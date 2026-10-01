# Ontologies and facts

OntoCast writes two kinds of graph: ontologies, which define the terms, and
facts, which state what a document says in those terms. This page explains how
each is identified, versioned and changed, and how to trace a fact back to the
text it came from.

## The catalog

The catalog is the set of ontologies OntoCast knows: the ones you load at
startup (`--ontology-dir` or
[`ONTOCAST_ONTOLOGY_DIRECTORY`](../reference/configuration/storage.md#ontocast_ontology_directory)),
the ones you upload to `/ontologies`, and the ones runs create.

An ontology is identified by its **IRI**, the subject of its `owl:Ontology`
statement. Two shorter names point to the same entry: its `ontology_id` and the
prefix its author bound in Turtle. They can differ (an ontology with id
`observation` may use the prefix `obs:`), and wherever OntoCast asks for an
ontology, such as
[`ONTOLOGY_CONTEXT_FIXED_ONTOLOGY_ID`](../reference/configuration/pipeline.md#ontology_context_fixed_ontology_id),
any of the three works.

## Versions

An ontology is never edited in place. Every change produces a new version,
stored as a named graph of its own, `<iri>#<hash>`, where the hash is computed
from the graph's content. Each version records:

| Triple on the ontology IRI | Value |
|---|---|
| `dcterms:identifier` | `hash:<hash>` of this version |
| `prov:wasDerivedFrom` | The version it was made from; two parents when two versions were merged |
| `dcterms:created` | When this version was created |
| `owl:versionInfo` | A semantic version. Major when more than five triples are deleted, including more than two classes or three properties; minor for any other deletion, or for five or more new classes or properties; patch otherwise |

A run always reads the newest version of each ontology, and older versions stay
in the store, so you can compare what a run changed with what was there before.
[Ontology catalog](../internals/ontology_catalog.md) describes how the catalog
picks the newest version safely when several processes share a store.

## Changes are patches

The model never rewrites an ontology. It returns a patch: the triples to insert
and the triples to delete. OntoCast checks the patch and applies it to the
graph, deletions first. Two things follow:

- A render's output is proportional to the change, not to the size of the
  ontology, so a large ontology costs no more output tokens than a small one.
- The model cannot drop a triple by forgetting to repeat it; a removal is always
  an explicit delete.

The facts critic works the same way: its fixes are applied as a patch to the
unit's facts. The model writes patches as compact JSON-LD by default, or as
Turtle; see
[`LLM_GRAPH_FORMAT`](../reference/configuration/pipeline.md#llm_graph_format).
Either way, the patch that is applied is the same.

## Facts and the two-namespace contract

Facts use two namespaces, and the rule between them is what keeps your
ontology clean:

| Namespace | What goes there |
|---|---|
| Your ontology's namespaces | Classes, properties, and the individuals the ontology already declares, such as the entries of a controlled vocabulary. Facts use these terms but never add to them |
| `cd:` | Every entity found in the text, even when its class comes from your ontology |

`cd:` is `https://growgraph.dev/facts/`. It is fixed, not a setting.

The model is instructed to:

- name every new entity in `cd:`, with a `lowercase_snake_case` local name and
  an `rdfs:label` taken from the text;
- reuse an ontology IRI for an individual only when the ontology declares that
  individual, never invent one in the ontology's namespace;
- treat a matching class as a type, not as an identity: two mentions of a
  sample are new `cd:` entities typed with the ontology's `Sample` class, not
  the class itself;
- never type a `cd:` entity as a class or a property.

You can add your own guidance on top of these rules with
`facts_user_instruction`; see [Writing user
instructions](../guides/user_instructions.md).

## Provenance

Every fact can be traced to the content unit it came from. OntoCast uses RDF
1.2 reification: each asserted triple gets a reifier that points to the unit,
or to several units when more than one stated it.

```turtle
_:r rdf:reifies <<( cd:sample_a ex:annealedAt cd:temperature_1 )>> ;
    prov:wasDerivedFrom <unit_iri> .
```

Each unit is described once:

| Predicate on the unit | Value |
|---|---|
| `rdf:type` | `prov:Entity`, `schema:Text` |
| `schema:position` | The unit's position in the document |
| `schema:identifier` | A hash of the unit's text |
| `prov:generatedAtTime` | When it was extracted |
| `schema:articleSection` | Its section label, when it has one |
| `ontocast:sectionLabelSource` | Which classifier step decided the label |
| `ontocast:sectionLabelConfidence` | How sure that step was, from 0 to 1 |

`ontocast:` is `https://growgraph.dev/ontocast#`, outside the facts namespace,
so pipeline metadata is never mistaken for extracted facts. Provenance of
ontology changes is kept apart from the ontology, so the ontology you load
elsewhere carries none of it.

Provenance adds at least two triples for every fact. To leave it out of what
you get back:

- over HTTP, pass `strip_provenance=true` to `/process` (provenance is kept by
  default);
- with `ontocast process`, the files are written without it by default; pass
  `--keep-provenance` to keep it.

A graph without provenance has no link back to the text, so nothing in it can
be checked against the source.

## Document identity

Each document gets an IRI built from
[`CURRENT_DOMAIN`](../reference/configuration/pipeline.md#current_domain) and a
hash of its content, and its units are linked to it. To state what the
document *is*, independent of its text, pass `document_metadata`: a JSON object
on `/process`, or `--document-metadata '{...}'` on `ontocast process`.

| Kind | Keys | Becomes |
|---|---|---|
| Bibliographic identifiers | `doi`, `isbn`, `pmid`, `arxiv_id`, `handle` | `dcterms:identifier` literals |
| Other identifiers | `identifiers: [{"scheme": "erp:doc", "value": "INV-001"}]`, or any key ending in an identifier word such as `_id`, `_ref`, `_no`, `_code` | A `dcterms:Identifier` node with `dcterms:type` and `rdf:value` |
| Description | `title`, `published` or `issued`, `source_system` | `dcterms:title`, `dcterms:issued`, `prov:wasAttributedTo` |
| Source | `source_uri`, `source_url`, `stable_source_iri` | `dcterms:source`, or `owl:sameAs` for `stable_source_iri` |
| People and other entities | `author`, `authors` or `creator`; `project`; any other key | A `schema:Person` (for authors) or `prov:Entity`, with an `rdfs:label` |

For example:

```json
{"doi": "10.1234/example", "author": ["Jane Doe"], "project": {"name": "Coatings survey", "identifier": "PRJ-7"}}
```

becomes

```turtle
<doc_iri> dcterms:identifier "10.1234/example" ;
    dcterms:creator <doc_iri/janeDoe> ;
    dcterms:relation <doc_iri/coatingsSurvey> .

<doc_iri/janeDoe> a schema:Person ; rdfs:label "Jane Doe" .
<doc_iri/coatingsSurvey> a prov:Entity ; rdfs:label "Coatings survey" ;
    dcterms:identifier "PRJ-7" .
```

Keys match regardless of case and of `snake_case`, `camelCase` or `kebab-case`,
so `DOI`, `doiId` and `doi_id` all mean `doi`. To give an entity another class,
pass a dict with a `type`, such as `{"name": "...", "type": "schema:Project"}`.
Entities are minted per document; OntoCast does not claim that the same author
in two documents is one person.

Document metadata is always in the facts graph, even when the text never
mentions it, and it stays there when provenance is stripped. Without
`document_metadata`, `ontocast process` sets `dcterms:title` to the file name.

## What to read next

- [How OntoCast works](index.md): where each of these steps happens.
- [Entity disambiguation](entity_disambiguation.md): how `cd:` entities from different units become one.
- [Triple stores](../guides/triple_stores.md): where ontologies and facts are stored.
