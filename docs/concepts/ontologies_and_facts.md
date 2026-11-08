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
statement: an absolute `http(s)` IRI with no fragment, since its versions are
named `<iri>#<key>`. An empty fragment (`http://www.w3.org/ns/prov#`) is
dropped. Its **namespace**, where its terms live, is separate: `<iri>#` for an
ontology OntoCast creates, and read from the file for one you load. Two
shorter names point to the same entry: its `ontology_id` and the prefix its
author bound in Turtle. They can differ (an ontology with id
`observation` may use the prefix `obs:`), and wherever OntoCast asks for an
ontology, such as
[`ONTOLOGY_CONTEXT_FIXED_ONTOLOGY_ID`](../reference/configuration/pipeline.md#ontology_context_fixed_ontology_id),
any of the three works.

An `ontology_id` is a lowercase handle matching `^[a-z][a-z0-9_-]{1,31}$`. It
is also the prefix the ontology's terms are shown under. The names OntoCast uses
itself (`cd`, `doc`, `facts`, `oc`, `ontocast`) are refused. A standard prefix
(`sosa`, `prov`, `skos`, `schema`, ...) names only the ontology whose namespace
is that prefix's standard one: the W3C SOSA file loads as `sosa` and PROV-O as
`prov`, while any other ontology, and every ontology OntoCast creates, needs
another id.

### The home ontology

Identity is minted by code, never written by the model. A run that can create
vocabulary (`RENDER_MODE` `ontology` or `ontology_and_facts`) resolves one
**home ontology** per document before its units are extracted, in one LLM
call: the model proposes an `ontology_id`, a title and a description. An id
already in the catalog means the document extends that ontology; any other id
names a new one, whose IRI is
[`CURRENT_DOMAIN`](../reference/configuration/pipeline.md#current_domain)`/<ontology_id>`.
A new id is overridden when the document's text names at least five terms of
one catalog ontology (by label or local name, plurals included): the document
then extends that ontology, so documents of one domain share a home
(`home_new_overridden`).
Every unit of the document mints its new terms under that one ontology. The
model never writes an `owl:Ontology` header; one it writes anyway is dropped
and counted (`ontology/model_header_triples_dropped`).

#### Owned and external ontologies

An ontology whose IRI lies under `CURRENT_DOMAIN` is **owned** and is extended
in place. Any other, such as a published vocabulary loaded as a seed, is
**external**: OntoCast never writes to it. When a document's home would be an
external ontology `<id>`, its terms go to the **extension ontology**
`CURRENT_DOMAIN/<id>-ext`, created on first use, which `owl:imports` the
external one; every document and every run that extends it shares the
extension (`home_extension`). A triple whose subject is a term of an external
ontology is refused at the end of the document and counted
(`external_subject_refused`): the external terms are used by IRI, never
redeclared or given new axioms. To extend a seed ontology in place, serve it
under `CURRENT_DOMAIN`.

A unit mints a class only at the level a second document in the same domain
would reuse. A case the document reports, such as a numbered sample or a
composition, is an individual or a property value in the facts. A distinction
that varies from one instance to the next is a property with values, not a
family of sibling subclasses, and an existing or well-known term is reused
before a new one is minted. The per-unit checks enforce the parts of this a
machine can see: hierarchy depth, value-set siblings, case-specific leaves and
duplicates of catalog terms (see
[Validation layers](../internals/validation_layers.md#ontology-findings)).

An ontology you load from a file or upload to `/ontologies` keeps its own IRI.
The file declares exactly one named `owl:Ontology` that describes itself; a
header that is only typed `owl:Ontology` beside it is dropped, as W3C PROV-O
publishes one. Its namespace is the file's `vann:preferredNamespaceUri` when
terms live under it, else the namespace in which the file describes the most
terms: OM-2's is `…/om-2/`, PROV-O's is `…/prov#`. Its id is its
`vann:preferredNamespacePrefix`, else derived from the IRI; its title
`dcterms:title`, else `rdfs:label`, else the id; its version label
`owl:versionInfo` when that is a semantic version, else `1.0.0`. It is stored
as a root version. An upload whose IRI is already in the catalog is refused;
delete the ontology first, or replace it with `PUT /ontologies/{iri}`.

## Versions

An ontology is never edited in place. Each version is an immutable named graph
`<iri>#<key>`, its **version IRI**. For versions OntoCast writes, the key is the
content hash of the graph, so the same content is never stored twice and a run
that leaves an ontology unchanged writes no version. Inside that graph, the
ontology IRI carries the version header:

| Triple on the ontology IRI | Value |
|---|---|
| `owl:versionIRI` | The named graph, `<iri>#<key>` |
| `owl:priorVersion` | The version(s) this one was made from: none on a root, one for an update, several for a merge |
| `owl:versionInfo` | A semantic version label: `1.0.0` on a root, then bumped per ontology from the updates applied to it. Major when more than five triples are deleted and the updates declare or remove more than two classes or more than three properties; minor for any other deletion, or for five or more new classes or properties; patch otherwise |
| `vann:preferredNamespacePrefix`, `vann:preferredNamespaceUri` | The `ontology_id` and the namespace |
| `dcterms:title`, `rdfs:label` | The title |
| `dcterms:description` | The description, when there is one |
| `dcterms:created` | When the root of the lineage was created, carried forward |
| `dcterms:modified` | When this version was created |
| `prov:wasDerivedFrom` | The source document's IRI, for versions a run wrote |

Author statements on the ontology IRI outside the header (`owl:imports`, a
licence) are kept. A removed version keeps its header and is marked
`<graph> a oc:RemovedVersion` (`oc:` is `https://growgraph.dev/ontocast#`),
so lineage through it never dangles.

The **head** of an ontology is a version that is not removed and from which no
other version that is not removed descends; descent passes through removed
versions, so removing the newest version makes its priors heads again. A
run reads one head per ontology, and older versions stay in the store, so you
can compare what a run changed with what was there before. An ontology with
several heads is **diverged**: `GET /ontologies` lists it and its versions can
be read, but no run extracts against it or extends it until a merge gives it
one head. Uploading the ontology with `PUT /ontologies/{iri}` is that merge: the
new version derives from every head. [Ontology catalog](../internals/ontology_catalog.md) describes how
the catalog picks heads safely when several processes share a store.

## Changes are patches

The model never rewrites an ontology. It returns a patch: the triples to insert
and the triples to delete. OntoCast checks the patch and applies it to the
graph, deletions first. Two things follow:

- A render's output is proportional to the change, not to the size of the
  ontology, so a large ontology costs no more output tokens than a small one.
- The model cannot drop a triple by forgetting to repeat it; a removal is always
  an explicit delete.

The facts critic works the same way: its fixes are applied as a patch to the
unit's facts. The model writes patches as Turtle by default, or as compact
JSON-LD; see
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

The prompt names your ontology's namespaces as the ones its terms live in, under
the prefixes its term sheet shows. A standard vocabulary you load as an
ontology (PROV-O, SOSA) is one of them; the RDF languages (`rdf`, `rdfs`,
`owl`, `xsd`) never are.

The model is instructed to:

- name every new entity in `cd:`, with a `lowercase_snake_case` local name and
  an `rdfs:label` taken from the text;
- reuse an ontology IRI for an individual only when the ontology declares that
  individual, never invent one in the ontology's namespace;
- treat a matching class as a type, not as an identity: two mentions of a
  sample are new `cd:` entities typed with the ontology's `Sample` class, not
  the class itself;
- never type a `cd:` entity as a class or a property;
- extract statements: mint an entity only when a statement is about it (it is
  linked to another entity or holds a value), never for a bare mention;
- write a structured value (a quantity, a range, a bound) as a blank node, the
  object of the property that carries it, never as an entity of its own.

The served graph holds statements only. After merging, an entity linked to
nothing and holding no value is dropped, and so is a value no statement
carries (`facts_unlinked_dropped`). Values are never merged with each other;
each is named under its carrier, `<carrier>.<property>` (then a number when a
carrier holds several values on one property), so it has a stable IRI.

You can add your own guidance on top of these rules with
`facts_user_instruction`; see [Writing user
instructions](../guides/user_instructions.md).

## Two vocabulary modes

`FACTS_VOCABULARY` says what the facts may be written in. Both modes use your
catalog first, for every class and property it has. Where it has none, the
model uses a well-known vocabulary, and then the namespaces you list in
`FACTS_ADDITIONAL_STANDARD_NAMESPACES`. The prompt names them all, with their
namespace IRIs:

| Prefix | Vocabulary | Terms checked |
|---|---|---|
| `rdf`, `rdfs`, `owl`, `xsd` | RDF, RDF Schema, OWL, XML Schema datatypes | yes |
| `skos` | SKOS | yes |
| `dcterms`, `dc` | DCMI Metadata Terms, Dublin Core elements | yes |
| `prov` | PROV-O | yes |
| `sosa`, `ssn` | SOSA, SSN | yes |
| `time` | OWL-Time | yes |
| `csvw` | CSVW | yes |
| `dcat` | DCAT | yes |
| `org` | the Organization Ontology | yes |
| `vcard` | vCard | no |
| `foaf` | FOAF | yes |
| `schema` | schema.org (`https://` and `http://`) | yes |
| `qudt`, `unit`, `quantitykind` | QUDT | no |

The model writes well-known terms from memory, so a term must exist in its
vocabulary: one it does not have, such as `schema:vulnerableTo`, is an
`unknown_term` finding with the vocabulary's near-misses as suggestions. A
vocabulary marked "no" is admitted by namespace, as are your own namespaces.

**`strict`** (the default) stops there. A thing that neither your catalog nor
these vocabularies can type or relate, and a quantity they have no value class
or carrier property for, is not extracted. A class or property from any other
vocabulary is an `unknown_term` finding the renderer must rewrite or remove,
and at the gate `non_catalog_vocabulary` is an error for it. The model sees the
unit's retrieved context, not the whole catalog, so a property retrieval missed
is a statement made in a well-known vocabulary or not made at all: watch
`non_catalog_vocabulary`, `facts_isolated` and `facts_unlinked_dropped`, and
widen the context (`ONTOLOGY_CONTEXT_MODE`) when they grow.

**`open`** also admits any other established vocabulary the model knows, as a
last resort after the well-known ones. Such terms pass the unit validator and
are listed per document as `non_catalog_vocabulary` warnings, so the gaps in
your catalog are visible.

In both modes the `carrier` role of `FACTS_QUANTITY_FALLBACK_VOCABULARY` names
the property that carries a quantity when your catalog has none, and an entity
linked to nothing and holding no value is dropped at the gate.

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
| `ontocast:sectionLabelStrength` | `strong` or `weak`: whether the label could take part in dropping the unit |

`ontocast:` is `https://growgraph.dev/ontocast#`, outside the facts namespace,
so pipeline metadata is never mistaken for extracted facts. Provenance of
ontology changes is kept apart from the ontology, so the ontology you load
elsewhere carries none of it.

Provenance adds at least two triples for every fact. It is the chunk nodes and
statement reifiers the pipeline mints, not the PROV namespace: facts you
extract with PROV-O as your ontology are facts like any others and are kept. To
leave provenance out of what you get back:

- over HTTP, pass `strip_provenance=true` to `/process` (provenance is kept by
  default);
- with `ontocast process`, the files are written without it by default; pass
  `--keep-provenance` to keep it, together with `<name>.units.jsonl`, the text
  of each unit the provenance points at (see [Reading run
  telemetry](../guides/telemetry.md#tracing-a-fact-to-its-text-and-the-catalog-across-a-batch)).

A graph without provenance has no link back to the text, so nothing in it can
be checked against the source.

Provenance is written in the formats that carry RDF 1.2 triple terms: Turtle,
N-Triples, N-Quads and TriG. JSON-LD and RDF/XML cannot carry them, so asking
for either on a graph with provenance raises `ProvenanceFormatError`; strip the
provenance first (`TripleStoreManager.strip_provenance`) to write those.

In Python, an `RDFGraph` keeps its reifications beside its triples, in
`graph.reifications`. Copying, merging (`+`, `+=`) and a pydantic dump and load
carry them.
`graph.hash()` covers the statements, not the provenance.

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

The facts are written to a named graph: by default the document namespace, so
two runs over the same converted document write one graph and the later one
replaces the earlier. A library caller that wants each run's facts kept apart
sets `AgentState.facts_graph_iri` (an absolute IRI or a URN) for the run; entity
IRIs stay in the document namespace either way, so the runs can be compared.

## What to read next

- [How OntoCast works](index.md): where each of these steps happens.
- [Entity disambiguation](entity_disambiguation.md): how `cd:` entities from different units become one.
- [Triple stores](../guides/triple_stores.md): where ontologies and facts are stored.
