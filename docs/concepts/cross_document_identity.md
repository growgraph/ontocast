# Cross-document identity

Every document's entities get IRIs in that document's namespace, so the same
court mentioned in forty judgments is forty entities. This page explains how
OntoCast links them without changing any document's facts, and how to read
which stored entities are one.

## What gets linked

Your ontology declares what identifies an entity across documents. Nothing is
linked until it does.

```turtle
@prefix ontocast: <https://growgraph.dev/ontocast#> .

ex:Court        ontocast:globallyNamed true .   # a court's name identifies it
ex:LocalCourt   rdfs:subClassOf ex:Court ;
                ontocast:globallyNamed false .  # but not a local court's
ex:Article      owl:hasKey ( ex:doi ) .         # an article is its DOI
ex:ecli         a owl:InverseFunctionalProperty .
```

| Declaration | Links an entity that | Match |
|---|---|---|
| `ontocast:globallyNamed true` on a class (inherited by subclasses; `false` stops it) | is of that class | each `rdfs:label` and `skos:altLabel`, normalised (case, accents, punctuation) |
| `owl:hasKey (p …)` on a class | is of that class and has a value for every `p` | the key values |
| `owl:InverseFunctionalProperty` | has a value for the property | the value |
| none: [document metadata](ontologies_and_facts.md#document-identity) with an `identifier` | the document node links to | the identifier, per type |

A key value counts when it is a literal (strings compared trimmed and
case-folded) or a term of your catalog. An IRI minted in one document never
counts, because it means nothing in another. Values (quantities, ranges) and
literals are never linked: you reach them through their entity. Labels like
"Sample 1" stay unlinked unless their class is declared, because the same name
in two documents usually names two things.

Declarations are read from every head of the catalog, not from the slice of
the ontology a document's extraction saw.

## The links graph

Each document's links are written to their own named graph beside its facts,
`<graph>/identity-links`. A link points to a **match node**, minted under
`<CURRENT_DOMAIN>/match/` from the class, the kind of match and the normalised
value, so every document that states the same name or key points to the same
node:

```turtle
<…/doc/a1…/echr> skos:closeMatch <…/match/7f3c…> .
<…/match/7f3c…> a ontocast:Match ;
    ontocast:matchKind "label" ;
    ontocast:matchClass ex:Court ;
    ontocast:matchValue "european court of human rights" .
```

The text that was matched is attached to each link through an RDF 1.2
reifier (`ontocast:matchedText`). `skos:closeMatch` is not transitive and no
document IRI is rewritten, so a wrong link never merges anything by itself.

## Keeping the links up to date

A document's links depend only on that document and the declarations, so
adding documents one at a time, in any order, gives the same links as building
them all at once.

| To | Do |
|---|---|
| Link each document as it is stored | [`LINK_ENABLED=true`](../reference/configuration/aggregation.md#link_enabled) |
| Link documents already stored | `ontocast link add --graph <facts graph IRI>` |
| Drop a document's links | `ontocast link remove --graph <facts graph IRI>` |
| Recompute everything, e.g. after changing a declaration | `ontocast link build` |

`build` also drops links whose document graph is gone. Each command takes
`--tenant` and `--project`, makes no model call, and needs a store with SPARQL.

## Which entities are one

Links propose; a **cluster** decides. Two entities that share a match node are
candidates. Candidates join in a fixed order, strongest evidence first (key,
inverse-functional value, identifier, then name), and two groups never join
when any pair across them is kept apart:

| Kept apart because | Example |
|---|---|
| same document | the document's own merging already decided they are two |
| disjoint types | `owl:disjointWith` between their classes or superclasses |
| functional conflict | a property your ontology allows one value for (`owl:FunctionalProperty`, max cardinality 1) has different values on each |
| key conflict | a key that applies to both has different complete values |

`GET /entities/cluster?iri=<entity>` (see the [HTTP API](../reference/http_api.md#cross-document-identity))
returns the members, the reason each joined, and the candidates kept apart
with the reason. From Python, `ontocast.link.cluster_of(store, declarations, iri)`
returns the same.

## Querying across documents

Each document is its own named graph, so a pattern that joins two documents
needs the union of the graphs. Library callers pass `union=True` to
`aselect` / `aconstruct`; on Fuseki this asks for `urn:x-arq:UnionGraph`.
Every article another document holds under the same key:

```sparql
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?other WHERE {
  <…/doc/a1…/paper> skos:closeMatch ?m .
  ?other skos:closeMatch ?m .
  FILTER(?other != <…/doc/a1…/paper>)
}
```

The union holds every graph of the facts partition, including the links and
ontology-provenance graphs. A raw join over links does not apply the
vetoes above; use clusters when that matters.

## What to read next

- [Entity disambiguation](entity_disambiguation.md): how entities within one document are merged.
- [Ontologies and facts](ontologies_and_facts.md): document namespaces and metadata.
- [Triple stores](../guides/triple_stores.md): the stores and their partitions.
