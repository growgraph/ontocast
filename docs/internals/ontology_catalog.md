# Ontology catalog

Several components touch stored ontologies, and it is easy to reach for the
wrong one. This page says which layer owns what, why the catalog cache is
safe, how tenancy scopes the catalog, and what a custom triple-store backend
has to implement.

## The boundary

| Layer | Owns | Must not |
|---|---|---|
| `TripleStoreManager` (`tool/triple_manager/`) | Persistence and query execution: `aselect`, `aconstruct`, named-graph reads and writes, tenancy partitions | Know about catalog semantics, lineage or aliases |
| `OntologyManager` (`tool/ontology_manager.py`) | The catalog: identity and aliases, version lineage and head selection, the author-prefix table, and the graph cache keyed by version IRI. The one answer to "give me ontology X's graph" | Execute queries itself, or hold more than one partition |
| `SPARQLTool` (`tool/sparql.py`) | Graph algorithms over graphs it is handed | Fetch |
| `OntologyPatchRetriever` (`tool/vector_store/patch_retriever.py`) | Vector retrieval, seed selection, reference expansion | Materialize catalog graphs directly |
| `ShapesCatalog` (`tool/shapes_catalog.py`) | The shapes partition: seeding it from `FACTS_SHAPES_DIR`, and the merged shapes graph the facts gate reads | Reach the ontologies dataset, or be indexed as ontology atoms |

## Why the cache is safe

`OntologyManager` caches ontology versions, never which version is the head.

- Head selection runs on headers read from the store on every call
  (`aget_catalog_headers`): a head is a live version that no live version
  descends from. Descent walks through tombstoned versions (`oc:RemovedVersion`),
  so removing a head makes its nearest live ancestors the heads; an ontology
  whose versions are all tombstoned has no heads. A version another process
  wrote is visible at once.
- A version is immutable, and graphs are cached under its version IRI, the
  named graph it was read from (`owl:versionIRI`). A concurrent writer
  produces a new version IRI, which is a cache miss, so no sequence of events
  yields a stale hit. Several processes can therefore share one Fuseki
  dataset, with the store as the authority and each catalog a cache in front
  of it.
- The key is a name read from the store, never a hash recomputed on the
  materialized `Ontology`. The content hash decides only whether a derived
  artifact is a new version, which is why `RDFGraph.hash()` canonicalizes
  literals onto the value space stores normalize to (`canonical_literal`):
  stores rewrite `xsd:decimal` lexical forms and collapse integer subtypes on
  insert.

The merged working graph is cached the same way, keyed by the `frozenset` of
contributing version IRIs. That is safe because the induced-subgraph builder
treats its merged input as read-only and writes only to its own result graph.

A diverged ontology (several heads) is read-only to runs: it is listed, its
versions can be read, and `get_head` raises `DivergedOntologyError` until a
merge makes one head. `ontologies` leaves it out, so nothing extracts against
it or extends it. `ingest_ontology_ttl(..., as_new_version=True)` (`PUT
/ontologies/{iri}`) stamps the uploaded file as a version derived from every
head, which is that merge.

## Shapes partition

The catalog read takes every named graph that carries an `owl:Ontology`
subject (`ONTOLOGY_HEADER_QUERY`) and is strict: a graph in the ontologies
dataset that is not a version (a header field missing, `owl:versionIRI` not
equal to the graph name, or several `owl:Ontology` subjects) raises
`CatalogContractError`, never a silent gap in the catalog. A SHACL shapes
document usually declares an `owl:Ontology` and carries no header. Shapes
therefore live in their own partition, `{tenant}--{project}--shapes`, and the
store selects a partition by `StoreKind` (`"facts"`, `"ontologies"` or
`"shapes"`). [Validation](../guides/validation.md) gives the user-facing
reasons.

Shapes carry no lineage: no version IRIs, no head selection, no alias ledger. A document is addressed by the ontology IRI it declares, or by
a `urn:shapes:` name derived from its path when it declares none, and
uploading it again replaces it. The gate reads the union of the partition,
because SHACL evaluates one shapes graph and shapes documents are independent.

## Tenancy

Everything `OntologyManager` and `ShapesCatalog` hold belongs to one
partition: the ontologies, the graph caches, and the alias ledger that
`validate_identity_uniqueness` enforces. A ToolBox is therefore bound to one
tenant and project, and serving several means several ToolBoxes.

- `ToolBox.for_scope(tenant, project)` returns the ToolBox for a partition:
  itself when the scope matches, otherwise one from its `ToolBoxRegistry`
  (`registry.py`). The server calls it for every request that names a
  `tenant` or `project`.
- The registry builds each scope's ToolBox over its own copy of `Config`
  (`Config.for_tenancy`), with dataset, collection and table names resolved for
  that partition, so scopes cannot alias each other's names. The expensive
  tools (LLM client, converter, embedding model, cache) live in a shared
  `ToolBoxRuntime` (`runtime.py`), so a new scope costs a store connection and
  a catalog.
- A new scope runs `initialize()`, which reads the partition's catalog and
  shapes. The registry keeps at most
  [`MAX_TENANCY_SCOPES`](../reference/configuration/pipeline.md#max_tenancy_scopes)
  scopes and closes the least recently used one beyond that.
- `ToolBox.update_tenancy_with_vector_mode` retargets an existing ToolBox in
  place. The CLI calls it once at startup to bind the startup ToolBox, before
  `initialize()` populates the catalog. On any later change of scope it calls
  `OntologyManager.reset_catalog()` and `ShapesCatalog.reset()` and reloads both
  from the new partition.

Seed files from `ONTOCAST_ONTOLOGY_DIRECTORY` are written into a partition
only where it serves no usable graph for that IRI. The test is whether the
stored ontology defines terms, not whether its IRI is listed: a catalog read
builds an ontology from its `owl:Ontology` subject and fills the graph
separately, so one whose graph is missing still answers with a few header
triples. A stored ontology that defines terms is never overwritten, because an
ontology-mode run writes its evolved version back and the seed on disk is
older. `FACTS_SHAPES_DIR` is synced into the shapes partition at the same
point.

## Reading the catalog

```python
headers = await tools.ontology_manager.aget_catalog_headers()  # metadata, no graphs
ontologies = await tools.ontology_manager.aget_ontologies_by_iri(iris)
merged, prefix_map = await tools.ontology_manager.aget_merged_graph(ontologies)
iri = tools.ontology_manager.resolve_ontology_ref("obs")  # IRI, ontology_id or prefix
head = tools.ontology_manager.get_head(
    "obs"
)  # raises DivergedOntologyError on several heads
heads = tools.ontology_manager.get_heads("obs")  # every head, newest first
```

Returned `Ontology` objects are shared, read-only references: catalog
selection uses the live objects, and the cache depends on nobody mutating them.

`resolve_ontology_ref` accepts the ontology IRI, the short `ontology_id`, or
the author's Turtle prefix. Prompt assembly works on `OntologySnapshot` views,
which have no catalog id; writeback applies insert complements onto the head
of each ontology by namespace ownership, and snapshots are never registered as
catalog entries. See [Choosing ontology context](../guides/ontology_context.md).

`catalog_cache_stats()` returns hit and miss counters; the retriever copies
them into `retrieval_metrics["patch_retrieval"]`.

## Targeted reads

`fetch_ontologies()` materializes every stored ontology, which suits startup
but not the per-unit retrieval path. `TripleStoreManager` offers narrower
reads:

| Method | Returns |
|---|---|
| `aselect(query, *, store="ontologies")` | One `dict[str, str]` per SPARQL SELECT solution |
| `aconstruct(query, *, store="ontologies")` | An `RDFGraph` of real RDF terms, without prefix bindings |
| `afetch_ontology_catalog()` | One `OntologyHeader` (`iri`, `ontology_id`, `title`, `description`, `version`, `version_iri`, `prior_version_iris`, `created_at`, `modified_at`, `removed`, `graph_uri`) per stored version, with no graphs |
| `afetch_ontologies_by_iri(iris)` | Ontologies with graphs, restricted to `iris` (empty means all) |

`aselect` rows carry each term's lexical value only, so constrain term kinds in
the query (`FILTER(isIRI(?x))`); unbound variables are absent from the row.
`aconstruct` keeps blank nodes and datatypes but not prefixes, which are
serialization metadata. Both raise on failure rather than return an empty
result, since empty means "nothing matched".

`OntologyHeader` is not an `Ontology`: building an `Ontology` recomputes its
content hash from the graph, so a graph-less one would carry a made-up hash.
Lineage is read from the header alone (`owl:versionIRI`, `owl:priorVersion`),
and `heads_by_iri()`, `catalog_heads()` and `select_relevant_ontologies()`
accept headers and ontologies alike, so heads can be picked without
downloading graphs.

## Custom triple-store backends

A `TripleStoreManager` subclass must implement `fetch_ontologies`,
`serialize_graph`, `serialize` and `clean`. Every read in the table above has
a base-class default built on `fetch_ontologies`, so a minimal backend works
and fetches more than it needs. A write the store refuses raises
`TripleStoreWriteError`; nothing downstream treats a refused write as done,
and a run writes its ontology versions before it registers and indexes them
and before it writes the facts. Reads of the ontologies dataset are strict:
a graph that is not a version raises `CatalogContractError`. Three predicates
opt into more:

| Predicate | With | Enables |
|---|---|---|
| `supports_sparql_select()` | `aselect()` | Targeted catalog reads, reference expansion, listing the shapes partition |
| `supports_sparql_construct()` | `aconstruct()` | Reading the shapes partition, and candidate pushdown ([`VECTOR_STORE_INDUCED_SUBGRAPH_CANDIDATE_PUSHDOWN`](../reference/configuration/ontology-retrieval.md#vector_store_induced_subgraph_candidate_pushdown)) |
| `supports_tenancy_partition()` | `update_tenancy()`, `clean_tenancy()` | Per-tenant partitions and `POST /flush?tenant=...` |

The SELECT and CONSTRUCT predicates are separate because a backend can answer
row queries without returning triples; Fuseki's SELECT path, for instance,
accepts only `application/sparql-results+json`. Without
`supports_sparql_construct()` the facts gate sees no shapes, and validation
reports that SHACL did not run.

## Related

- [Triple stores](../guides/triple_stores.md): choosing and configuring a backend.
- [Tenancy](../guides/tenancy.md): partition naming.
- [How ontology retrieval works](retrieval.md): what the retriever does with the catalog.
