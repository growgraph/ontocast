# Ontology Catalog: Responsibility Boundary

Four components touch stored ontologies, and it is easy to reach for the wrong one.
This page is the standing answer to "which layer owns what", and in particular to
"what is `OntologyManager` for".

## The boundary

| Layer | Owns | Must not |
|---|---|---|
| `TripleStoreManager` (`tool/triple_manager/`) | Persistence and **query execution**: `aselect`, `aconstruct`, named-graph read/write, tenancy partitions | Know anything about catalog semantics, lineage, or aliases |
| `OntologyManager` (`tool/ontology_manager.py`) | **The catalog**: identity and aliases, hash lineage and terminal selection, the author-prefix table, and the content-addressed graph cache. The single answer to "give me ontology X's graph" | Execute queries itself; survive a tenancy switch |
| `SPARQLTool` (`tool/sparql.py`) | Graph algorithms over graphs it is **handed** | Fetch |
| `OntologyPatchRetriever` (`tool/vector_store/patch_retriever.py`) | Vector retrieval, seed selection, reference expansion | Materialize catalog graphs directly |
| `ShapesCatalog` (`tool/shapes_catalog.py`) | **The shapes partition**: seeding it from `FACTS_SHAPES_DIR`, and the merged shapes graph the synchronous facts gate reads | Reach the ontologies dataset, or be indexed as ontology atoms |

## Why the cache is sound

`OntologyManager` caches ontology *versions*, never *which version is terminal*.

- Terminal selection runs on headers read fresh from the store on every call
  (`aget_catalog_headers`), so another process's writes are visible immediately.
- Graphs are cached under the header's `graph_uri`, which is `{iri}#{sha256}` — a
  content address. A concurrent writer necessarily produces a *new* graph URI, which
  is a cache miss. There is no sequence of events that yields a stale hit.
- The key is the `graph_uri` the graph was *read from*, not the hash recomputed on the
  materialized `Ontology`. Those agree only while hashing is round-trip stable, which is
  why `RDFGraph.hash()` canonicalizes literals onto the value space triple stores
  normalize to (`canonical_literal`): stores rewrite `xsd:decimal` lexical forms and
  collapse integer subtypes on insert. Keying on the recomputed hash instead made every
  lookup for an affected ontology miss forever.

This matters for the cloud stack, where several `ontocast-worker` processes share one
Fuseki dataset. An in-process catalog could not be authoritative there; a
content-addressed cache in front of an authoritative store can.

The merged working graph is cached the same way, keyed by the `frozenset` of contributing
`versioned_iri` values. That is only safe because the induced-subgraph builder treats its
merged input as a read-only oracle and writes exclusively to its own result graph — an
invariant pinned by `test_merged_graph_is_not_mutated_by_the_builder`.

## Why shapes are a separate partition

Terminal selection starts from "every named graph carrying an `owl:Ontology`
subject" (`ONTOLOGY_HEADER_QUERY`). A SHACL shapes document declares exactly
that — `<…/qqval-shapes> a owl:Ontology`, with its own `owl:versionIRI`. Kept
in the ontologies dataset, a catalog shipping six shapes modules would gain six
phantom catalog entries: vector-indexed, terminal-selected, and served to the
renderer as schema it may extend.

`{tenant}--{project}--shapes` removes the ambiguity at the storage layer rather
than filtering it out at every read. The store surface therefore selects a
partition by `StoreKind` (`"facts" | "ontologies" | "shapes"`) instead of the
boolean it used while there were only two.

Shapes carry no lineage machinery: no hash-versioned graph URIs, no terminal
selection, no alias ledger. A document is addressed by the ontology IRI it
declares (or a stable path/filename-derived `urn:shapes:` name when it declares
none), and re-uploading replaces it. The merged graph the gate reads is the
union of the partition, because SHACL evaluates one shapes graph and shapes
documents are independent.

## Why it resets on a tenancy switch

Everything `OntologyManager` holds is partition-scoped: the ontologies, the graph caches,
and the alias ledger that `validate_identity_uniqueness` enforces. Carrying any of it
across a `?tenant=` switch leaks one tenant's ontologies into another's requests, and the
alias ledger additionally starts rejecting a legitimately distinct ontology that happens
to reuse an `ontology_id`.

`ToolBox.update_tenancy_with_vector_mode` therefore calls `reset_catalog()` (and
`ShapesCatalog.reset()`) and reloads from the retargeted store whenever the tenancy
actually changes. The first assignment is
the exception: it happens at startup, before `initialize()`, which populates the catalog
itself — resyncing there would just fetch everything twice.

Seed TTLs from `ONTOCAST_ONTOLOGY_DIRECTORY` **are** replayed on a switch, but only where
the partition serves nothing of its own. Withholding them was the more conservative
choice on paper and the worse one in practice: a scope whose catalog is empty for want of
a bootstrap is the same fault as a startup one, and the tenant extracted against no
vocabulary at all rather than being surprised by a write.

The test is whether the partition serves *terms*, not whether it lists the IRI. A catalog
read builds an ontology from its `owl:Ontology` subject and fills the graph separately, so
one whose graph never arrived still answers with a few triples about itself — a non-empty
graph that defines nothing. An ontology that does define terms is never overwritten: a
previous run's evolved terminal outranks whatever is on disk.

`FACTS_SHAPES_DIR` follows the same rule.

## Reading the catalog

```python
headers = await tools.ontology_manager.aget_catalog_headers()  # metadata, no graphs
ontologies = await tools.ontology_manager.aget_ontologies_by_iri(iris)
merged, prefix_map = await tools.ontology_manager.aget_merged_graph(ontologies)
iri = tools.ontology_manager.resolve_ontology_ref(
    "obs"
)  # IRI, ontology_id, or author prefix
```

Returned `Ontology` objects are **shared, read-only references**. This was already true
before the cache existed — `select_ontology_catalog` uses the live catalog object and
`update_ontology_manager` enriches these objects in place — the cache only makes the
existing contract load-bearing.

`resolve_ontology_ref` accepts the canonical ontology IRI, the short `ontology_id`, or the
author Turtle prefix. Prompt assembly uses `OntologySnapshot` views (no catalog id); writeback
applies insert complements onto freshest catalog terminals by namespace ownership — snapshots
are never registered as catalog entries. See [Ontology Context](../user_guide/ontology_context.md#assemble-propose-apply).

`catalog_cache_stats()` exposes hit/miss counters; they also ride along in
`state.retrieval_metrics["patch_retrieval"]` (see
[Ontology Context — Catalog I/O](../user_guide/ontology_context.md#catalog-io)).

## Related

- [Triple Stores](../user_guide/triple_stores.md) — backend surface and custom backends
- [Ontology Context](../user_guide/ontology_context.md) — how snapshots are assembled
- [Tenancy](../user_guide/tenancy.md) — partition naming
- [Validation](../user_guide/validation.md#where-shapes-come-from) — how the shapes partition feeds the gate
