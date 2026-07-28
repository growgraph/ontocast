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

## Why the cache is sound

`OntologyManager` caches ontology *versions*, never *which version is terminal*.

- Terminal selection runs on headers read fresh from the store on every call
  (`aget_catalog_headers`), so another process's writes are visible immediately.
- Graphs are cached under `Ontology.versioned_iri`, which is `{iri}#{sha256}` — a
  content address. A concurrent writer necessarily produces a *new* graph URI, which
  is a cache miss. There is no sequence of events that yields a stale hit.

This matters for the cloud stack, where several `ontocast-worker` processes share one
Fuseki dataset. An in-process catalog could not be authoritative there; a
content-addressed cache in front of an authoritative store can.

The merged working graph is cached the same way, keyed by the `frozenset` of contributing
`versioned_iri` values. That is only safe because the induced-subgraph builder treats its
merged input as a read-only oracle and writes exclusively to its own result graph — an
invariant pinned by `test_merged_graph_is_not_mutated_by_the_builder`.

## Why it resets on a tenancy switch

Everything `OntologyManager` holds is partition-scoped: the ontologies, the graph caches,
and the alias ledger that `validate_identity_uniqueness` enforces. Carrying any of it
across a `?tenant=` switch leaks one tenant's ontologies into another's requests, and the
alias ledger additionally starts rejecting a legitimately distinct ontology that happens
to reuse an `ontology_id`.

`ToolBox.update_tenancy_with_vector_mode` therefore calls `reset_catalog()` and reloads
from the retargeted store whenever the tenancy actually changes. The first assignment is
the exception: it happens at startup, before `initialize()`, which populates the catalog
itself — resyncing there would just fetch everything twice.

Seed TTLs from `ONTOCAST_ONTOLOGY_DIRECTORY` are **not** replayed on a switch. They are a
startup bootstrap; materializing them into a different tenant as a side effect of a query
parameter would be a surprise.

## Reading the catalog

```python
headers = await tools.ontology_manager.aget_catalog_headers()      # metadata, no graphs
ontologies = await tools.ontology_manager.aget_ontologies_by_iri(iris)
merged, prefix_map = await tools.ontology_manager.aget_merged_graph(ontologies)
iri = tools.ontology_manager.resolve_ontology_ref("obs")  # IRI, ontology_id, or author prefix
```

Returned `Ontology` objects are **shared, read-only references**. This was already true
before the cache existed — `select_ontology_catalog` uses the live catalog object and
`update_ontology_manager` enriches these objects in place — the cache only makes the
existing contract load-bearing.

`resolve_ontology_ref` accepts the canonical ontology IRI, the short `ontology_id`, or the
author Turtle prefix. Prompt assembly uses `OntologySnapshot` views (no catalog id); writeback
applies insert complements onto freshest catalog terminals by namespace ownership — snapshots
are never registered as catalog entries. See [Ontology Context](../user_guide/ontology_context.md#assemble--propose--apply).

`catalog_cache_stats()` exposes hit/miss counters; they also ride along in
`state.retrieval_metrics["patch_retrieval"]` (see
[Ontology Context — Catalog I/O](../user_guide/ontology_context.md#catalog-io)).

## Related

- [Triple Stores](../user_guide/triple_stores.md) — backend surface and custom backends
- [Ontology Context](../user_guide/ontology_context.md) — how snapshots are assembled
- [Tenancy](../user_guide/tenancy.md) — partition naming
