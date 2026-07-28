# Ontology Context

Before the LLM renders ontology updates for each content unit, OntoCast assembles **ontology context** — the background TTL/JSON-LD the model sees when extracting concepts.

Context is assembled **per unit** inside the ontology loop, not at document level.

## Context Modes

Set via `ONTOLOGY_CONTEXT_MODE` (server default) or per-request `ontology_context_mode`.

### `selected_single_ontology` (default)

The LLM selects one catalog ontology per content unit from seed ontologies in the triple store / `ONTOCAST_ONTOLOGY_DIRECTORY`.

- Does **not** require a vector store
- Vector store initialization is skipped unless vector mode is requested
- Optional `ontology_selection_user_instruction` guides selection

### `selected_vector_search_ontology`

Retrieves a stitched ontology ensemble from the configured vector store using hybrid dense + BM25 retrieval, then expands an induced subgraph subject to triple budgets.

**Requires one vector backend:**

| Backend | Configuration |
|---------|---------------|
| **Qdrant** (server) | `QDRANT_URI` (and optionally `QDRANT_API_KEY`) |
| **LanceDB** (embedded) | `LANCEDB_ENABLED=true`, `LANCEDB_DATA_DIR=~/.lancedb_data` — install with `uv sync --extra lancedb` |

Also required: compatible `EMBEDDING_*` settings and indexed ontology atoms in the active tenant/project partition (Qdrant collection or LanceDB table).

On `ToolBox.initialize` in this mode:

- Orphan ontology IRIs (indexed but absent from the synchronized catalog) are pruned by default (`VECTOR_STORE_PRUNE_ORPHAN_IRIS_ON_INIT=true`) so renamed ontologies do not linger in retrieval.
- Optional clean slate: `VECTOR_STORE_WIPE_ON_INIT=true` or CLI `--wipe-vector-store` drops the current partition before recreate+reindex.

`QDRANT_URI` and `LANCEDB_ENABLED=true` are mutually exclusive.

If vector infrastructure is unavailable, the API returns **409** with `error_code: VECTOR_STORE_UNAVAILABLE`.

**Key budget settings** (full reference: [Configuration — Ontology Patch Retrieval](configuration.md#ontology-patch-retrieval)):

Default path: per-window channel fusion → max-score IRI dedupe → global score order → window-scaled hard cap → expand. Setting a non-zero `ONTOLOGY_PATCH_PER_ONTOLOGY_SEED_QUOTA` inserts a per-ontology round-robin (visiting ontologies best-scoring first) in place of plain score order.

| Variable | Default | Role |
|----------|---------|------|
| `VECTOR_STORE_TOP_K` | `20` | Hits per channel per proposition window |
| `VECTOR_STORE_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES` | `550` | Global triple cap for context |
| `VECTOR_STORE_INDUCED_SUBGRAPH_DEPTH` | `2` | BFS depth for hub seed expansion |
| `VECTOR_STORE_INDUCED_SUBGRAPH_HUB_SEED_COUNT` | `16` | Top seeds receiving full BFS budget |
| `VECTOR_STORE_INDUCED_SUBGRAPH_ANCESTOR_CLOSURE_DEPTH` | `3` | `rdfs:subClassOf` hops in schema shell |
| `VECTOR_STORE_INDUCED_SUBGRAPH_ESTIMATED_TRIPLES_PER_QUERY` | `24` | Per-entity BFS quota hint |
| `VECTOR_STORE_INDUCED_SUBGRAPH_CANDIDATE_PUSHDOWN` | `false` | Opt-in SPARQL neighborhood CONSTRUCT (see below) |
| `VECTOR_STORE_PROPOSITION_MAX_WINDOWS` | `16` | Window cap; long chunks sample evenly across the text |
| `ONTOLOGY_PATCH_CROSS_QUERY_MERGE_MODE` | `max_score` | Default merge; `sum_score` (rewards multi-window agreement) and `hybrid` are opt-in |
| `ONTOLOGY_PATCH_PER_ONTOLOGY_SEED_QUOTA` | `0` | Max seeds per ontology; `0` (default) uses global score order |
| `ONTOLOGY_PATCH_SEEDS_PER_WINDOW` | `4` | Scales effective atom cap with proposition windows |
| `ONTOLOGY_PATCH_MAX_ATOMS_BASE` | `96` | Floor for the effective atom cap |
| `ONTOLOGY_PATCH_MAX_ATOMS` | `96` | Hard cap: `min(max_atoms, max(base, seeds_per_window × n_queries))` |
| `ONTOLOGY_PATCH_MIN_MERGED_MAX_SCORE` | `0.18` | Empty patch when the best per-window fused score is below this |
| `ONTOLOGY_PATCH_MMR_LAMBDA` | `1.0` | `1.0` skips MMR (default); lower enables diversity rerank |

Advanced (off by default): `ONTOLOGY_PATCH_PER_QUERY_*_SCORE_RATIO`, `ONTOLOGY_PATCH_MERGED_SCORE_RATIO`, hybrid tier-1/tier-2 (`MAX_ATOMS_TIER1`, `MIN_ENTITY_SCORE`).

### Recommended preset for dense scientific text

Use vector search mode with the defaults above. For noisy catalogs, optionally tighten with advanced knobs:

```bash
ONTOLOGY_PATCH_MAX_ATOMS=24
ONTOLOGY_PATCH_MERGED_SCORE_RATIO=0.5
ONTOLOGY_PATCH_MMR_LAMBDA=0.85
VECTOR_STORE_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES=600
```

Tighten only against measured output — see [Diagnostics](#diagnostics). Lowering the seed
budget trades recall away directly, and on a large catalog the defaults are already the
recall-favouring choice.

Effective seed budget grows with proposition windows (`seeds_per_window × n_queries`, floored by `max_atoms_base`, capped by `max_atoms`). Set `ONTOLOGY_PATCH_MERGED_SCORE_RATIO` / per-query score ratios only when you need stricter precision.

Retrieval expands ontology scope beyond hit sources when seeds reference classes
in other catalog ontologies via `rdfs:subClassOf`, `rdfs:domain`, or `rdfs:range`.

### Diagnostics

Retrieval quality is measured by `test/test_retrieval_recall.py`, which runs real
embeddings against a real Qdrant collection and reports a per-stage funnel:

```bash
cd ontocast
bash -c 'set -a; source .env; set +a; uv run pytest test/test_retrieval_recall.py -v -s'
```

**Seed recall** covers vector search, cross-window merge, seed allocation (global score
order by default; optional per-ontology round-robin), and the atom cap: the expected term
reached `atoms_final`. **Snapshot recall** additionally covers induced-subgraph expansion:
the term is *defined* in the returned graph. A gap between them localises the loss to the
graph stage.

Each is reported at two granularities. The **case** figures ask whether *any* expected
term survived, and saturate as soon as cases carry several expected terms — a corpus can
show 100% snapshot recall while under half its terms are actually present. The **TERM**
figures count every expected term and are the numbers to compare configurations on.
Treat differences under roughly one percentage point as run-to-run noise: approximate
nearest-neighbour search is not bit-reproducible across index builds.

Ground truth is supplied externally so the core stays domain-agnostic. Point
`ONTOCAST_RECALL_CORPUS` at a prebuilt corpus directory:

```text
<corpus>/ontologies/*.ttl
<corpus>/cases.jsonl     # {"id", "text", "expected_iris": [...], "ontology_iri"}
```

Case text is split into proposition windows exactly as production does, so multi-sentence
passages issue several queries. `ontocast-validation/run/build_recall_corpus.py` builds a
corpus in this layout from an ontology directory and document chunks.

`ONTOCAST_RECALL_ROOT` additionally accepts a Text2KGBench-style corpus (`a_ontologies/` +
`b_gt_text/`), scaled with `ONTOCAST_RECALL_ONTOLOGIES` and `ONTOCAST_RECALL_CASES`;
catalog size matters most, because the atom cap does not grow with it. Without either, the
in-repo anchor fixtures still run. The test skips when Qdrant is unreachable.

Per-run metrics are also available in production on
`state.retrieval_metrics["patch_retrieval"]`: `atoms_after_dedupe`, `atoms_final`,
`seed_iris`, `seeds_by_ontology`, `snapshot_triple_count`, `snapshot_pruned_uri_count`,
`snapshot_uri_components`.

#### Catalog I/O

Ensemble retrieval runs once per content unit, so how it reads the ontology catalog
dominates its cost. It never materializes the whole catalog: the seeds' cross-ontology
`rdfs:subClassOf` / `domain` / `range` references are resolved with targeted SPARQL
SELECTs, and only the ontologies that survive that filter are then fetched as graphs.
Backends without SPARQL (see [Triple Stores](triple_stores.md#custom-backends)) fall back
to the full-catalog scan and return identical results, just more slowly.

Graphs and their merged union are then served from `OntologyManager`, which is the single
read path for ontology graphs — see [Ontology Catalog](../architecture/ontology_catalog.md)
for why that is sound and where the responsibility boundary sits. In practice a document
pays for each ontology once, not once per content unit.

| Key | Meaning |
|---|---|
| `catalog_access_mode` | `sparql`, or `full_fetch_fallback` when the backend has no SPARQL or a query failed |
| `catalog_select_queries` | SELECTs issued (header catalog + reference hops) |
| `catalog_graphs_fetched` | Ontology graphs materialized for the expansion step |
| `catalog_context_mode` | `merged_catalog` (default) or `sparql_candidate` (pushdown) |
| `catalog_context_triples` | Size of the working graph the snapshot was built from |
| `catalog_graph_cache_hits` / `_misses` | Per-version ontology graph cache |
| `catalog_merge_cache_hits` / `_misses` | Merged working-graph cache, keyed by version set |

A `full_fetch_fallback` on a Fuseki deployment means queries are failing — retrieval is
degrading to slow rather than failing outright, which is worth investigating.

#### Candidate pushdown (opt-in)

`VECTOR_STORE_INDUCED_SUBGRAPH_CANDIDATE_PUSHDOWN=true` builds the working graph from a
single SPARQL `CONSTRUCT` of the seeds' bounded neighborhood instead of merging whole
ontology graphs. It requires a backend with `supports_sparql_construct()`, and silently
uses the merge path otherwise.

Only *candidate generation* moves into the engine. The budgeted admission that follows —
per-seed quotas, the global triple cap, connectivity repair, component pruning — stays in
Python, because every triple it admits depends on how many have been admitted already.

**It is off by default, and you should measure before turning it on.** Compare
`catalog_context_triples` between the two modes on your own corpus. On the Text2KGBench
benchmark set — 6 ontologies, 976 triples merged, 36 seeds — the candidate graph is **83%
of the merged graph**, so there is nothing to gain there. The value is bounding memory and
wire volume on a *large* catalog, where the seeds' neighborhood is a small fraction of what
is stored.

One known asymmetry: the cross-component schema-path repair can search past the fetched
neighborhood, so a rare connectivity bridge may be missing. That makes a snapshot
*smaller*, never wrong.

### Induced-subgraph behavior (defaults)

After seeds are chosen, expansion builds a budgeted snapshot:

- **All seed-bearing components are kept.** Seedless components are dropped; references to
  dropped IRIs are removed so the snapshot never names a term it does not define.
- **Individual seeds stay alongside their classes.** An individual's `rdf:type` classes are
  promoted into the seed set without discarding the individual (needed for the facts
  two-namespace contract's reference individuals).
- **BFS admission prefers schema role** (label, `rdf:type`, hierarchy, domain/range, then
  descriptions) when a seed's quota cannot hold a full level — not alphabetical order.
- **Long chunks** split into up to `VECTOR_STORE_PROPOSITION_MAX_WINDOWS` windows sampled
  at an even stride from start to end, so the tail of a long passage still issues queries.

### BM25 / index recreate

Lexical retrieval uses Qdrant's BM25 with IDF and indexes split local names plus
`rdfs:label` / `skos:prefLabel` / `dcterms:title` / `skos:altLabel`, and — for
quantity vocabularies — `qudt:symbol` / `qudt:ucumCode`. Collections built
before that contract fail loudly with `EmbeddingContractMismatchError` — wipe and reindex
(`VECTOR_STORE_WIPE_ON_INIT` or `--wipe-vector-store`).

Surface forms are capped per atom (`VECTOR_STORE_MINIMAL_LABEL_LIMIT`, default 5) and
selected deterministically, ranked by predicate and then by language. Two consequences
are worth knowing when indexing an external vocabulary:

- **Symbols get their own budget.** They are not queued behind labels, so a term
  declaring many labels cannot crowd them out — QUDT's `unit:DEG_C` carries 23 labels
  against a cap of 5. Symbols lead the sparse lane and follow the primary label in the
  core representation, so the entity keeps a readable name.
- **Untagged and `en*` literals rank first.** Other languages are demoted, not dropped.
  Without this a term is named by whichever language sorts first alphabetically, and the
  cap fills with translations no English-language query matches.

This is what makes indexing a full external unit vocabulary practical: QUDT publishes an
authoritative symbol and UCUM code per unit, so a corpus need not restate them as
`skos:altLabel` to stay findable by the form its prose actually uses.

### `fixed_single_ontology`

Always uses one catalog ontology identified by `ontology_context_fixed_ontology_id` (env: `ONTOLOGY_CONTEXT_FIXED_ONTOLOGY_ID` or per-request parameter).

The value may be:

- the catalog **ontology IRI** (canonical),
- the short **`ontology_id`** (e.g. `observation`), or
- the author **prefix** (e.g. `obs` when the Turtle binds `obs:` to that ontology namespace).

Returns **400** if the mode is fixed but no ontology id is provided.

## Ontology identity

- **Canonical catalog key** is the ontology IRI (`owl:Ontology` subject).
- **`ontology_id`** and author **`prefix`** are aliases registered on ingest; they may differ (e.g. `lifecycle` vs `life`).
- **Prompt snapshots** (`OntologySnapshot`) are views of catalog graphs (or stitched ensembles) with `source_iris` / `writable_iris` provenance — they have **no** catalog id. Single/fixed modes wrap one ontology; vector/merged modes assemble a multi-source graph.
- **Writeback** (`U → O*`): insert complements (`U \\ S`) are partitioned by namespace ownership onto writable catalog IRIs and merged onto each ontology's freshest terminal. Snapshots are never registered in the catalog.

## Assemble → propose → apply

```text
assemble:  catalog ontologies O*  →  OntologySnapshot S
propose:   (S, text) → GraphUpdate U   (complement only; do not restate S)
apply:     (U, O*) → versioned catalog ontologies O*'
```

Ontology-update prompts use **complement** framing (single writable IRI or multi-source). Facts prompts treat S as read-only schema (`cd:` for instances).

## Per-Request Overrides

All modes can be overridden on `/process` and `/process_unit`:

```bash
curl -X POST "http://localhost:8999/process?ontology_context_mode=fixed_single_ontology&ontology_context_fixed_ontology_id=legal_core" \
  -F "file=@contract.pdf"
```

JSON body equivalent:

```json
{
  "text": "...",
  "ontology_context_mode": "selected_vector_search_ontology"
}
```

## Seeding the Catalog

1. Place TTL files in `ONTOCAST_ONTOLOGY_DIRECTORY`, or
2. Upload via `POST /ontologies` (see [API Endpoints](api.md))

Ontologies are synced to the triple store on startup when configured.

## Vector Indexing

When a vector backend is configured (Qdrant or LanceDB) and vector mode is used, ontology atoms are embedded (core + neighborhood representations) and upserted into the tenant/project ontologies partition. BM25 sparse vectors provide a lexical retrieval lane fused with dense scores.

- **Qdrant** — collections `{tenant}--{project}--ontologies` / `--facts`
- **LanceDB** — tables with the same naming pattern under `LANCEDB_DATA_DIR`

Dedup policy (`VECTOR_STORE_DEDUP_MODE`): `iri` (one point per entity key) or `atom_id` (every atom variant).

## Related

- [Configuration](configuration.md) — full env var reference
- [Ontology Catalog](../architecture/ontology_catalog.md) — `OntologyManager` read path and cache
- [Tenancy](tenancy.md) — collection naming and catalog reset on tenant switch
- [User Instructions](user_instructions.md) — selection and extraction guidance
