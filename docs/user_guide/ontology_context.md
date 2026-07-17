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

`QDRANT_URI` and `LANCEDB_ENABLED=true` are mutually exclusive.

If vector infrastructure is unavailable, the API returns **409** with `error_code: VECTOR_STORE_UNAVAILABLE`.

**Key budget settings** (full reference: [Configuration — Ontology Patch Retrieval](configuration.md#ontology-patch-retrieval)):

| Variable | Default | Role |
|----------|---------|------|
| `VECTOR_STORE_TOP_K` | `10` | Fused hits per proposition window |
| `VECTOR_STORE_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES` | `550` | Global triple cap for context |
| `VECTOR_STORE_INDUCED_SUBGRAPH_DEPTH` | `2` | BFS depth for hub seed expansion |
| `VECTOR_STORE_INDUCED_SUBGRAPH_HUB_SEED_COUNT` | `8` | Top seeds receiving full BFS budget |
| `VECTOR_STORE_INDUCED_SUBGRAPH_ANCESTOR_CLOSURE_DEPTH` | `3` | `rdfs:subClassOf` hops in schema shell |
| `VECTOR_STORE_INDUCED_SUBGRAPH_ESTIMATED_TRIPLES_PER_QUERY` | `24` | Per-entity BFS quota hint |
| `ONTOLOGY_PATCH_PER_QUERY_CORE_SCORE_RATIO` | `0.85` | Per-window core hit floor vs best core score |
| `ONTOLOGY_PATCH_PER_QUERY_NEIGHBORHOOD_SCORE_RATIO` | `0.85` | Per-window neighborhood hit floor vs best neighborhood score |
| `ONTOLOGY_PATCH_PER_QUERY_BM25_SCORE_RATIO` | `0.85` | Per-window BM25 hit floor vs best BM25 score |
| `ONTOLOGY_PATCH_MIN_MERGED_MAX_SCORE` | `0.18` | Empty patch when merged top score is below this |
| `ONTOLOGY_PATCH_MERGED_SCORE_RATIO` | `0.45` | Drop merged seeds below `top_score × ratio` |
| `ONTOLOGY_PATCH_CROSS_QUERY_MERGE_MODE` | `hybrid` | `hybrid`, `max_score`, or `rrf` |
| `ONTOLOGY_PATCH_MAX_ATOMS_TIER1` | `12` | Strong global seed cap for hybrid merge |
| `ONTOLOGY_PATCH_PER_ONTOLOGY_SEED_QUOTA` | `3` | Tier-2 seeds per ontology IRI |
| `ONTOLOGY_PATCH_MIN_ENTITY_SCORE` | `0.3` | Tier-2 minimum fused score |
| `ONTOLOGY_PATCH_MAX_ATOMS` | `25` | Total seed cap after merge/MMR |
| `ONTOLOGY_PATCH_MMR_LAMBDA` | `0.9` | MMR relevance vs diversity |

### Recommended preset for dense scientific text

Use vector search mode with the defaults above. For large catalogs or noisy retrieval, tighten:

```bash
ONTOLOGY_PATCH_MAX_ATOMS=20
ONTOLOGY_PATCH_MERGED_SCORE_RATIO=0.5
ONTOLOGY_PATCH_MMR_LAMBDA=0.85
VECTOR_STORE_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES=600
```

`ONTOLOGY_PATCH_MERGED_SCORE_RATIO=0.45` is the general-purpose default: it trims weak seeds that are less than 45% of the top merged score. Raise toward `0.5`–`0.6` for stricter recall control; set `0` to disable post-merge trimming.

Retrieval expands ontology scope beyond hit sources when seeds reference classes
in other catalog ontologies via `rdfs:subClassOf`, `rdfs:domain`, or `rdfs:range`.

### Diagnostics

Manual staged logging for matsci / perovskitemat coverage:

```bash
ONTOCAST_RUN_MANUAL_TESTS=1 cd ontocast && uv run pytest \\
  test/manual/test_perovskite_retrieval_diagnostics.py -v --log-cli-level=INFO
```

### `fixed_single_ontology`

Always uses one catalog ontology identified by `ontology_context_fixed_ontology_id` (env: `ONTOLOGY_CONTEXT_FIXED_ONTOLOGY_ID` or per-request parameter).

Returns **400** if the mode is fixed but no ontology id is provided.

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
- [Tenancy](tenancy.md) — collection naming
- [User Instructions](user_instructions.md) — selection and extraction guidance
