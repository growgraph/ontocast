# Ontology Context

Before the LLM renders ontology updates for each content unit, OntoCast assembles **ontology context** — the background TTL/JSON-LD the model sees when extracting concepts.

Context is assembled **per unit** inside the ontology loop, not at document level.

## Context Modes

Set via `ONTOLOGY_CONTEXT_MODE` (server default) or per-request `ontology_context_mode`.

### `selected_single_ontology` (default)

The LLM selects one catalog ontology per content unit from seed ontologies in the triple store / `ONTOCAST_ONTOLOGY_DIRECTORY`.

- Does **not** require Qdrant
- Vector store initialization is skipped unless vector mode is requested
- Optional `ontology_selection_user_instruction` guides selection

### `selected_vector_search_ontology`

Retrieves a stitched ontology ensemble from Qdrant using hybrid vector + BM25 retrieval, then expands an induced subgraph subject to triple budgets.

**Requires:**

- `QDRANT_URI` (and optionally `QDRANT_API_KEY`)
- Compatible `EMBEDDING_*` settings
- Indexed ontology atoms in the tenant/project collection

If vector infrastructure is unavailable, the API returns **409** with `error_code: VECTOR_STORE_UNAVAILABLE`.

**Key budget settings:**

| Variable | Role |
|----------|------|
| `QDRANT_TOP_K` | Fused hits per proposition window (default `10`) |
| `QDRANT_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES` | Global triple cap for context (default `550`) |
| `QDRANT_INDUCED_SUBGRAPH_DEPTH` | BFS depth for hub seed expansion (default `2`) |
| `QDRANT_INDUCED_SUBGRAPH_HUB_SEED_COUNT` | Top seeds receiving full BFS budget (default `8`) |
| `QDRANT_INDUCED_SUBGRAPH_ANCESTOR_CLOSURE_DEPTH` | `rdfs:subClassOf` hops included in schema shell (default `3`) |
| `QDRANT_INDUCED_SUBGRAPH_ESTIMATED_TRIPLES_PER_QUERY` | Per-entity BFS quota hint |
| `ONTOLOGY_PATCH_CROSS_QUERY_MERGE_MODE` | `hybrid` (default), `max_score`, or `rrf` |
| `ONTOLOGY_PATCH_MAX_ATOMS_TIER1` | Strong global seed cap for hybrid merge (default `12`) |
| `ONTOLOGY_PATCH_PER_ONTOLOGY_SEED_QUOTA` | Tier-2 seeds per ontology IRI (default `3`) |
| `ONTOLOGY_PATCH_MIN_ENTITY_SCORE` | Tier-2 minimum fused score (default `0.3`) |
| `ONTOLOGY_PATCH_MAX_ATOMS` | Total seed cap after merge/MMR (default `25`) |
| `ONTOLOGY_PATCH_MERGED_SCORE_RATIO` | Trim weak seeds vs top score (default `0.45`) |
| `ONTOLOGY_PATCH_MMR_LAMBDA` | MMR relevance vs diversity (default `0.9`) |

### Recommended preset for dense scientific text

Use vector search mode with defaults above, or tighten further:

- `ONTOLOGY_PATCH_MAX_ATOMS=20`
- `ONTOLOGY_PATCH_MERGED_SCORE_RATIO=0.5`
- `QDRANT_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES=600`

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

When Qdrant is configured and vector mode is used, ontology atoms are embedded (core + neighborhood representations) and upserted into the tenant/project ontologies collection. BM25 sparse vectors provide a lexical retrieval lane fused with dense scores.

Dedup policy (`QDRANT_DEDUP_MODE`): `iri` (one point per entity key) or `atom_id` (every atom variant).

## Related

- [Configuration](configuration.md) — full env var reference
- [Tenancy](tenancy.md) — collection naming
- [User Instructions](user_instructions.md) — selection and extraction guidance
