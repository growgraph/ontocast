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
| `QDRANT_TOP_K` | Fused hits per query |
| `QDRANT_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES` | Global triple cap for context |
| `QDRANT_INDUCED_SUBGRAPH_ESTIMATED_TRIPLES_PER_QUERY` | Per-query allocation hint |
| `ONTOLOGY_PATCH_*` | Post-retrieval scoring, MMR, atom caps |

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
