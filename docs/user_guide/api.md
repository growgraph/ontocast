# API Endpoints

OntoCast exposes a FastAPI server (CLI entry point: `ontocast`). Default port: **8999**.

## Health and Info

### `GET /health`

Returns service health. Use for load balancers and readiness probes.

### `GET /info`

Returns service metadata, including:

| Field | Description |
|-------|-------------|
| `version` | Package version |
| `llm_cache` | When the LLM tool is initialized: in-memory hit/miss counters plus on-disk cache file stats (`cache_hits`, `cache_misses`, `disk`) |
| `max_concurrent_processes` | Configured cap on simultaneous `/process` handlers, if `MAX_CONCURRENT_PROCESSES` is set |

```bash
curl http://localhost:8999/info
```

---

## Document Processing

### `POST /process`

Runs the full document pipeline: convert → chunk → ontology map/reduce → facts map/reduce → serialize.

**Content types:**

- `application/json` — body must include a `text` field (or file references as supported)
- `multipart/form-data` — upload files (`file` field) or form fields

**Common query / form / JSON parameters:**

| Parameter | Description |
|-----------|-------------|
| `tenant` | Tenant name for store partitioning (default: `ontocast`) |
| `project` | Project name (default: `test`) |
| `render_mode` | `ontology`, `facts`, or `ontology_and_facts` |
| `max_visits` | Per-request render/critic retry budget (≥ 1) |
| `strip_provenance` | When true, omit reification/provenance from returned Turtle |
| `llm_graph_format` | `turtle` or `jsonld` for this request |
| `ontology_context_mode` | Per-request ontology context mode |
| `ontology_context_fixed_ontology_id` | Required when mode is `fixed_single_ontology` |
| `ontology_user_instruction` | Guide ontology extraction |
| `ontology_selection_user_instruction` | Guide catalog ontology selection |
| `facts_user_instruction` | Guide facts extraction |
| `target_sections` | Comma-separated or JSON list; keep only these sections (enables section tagging) |
| `summarize_sections` | Sections to summarize before extraction; omit to skip. `*` or empty = all chunks |
| `summary_max_sentences` | Max sentences per summary when summarization runs (default `5`) |

**CLI file processing** (`ontocast --input-path …`) accepts the same structured-document flags:

```bash
ontocast --input-path ./papers/ \
  --target-sections results,methods \
  --summarize-sections results \
  --summary-max-sentences 5
```

**Examples:**

```bash
# JSON body
curl -X POST http://localhost:8999/process \
  -H "Content-Type: application/json" \
  -d '{"text": "Your document text here"}'

# PDF upload
curl -X POST http://localhost:8999/process \
  -F "file=@document.pdf"

# Strip provenance from API Turtle output
curl -X POST "http://localhost:8999/process?strip_provenance=true" \
  -F "file=@document.pdf"

# Multi-tenant request
curl -X POST "http://localhost:8999/process?tenant=acme&project=reports" \
  -F "file=@document.pdf"
```

**Response:** JSON with `data.facts` (Turtle), `data.ontology_artifacts` (list of ontology TTL payloads), and `metadata` (status, chunk counts, budget including `cache_hits` when applicable).

---

### `POST /process_unit`

Runs the ontology and/or facts loop for a **single content unit** without the full document graph. Useful for debugging prompts and unit-level behavior.

Accepts the same parameters as `/process` (including `strip_provenance`, user instructions, and ontology context settings).

```bash
curl -X POST http://localhost:8999/process_unit \
  -H "Content-Type: application/json" \
  -d '{"text": "Single paragraph to process."}'
```

---

## Ontology Catalog

Routes under `/ontologies` manage the seed ontology catalog in the configured triple store. All routes accept optional `tenant` and `project` query parameters (same semantics as `/process`).

### `POST /ontologies`

Upload a catalog ontology (Turtle file).

```bash
curl -X POST "http://localhost:8999/ontologies?tenant=ontocast&project=test" \
  -F "file=@my_ontology.ttl"
```

### `PUT /ontologies/{ontology_iri}`

Replace an ontology by IRI (URL-encoded path segment). The Turtle file's ontology IRI must match the path.

### `DELETE /ontologies/{ontology_iri}`

Remove an ontology from the catalog by IRI.

See [Tenancy](tenancy.md) for dataset naming.

---

## Triple Store Maintenance

### `POST /flush`

Clear triple store data.

```bash
# Fuseki: flush default datasets for active tenant/project
curl -X POST http://localhost:8999/flush

# Fuseki: flush a specific dataset
curl -X POST "http://localhost:8999/flush?dataset=my_dataset"
```

- **Fuseki:** without `dataset`, flushes facts and ontologies datasets for the resolved tenant/project scope.
- **Neo4j:** deletes all nodes and relationships (`dataset` is ignored).

!!! warning
    This operation is irreversible.

---

## Graph Matching

Benchmark-oriented endpoints for entity alignment and evaluation. Used by the standalone `match-dirs` CLI.

Alignment treats **identical IRIs** across input graphs as the same entity (score 1.0) before embedding similarity is considered — important when predicted and ground-truth graphs share ontology class or property IRIs.

### `POST /match/entities`

Align entities globally across a list of graphs (embedding + symbolic clustering).

```json
{
  "graphs": [
    {"id": "gt:doc1.ttl", "graph": "@prefix ex: <https://gt.example/> . ..."},
    {"id": "predicted:doc1.ttl", "graph": "@prefix ex: <https://pred.example/> . ..."}
  ],
  "regime": "ontology_loose",
  "similarity_threshold": 0.8
}
```

### `POST /match/derive-matches`

Derive 1:1 predicted↔ground-truth entity matches for one graph pair from alignment clusters.

### `POST /match/evaluate`

Compute triple and entity precision/recall/F1 given graphs and entity matches. Label triples (`rdfs:label`) are excluded from triple metrics.

Entity metrics: true positives = number of accepted entity matches; false positives = predicted entities not in the matched set; false negatives = ground-truth entities not in the matched set (set-based, so correctly matched shared vocabulary IRIs are not double-penalized).

**Standalone CLI:**

```bash
match-dirs \
  --gt ./benchmark \
  --predicted ./extracted \
  --url http://localhost:8999 \
  --regime ontology_strict \
  --similarity-threshold 0.8
```

---

## Error Responses

| Status | Condition |
|--------|-----------|
| `400` | Invalid parameters (e.g. missing fixed ontology id) |
| `409` | Vector store unavailable when vector ontology mode requested |
| `500` | Processing or store errors |
| `503` | Health check: LLM not initialized or health probe failure |

When `MAX_CONCURRENT_PROCESSES` is set, additional `/process` and `/process_unit` requests **wait** until a handler slot is free (they are not rejected with 503).

Vector mode unavailable:

```json
{
  "error_code": "VECTOR_STORE_UNAVAILABLE",
  "error": "..."
}
```

---

## Related

- [Configuration](configuration.md) — server and tool settings
- [LLM Caching](llm_caching.md) — disk cache, in-flight limits, batch pre-warming
- [User Instructions](user_instructions.md) — guiding extraction
- [Workflow](workflow.md) — what happens inside `/process`
