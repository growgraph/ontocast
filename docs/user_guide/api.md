# API Endpoints

OntoCast exposes a FastAPI server (CLI entry point: `ontocast`). Default port: **8999**.

## Health and Info

### `GET /health`

Returns service health. Use for load balancers and readiness probes.

### `GET /info`

Returns version, configuration summary, and active backend information.

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
| `target_sections` | Comma-separated or JSON list; section prepare + keep only listed sections |
| `summarize_sections` | Section prepare + summarization; `*` or empty = all chunks |
| `summary_max_sentences` | Max sentences per summary when summarization runs (default `5`) |
| `section_schema_id` | Section label schema (`academic`, `financial`, `legal`, …) |
| `document_type_hint` | Free-text hint to resolve schema when `section_schema_id` is omitted |

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

**Response:** JSON with `data.facts` (Turtle), `data.ontology_artifacts` (list of ontology TTL payloads), and `metadata` (status, chunk counts, budget).

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

Clear triple-store data (and vector-store partitions when a vector backend is configured).

| Query params | Behavior |
|--------------|----------|
| *(none)* | `clean()` on the triple store's **active scope** — Fuseki facts + ontologies datasets for the configured tenant/project, or the in-memory partition currently selected |
| `tenant`, `project` | `clean_tenancy()` on triple store **and** vector store for that partition (both must support tenancy; returns `400` otherwise) |

```bash
# Flush active triple-store scope (server startup tenant/project)
curl -X POST http://localhost:8999/flush

# Flush a specific tenant/project partition (triple + vector when configured)
curl -X POST "http://localhost:8999/flush?tenant=acme&project=reports"
```

**Backends:**

- **Fuseki** — persistent datasets; scope follows configured or retargeted tenant/project names (`{tenant}--{project}--facts` / `--ontologies`).
- **In-memory** (default when `FUSEKI_URI` / `FUSEKI_AUTH` are unset) — clears the active pyoxigraph partition; data is not persisted across process restarts.

The `dataset` query parameter is **not** supported. Use `tenant` and `project` instead.

!!! warning
    This operation is irreversible.

---

## Graph Matching

Benchmark-oriented endpoints for entity alignment and evaluation. Used by the standalone `match-dirs` CLI.

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
- [User Instructions](user_instructions.md) — guiding extraction
- [Workflow](workflow.md) — what happens inside `/process`
