# API Endpoints

OntoCast exposes a FastAPI server (CLI entry point: `ontocast`). Default port: **8999**.

## Health and Info

### `GET /health`

Returns service health. This is a **liveness** signal, not a readiness one: it
does not reach the LLM provider, the triple store, or the vector store.

### `GET /info`

Returns service name, version, description, capabilities, supported input and
output types, LLM-cache status, and the concurrent-process limit.

---

## Document Processing

### `POST /process`

Runs the full document pipeline: convert → chunk → ontology map/reduce → facts map/reduce → serialize.

**Content types:**

- `application/json` — body must include a `text` field (or file references as supported)
- `multipart/form-data` — upload files (`file` field) or form fields

**Common query / form / JSON parameters:**

Every parameter below is read identically from the query string, a JSON body,
and a multipart form field. Precedence is **query parameter > JSON/form body >
the server's environment default**. A value that is present but unrecognised is
rejected with **400**; it is never silently replaced by the default.

| Parameter | Description |
|-----------|-------------|
| `tenant` | Tenant name for store partitioning (default: `ontocast`) |
| `project` | Project name (default: `test`) |
| `render_mode` | Which pipeline blocks run: `ontology`, `facts`, or `ontology_and_facts`. Defaults to `RENDER_MODE` — see [Render Mode](configuration.md#render-mode-render_mode) |
| `max_visits` | Per-request render/critic retry budget (≥ 1) |
| `strip_provenance` | When true, omit reification/provenance from returned Turtle |
| `llm_graph_format` | `jsonld` (default) or `turtle` for this request |
| `ontology_context_mode` | Per-request ontology context mode. **Ignored when `ontology_context_fixed_ontology_id` is non-empty** — see [Ontology Context Mode](configuration.md#ontology-context-mode-ontology_context_mode) |
| `ontology_context_fixed_ontology_id` | Required when mode is `fixed_single_ontology` (IRI, `ontology_id`, or author prefix). Setting it **forces** fixed mode regardless of `ontology_context_mode` |
| `ontology_user_instruction` | Guide ontology extraction |
| `ontology_selection_user_instruction` | Guide catalog ontology selection |
| `facts_user_instruction` | Guide facts extraction |
| `target_sections` | Comma-separated or JSON list; section prepare + keep only listed sections |
| `exclude_sections` | Comma-separated or JSON list; section prepare + drop listed sections |
| `max_chunks` | Cap the number of content units processed for this request |
| `summarize_sections` | Section prepare + summarization; `*` or empty = all chunks |
| `summary_max_sentences` | Max sentences per summary when summarization runs (default `5`) |
| `section_schema_id` | Section label schema (`academic`, `financial`, `legal`, …) |
| `document_type_hint` | Free-text hint to resolve schema when `section_schema_id` is omitted |
| `document_metadata` | JSON object (or stringified JSON) of caller-asserted document identity — DOI/ISBN, scheme+value ids, title, and typed entities for `author`/`project`/custom keys. See [Concepts — Document-level identity metadata](concepts.md#document-level-identity-metadata). |

**Examples:**

```bash
# JSON body
curl -X POST http://localhost:8999/process \
  -H "Content-Type: application/json" \
  -d '{"text": "Your document text here"}'

# PDF upload
curl -X POST http://localhost:8999/process \
  -F "file=@document.pdf"

# Document identity metadata with upload (typed author/project entities)
curl -X POST http://localhost:8999/process \
  -F "file=@document.pdf" \
  -F 'document_metadata={"doi":"10.1234/example","author":["Jane Doe"],"project":{"name":"Perovskite Survey","identifier":"PRJ-1"},"identifiers":[{"scheme":"erp:doc","value":"INV-1"}]}'

# Strip provenance from API Turtle output
curl -X POST "http://localhost:8999/process?strip_provenance=true" \
  -F "file=@document.pdf"

# Multi-tenant request
curl -X POST "http://localhost:8999/process?tenant=acme&project=reports" \
  -F "file=@document.pdf"
```

**Response:** JSON with `data.facts` (Turtle), `data.ontology_artifacts` (list of ontology TTL payloads), and `metadata`:

| Field | Meaning |
|-------|---------|
| `status` | Terminal workflow status |
| `chunks_processed` / `chunks_remaining` | Content-unit counts |
| `budget` | LLM call, cache-hit, character and triple counters, plus `node_durations` and `counters` timing telemetry — see [Performance](performance.md) |
| `retrieval_metrics` | Retrieval, extraction and validation-gate counters — see [Observability](observability.md#retrieval-metrics) for the key table |
| `facts_repairs` | Deterministic machine rewrites per unit index — lets you tell machine-altered triples from what the model asserted |
| `failed_units` | Units that produced no output, with phase, stage and reason. Empty on a clean run |
| `improvement_suggestions` | Advisory notes from the structural check and consistency critic. Nothing in the pipeline acts on them |
| `facts_conformance` | Validation summary for the served graph: whether SHACL ran and it conforms, counts by finding kind / constraint component / shape, repairs applied — see [Validation](validation.md) |
| `facts_validation_findings` | Residual findings behind the summary, after every repair stage |
| `facts_gate_repairs` | LLM-free repairs the gate applied to the merged graph (retype, code resolution, prune, literal-variant dedupe) |

A run in which *no* unit produced output returns **422**, not a 200 with empty
facts.

---

### `POST /process_unit`

Runs the ontology and/or facts loop for a **single content unit** without the full document graph. Useful for debugging prompts and unit-level behavior.

Accepts the same parameters as `/process` (including `strip_provenance`, user instructions, and ontology context settings). The post-aggregation validation gate runs here too — the response carries `facts_conformance`, `facts_validation_findings`, and `facts_gate_repairs`, and the served graph includes the LLM-free SHACL repairs. Only the un-merge repair is skipped: it re-aggregates retained units against each other, which has no meaning for a single unit.

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

## Shapes

Routes under `/shapes` manage the tenant's SHACL shapes partition — what the
facts validation gate validates against. Same optional `tenant` / `project`
query parameters as `/ontologies`. Shapes live in their own partition rather
than the ontologies dataset; see
[Validation](validation.md#why-shapes-are-not-stored-with-the-ontologies).

### `GET /shapes`

List the named graphs in the shapes partition, and the merged triple count.

### `POST /shapes`

Upload a SHACL shapes document (Turtle file). A document declaring
`<iri> a owl:Ontology` is stored under that IRI, so uploading it again replaces
it; one with no header is named after the uploaded filename.

```bash
curl -X POST "http://localhost:8999/shapes?tenant=ontocast&project=test" \
  -F "file=@my_shapes.ttl"
```

### `DELETE /shapes/{graph_uri}`

Remove a shapes document by graph URI (URL-encoded path segment). The seed
directory (`FACTS_SHAPES_DIR`) is untouched, so a seeded document returns on the
next restart.

---

## Triple Store Maintenance

### `POST /flush`

Clear triple-store data (and vector-store partitions when a vector backend is configured).

| Query params | Behavior |
|--------------|----------|
| *(none)* | `clean()` on the triple store's **active scope** — Fuseki facts + ontologies datasets for the configured tenant/project, or the in-memory partition currently selected |
| `tenant`, `project` | `clean_tenancy()` on triple store **and** vector store for that partition (both must support tenancy; returns `400` otherwise) |
| `include_shapes` | Also drop the shapes partition. **Off by default:** facts and ontologies come back from a rerun, but dropping shapes disarms the SHACL gate silently — later runs report `shacl_evaluated: null` instead of failing |

```bash
# Flush active triple-store scope (server startup tenant/project)
curl -X POST http://localhost:8999/flush

# Flush a specific tenant/project partition (triple + vector when configured)
curl -X POST "http://localhost:8999/flush?tenant=acme&project=reports"
```

**Backends:**

- **Fuseki** — persistent datasets; scope follows configured or retargeted tenant/project names (`{tenant}--{project}--facts` / `--ontologies` / `--shapes`).
- **In-memory** (default when `FUSEKI_URI` / `FUSEKI_AUTH` are unset) — clears the active pyoxigraph partition; data is not persisted across process restarts.

The `dataset` query parameter is **not** supported. Use `tenant` and `project` instead.

!!! warning
    This operation is irreversible.

---

## Graph Matching

Endpoints for entity alignment and for scoring an extracted graph against a reference one. Used by the standalone `match-graphs` CLI.

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
match-graphs \
  --gt ./reference \
  --predicted ./extracted \
  --url http://localhost:8999 \
  --regime ontology_strict \
  --similarity-threshold 0.8
```

---

## Error Responses

| Status | Condition |
|--------|-----------|
| `400` | Invalid parameters (e.g. missing fixed ontology id, malformed `document_metadata`, non-positive `summary_max_sentences`, unparseable section list) |
| `409` | Vector store unavailable when vector ontology mode requested |
| `422` | The uploaded document could not be converted |
| `500` | Processing or store errors |
| `503` | Server not ready (`/health`) |

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
