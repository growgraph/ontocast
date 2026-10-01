# HTTP API

This page lists every route the OntoCast server exposes, with its parameters,
responses and error codes. Start the server with `ontocast serve`; it listens
on `http://127.0.0.1:8999` by default
([`HOST`](configuration/pipeline.md#host), [`PORT`](configuration/pipeline.md#port)).

The server has no authentication, and `POST /flush` deletes data. Keep it on
the loopback interface unless a proxy that authenticates sits in front of it.

FastAPI serves interactive documentation at `/docs` and the OpenAPI schema at
`/openapi.json`.

## Routes

| Route | Purpose |
|---|---|
| `GET /health` | Liveness probe |
| `GET /info` | Version, accepted input types, cache statistics |
| `POST /process` | Extract an ontology and facts from one document |
| `POST /process_unit` | Extract from a short text as a single content unit, without chunking |
| `GET /ontologies` | List the catalog's ontologies |
| `POST /ontologies` | Add an ontology to the catalog |
| `PUT /ontologies/{iri}` | Replace a catalog ontology |
| `DELETE /ontologies/{iri}` | Remove a catalog ontology |
| `GET /shapes` | List stored SHACL shapes documents |
| `POST /shapes` | Add a SHACL shapes document |
| `DELETE /shapes/{graph_uri}` | Remove a SHACL shapes document |
| `POST /flush` | Delete stored facts, ontologies and vectors |
| `POST /match/entities` | Align entities across several graphs |
| `POST /match/derive-matches` | Turn alignment clusters into one-to-one entity matches |
| `POST /match/evaluate` | Score a predicted graph against a reference graph |

## Tenant and project

`/process`, `/process_unit`, `/ontologies`, `/shapes` and `/flush` work on one
tenant and project partition. Pass `tenant` and `project` as **query
parameters** to choose it; they are not read from a JSON body or form. With
neither, a route uses the partition the server started with (`ontocast serve
--tenant ... --project ...`, `ontocast`/`test` by default). With only one, the
other takes its default. [Tenancy](../guides/tenancy.md) explains how
partitions map to datasets and collections.

## Health and info

### `GET /health`

Reports that the server is up and its LLM client was built. It does not
contact the LLM provider, the triple store or the vector store.

```json
{"status": "healthy", "version": "...", "llm_provider": "openai"}
```

Returns `503` with `{"status": "unhealthy", "error": "..."}` when the LLM
client is missing.

### `GET /info`

Returns `name`, `version`, `description`, `capabilities`, `input_types` (the
file extensions this install can convert), `output_types`, `llm_cache` (cache
statistics) and `max_concurrent_processes`
([`MAX_CONCURRENT_PROCESSES`](configuration/pipeline.md#max_concurrent_processes)).

## Extraction

### `POST /process`

Runs the whole pipeline on one document: convert, chunk, extract the
ontology, extract facts, merge. The request returns when the document is done.

Send the document in one of two ways:

| Content type | Body |
|---|---|
| `multipart/form-data` | One uploaded file (`.txt`, `.json`, or a suffix in [`CONVERTER_SUPPORTED_EXTENSIONS`](configuration/conversion.md#converter_supported_extensions) such as `.pdf`, `.docx` or `.md`; `/info` lists what this install accepts) plus any parameters as form fields |
| `application/json` | A JSON object. The document text is its `text` field, or else its longest top-level string field. A `url` field is recorded as the document's source |

Any other content type, or a form with no file, returns `400`.

#### Parameters

Each parameter is read from the query string, the JSON body or a form field.
The query string wins over the body, and the body wins over the server's
setting. A present but unrecognised value returns `400`; it is never replaced
by the default.

| Parameter | Value | Default |
|---|---|---|
| `render_mode` | `ontology`, `facts` or `ontology_and_facts` | [`RENDER_MODE`](configuration/pipeline.md#render_mode) |
| `ontology_context_mode` | `selected_single_ontology`, `selected_vector_search_ontology` or `fixed_single_ontology` | [`ONTOLOGY_CONTEXT_MODE`](configuration/pipeline.md#ontology_context_mode) |
| `ontology_context_fixed_ontology_id` | IRI, `ontology_id` or prefix of a catalog ontology. A non-empty value selects `fixed_single_ontology` whatever `ontology_context_mode` says | In fixed mode, [`ONTOLOGY_CONTEXT_FIXED_ONTOLOGY_ID`](configuration/pipeline.md#ontology_context_fixed_ontology_id) |
| `llm_graph_format` | `jsonld` or `turtle`: the format the model writes graphs in | [`LLM_GRAPH_FORMAT`](configuration/pipeline.md#llm_graph_format) |
| `max_visits` | Integer of at least 1: retries of a render that failed outright | [`MAX_VISITS_PER_NODE`](configuration/pipeline.md#max_visits_per_node) |
| `strip_provenance` | `true` to leave reification and provenance triples out of the returned Turtle | `false` |
| `ontology_user_instruction` | Text added to the ontology prompt | none |
| `ontology_selection_user_instruction` | Text added to the prompt that picks a catalog ontology | none |
| `facts_user_instruction` | Text added to the facts prompt | none |
| `target_sections` | Section labels to keep, comma-separated or a JSON array | all sections |
| `exclude_sections` | Section labels to drop. An empty value drops nothing, overriding the section schema's default exclusions | the section schema's default exclusions |
| `summarize_sections` | Section labels whose content units get a summary; `*` or an empty value summarizes every unit. Without `target_sections`, a list of labels also keeps only those sections | no summaries |
| `summary_max_sentences` | Positive integer: sentences per summary | `5` |
| `section_schema_id` | Section label schema: `academic`, `clinical`, `fiction`, `financial`, `general`, `legal`, `manual`, `news`, `patent` or `standard` | resolved from the document |
| `document_type_hint` | Free text describing the document, used to pick a section schema when `section_schema_id` is not given | none |
| `document_metadata` | JSON object (or a string holding one) of identity facts you assert about the document; see [Ontologies and facts](../concepts/ontologies_and_facts.md) | none |

Section labels in a list are matched loosely (case, spaces and hyphens are
ignored); unknown labels are dropped with a warning, and a list in which no
label is known returns `400`. [User instructions](../guides/user_instructions.md)
covers the three instruction fields.

To cap the number of content units each request processes, start the server
with `ontocast serve --head-chunks N`.

#### Examples

```bash
# Text in a JSON body
curl -X POST http://127.0.0.1:8999/process \
  -H "Content-Type: application/json" \
  -d '{"text": "Your document text here", "render_mode": "facts"}'

# File upload into a tenant partition, provenance stripped
curl -X POST "http://127.0.0.1:8999/process?tenant=acme&project=reports&strip_provenance=true" \
  -F "file=@document.pdf"

# Document identity metadata
curl -X POST http://127.0.0.1:8999/process \
  -F "file=@document.pdf" \
  -F 'document_metadata={"doi": "10.1234/example", "title": "Example Report", "author": ["Jane Doe"], "project": {"name": "Example Project", "identifier": "PRJ-1"}, "identifiers": [{"scheme": "example:doc", "value": "DOC-1"}]}'

# Keep only the methods and results sections
curl -X POST http://127.0.0.1:8999/process \
  -F "file=@paper.pdf" -F "target_sections=methods,results"
```

#### Response

```json
{
  "status": "success",
  "data": {
    "facts": "@prefix cd: <...> . ...",
    "ontology": null,
    "ontology_artifacts": [
      {"iri": "https://...", "ontology_id": "...", "title": "...", "triples": 42, "ttl": "..."}
    ]
  },
  "metadata": {"status": "...", "chunks_processed": 7, "chunks_remaining": 0, "budget": {}, "...": "..."}
}
```

`data.facts` is the merged facts graph in Turtle; it is empty in `ontology`
mode. `data.ontology_artifacts` holds one entry per ontology the run created or
changed. `data.ontology` is always `null`.

| `metadata` field | Content |
|---|---|
| `status` | Final status of the run |
| `chunks_processed`, `chunks_remaining` | Content units processed and left unprocessed |
| `budget` | LLM calls, cache hits, characters, triple counts and per-node timings; see [Reading run telemetry](../guides/telemetry.md) |
| `retrieval_metrics` | Retrieval, extraction and validation counters; see [Telemetry counters](../internals/telemetry_counters.md) |
| `facts_repairs` | Rule-based rewrites applied to each content unit's facts, keyed by unit index |
| `failed_units` | Content units that produced no output, with stage and reason; empty on a clean run |
| `improvement_suggestions` | Advisory notes from the structural check and the consistency critic; nothing acts on them |
| `facts_conformance` | Validation summary of the returned facts: whether SHACL ran and the graph conforms, counts by finding kind, constraint and shape, repairs applied; see [Validation](../guides/validation.md) |
| `facts_validation_findings` | The findings behind that summary, after every repair |
| `facts_gate_repairs` | LLM-free repairs applied to the merged graph |

A run in which no content unit produced output returns `422`, not an empty
`200`.

### `POST /process_unit`

Treats the whole input as one content unit: no chunking, section tagging,
summarization or normalization. Use it for a single passage, or to debug
prompts. It takes the same body and parameters as `/process`; the section
parameters have no effect.

```bash
curl -X POST http://127.0.0.1:8999/process_unit \
  -H "Content-Type: application/json" \
  -d '{"text": "Single paragraph to process."}'
```

The response has the shape of `/process`. `ontology_artifacts` holds at most
one entry, the triples the unit added. The facts go through the same
validation gate as in `/process`, except the repair that splits wrongly merged
entities, which needs more than one unit. A failed unit returns `422`.

## Ontology catalog

These routes manage the catalog of the selected tenant and project. Uploads
are Turtle files in a `file` form field. The path IRI is URL-encoded.

| Route | Behavior |
|---|---|
| `GET /ontologies` | Lists the current version of each ontology: `iri`, `ontology_id`, `title`, `description`, `version` and `hash` |
| `POST /ontologies` | Adds the ontology. Returns `iri`, `ontology_id`, `version` and `hash` |
| `PUT /ontologies/{iri}` | Replaces the ontology. The uploaded file must declare the same ontology IRI as the path, or the call returns `400` |
| `DELETE /ontologies/{iri}` | Removes the ontology. Returns its `iri` |

```bash
curl "http://127.0.0.1:8999/ontologies?tenant=acme&project=reports"

curl -X POST "http://127.0.0.1:8999/ontologies?tenant=acme&project=reports" \
  -F "file=@my-ontology.ttl"

curl -X DELETE "http://127.0.0.1:8999/ontologies/https%3A%2F%2Fexample.org%2Fonto"
```

## Shapes

These routes manage the SHACL shapes that extracted facts are validated
against. Shapes are stored in their own partition, apart from the catalog; see
[Validation](../guides/validation.md).

| Route | Behavior |
|---|---|
| `GET /shapes` | Returns `graph_uris`, the stored documents, and `triples`, the size of all shapes merged |
| `POST /shapes` | Adds a Turtle shapes document from the `file` form field. A document that declares `<iri> a owl:Ontology` is stored under that IRI, so uploading it again replaces it; otherwise it is named after the file. Returns `graph_uri` and `triples` |
| `DELETE /shapes/{graph_uri}` | Removes the document. Documents loaded from [`FACTS_SHAPES_DIR`](configuration/facts-validation.md#facts_shapes_dir) come back at the next start |

```bash
curl -X POST "http://127.0.0.1:8999/shapes?tenant=acme&project=reports" \
  -F "file=@my-shapes.ttl"
```

## Flush

### `POST /flush`

Deletes stored data. It cannot be undone.

| Query parameters | What is deleted |
|---|---|
| none | Facts and ontologies in the triple store partition the server started with |
| `tenant` and/or `project` | Facts and ontologies in that partition's triple store, and its vector-store collection. Returns `400` if the triple store has no partitions |
| `include_shapes=true` | Also the shapes partition. Off by default: without shapes the SHACL gate stops running, and later runs report `shacl_evaluated: null` instead of failing |

```bash
curl -X POST "http://127.0.0.1:8999/flush?tenant=acme&project=reports"
```

## Graph matching

These routes compare graphs, for example an extracted graph against a
reference. The `match-graphs` command drives them; see [Entity
disambiguation](../concepts/entity_disambiguation.md). Graphs are Turtle
strings in the JSON body.

| Route | Body | Returns in `data` |
|---|---|---|
| `POST /match/entities` | `graphs` (a list of `{id, graph}`), `regime` (`ontology_loose` (default) or `ontology_strict`), `similarity_threshold` (0 to 1; default `AGG_SIMILARITY_THRESHOLD`), `embedding_model` (default `AGG_EMBEDDING_MODEL`) | Entity clusters across the graphs |
| `POST /match/derive-matches` | `clusters` (from `/match/entities`), `predicted_graph_id`, `gt_graph_id`, `similarity_threshold` (default `0`) | `entity_matches`: one-to-one predicted-to-reference matches |
| `POST /match/evaluate` | `predicted_graph`, `gt_graph`, `entity_matches` | Triple, fact and entity precision, recall and F1. `rdfs:label` triples are not counted; entities are the subjects and objects that are not vocabulary. A ratio with a zero denominator (nothing predicted, or no reference) is `null` |

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

## Errors

Error responses share one shape:

```json
{"status": "error", "error": "...", "error_type": "...", "error_code": "..."}
```

`/process` and `/process_unit` add `error_details` where they have a failing
stage or per-unit failures. A body that does not match a `/match/*` model, or
an upload route called without `file`, gets FastAPI's standard `422` with a
`detail` list instead.

| Status | Cause | `error_code` |
|---|---|---|
| `400` | Unsupported content type, or a form with no file | none |
| `400` | Unrecognised `render_mode`, `ontology_context_mode`, `llm_graph_format` or `strip_provenance` value | `invalid_param:<name>` |
| `400` | A blank `tenant` or `project` | `invalid_param:<name>` on `/process`, `/process_unit` and `/flush` |
| `400` | `max_visits` or `summary_max_sentences` not a positive integer; `document_metadata` not a JSON object; a section list that is malformed JSON or names no known label | `invalid_param:<name>` |
| `400` | `fixed_single_ontology` mode with no ontology id in the request or in `ONTOLOGY_CONTEXT_FIXED_ONTOLOGY_ID` | none |
| `400` | No store configured on `/flush`; a Turtle file that does not parse, or whose ontology IRI does not match the path, on `/ontologies` or `/shapes` | none |
| `409` | `selected_vector_search_ontology` requested but no vector store is configured and ready | `VECTOR_STORE_UNAVAILABLE` |
| `422` | No content unit produced output, including when the document could not be converted (`/process`) | `no_units_extracted` |
| `422` | The uploaded file could not be converted (`/process_unit`) | `conversion_failed:<stage>` |
| `422` | The single unit failed (`/process_unit`) | none |
| `422` | A section selection (`target_sections`, `summarize_sections`, or `exclude_sections` including the schema's default exclusions) removed every part of the document, with [`CHUNK_SECTION_FILTER_ON_EMPTY`](configuration/chunking.md#chunk_section_filter_on_empty)`=error`; with the default `warn`, the run continues with nothing to extract (`/process`) | `empty_section_selection:<name>` |
| `500` | Any other failure during processing or in a store | none |
| `503` | `/health` only: the LLM client is missing | none |

## What to read next

- [Configuring OntoCast](../guides/configuration.md): the server settings behind each default.
- [User instructions](../guides/user_instructions.md): steering extraction per request.
- [How OntoCast works](../concepts/index.md): what happens inside `/process`.
