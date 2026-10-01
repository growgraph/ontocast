# Troubleshooting

This page starts from what you see, a missing graph, an error code or a slow
run, and says what usually causes it and what to change. Before changing
anything, read the run manifest (`<name>.run.json` from `ontocast process
--output-dir`) or the `metadata` of the `/process` response: it records the
settings the run used, so it separates "the model did badly" from "that stage
never ran".

| Symptom | Likely cause | What to change or check |
|---|---|---|
| `422` from `/process`, or no facts at all | Every part failed, the document did not convert, or facts were never requested | [No facts, or a 422](#no-facts-or-a-422) |
| Run stops with `EmptyOntologyContextError` | A part was shown no ontology, and the run requires one | [Empty ontology context](#empty-ontology-context) |
| Facts-only run produced almost nothing | Empty catalog, or a fixed ontology id that matches nothing | [Facts-only run produced nothing](#facts-only-run-produced-nothing) |
| New terms invented instead of yours | Catalog not loaded, or the wrong ontology shown to each part | [Extraction ignores my ontology](#extraction-ignores-my-ontology) |
| Duplicates survive, or distinct entities are merged | Disambiguation threshold | [Duplicate entities or wrong merges](#duplicate-entities-or-wrong-merges) |
| SHACL violations, or SHACL never runs | Shapes or the `shacl` extra missing; violations the autofix cannot repair | [SHACL violations](#shacl-violations) |
| `429` errors from the provider, or slow runs | Concurrency above the provider's limits | [Rate limits and slow runs](#rate-limits-and-slow-runs) |
| A setting has no effect | Not in the process environment, or overridden | [Settings seem ignored](#settings-seem-ignored) |
| `409` with `VECTOR_STORE_UNAVAILABLE` | Vector store not configured or not initialized | [Vector mode returns 409](#vector-mode-returns-409) |
| Model output does not parse | A syntax error the parser could not repair | `llm/parse_retry` and `llm/parse_abandoned` in `budget.counters`; see [How LLM responses are parsed](../internals/llm_responses.md) |
| Warning that the ontology context is still over budget | The ontology is too large to show whole | Split the catalog, or switch to `selected_vector_search_ontology`; see [Choosing ontology context](ontology_context.md) |
| Parts split mid-argument, or too coarse | Chunk size bounds | [`CHUNK_MIN_SIZE`](../reference/configuration/chunking.md#chunk_min_size), [`CHUNK_MAX_SIZE`](../reference/configuration/chunking.md#chunk_max_size) |
| Reference list extracted as domain facts | Bibliography routing | [`CHUNK_BIBLIOGRAPHY_MODE`](../reference/configuration/chunking.md#chunk_bibliography_mode) |
| Gaps inside words in PDF text (`di ff usion`) | Converter profile | `CONVERTER_PROFILE=born_digital` |
| Memory use higher than expected | Several local embedding models loaded | [Performance tuning](performance.md) |

## No facts, or a 422

`/process` answers `422` when no part of the document produced output. The
body says why. With `error_code` `no_units_extracted`, `error_details` gives the
failing stage and reason, and `unit_failures` lists each part that failed; a
document that could not be converted fails here too. An `error_code` starting
with `empty_section_selection` means `target_sections` or `exclude_sections`
left nothing to process. `/process_unit` reports a conversion failure as
`conversion_failed`.

For a conversion failure, check the file type against `input_types` in `GET
/info`: converted formats need the `doc-processing` extra, and
[`CONVERTER_SUPPORTED_EXTENSIONS`](../reference/configuration/conversion.md#converter_supported_extensions)
may narrow them.

A `200` with an empty `data.facts` has different causes:

- `RENDER_MODE=ontology`, which never extracts facts. `metadata.chunks_remaining` is `0` and the run manifest shows `render_mode`.
- Some parts failed. `metadata.failed_units` lists them with the stage and reason.
- Facts-only run against an empty or wrong catalog; see [below](#facts-only-run-produced-nothing).

## Empty ontology context

Each part of a document (content unit) is shown a slice of your ontologies, its
ontology context. When that slice is empty, an ontology part creates a new
ontology from the text, which is how OntoCast starts without a catalog. A facts
part has nothing to do that with: it falls back on generic vocabulary.

With [`ONTOLOGY_CONTEXT_REQUIRED=true`](../reference/configuration/pipeline.md#ontology_context_required)
an empty context for a facts part stops the run with `EmptyOntologyContextError`
instead. The message names the cause; `retrieval_metrics.empty_snapshot_reason`
records it too. The usual causes:

- The catalog is empty. The startup log names the seed directory OntoCast used and ends the sync with `Ontology sync finished: N ontolog(ies)`.
- In fixed mode, the id matches no ontology; see [below](#facts-only-run-produced-nothing).
- In vector mode, the index and the catalog disagree: the index holds ontologies while the triple store holds none, or the other way round. OntoCast refuses to start in both cases. Load the ontologies into the triple store, or rebuild the index with `--wipe-vector-store`.

## Facts-only run produced nothing

With `RENDER_MODE=facts` no ontology terms are created, so the run depends
entirely on the catalog. Two configurations produce almost nothing without
failing:

- **An empty catalog.** `ontocast process` logs a warning and extracts in generic vocabulary. Seed the catalog with `--ontology-dir` or `ONTOCAST_ONTOLOGY_DIRECTORY`, or run `ontology_and_facts` once.
- **A fixed id that matches nothing.** In `fixed_single_ontology` mode, an id that names no ontology logs `No catalog ontology match for ontology_context_fixed_ontology_id` and continues with an empty ontology. The id can be the ontology's IRI, its ontology id or its prefix. Check it after renaming an ontology.

Set `ONTOLOGY_CONTEXT_REQUIRED=true` to turn both into errors: `ontocast
process` then refuses to start on an empty catalog, and a part with no ontology
context stops the run.

## Extraction ignores my ontology

When the output invents terms that your ontology already has:

1. **Check the catalog loaded.** The startup log names the seed directory, or says none is configured, and reports how many ontologies the sync found. A directory that does not exist stops `ontocast process` and only warns under `ontocast serve`. On a server, a request with `?tenant=` or `?project=` reads a different partition from one without, so an ontology uploaded to one is not seen by the other.
2. **Check which ontology each part saw.** In the default `selected_single_ontology` mode the model picks one ontology per part, and may pick another or none. Steer it with `ontology_selection_user_instruction` (see [Writing user instructions](user_instructions.md)), fix the ontology with `fixed_single_ontology`, or switch to `selected_vector_search_ontology` when parts need terms from several ontologies.
3. **Check the context fits.** `retrieval_metrics.ontology_snapshot_triples` is the size of the slice shown. [`ONTOLOGY_CONTEXT_MAX_TRIPLES`](../reference/configuration/pipeline.md#ontology_context_max_triples) caps it; over the cap, comments and definitions go first. In vector mode, raise [`VECTOR_STORE_TOP_K`](../reference/configuration/ontology-retrieval.md#vector_store_top_k) or [`VECTOR_STORE_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES`](../reference/configuration/ontology-retrieval.md#vector_store_induced_subgraph_max_total_triples).

Facts always go in the `cd:` namespace, typed with your ontology's classes.
A `cd:` entity is expected; a `cd:` class or property is not. See [Ontologies
and facts](../concepts/ontologies_and_facts.md).

## Duplicate entities or wrong merges

The same entity extracted from several parts is merged when the labels are
similar enough and no guard objects.
[`AGG_CANDIDATE_SIMILARITY_THRESHOLD`](../reference/configuration/aggregation.md#agg_candidate_similarity_threshold)
sets how similar:

- **Duplicates survive:** lower it.
- **Distinct entities are merged:** raise it, and keep [`AGG_LITERAL_CONFLICT_GUARD`](../reference/configuration/aggregation.md#agg_literal_conflict_guard) and [`AGG_INITIALS_DISTINCT_GUARD`](../reference/configuration/aggregation.md#agg_initials_distinct_guard) on.

`AGG_SIMILARITY_THRESHOLD` does not affect this: it is only a fallback for
matching entities across graphs.
`retrieval_metrics.facts_rejected_merges` counts merges the guards refused.
[Entity disambiguation](../concepts/entity_disambiguation.md) explains the
guards.

## SHACL violations

If `metadata.facts_conformance.shacl_evaluated` is `null`, SHACL never ran:
no shapes were loaded for the partition, or the `shacl` extra is not
installed. Load shapes with `FACTS_SHAPES_DIR`, `--shapes-dir` or `POST
/shapes`. `POST /flush` keeps shapes unless you pass `include_shapes=true`.

When SHACL runs and reports violations:

- `metadata.facts_validation_findings` lists each one, and `metadata.facts_gate_repairs` what was repaired.
- [`FACTS_SHACL_AUTOFIX`](../reference/configuration/facts-validation.md#facts_shacl_autofix) repairs what it can without an LLM call (`prune` by default). It never invents a value, so a node with real data but a missing required property stays a finding.
- The model is shown the shapes by default, so it can extract conforming facts in the first place. Check that [`FACTS_SHAPES_PROMPT_CONTRACT`](../reference/configuration/facts-validation.md#facts_shapes_prompt_contract) is not `off`.
- [`FACTS_CRITIC_PASSES`](../reference/configuration/facts-validation.md#facts_critic_passes) sets how many review calls each part gets; the critic runs once by default. [Validation layers](../internals/validation_layers.md) explains what it sees and when it stops.

[Validation and SHACL](validation.md) covers shapes and the gate in full.

## Rate limits and slow runs

[`PARALLEL_WORKERS`](../reference/configuration/pipeline.md#parallel_workers)
parts of one document run at once, and
[`LLM_MAX_INFLIGHT`](../reference/configuration/llm.md#llm_max_inflight) caps
the calls in flight across all documents. When the provider answers `429`:

- Lower `LLM_MAX_INFLIGHT`. Lowering `PARALLEL_WORKERS` alone does not help a server that runs several documents at once.
- Set [`LLM_REQUESTS_PER_SECOND`](../reference/configuration/llm.md#llm_requests_per_second) to your provider tier's rate; a burst of short calls can exceed it with few calls in flight.
- Raise [`LLM_MAX_RETRIES`](../reference/configuration/llm.md#llm_max_retries): the provider SDK retries with backoff. OntoCast does not retry rate-limited calls itself.

`llm/rate_limited` in `budget.counters` counts the throttles that got through.

For slow runs, the `budget` shows where the time went: `node_durations` per
stage and `calls_count` against `cache_hits`. A hung call holds a worker until
[`LLM_REQUEST_TIMEOUT_SECONDS`](../reference/configuration/llm.md#llm_request_timeout_seconds).
Each extra review or completion pass adds a call per part. [Performance
tuning](performance.md) explains the concurrency layers and what to measure.

## Settings seem ignored

- **The settings file is not read.** OntoCast reads only the process environment, never a `.env` file. Load the file into the environment of the command you run: `set -a; source .env; set +a`.
- **The server reads settings once.** Restart `ontocast serve` after a change.
- **A request overrides it.** `render_mode`, `ontology_context_mode`, `ontology_context_fixed_ontology_id`, `llm_graph_format` and `max_visits` sent with a request win over the environment.
- **Dataset names come from the tenant and project.** `ontocast serve` and `ontocast process` name the Fuseki datasets and vector collections after the tenant and project (`--tenant`, `--project`), whatever `FUSEKI_DATASET` or a table setting says, and log a warning naming the ignored setting. See [Tenancy](tenancy.md).
- **Fixed mode needs the mode setting.** For `ontocast process`, set `ONTOLOGY_CONTEXT_MODE=fixed_single_ontology` as well as `ONTOLOGY_CONTEXT_FIXED_ONTOLOGY_ID`.
- **The name is wrong.** An unknown variable is ignored without a warning. Check the spelling against the [configuration reference](../reference/configuration/index.md); the run manifest shows the values the run used.

## Fuseki refuses the connection

`FusekiAccessError: ... answered 401 creating dataset` means the server wants
credentials OntoCast did not send, or rejected the ones it did. Set
`FUSEKI_AUTH` to a user with admin rights, or allow anonymous access in the
server's `shiro.ini`. See [Triple stores](triple_stores.md).

## Vector mode returns 409

`409` with `error_code: VECTOR_STORE_UNAVAILABLE` means the request asked for
`selected_vector_search_ontology` and this server has no working vector store.
The `error` text includes the last initialization error, if there was one.

- **None configured.** Set `LANCEDB_ENABLED=true` (`lancedb` extra) or `QDRANT_URI` (`qdrant` extra).
- **Configured but never initialized.** The index is built at startup only when the server starts in vector mode. To send vector-mode requests, start the server with `ONTOLOGY_CONTEXT_MODE=selected_vector_search_ontology`.
- **Initialization failed.** A Qdrant server that is down, or embedding settings that do not match the stored index. After changing `EMBEDDING_MODEL_NAME`, `EMBEDDING_DIMENSION` or the query and document prefixes, restart with `--wipe-vector-store` to rebuild the index.

[Choosing ontology context](ontology_context.md) explains when vector mode is
worth setting up.
