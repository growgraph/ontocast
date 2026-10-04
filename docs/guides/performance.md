# Performance tuning

This page helps you make runs faster and cheaper: which concurrency setting to
change when a provider throttles you or a document is slow, which settings make
a prompt shorter, and how to keep local embedding models from filling memory.
Every change here is something you can measure on your own runs; [Reading run
telemetry](telemetry.md) shows where to look.

## How work runs concurrently

Four settings bound how much work is in flight:

| Setting | What it bounds |
|---|---|
| [`MAX_CONCURRENT_PROCESSES`](../reference/configuration/pipeline.md#max_concurrent_processes) | `/process` and `/process_unit` requests the server runs at once. Further requests wait for a slot; they are not rejected. Unset means no limit |
| [`PARALLEL_WORKERS`](../reference/configuration/pipeline.md#parallel_workers) | Content units of one document processed at once |
| [`LLM_MAX_INFLIGHT`](../reference/configuration/llm.md#llm_max_inflight) | Provider calls in flight across the whole process, all documents together |
| [`LLM_REQUEST_TIMEOUT_SECONDS`](../reference/configuration/llm.md#llm_request_timeout_seconds) | How long one provider call may take before it is abandoned |

A content unit makes one LLM call at a time, so one document puts at most
`min(PARALLEL_WORKERS, LLM_MAX_INFLIGHT)` calls on the provider, and `K`
concurrent documents at most `min(K × PARALLEL_WORKERS, LLM_MAX_INFLIGHT)`.
Workers above the in-flight cap do not run; they queue, and OntoCast warns at
startup when `PARALLEL_WORKERS` exceeds `LLM_MAX_INFLIGHT`. `ontocast process`
handles its files one after another, so only the last three settings apply to
it.

A call that hangs holds a worker slot and an in-flight slot until the timeout
frees them. A timed-out call is sent once more, then the render fails and the
run counts it under `llm/timeouts`. Raise the timeout for slow local models or
reasoning models that write long answers.

### Which setting to change

| What you see | What to change |
|---|---|
| The provider answers `429`, or `llm/rate_limited` is above zero | Lower `LLM_MAX_INFLIGHT`, or pace requests (next section) |
| A document is slow, and the fan-out runs near `PARALLEL_WORKERS` wide with units waiting for a slot | Raise `PARALLEL_WORKERS`, and `LLM_MAX_INFLIGHT` if it is lower |
| A document is slow, the fan-out runs far narrower than `PARALLEL_WORKERS`, and `loop_lag_total` is high | Leave `PARALLEL_WORKERS` alone: CPU work is blocking every unit at once, and more workers only queue behind it |
| `llm/timeouts` is above zero | Raise `LLM_REQUEST_TIMEOUT_SECONDS`, or check the provider |
| A shared server runs out of memory or provider quota under many requests | Set `MAX_CONCURRENT_PROCESSES` |

[Measure your own runs](#measure-your-own-runs) explains how to read the fan-out
width and `loop_lag_total`.

## Provider rate limits and retries

A concurrency cap is not a rate cap: many short calls can exceed a provider
tier's requests per minute while few are ever in flight at once. Two settings
handle the rate:

- [`LLM_REQUESTS_PER_SECOND`](../reference/configuration/llm.md#llm_requests_per_second)
  paces the start of each request with a per-process token bucket. Set it a
  little below your tier's limit, leaving room for the provider SDK's own
  retries. Pacing happens inside an `LLM_MAX_INFLIGHT` slot, so it also lowers
  the effective concurrency, and the wait counts toward the request timeout.
  Unset means unpaced.
- [`LLM_MAX_RETRIES`](../reference/configuration/llm.md#llm_max_retries) is the
  retry budget of the provider SDK, which backs off and honors `Retry-After`
  on rate limits and dropped connections. Unset keeps the SDK's own default.
  Ollama ignores it.

OntoCast itself never retries a rate limit or a connection error: retrying at
that point would raise the request rate exactly when the provider asks for
less. A throttle that survives the SDK's retries fails that unit's render and
is counted under `llm/rate_limited`. Check that counter before you judge a
run's cost or quality: a throttled run's failure counts describe the throttle,
not the extraction. The run manifest records both pacing settings, so you can
tell a paced run from an unpaced one afterwards.

## What makes a prompt expensive

Each extraction call carries the same chapters in the same order: a fixed
preamble, the conformance requirements from your SHACL shapes, the ontology
chapter, then the task and the text of the content unit. The ontology chapter
is usually the largest part, and every call of every unit pays for it again.
The number of calls per unit is set by the critic and completion budgets, plus
one ontology selection call per unit in the default `selected_single_ontology`
mode (`llm/ontology_selection` in `budget.counters`); see
[Balance quality and cost](configuration.md#balance-quality-and-cost).

Every lever below changes the prompt text, so the first run after a change
misses the [LLM response cache](llm_caching.md). [How prompts are
built](../internals/prompt_construction.md) describes the mechanics.

### How the model reads and writes the graph

The defaults pair what the model reads with what it writes:

| Run | Ontology the model reads | Graph the model writes |
|---|---|---|
| `RENDER_MODE=facts` | A term sheet ([below](#the-ontology-chapter-as-a-term-sheet)) | Turtle, compact layout |
| `ontology` or `ontology_and_facts` | Turtle: its chapter follows the output format, because the loop patches what it reads | Turtle, compact layout |

[`LLM_GRAPH_FORMAT`](../reference/configuration/pipeline.md#llm_graph_format)
sets the syntax the model writes graphs in. **Keep the default, `turtle`.** It
spends fewer tokens per triple than JSON-LD, and an IRI object cannot turn into
a string by accident: `unit:NanoM` is an IRI and `"unit:NanoM"` is visibly a
literal. In JSON-LD a bare `"unit:NanoM"` is a literal too, and nothing in the
syntax warns the model. Switch to `jsonld` for a provider whose structured
output handles long strings worse than nested objects: escaping or truncating
a long string shows up as `llm/parse_retry` and `rdf/turtle_repair`. Read those
counters on your own documents before and after switching. The format can also
be set per request.

[`LLM_OUTPUT_LAYOUT`](../reference/configuration/pipeline.md#llm_output_layout)
sets the whitespace of the response. **Keep the default, `compact`.** It asks
for minified JSON and, under Turtle, one line per subject. Without it, models
indent their JSON, and every indentation token is billed output that the parser
discards. `free` exists to compare against; it changes no content, so compare
`output_tokens` with the layout on and off.

On a reasoning model, read `reasoning_share_of_output` first: reasoning tokens
are output that no encoding touches, so when they dominate, the reasoning budget
([`LLM_REASONING_EFFORT`](../reference/configuration/llm.md#llm_reasoning_effort))
matters more than either setting.

### The ontology chapter as a term sheet

[`ONTOLOGY_CHAPTER_FORMAT`](../reference/configuration/pipeline.md#ontology_chapter_format)
sets how the ontology chapter of the facts prompts is written. The model's own
output keeps `LLM_GRAPH_FORMAT`.

| Value | Chapter |
|---|---|
| `auto` (default) | `term_sheet` when `RENDER_MODE=facts`, `inherit` otherwise |
| `inherit` | A serialized graph in `LLM_GRAPH_FORMAT` |
| `turtle` | A serialized graph in Turtle, whatever the output format |
| `term_sheet` | One line per term: its name, other names, parent, domain and range, and one usage note. The shortest chapter |

A term sheet is legal only when `RENDER_MODE=facts`: the ontology loop edits
the statements in its chapter, so its chapter must stay a graph. An explicit
`term_sheet` with any other render mode stops OntoCast at startup.

### Size of the ontology context

[`ONTOLOGY_CONTEXT_MAX_TRIPLES`](../reference/configuration/pipeline.md#ontology_context_max_triples)
caps the triples in each chapter; [Keep the prompt in
bounds](configuration.md#keep-the-prompt-in-bounds) explains what it drops.
Do not lower it to shorten a term sheet: the triples it drops first never
appear in a term sheet, and the next ones it drops are the alternative names
and usage notes the sheet keeps on purpose. Use the text caps instead. In
`selected_vector_search_ontology` mode, retrieval bounds the context before
this cap does, and
[`ONTOLOGY_PATCH_SMALL_MODULE_CLOSURE_MAX_TOTAL_TRIPLES`](../reference/configuration/ontology-retrieval.md#ontology_patch_small_module_closure_max_total_triples)
caps how much whole small vocabularies add; see [Retrieval](../internals/retrieval.md).

### Text caps

Nothing else limits the length of a single label or comment, so a catalog with
long definitions makes every prompt long. Four settings bound the text in the
facts chapters:

| Setting | Bounds |
|---|---|
| [`ONTOLOGY_TEXT_MAX_CHARS_NAMING`](../reference/configuration/pipeline.md#ontology_text_max_chars_naming) | `rdfs:label`, `skos:prefLabel`, `skos:altLabel` |
| [`ONTOLOGY_TEXT_MAX_CHARS_CONTRACT`](../reference/configuration/pipeline.md#ontology_text_max_chars_contract) | `skos:scopeNote`, `skos:definition` |
| [`ONTOLOGY_TEXT_MAX_CHARS_PROSE`](../reference/configuration/pipeline.md#ontology_text_max_chars_prose) | `rdfs:comment` and the other SKOS notes |
| [`ONTOLOGY_TEXT_TOTAL_BUDGET`](../reference/configuration/pipeline.md#ontology_text_total_budget) | The summed length of all of them |

All four are unset by default, and unset they leave prompts byte-identical. A
cap shortens text and marks the cut with `…`; it never removes a statement.
Once you set a cap, the `chapter/text_chars_before` and
`chapter/text_chars_after` counters show whether it changed anything on your
catalog.

### One chapter per document, and warming the cache that serves it

Some providers bill a repeated prompt prefix at a reduced rate. OntoCast puts
the parts of a prompt that repeat first, so calls that see the same ontology
chapter share a prefix: the render and critic calls of one unit always do. In
an `ontology_and_facts` run, every facts unit also reads the same chapter, the
document's merged ontology. In a `facts` run each unit resolves its own
context, so units share a prefix only when they happen to get the same one.
Three settings make the whole fan-out share one:

- [`ONTOLOGY_CONTEXT_SCOPE=document`](../reference/configuration/pipeline.md#ontology_context_scope)
  shows every unit the union of all units' contexts. Each unit still sees
  every term its own retrieval chose, plus its siblings' terms, so each call
  is longer.
- [`FANOUT_WARMUP_UNITS=1`](../reference/configuration/pipeline.md#fanout_warmup_units)
  runs the first unit before the others. A cached prefix can be read only once
  the request that wrote it has completed, so units sent together all miss it.
  This is useful in `ontology_and_facts` runs too.
- [`LLM_PROMPT_CACHE_KEY`](../reference/configuration/llm.md#llm_prompt_cache_key)
  (OpenAI only) routes requests that share a prefix to the same cache. Use one
  fixed string per deployment, never one per request.

Keep [`FACTS_SHAPES_PROMPT_CONTRACT`](../reference/configuration/facts-validation.md#facts_shapes_prompt_contract)
at `full` with a document scope: `context` picks shapes per unit, which makes
the conformance chapter, and so the prefix, differ between units.

!!! warning "Check that the provider reads its cache"
    These settings pay off only if the provider serves cached prefixes and
    reports it. Turn them on, then read `prefix_cache_hit_rate` in the budget.
    If it stays at zero, the union chapter only adds tokens: unset all three.

## Local embedding models and memory

Three parts of OntoCast can load a local sentence-transformers model, each from
its own setting:

| Setting | Used by |
|---|---|
| [`CHUNK_EMBEDDING_MODEL`](../reference/configuration/chunking.md#chunk_embedding_model) | Semantic chunking and embedding-based schema detection |
| [`EMBEDDING_MODEL_NAME`](../reference/configuration/embeddings.md#embedding_model_name) | Ontology retrieval |
| [`AGG_EMBEDDING_MODEL`](../reference/configuration/aggregation.md#agg_embedding_model) | Entity disambiguation |

A process loads each distinct model name once and shares it between the
settings that name it. The name is compared as a literal string, so the same
model written two ways (with and without the `sentence-transformers/` prefix)
is loaded twice. With the defaults, chunking uses a different model from the
other two, so two models are resident. Name the same model in all three to
keep one:

```bash
export CHUNK_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
export EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
export AGG_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Before you change a model, know what else moves:

- Changing `CHUNK_EMBEDDING_MODEL` invalidates the chunk cache and moves chunk
  boundaries, so the same document is cut into different content units.
- Changing `EMBEDDING_MODEL_NAME` requires rebuilding the vector index, which
  stores vectors of that model's dimension.
- Retrieval adds
  [`EMBEDDING_DOCUMENT_PREFIX`](../reference/configuration/embeddings.md#embedding_document_prefix)
  and [`EMBEDDING_QUERY_PREFIX`](../reference/configuration/embeddings.md#embedding_query_prefix)
  to its texts; the other two uses do not. Sharing a model shares its weights,
  not its inputs.

Calls into one shared model run one at a time. That bounds peak memory when
many units and documents encode at once, and two different models never wait
for each other. To keep the retrieval model out of process entirely, set
[`EMBEDDING_PROVIDER`](../reference/configuration/embeddings.md#embedding_provider)
to `openai` or `ollama`.

## Measure your own runs

Every run reports a `budget`: `node_durations` in seconds and `counters` as
event counts, in the `/process` response and in the run manifest. Two
fan-out stages, `Update Ontology` and `Render Facts`, process units in
parallel. For each, `<stage>/unit_sum` divided by the stage's wall clock is the
number of units that effectively ran at once, logged as `Effective workers:
Render Facts <n>x`. Compare it with `PARALLEL_WORKERS`:

| Effective workers | Also high | Meaning |
|---|---|---|
| Close to `PARALLEL_WORKERS` | — | The stage runs at full width. Widen it, or shorten each unit's work |
| Well below | `<stage>/worker_wait` | Units wait for a slot. Raise `PARALLEL_WORKERS` |
| Well below | `<stage>/loop_lag_total` | Units are blocked by CPU work. More workers will not help |

`loop_lag_total` separates the two slow cases. Waiting for the provider lets
other units run and adds no lag, however slow the provider is; only work that
holds the CPU without yielding adds lag. `llm/provider` and `llm/inflight_wait`
show time spent in provider calls and queued behind `LLM_MAX_INFLIGHT`.

To measure a change to CPU-side work without paying for tokens, run the same
document twice:

```bash
ontocast process --input-path doc.pdf --head-chunks 15 --output-dir ./out
ontocast process --input-path doc.pdf --head-chunks 15 --output-dir ./out
```

The second run answers every call from the LLM response cache, so its stage
times are CPU and cache reads only, and its `cached_input_tokens` and
`cached_output_tokens` show what the first run cost. Vary `--head-chunks` to
see how a cost grows with the number of units. [Reading run
telemetry](telemetry.md) covers the token fields;
[Telemetry counters](../internals/telemetry_counters.md) lists every key.
