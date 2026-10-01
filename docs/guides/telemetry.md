# Reading run telemetry

Every run reports what it cost, where its time went, which parts of the
document failed, what ontology retrieval found and whether the facts passed
validation. This page shows where those reports are and which fields answer
each question. [Telemetry counters](../internals/telemetry_counters.md) lists
every key.

## Where to find it

| Where | What you get |
|---|---|
| `/process` and `/process_unit` responses | `metadata`: `status`, `budget`, `retrieval_metrics`, `failed_units`, `facts_conformance`, `facts_validation_findings`, `facts_repairs`, `facts_gate_repairs`, `improvement_suggestions` |
| `ontocast process` | Per document, beside `<name>.facts.ttl`: `<name>.run.json`, the run manifest, and `<name>.facts.validation.json` when validation has something to report |
| The log | A summary of calls, tokens, durations and effective workers at `INFO` when a document finishes |

`ontocast process` writes these files into `--output-dir` (or
`--facts-output-dir`), and next to the input file when neither is given. For a
`.jsonl` input with several records, each name carries the line number, such as
`<name>.L3.run.json`.

## What did it cost

`budget` counts calls and tokens:

| Field | Meaning |
|---|---|
| `calls_count` | Calls sent to the provider |
| `cache_hits` | Calls answered from the [LLM response cache](llm_caching.md) |
| `input_tokens`, `output_tokens` | Tokens billed for the calls sent |
| `cached_input_tokens`, `cached_output_tokens` | Tokens of cached answers: what those calls cost when they were first made. Not added to the billed totals |
| `reasoning_tokens` | Reasoning tokens, included in the output totals |
| `cache_read_input_tokens` | Input tokens the provider served from its own prompt cache, included in the input totals |
| `prefix_cache_hit_rate` | `cache_read_input_tokens` over all input tokens |
| `reasoning_share_of_output` | `reasoning_tokens` over all output tokens |

Providers report tokens differently, and one that reports none leaves the
token fields at `0` and both ratios at `null`. A `null` ratio means
unmeasured, not zero. The two ratios say where to aim a cost change: a high
`reasoning_share_of_output` points at the model's reasoning budget, a low
`prefix_cache_hit_rate` at prompt reuse; see [Performance
tuning](performance.md#what-makes-a-prompt-expensive). OntoCast reports tokens,
not money.

Before reading cost or quality, check three counters in `budget.counters`:

- `llm/rate_limited` and `llm/timeouts`: calls lost to throttling or to the
  request timeout. Each fails a render, so a run with many of them describes
  the provider, not the extraction.
- `llm/parse_abandoned`: calls whose answer never parsed. Each is a content
  unit that contributed nothing. `llm/parse_retry` counts the re-sent renders
  before that point; [How LLM responses are
  parsed](../internals/llm_responses.md) explains both.

## Where did the time go

`budget.node_durations` holds seconds per pipeline step. A plain step name,
such as `Chunk Text` or `Render Facts`, is that step's wall clock. The two
steps that process content units in parallel, `Update Ontology` and
`Render Facts`, have more keys:

| Key | Meaning |
|---|---|
| `<step>/unit_sum` | Time of every unit added up. Divided by the wall clock, it is the number of units that effectively ran at once |
| `<step>/worker_wait` | Time units waited for a `PARALLEL_WORKERS` slot |
| `<step>/loop_lag_total` | Time CPU work blocked every unit at once |
| `llm/provider` | Time inside provider calls |
| `llm/inflight_wait` | Time calls waited behind `LLM_MAX_INFLIGHT` |

[Measure your own runs](performance.md#measure-your-own-runs) explains how to
act on these.

## Which units failed

`metadata.failed_units` lists each content unit that produced nothing, with
its `unit_index`, `phase` (`ontology`, `facts` or `summarize`), `stage` and
`reason`. It is empty on a fully successful run. When no unit produced output,
`/process` answers `422` and `ontocast process` records the file as failed and
exits non-zero.

A request the provider refuses as configured, such as a wrong key, a model
your account cannot use, or an unsupported parameter, fails every call the
same way. OntoCast stops instead of finishing an empty run: the call is
counted under `llm/calls_rejected`, `/process` answers with an error, and
`ontocast process` exits with code `78` and writes no output files for that
document.

## Did retrieval find anything

`retrieval_metrics` describes the ontology context each content unit was shown:

| Key | What to look for |
|---|---|
| `ontology_context_mode` | The mode that chose the context |
| `ontology_snapshot_triples` | How large the context was. Compare it with [`ONTOLOGY_CONTEXT_MAX_TRIPLES`](../reference/configuration/pipeline.md#ontology_context_max_triples): at or above it, the context was condensed or passed through oversized |
| `empty_snapshot_reason` | Why a unit's context was empty. Only the last unit's reason is kept |
| `patch_retrieval` | In `selected_vector_search_ontology` mode, what retrieval matched: query count, terms kept (`atoms_final`), terms per ontology, and which whole small vocabularies were included or left out |

Set [`ONTOLOGY_PATCH_DUMP_ONTOLOGY_RANKS`](../reference/configuration/ontology-retrieval.md#ontology_patch_dump_ontology_ranks)
to add a per-query ranking of the ontologies to `patch_retrieval`.
[Retrieval](../internals/retrieval.md) explains what each number measures.

## Did validation pass

`metadata.facts_conformance` summarizes validation of the returned facts:
whether SHACL ran, whether the graph conforms, and counts by kind of finding.
`facts_validation_findings` lists the findings that remain after every repair,
and `facts_gate_repairs` and `facts_repairs` list the repairs OntoCast made
without the LLM. If `retrieval_metrics.validated_without_ontology_context` is
`true`, the facts were checked against no catalog vocabulary at all. The
`.facts.validation.json` file holds the same information for `ontocast
process`. [Validation](validation.md) explains the findings.

The run manifest's `critic` block records what the facts critic did. Read
`calls` before `accepted`: [`FACTS_CRITIC_PASSES`](../reference/configuration/facts-validation.md#facts_critic_passes)
is `1` by default, so `calls` is normally above zero, and `calls: 0` means the
critic was turned off or every unit skipped it. `accepted` counts the reviews
whose verdict let the unit leave the loop. The `ontology_critic`
block is the same for the ontology loop, whose critic is off by default.

## The run manifest

`ontocast process` writes one run manifest per document. It records the
settings, the cost and the outcome, so you can compare two runs by diffing
their manifests instead of rerunning them. The HTTP routes return the same
`budget` and `retrieval_metrics` but write no manifest.

```json
{
  "source": "paper.pdf",
  "ontocast_version": "<version>",
  "render_mode": "ontology_and_facts",
  "loops": {"max_visits": 1, "facts_critic_passes": 1, "ontology_critic_passes": 0},
  "llm": {"provider": "<provider>", "model_name": "<model>", "temperature": 0.0, "max_inflight": 16},
  "prompting": {"llm_graph_format": "jsonld", "ontology_chapter_format": "inherit",
                "ontology_context_scope": "unit", "fanout_warmup_units": 0,
                "parallel_workers": 16, "embedding_model_name": "<model>"},
  "budget": {"calls_count": 0, "input_tokens": 0, "output_tokens": 0,
             "node_durations": {"Render Facts": 0.0, "Render Facts/unit_sum": 0.0},
             "counters": {"llm/parse_retry": 0}},
  "selection": {"exclude_sections": ["references"], "bibliography_mode": "skip",
                "labeled_units": 0, "unlabeled_units": 0},
  "critic": {"calls": 0, "accepted": 0, "score_histogram": {}},
  "completion": {"calls": 0, "measurements_recovered": 0},
  "facts_triples": 0,
  "facts_triples_serialized": 0,
  "ontology_triples": 0,
  "retrieval_metrics": {"ontology_context_mode": "selected_single_ontology"}
}
```

The blocks you read most:

| Block | Answers |
|---|---|
| `llm`, `loops`, `prompting` | Which model and settings produced the run, as actually applied |
| `budget` | Cost and time, as above |
| `selection` | Which sections the run was given, how many units carried a section label, and how many were skipped as bibliography, too short or not content. An exclusion list cannot act on units that carry no label |
| `critic`, `ontology_critic`, `completion` | What the review and completion passes did |
| `validation_config` | The validation settings in effect, including the size of the loaded shapes |
| `graph_metrics` | How connected the written facts graph is: components, largest component, isolated nodes |

!!! warning "`facts_triples` is not the size of the `.facts.ttl` file"
    `facts_triples` counts the facts graph in memory, provenance included.
    `facts_triples_serialized` counts what the file holds after provenance is
    removed. Compare runs on `facts_triples_serialized` when the question is
    about extracted content.

## External tracing

OntoCast is a LangGraph graph over LangChain chat models and has no tracing
code of its own, so tools that trace LangChain show each pipeline step and
each LLM call with its prompt, answer and token usage. LangSmith is enabled
from the environment and works with `ontocast serve` and `ontocast process`:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=...
export LANGSMITH_PROJECT=ontocast
ontocast process --input-path doc.pdf --output-dir ./out
```

Langfuse and OpenTelemetry instrumentation attach as LangChain callbacks, which
OntoCast has no setting for. Attach them in your application when you embed
OntoCast: `make_ontocast_node` passes your callbacks, tags and run metadata
through to every step. See [Embedding OntoCast in your agent](embedding.md).

!!! note "Cached calls leave no LLM span"
    A call answered from the LLM response cache never reaches the provider, so
    it produces no LLM span, while `budget` still counts it. For a complete
    trace, set `LLM_CACHE_ENABLED=false`; every call is then billed.
