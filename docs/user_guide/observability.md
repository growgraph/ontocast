# Observability

Three layers, in increasing order of setup cost. OntoCast owns the first two and
deliberately owns none of the third.

| Layer | What it answers | Needs |
|---|---|---|
| `BudgetTracker` | What did this run cost, and where did the time go? | Nothing — always on |
| Run manifest | Same, but durable and comparable across runs | `--output-dir` |
| External tracing | What did each node and each LLM call actually do? | An env var and a collector |

## 1. In-run telemetry

Every run accumulates a `BudgetTracker`: LLM calls, cache hits, tokens, triples,
per-node wall clock, and effective-worker ratios. It comes back as
`metadata.budget` on `/process` and `/process_unit`, and is logged at `INFO` on
completion.

Reading it is covered in [Performance](performance.md) — the duration-key
convention, effective workers versus event-loop lag, and the token fields
(billed versus replayed, reasoning, provider-cache reads). Two of them are
derived rather than accumulated, and are the ones to read first:
`prefix_cache_hit_rate` and `reasoning_share_of_output` say whether a cost
problem lives in the prompt or in the model's thinking budget.

### Retrieval metrics

Alongside the budget, a run accumulates `retrieval_metrics`: counters covering
how ontology context was assembled, what the facts fan-out produced, and what
the validation gate found. It is returned as `metadata.retrieval_metrics` on
`/process` and `/process_unit`, and written into the run manifest below.

The keys are a wire contract, enumerated by `RetrievalMetric` in
`ontocast/onto/enum.py` — one registry rather than string literals spread over
three modules, where a typo used to mean a silently missing metric.

| Key | Meaning |
|-----|---------|
| `ontology_context_mode` | Which context mode resolved this unit's ontology |
| `patch_retrieval` | Nested per-retrieval telemetry from the patch retriever (query counts, atom funnel, seeds by ontology, and `ontology_rank_diagnostics` when `ONTOLOGY_PATCH_DUMP_ONTOLOGY_RANKS` is on) |
| `empty_snapshot_reason` | Why a unit's ontology snapshot came back empty. Written per unit, last writer wins |
| `ontology_writable_count` / `ontology_primary_units` | Writable anchors, and units assigned a primary anchor |
| `ontology_snapshot_triples` | Size of the ontology snapshot a unit was actually shown, written in **every** context mode. Previously only the vector resolver recorded a size, nested under `patch_retrieval` — so the two modes that bound nothing also reported nothing. Read it against `ONTOLOGY_CONTEXT_MAX_TRIPLES` to tell a condensed snapshot from an unbounded one |
| `facts_anchor_count` / `facts_anchor_units` | Same for the facts fan-out |
| `facts_critic_fixes_applied` / `_residual` / `_noop` | Proposed fixes that reached the graph, that still need judgement, and that removed exactly what they re-added. A critique dominated by the last is a critic producing motion rather than corrections |
| `facts_critic_patches_rolled_back` | Passes undone for leaving the unit worse. Non-zero means the critique is provoking data-destroying edits |
| `facts_repair_delete_only` | Repair renders rolled back for answering the findings prompt with deletions instead of an in-place rewrite. Non-zero means the findings prompt or the validator is provoking data-destroying responses — treat it as a release blocker, not a curiosity |
| `facts_findings_residual` / `facts_mandatory_residual` | Deterministic findings still open after the last critic pass, over every unit; the mandatory subset is the number that tracks defects |
| `facts_critic_calls` / `facts_critic_accepted` | The facts critic's ledger: calls billed, and calls whose verdict let the unit exit the loop |
| `ontology_findings_residual` / `ontology_mandatory_residual` | Same residuals for the ontology loop's delta validator (shadow mode — recorded, not yet gating) |
| `ontology_critic_calls` / `ontology_critic_accepted` | The ontology critic's ledger under its incumbent `success or score > 90` gate |
| `facts_rejected_merges` | Candidate merges the aggregator's guards refused, for the graph actually served |
| `facts_merge_repair_passes` / `facts_merge_vetoes` / `facts_merge_repairs_rejected` | Un-merge repair loop: accepted passes, veto pairs accumulated, passes reverted |
| `validated_without_ontology_context` | Facts were validated with no catalog vocabulary at all |
| `facts_validation_findings` / `facts_validation_errors` | Residual findings and error-severity findings |
| `facts_shacl_violations_before` / `_after` | Fact-scoped violation counts around the autofix pass |
| `facts_shacl_repairs` / `facts_shacl_autofix_passes` / `facts_shacl_autofix_reverted` | Machine repairs applied, passes kept, and whether a pass was reverted |
| `structural_ontology_components_max` | Largest disconnected-component count over the stitched ontology |
| `consistency_conflicts` | Conflicts the consistency critic reported |

The SHACL and validation counters are written by one shared function, so
`/process` and `/process_unit` emit the same set for the same graph and their
batch dumps stay comparable. The SHACL group is present only when the autofix
pass actually ran — absent means "did not run", which is not the same as zero.

## 2. The run manifest

`ontocast process --output-dir DIR` writes `<stem>.run.json` beside each
`<stem>.facts.ttl`:

```json
{
  "source": "paper.pdf",
  "ontocast_version": "0.6.0",
  "render_mode": "ontology_and_facts",
  "loops": {"max_visits": 1, "max_critic_visits": null, "llm_repair_visits": 1},
  "graph_metrics": {"nodes": 130, "edges": 112, "components": 24, "largest_component": 61, "isolated_nodes": 18},
  "llm": {"provider": "ollama", "model_name": "qwen3.6", "temperature": 0.0, "think": true},
  "budget": {
    "calls_count": 42, "cache_hits": 0,
    "input_tokens": 380174, "output_tokens": 51203, "reasoning_tokens": 39880,
    "prefix_cache_hit_rate": 0.41, "reasoning_share_of_output": 0.78,
    "node_durations": {"Render Facts": 61.4, "Render Facts/unit_sum": 240.1},
    "counters": {"llm/parse_retry": 3, "llm/json_bracket_repair": 1, "llm/parse_abandoned": 0}
  },
  "selection": {"target_sections": null, "exclude_sections": ["references"],
                "summarize_sections": null, "summary_max_sentences": null,
                "bibliography_mode": "exclude"},
  "critic": {"calls": 34, "accepted": 6, "score_min": 55, "score_median": 79,
             "score_max": 98,
             "score_histogram": {"50-59": 2, "70-79": 9, "80-89": 12, "90-99": 5},
             "fix_severity_histogram": {"critical": 27, "important": 99}},
  "ontology_critic": {"calls": 0, "accepted": 0, "score_min": null,
                      "score_median": null, "score_max": null,
                      "score_histogram": {}, "fix_severity_histogram": {}},
  "facts_triples": 1204, "facts_triples_serialized": 557, "ontology_triples": 318,
  "retrieval_metrics": {
    "ontology_context_mode": "selected_vector_search_ontology",
    "facts_shacl_violations_before": 232, "facts_shacl_violations_after": 221,
    "facts_shacl_repairs": 8,
    "patch_retrieval": {"atoms_final": 96, "snapshot_triple_count": 1183}
  }
}
```

`budget.counters` holds named event counts, summed across unit workers. Three
of them cover how the LLM's JSON survived parsing, and they are worth reading on
any run against a new model:

| Counter | Meaning |
|---|---|
| `llm/parse_retry` | Renders re-issued because the previous response would not parse or validate. Each is a full re-extraction, so a non-trivial count is a real share of the bill. |
| `llm/json_bracket_repair` | Responses recovered by rewriting mismatched closing brackets. Non-zero means the model is emitting structurally broken JSON that the deterministic repair caught — the run is correct, but the prompt or the schema shape is provoking it. |
| `llm/parse_abandoned` | Calls given up on: retries exhausted, or the same JSON syntax error recurred and further attempts were judged pure spend. Every one of these is a content unit that contributed nothing. |

A silent `llm/parse_abandoned` used to be visible only as a `failed without
usable output` warning scrolling past in the logs; it is now in the manifest.

This is the offline option: no service, no account, and it survives the process.
Two runs are comparable by diffing their manifests — which model, which
generation settings, how many tokens, how long per node, and now what retrieval
and the validation gate did. `loops` records the **effective** per-unit budgets
(a `--max-visits` override whose flag silently failed to apply is detectable
from the dump, not only from call arithmetic), `critic` and `ontology_critic`
summarize each loop's LLM-critic decisions (call and accept counts, score
histogram, fix-severity histogram — the evidence a gate recalibration reads),
`selection` records which
sections the run was actually given — a `--target-sections` typo that matched
nothing is otherwise indistinguishable from a document that genuinely had no
such section — and `graph_metrics` summarizes the connectivity of the
serialized facts graph so fragmentation regressions surface per document. It is
also what makes a sweep of runs auditable after the fact, rather than something
to be re-run.

!!! warning "`facts_triples` is not the size of the `.facts.ttl` beside it"
    `facts_triples` counts the aggregated graph in memory, provenance included;
    `facts_triples_serialized` counts what the dump actually holds, after the
    provenance split. The two routinely differ by a factor of several for the
    same document. Compare runs on `facts_triples_serialized` when the question is
    about extracted content, and on `facts_triples` when it is about pipeline
    volume — mixing them silently compares a graph with its own subset.

    Read `critic.calls` before `critic.accepted`. At the default
    `FACTS_CRITIC_PASSES=0` the critic never runs and `summarize_loop` returns an
    all-zero record, so `accepted: 0` there means *nothing was judged*, not
    *everything was rejected*. Score buckets are decade ranges keyed
    `"70-79"`, so an empty `score_histogram` alongside `calls > 0` means the
    critic ran and returned no parseable score.

`retrieval_metrics` carries the same payload the HTTP response returns, so a
batch run is no longer the blind path: before this, retrieval telemetry existed
only over HTTP, which left `ONTOLOGY_PATCH_DUMP_ONTOLOGY_RANKS` with no reader
for anyone running `ontocast process`.

Two blocks make an arm self-describing, so a sweep of output directories can be
compared without reconstructing each run's environment:

- **`validation_config`** records the validation-facing knobs the run actually
  used — `context_from_units`, `json_mode`, `shapes_prompt_contract`,
  `shapes_triples` (size of the merged shapes partition; 0 means neither the
  gate nor the prompt contract had shapes), `shacl_inference`,
  `numeric_coverage_mandatory`, and `facts_user_instruction_chars` (length
  only; the text can carry deployment secrets and the dump is shareable).
- **`selection.labeled_units` / `unlabeled_units` /
  `section_label_histogram`** record whether section filters could act at
  all: `--exclude-sections` is a denylist over *labels*, so against mostly
  unlabeled units it is a no-op that the arm's name — and previously its
  manifest — would never reveal.

!!! note "Salvaged-unit counts reflect the post-patch verdict"
    The reduce warning `Parallel facts map salvaged output from non-converged
    loop(s)` counts units that exited their loop `FAILED`. Unit status is
    re-evaluated after the critic's patch is applied (and after a rollback),
    so a unit whose patch resolved every material defect exits `SUCCESS` and
    is not counted. Earlier builds set the status from the critic's
    *pre-patch* verdict, which overstated non-convergence.

The HTTP path already returns the same `budget` in its response, so no manifest
is written there.

## 3. External tracing

The pipeline is a LangGraph graph over LangChain chat models, so
**LangSmith, Langfuse and OpenTelemetry GenAI collectors instrument it with no
OntoCast code** — every node becomes a span and every LLM call a child span with
its prompt, response and token usage. Set the env vars and run:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=...
export LANGSMITH_PROJECT=ontocast

ontocast process --input-path doc.pdf --output-dir ./out
```

Langfuse and OTel collectors attach the same way, through LangChain's global
tracer — consult their own setup docs for the variables; nothing on the OntoCast
side changes.

Writing a bespoke trace format here would reinvent what these standardise, so
OntoCast does not. What it does do is stay out of the way:

- **Name the graph when embedding it.** `create_agent_graph(tools, name="ontocast")`
  — an unnamed subgraph appears as `LangGraph` in every trace. See
  [Embedding OntoCast](embedding.md#building-the-graph-yourself).
- **Caller config is merged, not replaced.** `make_ontocast_node` merges your
  `RunnableConfig`, so the callbacks, tags and run metadata a host application
  attaches survive into the subgraph.

!!! warning "Cached calls produce no provider span"
    A disk-cache hit returns before the provider is called, so it emits no LLM
    span. A replayed run therefore shows a near-empty trace even though the
    budget reports the full workload — the two disagree by design, not by bug.

    For a fully traced run, disable the cache:

    ```bash
    LLM_CACHE_ENABLED=false ontocast process --input-path doc.pdf
    ```

    Note this costs provider tokens. The cache hit *does* still carry token
    usage into the budget and onto the returned message
    (see [LLM Caching](llm_caching.md)), so cost accounting stays correct
    either way; it is only the span timeline that thins out.

## What is not here

Cost in currency. OntoCast reports tokens; converting them to money needs a
price table that goes stale, and the deployment already knows its own rates.
`prefix_cache_hit_rate` and `reasoning_share_of_output` are deliberately the
only derived figures here: both are dimensionless ratios of counts this run
actually observed, so neither can go stale or leak a cost basis. Anything that
needs a number with a currency on it — margin, plans, billing — belongs to the
deployment, which knows what it pays per model.
