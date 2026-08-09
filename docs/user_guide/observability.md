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
(billed versus replayed, reasoning, provider-cache reads).

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
| `facts_anchor_count` / `facts_anchor_units` | Same for the facts fan-out |
| `facts_llm_repair_renders_total` / `_failed` | Finding-driven repair renders attempted, and those that crashed (a failed repair leaves the pre-repair graph and the unit still reports success) |
| `facts_findings_residual` | Deterministic findings still open after the last repair render |
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
  "llm": {"provider": "ollama", "model_name": "qwen3.6", "temperature": 0.0, "think": true},
  "budget": {
    "calls_count": 42, "cache_hits": 0,
    "input_tokens": 380174, "output_tokens": 51203, "reasoning_tokens": 39880,
    "node_durations": {"Render Facts": 61.4, "Render Facts/unit_sum": 240.1}
  },
  "facts_triples": 1204, "ontology_triples": 318,
  "retrieval_metrics": {
    "ontology_context_mode": "selected_vector_search_ontology",
    "facts_shacl_violations_before": 232, "facts_shacl_violations_after": 221,
    "facts_shacl_repairs": 8,
    "patch_retrieval": {"atoms_final": 96, "snapshot_triple_count": 1183}
  }
}
```

This is the offline option: no service, no account, and it survives the process.
Two runs are comparable by diffing their manifests — which model, which
generation settings, how many tokens, how long per node, and now what retrieval
and the validation gate did. It is also what makes a benchmark sweep auditable
after the fact, rather than something to be re-run.

`retrieval_metrics` carries the same payload the HTTP response returns, so a
batch run is no longer the blind path: before this, retrieval telemetry existed
only over HTTP, which left `ONTOLOGY_PATCH_DUMP_ONTOLOGY_RANKS` with no reader
for anyone running `ontocast process`.

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
