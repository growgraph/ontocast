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
  "facts_triples": 1204, "ontology_triples": 318
}
```

This is the offline option: no service, no account, and it survives the process.
Two runs are comparable by diffing their manifests — which model, which
generation settings, how many tokens, how long per node. It is also what makes a
benchmark sweep auditable after the fact, rather than something to be re-run.

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
