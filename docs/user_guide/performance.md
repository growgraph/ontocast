# Performance

OntoCast processes content units concurrently, but a fan-out that *looks* wide
can still behave like a serial loop. This page describes the telemetry that
tells the two apart, and the protocol for measuring a change without spending a
single provider token.

## The three concurrency layers

| Layer | Setting | Default | Bounds |
|---|---|---|---|
| Unit workers | `PARALLEL_WORKERS` | 16 | Content units in flight within one document |
| Provider calls | `LLM_MAX_INFLIGHT` | 16 | Concurrent provider requests, **process-wide** across all documents |
| Documents | `MAX_CONCURRENT_PROCESSES` | unset | Concurrent `/process` and `/process_unit` handlers |

A unit never issues two LLM calls at once, so within a *single* document the
effective provider concurrency is `PARALLEL_WORKERS`. Across `K` concurrent
documents it is `min(K x PARALLEL_WORKERS, LLM_MAX_INFLIGHT)` — which is why a
busy server can stop scaling with `PARALLEL_WORKERS` alone.

## Provider rate limits

Concurrency caps are not rate caps: a fan-out of short calls can exceed a
provider tier's requests-per-minute while never holding many connections at
once. Three knobs, three roles:

- **`LLM_REQUESTS_PER_SECOND`** (unset = unpaced) — the sustained rate, paced
  by a per-process token bucket on request *starts*. Set it from your
  provider tier with headroom (a 500 RPM tier is ~8 RPS; leave a margin for
  the SDK's own retries). The limiter is acquired inside an
  `LLM_MAX_INFLIGHT` slot, so aggressive pacing also lowers effective
  concurrency — that is the point.
- **`LLM_MAX_RETRIES`** (unset = each SDK's default: OpenAI 2, Anthropic 2,
  Google 6) — the provider SDK's transport-retry budget. The SDKs back off
  and honour `Retry-After`, so raising this is the correct response to
  residual throttling. The pipeline itself deliberately never retries
  transport failures: retrying at this layer multiplies the request rate
  exactly when the provider is asking for less.
- **`LLM_MAX_INFLIGHT`** — the burst ceiling, as above.

**What a throttle looks like.** A rate-limit error that survives the SDK's
retries fails the unit's render (one loop visit burned) and increments
`llm/rate_limited` in `budget.counters` — beside `llm/timeouts`, both in the
run manifest. **Read those counters before reading a run's cost or quality
figures**: a throttled run's token totals and failure counts describe the
throttle, not the pipeline. Both pacing knobs are recorded in the manifest's
`llm` block, so a paced run is distinguishable from an unpaced one after the
fact.

## Reading the metrics

Every run reports `budget.node_durations` (seconds) and `budget.counters`
(counts) in the `/process` response, and logs a summary at `INFO` on completion.

Duration keys follow a convention, because "how long did Render Facts take" has
two different answers:

| Key | Meaning |
|---|---|
| `<node>` | **Wall clock** for the node. Written only by the pipeline's node wrapper. |
| `<node>/unit_sum` | Per-unit loop time **summed over every worker**. Exceeds wall clock whenever the fan-out is doing its job. |
| `<node>/worker_wait` | Time units spent queued for a `PARALLEL_WORKERS` slot. |
| `<node>/loop_lag_total` | Time the event loop could not service ready callbacks. |
| `<node>/loop_lag_max` | Longest single such stall. Keys ending in `_max` take the maximum on merge, not the sum. |
| `llm/provider` | Time inside the provider call itself. |
| `llm/inflight_wait` | Time queued behind `LLM_MAX_INFLIGHT`. |
| `llm/cache_lookup` | Disk-cache read time. |

The headline number is **effective workers**, logged as
`Effective workers: Render Facts 4.0x (loop lag 8.0s)` and available
programmatically:

```python
budget.parallel_efficiency("Render Facts")  # unit_sum / wall clock
```

Compare it against `PARALLEL_WORKERS`:

- **Close to `PARALLEL_WORKERS`** — the stage is running at full width. To make
  it faster, widen the fan-out or reduce work per unit.
- **Well below, with high `worker_wait`** — units are queued. The width is the
  constraint; raise `PARALLEL_WORKERS`.
- **Well below, with high `loop_lag_total`** — units are *not* queued, they are
  being blocked. Synchronous CPU work on the event loop is stalling every unit
  at once, and **raising `PARALLEL_WORKERS` will not help** (it usually hurts,
  by piling more units onto the same serialized work).

`loop_lag` is the decisive signal because awaited I/O yields control and
therefore produces *zero* lag no matter how slow the provider is. A
`loop_lag_max` above ~0.3s is an unambiguous fingerprint of one long
synchronous block, and it cannot be confused with provider latency.

The accounting closes approximately:

```
wall(node) x effective_workers  ~=  sum(llm/provider)
                                  + sum(llm/inflight_wait)
                                  + sum(<node>/worker_wait)
                                  + <node>/loop_lag_total
                                  + residual
```

Named CPU suspects are timed individually so the lag can be attributed rather
than guessed at: `ctx/merge_document_ontology`, `ctx/snapshot_deepcopy`,
`ctx/working_graph_copy`, `prompt/ontology_index`, `prompt/ontology_chapter`,
`repair/deterministic`.

### Token counts

Token reporting is provider-dependent — a provider that stays silent leaves these
at zero, which is not the same as a run that used no tokens.

| Field | Meaning |
|---|---|
| `input_tokens` / `output_tokens` | **Billed**: live provider calls only. |
| `cached_input_tokens` / `cached_output_tokens` | Replayed from the OntoCast disk cache. Deliberately *not* added to the billed totals — a replay pays nothing — so these are what the workload would cost cold. |
| `reasoning_tokens` | Thinking tokens, counted **inside** the output totals. Dominates output cost for reasoning models (`LLM_THINK`). |
| `cache_read_input_tokens` | Served from the **provider's** prompt cache, counted inside the input totals and billed at a reduced rate. Unrelated to OntoCast's disk cache. |
| `cache_creation_input_tokens` | Written to the provider's prompt cache. |

`calls_count` counts billed calls and `cache_hits` counts replays, so a fully
replayed run reports `calls_count: 0` with non-zero `cached_*`.

### The two ratios that say where to aim a cost change

Two derived fields ride the same budget summary. Read them first: they decide
whether a cost problem is in the prompt or in the model's thinking budget, and
they cost nothing to look at.

| Field | Meaning |
|---|---|
| `prefix_cache_hit_rate` | `cache_read_input_tokens` over **all** input tokens, billed and replayed. |
| `reasoning_share_of_output` | `reasoning_tokens` over **all** output tokens. A decomposition of what was already paid for, never an addition to it. |

Both are `null` when the provider reported no tokens at all — unmeasured, which
is not the same as zero. Compute them from the fields yourself only if you must,
and mind the denominator: `reasoning_tokens` and `cache_read_input_tokens`
accumulate on billed *and* replayed calls, while `input_tokens` counts billed
only, so dividing by `input_tokens` alone can exceed 100% on a partly replayed
run.

**A low `prefix_cache_hit_rate` on a wide fan-out is expected, not a bug — and
it is money.** A provider cache entry only becomes readable once the first
response has begun, so `PARALLEL_WORKERS` units that issue together all miss the
prefix they share. The same code will show a far lower hit rate on a wide
document fan-out than on a longer, more sequential run — the difference is call
sequencing, not configuration, so compare this number only between runs of
similar shape.

**`reasoning_share_of_output` of `0.0` means the model is not a reasoning
model**, and no thinking-budget setting will change its bill; the lever is prompt
size or call count instead. On a reasoning model the share is typically large
enough that most of the output cost is thinking rather than triples, which makes
the thinking budget the first thing to tune.

### Counters

`budget.counters` records event counts. The one to watch for *concurrency*
regressions is `ctx/merge_document_ontology.calls`: the merged document ontology
depends only on document-level state, so this must be **1** per document. A
value that grows with the unit count means a per-unit regression has
reintroduced O(N) full rdflib merges into the fan-out.

The `llm/*` counters in the same map are about *spend* rather than concurrency —
`llm/parse_retry` is a re-issued render, `llm/parse_abandoned` is a unit that
contributed nothing. They are tabulated in
[Observability](observability.md#2-the-run-manifest), and the mechanism behind
them in [Configuration](configuration.md#what-happens-to-a-response-that-will-not-parse).

## Measuring a change without provider tokens

The LLM disk cache is on by default, so a document can be replayed exactly:

```bash
# 1. Populate the cache (costs tokens, once)
ontocast process --input-path doc.pdf --head-chunks 30 --output-dir ./out

# 2. Replay. Every call now hits cache, so llm/provider goes to ~0 and the
#    node wall clock becomes pure CPU plus cache I/O.
ontocast process --input-path doc.pdf --head-chunks 30 --output-dir ./out
```

The second run is the repeatable before/after number for any CPU-side change.
Vary `--head-chunks` (5, 15, 30) to check how a cost scales with unit count:
per-unit-invariant work shows up as a straight line through the origin, and it
should be flat instead.

The replay still reports the workload's token cost: cache entries carry the
provider's usage, so `cached_input_tokens` / `cached_output_tokens` on the second
run are what the first one paid. Entries written before usage was persisted report
nothing rather than zero — re-run once against the provider to refresh them.

## Local embedding models

Three subsystems use a local sentence-transformer, each with its own setting:

| Setting | Used by | Default |
|---|---|---|
| `CHUNK_EMBEDDING_MODEL` | semantic chunking, schema detection | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (~1.1 GB) |
| `EMBEDDING_MODEL_NAME` | dense retrieval | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (~458 MB) |
| `AGG_EMBEDDING_MODEL` | entity disambiguation | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (shared with the above) |

Checkpoints are cached process-wide by `(model name, device)`, so **settings that
name the same model share one resident copy** — at defaults that is two models.
The key is the literal string, so aligning them means matching the spelling
exactly: the same checkpoint written two ways loads twice, even though
`sentence-transformers` resolves a bare name and a prefixed one to the same
files. All three defaults now carry the `sentence-transformers/` prefix for
exactly this reason. Aligning all three drops it to one resident model:

```bash
CHUNK_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
AGG_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Aligning them holds **one** resident model instead of two. The saving is less
than the models' difference on disk, because both share the same vocabulary and
torch's allocator carries its own overhead either way — but it is the difference
between one checkpoint in memory and two.

Read [Configuration](configuration.md) first: changing `CHUNK_EMBEDDING_MODEL`
invalidates the chunk cache, shifts chunk boundaries, and affects the calibrated
schema-detection thresholds.

Inference on a shared model is **serialised per model**. That bounds peak
memory, which is what matters when `PARALLEL_WORKERS` units and several
documents encode at once — each concurrent encode would otherwise allocate its
own activation batch. On CPU it costs almost nothing, because parallel encodes
contend for one intra-op thread pool regardless. Two *different* checkpoints
never serialise against each other.

Note that sharing weights is not sharing semantics: retrieval applies the
`EMBEDDING_DOCUMENT_PREFIX` / `EMBEDDING_QUERY_PREFIX` instructions and the
other two do not. The retrieval model is also the one whose dimension is fixed
in the vector store's collection schema — changing it requires a reindex, while
changing the chunker's does not.

## Tuning

Fix the loop stall before widening the fan-out. Raising `PARALLEL_WORKERS`
while `loop_lag_total` is a large fraction of wall clock makes things worse, not
better — the extra units queue behind the same synchronous section.

Other knobs that change cost rather than concurrency:

- `MAX_VISITS` (default 1) — retries of a **failed** render, nothing else.
  Raising it costs nothing on units that render successfully, which is most of
  them.
- `FACTS_CRITIC_PASSES` (default 1) — the budget that actually buys critic
  calls. Each pass is one call and applies its own fixes, so a facts unit costs
  two calls at the defaults; each extra pass adds one. Passes stop early once
  one changes nothing. `ONTOLOGY_CRITIC_PASSES` defaults to `0`. See
  [Validation](validation.md#how-many-llm-calls-a-facts-unit-really-costs).
- `CONVERTER_PROFILE=born_digital` — skips OCR on digital PDFs.
- `ONTOLOGY_CONTEXT_MAX_TRIPLES` (default `4000`) — the budget for the ontology
  chapter in **every** mode. `ONTOLOGY_PATCH_MAX_ATOMS` and
  `VECTOR_STORE_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES` bound it further in vector
  mode, and bind first there.

## How much a triple costs

A triple's cost in the prompt is set by the wire format, and the two differ by
about a factor of two:

**`LLM_GRAPH_FORMAT=jsonld` roughly doubles chars per triple against `turtle`.**
That is the cost of the default: JSON-LD is more reliably parsed out of
structured output, and it buys that with context. If you are context-bound
rather than parse-bound, switching to `turtle` is the largest single lever
available — larger than any retrieval knob — and it changes no extraction
semantics, only the encoding. It does invalidate the LLM cache.

See [Configuration](configuration.md) for the full list and
[LLM Caching](llm_caching.md) for cache behavior.
