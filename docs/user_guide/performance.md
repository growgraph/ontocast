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
effective provider concurrency is `min(PARALLEL_WORKERS, LLM_MAX_INFLIGHT)`;
workers above the in-flight cap only queue on the semaphore, which shows up as
`llm/inflight_wait` in the node durations, and startup now warns when the
worker count exceeds the cap. Across `K` concurrent
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
| `reasoning_tokens` | Thinking tokens, counted **inside** the output totals. Dominates output cost for reasoning models; bounded by `LLM_THINK` (Ollama), `LLM_REASONING_EFFORT` (OpenAI and Gemini 3+) or `LLM_THINKING_BUDGET` (Gemini 2.5). When `reasoning_share_of_output` is large that knob is the first cost lever, not the prompt. |
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

The two calls a facts unit makes — render, then critic — carry the same
ontology chapter, and both prompts now open with the constant chapters
(preamble, conformance requirements) followed by that chapter, byte-identical,
before anything unit-specific. The critic's call can therefore be served the
prefix the render populated; `prefix_cache_hit_rate` is where that shows.

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
contributed nothing, `llm/calls_failed` is every call that raised (timeouts and
rate limits are its subsets), and a timed-out call is charged its prompt
characters so `calls_count = llm/calls_timed + llm/timeouts` holds. They are tabulated in
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

A triple's cost is set by the wire format. `LLM_GRAPH_FORMAT=turtle` spends
meaningfully fewer characters per triple than the `jsonld` default — it changes
no extraction semantics, only the encoding — and it invalidates the LLM cache.

Two things bound that saving, and both matter more than the ratio:

- **The encodings do not fail the same way.** JSON-LD is the default because it
  is parsed more reliably out of structured output; a nested payload that
  arrives with mismatched brackets is a lost render. Turtle removes that failure
  mode and introduces another — an IRI a model mints from a phrase containing a
  delimiter is a legal JSON string and not a legal Turtle IRI. Which one costs
  you more is a property of your model and your text, and the only way to know
  is to run both and compare `llm/parse_retry` and `llm/parse_abandoned`.
- **On a reasoning model, most of the output is not the graph.** Read
  `reasoning_share_of_output` first. Where it is high the encoding governs a
  minority of the output tokens, and the thinking budget is the larger lever.

`ONTOLOGY_CHAPTER_FORMAT=turtle` takes the density for the ontology chapter
alone, leaving the output wire as JSON-LD. Prefer `term_sheet` below: it is
cheaper still, and unlike the Turtle chapter it does not trade quality for the
saving.

### The ontology chapter as a term sheet

`ONTOLOGY_CHAPTER_FORMAT=term_sheet` goes further than a denser serialization
by not serializing the graph at all. **A facts-only run gets it by default** —
that is what the shipped `auto` resolves to; the rest of this section is what
that default buys, and how to opt out of it. The chapter becomes one line per term —
its name, the surface forms a document might spell it with, what it is, where
it sits in the hierarchy, what it connects, and the scope note saying when it
applies:

```
## Classes
  ex:PowderSample  "Powder sample"  < ex:Sample
## Properties  (domain -> range)
  ex:hasThickness  "has thickness"  ex:Sample -> ex:QuantityValue
## Individuals  (units, vocabulary values)
  ex:Approximate  "approximately"  : ex:EpistemicQualifier  ~ ~; ∼; ≈; ca.; about
```

The `~` list is why a term sheet can be cheaper without being poorer: those
surface forms are what a document match actually has to work with, they cost a
couple of dozen characters each, and a serialized chapter buries them under a
`skos:altLabel` predicate IRI per entry.

What that drops is the per-statement RDF scaffolding — a node wrapper or
subject block per term, a repeated predicate IRI per statement — which carries
nothing a reader of the sheet loses. What it keeps is everything that lets a
model pick a term, including one prose field each: the usage contract
(`skos:scopeNote`, `skos:definition`) where a term has one, and `rdfs:comment`
otherwise.

That fallback is deliberate and is not free — it is the difference between a
listing that re-encodes the chapter and one that cuts it. On a catalog where
most terms carry a comment and few carry a scope note, dropping comments would
reduce those terms to a name and a parent, and nothing downstream can recover a
description that was never shown. The caps below bound how long that prose may
be; nothing bounds it to zero.

This is admissible only because a facts prompt reads its ontology and writes an
unrelated graph. The ontology loop writes a *patch against the statements in
its chapter*, which a listing cannot express, so an explicit `term_sheet`
requires `RENDER_MODE=facts` and is rejected outright otherwise rather than
falling back to a graph. That restriction is also why the default is `auto`
rather than `term_sheet`: a mode-aware default can be the cheapest legal
chapter everywhere, where a fixed one would either fail on the ontology path or
overpay on the facts path. Set `ONTOLOGY_CHAPTER_FORMAT=inherit` to go back to
a chapter in the wire format.

### One chapter per document, and warming the cache that serves it

Even a cheap chapter is paid once per LLM call, and the facts pipeline makes
`units + critic passes + completion passes` of them per document. A provider's
prefix cache is the mechanism for paying for a repeated prefix once — but it can
only serve a prefix that actually repeats, and by default no two calls in a
document share one: with `ONTOLOGY_CONTEXT_MODE=selected_vector_search_ontology`
every unit retrieves its *own* context, so every unit gets a different chapter.

`ONTOLOGY_CONTEXT_SCOPE=document` resolves each unit's context as before and
then shows every unit the union. It is **recall-safe by construction** — the
union contains every atom each unit's own retrieval selected, so no unit is
shown less than it would have been. What it costs is precision, because a unit
also sees its siblings' terms, and per-call tokens, because the union is larger
than any one unit's slice. What it buys is that the chapter repeats.

That alone is not enough. A prefix cache is populated by a request that has
already *completed*, and the unit fan-out issues every call at once — so N calls
sharing a prefix all miss it, having each arrived before any of them wrote the
entry. `FANOUT_WARMUP_UNITS=1` runs the first unit to completion before fanning
out the rest, turning the other N−1 misses into hits at the cost of one
serialized call's wall-clock. The two settings are worth nothing apart and
should be set together.

`LLM_PROMPT_CACHE_KEY` completes the picture on OpenAI. The provider caches by
prefix regardless, but without a routing hint a wide simultaneous fan-out can be
spread across machines that each build their own entry. Any stable string works;
it must **not** vary per request, or it defeats itself.

Read the result from `budget.prefix_cache_hit_rate`. Note that a run with the
critic on already shows a non-trivial rate for an unrelated reason — the critic
re-reads the chapter its own render just built — so compare arms of the same
shape, and expect the change here to show up on the *render* fan-out.

### Capping what whole-module inclusion costs

`ONTOLOGY_PATCH_SMALL_MODULE_CLOSURE_MAX_TRIPLES` pulls a module's *whole* graph
into the snapshot once any of its atoms is admitted, because a
qualified-quantity or observation vocabulary is only useful whole: showing a
model `hasLowerBound` but not `hasUpperBound` is what makes it invent near-miss
property names. Whole-module inclusion can be most of the chapter, and the
chapter is paid on every call of every unit.

`ONTOLOGY_PATCH_SMALL_MODULE_CLOSURE_MAX_TOTAL_TRIPLES` caps what all closures
together may contribute. It is unset by default (unlimited, the historical
behaviour). When set, candidate modules are admitted **in order of retrieval
relevance** — the best score any of their atoms achieved — until the budget is
spent; a module too large for what remains is skipped rather than ending the
pass, so a smaller one behind it still gets in.

The ordering is relevance and deliberately **not** how many atoms a module won.
A module can win at most as many seeds as it has terms, so counting them ranks
modules by size, and would exclude precisely the case this closure exists for: a
small, sharply relevant vocabulary that is the document's actual subject can
never out-count a large peripheral one. Nothing is excluded for being small or
for being unpopular — a budget filled best-first also keeps the ordinary
situation working, which is a *combination* of modules rather than a single
winner.

Read it back from `module_closure_iris`, `module_closure_declined_iris` and
`module_closure_triples` in the run manifest: a question about a missing term is
answered by knowing which module the budget kept out.

!!! note "`ONTOLOGY_CONTEXT_MAX_TRIPLES` is not the lever it looks like here"
    Lowering the triple budget to force condensing is counterproductive on a
    term-sheet chapter. The condenser's noise and structural passes remove
    `owl:imports`, `dcterms:*` and stub restriction nodes — none of which a term
    sheet renders in the first place, so they save nothing. Its next pass drops
    `GLOSS_PREDICATES`, which includes `skos:altLabel` and `skos:scopeNote`:
    the surface forms and usage contracts the sheet deliberately keeps. Bound
    those with the text caps below, which shorten rather than remove them.

### Bounding the chapter's text

Nothing else in the pipeline caps a single literal, so without a cap the
chapter costs what a catalog's authors chose to write rather than what it
declares — one rambling `rdfs:comment` is paid for on every call of every unit,
and a catalog with kilobyte-long definitions has no ceiling at all. Four
settings put one in place:

| setting | governs |
|---|---|
| `ONTOLOGY_TEXT_MAX_CHARS_NAMING` | `rdfs:label`, `skos:prefLabel`, `skos:altLabel` |
| `ONTOLOGY_TEXT_MAX_CHARS_CONTRACT` | `skos:scopeNote`, `skos:definition` |
| `ONTOLOGY_TEXT_MAX_CHARS_PROSE` | `rdfs:comment` and the remaining SKOS notes |
| `ONTOLOGY_TEXT_TOTAL_BUDGET` | the summed length of all of them |

They apply to every chapter the facts loop builds, term sheet and serialized
graph alike, and are unset by default: on a tersely authored catalog they are a
no-op, which is the intended shape. These are bounds, not reductions. Clipping
happens on a word boundary and leaves a visible marker, so the model can tell a
clipped definition from a complete one — worth preferring to dropping the
statement, because a scope note's first sentence usually carries the contract
and the rest elaborates.

Over the total budget, prose is tightened first, then contracts, then names —
each fitted to the *largest* cap that still meets the budget rather than stepped
down through preset tiers, because every character under the budget is context
it never asked to lose. Nothing is ever removed: a clipped definition still says
the term has one and still carries the words that say when it applies, while a
removed one says nothing at all. Shedding whole statements is
`ONTOLOGY_CONTEXT_MAX_TRIPLES`'s job, on the axis where statements are what is
over budget.

Names have the highest floor of the three, and on a catalog whose labels are
already shorter than it that role is reached and does nothing. A chapter that
still does not fit is then passed through with a warning — correctly, because
what remains at that point is the vocabulary itself, and the fix is to send
fewer terms, not shorter names.

Read the effect back from the run manifest's `budget.counters`:
`chapter/text_chars_before` and `chapter/text_chars_after`,
`chapter/literals_clipped`, `chapter/literals_dropped`, and
`chapter/text_over_budget` when a chapter could not be made to fit. They are
recorded once per chapter built rather than per call that reads it, and the
before/after is reported even when nothing was clipped — which is the case that
tells you your caps are inert on this catalog. Sizing a cap is not something a
default can do for you: whether a catalog reaches one is a property of that
catalog.

See [Configuration](configuration.md) for the full list and
[LLM Caching](llm_caching.md) for cache behavior.
