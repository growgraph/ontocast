# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **LLM JSON parse failures now repair deterministically or fail with
  actionable feedback, instead of the informationless `input_value=None`
  retry loop.** Root-caused from the cached wire traffic of the 2026-08-28
  `--max-visits 2` matsci run, where 4 of 7 units wasted a ~50k-token call
  each on attempt 1: gpt-5-mini escapes the quotes that *delimit* JSON
  strings (`"text_fragment": \"…\",`) and the whitespace between tokens
  (`\",\n  "action"`), and occasionally under-closes one `{`. Langchain's
  `parse_json_markdown` degraded all of these to a silent `None` — or a
  silently *truncated* prefix of the object — so the retry prompt carried a
  pydantic `input_value=None` error naming nothing, and retries repeated the
  identical malformation. Three changes in `agent/common.py`:
  - `unescape_json_delimiters` joins the sanitizer chain and repairs the
    escaped-delimiter and escaped-whitespace classes without a retry
    (string-aware scan; legitimate in-string `\"` escapes untouched).
  - `parse_json_object` replaces the silently-lenient parse for every
    `PydanticOutputParser` call: strict parse first (keeping only the
    `strict=False` control-character leniency), fenced-block extraction as
    the sole fallback, and any remaining failure raises with line/column and
    a ±150-char context window — the same feedback shape that made the
    Turtle-level retries recover on the first try. Partial recoveries and
    non-object JSON (`null`) are rejected instead of validating silently.
  - An LLM request *timeout* is re-issued identically exactly once per call
    before propagating — a timeout is not a rate-limit "send less" signal,
    and at `--max-visits 2` a lost render silently cost a unit its entire
    critique. Rate limits and connection errors still propagate immediately.

- **A false mandatory `UNKNOWN_TERM` finding ordered repair renders to destroy
  correct numeric values.** Root-caused from the cached LLM wire traffic of the
  2026-08-11 matsci runs: the catalog *references* `qudt:QuantityValue` and
  `qudt:unit` (in `rdfs:subClassOf`/`owl:onProperty`), which made
  `http://qudt.org/schema/qudt/` a catalog namespace, while `qudt:numericValue`
  appears only in qqval's prose — so every unit carrying the canonical scalar
  property got "`qudt:numericValue` does not exist in its ontology … Candidates:
  `qudt:QuantityValue`" as a MANDATORY item, with a **class** suggested for a
  predicate slot. Of 58 cached repair responses carrying the finding, 25
  deleted the valid values outright, 28 re-encoded scalars as equal-bound fake
  ranges, and 1 wrote the class as a predicate; the two benchmark arms measured
  the fallout as 38–64% of value nodes with no number in any numeric slot and
  0–5 SPARQL-answerable measurement records per corpus. Three rules changed:
  - A namespace is **closed** (members eligible for `UNKNOWN_TERM` /
    near-miss alias repair) only when the catalog *declares* terms in it —
    subject-position statements — not when it merely references them
    (`collect_declared_namespaces`).
  - The configured quantity fallback vocabulary and `FACTS_CODE_PREDICATES`
    are exempt: the validator must never order the renderer to remove the
    vocabulary the facts prompt itself recommends. All deployment-blessed
    exemptions travel as one `ValidationPolicy` object on the toolbox instead
    of loose parameters.
  - Alias candidates are **role-filtered** against the catalog's declarations:
    a predicate never gets a known class suggested, and vice versa.
- The findings prompt now states the repair contract explicitly — every
  MANDATORY item must be fixed by rewriting in place, never by deleting the
  statement — and a repair render that shrinks the unit graph without
  resolving any mandatory finding is logged as data destruction rather than
  passing as a successful repair.
- `RunManifest.facts_triples` was not comparable to the manifest's own
  `.facts.ttl` (raw aggregated count vs provenance-stripped dump; 1711 vs 557
  on the matsci runs). `facts_triples_serialized` now records what the file
  actually holds.

### Added

- **`BudgetTracker` reports the two ratios that locate a cost problem**:
  `prefix_cache_hit_rate` (input tokens served from the *provider's* prompt
  cache) and `reasoning_share_of_output` (thinking tokens as a share of output).
  Both are `computed_field`s, so they ride the `/process` response and the run
  manifest without a new wire shape, and both are `null` — unmeasured, not zero
  — when the provider reports no tokens. The denominators span billed *and*
  replayed tokens, because `cache_read_input_tokens` and `reasoning_tokens`
  accumulate on both while `input_tokens` counts billed only; dividing by
  `input_tokens` alone reports over 100% cache hits on a partly replayed run,
  which is the reason these are computed here rather than by each consumer.
  They make existing manifests retrospectively comparable: across the manifests
  in `benchmarking/`, reasoning is **62–73% of all output tokens**, and the
  provider prefix cache serves **35–44%** of input on the wide document
  fan-outs against **91%** on a longer, more sequential run — same code, so the
  gap is call sequencing, not configuration. No currency is reported; per
  `docs/user_guide/observability.md` that stays with the deployment.
- **SHACL-vs-catalog contradiction lint** (`shacl_catalog_contradictions`),
  run when the validation gate loads shapes: any property the shapes require
  (`sh:minCount >= 1`) that the term validator would flag as unknown is logged
  as a configuration error — data cannot satisfy both sides. This exact
  contradiction (shapes requiring `qudt:numericValue` while the validator
  mandated its removal) is what silently destroyed the matsci numerics.
- **`LABEL_ONLY_NUMBER` mandatory finding**: a node carrying the fallback
  vocabulary's unit property but no numeric literal on any property, with a
  number in its label, is a measurement invisible to every query — previously
  numbers inside labels counted as "covered" and nothing ever flagged it.
- **Degenerate-bound promotion** at parse time: equal lower/upper bounds
  collapse to a single scalar on the configured numeric-value property.
  Config-driven and off by default — activates when the quantity fallback
  vocabulary names `numeric_value`, `lower_bound` and `upper_bound` roles
  (58% of all bound pairs in the matsci runs were degenerate).
- **The run manifest records the effective configuration and output shape**:
  `loops` (`max_visits`, `max_critic_visits`, `llm_repair_visits`), `selection`
  (`target_sections`, `exclude_sections`, `summarize_sections`,
  `summary_max_sentences`, `bibliography_mode`), and `graph_metrics`
  (connectivity of the serialized facts graph, via the new
  `util/graph_metrics.py`). The 2026-08 `--max-visits 1` vs `2` ablation turned
  out to be an A/A comparison — call accounting shows the critic never ran in
  the second arm — and nothing in either dump could confirm or refute what the
  run received; now the arm is auditable from its own manifest.
  `test_max_visits_critic_propagation.py` additionally pins both ends of the
  chain: the batch entry path writes the flag onto `AgentState`, and a unit
  loop at `max_visits=2` observably spends a critic call.

### Changed

- **`tool/facts_invariants.py` (2,760 lines) is split into the
  `tool/facts_validation/` package** — `terms` (catalog inventory, namespace
  closure, `ValidationPolicy`, alias candidates), `literal_repair` (parse-time
  LLM-free rewrites), `unit_findings` (per-unit findings), `shacl` (execution,
  autofix, catalog lint), `gate` (document-level validation). The package
  `__init__` is the public surface; the old module name is gone.
- `_normalize_and_repair_graph` and `_collect_facts_findings` take the atomic
  toolbox instead of a growing list of unpacked scalars.

### Changed

- **`LLM_GRAPH_FORMAT` now defaults to `jsonld`.** Turtle remains supported as
  the legacy encoding, for providers whose structured output handles strings
  more reliably than nested objects. This changes behaviour for anyone who
  never set the variable. The default was declared in four independent places —
  `ServerConfig`, `AgentState`, `UnitState`, and the `llm_graph_format_ctx`
  ContextVar that `coerce_llm_graph_wire` falls back to when `model_validate`
  is called without a validation context — and all four moved together;
  `test_llm_graph_format_default.py` now pins them against drift.
- **`ONTOLOGY_MAX_TRIPLES` now defaults to unlimited.** At `50000` it could not
  bind: ~634k tokens as Turtle, ~1.28M as JSON-LD, against a largest real
  ontology of 1409 triples — a graph that size became unusable as context ~40×
  earlier. It is a runaway-growth backstop on the per-unit ontology working
  graph, not a context cap, and its description said otherwise; use
  `ONTOLOGY_CONTEXT_MAX_TRIPLES` for prompt size. Still available, off by
  default, and now tested — it previously had no test at all.
- `ontocast_extract` (LangChain/MCP tool) takes its `render_mode` default from
  `RENDER_MODE` instead of hardcoding `ontology_and_facts`, and parses it
  through `parse_render_mode_param` like every other entry point. Omitting the
  argument now honours the server's configuration.

### Fixed

- **`.env.example` named two local encoder models without the
  `sentence-transformers/` prefix.** `SharedEncoder` caches by the literal
  `(model name, device)` string, so `AGG_EMBEDDING_MODEL` and
  `EMBEDDING_MODEL_NAME` as shipped would not share a resident model with the
  prefixed defaults — and a user who copied the file and then followed the
  performance guide's advice to align all three got **two copies of the same
  checkpoint**, the exact outcome the alignment exists to prevent. Both spellings
  are valid and both work, so nothing surfaced it. Now prefixed, with a test
  asserting all three encoder settings keep the prefix.
- **`ONTOLOGY_MAX_TRIPLES` could lock the ontology loop out silently.** The
  guard compared absolute post-apply size, so a working graph seeded above the
  cap failed on every subsequent update — discarding the LLM's work for the rest
  of the run with only a WARNING, and with no way back under, since deletions
  were rejected too. It now rejects only updates that *grow* the graph past the
  cap, and logs the already-over case distinctly.
- **`pytest` no longer loads the developer's live `.env`.** Removing `env_files`
  from `[tool.pytest.ini_options]` had not been enough: `pytest-dotenv` loads a
  discovered dotenv file even with no `env_files` set, so every `BaseSettings`
  built in a test silently took local configuration (a local `RENDER_MODE=facts`
  left the ontology block untested) and a real `LLM_API_KEY` sat in the
  environment. The plugin is uninstalled and blocked via `-pno:dotenv` —
  deliberately **one token**, because `toml-sort` runs with `--all` and sorts
  array values, which split a two-token `-p no:dotenv` apart and left pytest
  unable to start. `test/conftest.py` now also asserts the pipeline mode
  selectors read their declared defaults, so a future mangling fails loudly with
  a pointer rather than silently readmitting the environment.
- The LangChain `apply_graph_update` tool pins its own Turtle coercion. Its
  interface is Turtle-in by parameter name (`insert_ttl` / `delete_ttl`) and was
  incorrectly tracking the LLM wire format.

### Added

- **`ONTOLOGY_CONTEXT_MAX_TRIPLES` (default `4000`) bounds the ontology context
  in every mode.** Only `selected_vector_search_ontology` had ever bounded it;
  `selected_single_ontology` (the default) and `fixed_single_ontology`
  serialized the whole selected ontology into every prompt, and the facts
  fan-out serialized the union of every artifact, with no cap. Over budget,
  `onto/ontology_condense.py` drops in increasing order of harm — header/list
  noise, then redundant structure, then glosses — and **never** drops labels,
  types, hierarchy or domain/range. It is best-effort by design: a graph that
  cannot fit is passed through with a warning naming the way out, because
  cutting load-bearing schema to hit a number produces an extraction failure
  that reads as a bad model. Enforced at `format_ontology_chapter`, the one
  point every chapter passes through, and part of the snapshot memo key so a
  shared snapshot cannot serve one unit's budget to the next.
- `ONTOLOGY_SNAPSHOT_TRIPLES` retrieval metric, written for every context mode.
  Previously only the vector resolver recorded a size, nested under
  `patch_retrieval` — so the two modes that bounded nothing also reported
  nothing.
- The seed-free graph pruners and predicate vocabularies move to
  `onto/graph_prune.py`, shared by induced-subgraph retrieval and the condenser
  rather than duplicated.
- **`.env.example.minimal` and a Configuration Playbooks guide.** The full
  configuration surface is ~200 variables, which is not a thing anyone can
  optimise at once. The minimal file carries 29, grouped by the decision they
  belong to rather than by config class, and the guide gives a playbook per task
  — evaluate, build an ontology, populate facts, scale the catalog, serve it —
  each listing only what it changes, plus a symptom-to-knob triage table. Three
  tests keep the curated file honest: every name resolves to a real setting, it
  stays a subset of `.env.example`, and it stays under a variable ceiling so it
  cannot accrete back toward the full surface. A fourth checks that exact
  variable counts quoted in prose still match reality (hedged phrasing like
  "around 200" is exempt).
- The minimal file covers conversion and chunking (`CHUNK_MIN_SIZE` /
  `CHUNK_MAX_SIZE`, `CHUNK_SEGMENTER`, `CHUNK_SECTION_CLASSIFIER`,
  `CHUNK_BIBLIOGRAPHY_MODE`, `CONVERTER_PROFILE`), the local-encoder alignment,
  SHACL shapes, LLM caching and the web-search toggle — all decisions a
  first-time user faces that the first cut omitted.

### Documentation

- **`RENDER_MODE` has prose for the first time.** A `## Render Mode` section in
  the configuration guide covering what each value actually skips — `ontology`
  writes no facts to the triple store at all, `facts` bypasses the ontology
  block and depends wholly on the existing catalog — plus the per-request
  precedence chain and the 400-on-typo contract.
- Ontology context behaviour that was only visible in code: a non-empty
  `ontology_context_fixed_ontology_id` silently forces fixed mode over an
  explicit `ontology_context_mode`; a fixed id matching no catalog entry
  degrades to an empty snapshot with a warning rather than an error;
  `selected_single_ontology` costs one extra LLM call per content unit; the
  consistency critic runs *only* under `selected_vector_search_ontology`; and
  facts units reuse a merged document-level context instead of re-resolving.
- **Docs search now finds environment variables.** The Material search separator
  did not split on `_`, so `ONTOLOGY_CONTEXT_MODE` indexed as one opaque token
  and "ontology context mode" matched nothing; titles are boosted 1000× but no
  heading carried the literal variable name. Separator updated, both mode
  selectors now name their variable in the heading, and the configuration and
  ontology-context pages carry a search boost.
- `README.md` and `docs/index.md` gained a Configuration section — between them
  they previously named five environment variables and not one mode selector.
- The 20-variable `WEB_SEARCH_*` block is now three annotated tables instead of
  a bare code fence, and states that the whole lane is inert at its default.
- `MAX_VISITS_PER_NODE` is documented under its canonical name, not only as the
  `MAX_VISITS` alias; stale "Qdrant"-only wording for vector mode corrected to
  cover LanceDB.

## [0.6.0] - 2026-08-10

*First release published to PyPI since 0.4.3: the 0.5.0 and 0.5.1
sections below were in-tree version bumps that were never tagged or
published.*

### Breaking

- Removed in-memory vector store (`VECTOR_STORE_BACKEND=memory`,
  `VectorStoreBackend.MEMORY`, `tool/vector_store/in_memory.py`). Retrieval
  requires Qdrant or LanceDB; default path unchanged (`AUTO` → `NONE` when
  neither is set). `Config.in_memory()` is triple-store only (pyoxigraph).
- Dropped unread `AgentState` fields (UnitState shadows, never-read writers,
  unused `graph_uri_override`); `graph_uri` is always `doc_namespace`. Removed
  unused status/progress helpers; `set_failure` no longer takes `success_score`.
  UnitState external-evidence mirrors removed — use `ExternalEvidenceCacheEntry`.
- Removed dead modules: `onto/context.py`, `tool/graph_version_manager.py`,
  `tool/graph_diff.py` (~1,222 lines).
- `AtomicToolBox` takes `WebSearchConfig`; `EmbeddingBasedAggregator` takes
  `AggregationConfig` (flat kwargs removed).
- Removed `test-api` console script / `cli/test_api.py`; dropped `requests`
  from the `server` extra.
- Removed unwired symbols: `route_after_convert`,
  `route_after_ontology_consolidation`, `WorkflowNode.AGGREGATE_FACTS` /
  `PARALLEL_MAP_UNITS`, mock triple-store managers, `aggregate_anchor_metrics`,
  stale prompt templates; refreshed `agent/__init__.py` exports.
- **A provenance unit node is typed `schema:Text`, not `schema:text`.**
  `https://schema.org/text` is the *property* `text`; the class is
  `schema:Text`. Every provenance node OntoCast has ever emitted was typed with
  a property IRI. Consistently so, which is why nothing caught it: the two
  matchers that key on it (`TripleStoreManager._provenance_source_nodes`,
  `normalize_ontology`'s chunk-node detection) used the same wrong IRI, and no
  test exercised `strip_provenance` against real rewriter output. Graphs already
  in a triple store keep the old type until re-extracted; a query filtering on
  `schema:text` must be updated.
- **`TripleStoreManager._PROVENANCE_METADATA_PREDICATES` is gone**, replaced by
  `ontocast.onto.constants.PROVENANCE_METADATA_TERMS` — a module-level
  `frozenset` naming the classes *and* predicates the pipeline mints on
  provenance nodes. The class attribute had zero references anywhere.
- **Removed confirmed-dead code (import-visible).** Every symbol below had
  exactly one reference repo-wide — its own definition — so nothing in-tree
  changes, but anything importing them out-of-tree breaks. Modules
  `ontocast/stategraph/util.py` (`count_visits_conditional_success`,
  `wrap_with`) and `ontocast/tool/agg/promoter.py` (`URIPromoter`), neither ever
  imported. The `prompt/graph_format.py` "backward-compatible" block —
  `critique_graph_format_instruction` plus seven eagerly-evaluated
  `output_instruction_*` aliases. The ten `*_description()` helpers in
  `onto/llm_graph_payload.py`, superseded by `GraphFormatProfile`. The
  `OntologyDecision` and `FactsDecision` enums. And the singletons
  `invalid_max_visits_response`, `SemanticTriplesFactsReport`,
  `EntityMetadata`, `PredicateMetadata`, `compare_versions`,
  `validate_and_connect_chunk`, `plot_ontology_graph`,
  `merge_terminal_ontologies`, `known_prefixes_for_llm_parse`,
  `collect_catalog_namespaces`, `role_from_predicate_usage`,
  `derive_pair_matches_with_embeddings`, `update_mermaid_graph_in_markdown`,
  `CHUNK_NULL_IRI` and `render_ontology_rank_diagnostics`. The last of these
  was the only presenter of the `ONTOLOGY_PATCH_DUMP_ONTOLOGY_RANKS` payload;
  the payload itself is still stored under `'ontology_rank_diagnostics'`, which
  is what the setting actually promises.
- **`ontology_directory` is now strictly read-only.** It is a seed fixture read
  once at startup, but two methods treated it as a writable store:
  `ingest_ontology_ttl` required it and created it while never writing a file,
  and `delete_ontology_by_iri` globbed it and **unlinked** any TTL declaring the
  deleted IRI — so `DELETE /ontologies/{iri}` destroyed curated input that the
  next startup reloads from. Ingestion no longer requires or touches the
  directory, and deletion no longer removes files from it. An ingested ontology
  lives in the triple store and vector index only and does not survive a rebuild
  from seeds; that is now the stated contract rather than an accident.
  `ToolBox._unlink_ttl_files_if_ontology_iri` is removed, as is the LangChain
  `ontology_directory is not configured` gate on
  `ontocast_ingest_ontology_ttl` — the tool is now always offered.
- **Removed the `cmp-states` console script** and `ontocast/cli/cmp_states.py`.
  It read `agent_state.onto.update*.json` dumps written by
  `BasePydanticModel.save_json`, which had no callers anywhere: the dumps were
  the same debug generation as the removed `working_directory`. Both
  `save_json` and its unused `load` counterpart are gone.
- **`CHUNK_BREAKPOINT_THRESHOLD_TYPE` and `CHUNK_BREAKPOINT_THRESHOLD_AMOUNT`
  removed** — the chunker clusters with PCA → UMAP → HDBSCAN and never read
  them. Invalidates the chunk cache; does **not** change chunk output.
- **`langchain-huggingface` is no longer a dependency.** The chunker uses a
  `langchain_core.embeddings.Embeddings` adapter over the shared encoder,
  reproducing `HuggingFaceEmbeddings._embed` including its newline collapse,
  which chunk boundaries depend on.
- **Removed:** `ChunkerTool.model` (use `ChunkConfig.embedding_model`),
  `WorkflowNode.SUMMARIZE_CHUNKS`, `route_after_chunk`,
  `make_summarize_chunks_node`, `resolve_effective_facts_ontology_context`.
  `EmbeddingTool._embed_unlocked` is renamed `_embed_raw`;
  `EntityClusterer.embedder` returns a `SharedEncoder`. Regenerate diagrams with
  `uv run plot-graph`.
- **Transport failures are no longer retried by `call_llm_with_retry`.** Its
  retry exists to show the model its own malformed output; retrying tripled the
  request rate when a provider was rate-limiting. Parse retries back off with
  jitter.
- **`ToolBox.serialize` raises inside a running event loop** — await
  `aserialize` there.
- **On Apple Silicon the chunker moves from CPU to MPS,** now that it
  auto-selects like every other consumer. Identical on CPU-only and CUDA hosts.
- **LLM cache key gained fields, invalidating every existing entry.** The key
  now carries a `cache_format_version` (now `2`) plus the Ollama generation
  knobs `think` / `num_predict` / `num_ctx`. Caches written by earlier releases
  will not be hit, so the first run after upgrading re-pays for every call.
- **The on-disk cache now evicts on its own,** capped at 1 GB by default
  (`ONTOCAST_CACHE_MAX_BYTES`). This is new deletion behaviour; set the variable
  to `0` to restore unbounded growth.
- **`CHUNK_SECTION_CLASSIFIER` now defaults to `heuristic`, not `llm`,** so
  chunking makes no LLM calls. The deterministic tiers now resolve the headings
  that previously required a model; set it back to `llm` to keep a model pass
  over headings none of them can name.
- **`SectionSpan.label` is now `str | None`.** A region whose section type is
  unknown is represented explicitly rather than being absorbed into its
  neighbour. Callers that assumed a non-null label must handle `None`.
- **`working_directory` / `ONTOCAST_WORKING_DIRECTORY` removed.** Nothing had
  read it since the filesystem triple-store backend was deleted in `9d3ab77`
  (2026-06) — but `ontocast serve` and `ontocast process` still *raised* without
  it and then created an empty directory, and `plot-graph` leaked a
  `tempfile.mkdtemp()` per invocation to satisfy a field `ToolBox` ignores.
  Both commands now start with no OntoCast env var set at all. Caches keep using
  `ONTOCAST_CACHE_DIR`; batch artifacts keep using the explicit `--output-dir`
  family. A stale `ONTOCAST_WORKING_DIRECTORY` in the environment is ignored.
- **`BudgetTracker.add_usage` / `add_cache_hit` take `usage: TokenUsage | None`**
  instead of `input_tokens` / `output_tokens` ints. `_usage_from_llm_result`
  returns a `TokenUsage` rather than a tuple.

### Added

- `MAX_CRITIC_VISITS_PER_NODE` — optional cap for the inner critic loop;
  unset keeps coupling to `MAX_VISITS_PER_NODE`.
- `test_agent_graph_topology_is_pinned` — asserts full document-graph
  node/edge topology (including conditional maps).
- Shared `run_facts_gate` (`merge_repair` flag) and
  `prepare_extraction_request` for `/process` and `/process_unit`.
- **Section labels reach the RDF output.** A source unit's `section_label`,
  `section_label_source` and `section_label_confidence` reached the summarizer
  and `ontocast sections` and stopped there, so a finished run could not be
  audited for *which part of the document* a fact came from. The provenance
  artifact now carries `schema:articleSection`, `ontocast:sectionLabelSource`
  and `ontocast:sectionLabelConfidence` (`xsd:decimal`) on each labeled unit
  node — emitted as a block, so an unlabeled unit gets none of the three rather
  than a bare confidence of `0.0`. `strip_provenance` removes them with the rest
  of the chunk metadata, which is now pinned by a test.
- **A minted `ontocast:` namespace** (`https://growgraph.dev/ontocast#`), the
  project's first. Deliberately outside `DEFAULT_IRI` so pipeline metadata is
  never a SHACL repair target. Used only where no standard vocabulary applies:
  "which classifier tier decided this label" and a bare confidence have no
  well-known predicate that is not a heavyweight `prov:qualifiedAttribution` or
  `schema:Rating` structure, and a `Rating` blank node would trip the gate's own
  placeholder-prune heuristic.
- **`CHUNK_SECTION_FILTER_ON_EMPTY`** (`warn` default, `error`) decides what
  happens when a section selection removes every segment. Under `warn` the run
  extracts zero chunks and reports success, which is indistinguishable from a
  document that genuinely had nothing to extract; `error` fails instead — HTTP
  `422` with `error_code=empty_section_selection:<param>`, or a non-zero exit
  for `ontocast process`, where the file joins `failed_files` so one unmatched
  selection does not kill the batch. Covers the `target_sections` /
  `summarize_sections` allowlist **and** the `exclude_sections` denylist, which
  had no empty guard at all and could blank a document from a schema's
  `default_exclude` with no caller involvement. `ontocast sections` always
  behaves as `warn`: a diagnostic has to survive the condition it diagnoses.
  New public `SectionSelectionEmptyError`.
- **`retrieval_metrics` in the run manifest.** `<stem>.run.json` now carries the
  same payload `/process` returns, so a batch run is no longer the blind path.
  This is also what finally gives `ONTOLOGY_PATCH_DUMP_ONTOLOGY_RANKS` a reader
  outside HTTP: its payload rides along under `patch_retrieval`.
- **`RetrievalMetric`**, a `StrEnum` in `onto/enum.py` enumerating the 24
  top-level `retrieval_metrics` keys. They are wire names — serialized verbatim
  into `ProcessResultMetadata` — and were bare string literals across three
  modules with no registry, so a typo was a silently missing metric and nothing
  said what a run should emit. The nested patch-retriever keys are deliberately
  left out: they belong to a different namespace with its own lifecycle.
- **`facts_llm_repair_renders_failed`** alongside
  `facts_llm_repair_renders_total`. A repair render that fails leaves the
  pre-repair graph intact and the unit reports `SUCCESS` by design, so the
  failure was recorded on the attempt log and aggregated nowhere.
- **Console-script reference** in the installation guide covering `ontocast
  serve` / `process` / `sections`, `pdfs-to-markdown`, `test-api`,
  `match-graphs` and `plot-graph`; three of them appeared in no doc page.
- Token accounting survives a cache replay. `CachedResponse` now stores the
  provider's usage alongside the response, and `BudgetTracker` reports it as
  `cached_input_tokens` / `cached_output_tokens`, kept apart from the billed
  `input_tokens` / `output_tokens` so a replay is not mistaken for spend.
  Previously `usage_metadata` — a separate `AIMessage` attribute, not part of
  the `response_metadata` that was persisted — was dropped on write, so the
  cache-replay protocol in `docs/user_guide/performance.md`, the mode the
  benchmark and ablation work runs in, reported zero tokens.
- `BudgetTracker.reasoning_tokens`, `cache_read_input_tokens` and
  `cache_creation_input_tokens`, read from LangChain's `UsageMetadata` detail
  keys (and the OpenAI `*_tokens_details` equivalents). Reasoning tokens
  dominate output cost for the thinking models `LLM_THINK` drives, and
  provider-cached input bills at a fraction of the fresh rate — folding either
  into a single total misstates cost in opposite directions.
- A cache hit now rebuilds its `AIMessage` with `usage_metadata`, so a replayed
  call looks identical to a live one to anything reading usage off the message.
  The Batch-API prefill (`ontocast.tool.llm_batch`) carries usage through too.
- `OllamaModel` presets for current Qwen and Kimi tags: `qwen3.6`, `qwen3.5`,
  `qwen3`, `qwen3-coder`, `qwen3-coder-next`, `qwen2.5`, `qwen2.5-coder`,
  `kimi-k3`, `kimi-k2.7-code`, `kimi-k2.6`, `kimi-k2.5`.
- **Run manifest.** `ontocast process --output-dir DIR` now writes
  `<stem>.run.json` beside each `<stem>.facts.ttl`: OntoCast version, render
  mode, tenancy, the LLM settings that shaped the output, the full
  `BudgetTracker`, and triple counts. The tracker was returned over HTTP and
  logged at INFO, then discarded, so a finished batch left its TTL with no
  record of what produced it or what it cost, and two runs could only be
  compared by rerunning them.
- `docs/user_guide/observability.md` — the three layers (in-run `BudgetTracker`,
  the run manifest, and external tracing via LangSmith/Langfuse/OTel, which work
  today with no OntoCast code) and the caveat that a cache hit emits no provider
  span, so a replayed run shows a thin trace while the budget shows the real
  workload.

Not bumped: `LLM_CACHE_FORMAT_VERSION` stays at 2. The `usage` field is additive
and optional, so entries written before it still load and report usage as
unknown rather than zero — bumping would have evicted every existing entry and
forced a paid re-run before any replay worked again.


- **`/process_unit` runs the post-aggregation validation gate.** The route
  shipped unvalidated facts while its docstring claimed otherwise; it now runs
  the same gate as the document pipeline and the CLI unit path — invariant
  findings, SHACL, and the LLM-free autofix, minus only the un-merge repair
  (meaningless for a single unit) — and returns `facts_conformance`,
  `facts_validation_findings` and `facts_gate_repairs`, serving the repaired
  graph.
- **LLM-free repair of SHACL violations** (`FACTS_SHACL_AUTOFIX`, default
  `prune`) in a bounded validate → repair → revalidate loop
  (`FACTS_SHACL_AUTOFIX_PASSES`); a pass that does not strictly reduce
  violations is reverted. `rewrite` retypes a literal to the `sh:datatype` it
  parses as, and replaces a string literal on an IRI-only path with the single
  catalog IRI declaring it as a surface form. `prune` additionally drops
  `sh:minCount` violators that assert nothing beyond `rdf:type`/`rdfs:label`
  and are referenced by at most one subject — a placeholder value node stands
  for an extraction that did not happen. `sh:maxCount`, `sh:not`,
  `sh:qualifiedValueShape` and SPARQL constraints are reported, never
  repaired. Nothing is invented: a node carrying real data but missing a
  required property stays a reported finding.
- **Code resolution at parse time** (`FACTS_CODE_PREDICATES`): a node carrying
  `qudt:ucumCode "d"` but no unit link gains `qudt:unit unit:DAY` when exactly
  one catalog individual declares that code. Schema-driven — the connecting
  property is whichever object property the context declares with a matching
  range and domain, falling back to the graph's own observed usage when a
  vendored projection declares no range (the shipped QUDT unit subset does
  not). Resolved all five ucum-coded nodes in the matsci pilot.
- **The validation result is reported, not just logged.** `ProcessResultMetadata`
  gains `facts_conformance` (whether SHACL ran, whether the graph conforms, and
  counts by finding kind, SHACL constraint component and shape),
  `facts_validation_findings` and `facts_gate_repairs`. Batch runs write the
  same payload beside the facts Turtle as `<name>.facts.validation.json`.
  Grouping by constraint is what makes a residue diagnosable: 36 violations on
  one shape are one modelling gap, not 36 defects.
- **[Facts Validation and SHACL](docs/user_guide/validation.md)** documents the
  three validation layers, which cost a provider call, where shapes come from,
  what the autofix will and will not do, and how to read the conformance
  report.


- **Automatic cache eviction.** `Cacher.prune()` drops TTL-expired entries, then
  evicts least-recently-*used* entries until the total fits under
  `ONTOCAST_CACHE_MAX_BYTES`. Recency comes from access time, so an entry written
  once and read constantly outlives one written recently and never touched.
  Runs at process start and after every `ONTOCAST_CACHE_PRUNE_EVERY` (256)
  writes, which keeps a long-lived `ontocast serve` bounded between restarts.
- **`ontocast cache` command group**: `stats` (per-tool size, flags orphaned
  subdirectories), `prune` (`--max-bytes`, `--ttl-days`, `--orphaned`), and
  `clear [--subdir]`.
- **`PathConfig` cache settings**: `ONTOCAST_CACHE_MAX_BYTES` (accepts `1GB` /
  `500MB` as well as a byte count), `ONTOCAST_CACHE_TTL_DAYS`,
  `ONTOCAST_CACHE_PRUNE_EVERY`.
- Converter cache entries carry a format version in the key, replacing the old
  practice of bumping the subdirectory name; the subdirectory is back to plain
  `converter/`. Stray `converter_v2/` and `converter_v3/` directories in an
  existing cache are cleared by `ontocast cache prune --orphaned`.
- Typed `CacheStats` / `PruneReport` models and `Cacher.cache_stats()`;
  `get_cache_stats()` still returns a plain dict for JSON responses.
- **Timing telemetry separating provider latency from event-loop stalls**, in
  `budget.node_durations` / `budget.counters`, documented in the new
  [Performance](docs/user_guide/performance.md) guide. `BudgetTracker` gains
  `counters`/`incr` and `parallel_efficiency`, plus a `<node>` (wall) vs
  `<node>/unit_sum` (summed across workers) key convention; `_max` keys take the
  maximum on merge. Adds `llm/provider`, `llm/inflight_wait`,
  `llm/cache_lookup`, per-stage `worker_wait`, and a `loop_lag` sampler
  (`ontocast/util/loop_lag.py`) — awaited I/O yields, so lag isolates
  synchronous blocking that wall clock cannot distinguish from slow providers.
- **`LLM_REQUEST_TIMEOUT_SECONDS`** (default 180). A hung call previously held a
  unit-worker slot and an `LLM_MAX_INFLIGHT` slot indefinitely. Raises
  `LLMRequestTimeoutError`, deliberately not an `asyncio.TimeoutError`, so it
  fails one unit instead of aborting the fan-out.
- **`CHUNK_EMBEDDING_MODEL`** (`ChunkConfig.embedding_model`), default unchanged.
  The chunker's checkpoint was a `ChunkerTool` field nothing ever set.
- **Section classification cascade** (`ontocast/tool/chunk/outline.py`,
  `density.py`). Classification runs over the document outline rather than per
  chunk, through tiers of increasing cost: outline → heading patterns → heading
  keywords → canonical-order fill → content density → batched LLM. Only the last
  costs anything, and it is off by default.
- **Heading genericity discrimination.** Docling reports a flat heading level
  for PDF conversions, so hierarchy cannot be read from the structure. Headings
  are instead classified by content-word count: generic section names open a new
  section, while descriptive subsection titles (and document titles) inherit
  their parent's label. Without this, a subsection such as `Cooperative ensemble
  breaks the population-inversion limitation` would split several thousand
  characters of results text out of the results section.
- **`keywords` and `order` in the section-label YAML schemas**, plus a schema
  level `ordered` flag. Keywords are the recall tier for compound and decorated
  headings; `order` is used only to refuse a fill that would run backwards.
  Both are optional, so existing schemas load unchanged.
- **Content-density classification** for regions with no usable heading
  (`CHUNK_SECTION_DENSITY`). `conservative` (default) recognises only reference
  lists and acknowledgements; `aggressive` additionally guesses
  methods/results/introduction and is opt-in, because those features do not
  separate those sections reliably and a wrong label is acted on silently.
- **Batched LLM section classification** (`CHUNK_SECTION_LLM_BATCH_SIZE`,
  default 40). When `CHUNK_SECTION_CLASSIFIER=llm`, one call now classifies a
  whole document's residual instead of one call per chunk; a response that
  cannot be used falls back to the per-chunk path.
- **`ontocast sections`** — prints the detected outline and every chunk's label,
  deciding tier and confidence, without running extraction. Makes no LLM calls
  and needs no provider credentials unless `--section-classifier llm` is passed.
- **`section_label_source` and `section_label_confidence`** on `ContentUnit`,
  `PreparedChunk` and `PrepareSegment`, recording which tier decided a label.
  The source is load-bearing, not diagnostic: it is what stops forward-fill from
  overwriting an explicitly unresolved section.
- **Plain-text heading detection** (`CHUNK_SECTION_TEXT_HEADINGS`) for documents
  whose conversion produced no markdown heading structure.
- **Automatic document-type schema detection**
  (`ontocast/tool/chunk/schema_detect.py`, `CHUNK_SECTION_SCHEMA_DETECT`,
  default `headings`). Section labels are only meaningful relative to a schema,
  and a 10-Q submitted without `section_schema_id` or a matching
  `document_type_hint` was scored against the academic default and came back
  entirely unlabeled. Three tiers, cheapest first: headings that only one schema
  recognises (free, no model), then embedding-based heading voting reusing the
  chunker's model, then body prose against document-type profiles. Precedence is
  explicit id → hint → detection → manifest default, so caller intent is never
  overridden. Every tier abstains rather than guessing — a wrong schema
  relabels an entire document silently.
  - The lexical tier scores on **exclusive** evidence only: a heading several
    schemas recognise counts zero, not a fraction. Weighting shared headings
    fractionally measured strictly worse (clinical 1.4× → 2.0×, standard
    4.2× → 14×, academic 6.1× → 600×) — `References` genuinely carries no
    information about which cell a document is in.
  - The content tier ships **off** (`auto`, not the default). It ranks 7/9 on
    the corpus but its one confident error is severe: chemistry prose scores
    `standard` over `academic` past the acceptance margin. It is gated to
    documents with essentially no headings, excludes `news` (a measured
    semantic attractor), and demands a 4.0 margin against the heading tiers'
    1.8.
- **Three new section-label schemas** — `patent`, `standard` and `news` —
  completing the document-type partition. No `thesis` cell: a thesis shares the
  IMRaD body of a paper and differs only in front and back matter, so it is a
  subtype of `academic` rather than a sibling and belongs to the planned
  `academic → paper → experimental` funnel. `thesis`/`dissertation` hints
  resolve to `academic`.
- **`document_profile` on `SectionLabelSchema`** — one sentence per cell stating
  what makes it exclusive. It is the artifact that enforces the partition (two
  profiles that could describe the same document mean the partition is broken)
  and doubles as the content tier's prototype. `general` deliberately has none,
  which is what keeps the residual cell out of detection entirely.
- **Verified keyword tiers for every schema.** All eight non-academic schemas
  gained corpus-grounded `keywords`, with `order`/`ordered` where a canonical
  order exists. Every keyword was authored against a real document in
  `test/data/schema_corpus.json` and cut if it matched nothing — the baseline
  before this was 4/9 cells detected, now 9/9.
- **Document-type detection corpus** (`test/data/schema_corpus.json`,
  `run/fetch_schema_samples.py`). One real document per cell — RFC 7231, *Pride
  and Prejudice*, a USPTO patent, the CC BY 4.0 legal code, the nginx guide, a
  Europe PMC trial protocol, a Wikinews article, plus the in-repo 10-Q and
  chemistry paper. Only heading sequences and sampled paragraphs are committed,
  each with its source URL and licence, so the suite stays offline and a few
  tens of kB. Tuning a nine-way classifier on the two document types previously
  in `data/` was not sound.
- **Schema reporting in `ontocast sections`** — the resolved schema, the tier
  that chose it, its margin over the runner-up, and the ranked candidate
  evidence. The only way to see a weak-but-accepted detection; free in
  `lexical` mode.

Measured on the Apple 10-Q, which is what the change is for: with detection off
the document resolves to the academic default and 1 of 102 chunks receives a
label — and that one is `methods`, i.e. wrong. With detection on it resolves to
`financial` on the free lexical tier at an 8.7× margin and 17 of 75 chunks are
labeled (`notes_to_financials`, `md_and_a`, `legal_proceedings`,
`financial_statements`, `business_overview`, `highlights`, `cover`), so
`--target-sections md_and_a` selects text for the first time.

- **Batch cache prewarming was a no-op.** `ontocast.tool.llm_batch` built its own
  cache-key config and dropped `base_url` when it was `None` — the default — so
  every imported entry hashed differently from what `LLMTool` looked up.
  Both sides now call one `llm_cache_config()`, covered by a regression test
  that asserts through the real read path rather than against the importer itself.
- **`complete()` and `extract()` mangled Anthropic and Google responses.** Those
  providers return a list of typed content blocks; the two methods stringified
  the list into a Python repr, returned it, and cached it. All four entry points
  (`__call__`, `acall`, `complete`, `extract`) now genuinely share
  `_invoke_cached`, so content normalisation cannot drift between them again.
- **Ollama generation settings were absent from the cache key,** so changing
  `num_ctx` returned the previous, truncated response from cache.
- **`remove_ontology_by_iri` evicted the ontology graph cache under the wrong
  key.** Entries are inserted under the header's `graph_uri` but were popped by
  `versioned_iri`; the two coincide only while content hashing is round-trip
  stable, so a removed ontology could stay resolvable. `_graph_cache` is now
  also LRU-bounded rather than an unbounded dict of rdflib graphs.
- **Cache writes are atomic** (temp file + `os.replace`). A truncating write with
  `PARALLEL_WORKERS` units in flight, or a Ctrl-C mid-write, left readers seeing
  half a JSON document.
- **Cache I/O no longer blocks the event loop.** Disk reads and writes on the
  async path run in a thread, and `GET /info` no longer `stat()`s every cache
  file inline — tens of thousands of syscalls per request on a warm cache.
- **Test cache isolation actually works.** Pytest was detected via the shell's
  `$_` variable, which holds the path to `uv` under `uv run pytest`, so the
  `.test_cache` branch never fired and tests read and wrote the developer's real
  cache.
- **Binary cache keys are no longer lossy.** PDF bytes were decoded with
  `errors="ignore"` before hashing, leaving the key resting on whichever bytes
  happened to form valid UTF-8.
- **The in-flight semaphore is per event loop.** `asyncio.Semaphore` binds to a
  loop on its first contended acquire, so a single process-wide instance raised
  "bound to a different event loop" on the second `asyncio.run` in a process.
- Cache-hit responses replay the provider's `response_metadata`, so a cached call
  is behaviourally identical to a fresh one.
- `ToolBoxRuntime.acreate` built a second `Cacher` instead of reusing the shared
  one, defeating the documented single-instance design.

### Changed

- `LLM_MODEL_NAME` accepts any string. The model enums are presets, not a
  whitelist: an unrecognised provider/model pairing now logs a warning and is
  passed through instead of raising. The closed whitelist rejected models newer
  than the installed package, and blocked the standard way to reach hosted
  Qwen/Kimi/DeepSeek — `LLM_PROVIDER=openai` plus a vendor `LLM_BASE_URL`
  (Moonshot, DashScope, OpenRouter, vLLM), which is what `base_url` exists for.
- A model name that matches a preset exactly is normalised to the enum member
  rather than warning about itself: `LLM_MODEL_NAME` always arrives as a string,
  and with `str` in the union pydantic had no reason to prefer the enum.
- Docs: `LLM_MODEL_NAME` guidance and OpenAI-compatible endpoint recipes in
  `docs/user_guide/configuration.md`; token-field table and replay-cost note in
  `docs/user_guide/performance.md`.
- **`FACTS_REPAIR_VISITS` → `FACTS_LLM_REPAIR_VISITS`**, and
  `_run_deterministic_repair` → `_run_finding_driven_repair`. The old name said
  "deterministic" for a loop whose every visit is a paid `render_facts_update`
  call — only its *trigger* is deterministic. This mattered: at the default
  `MAX_VISITS=1` a facts unit costs up to **two** provider calls, not one, and
  the documentation claimed otherwise. "Deterministic repair" now names only
  the LLM-free graph rewrites. Telemetry follows: attempt kind `repair` →
  `llm_repair`, metric `facts_repair_visits_total` →
  `facts_llm_repair_renders_total`.

### Fixed

- Process params unified via `_PARAM_SPECS` across query / JSON / multipart
  (`render_mode`, `ontology_context_mode`, `*_user_instruction`,
  `strip_provenance`); previously some were query-only or body-ignored.
- `render_mode` / `llm_graph_format` / `ontology_context_mode` parsers raise
  `RequestParamError` → HTTP 400 instead of silent default fallback.
- Finding-driven repair copies stage/reason onto `FactsLoopAttempt` before
  `clear_failure()`.
- `facts_loop` / `ontology_loop` report the failing stage, not always
  `*_CRITIQUE`.
- **Deterministic repairs no longer orphan provenance.** The 0.6.0 sweep covered
  `SHACL_PRUNE` only. `SHACL_RETYPE` and `SHACL_CODE_RESOLVED` also remove a
  triple — replacing it with a repaired one — so their
  `_:r rdf:reifies <<( s p o )>>` reifier was left describing the pre-repair
  statement, and no subject/object pattern matches a term sitting inside a
  triple term. New `retarget_reifiers` repoints the triple term onto the
  replacement instead of dropping it, so `prov:wasDerivedFrom` survives a repair
  rather than being lost or left dangling. `_shacl_repairs_for` returns a
  `_ShaclRepairPlan` pairing each removal with its replacement (first writer
  wins, since two violations can fire on one triple but a reifier reifies one
  statement). Retargeting runs after the accept test and before the prune sweep,
  which keeps "retyped *and* pruned in the same pass" correct by construction:
  the sweep matches the new term, so prune still wins.
- **`schema:position` and `schema:identifier` were reported as improvised
  vocabulary.** `_non_catalog_vocabulary_findings` skipped only `prov:`-prefixed
  *predicates*, so the chunk metadata the pipeline mints itself produced a
  `NON_CATALOG_VOCABULARY` warning per predicate on every run whose catalog does
  not happen to include schema.org — and a type is recorded against its object
  IRI, so the chunk node's own `prov:Entity` / `schema:Text` were flagged too,
  which the namespace guard on `rdf:type` could never catch. All of them are now
  skipped via `PROVENANCE_METADATA_TERMS`.
- **`facts_rejected_merges` stopped changing meaning mid-run.** The validate
  node overwrote the aggregator's guard count with `len(vetoes)` after any
  un-merge pass — a different quantity, already published one line earlier as
  `facts_merge_vetoes` — destroying the guard count for the graph actually
  served. It now republishes the count from the last accepted re-aggregation.
- **`/process_unit` could not report an empty ontology context.** The
  `validated_without_ontology_context` metric was written only by the document
  path, so the single-unit gate validated against an empty catalog silently.
  Both paths now write the SHACL and validation metric set through one shared
  `record_facts_gate_metrics`, so a counter added to one is present on both and
  batch dumps stay comparable — the two hand-maintained copies were the reason
  they had drifted. Pinned by a test asserting both paths emit the same key set.
- **The three local-embedding defaults are spelled consistently.**
  `AGG_EMBEDDING_MODEL` and `EMBEDDING_MODEL_NAME` now carry the
  `sentence-transformers/` prefix like `CHUNK_EMBEDDING_MODEL`. Identical
  checkpoint either way — `sentence-transformers` resolves a bare name to the
  same files — but `SharedEncoder` keys its process-wide cache on the literal
  string, so the same model written two ways loaded twice. Nine documentation
  snippets told you to align the three settings *using the unprefixed spelling*,
  which defeated the point; they are corrected. Does **not** invalidate the
  chunk cache: that is keyed on `CHUNK_EMBEDDING_MODEL`, which is unchanged.
- **`.env.example` documents every setting again.** Nine `CHUNK_*` variables
  were declared in `settings.py` and advertised nowhere, including
  `CHUNK_EMBEDDING_MODEL` — the one the performance guide tells you to align —
  and the four schema-detection thresholds. A new test diffs the two in both
  directions, so a knob added to settings or removed from it cannot drift from
  the advertised surface again.
- **`repair_ligature_gaps` is no longer labelled TEMP.** A knob that
  `CONVERTER_PROFILE=born_digital` turns *on* by default, that three doc pages
  describe, that four tests cover and that participates in the converter cache
  key is not temporary. The description now states a concrete removal condition
  instead.
- **The facts critic could not read the ontology it was judging.**
  `criticise_facts` built its ontology chapter with no index appendix, so on an
  opaque-IRI catalog (Wikidata-style `Q`/`P` codes) the critic was shown bare
  IRIs while facts guideline `6a` instructs the renderer to resolve them through
  the `# TERM INDEX`. It now uses the same memoised chapter the renderer does,
  which also stops re-serialising the ontology on every visit;
  `criticise_ontology` gets the same appendix. Costs nothing on readable
  ontologies — `build_ontology_index` returns `""` when no IRI is opaque.
- **Parse-time literal retyping is no longer numeric-only.**
  `normalize_literals_against_schema` retypes against any declared `rdfs:range`
  in an allowlist that now includes `xsd:date`, `dateTime`, `duration`, `gYear`
  and `gYearMonth`, so an `xsd:gYear` range receiving `"2019"^^xsd:string` is
  repaired at parse time rather than only at the SHACL gate, and only where
  shapes exist. Deliberately **not** "any XSD datatype", each case measured:
  every lexical form parses as `xsd:string` and `xsd:anyURI`;
  `Literal("2019", datatype=xsd:boolean).value` is `False`, not `None`; and
  `"2019"` parses as `xsd:time` 20:19 — so any of those as a range would let one
  sloppy declaration rewrite correctly typed values. The gregorian datatypes
  have no rdflib value parser at all (`.value` is always `None`) and are
  validated against their lexical space instead. Language-tagged literals are
  skipped, since retyping an `rdf:langString` discards the tag.
- **Guideline numbering no longer skips 11.** The JSON-LD clause in the facts
  prompt was numbered `12.` while `11.` came from the *conditional* search
  guidelines, absent whenever web grounding is off — the default. It is now
  `10a.`, matching the template's existing `1a.`/`6a.` convention, so it cannot
  collide with the injected rule or leave a gap.
- **Two error paths no longer report a cause they never checked.** Qdrant's
  `_ensure_payload_index` logged "already exists" for every failure including
  auth rejections and timeouts, leaving the index absent while the log said
  otherwise; the Ollama embedding fallback discarded the original exception with
  no log at any level, so a bad `EMBEDDING_BASE_URL` and an absent langchain
  integration were indistinguishable. Both now log the real exception. LanceDB's
  three equivalents were reworded to match.
- **Four documentation claims that were simply false.** `updated_at` /
  `dcterms:modified` "timestamp tracking" was advertised in `README.md` and
  `concepts.md` but exists nowhere — `Ontology` carries only `created_at`,
  serialized as `dcterms:created`. `ontology_context.md` listed
  `VECTOR_STORE_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES` as `550` against an actual
  `1200`. `create_vector_store_manager`'s docstring — which renders in the API
  reference — said `AUTO` falls back to the in-memory store when it resolves to
  `NONE`. And `performance.md` gave `CHUNK_EMBEDDING_MODEL` without its
  `sentence-transformers/` prefix, on the one page whose point is that settings
  naming the same model share one resident copy, keyed on the literal string.
- **Default test runs are offline, model-free, and reproducible.** Removed
  `env_files = [".env"]` from pytest configuration in `pyproject.toml` to
  prevent local test runs from automatically loading developer-specific `.env`
  files, avoiding accidental LLM-calling tests and environment leakage (local-green/CI-red).
  Added a dynamic deselection hook (`pytest_collection_modifyitems`) in `conftest.py`
  to automatically deselect tests marked as `slow` or `integration` when pytest
  is invoked without any marker selectors (e.g. standard `uv run pytest`),
  preserving the ability to run them via explicit commands like
  `uv run pytest -m slow` or `uv run pytest -m integration`.
- **SPARQL literals are escaped.** `GraphUpdate._serialize_rdf_term` wrapped a
  literal in bare double quotes, so any `"`, `\`, newline or carriage return —
  routine in extracted text — closed the string early and failed the whole
  update with a `ParseException` in `_apply_update_query`, losing every triple
  in the operation. Escapes are emitted explicitly rather than via
  `Literal.n3()`: n3 writes a raw tab, which rdflib's own SPARQL parser reads
  back as spaces.
- **Absolute IRIs outside `http` are no longer emitted as prefixed names.** The
  same method passed through any IRI containing `:` and not starting with
  `http`, so `urn:`, `doi:`, `file:` and `mailto:` IRIs reached the parser
  unbracketed as undefined prefixes. Abbreviations are now recognised against
  the prefixes the query actually declares instead of by shape.
- **SHACL retype fires for inline property shapes.** `sh:sourceShape` was kept
  only when it was a `URIRef`, so a violation from the common
  `sh:property [ sh:path … ; sh:datatype … ]` style arrived with no shape and
  the `sh:datatype` retype branch never ran. pyshacl reports the same blank node
  the shapes graph holds, so the datatype now resolves.
- **Blank-node SHACL violations reach the report.** Scope was decided on the
  projected finding, whose subject is a stringified blank node matching no
  namespace prefix, so every blank-node violation was dropped — while the repair
  pass acted on exactly those nodes. Report, repair, and the
  `violations_before`/`violations_after` metrics now share one scope predicate:
  IRIs by namespace, blank nodes by presence in the facts graph.
- **`SHACL_PRUNE` sweeps orphaned provenance.** A pruned node is also named
  inside `rdf:reifies <<( s p o )>>`, which no subject/object pattern matches,
  so the reifier and its `prov:wasDerivedFrom` survived describing a deleted
  statement. New `drop_reifiers_mentioning` clears them through pyoxigraph —
  `rdflib.Graph.remove` raises on a triple-term triple — and runs only after a
  pass is accepted, so a reverted pass leaves provenance intact.
- **`TripleFix.text_fragment` and `TripleFix.explanation` coerce free text.**
  Both are required with no default, so a provider answering either with a
  bulleted list raised and discarded every fix in the report. Also applied to
  `ExternalEvidencePlan.rationale`, which had diverged from the identically
  named field on `ExternalEvidenceRequest`. Graph-syntax fields
  (`incorrect_value` / `correct_value`) stay strict.
- **`bibliography.py` names the shipped default.** Its docstring marked
  `citations_only` as the default; `ChunkConfig.bibliography_mode` has defaulted
  to `skip` since the flag was introduced.
- `demo/README.md` and `docs/user_guide/llm_caching.md` documented CLI flags
  that do not exist (`--working-directory`, `--ontology-directory`), so the
  commands failed on invocation. Rewritten as env-var form.
- **The SHACL gate no longer crashes on the aggregated graph's RDF 1.2 triple
  terms.** The autofix copied the graph with bare `Graph.add`, and `run_shacl`
  handed the oxigraph-backed graph straight to pyshacl — both assert on the
  `rdf:reifies <<( s p o )>>` provenance the aggregator emits, so the default
  configuration (`FACTS_SHACL_AUTOFIX=prune`) failed `VALIDATE_FACTS` on any
  real document once shapes were configured. Validation now runs on a
  sanitised copy (reification carries no shape targets, so it loses nothing),
  and repairs are applied **in place with per-pass rollback** instead of on a
  copy, so the served graph keeps its provenance triple terms.
- **A focus node absent from the facts graph can no longer be "pruned".** With
  the ontology mixed into validation, pyshacl also reports catalog nodes;
  blank ones bypass the namespace scope check, and "no outgoing triples" read
  as "asserts nothing". The result was an empty repair record whose no-op pass
  tripped the strict-decrease revert and threw away genuine repairs in the
  same round. Scope is now presence-based for blank nodes, and an absent node
  is never an empty placeholder.
- **`conforms` and the `facts_shacl_violations_*` metrics count the same
  population.** The metrics counted raw pyshacl violations (facts + mixed-in
  ontology) while `conforms` was judged on fact-scoped findings, so a run
  could report `conforms: true` next to a nonzero `violations_after`. Both now
  use the fact-scoped population; the autofix loop's accept test still uses
  the raw count internally.
- **`crawl_directories` matches suffixes case-insensitively** (`report.PDF` is
  a PDF) and **raises when an explicitly named file is excluded by the prefix
  filter** instead of silently returning nothing — the exact no-op failure
  issue #53 removed. `pdfs-to-markdown` now exits non-zero on an empty crawl,
  like `ontocast process`.
- **The Docker image works again.** The build used `--no-group` flags against
  a project whose tiers are extras (installing the bare core: no server, no
  provider), the runtime stage lacked `uv` yet the entrypoint invoked it, and
  bare `ontocast` no longer starts the API. The image now installs every
  runtime extra and starts `ontocast serve` (override `CMD` for batch runs);
  the stale `ontocast-mcp-server` / `0.1.1` OCI labels are gone.
- **Documentation matched the code again** across README, docs and
  `.env.example`: settings that no longer exist removed
  (`CHUNK_BREAKPOINT_*`, `CHUNK_STRATEGY`, `PARALLEL_*_RETRIES`,
  `ONTOCAST_RECALL_*` and the deleted recall-harness instructions); wrong
  defaults corrected (`CHUNK_SECTION_CLASSIFIER`, fusion/dedup weights,
  `PARALLEL_WORKERS`, `QDRANT_UPSERT_BATCH_SIZE`, derived table names);
  examples that fail `Config()` validation fixed (`granite3.3`, the elliptical
  `FACTS_CODE_PREDICATES` sample); pre-0.5.0 CLI forms in the caching guide;
  the malformed `docker compose` command; `/health` documented as liveness,
  not readiness; the vector-store `auto` fallback and
  `FUSEKI_ONTOLOGIES_DATASET` descriptions un-contradicted.
- **SHACL findings no longer drive the un-merge repair.** Every error finding
  reached `_vetoes_from_findings`, and `sh:Violation` maps to error severity —
  so a missing required property could dissolve a legitimate identity cluster
  whenever the focus node happened to be merged, and the repair loop's
  accept test (violations must strictly decrease) was scored on constraints
  un-merging cannot fix. Vetoes and the loop's objective are now restricted to
  the merge-signature kinds (`FUNCTIONAL_VIOLATION`, `SUSPECT_MULTI_VALUE`,
  `DEGENERATE_COREFERENCE`).
- **SHACL now validates against the ontology context, not the facts alone.**
  A facts graph states that a value uses `unit:DAY`; that this individual *is*
  a `qudt:Unit` is stated only in the catalog, so every `sh:class` constraint
  pointing at a catalog term failed. On the three-document matsci pilot this
  was **128 of 360** reported violations — phantom findings describing the
  absent schema.
- **SHACL runs with RDFS inference by default** (`FACTS_SHACL_INFERENCE`,
  `FACTS_SHACL_ADVANCED`), matching how the shipped shapes are authored and
  validated by their own repo harness. SHACL property paths carry no
  `rdfs:subPropertyOf` entailment, so a shape naming a superproperty reported
  the specialised predicate the renderer emitted as missing: 268 violations at
  `none` against 232 at `rdfs` on the same pilot. A `FACTS_SHACL_MAX_TRIPLES`
  guard skips oversized graphs with a warning rather than stalling, and a
  skipped run is reported as "did not run", never as "conforms".
- **Generated API reference navigation.** `docs/gen_pages.py` indexed the
  literate-nav entries with a path already relative to `reference/`, doubling
  the prefix, so all 168 generated sidebar links 404'd while the pages
  themselves rendered — 336 build warnings, exit code 0.


- **A shared embedding model with a lock in only one of its users.** Once
  retrieval and clustering shared one `SentenceTransformer`, the lock in the
  retrieval module protected nothing: `tool/agg/clustering.py`,
  `tool/agg/entity_aligner.py` and semantic chunking called `.encode()` on the
  same object unguarded. The lock now belongs to the `SharedEncoder` that owns
  the model. Concurrent `encode()` is *correct* with default arguments, so this
  was a broken invariant and a peak-memory risk, not corruption.
- **One global embedding lock for all checkpoints** — locks are now per model,
  so unrelated checkpoints no longer queue behind one another.
- **The `semantic-chunking` extra required `langchain-huggingface` but not
  `sentence-transformers`,** so installing it alone degraded silently to naive
  chunking. It now requires `sentence-transformers`, and the probe checks
  `hdbscan` and `umap` rather than relying on `langchain-huggingface` as a proxy.
- **A failed chunker model load was retried on every call** — `None` meant both
  "not loaded" and "load failed". Failure is recorded once.
- **Section labels smeared across the document when a heading was
  unrecognised.** `_build_spans_from_heading_starts` ended each section span at
  the next *recognised* heading, so one unmatched heading let the previous label
  run on. On a paper with headings `Introduction / Experimental Section /
  Results and Discussion / Conclusions and Outlook / References`, only two spans
  were produced — `introduction` covering everything up to `References`. Because
  the label was stamped onto segments at split time and both the tagger and the
  LLM backfill skip already-labeled segments, the wrong label could never be
  corrected, and `target_sections=["results"]` returned introduction text or
  nothing. Every heading now closes the preceding span, and unrecognised ones
  open an explicitly unresolved span. Forward-fill and chunk merging respect
  that unresolved state, so neither reintroduces the smear.
- **Heading matching missed most real-world headings.** The anchored patterns
  required the heading to be exactly a canonical section name, so `Results and
  Discussion`, `RESULTS AND DISCUSSION`, `Experimental Section`, `Conclusions
  and Outlook`, `Device fabrication`, `Data Availability` and `Author
  Contributions` all failed. Publisher decoration defeated matching outright
  (`■ REFERENCES`, `■ ACKNOWLEDGMENTS`, `*sı Supporting Information`), as did
  section numbering (`2.1 Synthesis of thin films`). Measured on real docling
  conversions of three journal PDFs, 26 of 31 detected headings were unmatched
  and two of the three documents produced **no spans at all**.
- **Unheaded front matter was not recovered unless the paper opened with an
  IMRaD section.** `inject_front_matter_spans` bailed when the first labeled
  span was not `introduction`/`related_work`/`background`, leaving the title,
  abstract and introduction of papers that open with `Results` unlabeled.
- **`document_type_hint` matched needles buried inside longer words.** Hint
  matching was a bare substring test, so `epo` matched "r*epo*rt" (routing any
  unrecognised "…report" hint to the patent schema), `paper` matched
  "news*paper*", and `iso` matched "isotope". Needles now match on word
  boundaries, longest first, so the most specific hint wins regardless of YAML
  order. The `novel` → `fiction` needle was dropped rather than anchored: no
  word boundary separates the noun from the far more common adjective ("a study
  of novel materials" resolved to `fiction`), and detection scores real fiction
  at full share anyway. This gates detection as well as labeling — a false
  positive here suppressed detection entirely and imposed an unrelated schema.
- **The label schema was resolved three times per document, from the raw
  request each time** (`prepare.py` twice, `section_llm.py`, `inspect_sections.py`).
  Harmless while resolution was a pure function of the request, but it becomes
  silent label loss the moment it depends on document text: the deterministic
  tiers tag against the detected schema while the LLM backfill validates against
  the default, and `normalise_llm_label` *drops* labels absent from its schema
  rather than erroring. Resolution now happens once in
  `resolve_prepare_schema`, and the resolved schema is threaded to the backfill
  and to the CLI — so the schema `ontocast sections` reports is necessarily the
  one the chunks were labeled against.
- **`manual.yaml`'s `^using\s+` and `^how\s+to\s+` patterns were unbounded at
  the tail** — the catalog's only open-ended patterns, so `^using\s+` claimed
  headings such as "Using Creative Commons Public Licenses" for the manual
  schema. Harmless while schemas were only ever matched one at a time; a
  cross-schema scorer reads it as evidence. Both now bound the trailing words,
  matching short instructional headings ("Using the API") as intended.
- **`ontocast sections` could not read JSON or plain-text documents.** It called
  the Docling converter for every input, but the Convert node routes `.json` and
  `.txt` *around* the converter (Docling rejects them), so inspecting the files
  the pipeline is normally driven with — `data/json/*.json` — failed with
  "Input document is not valid". The routing and the JSON text heuristic now
  live in `onto/docling_helpers.py::json_payload_text` and are shared, so the
  CLI and the pipeline cannot disagree about what a file's text is. A JSON
  payload holding no document text (the shape of `clinical.trials.*.json`) now
  fails with a clear message rather than inspecting as an empty document.

### Performance

- **SHACL runs once or twice per document, not two to four times.** The
  autofix reuses the violations the reporting pass just computed
  (`apply_shacl_repairs(initial_violations=…)`) instead of re-validating the
  same graph — with RDFS inference and the ontology mixed in, each avoided run
  was a full materialisation.
- **`resolve_code_literals` no longer walks the whole catalog per rendered
  unit for nothing.** A surface index was built and never read (and doubled as
  a semantically wrong early exit); superclass closures are memoised per call;
  the no-declared-range fallback collects its usage evidence in one graph scan
  instead of rescanning per code literal.
- **Merged facts ontology is built once per document, not once per unit.** It
  reads only document-level state, but was rebuilt N+2 times — synchronously, on
  the event loop, so it stalled every other unit's in-flight provider call. At
  3.6k triples / 30 units the stage went from 6.07 s (5.97 s of it event-loop
  stall) to 0.21 s.
- **The ontology snapshot is shared by reference, not deep-copied per unit,** and
  its prompt chapter is serialised once per document rather than per unit per
  render attempt (`OntologySnapshot.prompt_chapter`).
- **Summarization runs inside the extraction fan-outs** instead of as a stage.
  A summary depends only on its own unit, so the node was a barrier that made
  the whole document wait for the slowest one.
- **Chunk preparation no longer blocks the event loop** — its synchronous
  segmentation and local-embedding phases run via `asyncio.to_thread`.
- **Aligning `CHUNK_EMBEDDING_MODEL` with `EMBEDDING_MODEL_NAME` and
  `AGG_EMBEDDING_MODEL` now leaves one resident local model instead of two**
  (measured: 1601 MB vs 2252 MB peak RSS). Defaults unchanged; opt-in recipe in
  [Performance](docs/user_guide/performance.md).
- **Fewer redundant loads and connections:** the docling `HybridChunker`
  tokenizer is cached instead of rebuilt per document; the docling converter
  lock guards only the lazy build, not the conversion (which serialised
  concurrent documents); BM25 moved to `ToolBoxRuntime` instead of one per
  tenancy scope; `ToolBox.serialize` uses one event loop and one backend
  connection per document rather than one per ontology.
- **The embedding lock no longer serialises remote providers**, which had
  nothing to protect.
- **`PARALLEL_WORKERS` defaults to 16** (was 8), matching `LLM_MAX_INFLIGHT`.

### Removed

- **`working_directory` / `ONTOCAST_WORKING_DIRECTORY`.** Nothing had read it since the
  filesystem triple-store backend was deleted; both commands now start with no OntoCast
  env var set at all. See Breaking.
- Unreachable CLI modules `cli/split_chunks.py`, `cli/merge_ontologies.py`,
  and `cli/batch_process.py`: click commands with no console-script entry,
  never registered on the `ontocast` group, referenced by nothing.
- Dead `track_llm_usage` decorator, which had no call sites and still used the
  instance-attribute budget-tracker pattern the `ContextVar` replaced.

## [0.5.1] - 2026-08-07

### Breaking

- **Base install is the light embeddable core.** `pip install ontocast` no longer ships the HTTP server, CLI, LLM provider SDKs, document stack, or vector-service clients. Install at least one LLM provider extra (`openai`, `anthropic`, `google`, `ollama`); also `server`, `documents`, `qdrant`, `sparse`, `graph`. Existing extras (`doc-processing`, `lancedb`, `semantic-chunking`, `shacl`, `web-search`, `all`) keep their intent.
- **Console scripts require `ontocast[server]`** (`ontocast`, `plot-graph`, `cmp-states`, `match-graphs`, `pdfs-to-markdown`, `test-api`); base install prints the missing-extra hint via `ontocast.cli._entry`.
- **Document convert/chunk needs `documents`.** `AgentState.docling_doc` is typed `Any` with lazy coercion. Prefer `run_unit_pipeline` for single-unit input without Docling.
- **`QdrantConfig.distance` is `ontocast.onto.enum.VectorDistance`** (env values unchanged). Sparse embeddings return `ontocast.onto.sparse.SparseVector`. Vector-store managers export lazily.
- **`LLMTool.create` and Fuseki sync wrappers raise inside a running event loop** with a named awaitable, instead of a bare `asyncio.run` failure.
- **Tenancy is a ToolBox per scope**, not a retargeted shared one. `apply_request_tenancy` returns `(ToolBox, tenant, project)`. `InMemoryVectorStoreManager.apply_tenancy` drops the index (re-index after switch).

### Added

- **LangChain / LangGraph embedding.** `ontocast.integrations.langchain.ontocast_tools` (capability-gated tools; mutating tools opt-in; SPARQL read-only) and `make_ontocast_node` / `text_in_turtle_out` for third-party graphs. Docs: `docs/user_guide/embedding.md`; runnable `examples/`.
- **Zero-external vector path.** `InMemoryVectorStoreManager`, `VECTOR_STORE_BACKEND` (`auto`|`memory`|`qdrant`|`lancedb`|`none`), `Config.in_memory(**overrides)`.
- **Async / graph helpers.** `ToolBox.acreate` / `aserialize` / `require_*`; `build_agent_graph` (uncompiled) + `create_agent_graph(checkpointer=, store=, name=)`; top-level lazy exports (`AgentState`, `ontocast_tools`, …).
- **Per-scope tenancy.** `ToolBoxRuntime`, `ToolBoxRegistry`, `ToolBox.for_scope`, `Config.for_tenancy`, `TenancyScope`, `MAX_TENANCY_SCOPES` — deep-copied config per scope, shared LLM/embedder/converter, concurrent tenants.
- **`ontocast.util.optional.require`** — missing optional deps name the install extra.
- **Facts findings.** `rdfs:domain` contradiction (`domain_violation`); `SCALAR_AS_BOUNDS` (duplicate functional numeric properties); `DANGLING_REFERENCE` post-aggregation; fact-namespace classes raise `UNKNOWN_TERM` like fact-namespace predicates.
- **Sections-first chunking** (always-on classification). `CHUNK_SEGMENTER` (`semantic` default | `docling`), `CHUNK_SECTION_CLASSIFIER` (`llm`|`heading`|`off`). `exclude_sections` on `/process`, `/process_unit`, and `ontocast process` (academic default: `acknowledgements`, `appendix`).
- **Telemetry / retrieval.** Per-node `BudgetTracker.node_durations`; `skos:scopeNote` in seed gloss, BFS, and atom text (reindex recommended).
- **Import-weight CI** (`test/test_import_weight.py` + base/server install jobs).

### Changed

- **Document metadata key aliases and identifier affixes.** `document_metadata`
  keys are resolved case-insensitively with camelCase / snake_case / kebab-case
  tolerance; bibliographic identifier and source keys also accept an optional
  leading or trailing `id` affix (e.g. `DOI`, `doi_id`, `arxivId`, bare
  `arxiv`). Unregistered keys with an identifier-shaped affix (`id`, `uid`,
  `uuid`, `guid`, `ref`, `reference`, `no`, `num`, `number`, `code`, `slug`,
  `handle`, `accession`, `key`) emit a structured `dcterms:identifier` blank
  node (scheme = stem) instead of minting a labeled `prov:Entity`. Companion
  pairs such as `project` + `project_id` attach the id onto the minted entity;
  registry `id`-affix aliasing is scoped to bibliographic identifier and source
  keys only (so `project_id` is not folded into `project`).
- **`CHUNK_SEGMENTER=semantic` default** (was Docling `HybridChunker`); **`CHUNK_BIBLIOGRAPHY_MODE=skip`** (was `citations_only`); academic `default_exclude` for acknowledgements/appendix.
- **schema.org canonicalized to `https://schema.org/`** (bindings + post-merge sanitize).
- **`PARALLEL_WORKERS` default 4 → 8.** Prompt examples drop `ex:`; quantity guidance defers to the ontology numeric-form contract.
- **Test suite.** Meaningful `slow` / `integration` markers; `test/manual/` ignored unless `ONTOCAST_RUN_MANUAL_TESTS=1`; duplicates collapsed; default run no longer loads sentence-transformers (~71 s → ~24 s CI).
- **CI unit matrix:** `UV_PYTHON` wired; 3.13 on `main` pushes only; uv cache on unit job. Pre-commit `ty check` syncs `dev,server,documents,qdrant,graph`.

### Removed

- Dead/duplicate tests and unused fixtures (~992 → 943). Recall harness (`test_retrieval_recall.py` + support) moved out of the unit suite — measurement belongs in `ontocast-validation`. Relative case5 aggregation “no damage” test dropped.
- Hard `langchain` dependency (unused).

### Fixed

- **RDF 1.2 triple terms no longer break graph copies or SPARQL generation**
  (#48, #49). Oxigraph-backed graphs yield triple terms as plain tuples, which
  rdflib's `Graph.add` rejects and SPARQL cannot express. `is_rdflib_triple` /
  `copy_triples` (`onto/rdfgraph.py`) filter them in `RDFGraph.copy`,
  `__add__`, and `__iadd__` — so `__deepcopy__` degrades instead of raising —
  and `GraphUpdate._serializable_triples` filters them out of INSERT/DELETE
  queries and diff summaries. `GraphUpdate._serialize_rdf_term` now raises
  `TypeError` on an unserialisable term rather than emitting a Python repr into
  the query, which only surfaced as a `ParseException` at apply time.
- **Critique reports accept free-text fields returned as lists** (#50). Several
  providers answer `systemic_critique_summary` (and
  `ExternalEvidenceRequest.rationale`) with an array of bullets; these are now
  joined with newlines by a `mode="before"` validator instead of failing
  validation and burning a critic retry.
- **`--input-path` accepts a single file and rejects a bad path loudly** (#53).
  `crawl_directories` returns the file itself when given one, and raises
  `ValueError` for a path that does not exist or has an unsupported suffix; the
  CLIs surface that as `BadParameter`. `ontocast process` also exits non-zero
  when a directory matches no supported input, instead of printing a line to
  stdout and exiting 0.
- Cross-chunk person/entity identity merge (initials-aware aliases; label-confirmed pairs bypass cosine gate).
- Per-unit `retrieval_metrics` fold back into document state; Docling chunker tokenizer budgeted from `CHUNK_MAX_SIZE`; semantic chunker guards for tiny sections.
- Path-dependent ontology/matsci tests; concurrency bound flake; dead tenancy self-assignment; `test-api` entry shim.

### Performance

- Fan-outs use slim `UnitLoopContext` instead of `AgentState.model_copy(deep=True)`.
- URDNA2015 hashing off the per-unit hot path (`working_graph_changed` via triple sets; lazy `OntologySnapshot.content_hash`).

## [0.5.0] - 2026-08-05

### Breaking

- **CLI is a Click group: `ontocast serve` / `ontocast process`.** Bare
  `ontocast` no longer starts the API; batch extraction moves from
  `ontocast --input-path …` to `ontocast process --input-path …`. Batch mode
  writes provenance-stripped `*.facts.ttl` / `*.ontology.ttl` beside each
  input (or under `--output-dir`, with optional `--facts-output-dir` /
  `--ontology-output-dir`). `--max-visits` overrides the server/batch visit
  budget. Filename → `dcterms:title` when `--document-metadata` is omitted is
  unchanged.
- **`ontocast serve` binds loopback by default.** Bind interface is `HOST`
  (default `127.0.0.1`; was hardcoded `0.0.0.0`). The server has no auth and a
  destructive `POST /flush`, so non-loopback bind is explicit. Containers must
  set `HOST=0.0.0.0` (logs a warning recommending an authenticating proxy).
- **`/process` answers 422 when no content unit produced output** (was HTTP
  200 with `status: "success"` and empty facts). Malformed request parameters
  answer 400 rather than 500; conversion failure maps to 422 for parity with
  `/process_unit`.
- **Install extras and hard dependencies reshaped.** A base
  `pip install ontocast` was unimportable (`docling_core` imported at module
  scope while only declared under `all`). `docling-core[chunking]` is now a
  hard dependency; heavy Docling/OCR/embeddings stay under `doc-processing`
  (the extra docs already named, which previously did not exist). `torch` /
  `hdbscan` / `umap-learn` / `langchain-huggingface` move to
  `semantic-chunking`; `duckduckgo-search` to `web-search`. Unused hard deps
  dropped (`asyncio` PyPI backport, `httpx2`, `numba`, `simsimd`,
  `rapidfuzz`, `owlready2`, `langchain-experimental`); previously transitive
  imports declared (`pydantic-settings`, `numpy`, `scikit-learn`,
  `starlette`, `python-dotenv`). Wheel no longer ships a top-level `data`
  package (~15 MB sample PDFs); sdist excludes `data/` / `docs/` / `demo/`.
  Runtime dependency ranges gain floors matching what is actually tested and
  upper bounds.
- **Embedding / surface-form contract through `sf6` — reindex required.**
  The surface-form contract versions the protocol defining how ontology terms are converted
  into indexed text (with `sf6` as the latest revision). Changing it changes stored vectors,
  so existing collections raise `EmbeddingContractMismatchError` and must be dropped
  (`VECTOR_STORE_WIPE_ON_INIT` / `--wipe-vector-store`). Across this release, the stored
  atom text and payloads changed to: atomize only IRIs an ontology *describes* (`sf3`→`sf4`);
  derive `entity_role` from property *declaration*, not incidental predicate use (`sf4`→`sf5`);
  index
  `dcterms:alternative`, case-preserved `symbol_surfaces`, and
  `qudt:symbol` / `qudt:ucumCode` as retrieval surfaces; BM25 with query
  encoder + IDF modifier and label-bearing minimal text; English-first
  literal ranking for multilingual labels. Ontology content hashes also move
  to RDF value-space canonicalization (round-trip stable) — `versioned_iri`
  and `atom_id` change, so Fuseki named graphs and the vector index need a
  one-time rebuild together.
- **`VECTOR_STORE_CONSISTENCY_CRITIC_SIMILARITY_THRESHOLD` →
  `VECTOR_STORE_CONSISTENCY_CRITIC_MIN_FUSED_SCORE`**, default `0.7` → `0.5`.
  The value was always a fused reciprocal-rank score, never a cosine cutoff;
  `0.5` means top-ranked in the dominant dense channel.
- **Public apply / aggregation shapes changed.** Ontology apply:
  `partition_triples_by_namespace` / `apply_partitioned_updates` (delete
  propagation + `base_overrides`); `build_ontology_delta_graph` returns
  `OntologyDelta`; `normalize_ontology_units` takes `delete_graph`;
  `repair_property_aliases` returns `(rewritten, findings, applied_records)`.
  Facts aggregation: `aggregate_graphs` / `postprocess_facts_units` return
  `AggregationResult` and accept `merge_vetoes`.
  `EntityNormalizer.extract_entity_context` returns `EntityContext` instead of
  a 6-tuple.

### Added

- **CI and publish guards.** Non-slow pytest on Python 3.12/3.13; wheel
  import-smoke with no extras; tag-vs-`pyproject.toml` version check on
  publish; `py.typed` marker; packaging metadata (`authors`, `keywords`,
  project URLs, trove classifiers); startup bounds on `PORT`,
  `BASE_RECURSION_LIMIT`, `ESTIMATED_CHUNKS`, `PARALLEL_WORKERS`,
  `ONTOLOGY_MAX_TRIPLES`.
- **Declared HTTP contract** on `/process`, `/process_unit`, `/flush`,
  `/health` (`response_model` + per-status bodies); uniform `StatusErrorBody`;
  `/info` reports converters the install actually supports.
- **Response / telemetry provenance.** `ProcessResultMetadata.failed_units`,
  `facts_repairs` (`GraphRepairRecord`s for deterministic rewrites), and
  `improvement_suggestions`; per-attempt facts-loop telemetry on
  `UnitFactsState.attempt_log` → `AgentState.facts_loop_telemetry`.
- **Document-level provenance from payload metadata.** Optional
  `document_metadata` on `/process`, `/process_unit`, and
  `ontocast process --document-metadata` attaches caller identity to
  `doc_iri` (`prov:Entity` / `foaf:Document`, bibliographic ids, typed
  people/projects). Survives chunk-level `strip_provenance`.
- **Post-aggregation validation gate (`VALIDATE_FACTS`).** After
  `MERGE_FACTS`, deterministic invariants on the merged graph (functional /
  cardinality, suspect multi-values, degenerate coreference, optional SHACL
  via `FACTS_SHAPES_DIR` / inline shapes; extra `shacl`). Error findings veto
  whole merge clusters and re-aggregate (`FACTS_MERGE_REPAIR_PASSES`). Closes
  the transitive coreference gap pairwise merge guards cannot see. Findings on
  `AgentState.facts_validation_findings`.
- **Bibliography routing (`CHUNK_BIBLIOGRAPHY_MODE`, default
  `citations_only`).** Reference-list chunks extract as citation metadata only
  (or `skip` / legacy `domain_facts`). Configurable citation vocabulary
  (`CHUNK_CITATION_VOCABULARY`); domain nouns removed from the citation prompt.
- **Configurable facts vocabulary policy.**
  `FACTS_QUANTITY_FALLBACK_VOCABULARY` (QUDT default; empty forbids out-of-
  context fallbacks), `FACTS_ADDITIONAL_STANDARD_NAMESPACES` (meta-vocabs
  built-in; domain vocabs opt-in). `NON_CATALOG_VOCABULARY` warning when the
  graph uses terms the ontology context never supplied.
- **Object-property literal quarantine** (`FACTS_OBJECT_PROPERTY_LITERAL_CHECK`,
  default on): string literals on class-ranged / `owl:ObjectProperty`
  predicates go to the critic with the declared range as hint.
- **Ontology snapshot / catalog read path.** `OntologySnapshot` prompt view;
  catalog key is ontology IRI (`ontology_id` / author prefix are aliases);
  `OntologyHeader` + targeted `TripleStoreManager` reads (`aselect`,
  `afetch_ontology_catalog`, `afetch_ontologies_by_iri`, `aconstruct`);
  `OntologyManager` caches graphs by `versioned_iri` and serves per-document
  rather than per-unit catalog merges. Author prefixes persist via SHACL
  `sh:declare`. Optional
  `VECTOR_STORE_INDUCED_SUBGRAPH_CANDIDATE_PUSHDOWN`. Documented in
  [Ontology Catalog](docs/architecture/ontology_catalog.md).
- **Lexical-trigger retrieval lane** (`VECTOR_STORE_LEXICAL_TRIGGER_*`):
  case-sensitive match on notation / symbol / UCUM (and optional code-shaped
  labels) injects additive seeds outside the semantic budget; calibrated
  score + `max_merge` fusion so trigger evidence is not discarded when the
  semantic lane already found the IRI. Optional query-side unit signals
  (`VECTOR_STORE_QUERY_UNIT_SIGNALS_ENABLED`, default off). BM25
  case-mismatch policy for symbol surfaces
  (`VECTOR_STORE_SYMBOL_CASE_MISMATCH_*`).
- **Snapshot assembly controls.** Schema closure over `rdfs:domain` /
  `rdfs:range` (`ONTOLOGY_PATCH_SCHEMA_CLOSURE_*`); per-ontology and
  per-role atom floors; small-module closure; window-scaled seed caps
  (`ONTOLOGY_PATCH_SEEDS_PER_WINDOW` / `ONTOLOGY_PATCH_MAX_ATOMS_BASE`);
  induced-subgraph symbol predicates and type-promotion score preservation;
  `sum_score` cross-window merge mode (non-default); `EMBEDDING_QUERY_PREFIX` /
  `EMBEDDING_DOCUMENT_PREFIX`; vector init hygiene
  (`VECTOR_STORE_WIPE_ON_INIT`, orphan prune, reindex concurrency).
- **Docling converter configuration** via `CONVERTER_*` (including
  `born_digital` preset).
- **Retrieval recall harness** (`test/test_retrieval_recall.py`,
  `test/retrieval_gt.py`): real embeddings + Qdrant; measures term-level recall and
  funnel metrics across both *seed recall* (the share of expected ontology terms that survive
  retrieval ranking and budget truncation) and *snapshot recall* (the share of expected terms
  actually defined in the ontology context handed to the model, including neighbor nodes);
  evaluates against Text2KGBench and prebuilt-corpus tiers (`ONTOCAST_RECALL_*`); and provides
  ablation controls that flip index/retrieval axes without editing corpus files on disk.

### Changed

- **Retrieval defaults retuned against measured recall** (evaluated against Text2KGBench and
  the internal materials-science corpus; configurations optimized only for the internal
  corpus are flagged as single-corpus fits in the configuration guide to caution against
  overfitting). Notable defaults: `ONTOLOGY_PATCH_MAX_ATOMS` / `_BASE` → 96;
  `VECTOR_STORE_TOP_K` 10 → 20; per-ontology seed quota 3 → 0; sparse fusion
  weight 0.2 → 0.8 and neighborhood 0.3 → 0.15; induced-subgraph triple
  budget 550 → 1200; per-ontology atom floor 0 → 2; small-module closure
  0 → 300. Patch-retrieval defaults lean toward max-score dedupe →
  best-first round-robin → window-scaled hard cap (relative floors / MMR /
  hybrid off). Label and symbol predicates are configuration-driven
  (`VECTOR_STORE_LABEL_PREDICATES` / `VECTOR_STORE_SYMBOL_PREDICATES`) and
  contribute to the embedding fingerprint when non-default.
- **Ontology snapshot / writeback decoupling.** Assemble
  `O* → OntologySnapshot`, propose complements on a scratchpad, apply
  namespace-owned updates onto catalog terminals. Cross-ontology reference
  ownership is deterministic (longest namespace, then lexicographic IRI).
  Seed round-robin visits best-scoring ontologies first.
- **Startup performance.** Batched dense embeds; overlap sparse with dense;
  lazy Docling converter; defer heavy ML imports; slim package `__init__`.
  OntologyManager patch/reindex paths are async-first.
- **Prefix / namespace hygiene for facts prompts.** Drop rdflib default
  bindings from the domain clause; keep author short prefixes canonical;
  leave reserved namespaces (`xml:`) alone; state the two-namespace contract
  once. Prompt JSON-LD `@context` lists only referenced prefixes.
- **Conversion.** `ConverterTool` builds Docling's PDF pipeline from typed
  config with config-aware cache keys; optional ligature-gap workaround.
- **Docs.** User-guide / `.env.example` defaults aligned with code; full
  patch-retrieval parameter table and tuning presets; snapshot vs catalog;
  catalog I/O metrics; pyoxigraph integer-subtype collapse documented as an
  accepted limitation.

### Removed

- **`PARALLEL_FACTS_RETRIES` / `PARALLEL_ONTOLOGY_RETRIES`.** Never read; the
  per-unit budget is `MAX_VISITS_PER_NODE`. Passing them now fails fast as
  unknown settings.
- **`FactsLoopAttempt.graph_hash`.** Replaced by `n_mandatory_findings` /
  `repair_failed`; the hash had no consumer and forced a URDNA2015
  canonicalization per attempt inside the async fan-out.
- **`ONTOLOGY_PATCH_CROSS_QUERY_MERGE_MODE=rrf`.** Re-ranked an unsorted
  concatenation (not reciprocal-rank fusion). Non-default and unused;
  `max_score` / `hybrid` unchanged.
- **`Ontology.from_working_context` identity lock** — snapshot / writeback
  decouple (see Changed).

### Fixed

- **Catalog / vector integrity.** Transient Fuseki list errors no longer
  look like an empty catalog and prune the ontology index; partial catalog
  fetches suppress prune; empty keep-sets are refused. Concurrent `?tenant=`
  retargets are locked; tenancy switches reset `ontology_manager`. Vector
  init is delayed until after wipe-on-init. Embedding fingerprint includes
  atomizer knobs that change stored payloads. Public retrieval wrappers
  default to configured budgets (were hardcoded 4× smaller).
- **Ontology delete and consolidation writeback.** GraphUpdate deletes
  propagate through map/reduce to catalog terminals (delete-then-reinsert
  nets out; cross-unit delete vs insert is conservative). Consolidation
  applies onto the map-stage artifact via `base_overrides`, not a stale
  pre-run base. Author-prefix collisions degrade instead of blocking ingest.
- **Facts repair and validation.** Deterministic repair runs at
  `MAX_VISITS=1` (`FACTS_REPAIR_VISITS`); gates on mandatory findings only;
  literal `rdf:type` objects coerced; parse-time numeric retyping and
  unambiguous property-alias rewrites; closed-range suggestions are
  case-exact. Validation gate output reaches the batch path and
  `/process_unit`; non-improving merge-repair passes revert;
  `DEGENERATE_COREFERENCE` / IRI multi-value vetoes read `values`;
  `owl:sameAs` excluded from multi-value checks; cross-document vetoes cover
  the whole canonical cluster; SHACL misconfiguration warns instead of
  silent pass; `FACTS_FUNCTIONAL_MIN_SINGLE_SUPPORT` is actually passed
  through. `facts_findings_residual` measures post-repair residual.
  `NON_CATALOG_VOCABULARY` and `UNKNOWN_TERM` agree on catalog membership;
  empty ontology context is reported by validate (and
  `empty_snapshot_reason` on the ensemble resolver).
- **Aggregation over-merge.** Merge guards block identity merges across
  disjoint numeric/temporal values, functional-ish IRI objects, co-objects,
  and fuzzy label matches on literal-bearing entities (`AGG_*`; regression
  fixtures under `test/data/case{4,5}`).
- **Retrieval correctness.** Induced subgraph keeps all seed-bearing
  components; individuals stay alongside promoted types; BFS admits by
  predicate role; proposition windows stride across long chunks; relative
  score floors handle negative similarities; version/hash filters relax per
  IRI instead of emptying context; snapshot prefixes bind only used
  namespaces; lexical triggers honour non-ASCII symbols and token
  boundaries; small-module closure works against triple-store catalogs and
  is blank-node idempotent; catalog identity no longer drifts after store
  round-trips; Turtle no longer rounds floats; `ontocast-turtle`
  serialization works for oxigraph stores with RDF 1.2 triple terms;
  ensemble retrieval no longer materializes the full catalog twice per unit.
- **Ops / CLI / config.** `FUSEKI_AUTH=user:password` accepted; `ontocast
  process` exits non-zero when every file fails; `AgentState(current_domain=…)`
  honours the caller; per-unit LLM budget uses a `ContextVar`;
  `LLM_MAX_INFLIGHT` aliased (was silently `LLM_LLM_MAX_INFLIGHT`);
  `ToolBox.aclose()` closes Fuseki and Qdrant (`QDRANT_TIMEOUT_SECONDS`);
  backend connections close on app shutdown.

## [0.4.3] - 2026-06-08

### Added
- **LanceDB** embedded vector store (`LANCEDB_ENABLED`, `LANCEDB_DATA_DIR`) as a local alternative to Qdrant.

### Changed
- **BREAKING**: Backend-agnostic vector retrieval settings moved from `QDRANT_*` to `VECTOR_STORE_*` (`top_k`, induced-subgraph limits, proposition windows, fusion weights, dedup mode, embedding batch size). `QDRANT_*` now covers connection/transport only (`URI`, `API_KEY`, collections, gRPC, `VECTOR_SIZE`, `DISTANCE`, `UPSERT_BATCH_SIZE`). Old `QDRANT_TOP_K`, `QDRANT_INDUCED_SUBGRAPH_*`, etc. are **ignored**.
- **BREAKING**: Configure **either** `QDRANT_URI` **or** `LANCEDB_ENABLED=true`, not both.

## [0.4.2] - 2026-06-08

### Added
- **In-memory triple store** — default pyoxigraph backend when Fuseki is not configured.

### Changed
- **BREAKING**: **Neo4j triple store removed** (`NEO4J_*` env vars no longer select a backend). Without Fuseki, OntoCast now uses the in-memory pyoxigraph store automatically.

### Removed
- `Neo4jTripleStoreManager` and `NEO4J_*` configuration.

## [0.4.1] - 2026-06-07

### Added
- **Structured documents** — section label catalog, section-aligned chunk prepare (segment → tag → filter → size), optional summarization (`target_sections`, `summarize_sections`, `section_schema_id`, `document_type_hint`).
- **Facts precision/recall/F1** on `POST /match/evaluate` (`fact_precision`, `fact_recall`, `fact_f1` and counts): relational triples only, excluding schema predicates and triples with ontological class/concept nodes in subject or object position.
- **Anthropic (Claude) and Google (Gemini) LLM providers** via `LLM_PROVIDER=anthropic|google`, with `ClaudeModel` and `GeminiModel` config enums.
- **Token usage reporting** in `BudgetTracker` when providers return `usage_metadata` on LLM responses (character counts remain the universal fallback).
- **LLM disk cache controls** on `LLMConfig`: `LLM_CACHE_ENABLED` (default on), `LLM_CACHE_READ_ONLY`, and in-memory plus on-disk stats via `LLMTool.get_cache_stats()`; `GET /info` exposes `llm_cache`.
- **Global LLM in-flight limit** (`LLM_MAX_INFLIGHT`, default 16) — shared semaphore caps concurrent provider requests across parallel unit workers.
- **Optional process concurrency cap** (`MAX_CONCURRENT_PROCESSES`) — limits simultaneous `/process` and `/process_unit` handlers (additional requests wait for a slot).
- **OpenAI Batch API helpers** (`ontocast.tool.llm_batch`) to export chat batch JSONL and import completed results into the LLM disk cache for offline benchmark pre-warming.
- **`BudgetTracker.cache_hits`** — disk-cache hits count toward character totals but not `calls_count`; included in budget summaries when non-zero.

### Changed
- **JSON-LD reinforced as internal exchange format** — compact JSON-LD (`@context` + `@graph`) when `LLM_GRAPH_FORMAT=jsonld`; prompt context, graph format instructions, and schema bindings share one format profile while runtime models stay canonical.
- **Section pipeline layout** — span detection and LLM backfill under `ontocast.tool.chunk`; section-label YAML in `ontocast.config.section_labels`.
- **Chunk prepare** — coalescing, section tagging, allowlist filtering, and size normalization in one pipeline inside the Chunk node.
- **LLM caching path** — `complete`, `extract`, `__call__`, and `acall` share one `_invoke_cached` implementation with consistent cache keys, optional disable/read-only modes, and provider calls gated by the global in-flight semaphore.
- **Facts extraction prompts** (`facts_guidelines.py`): clearer two-namespace contract — domain ontology is read-only schema plus optional **reference individuals**; all text-derived occurrences use `cd:` with `lowercase_snake_case` local names.

### Fixed
- **Entity alignment** (`EntityAligner`): identical `URIRef` across graphs always form a compatibility edge (score 1.0).
- **Match / evaluate API** (`match_models`, `triple_evaluator`, `match_common`): entity fields stay `URIRef` through Pydantic validation; triple projection and entity precision/recall use set-based unmatched counts.

### Documentation
- Structured documents, facts two-namespace model, entity alignment, LLM cache, and evaluate semantics (`concepts.md`, `workflow.md`, `api.md`, `configuration.md`, `aggregation.md`, `llm_caching.md`, `user_instructions.md`).

## [0.4.0] - 2026-05-26

### Added
- **Parallel map/reduce pipeline** for document processing: per-unit ontology and facts loops run concurrently with configurable `PARALLEL_WORKERS`, retry budgets (`PARALLEL_ONTOLOGY_RETRIES`, `PARALLEL_FACTS_RETRIES`), and a dedicated `/process_unit` endpoint for single-unit runs.
- **Robust semantic disambiguation across chunks**: embedding- and symbolic-aware entity alignment during aggregation (`EntityAligner`, connected-component clustering, `skos:altName` handling) with improved cross-unit identity resolution.
- **RDF 1.2 provenance support**: quoted-triple / reification syntax via `pyoxigraph`; provenance and alignment triples are split into a side artifact during ontology normalization; optional `strip_provenance` on `/process` and `/process_unit` omits reification scaffolding from API Turtle output.
- **Enhanced ontology update consolidation**: normalize → consolidate → structural check → consistency critic pipeline replaces the legacy sublimation stage; optional post-normalization consolidation pass via `ENABLE_ONTOLOGY_CONSOLIDATION`.
- **JSON-LD as LLM wire format**: `LLM_GRAPH_FORMAT=jsonld` emits compact JSON-LD (`@context` + `@graph`) for graph payloads while keeping canonical domain models (`GraphUpdate`, critique reports, etc.) at runtime; Turtle remains the default.
- Per-unit **ontology catalog selection** (`select_ontology_catalog`) with optional `ontology_selection_user_instruction`.
- **Ontology context modes**: `selected_single_ontology`, `selected_vector_search_ontology` (Qdrant stitched ensemble), and `fixed_single_ontology` (`ONTOLOGY_CONTEXT_FIXED_ONTOLOGY_ID`).
- **Qdrant vector retrieval** with dual-vector + BM25 hybrid fusion, patch-retrieval scoring/MMR caps (`ONTOLOGY_PATCH_*`), and induced-subgraph triple budgets (`VECTOR_STORE_INDUCED_SUBGRAPH_*` since 0.4.3; was `QDRANT_INDUCED_SUBGRAPH_*` in 0.4.0–0.4.2).
- **Embedding configuration** surface (`EMBEDDING_*`) and embedding-ready representation contracts for atomizer/retrieval pipelines.
- **Tenancy-aware storage**: `tenant` / `project` request parameters partition Fuseki datasets and Qdrant collections (`{tenant}--{project}--facts|ontologies`); defaults derive from built-in `ontocast` / `test`.
- REST **ontology management** routes: `POST/PUT/DELETE /ontologies` for catalog upload, replace, and delete.
- **Graph matching API**: `POST /match/entities`, `POST /match/derive-matches`, and `POST /match/evaluate` for entity alignment and triple/entity precision-recall evaluation.
- `match-dirs` standalone CLI client for batch benchmark evaluation against the match endpoints.

### Changed
- **BREAKING**: Ontology post-render processing now uses `normalize_ontology_units()` instead of `sublimate_ontology()`; provenance is extracted rather than inlined in the working ontology graph.
- **BREAKING**: CLI server module is `ontocast.cli.server` (entry point unchanged: `ontocast`); legacy `serve` module removed.
- Workflow graph restructured around parallel unit rendering, normalization, and optional consolidation before facts extraction.
- Fuseki/Qdrant dataset and collection names default from tenant/project naming when unset (explicit `FUSEKI_DATASET` / `FUSEKI_ONTOLOGIES_DATASET` still supported).
- Default `max_visits_per_node` is now `1` (override via `MAX_VISITS` or per-request `max_visits`).
- Graph format instructions, JSON Schema bindings, and prompt context chapters are driven by a shared format profile (`LLM_GRAPH_FORMAT`).
- Improved IRI policy, ontology access helpers, and atomizer coverage for facts and ontology cores.

### Removed
- `sublimate_ontology` agent stage and module (superseded by normalize + consolidate).
- Top-level `tool/aggregate` module path (aggregation lives under `tool/agg/`).

### Fixed
- GraphUpdate parsing and alignment edge cases across Turtle and JSON-LD encodings.
- Graceful initialization when vector store or optional backends are unavailable.
- Match endpoint robustness and evaluation semantics (label triples excluded from triple metrics).

### Documentation
- User guides updated for 0.4.0 pipeline (workflow, API, tenancy, ontology context, aggregation).
- API reference pages are generated at build time via `docs/gen_pages.py` (stale committed stubs removed).
- Workflow diagrams: `docs/assets/graph.png` (TB), `graph.lr.png` (LR); regenerate with `uv run plot-graph`.
- Configuration defaults aligned with `config.py` and `.env.example`.

## [0.3.0] - 2026-03-10

### Added
- `updated_at` timestamp field in Ontology properties for tracking last update time.
- Automatic semantic versioning with intelligent MAJOR/MINOR/PATCH increment analysis.
- Version analysis based on ontology changes (classes, properties, and instances).
- Hash-based versioning with parent hashes for git-style lineage tracking.
- `mark_as_updated()` in Ontology for version/timestamp management.
- `sync_properties_to_graph()` to persist `version` and `updated_at` in RDF.
- `versioned_iri` support for storing multiple ontology versions in triple stores.
- URL encoding for versioned IRIs in Fuseki to preserve `#` in named graph URIs.
- Multi-version ontology storage in Fuseki using separate named graphs.
- Automatic ontology synchronization from filesystem to triple store during initialization.
- `render_mode` processing options: `ontology`, `facts`, `ontology_and_facts`.
- Dedicated `serialize` workflow node; separated aggregation and serialization stages.
- API support for `render_mode` as a query parameter.
- **GraphUpdate** system with structured SPARQL insert/delete operations.
- `GraphUpdate`/`TripleOp` models for incremental graph modifications.
- `render_ontology_update()` and `render_facts_update()` GraphUpdate-based rendering.
- Automatic SPARQL generation from GraphUpdate operations.
- Budget tracking integrated in `AgentState`, including ontology/facts generation metrics.
- End-of-run budget summary reporting.
- Dependency-injected budget tracking for LLM calls.
- Shared caching architecture with a single `Cacher` instance and `ToolCacher` wrapper.
- `ONTOCAST_CACHE_DIR` environment variable for cache location.
- `serialize()` as a primary triple-manager interface for `Ontology` and `RDFGraph` objects.
- `ONTOLOGY_MAX_TRIPLES` guardrail to prevent unbounded ontology growth.
- Limit checks in `render_updated_graph()` and `sublimate_ontology()`.
- Parallel unit/chunk processing with configurable worker concurrency and retry behavior.
- More robust entity/property disambiguation across units/chunks during aggregation.
- Optional ontology consolidation switch via `ENABLE_ONTOLOGY_CONSOLIDATION`.
- Aggregation configuration via `AGG_EMBEDDING_MODEL` and `AGG_SIMILARITY_THRESHOLD`.
- Web grounding configuration surface (`WEB_SEARCH_*`) with planner, retry, evidence-budget, and domain filtering controls.
- `FUSEKI_ONTOLOGIES_DATASET` for separate ontology dataset configuration.

### Changed
- **BREAKING**: `serialize()` is now the primary interface for storing data in triple stores.
- **BREAKING**: `serialize()` now accepts `Ontology | RDFGraph` objects instead of raw `Graph` objects.
- **BREAKING**: `serialize_graph()` signature now uses `**kwargs` for backend-specific parameters.
- All triple store managers now implement both `serialize()` and `serialize_graph()`.
- **BREAKING**: Environment variables now use `ONTOCAST_` prefix:
  - `WORKING_DIRECTORY` → `ONTOCAST_WORKING_DIRECTORY`
  - `ONTOLOGY_DIRECTORY` → `ONTOCAST_ONTOLOGY_DIRECTORY`
  - `LLM_CACHE_DIR` → `ONTOCAST_CACHE_DIR`
- **BREAKING**: Ontology and facts rendering now use GraphUpdate/SPARQL operations instead of full TTL generation.
- LLM output now uses structured `GraphUpdate` + `TripleOp`, reducing token usage.
- Ontology version increments now derive from detected ontology diffs.
- Version updates now happen once at end of processing (`serialize`).
- LLM tool budget tracking refactored to dependency injection.
- Global `LLMBudgetTracker` replaced by AgentState-contained tracker.
- Agent functions updated to use injection-based budget plumbing.
- Server recursion control renamed to `BASE_RECURSION_LIMIT` (instead of `RECURSION_LIMIT`).
- `MAX_VISITS` remains supported as alias for `max_visits_per_node`.
- Default `ONTOLOGY_MAX_TRIPLES` increased to `50000`.
- Docs updated for new configuration sections and defaults (`Server`, `Aggregation`, and `Web Search`).

### Removed
- Global budget tracker state management.
- Manual budget tracker update calls inside agent functions.
- `set_budget_tracker()` and `get_budget_tracker()` functions.

## [0.1.7] - 2025-10

### Added
- Automatic LLM response caching for improved performance and cost reduction
- Platform-aware default cache directory selection
- Transparent caching with no configuration required

- Environment variable `SKIP_ONTOLOGY_DEVELOPMENT` to skip ontology critique step
- Environment variable `LLM_API_KEY` for LLM authentication (replaces `OPENAI_API_KEY`)
- Environment variable `MAX_VISITS` for controlling workflow behavior
- Environment variable `WORKING_DIRECTORY` for specifying working directory
- Environment variable `ONTOLOGY_DIRECTORY` for specifying ontology files
- Hierarchical configuration system with environment variable support
- Support for `.env` file configuration
- Python 3.12 type hint support (`str | None` syntax)
- `pathlib.Path` support for directory configurations
- Improved RDF graph operations with proper prefix binding

### Changed
- `OPENAI_API_KEY` environment variable renamed to `LLM_API_KEY`
- Configuration system refactored to use dependency injection
- `ToolBox` now accepts configuration objects directly
- `LLMTool` now accepts configuration objects directly
- Type annotations updated to Python 3.12 standards
- Path handling updated to use `pathlib.Path` objects
- Triple store configuration moved to environment variables

### Fixed
- RDF graph prefix binding issues
- Configuration validation errors
- Triple store initialization errors
- API key handling in LLM configuration
- Type annotation compatibility issues

### Removed
- Global configuration variable
- Support for `OPENAI_API_KEY` environment variable
- Individual parameter passing in tool initialization

### Security
- API keys now handled with secure string types
- Configuration validation prevents data exposure

## [0.1.5] - 2025-01-XX

### Added
- Automatic LLM response caching for improved performance and cost reduction
- Platform-aware default cache directory selection (avoids /tmp)
- Transparent caching with no configuration required

- Version bump to 0.1.5
- Various stability improvements

---

## Migration Guide

### Upgrading to 0.6.0 (from 0.4.3, the last published release)

Everything under 0.5.0, 0.5.1 and 0.6.0 above applies. The environment-level
breaks, in one place:

| Old | New |
|-----|-----|
| `FACTS_REPAIR_VISITS` | `FACTS_LLM_REPAIR_VISITS` (old name is a silent no-op) |
| `CHUNK_BREAKPOINT_THRESHOLD_TYPE` / `_AMOUNT` | removed (silent no-ops) |
| `ONTOCAST_WORKING_DIRECTORY` / `working_directory` | removed (ignored if set) |
| `CHUNK_STRATEGY` | `CHUNK_SEGMENTER` |
| `pip install ontocast` (full install) | base is the light core — add extras: `ontocast[server,openai]` |

Also: bare `ontocast` no longer starts the API (use `ontocast serve` /
`ontocast process`); `PARALLEL_WORKERS` default rose 8 → 16; the LLM cache key
format changed (`cache_format_version` 2), so existing on-disk caches re-fetch;
a 1 GB LRU cache ceiling applies by default (`ONTOCAST_CACHE_MAX_BYTES`); `ONTOCAST_WORKING_DIRECTORY` is gone (ignored if still set).

### Upgrading to 0.4.3

**Vector store env vars** — retrieval/indexing settings are backend-agnostic; rename `QDRANT_` → `VECTOR_STORE_` for:

| Old (ignored) | New |
|---------------|-----|
| `QDRANT_TOP_K` | `VECTOR_STORE_TOP_K` |
| `QDRANT_INDUCED_SUBGRAPH_DEPTH` | `VECTOR_STORE_INDUCED_SUBGRAPH_DEPTH` |
| `QDRANT_INDUCED_SUBGRAPH_HUB_SEED_COUNT` | `VECTOR_STORE_INDUCED_SUBGRAPH_HUB_SEED_COUNT` |
| `QDRANT_INDUCED_SUBGRAPH_ANCESTOR_CLOSURE_DEPTH` | `VECTOR_STORE_INDUCED_SUBGRAPH_ANCESTOR_CLOSURE_DEPTH` |
| `QDRANT_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES` | `VECTOR_STORE_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES` |
| `QDRANT_INDUCED_SUBGRAPH_ESTIMATED_TRIPLES_PER_QUERY` | `VECTOR_STORE_INDUCED_SUBGRAPH_ESTIMATED_TRIPLES_PER_QUERY` |
| `QDRANT_PROPOSITION_*` | `VECTOR_STORE_PROPOSITION_*` |
| `QDRANT_FUSION_*` | `VECTOR_STORE_FUSION_*` |
| `QDRANT_DEDUP_*` | `VECTOR_STORE_DEDUP_*` |
| `QDRANT_EMBEDDING_BATCH_SIZE` | `VECTOR_STORE_EMBEDDING_BATCH_SIZE` |
| `QDRANT_CONSISTENCY_CRITIC_SIMILARITY_THRESHOLD` | `VECTOR_STORE_CONSISTENCY_CRITIC_SIMILARITY_THRESHOLD` |

**Unchanged under `QDRANT_`:** `URI`, `API_KEY`, `ONTOLOGY_COLLECTION`, `FACTS_COLLECTION`, `GRPC_PORT`, `USE_GRPC`, `VECTOR_SIZE`, `DISTANCE`, `UPSERT_BATCH_SIZE`.

**LanceDB (optional):** `LANCEDB_ENABLED=true` and `LANCEDB_DATA_DIR=~/.lancedb_data` (`uv sync --extra lancedb`). Do not set `QDRANT_URI` at the same time.

### Upgrading to 0.4.2

**Triple store:**

```bash
# Old — Neo4j backend (removed)
NEO4J_URI=bolt://localhost:7687
NEO4J_AUTH=neo4j/password

# New — omit Fuseki for zero-config dev, or use Fuseki for persistence
FUSEKI_URI=http://localhost:3032
FUSEKI_AUTH=admin:password
# (no triple-store env vars → in-memory pyoxigraph)
```

### Upgrading to 0.4.0

**Environment variables:**

```bash
# Old (ignored in 0.4.0)
RECURSION_LIMIT=1000

# New
BASE_RECURSION_LIMIT=1000
```

**Defaults changed:**

| Setting | 0.3.x docs / `.env.example` | 0.4.0 code default |
|---------|----------------------------|-------------------|
| `MAX_VISITS` | often documented as `3` | `1` |
| `ONTOLOGY_MAX_TRIPLES` | sometimes `10000` | `50000` |
| Fuseki datasets | explicit `FUSEKI_DATASET` | derive `ontocast--test--facts` when unset |

**Removed APIs:**

- `ontocast.agent.sublimate_ontology` — use `normalize_ontology_units()` and optional consolidation instead.
- `ontocast.cli.serve` — server is `ontocast.cli.server` (CLI command `ontocast` unchanged).

**New request parameters:**

- `tenant`, `project` — partition Fuseki/Qdrant (query string on `/process`, `/ontologies`, etc.)
- `strip_provenance` — omit reification from API Turtle output
- `ontology_context_mode`, `ontology_context_fixed_ontology_id` — per-request ontology context

See [docs/user_guide/](docs/user_guide/) for full guides.

[0.6.0]: https://github.com/growgraph/ontocast/compare/v0.4.3...v0.6.0
[0.4.3]: https://github.com/growgraph/ontocast/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/growgraph/ontocast/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/growgraph/ontocast/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/growgraph/ontocast/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/growgraph/ontocast/releases/tag/v0.3.0
