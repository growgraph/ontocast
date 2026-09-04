# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.4] - unreleased

### Added

- **`ONTOLOGY_CHAPTER_FORMAT=term_sheet`** renders the `# ONTOLOGY` chapter as
  a line-per-term listing instead of a serialized graph: each term's name, the
  alternative surface forms a document may spell it with, its type, its place in
  the hierarchy, a property's domain and range, and the scope note saying when it
  applies. Dropped are the per-statement RDF scaffolding — a node wrapper or
  subject block per term and a repeated predicate IRI per statement — and
  `rdfs:comment`, which is written for someone browsing the ontology rather than
  for an extractor. The ontology chapter is the bulk of a facts prompt, so this
  is the largest context lever available, well beyond what `turtle` gives.
  Admissible because a facts prompt reads its ontology and writes an unrelated
  graph; the ontology loop writes a patch against the statements *in* its
  chapter, which a listing cannot express, so `term_sheet` requires
  `RENDER_MODE=facts` and a configuration asking for both is rejected at startup
  rather than silently falling back to a graph. Changes the LLM cache key for
  facts calls.

- **Character caps on the ontology chapter's text literals** —
  `ONTOLOGY_TEXT_MAX_CHARS_NAMING` (labels, preferred and alternative),
  `ONTOLOGY_TEXT_MAX_CHARS_CONTRACT` (scope notes, definitions),
  `ONTOLOGY_TEXT_MAX_CHARS_PROSE` (`rdfs:comment` and the remaining notes), and
  `ONTOLOGY_TEXT_TOTAL_BUDGET` across all of them. `ONTOLOGY_CONTEXT_MAX_TRIPLES`
  is a count and bounds no individual literal, so a chapter well inside it could
  still be arbitrarily long: chapter size tracked how much prose a catalog's
  authors wrote rather than how many terms it declares, and was paid on every
  call of every unit. These apply to every chapter the facts loop builds, term
  sheet and serialized graph alike, and are unset by default — inert when unset,
  byte-for-byte, so prompts and cache keys do not move for a deployment that
  sets none of them. Clipping is on a word boundary and leaves a visible marker,
  so a clipped definition reads as clipped; it is preferred to dropping the
  statement because a scope note's first sentence usually carries the contract.
  Over the total budget, prose is tightened then dropped, then contracts, and
  only then are names clipped to a floor — names are never dropped, and a
  chapter that still does not fit is passed through with a warning, matching how
  the triple budget refuses to cut into load-bearing structure. Reported in the
  run manifest as `text_chars_before`, `text_chars_after`, `literals_clipped`
  and `literals_dropped`.

- **`gpt-5.4` is the default OpenAI model** (`LLM_MODEL_NAME`), replacing
  `gpt-4o-mini`. Runs that do not set the variable move to it.

- **Reasoning controls for cloud providers.** `LLM_REASONING_EFFORT`
  (`none|minimal|low|medium|high|xhigh`) is the discrete depth knob, read by
  OpenAI reasoning models as `reasoning_effort` and by Gemini 3+ as
  `thinking_level`. The vocabulary is the union across providers and across
  model generations of one provider — the floor of the scale is spelled
  `minimal` by some models and `none` by others — so which levels a given model
  accepts stays the provider's business; an unsupported one is reported as a
  rejected request rather than guessed at or silently downgraded.
  `LLM_THINKING_BUDGET` is the Gemini 2.5 integer spelling (`0` disables where
  the model allows it, `-1` is model-chosen, a positive value is a cap),
  superseded from Gemini 3 on. Cloud equivalents of `LLM_THINK` for Ollama:
  reasoning tokens count toward the output total. Each knob joins the LLM
  disk-cache key only when set, so an unset knob leaves existing cache entries
  valid. A knob the configured model does not read logs a warning and is
  ignored — including `LLM_THINKING_BUDGET` on a Gemini 3+ model, where the
  thinking level supersedes it. On Google the two are mutually exclusive (the
  API's own rule) and setting both is rejected at startup rather than silently
  resolved, which would bill one setting while the run manifest recorded the
  other. Both are recorded in the run manifest `llm` block.

- **Unit-scoped fact IRIs before aggregation** (`AGG_UNIT_SCOPED_FACT_IRIS`,
  default `true`). After sanitization, instance IRIs under the fact
  namespaces are rewritten to `<local>__u<unit index>`. Aggregation keys by
  that scoped IRI, so a shared local name is a merge decision (cluster,
  then guard) rather than a dictionary collision. Served IRIs never carry
  the suffix: unmerged same-name entities mint `<name>` and `<name>_1` in
  unit order. Reifiers, `prov:wasDerivedFrom`, and `owl:sameAs` reference
  unscoped IRIs. `aggregation_clusters` and `AggregationResult.decisions`
  report scoped source IRIs. Predicates, `rdf:type` objects, schema
  targets, and terms typed as class or property are exempt.
  `unit_scope.strip_unit_scope` unwraps scoped IRIs for consumers that read
  per-unit graphs after aggregation. `false` restores name-keyed identity.

- **Inert-threshold warning.** The aggregator logs a warning when
  `AGG_SIMILARITY_THRESHOLD` is changed while
  `AGG_CANDIDATE_SIMILARITY_THRESHOLD` is at its default. The former
  belongs to the cross-graph `EntityAligner`; the in-pipeline aggregator
  does not read it.

- **`AtomicToolBox.catalog_terms()`** — memoised union of catalog-declared
  terms, built lazily by `OntologyManager.catalog_terms()` and keyed on
  content-addressed ontology ids so it rebuilds when the catalog changes.
  `ToolBox.get_atomic_tools()` returns a per-call copy bound to the
  requesting tenancy's catalog. Parse-time repairs use this to distinguish
  terms absent from the unit snapshot from terms absent from the catalog.

- **Batch validation dump records repairs and failures.**
  `*.facts.validation.json` gains `unit_repairs` (per-unit
  `GraphRepairRecord`s applied at parse: kind, source, target, triple count
  — same shape as HTTP `facts_repairs`) and `unit_failures` (unit index,
  phase, stage, reason).

- **`llm/calls_failed` budget counter.** Counts every provider call that
  raised. Timeouts and rate limits remain subsets (`llm/timeouts`,
  `llm/rate_limited`). Invariant: `calls_count = llm/calls_timed +
  llm/timeouts`. A timed-out call is charged its prompt characters.

- **`ONTOLOGY_CHAPTER_FORMAT`** (`inherit|turtle`, default `inherit`). Pins
  the `# ONTOLOGY` chapter of the facts render and critic prompts to Turtle
  regardless of `LLM_GRAPH_FORMAT`. Does not change the output wire, the
  facts chapter, or the ontology loop's chapters. The snapshot
  prompt-chapter memo is keyed on the chapter wire; the setting invalidates
  the LLM cache for facts calls.

- **Startup warning when `PARALLEL_WORKERS` exceeds `LLM_MAX_INFLIGHT`.** A
  unit never issues two provider calls at once, so extra workers only queue
  on the semaphore (`llm/inflight_wait`).

- **Front/back-matter routing (`CHUNK_NON_CONTENT_MODE`).** Units headed
  author information, notes, ORCID, data availability, competing interests,
  licence, supporting information, and similar (or whose tokens are mostly
  emails, URLs, ORCIDs, and initials) are recognised alongside bibliography
  detection. `extract` (default) keeps them and sets
  `SourceUnit.is_non_content`; `skip` drops them before fan-out. Routing
  order: `CHUNK_MIN_UNIT_CHARS` → bibliography → non-content. Each decision
  is logged; the run manifest `selection` block counts
  `undersized_units_skipped`, `bibliography_units_skipped`, and
  `non_content_units_skipped`.

- **Density-aware chunk split (`CHUNK_MAX_MEASUREMENTS_PER_UNIT`, default
  off).** A sized unit that states more unit-adjacent numbers than the cap
  is split at the sentence or paragraph boundary nearest its midpoint,
  recursively, never below `CHUNK_MIN_SIZE`. Pieces inherit headings,
  references, and section label; each split is logged.

- **`CONVERTER_REPAIR_NUMERIC_ARTIFACTS`** (default off, not part of
  `born_digital`). Pattern-local conversion repairs inside values: HTML
  entities (`&lt;` `&gt;` `&amp;` `&quot;` `&apos;`), carriage-return
  column wraps, flattened exponents (`2 × 10 6` → `2 × 10^6`; a bare
  `10 6` only after `~`/`≈`/"order of"), and single-sided ligature gaps
  with one reading. Superscript/subscript duplication and citation markers
  fused into values are left unchanged. The flag joins the converter cache
  key only when on.

- **`ontocast.util.measurement_lexicon`.** Shared scanner for unit-adjacent
  numbers (built-in SI/prefix/percent/time lexicon plus caller-supplied
  unit surfaces; compound tokens matched factor by factor), used by the
  density split and the numeric-coverage lane.

- **Insert-only facts completion pass (`FACTS_COMPLETION_PASSES`, default
  `0`).** After the critic loop, while the unit's numeric-coverage inventory
  still lists a measurement (number with unit) absent from the graph, a
  narrower pass runs. Each pass is shown a compact term sheet (the unit's
  quantity/observation/condition classes and unit individuals) plus
  existing catalog-typed subjects, not the full ontology chapter. Proposed
  fixes are insert-only (`action=ADD`); `REMOVE`/`REPLACE` are dropped.
  Each new subject closure goes through the same per-subject regression
  check as a critic fix; an insert that worsens the unit is rolled back.
  The loop stops when the inventory is empty. Telemetry: run manifest
  `completion` block and `retrieval_metrics` (`facts_completion_calls`,
  `facts_completion_triples_inserted`,
  `facts_completion_measurements_recovered`).

### Changed

- **Near-miss predicate repair requires token containment.**
  `repair_property_aliases` no longer rewrites a catalog-namespace
  predicate absent from the unit snapshot to the single `SequenceMatcher`
  candidate above `FACTS_PROPERTY_ALIAS_MIN_RATIO`. The repair now (1)
  never rewrites a predicate declared anywhere in the catalog
  (`catalog_terms()`), (2) rewrites only when exactly one candidate
  qualifies by token containment or equality (case and separator folded),
  and (3) uses the ratio only to break ties. Default
  `FACTS_PROPERTY_ALIAS_MIN_RATIO` is `0.95` (tie-break floor). Other
  cases remain mandatory findings with suggestions.

- **Facts prompts put constant chapters first.** Render and critic
  templates are now `preamble → conformance requirements → ontology → TASK
  → phase instruction → user instruction → text …`. Shared ontology
  chapters therefore share a byte-identical prefix through the end of that
  chapter. Placeholder names are unchanged; `prefix_cache_hit_rate` reports
  the effect. Cached prompts are invalidated by the reorder.

- **`chars_received` counts characters for every provider.** Previously
  `len(result.content)`, which counted content blocks for list-valued
  providers. It now measures the normalised text.

- **Batch run manifests populate the selection census.** The batch state
  merge-back now carries `content_units`, `unit_failures`,
  `facts_repairs_applied`, and aggregation clusters from workflow state, so
  `selection.labeled_units`, `unlabeled_units`, and
  `section_label_histogram` are no longer built from the pre-run empty
  list. `selection.summary_max_sentences` is emitted only when
  summarization ran.

- **Facts-mode companions are now the defaults.** `FACTS_CONTEXT_FROM_UNITS`
  and `FACTS_NUMERIC_IDENTIFIER_GUARD` default to `true`. In
  `RENDER_MODE=facts` there is no ontology stage; without the first,
  aggregator guards and the SHACL gate ran against an empty vocabulary
  (`validated_without_ontology_context`). The second keeps identifier digit
  groups out of the numeric-coverage inventory. Set either to `false` to
  restore previous behaviour. `LLM_JSON_MODE` remains off.

### Fixed

- **A request the provider refuses now stops the run instead of emptying it.**
  A rejected request — an unsupported parameter value, a model the account
  cannot reach, a missing or wrong key — is a property of the deployment:
  identical for every content unit and every retry. The unit loops isolated it
  the way they isolate a bad render, so every unit failed the same way, the
  document serialized an empty graph, and the run wrote a manifest and a
  validation report next to no facts and exited 0 — output a downstream
  aggregator cannot tell from a clean run. Such a rejection is now re-typed at
  the single call funnel as `LLMConfigurationError`, propagates through the
  unit loops and the parallel fan-out (once the gather has drained, so no
  sibling is orphaned), aborts the batch, and exits `78` (`EX_CONFIG`) with a
  message naming the provider, the model and the rejected parameter — no
  dumps, so the absence of output is the signal. It is never retried and never
  spends the timeout re-issue. Deliberately narrow: throttling (`429`) and a
  `400` that names the *input* rather than a parameter — an over-long chunk —
  stay per-unit faults. Counted as `llm/calls_rejected` alongside
  `llm/calls_failed`.

- **A document whose every unit failed is reported as a failed file.** The map
  stages already computed `FAILED` for one that produced nothing, and
  `merge_facts` preserved it, but the batch path never read the status — so a
  run that extracted nothing still exited 0. It now lands in the failed-file
  list, which the CLI already turns into a non-zero exit. The dumps still
  happen: an empty graph beside its manifest is the diagnostic.

- **The gpt-5 temperature pin no longer catches later families.** The
  series is provider-pinned to temperature 1.0, and the override matched any
  name starting `gpt-5` — which swallowed `gpt-5.4*` as well, forcing 1.0 on
  models that accept a temperature. The match is now anchored to the series
  itself (`gpt-5`, `gpt-5-mini`, `gpt-5-nano`). The override mutates the
  config in place, so it also reached the cache key and the run manifest: an
  affected run recorded the substituted temperature, not the one it asked
  for.


## [0.6.3] - unreleased

### Added

- **Provider rate-limit safeguards.** `LLM_REQUESTS_PER_SECOND`
  (per-process token-bucket pacing, passed as a langchain
  `InMemoryRateLimiter` to every provider) and `LLM_MAX_RETRIES` (provider
  SDK transport-retry budget; previously each SDK's default).
  `LLM_MAX_INFLIGHT` caps concurrency, not rate. Throttles are counted
  (`llm/rate_limited` in `budget.counters`, beside `llm/timeouts`). Both
  knobs are recorded in the run manifest `llm` block. The pipeline does not
  retry transport failures itself; the SDK backoff honours `Retry-After`.

- **Shapes-driven prompt contract** (`FACTS_SHAPES_PROMPT_CONTRACT`, default
  `auto`). Loaded SHACL shapes are rendered into a
  `# CONFORMANCE REQUIREMENTS` chapter for the facts renderer and critic.
  Messages come from `sh:message` where present, otherwise a synthesized
  structural line; message-less SPARQL constraints are omitted with a
  warning. Capped by `FACTS_SHAPES_PROMPT_MAX_LINES`. With no shapes
  loaded, or `off`, the prompt is unchanged. Terms the shapes require join
  `ValidationPolicy.contract_exempt_terms` (full-catalog, every mode).

  Chapter selection is by context join, not truncation: a small catalog is
  rendered whole (memoized per tenancy); above the line cap, only shapes
  whose terms intersect the unit snapshot IRIs are kept. Modes: `full`,
  `context`, `auto` (size-switched, default), `off`. The run manifest
  records `validation_config.shapes_prompt_selection`. Shapes are not
  vector-indexed.

  New module `prompt/shapes_contract.py`; wiring in
  `tool/shapes_catalog.py`, `toolbox.py`, `onto/unit_states.py`,
  `stategraph/atomic.py`, `agent/render_facts.py`,
  `agent/criticise_facts.py`.

- **Quantitative-completeness rule** in the facts prompt (rule 3a) and an
  actionable completeness guideline in the critic prompt. Rule 3a requires
  every quantitative statement (rule-8 fallback when no term covers it)
  while keeping rule 4's anti-junk guard. On a NUMERIC COVERAGE finding the
  critic proposes an ADD fix with the exact `text_fragment` and verbatim
  value+unit, or classifies the mention as typography — it must not invent
  a subject for a bare token.

- **`FACTS_NUMERIC_COVERAGE_LIMIT`** (default 30, previously hard-coded) and
  **`FACTS_NUMERIC_COVERAGE_MANDATORY`** (default off): cap on mentions a
  NUMERIC_COVERAGE finding lists, and whether those findings block unit
  acceptance.

- **`--facts-user-instruction` on `ontocast process`.** Exposes the
  per-request deployment-guidance slot already available on the HTTP API.
  The manifest records the instruction's length only.

- **Self-describing run manifest.** New `validation_config` block
  (`context_from_units`, `json_mode`, `shapes_prompt_contract`,
  `shapes_triples`, `shacl_inference`, `numeric_coverage_mandatory`,
  `facts_user_instruction_chars`) and section-label census on `selection`
  (`labeled_units`, `unlabeled_units`, `section_label_histogram`).

- **`--ontology-dir` and `--shapes-dir` on `ontocast serve` and `ontocast
  process`**, overriding `ONTOCAST_ONTOLOGY_DIRECTORY` and `FACTS_SHAPES_DIR`
  for one run. Three states: omitted leaves the environment in force; a
  path overrides it; the empty string clears it (no seed ontologies). Typed
  as strings: `Path("")` is `.`. `--ontology-dir` is the input catalog;
  `--ontology-output-dir` is where results are written.

  Touches: `cli/server.py::_shared_runtime_options`, `_prepare_path_config`,
  `_resolve_seed_directory`, `_bootstrap_tools`.
  Test: `test/test_cli_seed_directories.py`.

- **A missing seed directory stops `ontocast process` at startup.** The
  error names the flag or environment variable and the resolved absolute
  path. An empty directory still only warns. `ontocast serve` warns in both
  cases.

- **`FACTS_CRITIC_PASSES` (default `1`) and `ONTOLOGY_CRITIC_PASSES`
  (default `0`).** Review-and-patch passes per unit. A facts unit costs two
  provider calls at the defaults (one extraction, one review), independent
  of `MAX_VISITS`. The ontology default is opt-in. A pass ends the loop
  early when it changed nothing or was rolled back.

- **Screening limits on critic deletions.**
  `FACTS_CRITIC_MAX_DELETE_SHARE` / `ONTOLOGY_CRITIC_MAX_DELETE_SHARE` with
  a `*_MIN_DELETES` floor. A `REMOVE` may not empty a subject; a `REPLACE`
  may not write about a different subject than it deletes unless
  `FACTS_CRITIC_ALLOW_SUBJECT_RENAME`; a blank node is deleted whole or not
  at all. A pass is rolled back if it deleted without writing, shrank the
  unit without resolving anything, or created new mandatory findings.

- **No-op fix detection.** A fix whose delete set equals its insert set is
  dropped and reported as `facts_critic_fixes_noop`.

- **Patch telemetry.** `LoopAttempt(kind="critic_patch")` with applied/no-op
  and inserted/deleted counts, rollback and delete-cap flags;
  `RunManifestCritic` gains `patch_passes`, `fixes_applied`, `fixes_noop`,
  `patches_rolled_back`, `triples_deleted`, `triples_inserted`,
  `incumbent_accepted`; retrieval metrics `facts_critic_fixes_noop`,
  `facts_critic_patches_rolled_back`, and `ontology_critic_*` equivalents.

- **`ONTOLOGY_ACCEPT_BLOCKING_FINDING_KINDS`** — which deterministic
  ontology findings block acceptance.

- **`ONTOLOGY_CONTEXT_REQUIRED` (default `false`) stops a facts run whose
  ontology context resolves to zero triples.** An empty facts context
  yields generic vocabulary, `UNKNOWN_TERM` exemption, and a vacuous SHACL
  pass. Set `true` for deployments that extract against a curated catalog.
  Ontology units are not gated: `render_ontology_fresh` is the path when
  there is no seed. Applies to facts units only, even when set.

  Touches: `stategraph/context_resolver.py`;
  `onto/retrieval_capabilities.py::EmptyOntologyContextError`;
  `config/settings.py::ServerConfig`.
  Test: `test/test_stategraph_context_resolver.py`.

- **`shacl_focus_nodes` and `shacl_vacuous` in the conformance summary.**
  `conforms` is `null` rather than `true` when the focus set is empty.
  Target matching follows `rdfs:subClassOf`, as `sh:targetClass` does.

  Touches: `tool/facts_validation/gate.py::count_shacl_focus_nodes`,
  `summarize_conformance`; `stategraph/facts_gate.py`.
  Test: `test/facts/test_grounding_guards.py`.

- **`DOMAIN_ADHERENCE`**, a mandatory per-unit finding when a render barely
  used the catalog (`FACTS_DOMAIN_ADHERENCE_MIN_SHARE`, default `0.15`; `0`
  disables). Measured as a share of distinct schema terms (predicates and
  `rdf:type` objects), excluding minted instances and
  RDF/RDFS/OWL/XSD/SKOS/DC/PROV plumbing. Generic content vocabularies stay
  in the denominator. Silent when the unit has no catalog.

  Touches: `tool/facts_validation/unit_findings.py::domain_vocabulary_share`,
  `_domain_adherence_findings`; `tool/facts_validation/terms.py`;
  `config/settings.py::FactsValidationConfig`.

- **Startup check that the vector index and ontology catalog agree.** An
  empty catalog beside a populated index fails initialization, naming what
  the index still holds. Extra indexed IRIs beside a populated catalog warn
  (ordinary staleness). Fires before the first LLM call.

  Touches: `toolbox.py::_check_catalog_index_agreement`.
  Test: `test/test_toolbox_sync.py`.

- **Critic telemetry:** `fix_action_severity_histogram` (fixes keyed
  `ACTION:severity`) and `accept_reason_histogram` on the critic manifest
  block, plus `facts_critic_fixes_applied` / `facts_critic_fixes_residual`.

  Touches: `onto/model.py::LoopAttempt`; `onto/run_manifest.py`;
  `agent/criticise_facts.py`; `stategraph/node_factories.py`; `onto/enum.py`.

- `FACTS_CONTEXT_FROM_UNITS` (default off) seeds the merge/validate ontology
  context from the snapshots the facts units resolved. With
  `RENDER_MODE=facts` no ontology stage runs, so without this both the
  aggregator and the gate received an empty vocabulary. Snapshots are
  deduplicated by contributing catalog IRI. `validated_without_ontology_context`
  and `ontology_snapshot_triples` report which side a run is on.

  Touches: `stategraph/node_factories.py::_union_unit_ontology_context`,
  `make_render_facts_node`; `config/settings.py::FactsValidationConfig`.
  Test: `test/facts/test_context_computed_once.py`.

- `FACTS_SUSPECT_MULTI_VALUE_REQUIRE_CROSS_UNIT` (default off) requires
  merge-created evidence before the IRI branch of `SUSPECT_MULTI_VALUE`
  reports an error. `AggregationResult` gains `cross_unit_object_pairs`,
  canonicalized through the same mapping as `merged_clusters`; the un-merge
  loop recomputes it per pass. Numeric and string branches are unchanged.

  Touches: `tool/agg/aggregate.py::build_cross_unit_object_pairs`;
  `onto/state.py`; `stategraph/node_factories.py`; `stategraph/facts_gate.py`;
  `tool/facts_validation/gate.py::validate_aggregated_facts`.
  Test: `test/facts/test_gate.py`.

- `FACTS_NUMERIC_IDENTIFIER_GUARD` (default off) keeps identifier digit
  groups out of the numeric-coverage inventory. A group against `/` or `:`
  is treated as part of an identifier, not a magnitude. A digit group that
  is its own token is not covered. Values with units and hyphenated ranges
  are unaffected.

  Touches: `util/numeric_inventory.py::_is_identifier_fragment`,
  `extract_numeric_tokens`, `missing_numeric_mentions`;
  `tool/facts_validation/terms.py::ValidationPolicy`; `tool/atomic.py`.
  Test: `test/test_numeric_inventory.py`.

- `CHUNK_MIN_UNIT_CHARS` (default `0`, disabled) drops content units below a
  character floor before the extraction fan-out. Distinct from
  `CHUNK_MIN_SIZE` (chunker merge target). Each drop is logged.

  Touches: `agent/chunk_text.py`; `config/settings.py::ChunkConfig`.
  Test: `test/chunking/test_section_pipeline.py`.

- `LLM_JSON_MODE` (default off) constrains OpenAI decoding to syntactically
  valid JSON via `response_format: json_object`. Off by default because
  OpenAI rejects the request unless the prompt mentions JSON. Strict
  `json_schema` mode is not offered: it requires closed schemas and the
  graph fields are open.

  Touches: `tool/llm.py::LLMTool.setup`; `config/settings.py::LLMConfig`.
  Test: `test/test_llm_json_mode.py`.

- `ontocast process --keep-provenance` retains chunk-level provenance in the
  facts dump. Stripping remains the default; the flag mirrors the HTTP
  `strip_provenance` parameter.

  Touches: `api/process_helpers.py::dump_facts_ttl`, `process_files_input`;
  `cli/server.py`.
  Test: `test/test_cli_server.py`.

### Changed

- **Startup catalog check follows render mode.** `ontocast process` requires
  a populated catalog only under `RENDER_MODE=facts`. Other modes log that
  they will build ontologies from the corpus and start.

  Touches: `cli/server.py::_bootstrap_tools` (`batch`);
  `toolbox.py::_check_catalog_ready`.
  Test: `test/test_toolbox_ontology_seed.py`.

- **Retrieval-integrity checks no longer consult
  `ONTOLOGY_CONTEXT_REQUIRED`.** A populated vector index beside an empty
  catalog, or an empty index over a populated catalog, is a retrieval
  disagreement, not a preference. Neither fires during bootstrap when both
  are empty.

  Touches: `toolbox.py::_check_catalog_index_agreement`, `_check_catalog_ready`.
  Test: `test/test_toolbox_sync.py`.

- **Seed TTLs are replayed into a new tenancy scope**, where the partition
  serves no terms for them; `FACTS_SHAPES_DIR` is seeded on the same switch.
  An ontology the scope already defines is never overwritten by the on-disk
  copy.

  Touches: `toolbox.py::_update_tenancy_with_vector_mode_locked`.
  Docs: `docs/user_guide/tenancy.md`, `docs/architecture/ontology_catalog.md`.

- **The critic cites statement ids and applies its own fixes.** Every graph
  chapter in a critic prompt carries a number per statement — inline in
  Turtle, in a `TRIPLE INDEX` table under JSON-LD — and
  `TripleFix.triple_ids` names them. `incorrect_value` remains as a fallback
  for a fix that cites no id. `correct_value` stays free-hand. On the
  ontology side, ids are scoped to the unit's own delta: retrieved catalog
  statements carry no id, so a delete that would propagate onto a shared
  terminal is not expressible.

  Touches: `onto/triple_index.py`; `prompt/graph_index.py`;
  `prompt/graph_format.py`; `onto/model.py::TripleFix`;
  `tool/facts_validation/critic_patch.py`; both critic agents.
  Tests: `test/test_triple_index.py`, `test/facts/test_critic_patch.py`.

- **The separate finding-driven repair render is gone; the critic pass is
  the repair.** Fixes are compiled into a validated `GraphUpdate` and
  applied with no LLM call. The invariant is unchanged: every mutation is
  still a compiled, validated `GraphUpdate`.

  Removed: `stategraph/atomic.py::_run_finding_driven_repair`,
  `tool/facts_validation/critic_findings.py`.

- **Fix payloads are parsed format-tolerantly rather than by dispatch.**
  `_parse_fragment` maps `@value`/`@language`/`@type` onto the equivalent
  Turtle literal forms, unwraps `@id`, and brackets bare absolute IRIs. A
  payload that is not a statement stays unparseable.

- **The facts and ontology unit loops are one implementation.**
  `stategraph/atomic.py::run_unit_loop` serves both, with a `LoopPhase`
  adapter for the differences. `facts_loop` and `ontology_loop` remain as
  wrappers.

- **`MAX_VISITS` retries a failed render and nothing else.** A successful
  render is never repeated. Improving it is the critic's job. `MAX_VISITS`
  bounds renders only.

- **The ontology critic gates on deterministic findings, not on its own
  score.** The blocking set is the destructive-or-lossy subset of findings;
  it does not include `missing_label`. The retired gate's verdict is
  recorded as `incumbent_accepted`.

- **An accepting critic keeps its fixes.** Accepting means no defect worth
  another render, not that the critique was empty. Fixes that cannot be
  applied are counted as residual.

  Touches: `agent/criticise_facts.py`; `stategraph/atomic.py`.
  Test: `test/facts/test_suggestions_lifecycle.py`.

- **`MAX_VISITS` no longer gates the facts critic.** The facts critic runs
  whenever `FACTS_CRITIC_PASSES > 0` (and is skipped only at `0`). The
  ontology loop is unchanged.

  Touches: `stategraph/atomic.py::_skip_critic_after_final_render`;
  `config/settings.py::ServerConfig`; `docs/user_guide/{validation,workflow,
  configuration,performance,observability}.md`.
  Test: `test/test_max_visits_critic_propagation.py`.

- **The facts prompt states vocabulary precedence.** Catalog terms take
  precedence over generic vocabularies, which are a last resort. The block
  no longer points "above" at an ONTOLOGY section emitted below it, and
  says not to mint entities for bare numbers or citation markers.

  Touches: `prompt/facts_guidelines.py`.
  Test: `test/test_prefix_namespace_hygiene.py`.

- Test markers now match `pyproject.toml`. Every collected test outside
  `test/manual/` carries a kind marker, applied at module level, with
  `slow` assigned from isolated runtime. Markers are not mutually
  exclusive: a service test is both `integration` and `slow`.

- `ontocast/prompt/` has smoke coverage. Templates are discovered from
  `str.format` call sites and `PromptTemplate` construction, so literal
  blocks substituted into a slot (raw JSON braces) are not treated as
  templates.

  Test: `test/test_prompt_templates.py`.

### Removed

- **`render_facts_update` and the facts update-render mode.** `render_facts`
  dispatched on the unit graph being non-empty, and nothing populates that
  graph except a successful render, after which the render loop has already
  finished. `render_ontology_update` is unchanged: it dispatches on the
  retrieved snapshot, a different field from the one it writes.

  Also removed: `_findings_instruction` and
  `GraphFormatProfile.format_facts_chapter`, whose only caller it was. The
  critic uses `format_facts_chapter_indexed`.

### Deprecated

- `FACTS_LLM_REPAIR_VISITS` — alias for `FACTS_CRITIC_PASSES`, honoured for
  one release.
- `MAX_CRITIC_VISITS_PER_NODE` — inert. It capped critic retries within one
  render attempt; the loop no longer retries a critic inside a pass. Still
  recorded in the run manifest so an existing setting stays auditable.

### Fixed

- **Unit status now reflects the critic's patch, not its pre-patch
  verdict.** `_apply_critic_patch` re-runs `material_defects` on the
  post-patch findings and outstanding fixes (including rollback, which
  recomputes the pre-patch verdict) and sets the unit's status accordingly.
  `LoopPhase` carries the phase's acceptance policy.

- Stale docstring on `material_defects` claiming the critic never runs at
  `MAX_VISITS=1` — the critic budget (`FACTS_CRITIC_PASSES`) is independent
  of the render-failure bound.

- **A facts-only batch run against an empty catalog now fails at startup.**
  `ontocast process` under `RENDER_MODE=facts` refuses to start when the
  catalog resolves to zero ontologies, and any run refuses when the vector
  index is still empty after materialization. `ontocast serve` warns about
  an empty catalog instead of failing: starting empty and filling through
  `POST /ontologies` is supported.

  Touches: `toolbox.py::_check_catalog_ready`, `_catalog_sources_description`,
  `initialize(require_populated_catalog=...)`; `cli/server.py::_bootstrap_tools`.
  Test: `test/test_toolbox_sync.py`.

- **`EmptyOntologyContextError` was caught as a per-unit failure.**
  `OntologyContextConfigError` now propagates out of both the unit loop and
  the fan-out, re-raised after the gather has drained.

  Touches: `stategraph/atomic.py::run_unit_loop`;
  `stategraph/node_factories.py::_gather_units`.
  Tests: `test/test_atomic_loop_bounds.py`, `test/test_unit_fanout_failures.py`.

- **A seed ontology can repair a partition that listed its IRI but served
  no terms.** Sync now tests whether the served ontology defines terms,
  rather than only its own `owl:Ontology` header (which a catalog read
  synthesizes regardless). An ontology that does define terms is never
  overwritten.

  Touches: `toolbox.py::_seed_ontologies_missing_from`, `_defines_terms`,
  `_synchronize_ontologies`.
  Test: `test/test_toolbox_ontology_seed.py`.

- **A threshold rejection reported as "no candidate atoms matched".**
  `atoms_after_dedupe` is counted after the score gate, so the branch
  naming the thresholds was unreachable. `candidate_hits` and
  `threshold_rejected`, both counted before the gate, are now recorded and
  used instead. When retrieval short-circuits on zero atoms,
  `catalog_context_triples` is absent rather than `0`; the catalog is now
  asked directly when the metrics cannot answer.

  Touches: `stategraph/context_resolver.py::_diagnose_empty_snapshot`;
  `tool/vector_store/patch_retriever.py::_filter_and_merge_patch_hits`.
  Test: `test/test_stategraph_context_resolver.py`.

- **An ontology that indexed to zero atoms looked like a successful
  reindex.** `reindex_ontology` deletes before it indexes and discarded the
  count. The count is now logged per ontology and warned on when it is
  zero.

  Touches: `toolbox.py::_materialize_ontology`.

- **The empty-ontology-context diagnostic named the wrong subsystem.** It
  inspected the vector index and atom counts and never consulted catalog
  state. Catalog-side causes are now checked first and named, including
  vector-index and triple-store disagreement.

  Touches: `stategraph/context_resolver.py::_diagnose_empty_snapshot`.

- A section list whose labels are all unrecognised is now rejected instead
  of silently disabling section handling. Unknown tokens were dropped with
  a warning and the empty result replaced the resolved schema's
  `default_exclude`. A partly recognised list still warns and continues.
  The CLI reports it as a usage error rather than a traceback.

  Touches: `api/parse.py::_resolve_section_tokens`,
  `_normalise_section_tokens`; `cli/inspect_sections.py`.
  Test: `test/chunking/test_section_pipeline.py`.

- Serializing a graph that carries an unusable term raises a diagnostic
  naming the term, its rdflib type, its triple position, and its graph,
  instead of a bare `AssertionError`. The reachable case is a term that
  converts to something which is not a term: `to_ox` maps
  `urn:x-rdflib:default` to a `DefaultGraph`. The asserts also admitted
  `ox.Triple`, which `to_ox` cannot return.

  Touches: `tool/triple_manager/in_memory.py::_to_ox_term`,
  `_rdflib_graph_to_quads`.
  Test: `test/test_in_memory_manager.py`.

- `FusekiTripleStoreManager._initialize_datasets` falls back to the default
  facts dataset rather than passing `None` to Fuseki as a dataset name.

  Touches: `tool/triple_manager/fuseki.py`.

### Documentation

- **Facts-loop diagrams regenerated** (`docs/assets/facts_loop*`). They now
  show the repair lane: the gate on the repair budget, the LLM-free tier,
  and both critic outcomes landing in the same place. The ontology-loop
  diagrams are unchanged.

  Touches: `cli/plot_graph.py::_facts_loop_core_edges`,
  `_facts_loop_evidence_edges`, `facts_loop_flow`;
  `docs/user_guide/workflow.md`.

- `demo/README.md` named sample files that do not exist (`sample.pdf`,
  `sample.txt`, `sample.ttl`). It now names the two PDFs the directory
  actually holds, states that commands run from the repository root, and
  lists the recorded response and figure.

- Every public callable reported by griffe now carries parameter and return
  annotations, and `uv run mkdocs build` is warning-free.

  Touches: `runtime.py`, `toolbox.py`, `tool/llm.py`, `tool/converter.py`,
  `tool/validate.py`, `tool/onto.py`, `tool/ontology_manager.py`,
  `tool/chunk/chunker.py`, `tool/triple_manager/{core,fuseki}.py`,
  `onto/{content_unit,ontology,rdfgraph,state}.py`.

- `docs/user_guide/configuration.md` documents `LLM_JSON_MODE`,
  `CHUNK_MIN_UNIT_CHARS` and the fail-closed section-label rule;
  `docs/user_guide/validation.md` documents the three facts-validation arms
  and why each defaults to off; `docs/user_guide/concepts.md` documents
  `--keep-provenance`.

## [0.6.2] - 2026-08-29

### Breaking

- The ontology update wire is flat. `GraphUpdateRenderReport` carries
  `insert_graph` and `delete_graph` as sibling graph fields, replacing
  `graph_update.triple_operations[]`. `to_graph_update()` compiles them
  delete-then-insert into the unchanged internal model, so `apply()`, the SPARQL
  compiler and the LangChain tool are unaffected. Interleaving inserts and
  deletes within a single render is no longer expressible. Cached ontology-render
  responses are invalidated.

  Touches: `onto/model.py::GraphUpdateRenderReport`, `to_graph_update`;
  `prompt/graph_format.py`; `prompt/llm_json_schema.py`.
  Test: `test/test_graph_update_wire_regression.py`.

- `tool/facts_invariants.py` is removed and split into the
  `tool/facts_validation/` package: `terms` (catalog inventory, namespace
  closure, `ValidationPolicy`, alias candidates), `literal_repair` (parse-time
  repairs), `unit_findings`, `shacl` (execution, autofix, catalog lint), `gate`
  (document-level validation) and `acceptance`. The package `__init__` is the
  public surface; the previous module name no longer resolves.

- The `data/` directory is removed, together with the importable top-level
  `data` package it contained. It was already excluded from the source
  distribution while `test/` was included, so tests reading it could not run
  from a published sdist. The two TTL fixtures required by tests are now in
  `test/data/ontologies/`. `run/fetch_schema_samples.py` resolves its two
  local-source entries through `ONTOCAST_SCHEMA_SAMPLE_DIR` and skips them when
  that variable is unset; the corresponding extracts in
  `test/data/schema_corpus.json` are unchanged.

  Touches: `pyproject.toml`; `run/fetch_schema_samples.py`; `demo/README.md`;
  `cli/inspect_sections.py`.
  Test: `test/test_repo_isolation.py`.

- SHACL shapes are stored in the triple store, in a third tenancy partition
  `{tenant}--{project}--shapes` beside facts and ontologies
  (`FUSEKI_SHAPES_DATASET`). `FACTS_SHAPES_DIR` keeps its name but changes
  meaning: it is now a read-only **seed** directory materialized into that
  partition at startup -- the contract `ONTOCAST_ONTOLOGY_DIRECTORY` already
  had -- and the validation gate reads the partition, not the directory. A
  containerised worker therefore needs no shapes directory, and a per-tenant
  catalog carries its own shapes.

  Shapes get a partition of their own because catalog discovery claims every
  named graph holding an `owl:Ontology` subject, and a shapes document declares
  one; co-located, each would register as a catalog ontology, be vector-indexed,
  and be offered to the renderer as schema.

  `collect_shacl_shapes(ontology_graph, shapes_dir)` now takes
  `(ontology_graph, stored_shapes: RDFGraph | None)` and performs no disk I/O,
  which also removes the per-document re-glob and re-parse. The inline
  `sh:NodeShape` source is unchanged.

  Touches: `onto/tenancy.py`; `onto/constants.py`; `config/settings.py`;
  `tool/shapes_catalog.py` (new); `tool/facts_validation/shacl.py`;
  `stategraph/facts_gate.py`; `toolbox.py`; `api/shapes.py` (new); `api/app.py`.
  Test: `test/facts/test_shapes_catalog.py`.

- The triple-store partition selector is a `StoreKind`
  (`"facts" | "ontologies" | "shapes"`), replacing the two-valued
  `use_ontologies_dataset: bool` on `aselect`, `aconstruct`,
  `drop_named_graph`, `drop_all_ontology_graphs_for_iri`, `serialize_graph` and
  `serialize`. `aserialize(ontology)` previously hard-coded the ontologies
  dataset and silently overwrote a caller's `graph_uri`; it now honours a
  `store=` override. Fuseki's `serialize_graph` took a pre-built `dataset_url`
  while the in-memory backend took a boolean -- both now take `store`.
  The LangChain `ontocast_sparql_select` / `ontocast_sparql_construct` tools
  expose `store` in place of `use_ontologies_dataset`.

  Touches: `tool/triple_manager/{core,fuseki,in_memory,mock}.py`;
  `integrations/{langchain,schemas}.py`.

### Added

- `/shapes` routes -- `GET` (list stored documents), `POST` (upload Turtle),
  `DELETE /{graph_uri}` -- mirroring `/ontologies`, tenancy-scoped the same way.
  A document declaring `<iri> a owl:Ontology` is stored under that IRI so
  re-uploading replaces it; a headerless one is named after its seed path or
  uploaded filename, stable across edits. The seed directory is never written
  to, and `DELETE` leaves it untouched.

  Touches: `api/shapes.py`, `api/schemas.py`, `api/app.py`, `toolbox.py`.
  Test: `test/test_api_tenancy_resolution.py`.

- `POST /flush?include_shapes=true`. Flush **retains** the shapes partition by
  default: facts and ontologies come back from a rerun, but dropping shapes
  disarms the SHACL gate without an error -- later runs report
  `shacl_evaluated: null` instead of failing. `TripleStoreManager.clean()` and
  `clean_tenancy()` take the matching `include_shapes` flag.

  Touches: `api/app.py`; `toolbox.py::clean_tenancy_data`;
  `tool/triple_manager/{core,fuseki,in_memory,mock}.py`.

- Reduce-time ontology update semantics for retrieved snapshots. Under
  `ONTOLOGY_CONTEXT_MODE=selected_vector_search_ontology` a per-unit snapshot is
  a subset of the catalog while the delta applies to the full terminal, so
  judgements requiring knowledge of the whole catalog are made at reduce:

  - Minted-duplicate reconciliation. Each newly minted term is checked against
    the full terminals by exact surface form (`rdfs:label`, `skos:prefLabel`,
    `skos:notation`), unique resolution and compatible role.
    `ONTOLOGY_RECONCILE_MINTED_TERMS` selects `off`, `detect` (default; records
    matches without altering the delta) or `rewrite` (substitutes the catalog
    IRI in subject and object position).
  - Redeclare-only deletes. A merged delete whose subject the merged inserts do
    not redeclare is dropped, since it was judged against a partial view and
    would otherwise propagate onto shared catalog terminals. Applies to vector
    mode only; single-ontology modes are unaffected.
  - Partial-context prompts. Ensemble-assembled contexts carry an explicit
    notice in the render intro and the critic criteria, and intro selection keys
    on `OntologySnapshot.assembly_mode` rather than on the count of writable
    IRIs.
  - Divergence counters for delete triples absent from the terminal at apply
    time.

  Touches: `tool/ontology_validation/reconcile.py`;
  `stategraph/helpers.py::enforce_redeclared_deletes`; `onto/ontology_apply.py`;
  `prompt/render_ontology.py`; `prompt/criticise_ontology.py`.
  Metrics: `minted_duplicates`, `minted_duplicate_pairs`,
  `minted_duplicates_rewritten`, `deletes_dropped_unredeclared`,
  `apply_deletes_no_match`.
  Tests: `test/test_ontology_reconcile.py`,
  `test/test_ontology_prompt_context.py`.

- Fresh-path reconciliation. Ontology artifacts minted by different units under
  the same IRI are union-merged into a single lineage root rather than resolved
  last-writer-wins. Term overlap across distinct fresh IRIs is counted, not
  merged.

  Touches: `onto/ontology.py::Ontology.union_fresh`;
  `stategraph/node_factories.py`.
  Metrics: `fresh_ontologies_merged`, `fresh_minted_duplicates`.

- Deterministic per-unit validation of ontology deltas, in shadow mode. Findings
  are validated against the unit's net insert/delete delta relative to the
  prompt snapshot, not against the working graph, which is snapshot plus delta
  and would attribute pre-existing catalog defects to the unit. Checks: terms
  minted under namespaces no context ontology declares, degenerate
  `owl:Restriction` stubs, new terms without a label, subclass cycles across
  snapshot and delta, class/property role confusion, functional versus
  min-cardinality contradictions, deletes of catalog content the unit does not
  redeclare, and advisory label collisions. `UNKNOWN_TERM` is deliberately not
  ported, since minting terms is the ontology renderer's function; connectivity
  remains with the document-level structural check. Findings are collected
  before each critic call and at loop exit, so a residual exists at the
  `MAX_VISITS=1` default, and are injected into the critic prompt as mandatory
  items. **Acceptance is unchanged**; the findings are recorded for a later
  recalibration.

  Touches: `tool/ontology_validation/unit_findings.py`;
  `onto/model.py::OntologyUnitFindingKind`.
  Metrics: `ontology_findings_residual`, `ontology_mandatory_residual`.
  Test: `test/test_ontology_unit_findings.py`.

- Ontology critic telemetry. Each critic call appends a `LoopAttempt` recording
  score, severity mix, findings counts, delta size and the number of proposed
  fixes targeting snapshot content the delta does not touch. The ontology critic
  continues to accept on `critique.success or critique.score > 90`. That
  threshold is unmeasured: the ontology critic does not run under
  `render_mode: facts`, so no recorded data covers it.

  Touches: `onto/model.py::LoopAttempt` (renamed from `FactsLoopAttempt`);
  `onto/run_manifest.py::summarize_loop` (renamed from
  `summarize_facts_loop`).
  Metrics: `ontology_critic_calls`, `ontology_critic_accepted`.
  Manifest: `ontology_critic`.
  Test: `test/test_ontology_loop_telemetry.py`.

- `RunManifestCritic` on the run manifest, under `critic` and
  `ontology_critic`: call count, accepted count, score minimum, median and
  maximum, a decade-bucketed score histogram and a proposed-fix severity
  histogram. `summarize_loop` returns an all-zero record when no critic call
  ran, so `accepted: 0` at `MAX_VISITS=1` indicates that nothing was judged.

  Touches: `onto/run_manifest.py`; `api/process_helpers.py`.
  Metrics: `facts_critic_calls`, `facts_critic_accepted`.
  Test: `test/test_run_manifest_critic.py`.

- Effective configuration and output shape on the run manifest: `loops`
  (`max_visits`, `max_critic_visits`, `llm_repair_visits`), `selection`
  (`target_sections`, `exclude_sections`, `summarize_sections`,
  `summary_max_sentences`, `bibliography_mode`) and `graph_metrics`
  (connectivity of the serialized facts graph). A run whose per-unit budget
  differs from the requested one is now detectable from its own dump.

  Touches: `onto/run_manifest.py::RunManifestLoops`, `RunManifestSelection`;
  `util/graph_metrics.py`.
  Test: `test/test_max_visits_critic_propagation.py`.

- `BudgetTracker.prefix_cache_hit_rate` and
  `BudgetTracker.reasoning_share_of_output`, reported on the `/process` response
  and the run manifest as computed fields. Both denominators span billed and
  replayed tokens, because `cache_read_input_tokens` and `reasoning_tokens`
  accumulate on both while `input_tokens` counts billed calls only. Both are
  `null` when the provider reports no token usage.

  Touches: `onto/state.py::BudgetTracker`.
  Test: `test/test_budget_tracker.py`.

- Natural-key identity evidence for entity aggregation
  (`AGG_NATURAL_KEY_MERGE`, default enabled). Instances asserting an identical
  short string value (at most 64 normalized characters, shared by at most 8
  entities) on a single-valued identifier-like predicate — declared max-1, or
  observed single-valued on every subject — become merge candidates and satisfy
  the lexical bar even when labels and embeddings disagree. All distinctness
  guards continue to apply. Key-supported clusters are reported, and the
  validation gate downgrades string multi-value findings on those subjects to
  warnings.

  Touches: `tool/agg/aggregate.py`; `tool/agg/signatures.py`;
  `AggregationResult.key_supported_clusters`;
  `AgentState.aggregation_key_clusters`; `stategraph/facts_gate.py`.
  Test: `test/aggregation/test_merge_regressions.py`.

- String branch for the `SUSPECT_MULTI_VALUE` gate finding. Short name-like
  values (at most 64 normalized characters) on a predicate that is
  string-single-valued for a dominant majority of subjects yield an
  error-severity finding when any pair is not alias-compatible. The detector was
  previously numeric and IRI only, so a node carrying several irreconcilable
  names produced no finding and could not trigger the un-merge repair.

  Touches: `tool/facts_validation/gate.py`;
  `FACTS_SUSPECT_MULTI_VALUE_SEVERITY`.

- `MIXED_OBJECT_KINDS` warning finding: a predicate used with both IRI and
  literal objects across the facts graph, reported as telemetry because no
  single query shape matches both usages.

  Touches: `tool/facts_validation/gate.py`.

- Literal-variant deduplication at the validation gate
  (`FACTS_LITERAL_VARIANT_DEDUPE`, default enabled). Duplicate literals
  differing only in language tag or datatype on one subject-predicate pair are
  collapsed before validation; the language-tagged form is preferred, then the
  plain form. Reified provenance is retargeted onto the surviving triple and
  each removal is recorded as a `literal_variant_pruned` repair.

  Touches: `tool/facts_validation/gate.py`.
  Test: `test/test_facts_gate_repairs.py`.

- `LABEL_ONLY_NUMBER` mandatory finding: a node carrying the quantity fallback
  vocabulary's unit property, no numeric literal on any property, and a number
  in its label. Such nodes are not reachable by any numeric query; numbers
  inside labels previously counted as covered.

  Touches: `tool/facts_validation/unit_findings.py`;
  `util/numeric_inventory.py`.

- SHACL-versus-catalog contradiction lint (`shacl_catalog_contradictions`), run
  when the gate loads shapes. Any property the shapes require
  (`sh:minCount >= 1`) that the term validator would report as unknown is logged
  as a configuration error, since no data can satisfy both rules.

  Touches: `tool/facts_validation/shacl.py`.

- Degenerate-bound promotion at parse time: equal lower and upper bounds
  collapse to a single scalar on the configured numeric-value property. Active
  only when the quantity fallback vocabulary declares `numeric_value`,
  `lower_bound` and `upper_bound` roles.

  Touches: `tool/facts_validation/literal_repair.py`;
  `FACTS_QUANTITY_FALLBACK_VOCABULARY`.

- Isolation flags for the merge guards: `AGG_LITERAL_CONFLICT_GUARD` (default
  enabled) toggles the literal-conflict veto so its contribution to
  `facts_rejected_merges` can be separated without a code change, and
  `AGG_TYPE_GUARD_UNTYPED=strict` fails typed-versus-untyped pairs closed
  instead of the default open.

  Touches: `config/settings.py::AggregationConfig`; `tool/agg/aggregate.py`.
  Test: `test/aggregation/test_merge_guards.py`.

- `UnitOntologyState.build_delta()`, the per-unit insert/delete delta
  extraction, moved from `stategraph/helpers.py::build_ontology_delta_graph`
  onto the state so agents can call it without a layering cycle.

  Touches: `onto/unit_states.py`.

- `test/test_repo_isolation.py`, which asserts that no file under `test/`
  resolves a path outside `test/`, with an explicit allowlist for the
  declaration files the source distribution ships.

### Changed

- A rejecting facts critic now schedules a repair pass rather than a
  re-extraction, and a unit's worst-case call count no longer grows with
  `MAX_VISITS`. Previously a rejection fell through to the next render attempt,
  re-rendering the whole unit, giving a per-unit ledger of `2 * max_visits - 1`.
  Blocking critic fixes now enter the bounded rewrite-in-place pass already used
  for deterministic findings, and the outer loop retries on render failure only.
  With web grounding disabled (the default) a unit costs one render, one
  critique and the repair budget at any bound. Critic fixes join the findings for
  the first repair pass only.

  Touches: `stategraph/atomic.py`; `tool/facts_validation/critic_findings.py`;
  `FactsUnitFindingKind.CRITIC_FIX`.
  Test: `test/test_atomic_loop_bounds.py`.

- Facts render acceptance is decided by verifiable defects rather than by the
  critic's score. `criticise_facts` previously gated on `critique.success or
  critique.score > 90`, comparing a model-assigned value against a threshold the
  model is not shown, from a prompt that does not describe scoring. A model asked
  to propose improvements proposes them, so the gate rejected nearly every
  render and each rejection bought a second full extraction. It was also
  inverted: deterministic findings, which carry an explicit `mandatory` flag and
  were already injected into the critic prompt, took no part in the decision, so
  the expensive action depended on the unreliable signal and the cheap one on
  the reliable signal. Acceptance is now `material_defects()` over the
  deterministic findings and the critic's own `TripleFix` severities. `score`
  and `success` are still recorded and are no longer consulted.

  - `FACTS_ACCEPT_BLOCKING_SEVERITY` (default `critical`) sets the cut on critic
    fixes. `critical` is the only severity the critic applies selectively
    enough to discriminate on; `important` rejects almost everything. `never`
    lets deterministic findings gate alone.
  - A `REMOVE` fix never blocks, at any severity: the repair prompt states that
    a finding is not resolved by deleting the statement it refers to.
  - `FactsAcceptancePolicy.blocking_finding_kinds` silences a finding lane for
    acceptance purposes without silencing its telemetry.

  Touches: `tool/facts_validation/acceptance.py::material_defects`,
  `FactsAcceptancePolicy`; `agent/criticise_facts.py`; `stategraph/atomic.py`.
  Test: `test/test_facts_acceptance.py`.

- The repair prompt states a single contract. `improvement_instruction_template`
  previously described critic suggestions as advisory and instructed the model
  to identify additional problems beyond the critique, while
  `format_findings_for_prompt` in the same prompt required every mandatory item
  to be applied by rewriting in place. The permissive half is replaced by one
  correction-pass contract with a single bounded exception: an item contradicted
  by the source text may be skipped with a reason, and never by altering other
  statements.

  Touches: `prompt/render_facts.py`.

- Parse-retry feedback is a bounded excerpt: a window around the decode position
  for syntax errors, or a head-and-tail excerpt for schema errors, rather than
  the full response body on every retry.

  Touches: `agent/common.py::_feedback_excerpt`.

- A repeated JSON syntax-error class ends the call rather than consuming the
  remaining attempts. Schema `ValidationError`s retain the full retry budget.

  Touches: `agent/common.py::LLMJsonParseError`;
  `tool/llm.py::record_active_count`.
  Counters: `llm/parse_retry`, `llm/parse_abandoned`.
  Test: `test/test_llm_resilience.py`.

- A failed render logs one summary line per attempt, with the error context
  window at `DEBUG`. The window was previously emitted by every parse attempt,
  by the exhaustion branch and by the calling agent.

  Touches: `agent/common.py`.

- `_normalize_and_repair_graph` and `_collect_facts_findings` take the atomic
  toolbox rather than a list of unpacked scalars.

  Touches: `agent/render_facts.py`; `stategraph/atomic.py`.

- The test suite is organised into packages mirroring the source layout:
  `test/facts/`, `test/ontology/` and `test/chunking/` join the existing
  `test/aggregation/`, taking 40 modules off the top level. Module names lose
  the prefix the directory now carries (`test_facts_term_policy.py` becomes
  `test/facts/test_term_policy.py`). No test was added, removed or rewritten.

  The families are packages rather than concatenated modules by measurement:
  the facts family alone defines 12 top-level names divergently across its
  files (`_tools`, `_fake_tools`, `_ontology`, `_unit_state_with_violation`,
  `CD`, `EX` and others), so merging the sources would shadow one fixture with
  another while every test continued to pass. Where a merge is provably free of
  that hazard it was taken: the four `test_graph_atomizer_*.py` modules are now
  `test/test_graph_atomizer.py`, one section per original module, each retaining
  its docstring.

### Fixed

- The batch path dropped the critic telemetry between the graph and the
  manifest. `_merge_workflow_state_into_agent_state` copies an explicit field
  list off the astream chunk, and `facts_loop_telemetry` /
  `ontology_loop_telemetry` / `ontology_reduce_metrics` were not on it — so
  a batch run's manifests reported `critic: {calls: 0}` while their own
  `retrieval_metrics` recorded the calls that had actually been billed, and
  the score distributions had to be mined out of the LLM cache again, which
  is exactly what the manifest blocks exist to prevent. The three fields are now copied, and `RunManifest` gains
  `ontology_reduce_metrics` — the reduce policies' evidence
  (`minted_duplicates` and pairs, `deletes_dropped_unredeclared`,
  `apply_deletes_no_match`, `fresh_ontologies_merged`) was computed and then
  recorded nowhere.

- Ontology renders failed to parse for units producing a single large update.
  Responses were complete (`finish_reason: stop`) and bracket counts balanced;
  only the closing bracket *kinds* were transposed at the tail of a JSON-LD
  document nested in a singleton list, so the failure was concentrated in
  updates carrying one long operation rather than several short ones. Addressed by the
  flat wire above, by presenting a literal envelope skeleton in the prompt
  instead of prose over a `$ref`-indirected schema, and by
  `repair_bracket_kinds`, which rewrites each closing bracket to the kind its
  opener requires without inserting, deleting or reordering, and abandons repair
  on an unopened closer or on frames still open at end of input.

  Touches: `agent/common.py::repair_bracket_kinds`; `prompt/graph_format.py`.
  Counter: `llm/json_bracket_repair`.
  Test: `test/test_graph_update_wire_regression.py`, with the recorded failures
  in `test/data/llm_malformed_graph_updates.json`.

- Malformed LLM JSON now repairs deterministically or fails with actionable
  feedback. LangChain's `parse_json_markdown` degraded malformed input to `None`
  or to a truncated prefix, so retry prompts carried a validation error naming
  no location and retries reproduced the same malformation.
  `unescape_json_delimiters` joins the sanitizer chain, repairing escaped string
  delimiters and escaped inter-token whitespace with a string-aware scan that
  leaves legitimate in-string escapes intact. `parse_json_object` replaces the
  lenient parse for every `PydanticOutputParser` call: strict parsing first,
  fenced-block extraction as a fallback, and any remaining failure raising with
  line, column and a surrounding context window. Partial recoveries and
  non-object JSON are rejected rather than validated. A request timeout is
  re-issued once per call before propagating; rate-limit and connection errors
  propagate immediately.

  Touches: `agent/common.py`; `tool/llm.py::LLMRequestTimeoutError`.
  Test: `test/test_llm_resilience.py`.

- Ontology updates applied no post-parse hygiene. `render_ontology_update` did
  not call `finalize_llm_graph`, so an invalid XSD typed literal reached the
  working graph and the compiled SPARQL update. Both update agents now share one
  mechanism: prefix sanitization, removal of invalid typed literals, then domain
  repairs applied to the insert side only, since a delete must match the stored
  triple exactly. `quarantined_literal_triples` moved to `UnitState`.

  Touches: `agent/update_common.py::finalize_update_report`;
  `agent/render_ontology.py`; `agent/render_facts.py`.
  Test: `test/test_update_agents_aligned.py`.

- `ToolBox.aserialize` raised `RuntimeError` whenever a document produced an
  ontology version under vector retrieval. It called the synchronous
  `add_ontology()`, whose guard refuses to reindex the vector store from within
  a running event loop. It now awaits `aadd_ontology()`.

  Touches: `toolbox.py`.
  Test: `test/test_serialize_vector_mode.py`.

- Critic suggestions leaked across facts renders. `state.suggestions` was
  written by `criticise_facts` and read by `render_facts_update` but never
  cleared, so a rejected unit's suggestions reached every later render of that
  unit, including the finding-driven repair, which then held both the
  improvement template's permissive contract and the findings block's
  rewrite-in-place requirement. Cleared at both writers, since the repair is
  reached either by a render consuming the suggestions or by the critic
  accepting on a later attempt of the same render.

  Touches: `agent/criticise_facts.py`; `agent/render_facts.py`.
  Test: `test/test_facts_suggestions_lifecycle.py`.

- The same suggestion leak in the ontology loop: neither
  `render_ontology_update` nor the accept branch of `criticise_ontology` cleared
  `state.suggestions`. Cleared at both writers.

  Touches: `agent/criticise_ontology.py`; `agent/render_ontology.py`.
  Test: `test/test_ontology_suggestions_lifecycle.py`.

- `deterministic_findings` survived an ontology render, carrying findings raised
  against the previous extract into the next iteration. Both update agents now
  consume findings and suggestions identically.

  Touches: `agent/render_ontology.py`.
  Test: `test/test_update_agents_aligned.py`.

- A false mandatory `UNKNOWN_TERM` finding caused repair renders to remove valid
  numeric values. The catalog referenced `qudt:QuantityValue` and `qudt:unit` in
  `rdfs:subClassOf` and `owl:onProperty` position, which made the QUDT namespace
  closed, while `qudt:numericValue` was declared nowhere in it. Every unit using
  that property therefore received a mandatory finding naming a class as the
  candidate for a predicate slot. Repair renders answered it by deleting the
  values outright or by re-encoding scalars as equal-bound ranges. Three rules
  changed: a namespace is
  closed only when the catalog declares terms in it in subject position; the
  configured quantity fallback vocabulary and `FACTS_CODE_PREDICATES` are exempt;
  and alias candidates are role-filtered against the catalog's declarations.
  Deployment-configured exemptions now travel as a single `ValidationPolicy`
  object on the toolbox.

  Touches: `tool/facts_validation/terms.py::collect_declared_namespaces`,
  `ValidationPolicy`; `FACTS_ADDITIONAL_STANDARD_NAMESPACES`;
  `FACTS_QUANTITY_FALLBACK_VOCABULARY`; `FACTS_CODE_PREDICATES`.
  Test: `test/test_facts_term_policy.py`.

- A repair render that answered the findings prompt by deleting triples was
  detected but retained. The guard now rolls back to the pre-repair graph and
  re-collects findings against it, so the recorded residual describes the graph
  that was kept. Its predicate was strengthened: the previous condition
  (`mandatory_after >= mandatory_before`) scored the deletion of the statement a
  finding refers to as a successful repair. It now also fires when a render
  removed triples and wrote none back, while remaining quiet for a rewrite that
  shrinks the graph by collapsing a duplicate. The findings prompt states the
  repair contract explicitly.

  Touches: `stategraph/atomic.py`; `prompt/render_facts.py`.
  Metric: `facts_repair_delete_only`.
  Test: `test/test_facts_repair_rollback.py`.

- Merge-guard vetoes now hold across a cluster. `_build_identity_clusters`
  applied distinctness guards pairwise and then union-found the accepted edges,
  so a vetoed pair A–C was still united through accepted edges A–B and B–C. This
  defeated every guard, including the validation gate's own un-merge vetoes.
  Vetoes are precomputed per candidate cluster and checked across both
  components before a union; blocked unions are recorded with reason
  `cluster_veto`.

  Touches: `tool/agg/aggregate.py::_build_identity_clusters`.
  Metric: `facts_rejected_merges`.
  Test: `test/aggregation/test_merge_regressions.py`.

- String literals on arbitrary domain predicates no longer count as label
  agreement. Every untyped or language-tagged string of at least 3 characters
  was harvested as an `alt_label` and fed into the label-intersection merge
  tier. `alt_labels` now apply only to entities carrying no `rdfs:label`.

  Touches: `tool/agg/signatures.py`.

- Labels differing only by conflicting initials are treated as distinctness
  evidence. `_tokenize` dropped tokens of at most 2 characters, making such
  label token sets identical. Short tokens are retained, token comparisons strip
  edge punctuation, and a guard vetoes pairs whose labels are identical except
  for non-alias-compatible short tokens.

  Touches: `tool/agg/aggregate.py`; `AGG_INITIALS_DISTINCT_GUARD` (default
  enabled).

- Merge-created self-loops are dropped rather than asserted. A triple whose
  subject and object became identical only through the identity mapping is
  removed with a warning; reflexive triples asserted by the source are retained.

  Touches: `tool/agg/rewriter.py`.

- `AGG_SIMILARITY_THRESHOLD` no longer appears to control in-pipeline
  clustering. The aggregator overrode it with
  `AGG_CANDIDATE_SIMILARITY_THRESHOLD` for its only clustering call. The
  pipeline clusterer is constructed at the candidate threshold directly, and the
  field is documented as what it controls: the cross-graph `EntityAligner`
  threshold used by `/align_entities` and `match-graphs`.

  Touches: `tool/agg/aggregate.py`; `.env.example`;
  `docs/user_guide/aggregation.md`.

- `update_ontology()` returned `None` whether or not the update applied, so a
  batch discarded by the `ONTOLOGY_MAX_TRIPLES` backstop still produced
  `Status.SUCCESS` on an unchanged working graph. It now returns `bool` and the
  caller records the discard. The status remains `SUCCESS`, since the
  pre-update graph is intact and a re-render would meet the same ceiling.

  Touches: `onto/unit_states.py::update_ontology`; `agent/render_ontology.py`.
  Metric: `ontology/update_rejected_over_budget`.

- `facts_findings_residual` was computed over the wrong population. It summed
  `attempts[-1].n_deterministic_findings` only for units whose last attempt was
  an LLM repair, so clean units and retry-exhausted units contributed zero, and
  it summed total rather than mandatory findings. It is now read from each
  unit's final findings, with `facts_mandatory_residual` reporting the mandatory
  subset. Values from earlier runs are not comparable on this key.

  Touches: `onto/run_manifest.py`; `stategraph/atomic.py`.

- `RunManifest.facts_triples` was not comparable to the `.facts.ttl` file beside
  it, counting the aggregated graph including provenance rather than the
  serialized output. `facts_triples_serialized` records the file's triple count.

  Touches: `onto/run_manifest.py`; `api/process_helpers.py`.

### Documentation

- Shapes-as-a-stored-artifact documented across the set:
  `docs/user_guide/validation.md` (three sources, why the partition is separate,
  the flush policy), `docs/user_guide/tenancy.md` and
  `docs/user_guide/triple_stores.md` (the third partition, seed vs persistence),
  `docs/user_guide/api.md` (`/shapes`, `include_shapes` on `/flush`),
  `docs/user_guide/configuration.md` and `docs/architecture/ontology_catalog.md`
  (a `ShapesCatalog` row in the responsibility table, plus the `owl:Ontology`
  collision that motivates the separation).

- Full audit of the documentation set against the code. Verified: every variable
  in `.env.example`, every `RetrievalMetric` value, every module path and symbol
  named in prose, every documented default against its declared field, and every
  internal link and anchor. Corrections:

  - `docs/contributing.md` instructed contributors to source `.env` before
    running the suite, which this release blocks. It now documents the offline
    run, the marker semantics, why sourcing `.env` invalidates the suite, and
    the fixture-location rule.
  - `docs/user_guide/workflow.md` stated that the ontology loop has no
    deterministic finding lane. It has one, in shadow mode.
  - `docs/user_guide/validation.md` stated that the ontology reduce counters
    appear in the run manifest. They are not carried to any output surface;
    the page now says so.
  - `docs/user_guide/performance.md` stated that raising `MAX_VISITS` to 2
    approximately doubles the calls per unit. It adds one call to a facts unit
    and doubles only for the ontology loop.
  - `docs/user_guide/playbooks.md` attributed unparsable model output to a
    disabled critic; that path is handled by the parser.
  - The validation page is retitled "Validation: Facts, Ontology Deltas and
    SHACL", having covered both lanes since this release.

- New sections: LLM response handling in the configuration guide (sanitizer
  chain, strict parsing, fenced-block and bracket-kind fallbacks, retry bounds);
  merge-refusal reason codes and key-supported clusters in the aggregation
  guide; `ontology_snapshot_triples` and the corrected run-manifest sample,
  including `selection`, `critic`, `ontology_critic` and
  `facts_triples_serialized`, in the observability guide.

## [0.6.1] - 2026-08-10

### Changed

- `LLM_GRAPH_FORMAT` defaults to `jsonld`. Turtle remains supported for
  providers whose structured output handles strings more reliably than nested
  objects. This changes behaviour for deployments that never set the variable.
  The default was declared in four places — `ServerConfig`, `AgentState`,
  `UnitState`, and the `llm_graph_format_ctx` context variable that
  `coerce_llm_graph_wire` falls back to when `model_validate` is called without
  a validation context — and all four moved together.

  Test: `test/test_llm_graph_format_default.py`.

- `ONTOLOGY_MAX_TRIPLES` defaults to unlimited. At its previous value of 50,000
  it could not bind: such a graph is approximately 634,000 tokens as Turtle,
  against a largest observed ontology of 1,409 triples. It is a runaway-growth
  backstop on the per-unit working graph, not a context bound; use
  `ONTOLOGY_CONTEXT_MAX_TRIPLES` for prompt size.

  Test: `test/test_ontology_max_triples.py`.

- `ontocast_extract` (LangChain/MCP tool) takes its `render_mode` default from
  `RENDER_MODE` instead of hardcoding `ontology_and_facts`, and parses it
  through `parse_render_mode_param` like every other entry point. Omitting the
  argument honours the server's configuration.

  Touches: `tool/langchain_tools.py`.

### Added

- `ONTOLOGY_CONTEXT_MAX_TRIPLES` (default 4,000) bounds the ontology context in
  every context mode. Previously only `selected_vector_search_ontology` bounded
  it: `selected_single_ontology` and `fixed_single_ontology` serialized the
  whole selected ontology into every prompt, and the facts fan-out serialized
  the union of every artifact without a cap. Over budget, the condenser drops in
  increasing order of harm — header and list noise, then redundant structure,
  then glosses — and never drops labels, types, hierarchy or domain and range
  declarations. A graph that cannot be reduced to fit is passed through with a
  warning rather than truncated, since removing load-bearing schema produces an
  extraction failure. Enforced at `format_ontology_chapter` and included in the
  snapshot memoization key.

  Touches: `onto/ontology_condense.py`.
  Test: `test/test_ontology_condense.py`.

- `ONTOLOGY_SNAPSHOT_TRIPLES` retrieval metric, written for every context mode.
  Previously only the vector resolver recorded a snapshot size, nested under
  `patch_retrieval`.

- The seed-free graph pruners and predicate vocabularies move to
  `onto/graph_prune.py`, shared by induced-subgraph retrieval and the condenser.

- `.env.example.minimal` and a Configuration Playbooks guide. The full
  configuration surface is approximately 200 variables. The minimal file carries
  29, grouped by the decision they belong to rather than by configuration class,
  and the guide provides a playbook per task — evaluate, build an ontology,
  populate facts, scale the catalog, serve it — with a symptom-to-setting triage
  table. It covers conversion and chunking (`CHUNK_MIN_SIZE`, `CHUNK_MAX_SIZE`,
  `CHUNK_SEGMENTER`, `CHUNK_SECTION_CLASSIFIER`, `CHUNK_BIBLIOGRAPHY_MODE`,
  `CONVERTER_PROFILE`), local-encoder alignment, SHACL shapes, LLM caching and
  the web-search toggle.

  Touches: `.env.example.minimal`; `docs/user_guide/playbooks.md`.
  Tests: `test/test_env_example_coverage.py` — every name resolves to a real
  setting, the file remains a subset of `.env.example`, it stays under a
  variable ceiling, and exact variable counts quoted in prose match reality.

### Fixed

- `.env.example` named two local encoder models without the
  `sentence-transformers/` prefix. `SharedEncoder` caches by the literal
  `(model name, device)` pair, so `AGG_EMBEDDING_MODEL` and
  `EMBEDDING_MODEL_NAME` as shipped would not share a resident model with the
  prefixed defaults, loading two copies of the same checkpoint. Both spellings
  are valid, so nothing surfaced the difference.

  Test: `test/test_env_example_coverage.py`.

- `ONTOLOGY_MAX_TRIPLES` could lock the ontology loop out. The guard compared
  absolute post-apply size, so a working graph seeded above the cap failed every
  subsequent update, including deletions, for the remainder of the run. It now
  rejects only updates that grow the graph past the cap, and logs the
  already-over case distinctly.

  Touches: `onto/unit_states.py`.

- The test suite no longer loads a developer's `.env`. Removing `env_files` from
  `[tool.pytest.ini_options]` was insufficient, because `pytest-dotenv` loads a
  discovered dotenv file even with no `env_files` set: every `BaseSettings`
  instance built in a test took local configuration, and a real `LLM_API_KEY`
  was present in the environment. The plugin is uninstalled and blocked via
  `-pno:dotenv`, written as a single token because `toml-sort --all` sorts array
  values and would otherwise separate a two-token form. `test/conftest.py`
  asserts that the pipeline mode selectors read their declared defaults.

  Touches: `pyproject.toml`; `test/conftest.py`.

- The LangChain `apply_graph_update` tool pins its own Turtle coercion. Its
  interface is Turtle-in by parameter name (`insert_ttl`, `delete_ttl`) and was
  incorrectly tracking the configured LLM wire format.

  Touches: `tool/langchain_tools.py`.

### Documentation

- `RENDER_MODE` is documented: a Render Mode section in the configuration guide
  covering what each value skips — `ontology` writes no facts to the triple
  store, `facts` bypasses the ontology block and depends on the existing catalog
  — with the per-request precedence chain and the 400-on-invalid-value contract.

- Ontology context behaviour previously visible only in code: a non-empty
  `ontology_context_fixed_ontology_id` forces fixed mode over an explicit
  `ontology_context_mode`; a fixed id matching no catalog entry degrades to an
  empty snapshot with a warning rather than an error; `selected_single_ontology`
  costs one additional LLM call per content unit; the consistency critic runs
  only under `selected_vector_search_ontology`; and facts units reuse a merged
  document-level context rather than re-resolving it.

- Documentation search indexes environment variables. The Material search
  separator did not split on underscores, so `ONTOLOGY_CONTEXT_MODE` was indexed
  as a single token. The separator is updated, both mode selectors name their
  variable in a heading, and the configuration and ontology-context pages carry
  a search boost.

- `README.md` and `docs/index.md` gained a Configuration section.

- The `WEB_SEARCH_*` block is presented as three annotated tables rather than a
  code fence, and states that the lane is inactive at its defaults.

- `MAX_VISITS_PER_NODE` is documented under its canonical name as well as the
  `MAX_VISITS` alias. Vector-mode wording covers LanceDB as well as Qdrant.

## [0.6.0] - 2026-08-10

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
  — but `ontocast serve` and `ontocast process` still *raised* without
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
  replayed runs are measured in, reported zero tokens.
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
  not).
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
    fractionally is strictly worse: `References` carries no information about
    which cell a document is in, so scoring it only narrows the margins the
    tier decides on.
  - The content tier ships **off** (`auto`, not the default). Its errors are
    few but severe — chemistry prose scores `standard` over `academic` past the
    acceptance margin — so it is gated to documents with essentially no
    headings, excludes `news` as a semantic attractor, and demands a 4.0 margin
    against the heading tiers' 1.8.
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
  gained `keywords`, with `order`/`ordered` where a canonical order exists.
  Every keyword was authored against a real document in
  `test/data/schema_corpus.json` and cut if it matched nothing, so no cell rests
  on invented vocabulary.
- **Document-type detection corpus** (`test/data/schema_corpus.json`,
  `run/fetch_schema_samples.py`). One real document per cell — RFC 7231, *Pride
  and Prejudice*, a USPTO patent, the CC BY 4.0 legal code, the nginx guide, a
  Europe PMC trial protocol, a Wikinews article, plus the in-repo 10-Q and
  chemistry paper. Only heading sequences and sampled paragraphs are committed,
  each with its source URL and licence, so the suite stays offline and a few
  tens of kB. A nine-way classifier cannot be tuned on two document types.
- **Schema reporting in `ontocast sections`** — the resolved schema, the tier
  that chose it, its margin over the runner-up, and the ranked candidate
  evidence. The only way to see a weak-but-accepted detection; free in
  `lexical` mode.

What this changes for a financial document: with detection off it resolves to
the academic default and almost nothing is labeled, since the academic
vocabulary recognises none of its headings. With detection on it resolves to
`financial` on the free lexical tier and its sections are labeled
(`notes_to_financials`, `md_and_a`, `legal_proceedings`,
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
  pointing at a catalog term failed, producing phantom findings describing the
  absent schema.
- **SHACL runs with RDFS inference by default** (`FACTS_SHACL_INFERENCE`,
  `FACTS_SHACL_ADVANCED`), matching how the shipped shapes are authored and
  validated by their own repo harness. SHACL property paths carry no
  `rdfs:subPropertyOf` entailment, so a shape naming a superproperty reported
  the specialised predicate the renderer emitted as missing: 268 violations at
  `none` than at `rdfs`. A `FACTS_SHACL_MAX_TRIPLES`
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
  section numbering (`2.1 Synthesis of thin films`). On real converted journal
  PDFs the great majority of detected headings were unmatched
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
  holding one resident model instead of two. Defaults unchanged; opt-in recipe in
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

- Dead/duplicate tests and unused fixtures. Recall harness (`test_retrieval_recall.py` + support) moved out of the unit suite — measurement belongs in `ontocast-validation`. A relative aggregation “no damage” test dropped.
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
- Path-dependent ontology tests; concurrency bound flake; dead tenancy self-assignment; `test-api` entry shim.

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
  evaluates against public and prebuilt corpus tiers (`ONTOCAST_RECALL_*`); and provides
  controls that flip index/retrieval axes without editing corpus files on disk.

### Changed

- **Retrieval defaults retuned against measured recall**; configurations that
  fit only one document collection are flagged as such in the configuration guide to caution against
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
- **OpenAI Batch API helpers** (`ontocast.tool.llm_batch`) to export chat batch JSONL and import completed results into the LLM disk cache for offline pre-warming.
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
- `match-dirs` standalone CLI client for batch evaluation against the match endpoints.

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

### Upgrading to 0.6.2

- **Ontology render output shape.** A caller that constructs or inspects
  `GraphUpdateRenderReport` directly must move from
  `graph_update.triple_operations[]` to the `insert_graph` / `delete_graph`
  fields. `to_graph_update()` produces the unchanged internal `GraphUpdate`, so
  code downstream of it is unaffected. Interleaved insert/delete sequences
  within one render are no longer representable. On-disk LLM cache entries for
  ontology renders are invalidated and will re-fetch.
- **`ontocast.tool.facts_invariants` no longer exists.** Import from
  `ontocast.tool.facts_validation` instead; the package `__init__` re-exports
  the previous public names.
- **`FACTS_ACCEPT_BLOCKING_SEVERITY`** (default `critical`) replaces the
  critic's score threshold as the facts acceptance gate. Set it to `never` to
  let deterministic findings gate alone. The critic's `score` and `success` are
  still reported and are no longer consulted.
- **The `data/` directory is gone.** Anything referencing `data/ontologies`,
  `data/json` or `data/pdf` — including a local `ONTOCAST_ONTOLOGY_DIRECTORY`
  pointing at it — must be repointed. The two TTL fixtures used by the test
  suite are in `test/data/ontologies/`.
- `facts_findings_residual` changed population and is not comparable with
  values from earlier runs; `facts_triples` is joined by
  `facts_triples_serialized`, which is the count matching the emitted
  `.facts.ttl`.

### Upgrading to 0.6.0 (from 0.4.3)

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

[0.6.2]: https://github.com/growgraph/ontocast/compare/v0.6.1...HEAD
[0.6.1]: https://github.com/growgraph/ontocast/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/growgraph/ontocast/compare/v0.4.3...v0.6.0
[0.4.3]: https://github.com/growgraph/ontocast/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/growgraph/ontocast/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/growgraph/ontocast/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/growgraph/ontocast/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/growgraph/ontocast/releases/tag/v0.3.0
