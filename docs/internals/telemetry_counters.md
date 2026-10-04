# Telemetry counters

This page lists every field and counter a run reports, and the code that
emits it, so you can read a value you have not seen before or check that a
change emits what it should. [Reading run telemetry](../guides/telemetry.md)
explains which fields answer which question.

## Budget fields

`BudgetTracker` (`ontocast/onto/state.py`) is returned as `metadata.budget` by
`/process` and `/process_unit` and written as the run manifest's `budget`. Each
content unit runs with its own tracker, merged into the document's when the
unit finishes: fields and counters are summed, and duration keys ending in
`_max` keep the maximum.

| Field | Meaning |
|---|---|
| `calls_count` | Calls sent to the provider, including calls that timed out |
| `cache_hits` | Calls answered from the LLM response cache |
| `chars_sent`, `chars_received` | Prompt and answer characters, for sent and cached calls alike. A timed-out call adds its prompt |
| `input_tokens`, `output_tokens` | Tokens billed for sent calls |
| `cached_input_tokens`, `cached_output_tokens` | Tokens stored with cached answers. Cache entries written without usage add nothing |
| `reasoning_tokens` | Reasoning tokens, inside the output totals; sent and cached calls |
| `cache_read_input_tokens` | Input tokens served from the provider's prompt cache, inside the input totals; sent and cached calls |
| `cache_creation_input_tokens` | Input tokens written to the provider's prompt cache; sent and cached calls |
| `ontology_triples_generated`, `ontology_operations_count` | Triples and update operations the ontology renders produced |
| `facts_triples_generated`, `facts_operations_count` | The same for facts renders |
| `node_durations` | Seconds per key; see [Duration keys](#duration-keys) |
| `counters` | Event counts per key; see [Budget counters](#budget-counters) |

Two fields are derived when the budget is serialized:

| Field | Formula | `null` when |
|---|---|---|
| `prefix_cache_hit_rate` | `cache_read_input_tokens / (input_tokens + cached_input_tokens)` | No input tokens were reported |
| `reasoning_share_of_output` | `reasoning_tokens / (output_tokens + cached_output_tokens)` | No output tokens were reported |

The denominators include cached tokens because the numerators do. Dividing by
`input_tokens` alone can exceed 1 on a run that was partly answered from the
cache.

## Duration keys

`node_durations` holds seconds. A plain pipeline node name is that node's wall
clock, recorded by the `_timed` wrapper in `ontocast/stategraph/create.py`:
`Convert to Text`, `Chunk Text`, `Update Ontology`,
`Normalize Ontology Updates`, `Consolidate Ontology`, `Render Facts`,
`Merge Facts`, `Validate Facts`, `Structural Check`, `Consistency Critic`,
`Serialize`.

The two fan-out nodes, `Update Ontology` and `Render Facts`, add keys written
in `ontocast/stategraph/node_factories.py`:

| Key | Meaning |
|---|---|
| `<node>/unit_sum` | Each unit's loop time, summed over units |
| `<node>/worker_wait` | Time units waited for a `PARALLEL_WORKERS` slot, summed |
| `<node>/loop_lag_total` | Time the event loop ran late while the units ran (`ontocast/util/loop_lag.py`) |
| `<node>/loop_lag_max` | The longest single delay |

`BudgetTracker.parallel_efficiency(node)` returns `<node>/unit_sum` divided by
the node's wall clock, the effective number of workers, which the end-of-run
log prints as `Effective workers`. Waiting on I/O yields the loop and adds no
lag, so lag measures only CPU work done on the event loop.

Stage keys, summed over every call or unit:

| Key | Written in | Time spent |
|---|---|---|
| `llm/provider` | `ontocast/tool/llm.py` | Inside the provider call, including rate-limiter waits |
| `llm/inflight_wait` | `ontocast/tool/llm.py` | Waiting for an `LLM_MAX_INFLIGHT` slot |
| `llm/cache_lookup` | `ontocast/tool/llm.py` | Reading the LLM response cache |
| `chunk section classify/worker_wait` | `ontocast/tool/chunk/section_llm.py` | Section classification calls waiting for a worker slot |
| `ctx/merge_document_ontology` | `ontocast/stategraph/context_resolver.py` | Merging the document's ontology for the facts fan-out |
| `ctx/union_document_ontology` | `ontocast/stategraph/context_resolver.py` | Building the union chapter (`ONTOLOGY_CONTEXT_SCOPE=document`) |
| `ctx/working_graph_copy` | `ontocast/stategraph/atomic.py` | Copying the ontology context into a unit's working graph |
| `prompt/ontology_chapter` | `ontocast/agent/render_facts.py` | Building or fetching the facts ontology chapter |
| `repair/deterministic` | `ontocast/agent/render_facts.py`, `ontocast/stategraph/atomic.py` | LLM-free repairs and finding collection in the unit loops |
| `ontology_validation/unit_findings` | `ontocast/stategraph/atomic.py` | Validating an ontology unit's update |

For one fan-out node the time roughly adds up as

```
wall × effective workers ≈ llm/provider + llm/inflight_wait
                           + <node>/worker_wait + <node>/loop_lag_total + other
```

## Budget counters

`budget.counters` holds event counts.

| Counter | Written in | Counts |
|---|---|---|
| `llm/calls_timed` | `ontocast/tool/llm.py` | Provider calls that returned |
| `llm/calls_failed` | `ontocast/tool/llm.py` | Provider calls that raised, for any reason |
| `llm/timeouts` | `ontocast/tool/llm.py` | Calls abandoned at `LLM_REQUEST_TIMEOUT_SECONDS`; part of `llm/calls_failed` |
| `llm/rate_limited` | `ontocast/tool/llm.py` | Throttles that survived the SDK's retries; part of `llm/calls_failed` |
| `llm/calls_rejected` | `ontocast/tool/llm.py` | Requests the provider refused as configured; part of `llm/calls_failed`. Any non-zero value means the run stopped |
| `llm/parse_retry` | `ontocast/agent/common.py` | Calls re-sent because the previous answer did not parse or validate |
| `llm/json_bracket_repair` | `ontocast/agent/common.py` | Answers recovered by correcting mismatched closing brackets |
| `llm/parse_abandoned` | `ontocast/agent/common.py` | Calls given up after the retries ran out or the same JSON error recurred |
| `rdf/turtle_repair` | `ontocast/onto/rdfgraph.py` | Turtle payloads that parsed only after repair (truncated statement, unknown prefix, update-query wrapper) |
| `rdf/jsonld_rdflib_fallback` | `ontocast/onto/rdfgraph.py` | JSON-LD payloads that failed URDNA2015 normalization and were parsed by rdflib instead |
| `repair/compact_iri_literal` | `ontocast/tool/facts_validation/literal_repair.py` | Compact IRIs written as plain strings on an IRI-valued position and coerced back to IRIs, in renders and in critic and completion patches |
| `llm/ontology_selection` | `ontocast/agent/select_ontology_catalog.py` | Ontology selection calls, one per unit in `selected_single_ontology` mode |
| `chapter/text_chars_before` | `ontocast/agent/render_facts.py` | Summed length of capped text literals before the text caps |
| `chapter/text_chars_after` | `ontocast/agent/render_facts.py` | The same after the caps |
| `chapter/literals_clipped` | `ontocast/agent/render_facts.py` | Literals the caps shortened |
| `chapter/text_over_budget` | `ontocast/agent/render_facts.py` | Chapters still over `ONTOLOGY_TEXT_TOTAL_BUDGET` after every role was tightened |
| `ctx/merge_document_ontology.calls` | `ontocast/stategraph/context_resolver.py` | Document ontology merges; `1` per document |
| `ctx/union_document_ontology.calls` | `ontocast/stategraph/context_resolver.py` | Union chapters built; `1` per facts-only document with a document scope |
| `<node>/unit_errors` | `ontocast/stategraph/node_factories.py` | Units of a fan-out node that raised |
| `ontology/update_rejected_over_budget` | `ontocast/agent/render_ontology.py` | Ontology updates discarded for exceeding `ONTOLOGY_MAX_TRIPLES` |

`calls_count` equals `llm/calls_timed + llm/timeouts`: a call that raised for
any other reason is not billed and not counted in `calls_count`. The
`chapter/*` counters are recorded once per chapter built, not per call that
reads it, and only when at least one text cap is set; [How prompts are
built](prompt_construction.md#text-caps) describes the caps.

## Retrieval metrics

`AgentState.retrieval_metrics` is returned as `metadata.retrieval_metrics` and
written into the run manifest. Its top-level keys are the members of
`RetrievalMetric` in `ontocast/onto/enum.py`. A key appears only when the stage
that writes it ran.

Ontology context, written per unit in `ontocast/stategraph/context_resolver.py`
and merged onto the document, the last unit winning on shared keys:

| Key | Meaning |
|---|---|
| `ontology_context_mode` | The mode that resolved the context |
| `ontology_snapshot_triples` | Triples in the resolved context; for the facts fan-out, the shared document context when there is one |
| `empty_snapshot_reason` | Why a unit's context was empty |
| `patch_retrieval` | Retrieval detail in `selected_vector_search_ontology` mode; see below |

Ontology fan-out, written in `ontocast/stategraph/node_factories.py`:

| Key | Meaning |
|---|---|
| `ontology_writable_count`, `ontology_primary_units` | Distinct writable ontologies, and units assigned a primary one |
| `ontology_findings_residual`, `ontology_mandatory_residual` | Findings left on the ontology units after their loops, all and mandatory |
| `ontology_critic_calls`, `ontology_critic_accepted` | Ontology critic calls, and those whose verdict let the unit leave the loop |
| `ontology_critic_fixes_applied`, `ontology_critic_fixes_noop` | Critic fixes that reached the graph, and fixes that removed exactly what they added |
| `ontology_critic_patches_rolled_back` | Critic patch passes in which a fix was undone |

Facts fan-out, written in `ontocast/stategraph/node_factories.py`:

| Key | Meaning |
|---|---|
| `facts_anchor_count`, `facts_anchor_units` | Distinct anchor ontologies, and units assigned one |
| `facts_findings_residual`, `facts_mandatory_residual` | Findings left on the facts units after their loops, all and mandatory |
| `facts_critic_calls`, `facts_critic_accepted` | Facts critic calls, and those whose verdict let the unit leave the loop |
| `facts_critic_fixes_applied`, `facts_critic_fixes_residual`, `facts_critic_fixes_noop` | Fixes that reached the graph, fixes left for judgement, and fixes that removed exactly what they added |
| `facts_critic_fixes_rolled_back`, `facts_critic_patches_rolled_back` | Fixes undone for leaving the unit worse, and passes in which that happened |
| `facts_critic_fixes_junk_refused` | Inserts refused for minting a placeholder node |
| `facts_critic_fixes_unresolved_prefix` | Fixes naming a prefix nothing declares |
| `facts_critic_units_unreviewed`, `facts_critic_units_skipped` | Units whose critic call failed, and units not sent to the critic |
| `facts_completion_calls`, `facts_completion_triples_inserted`, `facts_completion_measurements_recovered` | The completion pass: calls, triples that stayed, and missed measurements found |

Aggregation and validation, written in `ontocast/stategraph/node_factories.py`,
`ontocast/stategraph/facts_gate.py` and `ontocast/tool/facts_validation/gate.py`:

| Key | Meaning |
|---|---|
| `facts_rejected_merges` | Entity merges the aggregator's guards refused, for the graph returned |
| `facts_merge_repair_passes`, `facts_merge_vetoes`, `facts_merge_repairs_rejected` | The un-merge repair: passes kept, merge pairs vetoed, passes reverted |
| `validated_without_ontology_context` | `true` when the facts were validated with an empty ontology context |
| `facts_validation_findings`, `facts_validation_errors` | Findings after validation, and those of error severity |
| `facts_shacl_violations_before`, `facts_shacl_violations_after` | SHACL violations around the automatic repair. Present only when the repair ran |
| `facts_shacl_repairs`, `facts_shacl_autofix_passes`, `facts_shacl_autofix_reverted` | Repairs applied, passes kept, and whether a pass was reverted. Present only when the repair ran |
| `structural_ontology_components_max` | The highest connected-component count among the ontologies the structural check examined |
| `consistency_conflicts` | Conflicts the consistency critic reported |

`/process` and `/process_unit` write the validation keys through the same
function, so the two routes report the same set for the same graph.

### patch_retrieval

`patch_retrieval` is the retriever's own record
(`ontocast/tool/vector_store/patch_retriever.py`) of the unit that wrote last.
Its main keys:

| Key | Meaning |
|---|---|
| `query_count`, `top_k`, `effective_max_atoms` | Queries run, hits per query, and the term budget |
| `candidate_hits`, `threshold_rejected`, `atoms_after_dedupe`, `atoms_final` | The funnel from raw hits to terms kept |
| `seed_iris`, `seeds_by_ontology`, `relevance_by_ontology` | Terms kept, their count per ontology, and each ontology's best score |
| `lexical_trigger_hits`, `lexical_trigger_promoted`, `lexical_trigger_appended` | Terms found by exact name match, and how they entered the result |
| `module_closure_iris`, `module_closure_triples`, `module_closure_declined_iris` | Small ontologies included whole, what they added, and those the closure budget left out |
| `snapshot_triple_count` | Triples in the assembled context |
| `ontology_rank_diagnostics` | Per-query ontology ranking, when `ONTOLOGY_PATCH_DUMP_ONTOLOGY_RANKS` is on |

[Retrieval](retrieval.md) explains the funnel.

## Run manifest blocks

`RunManifest` (`ontocast/onto/run_manifest.py`) is written by
`dump_run_manifest` in `ontocast/api/process_helpers.py` as `<name>.run.json`
beside each `<name>.facts.ttl`, by `ontocast process` only. Fields that are `null` are
left out.

| Field | Contents |
|---|---|
| `source`, `line_number` | Input file name, and the line for a multi-record `.jsonl` |
| `ontocast_version`, `render_mode`, `current_domain`, `doc_iri`, `tenant`, `project` | What ran, and where its IRIs and graphs went |
| `loops` | The effective `max_visits`, `facts_critic_passes`, `ontology_critic_passes` |
| `llm` | Provider, model, temperature, the Ollama and reasoning settings, `requests_per_second`, `max_retries`, `prompt_cache_key`, `max_inflight` |
| `prompting` | `llm_graph_format`, `llm_output_layout`, the resolved `ontology_chapter_format`, `ontology_context_scope`, `fanout_warmup_units`, `parallel_workers`, `embedding_model_name`. Scope and warm-up are left out on the single-unit path |
| `budget` | The full budget, as above |
| `critic`, `ontology_critic` | Per loop: `calls`, `accepted`, score minimum, median, maximum and histogram, fix severity histograms, `accept_reason_histogram`, `incumbent_accepted`, patch passes, fix outcomes (`fixes_applied`, `fixes_noop`, `fixes_rolled_back`, `fixes_junk_refused`, `fixes_unresolved_prefix`, `patches_rolled_back`), `units_unreviewed`, `units_skipped`, triples deleted and inserted |
| `completion` | `calls`, `units`, `subjects_inserted`, `subjects_rolled_back`, `triples_inserted`, `measurements_recovered` |
| `selection` | `target_sections`, `exclude_sections`, `summarize_sections`, `summary_max_sentences`, `bibliography_mode`, `non_content_mode`, `labeled_units`, `unlabeled_units`, `section_label_histogram`, and the units skipped as bibliography, undersized or non-content |
| `validation_config` | `context_from_units`, `json_mode`, `shapes_prompt_contract`, `shapes_prompt_selection`, `shapes_triples`, `shacl_inference`, `numeric_coverage_mandatory`, and the length (not the text) of the facts user instruction |
| `ontology_reduce_metrics` | Counts from reducing the units' ontology updates: duplicate minted terms and their pairs, deletes dropped, unattributed inserts and deletes, artifacts produced, and the apply step's own counts |
| `graph_metrics` | The written facts graph: `nodes`, `edges`, `components`, `largest_component`, `isolated_nodes` |
| `ontology_triples`, `facts_triples`, `facts_triples_serialized` | Ontology triples written; facts triples in memory, provenance included; facts triples in the `.facts.ttl` file |
| `retrieval_metrics` | The same mapping `/process` returns |

Score buckets in `score_histogram` are decades keyed like `"70-79"`; an empty
histogram with `calls` above zero means the critic returned no score that
could be read.
