# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Terms used in these entries

Retrieval and aggregation changes are justified against measurements, and the
entries name the evaluation sets and metrics involved. For readers outside the
project:

- **Evaluation corpora.** *Text2KGBench* is a public benchmark used here as a
  regression guard. The *materials-science corpus* is an internal evaluation
  set: eight mutually referencing ontology modules (a domain vocabulary, a
  units vocabulary, a qualified-value vocabulary, and others) with passages of
  real scientific prose. Individual passages used for tuning are referred to by
  a case number. Results measured only on the internal corpus are single-corpus
  fits and are flagged as such.
- **Seed recall vs. snapshot recall.** *Seed recall* is the share of expected
  ontology terms that survive retrieval ranking and budget truncation.
  *Snapshot recall* is the share that are actually defined in the ontology
  graph handed to the model — a term can be absent from the seeds yet still
  reach the model by being pulled in as a neighbour of one that was retrieved.
- **On-topic precision.** The share of terms in the assembled ontology context
  that belong to the ontology a given passage is about. It is reported for
  context, not optimised: a missing term cannot be used at all, whereas a
  surplus one only consumes prompt space. The measure also penalises correct
  multi-ontology contexts, since a units or provenance term legitimately
  drawn in from a sibling module counts against it.
- **Surface-form contract (`sf3`, `sf4`, …).** A version stamp on how ontology
  terms are converted into indexed text. Changing it changes the stored
  vectors, so a bump requires re-indexing existing collections; entries that
  bump it say so explicitly.


## [Unreleased]

### Breaking

- **LLM cache key gained fields, invalidating every existing entry.** The key
  now carries a `cache_format_version` (now `2`) plus the Ollama generation
  knobs `think` / `num_predict` / `num_ctx`. Caches written by earlier releases
  will not be hit, so the first run after upgrading re-pays for every call.
- **The on-disk cache now evicts on its own,** capped at 1 GB by default
  (`ONTOCAST_CACHE_MAX_BYTES`). This is new deletion behaviour; set the variable
  to `0` to restore unbounded growth.

### Fixed

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

### Added

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

### Removed

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

- Cross-chunk person/entity identity merge (initials-aware aliases; label-confirmed pairs bypass cosine gate).
- Per-unit `retrieval_metrics` fold back into document state; Docling chunker tokenizer budgeted from `CHUNK_MAX_SIZE`; semantic chunker guards for tiny sections.
- Path-dependent ontology/matsci tests; concurrency bound flake; dead tenancy self-assignment; `test-api` entry shim.

### Performance

- Fan-outs use slim `UnitLoopContext` instead of `AgentState.model_copy(deep=True)`.
- URDNA2015 hashing off the per-unit hot path (`working_graph_changed` via triple sets; lazy `OntologySnapshot.content_hash`).

## [0.5.0]

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
  Existing collections raise `EmbeddingContractMismatchError` and must be
  dropped (`VECTOR_STORE_WIPE_ON_INIT` / `--wipe-vector-store`). Across this
  release the stored atom text and payloads changed to: atomize only IRIs an
  ontology *describes* (`sf3`→`sf4`); derive `entity_role` from property
  *declaration*, not incidental predicate use (`sf4`→`sf5`); index
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
  `test/retrieval_gt.py`): real embeddings + Qdrant; seed / snapshot /
  term-level recall and per-stage funnel; Text2KGBench and prebuilt-corpus
  tiers (`ONTOCAST_RECALL_*`); ablation controls that flip index/retrieval
  axes without editing corpus files on disk.

### Changed

- **Retrieval defaults retuned against measured recall** (Text2KGBench +
  materials-science corpus; single-corpus fits flagged in the configuration
  guide). Notable defaults: `ONTOLOGY_PATCH_MAX_ATOMS` / `_BASE` → 96;
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

### Upgrading from 0.1.x / 0.3.x (general)
```bash
# Old
OPENAI_API_KEY=your_key_here

# New  
LLM_API_KEY=your_key_here
```

### Configuration Usage

```python
# Old way (no longer supported)
from ontocast.config import config

llm_provider = config.llm_config.provider

# New way
from ontocast.config import Config

config = Config()
llm_provider = config.tool_config.llm_config.provider
```

### ToolBox Initialization
```python
# Old way (no longer supported)
tools = ToolBox(
    llm_provider="openai",
    model_name="gpt-4",
    # ... many individual parameters
)

# New way
tools = ToolBox(config)
```

### CLI Parameters

### LLM Caching
```python
# Caching is now automatic - no configuration needed
```

```bash
# Skip ontology critique step
ontocast --skip-ontology-critique

# Or set environment variable
export SKIP_ONTOLOGY_DEVELOPMENT=true
ontocast --env-path .env
```
