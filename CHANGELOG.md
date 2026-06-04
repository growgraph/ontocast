# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Facts precision/recall/F1** on `POST /match/evaluate` (`fact_precision`, `fact_recall`, `fact_f1` and counts): relational triples only, excluding schema predicates and triples with ontological class/concept nodes in subject or object position.
- **Anthropic (Claude) and Google (Gemini) LLM providers** via `LLM_PROVIDER=anthropic|google`, with `ClaudeModel` and `GeminiModel` config enums.
- **Token usage reporting** in `BudgetTracker` when providers return `usage_metadata` on LLM responses (character counts remain the universal fallback).
- **LLM disk cache controls** on `LLMConfig`: `LLM_CACHE_ENABLED` (default on), `LLM_CACHE_READ_ONLY`, and in-memory plus on-disk stats via `LLMTool.get_cache_stats()`; `GET /info` exposes `llm_cache`.
- **Global LLM in-flight limit** (`LLM_MAX_INFLIGHT`, default 16) — shared semaphore caps concurrent provider requests across parallel unit workers.
- **Optional process concurrency cap** (`MAX_CONCURRENT_PROCESSES`) — limits simultaneous `/process` and `/process_unit` handlers (additional requests wait for a slot).
- **OpenAI Batch API helpers** (`ontocast.tool.llm_batch`) to export chat batch JSONL and import completed results into the LLM disk cache for offline benchmark pre-warming.
- **`BudgetTracker.cache_hits`** — disk-cache hits count toward character totals but not `calls_count`; included in budget summaries when non-zero.
- **Multi-domain section label catalog** (`data/section_labels/`) — versioned YAML schemas for academic, financial, legal, clinical, manual, fiction, and general documents; `section_schema_id` and `document_type_hint` on `/process` and CLI.
- **Structured-document preprocessing** for heading-structured text: the **Chunk** node runs prepare (segment → tag → filter → size) with section-aligned labels via document-wide regex spans and parallel LLM backfill; `target_sections` filters units before extraction.
- **Optional chunk summarization** — `summarize_sections` and `summary_max_sentences` on `/process` and CLI (`--summarize-sections`, `--summary-max-sentences`) run a **Summarize Chunks** graph node; ontology/facts render and critic prompts use `ContentUnit.extraction_text` (summary when present, else full chunk text).

### Changed
- **Section pipeline layout** — span detection and LLM backfill live under `ontocast.tool.chunk` (`sections.py`, `section_llm.py`, `segment.py`); section-label YAML and loader live in `ontocast.config.section_labels`; runtime settings remain `from ontocast.config import Config`; `SectionSpan` in `ontocast.onto.section_models`.
- **Chunk prepare** — pre-tag coalescing merges undersized hybrid segments into the right neighbor (trailing tiny segments merge left); LLM section backfill skips fragments below `CHUNK_SECTION_TAG_MIN_CHARS`; academic abstract detection uses relaxed heading patterns and optional front-matter span injection.
- **Chunk prepare** — section tagging, allowlist filtering, and size normalization run in one pipeline inside the Chunk node (removed separate Tag Sections workflow node).
- **LLM caching path** — `complete`, `extract`, `__call__`, and `acall` share one `_invoke_cached` implementation with consistent cache keys (normalized prompt text), optional disable/read-only modes, and provider calls gated by the global in-flight semaphore.
- **Facts extraction prompts** (`facts_guidelines.py`): clearer two-namespace contract — domain ontology is read-only schema plus optional **reference individuals**; all text-derived occurrences use `cd:` with `lowercase_snake_case` local names. New rules separate **classes** from **instances** (no PascalCase class IRIs in subject/object slots), forbid typing `cd:` entities as `rdfs:Class` / `rdf:Property`, and add a final structural validation checklist before output.

### Fixed
- **Entity alignment** (`EntityAligner`): identical `URIRef` across graphs always form a compatibility edge (score 1.0), so shared ontology terms (e.g. a class used in both predicted and ground-truth graphs) cluster correctly even when label embeddings differ.
- **Match / evaluate API** (`match_models`, `triple_evaluator`, `match_common`): entity fields stay `URIRef` through Pydantic validation; triple projection and entity precision/recall use set-based unmatched counts so shared-vocabulary IRIs are not double-counted as false positives/negatives.

### Documentation
- User guide: facts two-namespace model (`concepts.md`), facts guidelines vs `facts_user_instruction` (`user_instructions.md`), entity alignment and evaluate semantics (`aggregation.md`, `api.md`, `workflow.md`).
- User guide: LLM cache configuration, in-flight/process limits, batch pre-warming, and `/info` cache stats (`llm_caching.md`, `configuration.md`, `api.md`, `concepts.md`, `workflow.md`).
- User guide: structured documents — section tagging, section-aligned chunk labels, `target_sections` / `summarize_sections` (`concepts.md`, `workflow.md`, `api.md`, `configuration.md`).

## [0.4.0] - 2026-05-26

### Added
- **Parallel map/reduce pipeline** for document processing: per-unit ontology and facts loops run concurrently with configurable `PARALLEL_WORKERS`, retry budgets (`PARALLEL_ONTOLOGY_RETRIES`, `PARALLEL_FACTS_RETRIES`), and a dedicated `/process_unit` endpoint for single-unit runs.
- **Robust semantic disambiguation across chunks**: embedding- and symbolic-aware entity alignment during aggregation (`EntityAligner`, connected-component clustering, `skos:altName` handling) with improved cross-unit identity resolution.
- **RDF 1.2 provenance support**: quoted-triple / reification syntax via `pyoxigraph`; provenance and alignment triples are split into a side artifact during ontology normalization; optional `strip_provenance` on `/process` and `/process_unit` omits reification scaffolding from API Turtle output.
- **Enhanced ontology update consolidation**: normalize → consolidate → structural check → consistency critic pipeline replaces the legacy sublimation stage; optional post-normalization consolidation pass via `ENABLE_ONTOLOGY_CONSOLIDATION`.
- **JSON-LD as LLM wire format**: `LLM_GRAPH_FORMAT=jsonld` emits compact JSON-LD (`@context` + `@graph`) for graph payloads while keeping canonical domain models (`GraphUpdate`, critique reports, etc.) at runtime; Turtle remains the default.
- Per-unit **ontology catalog selection** (`select_ontology_catalog`) with optional `ontology_selection_user_instruction`.
- **Ontology context modes**: `selected_single_ontology`, `selected_vector_search_ontology` (Qdrant stitched ensemble), and `fixed_single_ontology` (`ONTOLOGY_CONTEXT_FIXED_ONTOLOGY_ID`).
- **Qdrant vector retrieval** with dual-vector + BM25 hybrid fusion, patch-retrieval scoring/MMR caps (`ONTOLOGY_PATCH_*`), and induced-subgraph triple budgets (`QDRANT_INDUCED_SUBGRAPH_*`).
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
