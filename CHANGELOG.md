# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


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

### Environment Variables
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
