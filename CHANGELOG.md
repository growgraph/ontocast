# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `updated_at` timestamp field to Ontology properties for tracking last update time
- Automatic semantic versioning with intelligent increment analysis (MAJOR/MINOR/PATCH)
- Version analysis based on ontology changes (classes, properties, instances)
- Hash-based versioning with parent hashes for git-style lineage tracking
- `mark_as_updated()` method in Ontology class for version and timestamp management
- `sync_properties_to_graph()` method updates version and `updated_at` in RDF graph
- `versioned_iri` property on Ontology class for storing multiple versions in triple stores
- URL encoding for versioned IRIs in Fuseki to preserve `#` characters in URIs
- Multiple ontology version support in Fuseki triple store (versions stored as separate named graphs)
- Automatic ontology synchronization from filesystem to triple store during initialization
- `skip_facts_rendering` parameter to skip facts extraction and go straight to serialization
- Separated `aggregate` and `serialize` into distinct workflow nodes for better control
- Added `serialize` node to the workflow graph
- API support for `skip_facts_rendering` and `skip_ontology_development` as query parameters
- **GraphUpdate system**: Structured SPARQL operations (insert/delete) for token-efficient graph updates
- `GraphUpdate` model with `TripleOp` operations for incremental graph modifications
- `render_ontology_update()` and `render_facts_update()` functions using GraphUpdate instead of full TTL
- Automatic SPARQL query generation from GraphUpdate operations
- Budget tracking system integrated into AgentState with usage statistics
- Triple generation metrics (ontology and facts tracking)
- Budget tracker summary reports at end of processing
- Clean dependency injection for LLM budget tracker
- Shared caching architecture with single Cacher instance
- ToolCacher wrapper for tool-specific cache access
- Environment variable `ONTOCAST_CACHE_DIR` for cache directory configuration
- New `serialize()` method in triple store managers as primary interface for storing Ontology and RDFGraph objects
- `ONTOLOGY_MAX_TRIPLES` configuration parameter to limit ontology graph size (default: 10000)
- Automatic limit checking in `render_updated_graph()` to prevent unbounded ontology growth
- Limit enforcement in `sublimate_ontology()` for direct graph modifications
- Updates exceeding the limit are silently skipped with warning logs

### Changed
- **BREAKING**: `serialize()` method is now the primary interface for storing data in triple stores
- **BREAKING**: `serialize()` method now accepts `Ontology | RDFGraph` objects instead of raw `Graph` objects
- **BREAKING**: `serialize_graph()` method signature changed to use `**kwargs` for implementation-specific parameters
- All triple store managers now implement both `serialize()` and `serialize_graph()` methods
- **BREAKING**: Environment variables now use `ONTOCAST_` prefix:
  - `WORKING_DIRECTORY` → `ONTOCAST_WORKING_DIRECTORY`
  - `ONTOLOGY_DIRECTORY` → `ONTOCAST_ONTOLOGY_DIRECTORY`
  - `LLM_CACHE_DIR` → `ONTOCAST_CACHE_DIR`
- **BREAKING**: Ontology and facts rendering now use GraphUpdate (SPARQL operations) instead of full TTL generation
- LLM now outputs structured `GraphUpdate` objects with `TripleOp` operations, dramatically reducing token usage
- Ontology version increment now analyzes changes to determine appropriate version bump
- Version updates happen once at end of processing (in `serialize`)
- Refactored LLM tool to accept budget tracker via dependency injection
- Removed global LLMBudgetTracker in favor of AgentState-contained tracker
- Updated all agent functions to use clean injection pattern
- Improved Python 3.12 typing with `|` union syntax instead of `Union`

### Removed
- Global budget tracker state management
- Manual budget tracker update calls in agent functions
- `set_budget_tracker()` and `get_budget_tracker()` functions

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
