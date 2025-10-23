# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Automatic LLM response caching for improved performance and cost reduction
- Platform-aware default cache directory selection (avoids /tmp)
- Transparent caching with no configuration required
- Custom cache directory support via `LLM_CACHE_DIR` environment variable

- CLI parameter `--skip-ontology-critique` to skip ontology critique step
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
- Custom cache directory support via `LLM_CACHE_DIR` environment variable

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
# Custom cache directory (optional)
export LLM_CACHE_DIR=/path/to/custom/cache
```

```bash
# Skip ontology critique step
ontocast serve --skip-ontology-critique

# Or set environment variable
export SKIP_ONTOLOGY_DEVELOPMENT=true
ontocast serve
```
