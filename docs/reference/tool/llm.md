# `ontocast.tool.llm`

::: ontocast.tool.llm

## LLM Caching

The `LLMTool` includes automatic response caching to improve performance and reduce API costs. Caching is enabled by default and requires no configuration.

### Default Cache Directory

The system automatically selects an appropriate cache directory:

- **Tests**: `.test_cache/llm/` in the current working directory
- **Windows**: `%USERPROFILE%\AppData\Local\ontocast\llm\`
- **Unix/Linux**: `~/.cache/ontocast/llm/` (or `$XDG_CACHE_HOME/ontocast/llm/`)

### Custom Cache Directory

```python
from pathlib import Path
from ontocast.tool.llm import LLMTool

# Specify custom cache directory
llm_tool = LLMTool.create(
    config=llm_config
)
```

### Environment Variable

```bash
export ONTOCAST_CACHE_DIR=/path/to/custom/cache
```

### Benefits

- **Performance**: Cached responses return instantly
- **Cost Reduction**: Avoids duplicate API calls
- **Offline Testing**: Tests can run without API access
- **Transparent**: No configuration required

For detailed information about LLM caching, see the [LLM Caching Guide](../user_guide/llm_caching.md).

