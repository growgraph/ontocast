# LLM Caching

OntoCast includes automatic LLM response caching to improve performance, reduce API costs, and enable offline testing capabilities.

---

## Overview

The LLM caching system automatically caches responses from language model providers, ensuring that identical queries return cached results instead of making new API calls. This provides several benefits:

- **Performance**: Cached responses return instantly
- **Cost Reduction**: Avoids duplicate API calls
- **Offline Testing**: Tests can run without API access
- **Transparency**: No configuration required - works automatically

---

## Shared Caching Architecture

OntoCast uses a **shared caching architecture** where:

- **Single Cacher Instance**: One `Cacher` object manages all caching for all tools
- **Tool-Specific Subdirectories**: Each tool gets its own subdirectory within the shared cache
- **Dependency Injection**: Tools receive the shared Cacher instance through their constructors
- **Organized Storage**: Cache files are organized by tool type (llm/, converter/, chunker/)

### Benefits

1. **Memory Efficiency**: Single cache instance instead of multiple
2. **Consistent Configuration**: All tools use the same cache directory settings
3. **Centralized Management**: Easy to clear, monitor, and manage all caches
4. **Better Organization**: Clear separation of cache files by tool type

---

## How It Works

### Shared Caching

OntoCast uses a shared caching system where all tools share a single Cacher instance:

```python
from ontocast.tool.llm import LLMTool
from ontocast.config import LLMConfig
from ontocast.tool.cache import Cacher

# Create shared cache instance
shared_cache = Cacher()

# Create LLM tool with shared cache
llm_config = LLMConfig(
    provider="openai",
    model_name="gpt-4o-mini",
    api_key="your-api-key"
)

llm_tool = LLMTool.create(config=llm_config, cache=shared_cache)

# First call - hits API and caches response
response1 = llm_tool("What is the capital of France?")

# Second call - returns cached response instantly
response2 = llm_tool("What is the capital of France?")
```

### Cache Key Generation

Cache keys are generated based on:
- LLM provider and model
- Prompt text
- Temperature and other parameters
- API endpoint URL

This ensures that different configurations or parameters result in separate cache entries.

---

## Cache Locations

### Default Locations

The system automatically selects appropriate cache directories:

- **Tests**: `.test_cache/llm/` in the current working directory
- **Windows**: `%USERPROFILE%\AppData\Local\ontocast\llm\`
- **Unix/Linux**: `~/.cache/ontocast/llm/` (or `$XDG_CACHE_HOME/ontocast/llm/`)

### Environment Variables

Set the cache directory via environment variables:

```bash
# OntoCast cache directory (recommended)
export ONTOCAST_CACHE_DIR=/path/to/custom/cache

# Or use XDG cache home (affects all XDG-compliant applications)
export XDG_CACHE_HOME=/path/to/custom/cache
```

### CLI Parameter

Specify cache directory via command line:

```bash
ontocast --env-path .env --working-directory ./work --cache-dir /custom/cache/path
```

---

## Cache Management

### Cache Structure

The cache directory contains organized subdirectories:

```
cache_dir/
├── openai/
│   ├── gpt-4o-mini/
│   │   ├── prompt_hash_1.json
│   │   └── prompt_hash_2.json
│   └── gpt-4/
│       └── prompt_hash_3.json
└── ollama/
    └── llama2/
        └── prompt_hash_4.json
```

### Cache Files

Each cached response is stored as a JSON file containing:
- Original prompt and parameters
- Response content
- Metadata (timestamp, model info)
- Cache key hash

---

## Testing with Caching

### Offline Testing

Cached responses enable offline testing:

```python
# First run - with API access
pytest test_llm_functionality.py

# Subsequent runs - offline (uses cached responses)
pytest test_llm_functionality.py
```

### Test Isolation

Each test run uses a separate cache directory (`.test_cache/llm/`) to avoid interference between tests.

---

## Performance Benefits

### Speed Improvements

- **First Call**: Normal API response time
- **Cached Calls**: Near-instant response (< 1ms)
- **Batch Processing**: Significant speedup for repeated operations

### Cost Savings

- **Development**: Avoid repeated API calls during development
- **Testing**: Run tests without API costs
- **Production**: Reduce API usage for common queries

---

## Best Practices

### Development

1. **Use Default Locations**: Let the system choose appropriate cache directories
2. **Version Control**: Add cache directories to `.gitignore`
3. **Cleanup**: Periodically clean old cache files

### Production

1. **Persistent Storage**: Use persistent cache directories
2. **Monitoring**: Monitor cache hit rates
3. **Maintenance**: Implement cache cleanup strategies

### Testing

1. **Isolated Caches**: Each test run gets its own cache
2. **Deterministic**: Cached responses ensure consistent test results
3. **Offline Capability**: Tests can run without API access

---

## Troubleshooting

### Common Issues

1. **Cache Not Working**: Check directory permissions
2. **Stale Responses**: Clear cache directory
3. **Disk Space**: Monitor cache directory size

### Debug Cache

```python
from ontocast.tool.llm import LLMTool

# Check cache directory
llm_tool = LLMTool.create(config=llm_config)
print(f"Cache directory: {llm_tool.cache.tool_cache_dir}")

# List cached files
cache_files = list(llm_tool.cache.tool_cache_dir.glob("**/*.json"))
print(f"Cached responses: {len(cache_files)}")
```

### Clear Cache

```python
import shutil
from pathlib import Path

# Clear entire cache
cache_dir = Path.home() / ".cache" / "ontocast" / "llm"
if cache_dir.exists():
    shutil.rmtree(cache_dir)
    print("Cache cleared!")
```

---

## Advanced Usage

### Custom Cache Implementation

For advanced use cases, you can implement custom caching by extending the Cacher class:

```python
from ontocast.tool.llm import LLMTool
from ontocast.tool.cache import Cacher
from pathlib import Path

class CustomLLMTool(LLMTool):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        # Override with custom cache
        self.cache = Cacher(subdirectory="llm", cache_dir=Path("/custom/cache"))
```

### Cache Statistics

```python
from ontocast.tool.llm import LLMTool

# Get cache statistics
llm_tool = LLMTool.create(config=llm_config)
stats = llm_tool.cache.get_cache_stats()
print(f"Cache stats: {stats}")
```

---

## Integration with Other Tools

### ToolBox Integration

Caching works seamlessly with the ToolBox through a shared Cacher instance:

```python
from ontocast.toolbox import ToolBox
from ontocast.config import Config

# ToolBox automatically creates and uses a shared Cacher
config = Config()
tools = ToolBox(config)

# All tools (LLM, Converter, Chunker) share the same cache instance
result = tools.llm("Process this document")
converted = tools.converter(document_file)
chunks = tools.chunker(text)
```

### Server Integration

The server automatically uses caching for all LLM operations:

```bash
# Start server with automatic caching
ontocast --env-path .env --working-directory /data/working
```

---

## Security Considerations

### Sensitive Data

- Cache files may contain sensitive prompt data
- Ensure proper file permissions on cache directories
- Consider encryption for sensitive deployments

### Access Control

- Restrict access to cache directories
- Use appropriate file system permissions
- Consider network security for shared cache directories


---

## Converter and Chunker Caching

In addition to LLM response caching, OntoCast also includes caching for document conversion and text chunking operations. This helps avoid redundant processing when the same documents or text are processed multiple times.

### Converter Caching

The `ConverterTool` automatically caches document conversion results based on the input file content. This means:

- **PDF files**: If the same PDF is processed multiple times, the conversion to markdown is cached
- **Other documents**: PowerPoint, Word documents, etc. are also cached after conversion
- **Plain text**: Text input is not cached as it doesn't require conversion

### Chunker Caching

The `ChunkerTool` caches chunking results based on:
- **Input text content**: The exact text being chunked
- **Chunking configuration**: All chunking parameters (max_size, min_size, model, etc.)
- **Chunking mode**: Whether semantic or naive chunking is used

This ensures that identical text with identical chunking parameters will return cached results.

### Cache Organization

Caching is organized in subdirectories:

```
~/.cache/ontocast/
├── llm/           # LLM response cache
├── converter/     # Document conversion cache
└── chunker/       # Text chunking cache
```

### Cache Benefits

1. **Faster Processing**: Repeated operations return instantly from cache
2. **Cost Reduction**: Avoids redundant LLM API calls and processing
3. **Consistency**: Identical inputs always produce identical outputs
4. **Offline Capability**: Cached operations work without API access

### Cache Management

You can access cache statistics and management through the tool instances:

```python
from ontocast.tool.converter import ConverterTool
from ontocast.tool.chunk.chunker import ChunkerTool

# Get cache statistics
converter = ConverterTool()
stats = converter.cache.get_cache_stats()
print(f"Converter cache: {stats['total_files']} files, {stats['total_size_bytes']} bytes")

# Clear cache if needed
converter.cache.clear()

# Chunker cache management
chunker = ChunkerTool()
chunker.cache.clear()  # Clear chunker cache
```

### Custom Cache Directories

You can specify custom cache directories in several ways:

#### 1. Environment Variables

```bash
# OntoCast cache directory (recommended)
export ONTOCAST_CACHE_DIR=/custom/cache/path

# Or use XDG cache home (affects all XDG-compliant applications)
export XDG_CACHE_HOME=/custom/cache/path
```

#### 2. CLI Parameter

```bash
ontocast --env-path .env --working-directory ./work --cache-dir /custom/cache/path
```

#### 3. Programmatic Configuration

```python
from ontocast.toolbox import ToolBox
from ontocast.config import Config
from pathlib import Path

# Create config and set cache directory
config = Config()
config.tool_config.path_config.cache_dir = Path("/custom/cache/path")

# Create ToolBox with config (cache directory is automatically used)
tools = ToolBox(config)

# All tools will use the same custom cache directory
result = tools.llm("Process this document")
converted = tools.converter(document_file)
chunks = tools.chunker(text)
```

### Cache Key Generation

Cache keys are generated based on:
- **Content hash**: SHA256 hash of the input content
- **Configuration**: All relevant parameters that affect the output
- **Tool-specific parameters**: Model names, chunking modes, etc.

This ensures that different configurations produce different cache entries, even for the same input content.

### Best Practices

1. **Let caching work automatically**: No configuration needed for basic usage
2. **Monitor cache size**: Check cache statistics periodically
3. **Clear cache when needed**: If you change tool configurations significantly
4. **Use custom directories**: For testing or specific deployment scenarios
5. **Cache persistence**: Caches persist between runs for maximum benefit

