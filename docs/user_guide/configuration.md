# Configuration System

OntoCast uses a hierarchical configuration system that provides type safety, environment variable support, and clear separation of concerns.

---

## Overview

The configuration system is built on Pydantic BaseSettings and provides:

- **Type Safety**: All configuration values are properly typed
- **Environment Variables**: Automatic loading from `.env` files and environment
- **Hierarchical Structure**: Clear separation between tool and server configuration
- **Validation**: Automatic validation of configuration values
- **Documentation**: Self-documenting configuration classes

---

## Configuration Structure

```python
Config
├── tools: ToolConfig          # Tool-related configuration
│   ├── llm: LLMConfig         # LLM settings
│   ├── neo4j: Neo4jConfig     # Neo4j triple store
│   ├── fuseki: FusekiConfig   # Fuseki triple store
│   ├── domain: DomainConfig   # Domain and URI settings
│   └── paths: PathConfig       # Path settings
└── server: ServerConfig       # Server-related configuration
    ├── port: int              # Server port
    ├── recursion_limit: int   # Workflow recursion limit
    ├── estimated_chunks: int  # Estimated number of chunks
    ├── max_visits: int        # Maximum visits per node
    ├── skip_ontology_development: bool  # Skip ontology critique
    ├── skip_facts_rendering: bool  # Skip facts rendering
    └── ontology_max_triples: int | None  # Maximum triples in ontology graph
```

---

## Environment Variables

### LLM Configuration

# LLM Caching
ONTOCAST_CACHE_DIR=/path/to/cache         # Custom cache directory (optional)


```bash
# LLM Provider and Model
LLM_PROVIDER=openai                    # or "ollama"
LLM_MODEL_NAME=gpt-4o-mini            # Model name
LLM_TEMPERATURE=0.1                    # Temperature setting
LLM_API_KEY=your-api-key-here         # API key (replaces OPENAI_API_KEY)
LLM_BASE_URL=http://localhost:11434    # Base URL for Ollama
```

### Server Configuration

```bash
# Server Settings
PORT=8999                              # Server port
RECURSION_LIMIT=1000                   # Workflow recursion limit
ESTIMATED_CHUNKS=30                    # Estimated number of chunks
MAX_VISITS=3                           # Maximum visits per node
SKIP_ONTOLOGY_DEVELOPMENT=false        # Skip ontology critique step
SKIP_FACTS_RENDERING=false             # Skip facts extraction and go straight to serialization
ONTOLOGY_MAX_TRIPLES=10000             # Maximum triples allowed in ontology graph (set empty for unlimited)
```

### Backend Configuration

Backend selection is **automatically inferred** from available configuration - no explicit flags needed:

```bash
# Fuseki Backend (auto-detected if both provided)
FUSEKI_URI=http://localhost:3032/test
FUSEKI_AUTH=admin:password

# Neo4j Backend (auto-detected if both provided)  
NEO4J_URI=bolt://localhost:7689
NEO4J_AUTH=neo4j:password

# Filesystem Triple Store (auto-detected if both provided)
ONTOCAST_WORKING_DIRECTORY=/path/to/working
ONTOCAST_ONTOLOGY_DIRECTORY=/path/to/ontologies

# Filesystem Manager (auto-detected if working directory provided)
# Can be combined with Fuseki or Neo4j for debugging
ONTOCAST_WORKING_DIRECTORY=/path/to/working
```

### Path Configuration

```bash
# Path Settings (all configured via .env file)
ONTOCAST_WORKING_DIRECTORY=/path/to/working     # Working directory (required for filesystem backends)
ONTOCAST_ONTOLOGY_DIRECTORY=/path/to/ontologies # Ontology files directory (required for filesystem backends)
ONTOCAST_CACHE_DIR=/path/to/cache               # Cache directory (optional)
```

**Note:** All paths are configured via the `.env` file - no CLI overrides available.

### Triple Store Configuration

```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7689        # Neo4j URI
NEO4J_AUTH=neo4j:password             # Neo4j authentication

# Fuseki Configuration
FUSEKI_URI=http://localhost:3032/test # Fuseki URI
FUSEKI_AUTH=admin:password            # Fuseki authentication
FUSEKI_DATASET=dataset_name           # Fuseki dataset name
```

### Domain Configuration

```bash
# Domain Settings
CURRENT_DOMAIN=https://example.com     # Domain for URI generation
```

---


## LLM Caching

OntoCast includes automatic LLM response caching to improve performance and reduce API costs. Caching is enabled by default and requires no configuration.

### Default Cache Locations

- **Tests**: `.test_cache/llm/` in the current working directory
- **Windows**: `%USERPROFILE%\AppData\Local\ontocast\llm\`
- **Unix/Linux**: `~/.cache/ontocast/llm/` (or `$XDG_CACHE_HOME/ontocast/llm/`)

### Environment Variables

```bash
# Optional: Custom cache directory
ONTOCAST_CACHE_DIR=/path/to/custom/cache
```

### Benefits

- **Faster Execution**: Repeated queries return cached responses instantly
- **Cost Reduction**: Identical requests don't hit the LLM API
- **Offline Capability**: Tests can run without API access if responses are cached
- **Transparent**: No configuration required - works automatically

### Custom Cache Directory

```python
from pathlib import Path
from ontocast.tool.llm import LLMTool
from ontocast.config import LLMConfig

# Create LLM configuration
llm_config = LLMConfig(
    provider="openai",
    model_name="gpt-4o-mini",
    api_key="your-api-key"
)

# Cache directory is managed automatically by Cacher
llm_tool = LLMTool.create(
    config=llm_config
)
```


## Usage Examples

### Basic Configuration

```python
from ontocast.config import Config

# Load configuration from environment
config = Config()

# Access configuration sections
tool_config = config.get_tool_config()
server_config = config.get_server_config()

# Access specific settings
llm_provider = config.tool_config.llm_config.provider
working_dir = config.tool_config.path_config.working_directory
server_port = config.server.port
```

### ToolBox Initialization

```python
from ontocast.config import Config
from ontocast.toolbox import ToolBox

# Create configuration
config = Config()

# Initialize ToolBox with configuration
tools = ToolBox(config)

# ToolBox automatically uses the configuration
print(f"LLM Provider: {tools.llm_provider}")
print(f"Model: {tools.llm.config.model_name}")
```

### Server Configuration

```python
from ontocast.config import Config

config = Config()

# Get server configuration
server_config = config.get_server_config()

# Access server settings
port = server_config["port"]
max_visits = server_config["max_visits"]
recursion_limit = server_config["recursion_limit"]
```

---

## Configuration Classes

### LLMConfig

```python
class LLMConfig(BaseSettings):
    provider: str = "openai"                    # LLM provider
    model_name: str = "gpt-4o-mini"            # Model name
    temperature: float = 0.1                    # Temperature
    base_url: str | None = None                 # Base URL
    api_key: str | None = None                  # API key
```

### ServerConfig

```python
class ServerConfig(BaseSettings):
    port: int = 8999                           # Server port
    recursion_limit: int = 1000                 # Recursion limit
    estimated_chunks: int = 30                  # Estimated chunks
    max_visits: int = 3                        # Max visits
    skip_ontology_development: bool = False     # Skip critique
    skip_facts_rendering: bool = False         # Skip facts rendering
    ontology_max_triples: int | None = 10000    # Max triples in ontology graph
```


### PathConfig

```python
class PathConfig(BaseSettings):
    working_directory: Path | None = None       # Working directory (required for filesystem backends)
    ontology_directory: Path | None = None      # Ontology directory (required for filesystem backends)
    cache_dir: Path | None = None               # Cache directory (optional)
```

**Note:** All paths are configured via environment variables in the `.env` file - no CLI overrides available.

---

## Validation

The configuration system includes automatic validation:

```python
from ontocast.config import Config

try:
    config = Config()
    config.validate_llm_config()  # Validate LLM configuration
    print("Configuration is valid!")
except ValueError as e:
    print(f"Configuration error: {e}")
```

### Common Validation Errors

- **Missing API Key**: `LLM_API_KEY` environment variable is required for OpenAI
- **Invalid Provider**: LLM provider must be "openai" or "ollama"
- **Missing Working Directory**: `ONTOCAST_WORKING_DIRECTORY` must be set when filesystem backends are enabled
- **Missing Ontology Directory**: `ONTOCAST_ONTOLOGY_DIRECTORY` must be set when filesystem backends are enabled
- **No Backend Available**: At least one backend (triple store or filesystem) must be configured
- **Invalid Paths**: Paths must exist and be accessible

---

## Migration from Previous Versions

### Environment Variable Changes

```bash
# Old (deprecated)
LLM_API_KEY=your-key

# New (current)
LLM_API_KEY=your-key
```

### Configuration Usage Changes

```python
# Old way (deprecated)
from ontocast.config import config

llm_provider = config.tool_config.llm_config.provider

# New way (current)
from ontocast.config import Config

config = Config()
llm_provider = config.tool_config.llm_config.provider
```

### ToolBox Initialization Changes

```python
# Old way (deprecated)
tools = ToolBox(
    llm_provider="openai",
    model_name="gpt-4",
    # ... many parameters
)

# New way (current)
tools = ToolBox(config)
```

---

## Best Practices

1. **Use Environment Variables**: Store sensitive data in `.env` files
2. **Validate Configuration**: Always validate configuration before use
3. **Type Safety**: Use the typed configuration classes
4. **Documentation**: Document your configuration in your project
5. **Testing**: Test configuration loading in your tests

### Example .env File

```bash
# LLM Configuration
LLM_PROVIDER=openai
LLM_API_KEY=your-openai-api-key
LLM_MODEL_NAME=gpt-4o-mini
LLM_TEMPERATURE=0.1

# Server Configuration
PORT=8999
MAX_VISITS=3
RECURSION_LIMIT=1000
ESTIMATED_CHUNKS=30
ONTOLOGY_MAX_TRIPLES=10000

# Backend Configuration (auto-detected)
FUSEKI_URI=http://localhost:3032/test
FUSEKI_AUTH=admin:password
ONTOCAST_WORKING_DIRECTORY=/path/to/working

# Path Configuration
ONTOCAST_WORKING_DIRECTORY=/path/to/working
ONTOCAST_ONTOLOGY_DIRECTORY=/path/to/ontologies
ONTOCAST_CACHE_DIR=/path/to/cache

# Triple Store Configuration (Optional)
FUSEKI_URI=http://localhost:3032/test
FUSEKI_AUTH=admin:password
FUSEKI_DATASET=ontocast

# Domain Configuration
CURRENT_DOMAIN=https://example.com
```

---

## Troubleshooting

### Common Issues

1. **Configuration Not Loading**: Check `.env` file location and format
2. **Type Errors**: Ensure environment variables match expected types
3. **Missing Variables**: Check required environment variables are set
4. **Path Issues**: Verify paths exist and are accessible

### Debug Configuration

```python
from ontocast.config import Config

# Load and inspect configuration
config = Config()

print("Configuration loaded successfully!")
print(f"LLM Provider: {config.tool_config.llm_config.provider}")
print(f"Working Directory: {config.tool_config.path_config.working_directory}")
print(f"Server Port: {config.server.port}")

# Validate configuration
try:
    config.validate_llm_config()
    print("LLM configuration is valid!")
except ValueError as e:
    print(f"LLM configuration error: {e}")
```
