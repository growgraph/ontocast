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
    ├── clean: bool            # Clean triple store on startup
    └── skip_ontology_development: bool  # Skip ontology critique
```

---

## Environment Variables

### LLM Configuration

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
CLEAN=false                            # Clean triple store on startup
SKIP_ONTOLOGY_DEVELOPMENT=false        # Skip ontology critique step
```

### Path Configuration

```bash
# Path Settings
WORKING_DIRECTORY=/path/to/working     # Working directory (required)
ONTOLOGY_DIRECTORY=/path/to/ontologies # Ontology files directory (optional)
```

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
llm_provider = config.tools.llm.provider
working_dir = config.tools.paths.working_directory
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
    clean: bool = False                        # Clean startup
    skip_ontology_development: bool = False     # Skip critique
```

### PathConfig

```python
class PathConfig(BaseSettings):
    working_directory: Path | None = None       # Working directory
    ontology_directory: Path | None = None        # Ontology directory
```

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
- **Missing Working Directory**: `WORKING_DIRECTORY` must be set
- **Invalid Paths**: Paths must exist and be accessible

---

## Migration from Previous Versions

### Environment Variable Changes

```bash
# Old (deprecated)
OPENAI_API_KEY=your-key

# New (current)
LLM_API_KEY=your-key
```

### Configuration Usage Changes

```python
# Old way (deprecated)
from ontocast.config import config
llm_provider = config.llm.provider

# New way (current)
from ontocast.config import Config
config = Config()
llm_provider = config.tools.llm.provider
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

# Path Configuration
WORKING_DIRECTORY=/path/to/working
ONTOLOGY_DIRECTORY=/path/to/ontologies

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
print(f"LLM Provider: {config.tools.llm.provider}")
print(f"Working Directory: {config.tools.paths.working_directory}")
print(f"Server Port: {config.server.port}")

# Validate configuration
try:
    config.validate_llm_config()
    print("LLM configuration is valid!")
except ValueError as e:
    print(f"LLM configuration error: {e}")
```
