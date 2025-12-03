# OntoCast <img src="https://raw.githubusercontent.com/growgraph/ontocast/refs/heads/main/docs/assets/favicon.ico" alt="Agentic Ontology Triplecast logo" style="height: 32px; width:32px;"/>

### Agentic ontology-assisted framework for semantic triple extraction

![Python](https://img.shields.io/badge/python-3.12-blue.svg) 
[![PyPI version](https://badge.fury.io/py/ontocast.svg)](https://badge.fury.io/py/ontocast)
[![PyPI Downloads](https://static.pepy.tech/badge/ontocast)](https://pepy.tech/projects/ontocast)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![pre-commit](https://github.com/growgraph/ontocast/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/growgraph/ontocast/actions/workflows/pre-commit.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17796467.svg)](https://doi.org/10.5281/zenodo.17796467)

---

## Overview

OntoCast is a framework for extracting semantic triples (creating a Knowledge Graph) from documents using an agentic, ontology-driven approach. It combines ontology management, natural language processing, and knowledge graph serialization to turn unstructured text into structured, queryable data.

---

## Key Features

- **Ontology-Guided Extraction**: Ensures semantic consistency and co-evolves ontologies
- **Entity Disambiguation**: Resolves references across document chunks
- **Multi-Format Support**: Handles text, JSON, PDF, and Markdown
- **Semantic Chunking**: Splits text based on semantic similarity
- **MCP Compatibility**: Implements Model Control Protocol endpoints
- **RDF Output**: Produces standardized RDF/Turtle
- **Triple Store Integration**: Supports Neo4j (n10s) and Apache Fuseki
- **Hierarchical Configuration**: Type-safe configuration system with environment variable support
- **CLI Parameters**: Flexible command-line interface with `--skip-ontology-critique` option
- **Automatic LLM Caching**: Built-in response caching for improved performance and cost reduction
- **GraphUpdate Operations**: Token-efficient SPARQL-based updates instead of full graph regeneration
- **Budget Tracking**: Comprehensive tracking of LLM usage and triple generation metrics
- **Ontology Versioning**: Automatic semantic versioning with hash-based lineage tracking

---

## Applications

OntoCast can be used for:

- **Knowledge Graph Construction**: Build domain-specific or general-purpose knowledge graphs from documents
- **Semantic Search**: Power search and retrieval with structured triples
- **GraphRAG**: Enable retrieval-augmented generation over knowledge graphs (e.g., with LLMs)
- **Ontology Management**: Automate ontology creation, validation, and refinement
- **Data Integration**: Unify data from diverse sources into a semantic graph

---

## Installation

```sh
uv add ontocast 
# or
pip install ontocast
```

---

## Quick Start

### 1. Configuration

Create a `.env` file with your configuration:

```bash
# LLM Configuration
LLM_PROVIDER=openai
LLM_API_KEY=your-api-key-here
LLM_MODEL_NAME=gpt-4o-mini
LLM_TEMPERATURE=0.1

# Server Configuration
PORT=8999
MAX_VISITS=3
RECURSION_LIMIT=1000
ESTIMATED_CHUNKS=30
ONTOLOGY_MAX_TRIPLES=10000

# Path Configuration
ONTOCAST_WORKING_DIRECTORY=/path/to/working
ONTOCAST_ONTOLOGY_DIRECTORY=/path/to/ontologies
ONTOCAST_CACHE_DIR=/path/to/cache

# Optional: Triple Store Configuration
FUSEKI_URI=http://localhost:3032/test
FUSEKI_AUTH=admin:password
FUSEKI_DATASET=ontocast

# Optional: Skip ontology critique
SKIP_ONTOLOGY_DEVELOPMENT=false
# Optional: Maximum triples allowed in ontology graph (set empty for unlimited)
ONTOLOGY_MAX_TRIPLES=10000
```

### 2. Start Server

```bash
ontocast \
    --env-path .env \
    --working-directory /path/to/working \
    --ontology-directory /path/to/ontologies
```

### 3. Process Documents

```bash
curl -X POST http://localhost:8999/process -F "file=@document.pdf"
```

### 4. API Endpoints

The OntoCast server provides the following endpoints:

- **POST /process**: Process documents and extract semantic triples
  ```bash
  curl -X POST http://localhost:8999/process -F "file=@document.pdf"
  ```

- **POST /flush**: Flush/clean triple store data
  ```bash
  # Clean all datasets (Fuseki) or entire database (Neo4j)
  curl -X POST http://localhost:8999/flush
  
  # Clean specific Fuseki dataset
  curl -X POST "http://localhost:8999/flush?dataset=my_dataset"
  ```
  **Note:** For Fuseki, you can specify a `dataset` query parameter to clean a specific dataset. If omitted, all datasets are cleaned. For Neo4j, the `dataset` parameter is ignored and all data is deleted.

- **GET /health**: Health check endpoint
- **GET /info**: Service information endpoint

---


## LLM Caching

OntoCast includes automatic LLM response caching to improve performance and reduce API costs. Caching is enabled by default and requires no configuration.

### Cache Locations

- **Tests**: `.test_cache/llm/` in the current working directory
- **Windows**: `%USERPROFILE%\AppData\Local\ontocast\llm\`
- **Unix/Linux**: `~/.cache/ontocast/llm/` (or `$XDG_CACHE_HOME/ontocast/llm/`)

### Benefits

- **Faster Execution**: Repeated queries return cached responses instantly
- **Cost Reduction**: Identical requests don't hit the LLM API
- **Offline Capability**: Tests can run without API access if responses are cached
- **Transparent**: No configuration required - works automatically

### Custom Cache Directory

If you need to specify a custom cache directory:

```python
from pathlib import Path
from ontocast.tool.llm import LLMTool

# Cache directory is managed automatically by Cacher
llm_tool = LLMTool.create(
    config=llm_config
)
```


## Configuration System

OntoCast uses a hierarchical configuration system built on Pydantic BaseSettings:

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `LLM_API_KEY` | API key for LLM provider | - | Yes |
| `LLM_PROVIDER` | LLM provider (openai, ollama) | openai | No |
| `LLM_MODEL_NAME` | Model name | gpt-4o-mini | No |
| `LLM_TEMPERATURE` | Temperature setting | 0.1 | No |
| `ONTOCAST_WORKING_DIRECTORY` | Working directory path | - | Yes |
| `ONTOCAST_ONTOLOGY_DIRECTORY` | Ontology files directory | - | No |
| `PORT` | Server port | 8999 | No |
| `MAX_VISITS` | Maximum visits per node | 3 | No |
| `SKIP_ONTOLOGY_DEVELOPMENT` | Skip ontology critique | false | No |
| `ONTOLOGY_MAX_TRIPLES` | Maximum triples allowed in ontology graph | 10000 | No |
| `SKIP_FACTS_RENDERING` | Skip facts rendering and go straight to aggregation | false | No |
| `ONTOCAST_CACHE_DIR` | Custom cache directory for LLM responses | Platform default | No |

### Triple Store Configuration

```bash
# Fuseki (Preferred)
FUSEKI_URI=http://localhost:3032/test
FUSEKI_AUTH=admin:password
FUSEKI_DATASET=dataset_name

# Neo4j (Alternative)
NEO4J_URI=bolt://localhost:7689
NEO4J_AUTH=neo4j:password
```

### CLI Parameters

```bash
# Skip ontology critique step
ontocast --skip-ontology-critique

# Process only first N chunks (for testing)
ontocast --head-chunks 5

```

---

## Triple Store Setup

OntoCast supports multiple triple store backends with automatic fallback:

1. **Apache Fuseki** (Recommended) - Native RDF with SPARQL support
2. **Neo4j with n10s** - Graph database with RDF capabilities  
3. **Filesystem** (Fallback) - Local file-based storage

When multiple triple stores are configured, **Fuseki is preferred over Neo4j**.

### Quick Setup with Docker

**Fuseki:**
```bash
cd docker/fuseki
cp .env.example .env
# Edit .env with your values
docker compose --env-file .env fuseki up -d
```

**Neo4j:**
```bash
cd docker/neo4j
cp .env.example .env
# Edit .env with your values
docker compose --env-file .env neo4j up -d
```

See [Triple Store Setup](docs/user_guide/triple_stores.md) for detailed instructions.

---

## Documentation

- [Quick Start Guide](docs/getting_started/quickstart.md) - Get started quickly
- [Configuration System](docs/user_guide/configuration.md) - Detailed configuration guide
- [Triple Store Setup](docs/user_guide/triple_stores.md) - Triple store configuration
- [User Guide](docs/user_guide/concepts.md) - Core concepts and workflow
- [API Reference](docs/reference/onto.md) - Detailed API documentation

---

## Recent Changes

### Ontology Management Improvements

- **Automatic Versioning**: Semantic version increment based on change analysis (MAJOR/MINOR/PATCH)
- **Hash-Based Lineage**: Git-style versioning with parent hashes for tracking ontology evolution
- **Multiple Version Storage**: Versions stored as separate named graphs in Fuseki triple stores
- **Timestamp Tracking**: `updated_at` field tracks when ontology was last modified
- **Smart Version Analysis**: Analyzes ontology changes (classes, properties, instances) to determine appropriate version bump

### GraphUpdate System

- **Token Efficiency**: LLM outputs structured SPARQL operations (insert/delete) instead of full TTL graphs
- **Incremental Updates**: Only changes are generated, dramatically reducing token usage
- **Structured Operations**: TripleOp operations with explicit prefix declarations for precise updates
- **SPARQL Generation**: Automatic conversion of operations to executable SPARQL queries

### Budget Tracking

- **LLM Statistics**: Tracks API calls, characters sent/received for cost monitoring
- **Triple Metrics**: Tracks ontology and facts triples generated per operation
- **Summary Reports**: Budget summaries logged at end of processing
- **Integrated Tracking**: Budget tracker integrated into AgentState for clean dependency injection

### Configuration System Overhaul

- **Hierarchical Configuration**: New `ToolConfig` and `ServerConfig` structure
- **Environment Variables**: Support for `.env` files and environment variables
- **Type Safety**: Full type safety with Python 3.12 union syntax
- **API Key**: Changed from `OPENAI_API_KEY` to `LLM_API_KEY` for consistency
- **Dependency Injection**: Removed global variables, implemented proper DI

### Enhanced Features

- **CLI Parameters**: New `--skip-ontology-critique` and `--skip-facts-rendering` parameters
- **RDFGraph Operations**: Improved `__iadd__` method with proper prefix binding
- **Triple Store Management**: Better separation between filesystem and external stores
- **Serialization Interface**: Unified `serialize()` method for storing Ontology and RDFGraph objects
- **Error Handling**: Improved error handling and validation

See [CHANGELOG.md](CHANGELOG.md) for complete details.

---

## Examples

### Basic Usage

```python
from ontocast.config import Config
from ontocast.toolbox import ToolBox

# Load configuration
config = Config()

# Initialize tools
tools = ToolBox(config)

# Process documents
# ... (use tools for processing)
```

### Server Usage

```bash
# Start server with custom configuration
ontocast \
    --env-path .env \
    --working-directory /data/working \
    --ontology-directory /data/ontologies \
    --skip-ontology-critique \
    --head-chunks 10
```

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/contributing.md) for details.

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/growgraph/ontocast/issues)
- **Discussions**: [GitHub Discussions](https://github.com/growgraph/ontocast/discussions)
