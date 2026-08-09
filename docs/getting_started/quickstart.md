# Quick Start

This guide will help you get started with OntoCast quickly. We'll walk through a simple example of processing a document and viewing the results.

## Prerequisites

- OntoCast installed (see [Installation](installation.md))
- A sample document to process (e.g., a pdf or a markdown file)

## Basic Example

### Query the Server

```bash
curl -X POST http://url:port/process -F "file=@sample.pdf"

curl -X POST http://url:port/process -F "file=@sample.json"
```

`url` would be `localhost` for a locally running server, default port is 8999

### Running a Server

To start an OntoCast server:

```bash
# Backend automatically detected from .env configuration
ontocast serve

# Process specific file (local batch)
ontocast process --input-path ./document.pdf --output-dir ./out

# Process with chunk limit (for testing)
ontocast process --input-path ./document.pdf --head-chunks 5

# Override render/critic retry budget
ontocast process --input-path ./document.pdf --max-visits 2

# Clean-slate vector reindex (embedding-contract / BM25 schema changes)
ontocast serve --wipe-vector-store
```

- Triple store: Fuseki when `FUSEKI_URI` is set; otherwise in-memory pyoxigraph
- Vector store: Qdrant (`QDRANT_URI`) or LanceDB (`LANCEDB_ENABLED=true`), not both
- Paths and directories are configured via `.env`
- `--input-path` takes a single file or a directory (searched recursively). A
  path that does not exist, a file whose extension is not supported, or a
  directory holding no supported input is a hard error with a non-zero exit —
  never a silent no-op

### Configuration

OntoCast uses a hierarchical configuration system with environment variables. Create a `.env` file in your project directory (or copy `.env.example`):

```bash
# Domain configuration (used for URI generation)
CURRENT_DOMAIN=https://example.com
PORT=8999

# LLM Configuration
LLM_PROVIDER=openai
LLM_API_KEY=your-api-key-here
LLM_MODEL_NAME=gpt-4o-mini
LLM_TEMPERATURE=0.0

# Server Configuration
MAX_VISITS=1
BASE_RECURSION_LIMIT=1000
ESTIMATED_CHUNKS=30
RENDER_MODE=ontology_and_facts
ONTOLOGY_MAX_TRIPLES=50000
PARALLEL_WORKERS=4
PARALLEL_FACTS_RETRIES=3
PARALLEL_ONTOLOGY_RETRIES=3
ENABLE_ONTOLOGY_CONSOLIDATION=false

# Paths
ONTOCAST_WORKING_DIRECTORY=/path/to/working/directory
ONTOCAST_ONTOLOGY_DIRECTORY=/path/to/ontology/files
# ONTOCAST_CACHE_DIR=/path/to/cache/directory

# Triple store (optional — omit FUSEKI_URI for in-memory pyoxigraph)
# FUSEKI_URI=http://localhost:3030
# FUSEKI_AUTH=admin/admin

# Optional aggregation controls
AGG_EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
AGG_SIMILARITY_THRESHOLD=0.80

# Optional web-search grounding
WEB_SEARCH_ENABLED=false
WEB_SEARCH_PROVIDER=duckduckgo
WEB_SEARCH_TOP_K=3
```

#### Alternative: Ollama Configuration

```bash
# For Ollama
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434
LLM_MODEL_NAME=granite3.3
```

#### Alternative: Claude / Gemini

```bash
# Anthropic Claude
LLM_PROVIDER=anthropic
LLM_MODEL_NAME=claude-sonnet-4-20250514
LLM_API_KEY=your-anthropic-api-key

# Google Gemini
LLM_PROVIDER=google
LLM_MODEL_NAME=gemini-2.0-flash
LLM_API_KEY=your-google-api-key
```

### CLI Parameters

```bash
# Start the API server (config from .env / environment)
ontocast serve

# Process specific input file; dump TTLs under ./out
ontocast process --input-path /path/to/document.pdf --output-dir ./out

# Process only first 5 chunks (for testing)
ontocast process --input-path /path/to/document.pdf --head-chunks 5

# Override MAX_VISITS for this run
ontocast process --input-path /path/to/document.pdf --max-visits 2

# Drop and recreate the vector partition before reindex
ontocast serve --wipe-vector-store

# Separate facts vs ontology dump folders
ontocast process --input-path ./docs \
  --facts-output-dir ./out/facts \
  --ontology-output-dir ./out/ontologies
```

**Note:** Paths and directories are configured via the `.env` file.

### Receive Results

After processing, the ontology and the facts graph are returned in turtle format

```json
{
    "data": {
        "facts": "# facts in turtle format",
        "ontology": "# ontology in turtle format"
    }
  ...
}
```

## Configuration System

OntoCast uses a hierarchical configuration system:

- **ToolConfig**: Configuration for tools (LLM, triple stores, paths)
- **ServerConfig**: Configuration for server behavior
- **Environment Variables**: Override defaults via `.env` file or environment

### Key Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_API_KEY` | API key for LLM provider | Required for openai / anthropic / google |
| `LLM_PROVIDER` | `openai`, `ollama`, `anthropic`, or `google` | openai |
| `LLM_MODEL_NAME` | Model name | gpt-4o-mini |
| `FUSEKI_URI` + `FUSEKI_AUTH` | Persistent triple store | Omit for in-memory (default) |
| `ONTOCAST_ONTOLOGY_DIRECTORY` | Seed ontology TTL files | Optional bootstrap |
| `MAX_VISITS` | Maximum visits per node | 1 |
| `BASE_RECURSION_LIMIT` | Base recursion limit for workflow | 1000 |
| `ONTOLOGY_MAX_TRIPLES` | Maximum triples allowed in ontology graph | 50000 |
| `ENABLE_ONTOLOGY_CONSOLIDATION` | Run ontology consolidation pass | false |

Full reference: [Configuration](../user_guide/configuration.md).

## Next Steps

Now that you've processed your first document, you can:

1. Try processing different types of documents (PDF, Word)
2. Configure Fuseki for persistent triple storage (see [Triple Stores](../user_guide/triple_stores.md))
3. Check the [API Endpoints](../user_guide/api.md) for REST usage
4. Explore the [User Guide](../user_guide/concepts.md) for advanced usage
