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
ontocast --env-path .env

# Process specific file
ontocast --env-path .env --input-path ./document.pdf

# Process with chunk limit (for testing)
ontocast --env-path .env --head-chunks 5
```

- Backend selection is **fully automatic** based on available configuration
- No explicit backend flags needed - just provide the required credentials/paths in .env
- All paths and directories are configured via .env file

### Configuration

OntoCast uses a hierarchical configuration system with environment variables. Create a `.env` file in your project directory:

```bash
# Domain configuration (used for URI generation) 
CURRENT_DOMAIN=https://example.com
PORT=8999
LLM_TEMPERATURE=0.1

# LLM Configuration
LLM_PROVIDER=openai
LLM_API_KEY=your-api-key-here
LLM_MODEL_NAME=gpt-4o-mini

# Server Configuration
MAX_VISITS=3
RECURSION_LIMIT=1000
ESTIMATED_CHUNKS=30

# Backend Configuration (auto-detected)
FUSEKI_URI=http://localhost:3032/test
FUSEKI_AUTH=admin:password
ONTOCAST_WORKING_DIRECTORY=/path/to/working

# Path Configuration (required for filesystem backends)
ONTOCAST_WORKING_DIRECTORY=/path/to/working/directory
ONTOCAST_ONTOLOGY_DIRECTORY=/path/to/ontology/files
ONTOCAST_CACHE_DIR=/path/to/cache/directory

# Triple Store Configuration (Optional)
# For Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_AUTH=username:password

# For Fuseki
FUSEKI_URI=http://localhost:3030
FUSEKI_AUTH=username:password
FUSEKI_DATASET=dataset_name

# Skip ontology critique (optional)
SKIP_ONTOLOGY_DEVELOPMENT=false
# Maximum triples allowed in ontology graph (optional, set empty for unlimited)
ONTOLOGY_MAX_TRIPLES=10000
```

#### Alternative: Ollama Configuration

```bash
# For Ollama
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434
LLM_MODEL_NAME=granite3.3
```

### CLI Parameters

You can use these CLI parameters:

```bash
# Use custom .env file
ontocast --env-path /path/to/custom.env

# Process specific input file
ontocast --env-path .env --input-path /path/to/document.pdf

# Process only first 5 chunks (for testing)
ontocast --env-path .env --head-chunks 5
```

**Note:** All paths and directories are configured via the `.env` file - no CLI overrides needed.

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
| `LLM_API_KEY` | API key for LLM provider | Required |
| `LLM_PROVIDER` | LLM provider (openai, ollama) | openai |
| `LLM_MODEL_NAME` | Model name | gpt-4o-mini |
| `FUSEKI_URI` + `FUSEKI_AUTH` | Use Fuseki as main triple store | Auto-detected |
| `NEO4J_URI` + `NEO4J_AUTH` | Use Neo4j as main triple store | Auto-detected |
| `ONTOCAST_WORKING_DIRECTORY` + `ONTOCAST_ONTOLOGY_DIRECTORY` | Use filesystem as main triple store | Auto-detected |
| `ONTOCAST_ONTOLOGY_DIRECTORY` | Ontology files directory | Provide seed ontologies |
| `MAX_VISITS` | Maximum visits per node | 3 |
| `SKIP_ONTOLOGY_DEVELOPMENT` | Skip ontology critique | false |
| `ONTOLOGY_MAX_TRIPLES` | Maximum triples allowed in ontology graph | 10000 |

## Next Steps

Now that you've processed your first document, you can:

1. Try processing different types of documents (PDF, Word)
2. Configure triple stores (Neo4j, Fuseki) for persistent storage
3. Check the [API Reference](../reference/onto.md) for more details
4. Explore the [User Guide](../user_guide/concepts.md) for advanced usage
