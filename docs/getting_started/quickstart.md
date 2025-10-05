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
ontocast serve --env-path .env
```

- Configuration is provided via `.env` file
- LLM settings are configured in the `.env` file

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

# Path Configuration
WORKING_DIRECTORY=/path/to/working/directory
ONTOLOGY_DIRECTORY=/path/to/ontology/files

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
```

#### Alternative: Ollama Configuration

```bash
# For Ollama
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434
LLM_MODEL_NAME=granite3.3
```

### CLI Parameters

You can override configuration via CLI parameters:

```bash
# Use custom .env file
ontocast serve --env-path /path/to/custom.env

# Process specific input file
ontocast serve --env-path .env --input-path /path/to/document.pdf

# Process only first 5 chunks (for testing)
ontocast serve --env-path .env --head-chunks 5
```

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
| `WORKING_DIRECTORY` | Working directory path | Required |
| `ONTOLOGY_DIRECTORY` | Ontology files directory | Optional |
| `MAX_VISITS` | Maximum visits per node | 3 |
| `SKIP_ONTOLOGY_DEVELOPMENT` | Skip ontology critique | false |

## Next Steps

Now that you've processed your first document, you can:

1. Try processing different types of documents (PDF, Word)
2. Configure triple stores (Neo4j, Fuseki) for persistent storage
3. Check the [API Reference](../reference/onto.md) for more details
4. Explore the [User Guide](../user_guide/concepts.md) for advanced usage
