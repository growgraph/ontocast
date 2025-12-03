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

## Configuration

## Documentation

- [Quick Start Guide](getting_started/quickstart.md) - Get started quickly
- [Configuration System](user_guide/configuration.md) - Detailed configuration guide
- [LLM Caching](user_guide/llm_caching.md) - Automatic response caching
- [Triple Store Setup](user_guide/triple_stores.md) - Triple store configuration
- [User Guide](user_guide/concepts.md) - Core concepts and workflow
- [API Reference](reference/onto.md) - Detailed API documentation


### Environment Variables

Copy the example file and edit as needed:

```bash
cp env.example .env
# Edit with your values
```

**Main options:**
```bash
# LLM Configuration
# common
LLM_PROVIDER=openai # or ollama
LLM_MODEL_NAME=gpt-4o-mini # ollama model
LLM_TEMPERATURE=0.0

# openai
LLM_API_KEY=your_openai_api_key_here

# ollama
LLM_BASE_URL=

# Server
PORT=8999
RECURSION_LIMIT=1000
ESTIMATED_CHUNKS=30

# Backend Configuration (auto-detected)
FUSEKI_URI=http://localhost:3032/test
FUSEKI_AUTH=admin:password
ONTOCAST_WORKING_DIRECTORY=/path/to/working

# Optional: Triple Store Configuration (Fuseki preferred over Neo4j)
FUSEKI_URI=http://localhost:3032/test
FUSEKI_AUTH=admin/abc123-qwe

NEO4J_URI=bolt://localhost:7689
NEO4J_AUTH=neo4j/test!passfortesting
```

---

## Triple Store Setup

OntoCast supports multiple triple store backends. When both Fuseki and Neo4j are configured, **Fuseki is preferred**.

- See [Triple Store Setup](user_guide/triple_stores.md) for detailed Docker Compose instructions and sample `.env.example` files.
- Quick summary: copy and edit the provided `.env.example` in `docker/fuseki` or `docker/neo4j`, then run `docker compose --env-file .env <service> up -d` in the respective directory.

---

## Running OntoCast Server

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

---

## API Usage

- **POST /process**: Accepts `application/json` or file uploads (`multipart/form-data`).
- Returns: JSON with extracted facts (Turtle), ontology (Turtle), and processing metadata. Triples are also serialized to the configured triple store.

**Example:**
```bash
curl -X POST http://localhost:8999/process \
    -H "Content-Type: application/json" \
    -d '{"text": "Your document text here"}'
    
# Process a PDF file
curl -X POST http://url:port/process -F "file=@data/pdf/sample.pdf"

# Process a json file
curl -X POST http://url:port/process -F "file=@test2/sample.json"
```

---

## MCP Endpoints

- `GET /health`: Health check
- `GET /info`: Service info
- `POST /process`: Document processing
- `POST /flush`: Flush/clean triple store data (optional `dataset` query parameter for Fuseki)

---

## Filesystem Mode

If no triple store is configured, OntoCast stores ontologies and facts as Turtle files in the working directory.

---

## Notes

- JSON documents must contain a `text` field, e.g.:
  ```json
  { "text": "abc" }
  ```
- `recursion_limit` is calculated as `max_visits * estimated_chunks` (default 30, or set via `.env`)
- Default port: 8999

---

## Docker

To build the OntoCast Docker image:
```sh
docker buildx build -t growgraph/ontocast:0.1.4 . 2>&1 | tee build.log
```

---

## Project Structure

```
ontocast/
├── agent/           # Agent workflow and orchestration
├── cli/             # CLI utilities and server
├── prompt/          # LLM prompt templates
├── stategraph/      # State graph logic
├── tool/            # Triple store, chunking, and ontology tools
├── toolbox.py       # Toolbox for agent tools
├── onto.py          # Ontology and RDF graph handling
├── util.py          # Utilities
```
Other directories:
- `docker/` – Docker Compose and .env.example files for triple stores
- `data/` – Example data, ontologies, and test files
- `docs/` – Documentation and user guides
- `test/` – Test suite

---

## Workflow

The extraction follows a multi-stage workflow:

<img src="https://github.com/growgraph/ontocast/blob/main/docs/assets/graph.png?raw=true" alt="Workflow diagram" width="350" style="float: right; margin-left: 20px;"/>

1. **Document Preparation**
    - [Optional] Convert to Markdown
    - Text chunking
2. **Ontology Processing**
    - Ontology selection
    - Text to ontology triples
    - Ontology critique
3. **Fact Extraction**
    - Text to facts
    - Facts critique
    - Ontology sublimation
4. **Chunk Normalization**
    - Chunk KG aggregation
    - Entity/Property Disambiguation
5. **Storage**
    - Triple/KG serialization

---


## Roadmap

- [x] Add Jena Fuseki triple store for triple serialization
- [x] Add Neo4j n10s for triple serialization
- [ ] Replace triple prompting with a tool for local graph retrieval

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## Acknowledgments

- Uses RDFlib for semantic triple management
- Uses docling for pdf/pptx conversion
- Uses OpenAI language models / open models served via Ollama for fact extraction
- Uses langchain/langgraph
