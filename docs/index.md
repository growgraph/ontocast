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

OntoCast extracts semantic triples from documents using an agentic, ontology-driven pipeline. It co-evolves ontologies and facts graphs with parallel per-chunk processing, RDF 1.2 provenance, and optional vector-backed ontology retrieval.

---

## Key Features

- **Parallel map/reduce pipeline** — concurrent per-unit ontology and facts loops
- **Robust entity disambiguation** — embedding + symbolic alignment across chunks
- **RDF 1.2 provenance** — quoted triples, provenance artifacts, optional `strip_provenance`
- **GraphUpdate operations** — token-efficient structured insert/delete triple patches instead of full graph regeneration
- **JSON-LD wire format** — optional `LLM_GRAPH_FORMAT=jsonld` for LLM payloads
- **Ontology context modes** — catalog selection, vector retrieval, or fixed ontology
- **Triple store integration** — Fuseki, Neo4j (n10s), or filesystem fallback
- **Tenancy** — partition datasets/collections by tenant and project
- **REST API** — document processing, ontology catalog management, graph matching
- **Automatic LLM caching** — disk cache with optional read-only mode, global in-flight limiting, and OpenAI Batch API pre-warming for benchmarks
- **Structured documents** — optional section tagging, section-aligned chunk labels, section filtering, and LLM summarization before extraction

---

## Documentation

- [Quick Start](getting_started/quickstart.md)
- [Workflow](user_guide/workflow.md)
- [Core Concepts](user_guide/concepts.md)
- [Configuration](user_guide/configuration.md)
- [API Endpoints](user_guide/api.md)
- [Tenancy](user_guide/tenancy.md)
- [Ontology Context](user_guide/ontology_context.md)
- [Triple Stores](user_guide/triple_stores.md)
- [LLM Caching](user_guide/llm_caching.md)
- [API Reference](reference/onto/state.md)

---

## Installation

```sh
uv add ontocast
# or
pip install ontocast
```

Optional PDF/DOCX conversion: `pip install "ontocast[doc-processing]"`

---

## Quick Start

```bash
cp .env.example .env
# Edit LLM_API_KEY and paths

ontocast --env-path .env

curl -X POST http://localhost:8999/process -F "file=@document.pdf"
```

See [Quick Start Guide](getting_started/quickstart.md) for full configuration.

---

## REST API (Summary)

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Health check |
| `GET` | `/info` | Service metadata |
| `POST` | `/process` | Full document pipeline |
| `POST` | `/process_unit` | Single content unit |
| `POST` | `/flush` | Clear triple store data |
| `POST` | `/ontologies` | Upload catalog ontology |
| `PUT/DELETE` | `/ontologies/{iri}` | Replace or delete ontology |
| `POST` | `/match/entities` | Global entity alignment |
| `POST` | `/match/derive-matches` | Pairwise entity matching |
| `POST` | `/match/evaluate` | Triple/entity metrics |

Details: [API Endpoints](user_guide/api.md).

---

## Workflow

Document-level pipeline (regenerated via `uv run plot-graph`):

![Workflow diagram](assets/graph.png)

Landscape variant: [graph.lr.png](assets/graph.lr.png). Per-unit loops: [ontology_loop.png](assets/ontology_loop.png), [facts_loop.png](assets/facts_loop.png) — details in [Workflow](user_guide/workflow.md#per-unit-atomic-loop).

1. Convert → chunk prepare (segment, tag, filter, size) → optional summarize chunks
2. Parallel ontology render per unit → normalize → optional consolidate → validate
3. Parallel facts render per unit → merge with disambiguation
4. Serialize to triple store; return Turtle in API response

---

## Project Structure

```
ontocast/
├── agent/           # Render, critic, normalize, serialize agents
├── api/             # FastAPI routers (ontologies, schemas, tenancy)
├── cli/             # Server and utility CLIs
├── onto/            # Ontology, RDFGraph, state models
├── prompt/          # LLM prompt templates
├── stategraph/      # LangGraph workflow
├── tool/            # Triple stores, chunking, vector store, aggregation
├── config.py        # Pydantic settings
└── toolbox.py       # Tool dependency container
```

---

## Contributing

See [Contributing](contributing.md) and [CHANGELOG](https://github.com/growgraph/ontocast/blob/main/CHANGELOG.md).
