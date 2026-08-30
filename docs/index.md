# OntoCast <img src="https://raw.githubusercontent.com/growgraph/ontocast/refs/heads/main/docs/assets/favicon.ico" alt="OntoCast logo" style="height: 32px; width:32px;"/>

**Agentic ontology-assisted extraction of RDF knowledge graphs from documents.**

![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)
[![PyPI version](https://badge.fury.io/py/ontocast.svg)](https://badge.fury.io/py/ontocast)
[![PyPI Downloads](https://static.pepy.tech/badge/ontocast)](https://pepy.tech/projects/ontocast)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![pre-commit](https://github.com/growgraph/ontocast/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/growgraph/ontocast/actions/workflows/pre-commit.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17796467.svg)](https://doi.org/10.5281/zenodo.17796467)

OntoCast turns unstructured text into queryable RDF: it **co-evolves** domain ontologies and fact graphs in a parallel map/reduce pipeline, with RDF 1.2 provenance, entity disambiguation across chunks, and optional vector-backed ontology retrieval. Run it as a REST service, a batch CLI, or embed the pipeline in your own LangChain / LangGraph agent.

---

## Why OntoCast

Most extractors dump triples and leave ontology drift to you. OntoCast treats schema and instance data as one loop: per-chunk render → critic → merge, with GraphUpdate patches (insert/delete) instead of regenerating whole graphs, SHACL validation with LLM-free autofix, and a light install so you can embed the core without pulling Docling, gRPC, or ONNX.

---

## Features

- **Parallel ontology + facts loops** — concurrent per-unit render/critic with configurable workers
- **GraphUpdate patches** — token-efficient insert/delete ops, not full-graph regeneration
- **Entity disambiguation** — embedding + symbolic alignment across chunks
- **RDF 1.2 provenance** — quoted triples / provenance artifacts; optional `strip_provenance`
- **Ontology context** — catalog selection, vector retrieval (LanceDB or Qdrant), or a fixed ontology
- **Facts validation** — invariants, SHACL, and machine repairs without an extra LLM pass
- **Stores** — in-memory pyoxigraph by default; Fuseki for persistence; tenancy by tenant/project
- **LLM caching** — disk cache, in-flight limits, optional read-only / batch pre-warm
- **Embeddable** — `ontocast_tools`, `run_unit_pipeline`, or a LangGraph node

---

## Install

Pick at least one LLM provider extra. Add `server` for the CLI and HTTP API:

```sh
uv add "ontocast[server,openai]"
# or: pip install "ontocast[server,openai]"
```

Common add-ons: `doc-processing` (PDF/DOCX), `lancedb` or `qdrant` (ontology retrieval), `shacl` (shape validation).

```sh
uv add "ontocast[server,openai,doc-processing,lancedb,shacl]"
```

See [Installation](getting_started/installation.md) for the full extras table.

---

## Quick Start

```bash
cp .env.example .env
# Set LLM_API_KEY (and LLM_PROVIDER / LLM_MODEL_NAME as needed)

ontocast serve
curl -X POST http://localhost:8999/process -F "file=@document.pdf"
```

Batch without a server:

```bash
ontocast process --input-path ./document.pdf --head-chunks 5 --output-dir ./out
```

Omit `FUSEKI_URI` for in-memory pyoxigraph. Details: [Quick Start Guide](getting_started/quickstart.md).

### Supplying Your Ontologies

OntoCast uses seed ontologies (in Turtle `.ttl` format) to guide extraction. Provide yours in two ways:

1. **Directory Seed:** Set `ONTOCAST_ONTOLOGY_DIRECTORY=/path/to/your/ontologies` in your `.env`. All `.ttl` files in that folder sync automatically on startup.
2. **API Upload:** Register schemas dynamically with the running server:
   ```bash
   curl -X POST "http://localhost:8999/ontologies?tenant=ontocast&project=test" -F "file=@my_ontology.ttl"
   ```

---

## Configuration

Start from `.env.example.minimal` — 47 variables instead of 202, grouped by the
decision they belong to. Then pick a [playbook](user_guide/playbooks.md) for what
you are actually doing: evaluating, building an ontology, populating facts,
scaling to a large catalog, or serving it.

The knobs that change *what the pipeline does* — as opposed to where it stores
things:

| Variable | Default | What it controls |
|---|---|---|
| `RENDER_MODE` | `ontology_and_facts` | Which halves run. `ontology` writes no facts; `facts` skips the ontology block and extracts only against the catalog you already have — see [Render Mode](user_guide/configuration.md#render-mode-render_mode) |
| `ONTOLOGY_CONTEXT_MODE` | `selected_single_ontology` | Where each unit's schema comes from: LLM catalog selection, vector retrieval, or one pinned ontology — see [Ontology Context](user_guide/ontology_context.md) |
| `LLM_GRAPH_FORMAT` | `jsonld` | Wire encoding the LLM emits graphs in; `turtle` is the legacy alternative |
| `MAX_VISITS_PER_NODE` | `1` | **Render** attempts per unit. The facts critic still runs at `1` (see `FACTS_LLM_REPAIR_VISITS`) |
| `PARALLEL_WORKERS` | `16` | Concurrent content-unit workers |
| `LLM_PROVIDER` / `LLM_MODEL_NAME` / `LLM_API_KEY` | `openai` | Provider selection and credentials |
| `ONTOCAST_ONTOLOGY_DIRECTORY` | — | Seed ontologies synced on startup |
| `FUSEKI_URI` | — | Triple store; unset means in-memory pyoxigraph |

`RENDER_MODE`, `ONTOLOGY_CONTEXT_MODE` and `LLM_GRAPH_FORMAT` are also
per-request parameters on `/process`. Full surface, including chunking,
retrieval and validation: [Configuration System](user_guide/configuration.md).

---

## Embed in your agent

```python
from langchain.agents import create_agent
from ontocast import Config, ToolBox, ontocast_tools

tools = await ToolBox.acreate(Config.in_memory())
await tools.initialize()

agent = create_agent(
    model,
    tools=[*ontocast_tools(tools)],
    prompt="Edit the ontology from the user's text.",
)
```

Also: `run_unit_pipeline` for a single passage, or `make_ontocast_node` inside your own LangGraph — see [Embedding OntoCast](user_guide/embedding.md).

---

## Workflow

![Workflow diagram](assets/graph.png)

1. Convert → chunk prepare (segment, tag, filter, size)
2. Parallel ontology render → normalize → consolidate → structural check → critic
3. Parallel facts render → merge / disambiguate → validate (invariants, SHACL, autofix)
4. Serialize to the triple store; return Turtle from the API

[Workflow Guide](user_guide/workflow.md) · landscape: [`graph.lr.png`](assets/graph.lr.png) · per-unit: [`ontology_loop`](assets/ontology_loop.png), [`facts_loop`](assets/facts_loop.png)

---

## Documentation

Browse the complete documentation using the sidebar or start with these core guides:

| | |
|---|---|
| [Installation](getting_started/installation.md) · [Quick Start](getting_started/quickstart.md) | Getting started |
| [Core Concepts](user_guide/concepts.md) · [Workflow Guide](user_guide/workflow.md) · [Configuration System](user_guide/configuration.md) | How it works |
| [API Endpoints](user_guide/api.md) · [Embedding OntoCast](user_guide/embedding.md) · [Tenancy](user_guide/tenancy.md) | Integrate |
| [Ontology Context](user_guide/ontology_context.md) · [Validation / SHACL](user_guide/validation.md) · [Triple Stores](user_guide/triple_stores.md) | Operate |
| [API Reference](reference/) | Python API |

Release notes: [CHANGELOG](https://github.com/growgraph/ontocast/blob/main/CHANGELOG.md)

---

## Contributing

We welcome contributions! See the [Contributing Guide](contributing.md) for guidelines, and feel free to open issues or discussions on [GitHub](https://github.com/growgraph/ontocast).

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](https://github.com/growgraph/ontocast/blob/main/LICENSE) file for details.
