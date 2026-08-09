# OntoCast <img src="https://raw.githubusercontent.com/growgraph/ontocast/refs/heads/main/docs/assets/favicon.ico" alt="OntoCast logo" style="height: 32px; width:32px;"/>

**Agentic ontology-assisted extraction of RDF knowledge graphs from documents.**

![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)
[![PyPI version](https://badge.fury.io/py/ontocast.svg)](https://badge.fury.io/py/ontocast)
[![PyPI Downloads](https://static.pepy.tech/badge/ontocast)](https://pepy.tech/projects/ontocast)
[![Docs](https://img.shields.io/badge/docs-growgraph.github.io-orange.svg)](https://growgraph.github.io/ontocast/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![pre-commit](https://github.com/growgraph/ontocast/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/growgraph/ontocast/actions/workflows/pre-commit.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17796467.svg)](https://doi.org/10.5281/zenodo.17796467)

OntoCast turns unstructured text into queryable RDF: it **co-evolves** domain ontologies and fact graphs in a parallel map/reduce pipeline, with RDF 1.2 provenance, entity disambiguation across chunks, and optional vector-backed ontology retrieval. Run it as a REST service, a batch CLI, or embed the pipeline in your own LangChain / LangGraph agent.

**Documentation:** [growgraph.github.io/ontocast](https://growgraph.github.io/ontocast/)

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

Full extras table: [Installation](https://growgraph.github.io/ontocast/getting_started/installation/).

---

## Quick start

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

Omit `FUSEKI_URI` for in-memory pyoxigraph. Details: [Quick Start](https://growgraph.github.io/ontocast/getting_started/quickstart/).

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

Also: `run_unit_pipeline` for a single passage, or `make_ontocast_node` inside your own LangGraph — see [Embedding OntoCast](https://growgraph.github.io/ontocast/user_guide/embedding/).

---

## Workflow

![Workflow diagram](docs/assets/graph.png)

1. Convert → chunk prepare (segment, tag, filter, size)
2. Parallel ontology render → normalize → consolidate → structural check → critic
3. Parallel facts render → merge / disambiguate → validate (invariants, SHACL, autofix)
4. Serialize to the triple store; return Turtle from the API

[Workflow guide](https://growgraph.github.io/ontocast/user_guide/workflow/) · landscape: [`graph.lr.png`](docs/assets/graph.lr.png) · per-unit: [`ontology_loop`](docs/assets/ontology_loop.png), [`facts_loop`](docs/assets/facts_loop.png)

---

## Documentation

Everything lives at **[growgraph.github.io/ontocast](https://growgraph.github.io/ontocast/)**:

| | |
|---|---|
| [Installation](https://growgraph.github.io/ontocast/getting_started/installation/) · [Quick Start](https://growgraph.github.io/ontocast/getting_started/quickstart/) | Getting started |
| [Core Concepts](https://growgraph.github.io/ontocast/user_guide/concepts/) · [Workflow](https://growgraph.github.io/ontocast/user_guide/workflow/) · [Configuration](https://growgraph.github.io/ontocast/user_guide/configuration/) | How it works |
| [API](https://growgraph.github.io/ontocast/user_guide/api/) · [Embedding](https://growgraph.github.io/ontocast/user_guide/embedding/) · [Tenancy](https://growgraph.github.io/ontocast/user_guide/tenancy/) | Integrate |
| [Ontology Context](https://growgraph.github.io/ontocast/user_guide/ontology_context/) · [Validation / SHACL](https://growgraph.github.io/ontocast/user_guide/validation/) · [Triple Stores](https://growgraph.github.io/ontocast/user_guide/triple_stores/) | Operate |
| [API Reference](https://growgraph.github.io/ontocast/reference/) | Python API |

Release notes: [CHANGELOG.md](CHANGELOG.md)

---

## Contributing

See [Contributing](https://growgraph.github.io/ontocast/contributing/). Issues and discussion: [GitHub](https://github.com/growgraph/ontocast).

## License

Apache License 2.0 — see [LICENSE](LICENSE).
