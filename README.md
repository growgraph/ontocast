# OntoCast <img src="https://raw.githubusercontent.com/growgraph/ontocast/refs/heads/main/docs/assets/logo.png" alt="OntoCast logo" style="height: 32px; width:32px;"/>

**Ontology-guided extraction of RDF knowledge graphs from documents.**

![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)
[![PyPI version](https://badge.fury.io/py/ontocast.svg)](https://badge.fury.io/py/ontocast)
[![PyPI Downloads](https://static.pepy.tech/badge/ontocast)](https://pepy.tech/projects/ontocast)
[![Docs](https://img.shields.io/badge/docs-growgraph.github.io-224777.svg)](https://growgraph.github.io/ontocast/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![pre-commit](https://github.com/growgraph/ontocast/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/growgraph/ontocast/actions/workflows/pre-commit.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17796467.svg)](https://doi.org/10.5281/zenodo.17796467)

OntoCast reads documents and writes an RDF knowledge graph: an ontology that
describes the domain, and the facts the documents state in its terms. Give it
your ontologies and it extracts facts against them; give it none and it builds
one as it reads. Run it as an HTTP service, as a batch command, or inside your
own LangChain or LangGraph agent.

**Documentation:** [growgraph.github.io/ontocast](https://growgraph.github.io/ontocast/)

## Features

- **Ontology and facts together.** Each part of a document goes through a
  language model in a render-and-critique loop, in parallel; ontology changes
  are merged, versioned and checked before facts are extracted against them.
- **Patches, not rewrites.** The model emits insert/delete updates to a graph
  rather than regenerating it.
- **Entity disambiguation.** Mentions of the same entity across a document are
  merged into one.
- **Validation.** Deterministic checks, SHACL shapes, and repairs that need no
  extra model call.
- **Provenance.** Facts carry RDF 1.2 provenance back to the text, which you can
  strip on output.
- **Ontology context.** Each part of a document is shown the ontology it needs:
  chosen from your catalog, retrieved from a vector store (LanceDB or Qdrant),
  or fixed.
- **Storage.** In memory by default, Apache Jena Fuseki for persistence,
  partitioned by tenant and project.
- **A light core.** The base install embeds without a document-processing stack
  or an ML runtime; those are extras.

## Install

```sh
pip install "ontocast[server,openai,doc-processing]"
```

`server` provides the `ontocast` command and HTTP API, `openai` the model
provider (`anthropic`, `google` and `ollama` also exist), and `doc-processing`
the document converter (PDF, Office, HTML, Markdown, images). All extras:
[Installation](https://growgraph.github.io/ontocast/getting_started/installation/).

## Quick start

OntoCast reads its settings from environment variables:

```bash
export LLM_API_KEY=sk-...
ontocast serve
curl -X POST http://127.0.0.1:8999/process -F "file=@document.pdf" -o result.json
```

The response holds the facts and the ontology as Turtle. To process files
without a server:

```bash
ontocast process --input-path ./papers --output-dir ./out
```

To keep settings in a file, copy [`.env.example.minimal`](.env.example.minimal),
edit it, and pass it with `ontocast --env-file my.env serve` (repeatable; later
files win). Step by step:
[Quick start](https://growgraph.github.io/ontocast/getting_started/quickstart/).

## Your own ontologies

Put Turtle files in a directory and pass it at startup, or upload them to a
running server:

```bash
ontocast serve --ontology-dir ./my-ontologies
curl -X POST http://127.0.0.1:8999/ontologies -F "file=@my-ontology.ttl"
```

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

`run_unit_pipeline` processes a single passage, and `make_ontocast_node` adds
OntoCast to your own LangGraph. See [Embedding
OntoCast](https://growgraph.github.io/ontocast/guides/embedding/).

## How it works

![The OntoCast pipeline: convert, chunk, then the ontology stages and the facts stages, then serialize](https://raw.githubusercontent.com/growgraph/ontocast/refs/heads/main/docs/assets/graph.lr.png)

A document is converted to text and cut into parts. Each part updates the
ontology; the updates are normalized, consolidated and checked. Each part then
yields facts in the ontology's terms; the facts are merged, disambiguated and
validated, and the result is written to the triple store. [How OntoCast
works](https://growgraph.github.io/ontocast/concepts/).

## Documentation

| | |
|---|---|
| Getting started | [Installation](https://growgraph.github.io/ontocast/getting_started/installation/) · [Quick start](https://growgraph.github.io/ontocast/getting_started/quickstart/) |
| Concepts | [How OntoCast works](https://growgraph.github.io/ontocast/concepts/) · [Ontologies and facts](https://growgraph.github.io/ontocast/concepts/ontologies_and_facts/) |
| Guides | [Configuring OntoCast](https://growgraph.github.io/ontocast/guides/configuration/) · [Recipes](https://growgraph.github.io/ontocast/guides/recipes/) · [Validation and SHACL](https://growgraph.github.io/ontocast/guides/validation/) · [Triple stores](https://growgraph.github.io/ontocast/guides/triple_stores/) |
| Reference | [Configuration](https://growgraph.github.io/ontocast/reference/configuration/) · [HTTP API](https://growgraph.github.io/ontocast/reference/http_api/) · [Python API](https://growgraph.github.io/ontocast/reference/python/) |

Release notes: [CHANGELOG.md](CHANGELOG.md)

## Contributing

See [Contributing](https://growgraph.github.io/ontocast/contributing/). Issues
and discussion: [GitHub](https://github.com/growgraph/ontocast). Contributors
accept the [Contributor License Agreement](CLA.md) once, by commenting on
their first pull request.

## License

Apache License 2.0; see [LICENSE](LICENSE).
