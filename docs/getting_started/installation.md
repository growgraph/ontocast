# Installation

This page helps you choose which parts of OntoCast to install: the base
package is small, and everything heavy sits behind an extra you add only when
you need it.

## Requirements

- Python 3.12 or later
- `pip`, or `uv` if you manage your project with it

## Install

Pick the install by what you are doing:

```bash
# Run the server or the CLI on PDF and PowerPoint files
pip install "ontocast[server,openai,doc-processing]"

# Embed OntoCast in your own application; see Embedding OntoCast in your agent
pip install "ontocast[openai]"
```

With `uv`, use `uv add` with the same specifiers.

The base package holds the extraction pipeline, the RDF stack, the in-memory
triple and vector stores, and the ontology tooling. Anything that pulls a
service SDK, a document-processing stack or an ML runtime sits behind an
extra, so embedding OntoCast in another application installs only what it
uses.

**Pick at least one LLM provider extra.** OntoCast does not install one for
you.

## Extras

| Extra | Enables | Notes |
|-------|---------|-------|
| `openai` / `anthropic` / `google` / `ollama` | The matching LLM provider | One is required |
| `server` | The `ontocast` command, every console script, and the HTTP API | FastAPI, uvicorn, click, rich. Without it the console scripts print an install hint and exit |
| `documents` | Representing and chunking converted documents (`docling-core`) | Included by `server` and `doc-processing`; pulls pandas, pyarrow, transformers |
| `doc-processing` | PDF and PowerPoint conversion (Docling), OCR, and the `sentence-transformers` backend used by the default `EMBEDDING_PROVIDER=huggingface` | Includes `documents` |
| `qdrant` | Qdrant vector store | Pulls `qdrant-client` and gRPC |
| `lancedb` | Embedded LanceDB vector store, no external service | |
| `sparse` | `fastembed` BM25 sparse embeddings | Included by `qdrant` and `lancedb`; pulls an ONNX runtime |
| `semantic-chunking` | The clustering-based chunker (`CHUNK_SEGMENTER=semantic`, the default) | Without it the chunker falls back to paragraph and sentence splitting and logs a warning. Pulls `torch` and `sentence-transformers`, a multi-gigabyte download |
| `graph` | `networkx` ontology lineage graphs | |
| `shacl` | SHACL validation of extracted facts, plus the shape-driven autofix | Without it, shape validation logs a warning and does nothing; see [Validation](../guides/validation.md) |
| `web-search` | Web grounding (`WEB_SEARCH_ENABLED=true`) | |
| `plot` | `plot-graph` workflow diagrams | `pygraphviz`, whose wheel bundles Graphviz |
| `all` | Every extra above except `plot` | `plot` is not included in `all` because it is a large optional dependency |

Vector retrieval is off in a base install: each content unit is shown one
working ontology, which is the default. To turn retrieval on, install one of
the two backends, `lancedb` (embedded) or `qdrant` (a server), and set
`ONTOLOGY_CONTEXT_MODE=selected_vector_search_ontology`; see [Choosing ontology
context](../guides/ontology_context.md).

```bash
# Document conversion plus an embedded vector store
pip install "ontocast[doc-processing,lancedb]"

# Everything except plotting
pip install "ontocast[all]"

# Plotting
pip install "ontocast[plot]"
```

## Console scripts

The `server` extra puts these commands on your `PATH`. Without it they print
an install hint instead of failing with an import error.

| Command | Purpose |
|---------|---------|
| `ontocast serve` | Start the HTTP server; see the [HTTP API](../reference/http_api.md) |
| `ontocast process` | Extract from a local file or directory, without a server |
| `ontocast sections` | Print the section outline detected in a document, without running extraction |
| `ontocast cache` | Inspect (`stats`), trim (`prune`) or empty (`clear`) the on-disk cache |
| `pdfs-to-markdown` | Convert a directory of PDFs to Markdown JSON once, so later runs reuse the conversion |
| `match-graphs` | Match the entities of two Turtle graphs; see [Entity disambiguation](../concepts/entity_disambiguation.md) |
| `plot-graph` | Regenerate the workflow diagrams (needs the `plot` extra) |

## Next steps

- [Quick start](quickstart.md): take one document to a knowledge graph.
- [Configuring OntoCast](../guides/configuration.md): the settings that change what a run does.
- [Python API reference](../reference/python/index.md): the modules you call when embedding OntoCast.
