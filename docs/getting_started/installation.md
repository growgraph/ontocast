# Installation

This guide will help you install OntoCast and its dependencies.

## System Requirements

- Python 3.12 or higher
- uv (Python package installer)

## Installation Steps

Pick your install by what you are doing.

```bash
# Running the server or the CLI
uv add "ontocast[server,openai,documents]"

# Embedding OntoCast in your own application -- see the Embedding guide
uv add "ontocast[openai]"
```

The base `ontocast` package is deliberately light: the extraction pipeline, the
RDF stack, the in-memory triple and vector stores, and the ontology tooling.
Anything that pulls a service SDK, a document-processing stack or an ML runtime
sits behind an extra, so that embedding OntoCast in another application does not
install a gRPC stack and an ONNX runtime.

**You must pick at least one LLM provider extra** — OntoCast does not choose one
for you.

| Extra | Enables | Notes |
|-------|---------|-------|
| `openai` / `anthropic` / `google` / `ollama` | The matching LLM provider | One is required |
| `server` | The `ontocast` command, every console script, and the HTTP API | FastAPI, uvicorn, click, rich. Without it the console scripts print an install hint and exit |
| `documents` | `docling-core`: representing and chunking converted documents | Required to chunk anything; pulls pandas, pyarrow, transformers |
| `doc-processing` | PDF / DOCX / PPT conversion (Docling), OCR, and the `sentence-transformers` backend used by the default `EMBEDDING_PROVIDER=huggingface` | Implies `documents` |
| `qdrant` | Qdrant vector store | Pulls `qdrant-client` and gRPC |
| `lancedb` | Embedded LanceDB vector store (no external service) | |
| `sparse` | `fastembed` BM25 sparse embeddings | Implied by `qdrant` and `lancedb`; pulls an ONNX runtime |
| `semantic-chunking` | Clustering-based chunker (`CHUNK_STRATEGY=semantic`) | Pulls `torch` and `sentence-transformers`; multi-GB download. The model is shared with retrieval and disambiguation when `CHUNK_EMBEDDING_MODEL` matches theirs |
| `graph` | `networkx` ontology lineage graphs | |
| `shacl` | SHACL validation of aggregated facts via `FACTS_SHAPES_DIR` | Without it, shape validation logs a warning and does nothing |
| `web-search` | Optional web grounding (`WEB_SEARCH_ENABLED=true`) | |
| `plot` | `plot-graph` workflow diagrams | Builds `pygraphviz` from source; needs system graphviz headers |
| `all` | Everything above **except** `plot` | `plot` is excluded because its source build fails without system headers |

Vector retrieval needs no extra at all if you use the in-memory backend
(`VECTOR_STORE_BACKEND=memory`) with an API-based embedding provider
(`EMBEDDING_PROVIDER=openai` or `=ollama`). See
[Embedding OntoCast](../user_guide/embedding.md).

```bash
# Typical: document conversion plus an embedded vector store
uv add "ontocast[doc-processing,lancedb]"

# Everything except the graphviz-dependent plotting extra
uv add "ontocast[all]"

# Plotting requires system graphviz first, e.g. apt install graphviz graphviz-dev
uv add "ontocast[plot]"
```

## Next Steps

After installation, you can:

1. Read the [Quick Start](quickstart.md) guide
2. Check the [Configuration](../user_guide/configuration.md) reference
3. Browse the generated [API Reference](../reference/) after `uv run mkdocs build`
