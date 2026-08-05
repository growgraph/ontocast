# Installation

This guide will help you install OntoCast and its dependencies.

## System Requirements

- Python 3.12 or higher
- uv (Python package installer)

## Installation Steps

```bash
uv add ontocast
```

or

```bash
pip install ontocast
```

Optional extras:

| Extra | Enables | Notes |
|-------|---------|-------|
| `doc-processing` | PDF / DOCX / PPT conversion (Docling), OCR, and the `sentence-transformers` embedding backend used by the default `EMBEDDING_PROVIDER=huggingface` | Needed for any vector-retrieval mode, not only for document conversion |
| `lancedb` | Embedded LanceDB vector store (no external service) | Pair with `doc-processing` for the embedding backend |
| `semantic-chunking` | Clustering-based chunker (`CHUNK_STRATEGY=semantic`) | Pulls `torch`; multi-GB download |
| `shacl` | SHACL validation of aggregated facts via `FACTS_SHAPES_DIR` | Without it, shape validation logs a warning and does nothing |
| `web-search` | Optional web grounding (`WEB_SEARCH_ENABLED=true`) | |
| `plot` | `plot-graph` workflow diagrams | Builds `pygraphviz` from source; needs system graphviz headers |
| `all` | Everything above **except** `plot` | `plot` is excluded because its source build fails without system headers |

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
