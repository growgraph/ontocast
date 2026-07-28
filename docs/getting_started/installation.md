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

```bash
# PDF / DOCX conversion (Docling)
uv add "ontocast[doc-processing]"

# Embedded LanceDB vector store
uv add "ontocast[lancedb]"
```

## Next Steps

After installation, you can:

1. Read the [Quick Start](quickstart.md) guide
2. Check the [Configuration](../user_guide/configuration.md) reference
3. Browse the generated [API Reference](../reference/) after `uv run mkdocs build`
