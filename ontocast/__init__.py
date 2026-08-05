"""OntoCast: Agentic ontology-assisted framework for semantic triple extraction.

OntoCast is a comprehensive framework for extracting semantic triples from
documents using ontology assistance. It provides a complete pipeline for
document processing, ontology management, and knowledge graph construction.

The framework includes:
- Document conversion and chunking
- Ontology selection and management
- Fact extraction and validation
- Triple store integration (Fuseki, In-Memory)
- LLM-powered semantic analysis
- REST API server for document processing

For more information, see the documentation at https://growgraph.github.io/ontocast/
"""

from typing import Any

from ontocast._version import __version__

__all__ = ["Config", "ToolBox", "__version__", "facts_loop", "ontology_loop"]

# Everything is resolved lazily so `import ontocast` stays cheap: ToolBox pulls
# in the whole tool tree and Config the settings tree, and neither is wanted
# just to read __version__.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "Config": ("ontocast.config", "Config"),
    "ToolBox": ("ontocast.toolbox", "ToolBox"),
    "facts_loop": ("ontocast.stategraph.atomic", "facts_loop"),
    "ontology_loop": ("ontocast.stategraph.atomic", "ontology_loop"),
}


def __getattr__(name: str) -> Any:
    """Lazily export the documented entry points."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_name, attribute = target
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
