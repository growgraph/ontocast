"""Tool package for OntoCast.

The Qdrant and LanceDB vector managers are re-exported lazily. Naming them in a
plain ``from .vector_store import ...`` would defeat that subpackage's own lazy
export, because a from-import resolves every name in its list immediately.
"""

from typing import Any

from ontocast.tool.chunk.chunker import ChunkerTool

from .agg.aggregate import EmbeddingBasedAggregator
from .atomic import AtomicToolBox, SearchHit
from .converter import ConverterTool
from .llm import LLMTool
from .onto import Tool
from .ontology_manager import OntologyManager
from .triple_manager import (
    FusekiTripleStoreManager,
    InMemoryTripleStoreManager,
    TripleStoreManager,
)
from .vector_store import (
    EmbeddingTool,
    InMemoryVectorStoreManager,
    OntologyPatchRetriever,
    VectorStoreManager,
)

__all__ = [
    "LLMTool",
    "OntologyManager",
    "TripleStoreManager",
    "FusekiTripleStoreManager",
    "InMemoryTripleStoreManager",
    "ConverterTool",
    "ChunkerTool",
    "Tool",
    "AtomicToolBox",
    "SearchHit",
    "EmbeddingTool",
    "InMemoryVectorStoreManager",
    "QdrantVectorStoreManager",
    "LanceDBVectorStoreManager",
    "VectorStoreManager",
    "OntologyPatchRetriever",
    "EmbeddingBasedAggregator",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "QdrantVectorStoreManager": (
        "ontocast.tool.vector_store.qdrant",
        "QdrantVectorStoreManager",
    ),
    "LanceDBVectorStoreManager": (
        "ontocast.tool.vector_store.lancedb",
        "LanceDBVectorStoreManager",
    ),
}


def __getattr__(name: str) -> Any:
    """Resolve the optional-backend managers on first access."""
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
