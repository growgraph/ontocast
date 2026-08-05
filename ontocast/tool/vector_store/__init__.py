"""Vector store package for ontology patch retrieval.

The Qdrant and LanceDB managers are exported lazily: importing either pulls its
backend SDK, and neither ships in OntoCast's base install. Everything else --
the abstract manager, the atom model, the embedding tools, and the in-memory
backend -- is dependency-light and imported eagerly.
"""

from typing import Any

from .atomizer import GraphAtomizer
from .core import GraphAtom, OntologySearchHit, VectorStoreManager
from .embedding import (
    EmbeddingTool,
    FastembedBm25SparseTool,
    HuggingFaceEmbeddingTool,
    OllamaEmbeddingTool,
    OpenAIEmbeddingTool,
)
from .factory import create_vector_store_manager
from .in_memory import InMemoryVectorStoreManager
from .patch_retriever import OntologyPatchRetriever
from .util import EmbeddingContractMismatchError

__all__ = [
    "EmbeddingTool",
    "FastembedBm25SparseTool",
    "HuggingFaceEmbeddingTool",
    "OllamaEmbeddingTool",
    "OpenAIEmbeddingTool",
    "GraphAtom",
    "OntologySearchHit",
    "GraphAtomizer",
    "OntologyPatchRetriever",
    "EmbeddingContractMismatchError",
    "InMemoryVectorStoreManager",
    "QdrantVectorStoreManager",
    "LanceDBVectorStoreManager",
    "VectorStoreManager",
    "create_vector_store_manager",
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
    """Resolve the backend managers on first access."""
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
