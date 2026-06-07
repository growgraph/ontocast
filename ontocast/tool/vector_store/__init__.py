"""Vector store package for ontology patch retrieval."""

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
from .lancedb import LanceDBVectorStoreManager
from .patch_retriever import OntologyPatchRetriever
from .qdrant import QdrantVectorStoreManager
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
    "QdrantVectorStoreManager",
    "LanceDBVectorStoreManager",
    "VectorStoreManager",
    "create_vector_store_manager",
]
