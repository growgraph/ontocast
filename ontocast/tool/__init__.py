"""Tool package for OntoCast.

This package provides a collection of tools that support the OntoCast workflow,
including document processing, ontology management, triple store operations,
and LLM interactions.

The package includes:
- LLMTool: Language model interaction and prompting
- OntologyManager: Ontology loading and management
- TripleStoreManager: Abstract interface for triple store operations
- FusekiTripleStoreManager: Fuseki-specific triple store implementation (preferred)
- Neo4jTripleStoreManager: Neo4j-specific triple store implementation
- FilesystemTripleStoreManager: Filesystem-based triple store implementation
- ConverterTool: Document format conversion utilities
- ChunkerTool: Text chunking and segmentation

All tools inherit from the base Tool class and provide standardized
interfaces for integration into the OntoCast workflow.

Example:
    >>> from ontocast.tool import LLMTool, OntologyManager
    >>> llm = LLMTool.create(provider="openai", model="gpt-4")
    >>> om = OntologyManager()
"""

from ontocast.tool.chunk.chunker import ChunkerTool

from .agg.aggregate import EmbeddingBasedAggregator
from .atomic import AtomicToolBox, SearchHit
from .converter import ConverterTool
from .llm import LLMTool
from .onto import Tool
from .ontology_manager import OntologyManager
from .triple_manager import (
    FilesystemTripleStoreManager,
    FusekiTripleStoreManager,
    Neo4jTripleStoreManager,
    TripleStoreManager,
)
from .vector_store import EmbeddingTool, OntologyPatchRetriever, QdrantVectorStore

__all__ = [
    "LLMTool",
    "OntologyManager",
    "TripleStoreManager",
    "FusekiTripleStoreManager",
    "Neo4jTripleStoreManager",
    "FilesystemTripleStoreManager",
    "ConverterTool",
    "ChunkerTool",
    "Tool",
    "AtomicToolBox",
    "SearchHit",
    "EmbeddingTool",
    "QdrantVectorStore",
    "OntologyPatchRetriever",
    "EmbeddingBasedAggregator",
]
