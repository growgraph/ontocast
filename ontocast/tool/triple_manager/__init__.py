"""Triple store management package for OntoCast.

This package provides a unified interface for managing RDF triple stores
across different backends. It includes abstract base classes and concrete
implementations for various triple store technologies.

The package supports:
- Abstract interfaces for triple store operations
- Neo4j implementation using the n10s plugin
- Fuseki implementation using Apache Fuseki
- Filesystem implementation for local storage

All implementations support:
- Fetching and storing ontologies
- Serializing and retrieving facts
- Authentication and connection management
- Error handling and logging

Example:
    >>> from ontocast.tool.triple_manager import Neo4jTripleStoreManager
    >>> manager = Neo4jTripleStoreManager(uri="bolt://localhost:7687")
    >>> ontologies = manager.fetch_ontologies()
"""

from .core import (
    TripleStoreManager,
)
from .filesystem_manager import (
    FilesystemTripleStoreManager,
)
from .fuseki import (
    FusekiTripleStoreManager,
)
from .mock import (
    MockFusekiTripleStoreManager,
    MockNeo4jTripleStoreManager,
    MockTripleStoreManager,
)
from .neo4j import (
    Neo4jTripleStoreManager,
)

__all__ = [
    "TripleStoreManager",
    "Neo4jTripleStoreManager",
    "FusekiTripleStoreManager",
    "FilesystemTripleStoreManager",
    "MockTripleStoreManager",
    "MockFusekiTripleStoreManager",
    "MockNeo4jTripleStoreManager",
]
