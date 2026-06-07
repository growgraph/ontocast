"""Triple store management package for OntoCast."""

from .core import TripleStoreManager
from .fuseki import (
    FusekiTripleStoreManager,
    normalize_fuseki_server_uri,
)
from .in_memory import InMemoryTripleStoreManager
from .mock import (
    MockFusekiTripleStoreManager,
    MockInMemoryTripleStoreManager,
    MockTripleStoreManager,
)
from .util import deterministic_turtle_serialization

TripleManager = TripleStoreManager


__all__ = [
    "TripleStoreManager",
    "TripleManager",
    "FusekiTripleStoreManager",
    "InMemoryTripleStoreManager",
    "MockTripleStoreManager",
    "MockFusekiTripleStoreManager",
    "MockInMemoryTripleStoreManager",
    "normalize_fuseki_server_uri",
    "deterministic_turtle_serialization",
]
