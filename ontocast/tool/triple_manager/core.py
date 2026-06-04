"""Triple store management tools for OntoCast.

This module provides functionality for managing RDF triple stores, including
abstract interfaces and concrete implementations for different triple store backends.
"""

import abc
import asyncio
import os
from typing import Any, ClassVar

from pydantic import Field
from rdflib import RDF, Graph

from ontocast.onto.constants import PROV, RDF_REIFIES, SCHEMA
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool import Tool


class TripleStoreManager(Tool):
    _PROVENANCE_METADATA_PREDICATES: ClassVar[set] = {
        PROV.generatedAtTime,
        SCHEMA.position,
        SCHEMA.identifier,
    }

    """Base class for managing RDF triple stores.

    This class defines the interface for triple store management operations,
    including fetching and storing ontologies and their graphs. All concrete
    triple store implementations should inherit from this class.

    This is an abstract base class that must be implemented by specific
    triple store backends (e.g., Neo4j, Fuseki, Filesystem).
    """

    def __init__(self, **kwargs):
        """Initialize the triple store manager.

        Args:
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)

    @abc.abstractmethod
    def fetch_ontologies(self) -> list[Ontology]:
        """Fetch all available ontologies from the triple store.

        This method should retrieve all ontologies stored in the triple store
        and return them as Ontology objects with their associated RDF graphs.

        Returns:
            list[Ontology]: List of available ontologies with their graphs.
        """
        return []

    async def afetch_ontologies(self) -> list[Ontology]:
        """Async fetch helper for backends without native async I/O."""
        return await asyncio.to_thread(self.fetch_ontologies)

    @abc.abstractmethod
    def serialize_graph(self, graph: Graph, **kwargs) -> bool | dict[str, Any] | None:
        """Store an RDF graph in the triple store.

        This method should store the given RDF graph in the triple store.
        The implementation may choose how to organize the storage (e.g., as named graphs,
        in specific collections, etc.).

        Args:
            graph: The RDF graph to store.
            **kwargs: Implementation-specific arguments (e.g., fname for filesystem, graph_uri for Fuseki).

        Returns:
            bool | None: Implementation-specific return value (bool for Fuseki, summary for Neo4j, None for Filesystem).
        """
        pass

    @abc.abstractmethod
    def serialize(
        self, o: Ontology | RDFGraph, **kwargs
    ) -> bool | dict[str, Any] | None:
        """Store an RDF graph in the triple store.

        This method should store the given RDF graph in the triple store.
        The implementation may choose how to organize the storage (e.g., as named graphs,
        in specific collections, etc.).

        Args:
            o: RDF graph or Ontology object to store.
            **kwargs: Implementation-specific arguments (e.g., graph_uri for Fuseki).

        Returns:
            bool | None: Implementation-specific return value (bool for Fuseki, summary for Neo4j, None for Filesystem).
        """
        pass

    async def aserialize(
        self, o: Ontology | RDFGraph, **kwargs
    ) -> bool | dict[str, Any] | None:
        """Async serialize helper for backends without native async I/O."""
        return await asyncio.to_thread(self.serialize, o, **kwargs)

    @classmethod
    def _provenance_source_nodes(cls, graph: Graph) -> set:
        """Return chunk/source nodes whose triples are provenance scaffolding."""
        derived_from = set(graph.objects(None, PROV.wasDerivedFrom))
        entity_nodes = set(graph.subjects(RDF.type, PROV.Entity))
        text_chunk_nodes = set(graph.subjects(RDF.type, SCHEMA.text))
        chunk_metadata_nodes = entity_nodes & text_chunk_nodes
        return derived_from | chunk_metadata_nodes

    @classmethod
    def strip_provenance(cls, graph: Graph) -> RDFGraph:
        """Return a graph without reification/provenance scaffolding triples."""
        clean = RDFGraph()
        for prefix, namespace in graph.namespaces():
            clean.bind(prefix, namespace)

        reifier_nodes = set(graph.subjects(RDF_REIFIES, None))
        source_nodes = cls._provenance_source_nodes(graph)

        for subject, predicate, object_ in graph:
            if predicate in {RDF_REIFIES, PROV.wasDerivedFrom}:
                continue
            if subject in reifier_nodes:
                continue
            if subject in source_nodes:
                continue
            clean.add((subject, predicate, object_))

        return clean

    @abc.abstractmethod
    async def clean(self) -> None:
        """Clean/flush data managed by this store (backend-specific scope).

        Warning: This operation is irreversible and will delete data.

        Raises:
            NotImplementedError: If the triple store doesn't support cleaning.
        """
        raise NotImplementedError("clean() method must be implemented by subclasses")

    def supports_tenancy_partition(self) -> bool:
        """True if this backend isolates facts/ontologies by :func:`tenant_project_*` names."""
        return False

    async def clean_tenancy(self, tenant: str, project: str) -> None:
        """Remove all triples for datasets derived from ``tenant`` / ``project``.

        Backends without per-tenant partitions raise :class:`NotImplementedError`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not isolate data by tenant/project"
        )


class TripleStoreManagerWithAuth(TripleStoreManager):
    """Base class for triple store managers that require authentication.

    This class provides common functionality for triple store managers that
    need URI and authentication credentials. It handles environment variable
    loading and credential parsing.

    Attributes:
        uri: The connection URI for the triple store.
        auth: Authentication tuple (username, password) for the triple store.
    """

    uri: str | None = Field(default=None, description="Triple store connection URI")
    auth: tuple | None = Field(
        default=None, description="Triple store authentication tuple (user, password)"
    )

    def __init__(self, uri=None, auth=None, env_uri=None, env_auth=None, **kwargs):
        """Initialize the triple store manager with authentication.

        This method handles loading URI and authentication credentials from
        either direct parameters or environment variables. It also parses
        authentication strings in the format "user/password".

        Args:
            uri: Direct URI for the triple store connection.
            auth: Direct authentication tuple or string in "user/password" format.
            env_uri: Environment variable name for the URI (e.g., "NEO4J_URI").
            env_auth: Environment variable name for authentication (e.g., "NEO4J_AUTH").
            **kwargs: Additional keyword arguments passed to the parent class.

        Raises:
            ValueError: If authentication string is not in "user/password" format.

        Example:
            >>> manager = TripleStoreManagerWithAuth(
            ...     env_uri="NEO4J_URI",
            ...     env_auth="NEO4J_AUTH"
            ... )
        """
        # Use env vars if not provided
        uri = uri or (os.getenv(env_uri) if env_uri else None)
        auth_env = auth or (os.getenv(env_auth) if env_auth else None)

        if auth_env and not isinstance(auth_env, tuple):
            if "/" in auth_env:
                user, password = auth_env.split("/", 1)
                auth = (user, password)
            else:
                raise ValueError(
                    f"{env_auth or 'TRIPLESTORE_AUTH'} must be in 'user/password' format"
                )
        elif isinstance(auth_env, tuple):
            auth = auth_env
        # else: auth remains None

        super().__init__(uri=uri, auth=auth, **kwargs)
