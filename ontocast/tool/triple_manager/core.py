"""Triple store management tools for OntoCast.

This module provides functionality for managing RDF triple stores, including
abstract interfaces and concrete implementations for different triple store backends.
"""

import abc
import os

from pydantic import Field
from rdflib import Graph

from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool import Tool


class TripleStoreManager(Tool):
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

    @abc.abstractmethod
    def serialize_graph(self, graph: Graph, **kwargs) -> bool | None:
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
    def serialize(self, o: Ontology | RDFGraph, **kwargs) -> bool | None:
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

    @abc.abstractmethod
    async def clean(self, dataset: str | None = None) -> None:
        """Clean/flush data from the triple store.

        This method removes data from the triple store. For Fuseki, the optional
        dataset parameter allows cleaning a specific dataset, or all datasets if None.
        For Neo4j and Filesystem, the dataset parameter is ignored.

        Args:
            dataset: Optional dataset name to clean (Fuseki only). If None, cleans
                all data. For other stores, this parameter is ignored.

        Warning: This operation is irreversible and will delete all data.

        Raises:
            NotImplementedError: If the triple store doesn't support cleaning.
        """
        raise NotImplementedError("clean() method must be implemented by subclasses")


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
