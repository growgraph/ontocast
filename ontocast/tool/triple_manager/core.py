"""Triple store management tools for OntoCast.

This module provides functionality for managing RDF triple stores, including
abstract interfaces and concrete implementations for different triple store backends.
"""

import abc
import os
from typing import Optional

from pydantic import Field
from rdflib import Graph

from ontocast.onto.ontology import Ontology
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
    def serialize_ontology(self, o: Ontology, **kwargs):
        """Store an ontology in the triple store.

        This method should store the given ontology and its associated RDF graph
        in the triple store. The implementation may choose how to organize
        the storage (e.g., as named graphs, in specific collections, etc.).

        Args:
            o: The ontology to store.
            **kwargs: Additional keyword arguments for serialization.
        """
        pass

    @abc.abstractmethod
    def serialize_facts(self, g: Graph, **kwargs):
        """Store a graph with facts in the triple store.

        This method should store the given RDF graph containing facts
        in the triple store. The implementation may choose how to organize
        the storage (e.g., as named graphs, in specific collections, etc.).

        Args:
            g: The RDF graph containing facts to store.
            **kwargs: Additional keyword arguments for serialization.
        """
        pass


class TripleStoreManagerWithAuth(TripleStoreManager):
    """Base class for triple store managers that require authentication.

    This class provides common functionality for triple store managers that
    need URI and authentication credentials. It handles environment variable
    loading and credential parsing.

    Attributes:
        uri: The connection URI for the triple store.
        auth: Authentication tuple (username, password) for the triple store.
    """

    uri: Optional[str] = Field(default=None, description="Triple store connection URI")
    auth: Optional[tuple] = Field(
        default=None, description="Triple store authentication tuple (user, password)"
    )
    clean: bool = Field(
        default=False, description="If True, clean the database on init."
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
