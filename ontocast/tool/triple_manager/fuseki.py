"""Fuseki triple store management for OntoCast.

This module provides a concrete implementation of triple store management
using Apache Fuseki as the backend. It supports named graphs for ontologies
and facts, with proper authentication and dataset management.
"""

import asyncio
import logging
import re
from collections import defaultdict
from urllib.parse import quote

import httpx
from pydantic import Field
from rdflib import Graph
from rdflib.namespace import OWL, RDF

from ontocast.onto.constants import DEFAULT_DATASET, DEFAULT_ONTOLOGIES_DATASET
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.triple_manager.core import TripleStoreManagerWithAuth

logger = logging.getLogger(__name__)


def deterministic_turtle_serialization(graph: Graph) -> str:
    """Create a deterministic Turtle serialization of an RDF graph.

    This function ensures that the same graph content will always produce
    the same Turtle output, regardless of the order triples were added or
    how they're stored in Fuseki. This is crucial for caching to work
    correctly.

    Args:
        graph: The RDF graph to serialize.

    Returns:
        str: Deterministically serialized Turtle string.
    """
    # Capture and sort namespaces
    prefix_lines = [
        f"@prefix {p}: <{ns}> ."
        for p, ns in sorted(graph.namespace_manager.namespaces())
    ]

    # Sort triples by their string representation
    triples_sorted = sorted(graph, key=lambda t: (str(t[0]), str(t[1]), str(t[2])))

    # Serialize triples using n3 format to get proper Turtle syntax
    triple_lines = [
        f"{s.n3(graph.namespace_manager)} {p.n3(graph.namespace_manager)} {o.n3(graph.namespace_manager)} ."
        for s, p, o in triples_sorted
    ]

    # Return sorted prefixes followed by sorted triples
    return "\n".join(prefix_lines + [""] + triple_lines)


def _compare_versions(ver1: str, ver2: str) -> int:
    """Compare two semantic version strings.

    Args:
        ver1: First version string (e.g., "1.2.3")
        ver2: Second version string (e.g., "1.3.0")

    Returns:
        int: Negative if ver1 < ver2, 0 if equal, positive if ver1 > ver2
    """

    def _parse_version(v: str) -> tuple:
        # Simple version parser - splits by dots and converts to int
        parts = v.split(".")
        result = []
        for part in parts:
            # Remove any non-numeric suffix
            numeric_part = re.sub(r"[^0-9].*$", "", part)
            result.append(int(numeric_part) if numeric_part else 0)
        # Pad to 3 components
        while len(result) < 3:
            result.append(0)
        return tuple(result)

    try:
        v1_parts = _parse_version(ver1)
        v2_parts = _parse_version(ver2)
        if v1_parts < v2_parts:
            return -1
        elif v1_parts > v2_parts:
            return 1
        return 0
    except Exception:
        # If parsing fails, use string comparison
        return 1 if ver1 > ver2 else (-1 if ver1 < ver2 else 0)


class FusekiTripleStoreManager(TripleStoreManagerWithAuth):
    """Fuseki-based triple store manager.

    This class provides a concrete implementation of triple store management
    using Apache Fuseki. It stores ontologies as named graphs using their
    URIs as graph names, and supports dataset creation and cleanup.

    The manager uses Fuseki's REST API for all operations, including:
    - Dataset creation and management
    - Named graph operations for ontologies
    - SPARQL queries for ontology discovery
    - Graph-level data operations

    Attributes:
        dataset: The Fuseki dataset name to use for storage.
        clean: Whether to clean the dataset on initialization.
    """

    dataset: str | None = Field(default=None, description="Fuseki dataset name")
    ontologies_dataset: str = Field(
        default=DEFAULT_ONTOLOGIES_DATASET,
        description="Fuseki dataset name for ontologies",
    )

    def __init__(
        self,
        uri=None,
        auth=None,
        dataset=None,
        ontologies_dataset=None,
        **kwargs,
    ):
        """Initialize the Fuseki triple store manager.

        This method sets up the connection to Fuseki and creates the dataset
        if it doesn't exist. The dataset is NOT cleaned on initialization.

        Args:
            uri: Fuseki server URI (e.g., "http://localhost:3030").
            auth: Authentication tuple (username, password) or string in "user/password" format.
            dataset: Dataset name to use for storage.
            ontologies_dataset: Dataset name for ontologies (defaults to separate dataset).
            **kwargs: Additional keyword arguments passed to the parent class.

        Raises:
            ValueError: If dataset is not specified in URI or as argument.

        Example:
            >>> manager = FusekiTripleStoreManager(
            ...     uri="http://localhost:3030",
            ...     dataset="test"
            ... )
            >>> # To clean the dataset, use the clean() method explicitly:
            >>> await manager.clean()
        """
        super().__init__(
            uri=uri, auth=auth, env_uri="FUSEKI_URI", env_auth="FUSEKI_AUTH", **kwargs
        )
        if dataset is None:
            self.dataset = DEFAULT_DATASET
        else:
            self.dataset = dataset
        self.ontologies_dataset = ontologies_dataset or DEFAULT_ONTOLOGIES_DATASET

        # Initialize httpx client for async operations
        self._client: httpx.AsyncClient | None = None

        # Initialize datasets synchronously (for backward compatibility)
        # In async contexts, use async_init() instead
        asyncio.run(self._async_init_with_cleanup())

    async def _async_init_with_cleanup(self):
        """Wrapper for async_init that ensures proper cleanup when using asyncio.run().

        This method creates a temporary client and ensures it's properly closed
        before returning, preventing "Event loop is closed" errors.
        """
        async with httpx.AsyncClient(
            auth=self._prepare_auth(), timeout=30.0
        ) as temp_client:
            # Temporarily replace the client
            original_client = self._client
            self._client = temp_client
            try:
                await self._async_init()
            finally:
                # Restore original client
                self._client = original_client

    async def _async_init(self):
        """Async initialization of datasets."""
        await self.init_dataset(self.dataset)
        if self.ontologies_dataset != self.dataset:
            await self.init_dataset(self.ontologies_dataset)

    def _prepare_auth(self) -> httpx.BasicAuth | None:
        """Prepare httpx BasicAuth from self.auth.

        Returns:
            httpx.BasicAuth instance or None if no auth is configured.
        """
        if self.auth:
            if isinstance(self.auth, tuple):
                return httpx.BasicAuth(*self.auth)
            elif isinstance(self.auth, str) and "/" in self.auth:
                parts = self.auth.split("/", 1)
                if len(parts) == 2:
                    username, password = parts[0], parts[1]
                    return httpx.BasicAuth(username, password)
        return None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the httpx async client."""
        if self._client is None:
            auth = self._prepare_auth()
            self._client = httpx.AsyncClient(auth=auth, timeout=30.0)
        return self._client

    async def close(self):
        """Close the httpx client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def update_dataset(self, new_dataset: str) -> None:
        """Update the dataset name for this manager.

        This method allows changing the dataset without recreating the entire
        manager, which is useful for API requests that specify different datasets.

        Args:
            new_dataset: The new dataset name to use.
        """
        if not new_dataset:
            raise ValueError("Dataset name cannot be empty")

        self.dataset = new_dataset
        await self.init_dataset(self.dataset)
        logger.info(f"Updated Fuseki dataset to: {self.dataset}")

    async def clean(self, dataset: str | None = None) -> None:
        """Clean/flush data from Fuseki dataset(s).

        This method removes all named graphs and clears the default graph
        from the specified dataset, or all datasets if no dataset is specified.

        Args:
            dataset: Optional dataset name to clean. If None, cleans both the main
                dataset and the ontologies dataset. If specified, cleans only that dataset.

        Warning: This operation is irreversible and will delete all data
        from the specified dataset(s).

        The method handles errors gracefully and logs the results of
        each cleanup operation.

        Example:
            >>> # Clean all datasets
            >>> await manager.clean()
            >>> # Clean specific dataset
            >>> await manager.clean(dataset="my_dataset")
        """
        if dataset is None:
            # Clean all datasets (main and ontologies)
            # self.dataset is guaranteed to be a string (set to DEFAULT_DATASET if None in __init__)
            assert self.dataset is not None, "Dataset should never be None"
            await self._clean_dataset_by_name(self.dataset)
            logger.info(f"Fuseki dataset '{self.dataset}' cleaned (all data deleted)")

            # Also clean the ontologies dataset if it's different
            if self.ontologies_dataset != self.dataset:
                await self._clean_dataset_by_name(self.ontologies_dataset)
                logger.info(
                    f"Fuseki ontologies dataset '{self.ontologies_dataset}' cleaned (all data deleted)"
                )
        else:
            # Clean only the specified dataset
            await self._clean_dataset_by_name(dataset)
            logger.info(f"Fuseki dataset '{dataset}' cleaned (all data deleted)")

    async def _clean_dataset_by_name(self, dataset_name: str) -> None:
        """Clean a specific dataset by name.

        This is a helper method that performs the actual cleaning of a single dataset.
        It deletes all named graphs and clears the default graph.

        Uses a temporary client to avoid event loop cleanup issues when called
        from different async contexts.

        Args:
            dataset_name: Name of the dataset to clean.

        Raises:
            Exception: If the cleanup operation fails.
        """
        # Use a temporary client to avoid event loop cleanup issues
        async with httpx.AsyncClient(auth=self._prepare_auth(), timeout=30.0) as client:
            try:
                dataset_url = f"{self.uri}/{dataset_name}"
                sparql_update_url = f"{dataset_url}/update"
                sparql_url = f"{dataset_url}/sparql"

                # Delete all named graphs
                query = """
                SELECT DISTINCT ?g WHERE {
                  GRAPH ?g { ?s ?p ?o }
                }
                """
                response = await client.post(
                    sparql_url,
                    data={"query": query, "format": "application/sparql-results+json"},
                )

                if response.status_code == 200:
                    results = response.json()
                    tasks = []
                    for binding in results.get("results", {}).get("bindings", []):
                        graph_uri = binding["g"]["value"]
                        # Delete the named graph using SPARQL UPDATE
                        drop_query = f"DROP GRAPH <{graph_uri}>"
                        tasks.append(
                            client.post(
                                sparql_update_url,
                                data={"update": drop_query},
                            )
                        )

                    # Execute all deletions in parallel
                    delete_responses = await asyncio.gather(
                        *tasks, return_exceptions=True
                    )
                    for i, delete_response in enumerate(delete_responses):
                        graph_uri = results["results"]["bindings"][i]["g"]["value"]
                        if isinstance(delete_response, Exception):
                            logger.warning(
                                f"Failed to delete graph {graph_uri}: {delete_response}"
                            )
                        elif isinstance(delete_response, httpx.Response):
                            if delete_response.status_code in (200, 204):
                                logger.debug(f"Deleted named graph: {graph_uri}")
                            else:
                                logger.warning(
                                    f"Failed to delete graph {graph_uri}: {delete_response.status_code}"
                                )

                # Clear the default graph using SPARQL UPDATE
                clear_query = "CLEAR DEFAULT"
                clear_response = await client.post(
                    sparql_update_url,
                    data={"update": clear_query},
                )
                if clear_response.status_code in (200, 204):
                    logger.debug(f"Cleared default graph in dataset '{dataset_name}'")
                else:
                    logger.warning(
                        f"Failed to clear default graph in dataset '{dataset_name}': {clear_response.status_code}"
                    )
            except Exception as e:
                logger.error(f"Failed to clean dataset '{dataset_name}': {e}")
                raise

    async def init_dataset(self, dataset_name):
        """Initialize a Fuseki dataset.

        This method creates a new dataset in Fuseki if it doesn't already exist.
        It uses Fuseki's admin API to create the dataset with TDB2 storage.

        Uses a temporary client to avoid event loop cleanup issues when called
        from different async contexts.

        Args:
            dataset_name: Name of the dataset to create.

        Note:
            This method will not fail if the dataset already exists.
        """
        # Use a temporary client to avoid event loop cleanup issues
        async with httpx.AsyncClient(auth=self._prepare_auth(), timeout=30.0) as client:
            fuseki_admin_url = f"{self.uri}/$/datasets"

            payload = {"dbName": dataset_name, "dbType": "tdb2"}

            headers = {"Content-Type": "application/x-www-form-urlencoded"}

            response = await client.post(
                fuseki_admin_url, data=payload, headers=headers
            )

            if response.status_code == 200 or response.status_code == 201:
                logger.info(f"Fuseki dataset '{dataset_name}' created successfully.")
            elif response.status_code == 409:
                logger.info(
                    f"Fuseki status code: {response.status_code}; {response.text.strip()}"
                )
            else:
                logger.error(
                    f"Failed to create dataset {dataset_name}. Status code: {response.status_code}"
                )
                logger.error(f"Response: {response.text.strip()}")

    def _get_dataset_url(self):
        """Get the full URL for the dataset.

        Returns:
            str: The complete URL for the dataset endpoint.
        """
        return f"{self.uri}/{self.dataset}"

    def _get_ontologies_dataset_url(self):
        """Get the full URL for the ontologies dataset.

        Returns:
            str: The complete URL for the ontologies dataset endpoint.
        """
        return f"{self.uri}/{self.ontologies_dataset}"

    def fetch_ontologies(self) -> list[Ontology]:
        """Synchronous wrapper for fetch_ontologies.

        For async usage, use afetch_ontologies() instead.
        """
        # Use a temporary client for this operation to avoid event loop cleanup issues
        return asyncio.run(self._fetch_ontologies_with_cleanup())

    async def afetch_ontologies(self) -> list[Ontology]:
        """Async version of fetch_ontologies.

        This is the preferred method when running in an async context.
        """
        return await self._fetch_ontologies_async()

    async def _fetch_ontologies_with_cleanup(self) -> list[Ontology]:
        """Wrapper that ensures proper cleanup when using asyncio.run().

        This method creates a temporary client and ensures it's properly closed
        before returning, preventing "Event loop is closed" errors.
        """
        async with httpx.AsyncClient(
            auth=self._prepare_auth(), timeout=30.0
        ) as temp_client:
            # Temporarily replace the client
            original_client = self._client
            self._client = temp_client
            try:
                return await self._fetch_ontologies_async()
            finally:
                # Restore original client
                self._client = original_client

    async def _fetch_ontologies_async(self) -> list[Ontology]:
        """Fetch all ontologies from their corresponding named graphs.

        This method discovers all ontologies in the Fuseki ontologies dataset and
        fetches each one from its corresponding named graph. For versioned ontologies,
        it returns only the latest version for each unique ontology IRI.

        1. Discovery: List all named graphs (which may be versioned URIs)
        2. Fetching: Retrieve each ontology from its named graph (in parallel)
        3. Deduplication: For versioned ontologies, keep only the latest version

        Returns:
            list[Ontology]: List of the latest version of each ontology found.

        Example:
            >>> ontologies = await manager.fetch_ontologies()
            >>> for onto in ontologies:
            ...     print(f"Found ontology: {onto.iri} v{onto.version}")
        """
        client = await self._get_client()
        sparql_url = f"{self._get_ontologies_dataset_url()}/sparql"

        # Step 1: List all named graphs
        list_query = """
        SELECT DISTINCT ?g WHERE {
          GRAPH ?g { ?s ?p ?o }
        }
        """
        response = await client.post(
            sparql_url,
            data={"query": list_query, "format": "application/sparql-results+json"},
        )
        if response.status_code != 200:
            logger.error(f"Failed to list graphs from Fuseki: {response.text}")
            return []

        results = response.json()
        graph_uris = []
        for binding in results.get("results", {}).get("bindings", []):
            graph_uri = binding["g"]["value"]
            graph_uris.append(graph_uri)

        logger.debug(f"Found {len(graph_uris)} named graphs: {graph_uris}")

        # Step 2: Fetch each ontology from its corresponding named graph (in parallel)
        async def fetch_single_ontology(graph_uri: str) -> Ontology | None:
            """Fetch a single ontology from a graph URI."""
            try:
                graph = RDFGraph()
                # URL encode the graph URI to handle special characters like #
                encoded_graph_uri = quote(str(graph_uri), safe="/:")
                export_url = f"{self._get_ontologies_dataset_url()}/get?graph={encoded_graph_uri}"
                export_resp = await client.get(
                    export_url, headers={"Accept": "text/turtle"}
                )

                if export_resp.status_code == 200:
                    graph.parse(data=export_resp.text, format="turtle")

                    # Re-serialize deterministically to ensure consistent cache keys
                    # This sorts both namespaces and triples alphabetically
                    deterministic_turtle = deterministic_turtle_serialization(graph)

                    # Re-parse from deterministic serialization to ensure we have RDFGraph
                    deterministic_graph = RDFGraph()
                    deterministic_graph.parse(
                        data=deterministic_turtle, format="turtle"
                    )

                    # Copy namespace bindings from original graph
                    for prefix, namespace in graph.namespaces():
                        if prefix:
                            deterministic_graph.bind(prefix, namespace)

                    graph = deterministic_graph

                    # Find the ontology IRI in the graph
                    for onto_subj, _, obj in graph.triples(
                        (None, RDF.type, OWL.Ontology)
                    ):
                        onto_iri = str(onto_subj)
                        # Extract base IRI if graph_uri is versioned
                        # Handle both hash fragments (#19193944...) and semantic versions (#v1.2.3)
                        if "#" in graph_uri:
                            base_iri = graph_uri.split("#")[0]
                            # Use base IRI from graph_uri (named graph identifier)
                            # The graph content should have simplified IRI, but use graph_uri as source of truth
                            onto_iri = base_iri

                        ontology = Ontology(
                            graph=graph,
                            iri=onto_iri,
                        )
                        # Load properties from graph (will strip any hash fragments if present)
                        ontology.sync_properties_from_graph()
                        logger.debug(
                            f"Successfully loaded ontology: {onto_iri} version: {ontology.version}"
                        )
                        return ontology
                else:
                    logger.warning(
                        f"Failed to fetch graph {graph_uri}: {export_resp.status_code}"
                    )
            except Exception as e:
                logger.warning(f"Error fetching ontology from {graph_uri}: {e}")
            return None

        # Fetch all ontologies in parallel
        all_ontologies_results = await asyncio.gather(
            *[fetch_single_ontology(uri) for uri in graph_uris], return_exceptions=True
        )

        # Filter out None and exceptions
        all_ontologies = []
        for result in all_ontologies_results:
            if isinstance(result, Exception):
                logger.warning(f"Exception fetching ontology: {result}")
            elif result is not None:
                all_ontologies.append(result)

        # Step 3: Deduplicate and keep latest terminal versions
        ontology_dict = defaultdict(list)

        for onto in all_ontologies:
            ontology_dict[onto.iri].append(onto)

        # Build set of all parent hashes to identify terminal ontologies
        # A terminal ontology is one that is not a parent for any other ontology
        all_parent_hashes = set()

        for onto in all_ontologies:
            if onto.hash:
                # Collect all parent hashes
                for parent_hash in onto.parent_hashes:
                    all_parent_hashes.add(parent_hash)

        # For each unique IRI, select the latest terminal ontology
        ontologies = []

        for iri, versions in ontology_dict.items():
            if len(versions) == 1:
                ontologies.append(versions[0])
            else:
                # Multiple versions - find terminal ontologies (not parents)
                terminal_versions = [
                    v for v in versions if v.hash and v.hash not in all_parent_hashes
                ]

                if not terminal_versions:
                    # No terminal ontologies found - all are parents
                    # Fall back to non-terminal versions
                    logger.warning(
                        f"No terminal ontologies found for {iri}, "
                        f"using all versions for selection"
                    )
                    terminal_versions = versions

                # Select latest by created_at among terminal ontologies
                try:
                    versions_with_created = [
                        v for v in terminal_versions if v.created_at is not None
                    ]

                    if versions_with_created:
                        # Sort by created_at (most recent first)
                        versions_with_created.sort(
                            key=lambda x: x.created_at, reverse=True
                        )
                        selected = versions_with_created[0]
                        hash_str = (
                            f"{selected.hash[:16]}..." if selected.hash else "no hash"
                        )
                        logger.debug(
                            f"Selected terminal ontology for {iri} "
                            f"by created_at: {selected.created_at} "
                            f"(hash: {hash_str})"
                        )
                        ontologies.append(selected)
                    else:
                        # No created_at available - fall back to version-based sorting
                        versions_with_ver = [v for v in terminal_versions if v.version]
                        if versions_with_ver:
                            versions_with_ver.sort(
                                key=lambda x: str(x.version), reverse=False
                            )
                            selected = versions_with_ver[-1]
                            logger.debug(
                                f"Selected terminal ontology for {iri} "
                                f"by version: {selected.version} "
                                f"(no created_at available)"
                            )
                            ontologies.append(selected)
                        else:
                            # No version info either - use first terminal ontology
                            selected = terminal_versions[0]
                            logger.debug(
                                f"Selected first terminal ontology for {iri} "
                                f"(no created_at or version available)"
                            )
                            ontologies.append(selected)
                except Exception as e:
                    logger.warning(
                        f"Could not select terminal ontology for {iri}: {e}, "
                        f"using first version"
                    )
                    ontologies.append(terminal_versions[0])

        logger.info(
            f"Successfully loaded {len(ontologies)} unique ontologies from Fuseki "
        )
        return ontologies

    def serialize_graph(self, graph: Graph, **kwargs) -> bool | None:
        """Synchronous wrapper for serialize_graph.

        For async usage, use aserialize_graph() instead.
        """
        return asyncio.run(self._serialize_graph_with_cleanup(graph, **kwargs))

    async def aserialize_graph(self, graph: Graph, **kwargs) -> bool | None:
        """Async version of serialize_graph.

        This is the preferred method when running in an async context.
        """
        return await self._serialize_graph_async(graph, **kwargs)

    async def _serialize_graph_with_cleanup(
        self, graph: Graph, **kwargs
    ) -> bool | None:
        """Wrapper that ensures proper cleanup when using asyncio.run().

        This method creates a temporary client and ensures it's properly closed
        before returning, preventing "Event loop is closed" errors.
        """
        async with httpx.AsyncClient(
            auth=self._prepare_auth(), timeout=30.0
        ) as temp_client:
            # Temporarily replace the client
            original_client = self._client
            self._client = temp_client
            try:
                return await self._serialize_graph_async(graph, **kwargs)
            finally:
                # Restore original client
                self._client = original_client

    async def _serialize_graph_async(self, graph: Graph, **kwargs) -> bool | None:
        """Store an RDF graph as a named graph in a specific Fuseki dataset.

        This is a private helper method that handles the common logic for storing
        graphs in Fuseki datasets.

        Args:
            graph: The RDF graph to store.
            **kwargs: Additional parameters including graph_uri, dataset_url, default_graph_uri, log_prefix.

        Returns:
            bool: True if the graph was successfully stored, False otherwise.
        """
        client = await self._get_client()
        graph_uri = kwargs.get("graph_uri")
        dataset_url = kwargs.get("dataset_url")
        default_graph_uri = kwargs.get("default_graph_uri")
        log_prefix = kwargs.get("log_prefix")

        turtle_data = graph.serialize(format="turtle")
        if graph_uri is None:
            graph_uri = default_graph_uri

        # URL encode the graph URI to handle special characters like #
        encoded_graph_uri = quote(str(graph_uri), safe="/:")
        url = f"{dataset_url}/data?graph={encoded_graph_uri}"
        headers = {"Content-Type": "text/turtle;charset=utf-8"}
        response = await client.put(url, headers=headers, content=turtle_data)
        if response.status_code in (200, 201, 204):
            logger.info(
                f"{log_prefix} graph {graph_uri} uploaded to Fuseki as named graph."
            )
            return True
        else:
            logger.error(
                f"Failed to upload {log_prefix.lower() if log_prefix else 'unknown'} graph {graph_uri}. Status code: {response.status_code}"
            )
            logger.error(f"Response: {response.text}")
            return False

    def serialize(self, o: Ontology | RDFGraph, **kwargs) -> bool | None:
        """Synchronous wrapper for serialize.

        For async usage, use aserialize() instead.
        """
        return asyncio.run(self._serialize_with_cleanup(o, **kwargs))

    async def aserialize(self, o: Ontology | RDFGraph, **kwargs) -> bool | None:
        """Async version of serialize.

        This is the preferred method when running in an async context.
        """
        return await self._serialize_async(o, **kwargs)

    async def _serialize_with_cleanup(
        self, o: Ontology | RDFGraph, **kwargs
    ) -> bool | None:
        """Wrapper that ensures proper cleanup when using asyncio.run().

        This method creates a temporary client and ensures it's properly closed
        before returning, preventing "Event loop is closed" errors.
        """
        async with httpx.AsyncClient(
            auth=self._prepare_auth(), timeout=30.0
        ) as temp_client:
            # Temporarily replace the client
            original_client = self._client
            self._client = temp_client
            try:
                return await self._serialize_async(o, **kwargs)
            finally:
                # Restore original client
                self._client = original_client

    async def _serialize_async(self, o: Ontology | RDFGraph, **kwargs) -> bool | None:
        """Store an RDF graph as a named graph in Fuseki.

        This method stores the given RDF graph as a named graph in Fuseki.
        The graph name is taken from the graph_uri parameter or defaults to
        "urn:data:default".

        Args:
            o: RDF graph or Ontology object.
            **kwargs: Additional parameters including graph_uri.

        Returns:
            bool: True if the graph was successfully stored, False otherwise.

        Example:
            >>> graph = RDFGraph()
            >>> success = await manager.serialize(graph)

            >>> success = await manager.serialize(graph, graph_uri="http://example.org/chunk1")
        """
        graph_uri = kwargs.get("graph_uri")

        if isinstance(o, Ontology):
            graph = o.graph
            # Use versioned IRI for storage to enable multiple versions to coexist
            graph_uri = o.versioned_iri
            default_graph_uri = "urn:ontology:default"
            log_prefix = "Ontology"
            # Use ontologies dataset for ontology storage
            dataset_url = self._get_ontologies_dataset_url()
        elif isinstance(o, RDFGraph):
            graph = o
            default_graph_uri = "urn:data:default"
            log_prefix = "Graph"
            # Use regular dataset for facts storage
            dataset_url = self._get_dataset_url()
        else:
            raise TypeError(f"unsupported obj of type {type(o)} received")

        return await self._serialize_graph_async(
            graph=graph,
            graph_uri=graph_uri,
            dataset_url=dataset_url,
            default_graph_uri=default_graph_uri,
            log_prefix=log_prefix,
        )
