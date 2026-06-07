"""Fuseki triple store management for OntoCast.

This module provides a concrete implementation of triple store management
using Apache Fuseki as the backend. It supports named graphs for ontologies
and facts, with proper authentication and dataset management.
"""

import asyncio
import logging
from urllib.parse import quote, urlparse, urlunparse

import httpx
from pydantic import Field
from rdflib import Graph

from ontocast.onto.constants import DEFAULT_DATASET, DEFAULT_ONTOLOGIES_DATASET
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.tenancy import (
    TENANCY_SEP,
    tenant_project_facts_name,
    tenant_project_ontologies_name,
)
from ontocast.tool.triple_manager.core import TripleStoreManagerWithAuth
from ontocast.tool.triple_manager.util import (
    dedupe_terminal_ontologies,
    ontology_from_named_graph,
)

logger = logging.getLogger(__name__)


def normalize_fuseki_server_uri(raw: str | None) -> str | None:
    """Normalize ``FUSEKI_URI`` to the Fuseki HTTP service root.

    SPARQL and Graph Store HTTP endpoints are ``{base}/{dataset}/sparql``,
    ``{base}/{dataset}/update``, and so on. The Fuseki web UI links look like
    ``http://host:port/#/dataset/dataset_name``; the fragment is client-side only
    and must not be sent with API requests. Trailing slashes on the base URL are
    removed so ``{base}`` and ``{dataset}`` concatenate to correct paths.

    Args:
        raw: Connection URI (e.g. from ``FUSEKI_URI``).

    Returns:
        Normalized base URL, or ``None`` if ``raw`` is ``None``. Malformed values
        without scheme/netloc are returned unchanged (after ``strip``).
    """
    if raw is None:
        return None
    text = raw.strip()
    parsed = urlparse(text)
    if not parsed.scheme or not parsed.netloc:
        return text
    path = (parsed.path or "").rstrip("/")
    return urlunparse(
        (parsed.scheme, parsed.netloc, path, parsed.params, parsed.query, "")
    )


class FusekiTripleStoreManager(TripleStoreManagerWithAuth):
    """Fuseki-based triple store manager.

    This class provides a concrete implementation of triple store management
    using Apache Fuseki. It stores ontologies as named graphs using their
    URIs as graph names, and supports dataset creation and cleanup.

    **URI shape:** ``uri`` must be the Fuseki **HTTP server root** (e.g.
    ``http://localhost:3032``), not a dataset path or UI URL. Dataset names are
    ``dataset`` / ``ontologies_dataset``; the client calls
    ``{uri}/{dataset_name}/sparql`` and similar. The UI route
    ``/#/dataset/dataset_name`` is only for the browser; paste the origin (and
    optional non-dataset path prefix) into ``FUSEKI_URI``, and set
    ``FUSEKI_DATASET`` to ``dataset_name``.

    The manager uses Fuseki's REST API for all operations, including:
    - Dataset creation and management
    - Named graph operations for ontologies
    - SPARQL queries for ontology discovery
    - Graph-level data operations

    Attributes:
        dataset: Facts dataset name (first path segment in Fuseki HTTP API).
        ontologies_dataset: Ontologies dataset name.
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
            uri: Fuseki HTTP service root (e.g. ``http://localhost:3030``), not
                ``.../dataset/name`` and not a ``#/dataset/...`` UI link.
            auth: Authentication tuple (username, password) or string in "user/password" format.
            dataset: Facts dataset name (Fuseki API path segment).
            ontologies_dataset: Ontologies dataset name (separate Fuseki dataset).
            **kwargs: Additional keyword arguments passed to the parent class.

        Example:
            >>> manager = FusekiTripleStoreManager(
            ...     uri="http://localhost:3030",
            ...     dataset="acme--demo--facts",
            ...     ontologies_dataset="acme--demo--ontologies",
            ... )
            >>> await manager.clean()
        """
        super().__init__(
            uri=uri, auth=auth, env_uri="FUSEKI_URI", env_auth="FUSEKI_AUTH", **kwargs
        )
        self.uri = normalize_fuseki_server_uri(self.uri)
        if dataset is None:
            self.dataset = DEFAULT_DATASET
        else:
            self.dataset = dataset
        self.ontologies_dataset = ontologies_dataset or DEFAULT_ONTOLOGIES_DATASET

        # Initialize httpx client for async operations (recreated per event loop;
        # httpx.AsyncClient is bound to the loop it was created on).
        self._client: httpx.AsyncClient | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None

    async def async_init(self) -> None:
        """Initialize configured Fuseki datasets explicitly.

        Constructors stay side-effect free so callers can resolve tenancy first
        and then create datasets for the final dataset names.
        """
        # Use a temporary client to keep initialization independent from any
        # loop-bound long-lived client state.
        async with httpx.AsyncClient(
            auth=self._prepare_auth(), timeout=30.0
        ) as temp_client:
            # Temporarily replace the client
            original_client = self._client
            self._client = temp_client
            try:
                await self._initialize_datasets()
            finally:
                # Restore original client
                self._client = original_client

    async def _initialize_datasets(self) -> None:
        """Create configured facts/ontologies datasets when missing."""
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
        """Get or create the httpx async client for the current running event loop."""
        loop = asyncio.get_running_loop()
        if self._client is not None and self._client_loop is loop:
            return self._client
        # Client from a prior asyncio.run() is bound to a closed loop; do not await
        # aclose() on it (that schedules callbacks on the dead loop).
        self._client = None
        self._client_loop = None
        auth = self._prepare_auth()
        self._client = httpx.AsyncClient(auth=auth, timeout=30.0)
        self._client_loop = loop
        return self._client

    async def close(self):
        """Close the httpx client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._client_loop = None

    def supports_tenancy_partition(self) -> bool:
        return True

    async def update_tenancy(
        self,
        tenant: str,
        project: str,
        *,
        sep: str = TENANCY_SEP,
    ) -> None:
        """Switch facts and ontologies Fuseki datasets for ``tenant`` / ``project``."""
        facts = tenant_project_facts_name(tenant, project, sep=sep)
        ontos = tenant_project_ontologies_name(tenant, project, sep=sep)
        self.dataset = facts
        self.ontologies_dataset = ontos
        await self.init_dataset(self.dataset)
        if self.ontologies_dataset != self.dataset:
            await self.init_dataset(self.ontologies_dataset)
        logger.info(
            "Fuseki tenancy set to tenant=%r project=%r (facts=%s ontologies=%s)",
            tenant,
            project,
            self.dataset,
            self.ontologies_dataset,
        )

    async def clean(self) -> None:
        """Clear the configured facts dataset and ontologies dataset (when distinct)."""
        assert self.dataset is not None, "Dataset should never be None"
        await self._clean_dataset_by_name(self.dataset)
        logger.info("Fuseki dataset '%s' cleaned (all data deleted)", self.dataset)

        if self.ontologies_dataset != self.dataset:
            await self._clean_dataset_by_name(self.ontologies_dataset)
            logger.info(
                "Fuseki ontologies dataset '%s' cleaned (all data deleted)",
                self.ontologies_dataset,
            )

    async def clean_tenancy(
        self,
        tenant: str,
        project: str,
        *,
        sep: str = TENANCY_SEP,
    ) -> None:
        """Flush facts and ontologies datasets for ``tenant`` / ``project`` (by derived names)."""
        facts = tenant_project_facts_name(tenant, project, sep=sep)
        ontos = tenant_project_ontologies_name(tenant, project, sep=sep)
        await self._clean_dataset_by_name(facts)
        if ontos != facts:
            await self._clean_dataset_by_name(ontos)
        logger.info(
            "Fuseki tenancy flush tenant=%r project=%r (facts=%s ontologies=%s)",
            tenant,
            project,
            facts,
            ontos,
        )

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

    async def drop_named_graph(
        self, graph_uri: str, *, use_ontologies_dataset: bool = True
    ) -> None:
        """Drop a single named graph in the ontologies or main dataset."""
        dataset_url = (
            self._get_ontologies_dataset_url()
            if use_ontologies_dataset
            else self._get_dataset_url()
        )
        update_url = f"{dataset_url}/update"
        drop_query = f"DROP GRAPH <{graph_uri}>"
        async with httpx.AsyncClient(auth=self._prepare_auth(), timeout=30.0) as client:
            response = await client.post(update_url, data={"update": drop_query})
            if response.status_code not in (200, 204):
                logger.warning(
                    "Fuseki DROP GRAPH failed for %s: %s %s",
                    graph_uri,
                    response.status_code,
                    response.text,
                )

    async def drop_all_ontology_graphs_for_iri(self, ontology_iri: str) -> None:
        """Remove named graphs for ``ontology_iri`` (base and ``iri#...`` versioned)."""
        prefix = f"{ontology_iri}#"
        async with httpx.AsyncClient(auth=self._prepare_auth(), timeout=30.0) as client:
            sparql_url = f"{self._get_ontologies_dataset_url()}/sparql"
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
                logger.error(
                    "Failed to list graphs from Fuseki ontologies dataset: %s",
                    response.text,
                )
                return
            to_drop: list[str] = []
            for binding in response.json().get("results", {}).get("bindings", []):
                g = binding["g"]["value"]
                if g == ontology_iri or g.startswith(prefix):
                    to_drop.append(g)
            update_url = f"{self._get_ontologies_dataset_url()}/update"
            for graph_uri in to_drop:
                drop_query = f"DROP GRAPH <{graph_uri}>"
                dr = await client.post(update_url, data={"update": drop_query})
                if dr.status_code not in (200, 204):
                    logger.warning(
                        "Failed to drop graph %s: %s %s",
                        graph_uri,
                        dr.status_code,
                        dr.text,
                    )

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
                    return ontology_from_named_graph(graph_uri, graph)
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
        all_ontologies: list[Ontology] = []
        for result in all_ontologies_results:
            if isinstance(result, Exception):
                logger.warning(f"Exception fetching ontology: {result}")
            elif isinstance(result, Ontology):
                all_ontologies.append(result)

        ontologies = dedupe_terminal_ontologies(all_ontologies)
        logger.info(
            "Successfully loaded %d unique ontologies from Fuseki", len(ontologies)
        )
        return ontologies

    def serialize_graph(self, graph: Graph, **kwargs) -> bool:
        """Synchronous wrapper for serialize_graph.

        For async usage, use aserialize_graph() instead.
        """
        return asyncio.run(self._serialize_graph_with_cleanup(graph, **kwargs))

    async def aserialize_graph(self, graph: Graph, **kwargs) -> bool:
        """Async version of serialize_graph.

        This is the preferred method when running in an async context.
        """
        return await self._serialize_graph_async(graph, **kwargs)

    async def _serialize_graph_with_cleanup(self, graph: Graph, **kwargs) -> bool:
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

    async def _serialize_graph_async(self, graph: Graph, **kwargs) -> bool:
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

        if isinstance(graph, RDFGraph):
            turtle_data = graph.serialize_canonical_turtle()
        else:
            rdf_graph = RDFGraph()
            for triple in graph:
                rdf_graph.add(triple)
            for prefix, namespace in graph.namespaces():
                rdf_graph.bind(prefix, namespace)
            turtle_data = rdf_graph.serialize_canonical_turtle()
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

    def serialize(self, o: Ontology | RDFGraph, **kwargs) -> bool:
        """Synchronous wrapper for serialize.

        For async usage, use aserialize() instead.
        """
        return asyncio.run(self._serialize_with_cleanup(o, **kwargs))

    async def aserialize(self, o: Ontology | RDFGraph, **kwargs) -> bool:
        """Async version of serialize.

        This is the preferred method when running in an async context.
        """
        return await self._serialize_async(o, **kwargs)

    async def _serialize_with_cleanup(self, o: Ontology | RDFGraph, **kwargs) -> bool:
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

    async def _serialize_async(self, o: Ontology | RDFGraph, **kwargs) -> bool:
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
