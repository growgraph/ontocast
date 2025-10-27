"""Fuseki triple store management for OntoCast.

This module provides a concrete implementation of triple store management
using Apache Fuseki as the backend. It supports named graphs for ontologies
and facts, with proper authentication and dataset management.
"""

import logging
import re
from collections import defaultdict
from urllib.parse import quote

import requests
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
        clean=False,
        **kwargs,
    ):
        """Initialize the Fuseki triple store manager.

        This method sets up the connection to Fuseki, creates the dataset
        if it doesn't exist, and optionally cleans all data from the dataset.

        Args:
            uri: Fuseki server URI (e.g., "http://localhost:3030").
            auth: Authentication tuple (username, password) or string in "user/password" format.
            dataset: Dataset name to use for storage.
            clean: If True, delete all data from the dataset on initialization.
            **kwargs: Additional keyword arguments passed to the parent class.

        Raises:
            ValueError: If dataset is not specified in URI or as argument.

        Example:
            >>> manager = FusekiTripleStoreManager(
            ...     uri="http://localhost:3030",
            ...     dataset="test",
            ...     clean=True
            ... )
        """
        super().__init__(
            uri=uri, auth=auth, env_uri="FUSEKI_URI", env_auth="FUSEKI_AUTH", **kwargs
        )
        if dataset is None:
            self.dataset = DEFAULT_DATASET
        else:
            self.dataset = dataset
        self.ontologies_dataset = ontologies_dataset or DEFAULT_ONTOLOGIES_DATASET
        self.clean = clean
        self.init_dataset(self.dataset)
        if self.ontologies_dataset != self.dataset:
            self.init_dataset(self.ontologies_dataset)

        # Clean dataset if requested
        if self.clean:
            self._clean_dataset()

    def update_dataset(self, new_dataset: str) -> None:
        """Update the dataset name for this manager.

        This method allows changing the dataset without recreating the entire
        manager, which is useful for API requests that specify different datasets.

        Args:
            new_dataset: The new dataset name to use.
        """
        if not new_dataset:
            raise ValueError("Dataset name cannot be empty")

        self.dataset = new_dataset
        self.init_dataset(self.dataset)
        logger.info(f"Updated Fuseki dataset to: {self.dataset}")

    def _clean_dataset(self):
        """Delete all data from the dataset.

        This method removes all named graphs and clears the default graph
        from the Fuseki dataset. It uses Fuseki's REST API to perform
        the cleanup operations.

        The method handles errors gracefully and logs the results of
        each cleanup operation.
        """
        try:
            # Get the SPARQL update endpoint
            sparql_update_url = f"{self._get_dataset_url()}/update"

            # Delete all named graphs
            sparql_url = f"{self._get_dataset_url()}/sparql"
            query = """
            SELECT DISTINCT ?g WHERE {
              GRAPH ?g { ?s ?p ?o }
            }
            """
            response = requests.post(
                sparql_url,
                data={"query": query, "format": "application/sparql-results+json"},
                auth=self.auth,
            )

            if response.status_code == 200:
                results = response.json()
                for binding in results.get("results", {}).get("bindings", []):
                    graph_uri = binding["g"]["value"]
                    # Delete the named graph using SPARQL UPDATE
                    drop_query = f"DROP GRAPH <{graph_uri}>"
                    delete_response = requests.post(
                        sparql_update_url,
                        data={"update": drop_query},
                        auth=self.auth,
                    )
                    if delete_response.status_code in (200, 204):
                        logger.debug(f"Deleted named graph: {graph_uri}")
                    else:
                        logger.warning(
                            f"Failed to delete graph {graph_uri}: {delete_response.status_code}"
                        )

            # Clear the default graph using SPARQL UPDATE
            clear_query = "CLEAR DEFAULT"
            clear_response = requests.post(
                sparql_update_url,
                data={"update": clear_query},
                auth=self.auth,
            )
            if clear_response.status_code in (200, 204):
                logger.debug("Cleared default graph")
            else:
                logger.warning(
                    f"Failed to clear default graph: {clear_response.status_code}"
                )

            logger.info(f"Fuseki dataset '{self.dataset}' cleaned (all data deleted)")

        except Exception as e:
            logger.warning(f"Fuseki cleanup failed: {e}")

    def init_dataset(self, dataset_name):
        """Initialize a Fuseki dataset.

        This method creates a new dataset in Fuseki if it doesn't already exist.
        It uses Fuseki's admin API to create the dataset with TDB2 storage.

        Args:
            dataset_name: Name of the dataset to create.

        Note:
            This method will not fail if the dataset already exists.
        """
        fuseki_admin_url = f"{self.uri}/$/datasets"

        payload = {"dbName": dataset_name, "dbType": "tdb2"}

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        response = requests.post(
            fuseki_admin_url, data=payload, headers=headers, auth=self.auth
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
        """Fetch all ontologies from their corresponding named graphs.

        This method discovers all ontologies in the Fuseki ontologies dataset and
        fetches each one from its corresponding named graph. For versioned ontologies,
        it returns only the latest version for each unique ontology IRI.

        1. Discovery: List all named graphs (which may be versioned URIs)
        2. Fetching: Retrieve each ontology from its named graph
        3. Deduplication: For versioned ontologies, keep only the latest version

        Returns:
            list[Ontology]: List of the latest version of each ontology found.

        Example:
            >>> ontologies = manager.fetch_ontologies()
            >>> for onto in ontologies:
            ...     print(f"Found ontology: {onto.iri} v{onto.version}")
        """
        sparql_url = f"{self._get_ontologies_dataset_url()}/sparql"

        # Step 1: List all named graphs
        list_query = """
        SELECT DISTINCT ?g WHERE {
          GRAPH ?g { ?s ?p ?o }
        }
        """
        response = requests.post(
            sparql_url,
            data={"query": list_query, "format": "application/sparql-results+json"},
            auth=self.auth,
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

        # Step 2: Fetch each ontology from its corresponding named graph
        all_ontologies = []
        for graph_uri in graph_uris:
            graph = RDFGraph()
            # URL encode the graph URI to handle special characters like #
            encoded_graph_uri = quote(str(graph_uri), safe="/:")
            export_url = (
                f"{self._get_ontologies_dataset_url()}/get?graph={encoded_graph_uri}"
            )
            export_resp = requests.get(
                export_url, auth=self.auth, headers={"Accept": "text/turtle"}
            )

            if export_resp.status_code == 200:
                graph.parse(data=export_resp.text, format="turtle")

                # Re-serialize deterministically to ensure consistent cache keys
                # This sorts both namespaces and triples alphabetically
                deterministic_turtle = deterministic_turtle_serialization(graph)

                # Re-parse from deterministic serialization to ensure we have RDFGraph
                deterministic_graph = RDFGraph()
                deterministic_graph.parse(data=deterministic_turtle, format="turtle")

                # Copy namespace bindings from original graph
                for prefix, namespace in graph.namespaces():
                    if prefix:
                        deterministic_graph.bind(prefix, namespace)

                graph = deterministic_graph

                # Find the ontology IRI in the graph
                for onto_subj, _, obj in graph.triples((None, RDF.type, OWL.Ontology)):
                    onto_iri = str(onto_subj)
                    # Extract base IRI if it's versioned
                    if "#v" in graph_uri:
                        # Versioned graph - extract base IRI
                        onto_iri = graph_uri.split("#v")[0]

                    ontology = Ontology(
                        graph=graph,
                        iri=onto_iri,
                    )
                    # Load properties from graph
                    ontology.sync_properties_from_graph()
                    all_ontologies.append(ontology)
                    logger.debug(
                        f"Successfully loaded ontology: {onto_iri} version: {ontology.version}"
                    )
                    break  # Only one ontology per graph
            else:
                logger.warning(
                    f"Failed to fetch graph {graph_uri}: {export_resp.status_code}"
                )

        # Step 3: Deduplicate and keep latest versions
        ontology_dict = defaultdict(list)

        for onto in all_ontologies:
            ontology_dict[onto.iri].append(onto)

        # For each unique IRI, select the latest version
        ontologies = []

        for iri, versions in ontology_dict.items():
            if len(versions) == 1:
                ontologies.append(versions[0])
            else:
                # Multiple versions - keep the latest
                try:
                    # Sort by version if available
                    versions_with_ver = [v for v in versions if v.version]
                    if versions_with_ver:
                        # Sort by version using custom comparison
                        versions_with_ver.sort(
                            key=lambda x: str(x.version), reverse=False
                        )
                        ontologies.append(versions_with_ver[-1])
                        logger.debug(
                            f"Selected latest version for {iri}: {versions_with_ver[-1].version}"
                        )
                    else:
                        # No version info, keep first one
                        ontologies.append(versions[0])
                except Exception as e:
                    logger.warning(f"Could not compare versions for {iri}: {e}")
                    ontologies.append(versions[0])

        logger.info(
            f"Successfully loaded {len(ontologies)} unique ontologies from Fuseki (latest versions)"
        )
        return ontologies

    def serialize_graph(self, graph: Graph, **kwargs) -> bool | None:
        """Store an RDF graph as a named graph in a specific Fuseki dataset.

        This is a private helper method that handles the common logic for storing
        graphs in Fuseki datasets.

        Args:
            graph: The RDF graph to store.
            **kwargs: Additional parameters including graph_uri, dataset_url, default_graph_uri, log_prefix.

        Returns:
            bool: True if the graph was successfully stored, False otherwise.
        """
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
        response = requests.put(url, headers=headers, data=turtle_data, auth=self.auth)
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
            >>> success = manager.serialize(graph)

            >>> success = manager.serialize(graph, graph_uri="http://example.org/chunk1")
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

        return self.serialize_graph(
            graph=graph,
            graph_uri=graph_uri,
            dataset_url=dataset_url,
            default_graph_uri=default_graph_uri,
            log_prefix=log_prefix,
        )
