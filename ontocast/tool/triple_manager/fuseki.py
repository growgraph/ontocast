"""Fuseki triple store management for OntoCast.

This module provides a concrete implementation of triple store management
using Apache Fuseki as the backend. It supports named graphs for ontologies
and facts, with proper authentication and dataset management.
"""

import logging

import requests
from pydantic import Field
from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF

from ontocast.onto.constants import DEFAULT_DATASET, DEFAULT_ONTOLOGIES_DATASET
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.util import derive_ontology_id
from ontocast.tool.triple_manager.core import TripleStoreManagerWithAuth

logger = logging.getLogger(__name__)


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
            logger.info(f"Dataset '{dataset_name}' created successfully.")
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
        fetches each one from its corresponding named graph. It uses
        a two-step process:

        1. Discovery: Query for all ontology URIs using SPARQL
        2. Fetching: Retrieve each ontology from its named graph

        The method handles both named graphs and the default graph,
        and verifies that each ontology is properly typed as owl:Ontology.

        Returns:
            list[Ontology]: List of all ontologies found in the ontologies dataset.

        Example:
            >>> ontologies = manager.fetch_ontologies()
            >>> for onto in ontologies:
            ...     print(f"Found ontology: {onto.iri}")
        """
        sparql_url = f"{self._get_ontologies_dataset_url()}/sparql"

        # Step 1: List all ontology URIs from all graphs
        list_query = """
        SELECT DISTINCT ?s WHERE {
          { GRAPH ?g { ?s a <http://www.w3.org/2002/07/owl#Ontology> } }
          UNION
          { ?s a <http://www.w3.org/2002/07/owl#Ontology> }
        }
        """
        response = requests.post(
            sparql_url,
            data={"query": list_query, "format": "application/sparql-results+json"},
            auth=self.auth,
        )
        if response.status_code != 200:
            logger.error(f"Failed to list ontologies from Fuseki: {response.text}")
            return []

        results = response.json()
        ontology_iris = []
        for binding in results.get("results", {}).get("bindings", []):
            onto_iri = binding["s"]["value"]
            ontology_iris.append(onto_iri)

        logger.debug(f"Found {len(ontology_iris)} ontology URIs: {ontology_iris}")

        # Step 2: Fetch each ontology from its corresponding named graph
        ontologies = []
        for onto_iri in ontology_iris:
            # Fetch the ontology from its corresponding named graph
            graph = RDFGraph()
            export_url = f"{self._get_ontologies_dataset_url()}/get?graph={onto_iri}"
            export_resp = requests.get(
                export_url, auth=self.auth, headers={"Accept": "text/turtle"}
            )

            if export_resp.status_code == 200:
                graph.parse(data=export_resp.text, format="turtle")
                # Verify the ontology is actually in this graph
                onto_iri_ref = URIRef(onto_iri)
                if (onto_iri_ref, RDF.type, OWL.Ontology) in graph:
                    ontology_id = derive_ontology_id(onto_iri)
                    ontologies.append(
                        Ontology(
                            graph=graph,
                            iri=onto_iri,
                            ontology_id=ontology_id,
                        )
                    )
                    logger.debug(f"Successfully loaded ontology: {onto_iri}")
                else:
                    logger.warning(f"Ontology {onto_iri} not found in its named graph")
            else:
                logger.warning(
                    f"Failed to fetch ontology graph {onto_iri}: {export_resp.status_code}"
                )

        logger.info(f"Successfully loaded {len(ontologies)} ontologies from Fuseki")
        return ontologies

    def _serialize_graph_to_dataset(
        self,
        graph: Graph,
        graph_uri: str | None,
        dataset_url: str,
        default_graph_uri: str,
        log_prefix: str,
    ) -> bool | None:
        """Store an RDF graph as a named graph in a specific Fuseki dataset.

        This is a private helper method that handles the common logic for storing
        graphs in Fuseki datasets.

        Args:
            graph: The RDF graph to store.
            graph_uri: URI to use as the named graph name (optional).
            dataset_url: The URL of the dataset to store the graph in.
            default_graph_uri: Default graph URI to use if none provided.
            log_prefix: Prefix for log messages to distinguish between datasets.

        Returns:
            bool: True if the graph was successfully stored, False otherwise.
        """
        turtle_data = graph.serialize(format="turtle")
        if graph_uri is None:
            graph_uri = default_graph_uri

        url = f"{dataset_url}/data?graph={graph_uri}"
        headers = {"Content-Type": "text/turtle;charset=utf-8"}
        response = requests.put(url, headers=headers, data=turtle_data, auth=self.auth)
        if response.status_code in (200, 201, 204):
            logger.info(
                f"{log_prefix} graph {graph_uri} uploaded to Fuseki as named graph."
            )
            return True
        else:
            logger.error(
                f"Failed to upload {log_prefix.lower()} graph {graph_uri}. Status code: {response.status_code}"
            )
            logger.error(f"Response: {response.text}")
            return False

    def serialize_ontology_graph(
        self, graph: Graph, graph_uri: str | None = None
    ) -> bool | None:
        """Store an RDF graph as a named graph in the ontologies dataset.

        This method stores the given RDF graph as a named graph in the Fuseki
        ontologies dataset. The graph name is taken from the graph_uri parameter
        or defaults to "urn:ontology:default".

        Args:
            graph: The RDF graph to store.
            graph_uri: URI to use as the named graph name (optional).

        Returns:
            bool: True if the graph was successfully stored, False otherwise.

        Example:
            >>> graph = RDFGraph()
            >>> success = manager.serialize_ontology_graph(graph)

            >>> success = manager.serialize_ontology_graph(graph, graph_uri="http://example.org/ontology1")
        """
        return self._serialize_graph_to_dataset(
            graph=graph,
            graph_uri=graph_uri,
            dataset_url=self._get_ontologies_dataset_url(),
            default_graph_uri="urn:ontology:default",
            log_prefix="Ontology",
        )

    def serialize_graph(
        self, graph: Graph, graph_uri: str | None = None
    ) -> bool | None:
        """Store an RDF graph as a named graph in Fuseki.

        This method stores the given RDF graph as a named graph in Fuseki.
        The graph name is taken from the graph_uri parameter or defaults to
        "urn:data:default".

        Args:
            graph: The RDF graph to store.
            graph_uri: URI to use as the named graph name (optional).

        Returns:
            bool: True if the graph was successfully stored, False otherwise.

        Example:
            >>> graph = RDFGraph()
            >>> success = manager.serialize_graph(graph)

            >>> success = manager.serialize_graph(graph, graph_uri="http://example.org/chunk1")
        """
        return self._serialize_graph_to_dataset(
            graph=graph,
            graph_uri=graph_uri,
            dataset_url=self._get_dataset_url(),
            default_graph_uri="urn:data:default",
            log_prefix="Graph",
        )
