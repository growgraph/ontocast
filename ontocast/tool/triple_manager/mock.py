"""Mock triple store implementations for testing.

This module provides mock implementations of triple store managers that simulate
the behavior of real triple stores (Fuseki, Neo4j) without requiring external
services. These mocks are useful for testing and development.

The mocks maintain in-memory storage and provide the same interface as the
real implementations, allowing tests to run without external dependencies.
"""

import logging
from typing import Any, Dict, List

from pydantic import Field
from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF

from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.util import derive_ontology_id
from ontocast.tool.triple_manager.core import (
    TripleStoreManager,
    TripleStoreManagerWithAuth,
)

logger = logging.getLogger(__name__)


class MockTripleStoreManager(TripleStoreManager):
    """Mock triple store manager for testing.

    This class provides an in-memory implementation of triple store operations
    that simulates the behavior of real triple stores without requiring external
    services. It stores ontologies and graphs in memory and provides the same
    interface as concrete implementations.

    Attributes:
        ontologies: In-memory storage for ontologies.
        graphs: In-memory storage for RDF graphs.
    """

    model_config = {"arbitrary_types_allowed": True}

    ontologies: List[Ontology] = Field(
        default_factory=list, description="In-memory storage for ontologies"
    )
    graphs: Dict[str, Graph] = Field(
        default_factory=dict, description="In-memory storage for RDF graphs"
    )

    def __init__(self, **kwargs):
        """Initialize the mock triple store manager.

        Args:
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)

    def fetch_ontologies(self) -> List[Ontology]:
        """Fetch all available ontologies from the mock store.

        Returns:
            List[Ontology]: List of available ontologies with their graphs.
        """
        return self.ontologies.copy()

    def serialize_graph(
        self, graph: Graph, graph_uri: str | None = None
    ) -> bool | None:
        """Store an RDF graph in the mock store.

        Args:
            graph: The RDF graph to store.
            graph_uri: Optional URI to use as the graph identifier.

        Returns:
            bool: True if the graph was stored successfully.
        """
        # Create a new Graph and copy all triples
        new_graph = Graph()
        for triple in graph:
            new_graph.add(triple)

        if graph_uri:
            self.graphs[graph_uri] = new_graph
        else:
            # Generate a default URI based on graph content
            graph_uri = f"mock://graph/{len(self.graphs)}"
            self.graphs[graph_uri] = new_graph

        # Try to extract ontology information from the graph
        ontology_id = self._extract_ontology_id(graph)
        if ontology_id:
            ontology = Ontology(
                ontology_id=ontology_id,
                title=f"Mock Ontology {ontology_id}",
                description="Mock ontology for testing",
                version="1.0.0",
                iri=graph_uri,
                graph=self._create_rdf_graph_from_graph(graph),
            )
            # Update existing ontology or add new one
            existing = next(
                (o for o in self.ontologies if o.ontology_id == ontology_id), None
            )
            if existing:
                existing.graph = self._create_rdf_graph_from_graph(graph)
                existing.iri = graph_uri
            else:
                self.ontologies.append(ontology)

        return True

    def serialize(self, o: Ontology | RDFGraph, **kwargs) -> bool | None:
        """Store an Ontology or RDFGraph in the mock store.

        Args:
            o: Ontology or RDFGraph object to store.
            **kwargs: Additional keyword arguments.

        Returns:
            bool: True if the object was stored successfully.
        """
        if isinstance(o, Ontology):
            graph = o.graph
            graph_uri = o.iri
        elif isinstance(o, RDFGraph):
            graph = o
            graph_uri = kwargs.get("graph_uri")
        else:
            raise TypeError(f"unsupported obj of type {type(o)} received")

        return self.serialize_graph(graph, graph_uri)

    def _extract_ontology_id(self, graph: Graph) -> str | None:
        """Extract ontology ID from graph content.

        Args:
            graph: The RDF graph to analyze.

        Returns:
            str | None: The extracted ontology ID, or None if not found.
        """
        # Look for owl:Ontology declarations
        for s, p, o in graph.triples((None, RDF.type, OWL.Ontology)):
            if isinstance(s, URIRef):
                return derive_ontology_id(str(s))
        return None

    def clear(self):
        """Clear all stored data."""
        self.ontologies.clear()
        self.graphs.clear()

    async def clean(self, dataset: str | None = None) -> None:
        """Clean/flush data from the mock triple store.

        Args:
            dataset: Optional dataset name (ignored for mock, kept for interface compatibility).
        """
        self.clear()

    def _create_rdf_graph_from_graph(self, graph: Graph) -> RDFGraph:
        """Create an RDFGraph from a regular Graph by copying all triples.

        Args:
            graph: The source graph to copy from.

        Returns:
            RDFGraph: A new RDFGraph with all triples copied.
        """
        rdf_graph = RDFGraph()
        for triple in graph:
            rdf_graph.add(triple)
        return rdf_graph


class MockFusekiTripleStoreManager(TripleStoreManagerWithAuth):
    """Mock Fuseki triple store manager for testing.

    This class simulates the behavior of FusekiTripleStoreManager without
    requiring an actual Fuseki server. It maintains in-memory storage and
    provides the same interface as the real implementation.

    Attributes:
        dataset: The mock dataset name.
        ontologies_dataset: The mock ontologies dataset name.
        ontologies: In-memory storage for ontologies.
        graphs: In-memory storage for RDF graphs.
    """

    model_config = {"arbitrary_types_allowed": True}

    dataset: str | None = None
    ontologies_dataset: str = "ontologies"
    ontologies: List[Ontology] = Field(
        default_factory=list, description="In-memory storage for ontologies"
    )
    graphs: Dict[str, Graph] = Field(
        default_factory=dict, description="In-memory storage for RDF graphs"
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
        """Initialize the mock Fuseki triple store manager.

        Args:
            uri: Mock URI (ignored but kept for interface compatibility).
            auth: Mock authentication (ignored but kept for interface compatibility).
            dataset: Mock dataset name.
            ontologies_dataset: Mock ontologies dataset name.
            clean: Whether to clean the store on initialization.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(uri=uri, auth=auth, **kwargs)
        self.dataset = dataset or "test"
        self.ontologies_dataset = ontologies_dataset or "ontologies"

        if clean:
            self.clear()

    def fetch_ontologies(self) -> List[Ontology]:
        """Fetch all available ontologies from the mock store.

        Returns:
            List[Ontology]: List of available ontologies with their graphs.
        """
        return self.ontologies.copy()

    def serialize_graph(
        self, graph: Graph, graph_uri: str | None = None
    ) -> bool | None:
        """Store an RDF graph in the mock store.

        Args:
            graph: The RDF graph to store.
            graph_uri: Optional URI to use as the graph identifier.

        Returns:
            bool: True if the graph was stored successfully.
        """
        # Create a new Graph and copy all triples
        new_graph = Graph()
        for triple in graph:
            new_graph.add(triple)

        if graph_uri:
            self.graphs[graph_uri] = new_graph
        else:
            # Generate a default URI based on graph content
            graph_uri = f"mock://{self.dataset}/graph/{len(self.graphs)}"
            self.graphs[graph_uri] = new_graph

        # Try to extract ontology information from the graph
        ontology_id = self._extract_ontology_id(graph)
        if ontology_id:
            ontology = Ontology(
                ontology_id=ontology_id,
                title=f"Mock Ontology {ontology_id}",
                description="Mock ontology for testing",
                version="1.0.0",
                iri=graph_uri,
                graph=self._create_rdf_graph_from_graph(graph),
            )
            # Update existing ontology or add new one
            existing = next(
                (o for o in self.ontologies if o.ontology_id == ontology_id), None
            )
            if existing:
                existing.graph = self._create_rdf_graph_from_graph(graph)
                existing.iri = graph_uri
            else:
                self.ontologies.append(ontology)

        return True

    def serialize(self, o: Ontology | RDFGraph, **kwargs) -> bool | None:
        """Store an Ontology or RDFGraph in the mock store.

        Args:
            o: Ontology or RDFGraph object to store.
            **kwargs: Additional keyword arguments.

        Returns:
            bool: True if the object was stored successfully.
        """
        if isinstance(o, Ontology):
            graph = o.graph
            graph_uri = o.iri
        elif isinstance(o, RDFGraph):
            graph = o
            graph_uri = kwargs.get("graph_uri")
        else:
            raise TypeError(f"unsupported obj of type {type(o)} received")

        return self.serialize_graph(graph, graph_uri)

    def _extract_ontology_id(self, graph: Graph) -> str | None:
        """Extract ontology ID from graph content.

        Args:
            graph: The RDF graph to analyze.

        Returns:
            str | None: The extracted ontology ID, or None if not found.
        """
        # Look for owl:Ontology declarations
        for s, p, o in graph.triples((None, RDF.type, OWL.Ontology)):
            if isinstance(s, URIRef):
                return derive_ontology_id(str(s))
        return None

    def clear(self):
        """Clear all stored data."""
        self.ontologies.clear()
        self.graphs.clear()

    async def clean(self, dataset: str | None = None) -> None:
        """Clean/flush data from the mock Fuseki triple store.

        Args:
            dataset: Optional dataset name (ignored for mock, kept for interface compatibility).
        """
        self.clear()

    def _create_rdf_graph_from_graph(self, graph: Graph) -> RDFGraph:
        """Create an RDFGraph from a regular Graph by copying all triples.

        Args:
            graph: The source graph to copy from.

        Returns:
            RDFGraph: A new RDFGraph with all triples copied.
        """
        rdf_graph = RDFGraph()
        for triple in graph:
            rdf_graph.add(triple)
        return rdf_graph


class MockNeo4jTripleStoreManager(TripleStoreManagerWithAuth):
    """Mock Neo4j triple store manager for testing.

    This class simulates the behavior of Neo4jTripleStoreManager without
    requiring an actual Neo4j server. It maintains in-memory storage and
    provides the same interface as the real implementation.

    Attributes:
        ontologies: In-memory storage for ontologies.
        graphs: In-memory storage for RDF graphs.
    """

    model_config = {"arbitrary_types_allowed": True}

    ontologies: List[Ontology] = Field(
        default_factory=list, description="In-memory storage for ontologies"
    )
    graphs: Dict[str, Graph] = Field(
        default_factory=dict, description="In-memory storage for RDF graphs"
    )

    def __init__(self, uri=None, auth=None, clean=False, **kwargs):
        """Initialize the mock Neo4j triple store manager.

        Args:
            uri: Mock URI (ignored but kept for interface compatibility).
            auth: Mock authentication (ignored but kept for interface compatibility).
            clean: Whether to clean the store on initialization.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(uri=uri, auth=auth, **kwargs)

        if clean:
            self.clear()

    def fetch_ontologies(self) -> List[Ontology]:
        """Fetch all available ontologies from the mock store.

        Returns:
            List[Ontology]: List of available ontologies with their graphs.
        """
        return self.ontologies.copy()

    def serialize_graph(
        self, graph: Graph, graph_uri: str | None = None
    ) -> Dict[str, Any] | None:
        """Store an RDF graph in the mock store.

        Args:
            graph: The RDF graph to store.
            graph_uri: Optional URI to use as the graph identifier.

        Returns:
            Dict[str, Any]: Mock summary of the operation.
        """
        # Create a new Graph and copy all triples
        new_graph = Graph()
        for triple in graph:
            new_graph.add(triple)

        if graph_uri:
            self.graphs[graph_uri] = new_graph
        else:
            # Generate a default URI based on graph content
            graph_uri = f"mock://neo4j/graph/{len(self.graphs)}"
            self.graphs[graph_uri] = new_graph

        # Try to extract ontology information from the graph
        ontology_id = self._extract_ontology_id(graph)
        if ontology_id:
            ontology = Ontology(
                ontology_id=ontology_id,
                title=f"Mock Ontology {ontology_id}",
                description="Mock ontology for testing",
                version="1.0.0",
                iri=graph_uri,
                graph=self._create_rdf_graph_from_graph(graph),
            )
            # Update existing ontology or add new one
            existing = next(
                (o for o in self.ontologies if o.ontology_id == ontology_id), None
            )
            if existing:
                existing.graph = self._create_rdf_graph_from_graph(graph)
                existing.iri = graph_uri
            else:
                self.ontologies.append(ontology)

        # Return mock summary similar to Neo4j
        return {
            "nodes_created": len(graph),
            "relationships_created": 0,
            "properties_set": len(graph),
            "labels_added": 1,
        }

    def serialize(self, o: Ontology | RDFGraph, **kwargs) -> Dict[str, Any] | None:
        """Store an Ontology or RDFGraph in the mock store.

        Args:
            o: Ontology or RDFGraph object to store.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: Mock summary of the operation.
        """
        if isinstance(o, Ontology):
            graph = o.graph
            graph_uri = o.iri
        elif isinstance(o, RDFGraph):
            graph = o
            graph_uri = kwargs.get("graph_uri")
        else:
            raise TypeError(f"unsupported obj of type {type(o)} received")

        return self.serialize_graph(graph, graph_uri)

    def _extract_ontology_id(self, graph: Graph) -> str | None:
        """Extract ontology ID from graph content.

        Args:
            graph: The RDF graph to analyze.

        Returns:
            str | None: The extracted ontology ID, or None if not found.
        """
        # Look for owl:Ontology declarations
        for s, p, o in graph.triples((None, RDF.type, OWL.Ontology)):
            if isinstance(s, URIRef):
                return derive_ontology_id(str(s))
        return None

    def clear(self):
        """Clear all stored data."""
        self.ontologies.clear()
        self.graphs.clear()

    async def clean(self, dataset: str | None = None) -> None:
        """Clean/flush data from the mock Neo4j triple store.

        Args:
            dataset: Optional dataset name (ignored for Neo4j mock, kept for interface compatibility).
        """
        self.clear()

    def _create_rdf_graph_from_graph(self, graph: Graph) -> RDFGraph:
        """Create an RDFGraph from a regular Graph by copying all triples.

        Args:
            graph: The source graph to copy from.

        Returns:
            RDFGraph: A new RDFGraph with all triples copied.
        """
        rdf_graph = RDFGraph()
        for triple in graph:
            rdf_graph.add(triple)
        return rdf_graph
