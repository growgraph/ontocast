"""Mock triple store implementations for testing."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import Field
from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF

from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.tenancy import TENANCY_SEP
from ontocast.onto.util import derive_ontology_id
from ontocast.tool.triple_manager.core import (
    TripleStoreManager,
    TripleStoreManagerWithAuth,
)

logger = logging.getLogger(__name__)


class _InMemoryGraphStoreMixin:
    """Shared in-memory graph storage for mock backends."""

    model_config = {"arbitrary_types_allowed": True}

    ontologies: list[Ontology] = Field(
        default_factory=list, description="In-memory storage for ontologies"
    )
    graphs: dict[str, Graph] = Field(
        default_factory=dict, description="In-memory storage for RDF graphs"
    )

    def _create_rdf_graph_from_graph(self, graph: Graph) -> RDFGraph:
        rdf_graph = RDFGraph()
        for triple in graph:
            rdf_graph.add(triple)
        return rdf_graph

    def _extract_ontology_id(self, graph: Graph) -> str | None:
        for s, _, _ in graph.triples((None, RDF.type, OWL.Ontology)):
            if isinstance(s, URIRef):
                return derive_ontology_id(str(s))
        return None

    def _store_graph(self, graph: Graph, graph_uri: str | None) -> str:
        new_graph = Graph()
        for triple in graph:
            new_graph.add(triple)
        if graph_uri is None:
            graph_uri = f"mock://graph/{len(self.graphs)}"
        self.graphs[graph_uri] = new_graph

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
            existing = next(
                (o for o in self.ontologies if o.ontology_id == ontology_id), None
            )
            if existing:
                existing.graph = self._create_rdf_graph_from_graph(graph)
                existing.iri = graph_uri
            else:
                self.ontologies.append(ontology)
        return graph_uri

    def clear(self) -> None:
        self.ontologies.clear()
        self.graphs.clear()

    def fetch_ontologies(self) -> list[Ontology]:
        return self.ontologies.copy()

    def serialize_graph(self, graph: Graph, **kwargs: Any) -> bool:
        self._store_graph(graph, kwargs.get("graph_uri"))
        return True

    def serialize(self, o: Ontology | RDFGraph, **kwargs: Any) -> bool:
        if isinstance(o, Ontology):
            graph = o.graph
            graph_uri = o.versioned_iri
        elif isinstance(o, RDFGraph):
            graph = o
            graph_uri = kwargs.get("graph_uri")
        else:
            raise TypeError(f"unsupported obj of type {type(o)} received")
        self.serialize_graph(graph, graph_uri=graph_uri)
        return True

    async def clean(self) -> None:
        self.clear()


class MockTripleStoreManager(_InMemoryGraphStoreMixin, TripleStoreManager):
    """Basic in-memory mock triple store manager."""


class MockFusekiTripleStoreManager(
    _InMemoryGraphStoreMixin, TripleStoreManagerWithAuth
):
    """Mock Fuseki triple store manager with tenancy support."""

    dataset: str | None = None
    ontologies_dataset: str = "ontologies"
    _tenant: str = "ontocast"
    _project: str = "test"

    def __init__(
        self,
        uri=None,
        auth=None,
        dataset=None,
        ontologies_dataset=None,
        clean=False,
        **kwargs,
    ):
        super().__init__(uri=uri, auth=auth, **kwargs)
        self.dataset = dataset or "test"
        self.ontologies_dataset = ontologies_dataset or "ontologies"
        if clean:
            self.clear()

    def supports_tenancy_partition(self) -> bool:
        return True

    async def update_tenancy(
        self,
        tenant: str,
        project: str,
        *,
        sep: str = TENANCY_SEP,
    ) -> None:
        _ = sep
        self._tenant = tenant.strip()
        self._project = project.strip()
        self.dataset = f"{self._tenant}--{self._project}--facts"
        self.ontologies_dataset = f"{self._tenant}--{self._project}--ontologies"

    async def clean_tenancy(self, tenant: str, project: str) -> None:
        _ = tenant, project
        self.clear()

    async def drop_named_graph(
        self, graph_uri: str, *, use_ontologies_dataset: bool = True
    ) -> None:
        _ = use_ontologies_dataset
        self.graphs.pop(graph_uri, None)

    async def drop_all_ontology_graphs_for_iri(self, ontology_iri: str) -> None:
        prefix = f"{ontology_iri}#"
        for graph_uri in list(self.graphs):
            if graph_uri == ontology_iri or graph_uri.startswith(prefix):
                self.graphs.pop(graph_uri, None)
        self.ontologies = [o for o in self.ontologies if o.iri != ontology_iri]


class MockInMemoryTripleStoreManager(MockFusekiTripleStoreManager):
    """Alias mock for the in-memory pyoxigraph backend."""

    def __init__(self, *, clean: bool = False, **kwargs):
        super().__init__(clean=clean, **kwargs)
