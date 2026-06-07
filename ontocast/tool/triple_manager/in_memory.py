"""In-memory triple store management using pyoxigraph."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import pyoxigraph as ox
from oxrdflib._converter import to_ox
from rdflib import Graph

from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.tenancy import (
    DEFAULT_PROJECT,
    DEFAULT_TENANT,
    TENANCY_SEP,
)
from ontocast.tool.triple_manager.core import TripleStoreManager
from ontocast.tool.triple_manager.util import (
    dedupe_terminal_ontologies,
    ontology_from_named_graph,
)

logger = logging.getLogger(__name__)


@dataclass
class _TenantPartition:
    facts: ox.Store = field(default_factory=ox.Store)
    ontologies: ox.Store = field(default_factory=ox.Store)


def _to_ox_graph(graph_uri: str) -> ox.NamedNode:
    return ox.NamedNode(graph_uri)


def _clear_named_graph(store: ox.Store, graph_ctx: ox.NamedNode) -> None:
    for quad in list(store.quads_for_pattern(None, None, None, graph_ctx)):
        store.remove(quad)


def _rdflib_graph_to_quads(graph: Graph, graph_ctx: ox.NamedNode) -> list[ox.Quad]:
    quads: list[ox.Quad] = []
    for subject, predicate, object_ in graph:
        s_ox = to_ox(subject)
        p_ox = to_ox(predicate)
        o_ox = to_ox(object_)
        assert isinstance(s_ox, (ox.NamedNode, ox.BlankNode, ox.Triple))
        assert isinstance(p_ox, ox.NamedNode)
        assert isinstance(o_ox, (ox.NamedNode, ox.BlankNode, ox.Literal, ox.Triple))
        quads.append(ox.Quad(s_ox, p_ox, o_ox, graph_ctx))
    return quads


def _export_named_graph(store: ox.Store, graph_uri: str) -> RDFGraph:
    graph_ctx = _to_ox_graph(graph_uri)
    tmp = ox.Store()
    for quad in store.quads_for_pattern(None, None, None, graph_ctx):
        tmp.add(ox.Quad(quad.subject, quad.predicate, quad.object, ox.DefaultGraph()))
    if len(tmp) == 0:
        return RDFGraph()
    raw = tmp.dump(format=ox.RdfFormat.TURTLE, from_graph=ox.DefaultGraph())
    if raw is None:
        return RDFGraph()
    result = RDFGraph()
    result.parse(data=raw.decode("utf-8"), format="turtle")
    return result


def _list_named_graph_uris(store: ox.Store) -> list[str]:
    uris: set[str] = set()
    for quad in store:
        graph = quad.graph_name
        if isinstance(graph, ox.NamedNode):
            uris.add(graph.value)
    return sorted(uris)


class InMemoryTripleStoreManager(TripleStoreManager):
    """pyoxigraph-backed in-memory triple store with tenant/project partitions."""

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._partitions: dict[tuple[str, str], _TenantPartition] = {}
        self._active: tuple[str, str] = (DEFAULT_TENANT, DEFAULT_PROJECT)
        self._lock = asyncio.Lock()
        self._ensure_partition(self._active[0], self._active[1])

    def _ensure_partition(self, tenant: str, project: str) -> _TenantPartition:
        key = (tenant.strip(), project.strip())
        if key not in self._partitions:
            self._partitions[key] = _TenantPartition()
        return self._partitions[key]

    def _active_partition(self) -> _TenantPartition:
        return self._ensure_partition(self._active[0], self._active[1])

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
        t, p = tenant.strip(), project.strip()
        if not t or not p:
            raise ValueError("tenant and project must be non-empty")
        async with self._lock:
            self._active = (t, p)
            self._ensure_partition(t, p)
        logger.info("In-memory tenancy set to tenant=%r project=%r", tenant, project)

    async def clean(self) -> None:
        async with self._lock:
            partition = self._active_partition()
            partition.facts = ox.Store()
            partition.ontologies = ox.Store()

    async def clean_tenancy(
        self,
        tenant: str,
        project: str,
        *,
        sep: str = TENANCY_SEP,
    ) -> None:
        _ = sep
        key = (tenant.strip(), project.strip())
        async with self._lock:
            self._partitions.pop(key, None)
            if self._active == key:
                self._ensure_partition(key[0], key[1])
        logger.info("In-memory tenancy flush tenant=%r project=%r", tenant, project)

    async def drop_named_graph(
        self, graph_uri: str, *, use_ontologies_dataset: bool = True
    ) -> None:
        async with self._lock:
            partition = self._active_partition()
            store = partition.ontologies if use_ontologies_dataset else partition.facts
            _clear_named_graph(store, _to_ox_graph(graph_uri))

    async def drop_all_ontology_graphs_for_iri(self, ontology_iri: str) -> None:
        prefix = f"{ontology_iri}#"
        async with self._lock:
            partition = self._active_partition()
            for graph_uri in _list_named_graph_uris(partition.ontologies):
                if graph_uri == ontology_iri or graph_uri.startswith(prefix):
                    _clear_named_graph(partition.ontologies, _to_ox_graph(graph_uri))

    def fetch_ontologies(self) -> list[Ontology]:
        partition = self._active_partition()
        all_ontologies: list[Ontology] = []
        for graph_uri in _list_named_graph_uris(partition.ontologies):
            graph = _export_named_graph(partition.ontologies, graph_uri)
            onto = ontology_from_named_graph(graph_uri, graph)
            if onto is not None:
                all_ontologies.append(onto)
        result = dedupe_terminal_ontologies(all_ontologies)
        logger.info("Loaded %d unique ontologies from in-memory store", len(result))
        return result

    def serialize_graph(self, graph: Graph, **kwargs) -> bool:
        graph_uri = kwargs.get("graph_uri")
        use_ontologies = kwargs.pop("use_ontologies_dataset", False)
        if graph_uri is None:
            graph_uri = kwargs.get("default_graph_uri", "urn:data:default")

        partition = self._active_partition()
        store = partition.ontologies if use_ontologies else partition.facts
        graph_ctx = _to_ox_graph(str(graph_uri))
        _clear_named_graph(store, graph_ctx)
        quads = _rdflib_graph_to_quads(graph, graph_ctx)
        if quads:
            store.extend(quads)
        return True

    def serialize(self, o: Ontology | RDFGraph, **kwargs) -> bool:
        if isinstance(o, Ontology):
            return self.serialize_graph(
                o.graph,
                graph_uri=o.versioned_iri,
                use_ontologies_dataset=True,
            )
        if isinstance(o, RDFGraph):
            graph_uri = kwargs.get("graph_uri", "urn:data:default")
            return self.serialize_graph(
                o,
                graph_uri=graph_uri,
                use_ontologies_dataset=False,
            )
        raise TypeError(f"unsupported obj of type {type(o)} received")
