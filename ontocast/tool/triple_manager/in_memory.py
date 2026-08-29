"""In-memory triple store management using pyoxigraph."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import pyoxigraph as ox
from oxrdflib._converter import to_ox
from rdflib import Graph, URIRef

from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_header import OntologyHeader
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.tenancy import (
    DEFAULT_PROJECT,
    DEFAULT_TENANT,
    TENANCY_SEP,
    StoreKind,
)
from ontocast.tool.triple_manager.core import TripleStoreManager
from ontocast.tool.triple_manager.util import (
    ONTOLOGY_HEADER_QUERY,
    dedupe_terminal_ontologies,
    headers_from_select_rows,
    ontology_from_named_graph,
)

logger = logging.getLogger(__name__)


@dataclass
class _TenantPartition:
    facts: ox.Store = field(default_factory=ox.Store)
    ontologies: ox.Store = field(default_factory=ox.Store)
    shapes: ox.Store = field(default_factory=ox.Store)


def _partition_store(partition: _TenantPartition, store: StoreKind) -> ox.Store:
    """Resolve a :data:`StoreKind` to the partition's backing oxigraph store."""
    if store == "ontologies":
        return partition.ontologies
    if store == "shapes":
        return partition.shapes
    return partition.facts


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


def _run_select(store: ox.Store, query: str) -> list[dict[str, str]]:
    """Evaluate a SPARQL SELECT against ``store``, returning lexical values."""
    solutions = store.query(query)
    if not isinstance(solutions, ox.QuerySolutions):
        # Drop the result *here*: pyoxigraph result handles are unsendable, and
        # letting one escape on a traceback frame gets it freed on the calling
        # thread, which raises a second, far more confusing error.
        del solutions
        raise TypeError("aselect() requires a SPARQL SELECT query")
    variables = [str(variable)[1:] for variable in solutions.variables]
    rows: list[dict[str, str]] = []
    for solution in solutions:
        row: dict[str, str] = {}
        for name in variables:
            term = solution[name]
            if term is not None:
                row[name] = term.value
        rows.append(row)
    return rows


def _run_construct(store: ox.Store, query: str) -> RDFGraph:
    """Evaluate a SPARQL CONSTRUCT against ``store``, returning real RDF terms."""
    triples = store.query(query)
    if not isinstance(triples, ox.QueryTriples):
        del triples  # see the note in _run_select
        raise TypeError("aconstruct() requires a SPARQL CONSTRUCT or DESCRIBE query")
    raw = triples.serialize(format=ox.RdfFormat.N_TRIPLES)
    result = RDFGraph()
    if raw is None:
        return result
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    if text.strip():
        result.parse(data=text, format="nt")
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
        self._full_catalog_fetches = 0
        self._graph_fetches = 0
        self._select_queries = 0
        self._construct_queries = 0
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

    async def clean(self, *, include_shapes: bool = False) -> None:
        async with self._lock:
            partition = self._active_partition()
            partition.facts = ox.Store()
            partition.ontologies = ox.Store()
            if include_shapes:
                partition.shapes = ox.Store()

    async def clean_tenancy(
        self,
        tenant: str,
        project: str,
        *,
        sep: str = TENANCY_SEP,
        include_shapes: bool = False,
    ) -> None:
        _ = sep
        key = (tenant.strip(), project.strip())
        async with self._lock:
            existing = self._partitions.pop(key, None)
            preserved = (
                existing.shapes if existing is not None and not include_shapes else None
            )
            if preserved is not None or self._active == key:
                fresh = self._ensure_partition(key[0], key[1])
                if preserved is not None:
                    fresh.shapes = preserved
        logger.info("In-memory tenancy flush tenant=%r project=%r", tenant, project)

    async def drop_named_graph(
        self, graph_uri: str, *, store: StoreKind = "ontologies"
    ) -> None:
        async with self._lock:
            partition = self._active_partition()
            ox_store = _partition_store(partition, store)
            _clear_named_graph(ox_store, _to_ox_graph(graph_uri))

    async def drop_all_ontology_graphs_for_iri(
        self, ontology_iri: str, *, store: StoreKind = "ontologies"
    ) -> None:
        prefix = f"{ontology_iri}#"
        async with self._lock:
            ox_store = _partition_store(self._active_partition(), store)
            for graph_uri in _list_named_graph_uris(ox_store):
                if graph_uri == ontology_iri or graph_uri.startswith(prefix):
                    _clear_named_graph(ox_store, _to_ox_graph(graph_uri))

    def supports_sparql_select(self) -> bool:
        return True

    async def aselect(
        self, query: str, *, store: StoreKind = "ontologies"
    ) -> list[dict[str, str]]:
        """Evaluate a SPARQL SELECT against the active partition."""
        async with self._lock:
            partition = self._active_partition()
            ox_store = _partition_store(partition, store)
        self._select_queries += 1
        return await asyncio.to_thread(_run_select, ox_store, query)

    def supports_sparql_construct(self) -> bool:
        return True

    async def aconstruct(
        self, query: str, *, store: StoreKind = "ontologies"
    ) -> RDFGraph:
        """Evaluate a SPARQL CONSTRUCT against the active partition."""
        async with self._lock:
            partition = self._active_partition()
            ox_store = _partition_store(partition, store)
        self._construct_queries += 1
        return await asyncio.to_thread(_run_construct, ox_store, query)

    async def afetch_ontology_catalog(self) -> list[OntologyHeader]:
        """Read one header per stored ontology version via a single SELECT."""
        rows = await self.aselect(ONTOLOGY_HEADER_QUERY)
        return headers_from_select_rows(rows)

    async def afetch_ontologies_by_iri(self, iris: Sequence[str]) -> list[Ontology]:
        """Materialize only the named graphs backing ``iris``."""
        if not iris:
            return await self.afetch_ontologies()
        wanted = set(iris)
        headers = dedupe_terminal_ontologies(await self.afetch_ontology_catalog())
        graph_uris = [header.graph_uri for header in headers if header.iri in wanted]
        if not graph_uris:
            return []
        return await asyncio.to_thread(self._materialize_graphs, graph_uris)

    def _materialize_graphs(self, graph_uris: Sequence[str]) -> list[Ontology]:
        """Build ontologies from an explicit list of named graph URIs."""
        partition = self._active_partition()
        ontologies: list[Ontology] = []
        for graph_uri in graph_uris:
            self._graph_fetches += 1
            graph = _export_named_graph(partition.ontologies, graph_uri)
            onto = ontology_from_named_graph(graph_uri, graph)
            if onto is not None:
                ontologies.append(onto)
        return ontologies

    def catalog_io_stats(self) -> dict[str, int]:
        """Counters for catalog I/O, for tests and diagnostics."""
        return {
            "full_catalog_fetches": self._full_catalog_fetches,
            "graph_fetches": self._graph_fetches,
            "select_queries": self._select_queries,
            "construct_queries": self._construct_queries,
        }

    def fetch_ontologies(self) -> list[Ontology]:
        self._full_catalog_fetches += 1
        partition = self._active_partition()
        result = dedupe_terminal_ontologies(
            self._materialize_graphs(_list_named_graph_uris(partition.ontologies))
        )
        logger.info("Loaded %d unique ontologies from in-memory store", len(result))
        return result

    def serialize_graph(self, graph: Graph, **kwargs) -> bool:
        graph_uri = kwargs.get("graph_uri")
        store: StoreKind = kwargs.pop("store", "facts")
        if graph_uri is None:
            graph_uri = kwargs.get("default_graph_uri", "urn:data:default")

        partition = self._active_partition()
        ox_store = _partition_store(partition, store)
        graph_ctx = _to_ox_graph(str(graph_uri))
        _clear_named_graph(ox_store, graph_ctx)
        quads = _rdflib_graph_to_quads(graph, graph_ctx)
        if quads:
            ox_store.extend(quads)
        return True

    def serialize(self, o: Ontology | RDFGraph, **kwargs) -> bool:
        if isinstance(o, Ontology):
            if o.iri and not o.is_null():
                # Persist author @prefix names as triples before they die at
                # the store boundary (idempotent, excluded from content hash).
                o.graph.materialize_prefix_declarations(URIRef(o.iri))
            return self.serialize_graph(
                o.graph,
                graph_uri=o.versioned_iri,
                store=kwargs.get("store", "ontologies"),
            )
        if isinstance(o, RDFGraph):
            graph_uri = kwargs.get("graph_uri", "urn:data:default")
            return self.serialize_graph(
                o,
                graph_uri=graph_uri,
                store=kwargs.get("store", "facts"),
            )
        raise TypeError(f"unsupported obj of type {type(o)} received")
