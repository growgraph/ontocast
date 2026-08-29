"""SHACL shapes catalog: the shapes partition of the triple store.

Shapes are a *deployment artifact*, not a per-process file path. A catalog that
declares constraints ships them alongside its schema, and a tenant that owns a
catalog owns its shapes -- so they live in the triple store, in their own
partition, seeded once from disk and mutable over HTTP thereafter.

Why a partition of their own rather than the ontologies dataset: catalog
discovery claims every named graph carrying an ``owl:Ontology`` subject, and a
shapes document declares one. Stored beside the ontologies, a shapes file
registers as a catalog entry, gets indexed as ontology atoms, and is offered to
the renderer as first-class schema.

The facts validation gate is synchronous, so it cannot read the store itself.
This catalog resolves the merged shapes graph once, asynchronously, and hands
the gate a plain :class:`~ontocast.onto.rdfgraph.RDFGraph`.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import pathlib

from rdflib import RDF, URIRef
from rdflib.namespace import OWL

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.onto import Tool
from ontocast.tool.triple_manager.core import TripleStoreManager
from ontocast.tool.triple_manager.util import LIST_NAMED_GRAPHS_QUERY

logger = logging.getLogger(__name__)

#: Every triple in the shapes partition, merged. SHACL evaluates one shapes
#: graph, and shapes documents are independent, so the union is the whole answer.
_ALL_SHAPES_QUERY = """
CONSTRUCT { ?s ?p ?o }
WHERE { GRAPH ?g { ?s ?p ?o } }
"""


def shapes_graph_uri(graph: RDFGraph, *, fallback: str) -> str:
    """Name the named graph a shapes document is stored under.

    A shapes document that declares an ``owl:Ontology`` header is addressed by
    that IRI, which makes replace and delete by IRI work the way they do for
    ontologies. A bare SHACL file has no identity of its own, so the caller's
    ``fallback`` names it -- stable across edits, so re-seeding replaces the
    document rather than accumulating stale copies beside it.

    Args:
        graph: The parsed shapes document.
        fallback: Graph name to use when the document declares no ontology IRI.

    Returns:
        str: The named-graph URI.
    """
    for subject, _, _ in graph.triples((None, RDF.type, OWL.Ontology)):
        if isinstance(subject, URIRef):
            return str(subject)
    return fallback


def seed_graph_uri(path: pathlib.Path, root: pathlib.Path) -> str:
    """Fallback graph name for a headerless shapes file, derived from its path."""
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        relative = path.name
    return f"urn:shapes:{relative}"


def content_graph_uri(ttl: bytes) -> str:
    """Fallback graph name for a headerless uploaded shapes document."""
    return f"urn:shapes:{hashlib.sha256(ttl).hexdigest()}"


class ShapesCatalog(Tool):
    """The shapes partition, and the merged graph the validation gate reads.

    Partition-scoped, like
    :class:`~ontocast.tool.ontology_manager.OntologyManager`: everything held
    here belongs to one tenant/project, so a tenancy switch must
    :meth:`reset` it.
    """

    def __init__(self, **kwargs):
        """Initialize an empty catalog with no triple store registered."""
        super().__init__(**kwargs)
        self._triple_store_manager: TripleStoreManager | None = None
        self._graph: RDFGraph | None = None

    def register_triple_store(self, manager: TripleStoreManager | None) -> None:
        """Register the triple store holding the shapes partition."""
        self._triple_store_manager = manager

    def reset(self) -> None:
        """Drop the merged graph. Call on a tenancy switch."""
        self._graph = None

    def graph(self) -> RDFGraph | None:
        """Return the merged shapes graph, or ``None`` when nothing is stored.

        ``None`` is load-bearing downstream: it is what keeps
        ``facts_conformance.shacl_evaluated`` at ``None`` ("never checked")
        rather than reporting a clean run against no shapes.
        """
        return self._graph if self._graph is not None and len(self._graph) else None

    async def sync(self, shapes_dir: str | None = None) -> None:
        """Seed from ``shapes_dir`` when needed, then materialize the merged graph.

        Seeding mirrors the ontology bootstrap in
        :meth:`ontocast.toolbox.ToolBox._synchronize_ontologies`: the directory
        is a read-only fixture, the store is the persistence. Unlike that path
        the search is recursive, matching what the validation gate accepted from
        ``FACTS_SHAPES_DIR`` before shapes were stored.

        Args:
            shapes_dir: Seed directory of ``.ttl`` shape files, or ``None``.
        """
        store = self._triple_store_manager
        if store is None:
            self._graph = None
            return
        if shapes_dir:
            await self._seed_from_directory(shapes_dir, store)
        self._graph = await self._materialize(store)
        if self._graph is not None and len(self._graph):
            logger.info("Shapes partition holds %d triples", len(self._graph))

    async def ingest(self, graph: RDFGraph, *, graph_uri: str) -> str:
        """Store one shapes document and refresh the merged graph.

        Args:
            graph: The parsed shapes document.
            graph_uri: Named graph to store it under.

        Returns:
            str: The graph URI it was stored at.
        """
        store = self._require_triple_store()
        await store.aserialize_graph(graph, graph_uri=graph_uri, store="shapes")
        self._graph = await self._materialize(store)
        return graph_uri

    async def delete(self, graph_uri: str) -> None:
        """Remove one shapes document and refresh the merged graph."""
        store = self._require_triple_store()
        await store.drop_named_graph(graph_uri, store="shapes")
        self._graph = await self._materialize(store)

    async def list_graph_uris(self) -> list[str]:
        """List the named graphs in the shapes partition.

        Uses a named-graph listing rather than the ontology header query: a
        shapes document is not required to declare an ``owl:Ontology`` header,
        and one stored without a header must still be visible.
        """
        store = self._require_triple_store()
        if not store.supports_sparql_select():
            return []
        rows = await store.aselect(LIST_NAMED_GRAPHS_QUERY, store="shapes")
        return sorted({row["g"] for row in rows if "g" in row})

    def _require_triple_store(self) -> TripleStoreManager:
        if self._triple_store_manager is None:
            raise RuntimeError(
                "ShapesCatalog has no triple store registered; "
                "call register_triple_store() before reading the shapes partition"
            )
        return self._triple_store_manager

    async def _materialize(self, store: TripleStoreManager) -> RDFGraph | None:
        if not store.supports_sparql_construct():
            return None
        try:
            return await store.aconstruct(_ALL_SHAPES_QUERY, store="shapes")
        except Exception as error:
            # A shapes read that fails must never look like "no shapes
            # configured": that silently downgrades the gate to a clean run.
            logger.error("Failed to read the shapes partition: %s", error)
            raise

    async def _seed_from_directory(
        self, shapes_dir: str, store: TripleStoreManager
    ) -> None:
        directory = pathlib.Path(shapes_dir).expanduser()
        if not directory.is_dir():
            logger.warning(
                "FACTS_SHAPES_DIR points at %s, which is not a directory; "
                "no SHACL shapes seeded",
                shapes_dir,
            )
            return
        files = sorted(directory.glob("**/*.ttl"))
        if not files:
            logger.warning(
                "FACTS_SHAPES_DIR %s contains no .ttl shape files", shapes_dir
            )
            return
        documents = await asyncio.to_thread(self._parse_seed_files, files, directory)
        for graph_uri, graph in documents:
            await store.aserialize_graph(graph, graph_uri=graph_uri, store="shapes")
        logger.info(
            "Seeded %d shapes document(s) from %s into the shapes partition",
            len(documents),
            shapes_dir,
        )

    @staticmethod
    def _parse_seed_files(
        files: list[pathlib.Path], root: pathlib.Path
    ) -> list[tuple[str, RDFGraph]]:
        documents: list[tuple[str, RDFGraph]] = []
        for path in files:
            graph = RDFGraph()
            try:
                graph.parse(path.as_posix(), format="turtle")
            except Exception as error:
                logger.warning("Failed to parse shapes file %s: %s", path, error)
                continue
            documents.append(
                (
                    shapes_graph_uri(graph, fallback=seed_graph_uri(path, root)),
                    graph,
                )
            )
        return documents
