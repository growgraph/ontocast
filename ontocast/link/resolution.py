"""The resolution graph: every document's identity links, one named graph each.

Three operations keep it in step with the document graphs:

- :func:`add_documents` (update): write the links graph of each given document;
- :func:`remove_documents`: drop the links graphs of the given documents;
- :func:`build`: recompute every links graph from the stored documents, and
  drop links graphs whose document is gone.

Each links graph depends only on its document and the catalog's declarations,
so adding documents one at a time, in any order, gives the same resolution
graph as building it from all of them.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from pydantic import BaseModel, Field
from rdflib import SKOS

from ontocast.onto.constants import (
    IDENTITY_LINKS_SUFFIX,
    is_auxiliary_graph,
)
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.triple_manager.core import TripleStoreManager

from .declarations import IdentityDeclarations
from .links import identity_links_graph_uri, links_for

logger = logging.getLogger(__name__)

_NAMED_GRAPHS_QUERY = "SELECT DISTINCT ?g WHERE { GRAPH ?g { ?s ?p ?o } }"


class ResolutionReport(BaseModel):
    """What one operation on the resolution graph did."""

    documents: int = Field(default=0, description="Documents whose links were written.")
    links: int = Field(default=0, description="Links written.")
    removed: int = Field(default=0, description="Links graphs dropped.")


async def write_document_links(
    store: TripleStoreManager,
    graph_uri: str,
    links: RDFGraph,
) -> None:
    """Replace the links graph of the document ``graph_uri`` with ``links``."""
    target = identity_links_graph_uri(graph_uri)
    if len(links) == 0:
        await store.drop_named_graph(target, store="facts")
        return
    await store.aserialize(links, graph_uri=target)


async def document_graphs(store: TripleStoreManager) -> list[str]:
    """Every document graph in the facts partition, auxiliary graphs excluded."""
    rows = await store.aselect(_NAMED_GRAPHS_QUERY, store="facts")
    return sorted(
        {g for row in rows if (g := row.get("g")) and not is_auxiliary_graph(g)}
    )


async def links_graphs(store: TripleStoreManager) -> list[str]:
    """Every identity links graph in the facts partition."""
    rows = await store.aselect(_NAMED_GRAPHS_QUERY, store="facts")
    return sorted(
        {g for row in rows if (g := row.get("g")) and g.endswith(IDENTITY_LINKS_SUFFIX)}
    )


async def add_documents(
    store: TripleStoreManager,
    graph_uris: Iterable[str],
    declarations: IdentityDeclarations,
    *,
    domain: str,
    value_classes: frozenset[str] = frozenset(),
) -> ResolutionReport:
    """Write the links graph of each document graph in ``graph_uris``.

    A document graph that is not stored has its links graph dropped.
    """
    report = ResolutionReport()
    for graph_uri in sorted(set(graph_uris)):
        graph = await store.afetch_named_graph(graph_uri, store="facts")
        if graph is None:
            await store.drop_named_graph(
                identity_links_graph_uri(graph_uri), store="facts"
            )
            report.removed += 1
            continue
        links = links_for(
            graph, declarations, domain=domain, value_classes=value_classes
        )
        await write_document_links(store, graph_uri, links)
        report.documents += 1
        report.links += count_links(links)
    return report


async def remove_documents(
    store: TripleStoreManager, graph_uris: Iterable[str]
) -> ResolutionReport:
    """Drop the links graphs of the document graphs in ``graph_uris``."""
    report = ResolutionReport()
    for graph_uri in sorted(set(graph_uris)):
        await store.drop_named_graph(identity_links_graph_uri(graph_uri), store="facts")
        report.removed += 1
    return report


async def build(
    store: TripleStoreManager,
    declarations: IdentityDeclarations,
    *,
    domain: str,
    value_classes: frozenset[str] = frozenset(),
) -> ResolutionReport:
    """Recompute the whole resolution graph from the stored document graphs.

    Links graphs whose document graph is gone are dropped. Running it twice
    gives the same store.
    """
    documents = await document_graphs(store)
    expected = {identity_links_graph_uri(g) for g in documents}
    stale = [g for g in await links_graphs(store) if g not in expected]
    for graph_uri in stale:
        await store.drop_named_graph(graph_uri, store="facts")
    report = await add_documents(
        store, documents, declarations, domain=domain, value_classes=value_classes
    )
    report.removed += len(stale)
    logger.info(
        "Resolution graph built: %d document(s), %d link(s), %d stale graph(s) dropped",
        report.documents,
        report.links,
        len(stale),
    )
    return report


def count_links(links: RDFGraph) -> int:
    """The number of ``skos:closeMatch`` links in a links graph."""
    return sum(1 for _ in links.triples((None, SKOS.closeMatch, None)))
