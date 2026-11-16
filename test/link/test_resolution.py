"""The resolution graph's three modes, and update equals build.

Build ``[docs] -> R``, update ``[docs], R -> R`` and remove
``[doc ids], R -> R`` must agree: adding documents one at a time, in any order
or batching, gives the same resolution graph as building it at once.
"""

import asyncio
from itertools import permutations

import pytest
from rdflib import SKOS

from ontocast.link import (
    add_documents,
    build,
    document_graphs,
    identity_links_graph_uri,
    remove_documents,
)
from ontocast.onto.enum import RenderMode
from ontocast.onto.state import AgentState
from ontocast.toolbox import ToolBox
from test.link.helpers import (
    DOC1,
    DOC2,
    DOC3,
    DOMAIN,
    VALUE_CLASSES,
    canonical,
    catalog_graph,
    document,
    make_tools,
    store_documents,
)
from test.ontology_helpers import stamped_ontology

pytestmark = pytest.mark.unit

DOCS = [DOC1, DOC2, DOC3]


def _resolution(tools: ToolBox) -> dict[str, str]:
    """Every links graph in the store, by name, in canonical form."""

    async def read() -> dict[str, str]:
        store = tools.triple_store_manager
        assert store is not None
        rows = await store.aselect(
            "SELECT DISTINCT ?g WHERE { GRAPH ?g { ?s ?p ?o } }", store="facts"
        )
        out: dict[str, str] = {}
        for row in rows:
            name = row["g"]
            if name.endswith("/identity-links"):
                graph = await store.afetch_named_graph(name, store="facts")
                assert graph is not None
                out[name] = canonical(graph)
        return out

    return asyncio.run(read())


def _stored(store, graph_uri: str) -> str:
    graph = asyncio.run(store.afetch_named_graph(graph_uri, store="facts"))
    assert graph is not None
    return canonical(graph)


def _add(tools: ToolBox, docs: list[str]) -> None:
    store = tools.triple_store_manager
    assert store is not None
    asyncio.run(
        add_documents(
            store,
            docs,
            tools.ontology_manager.identity_declarations(),
            domain=DOMAIN,
            value_classes=VALUE_CLASSES,
        )
    )


def _build(tools: ToolBox) -> None:
    store = tools.triple_store_manager
    assert store is not None
    asyncio.run(
        build(
            store,
            tools.ontology_manager.identity_declarations(),
            domain=DOMAIN,
            value_classes=VALUE_CLASSES,
        )
    )


def _built_from(docs: list[str]) -> dict[str, str]:
    tools = make_tools()
    asyncio.run(store_documents(tools, docs))
    _build(tools)
    return _resolution(tools)


def test_build_writes_one_links_graph_per_linked_document() -> None:
    resolution = _built_from(DOCS)
    assert set(resolution) == {identity_links_graph_uri(d) for d in DOCS}


def test_document_graphs_exclude_auxiliary_graphs() -> None:
    tools = make_tools()
    asyncio.run(store_documents(tools, DOCS))
    _build(tools)
    store = tools.triple_store_manager
    assert store is not None
    assert asyncio.run(document_graphs(store)) == sorted(DOCS)


@pytest.mark.parametrize("order", list(permutations(DOCS)))
def test_adding_one_at_a_time_in_any_order_equals_building(
    order: tuple[str, ...],
) -> None:
    expected = _built_from(DOCS)
    tools = make_tools()
    for doc in order:
        asyncio.run(store_documents(tools, [doc]))
        _add(tools, [doc])
    assert _resolution(tools) == expected


def test_adding_in_batches_equals_building() -> None:
    expected = _built_from(DOCS)
    tools = make_tools()
    asyncio.run(store_documents(tools, DOCS))
    _add(tools, [DOC3])
    _add(tools, [DOC2, DOC1])
    assert _resolution(tools) == expected


def test_remove_equals_a_build_without_the_document() -> None:
    tools = make_tools()
    asyncio.run(store_documents(tools, DOCS))
    _build(tools)
    store = tools.triple_store_manager
    assert store is not None
    asyncio.run(remove_documents(store, [DOC1]))
    assert _resolution(tools) == _built_from([DOC2, DOC3])


def test_remove_then_add_restores_the_resolution_graph() -> None:
    tools = make_tools()
    asyncio.run(store_documents(tools, DOCS))
    _build(tools)
    before = _resolution(tools)
    store = tools.triple_store_manager
    assert store is not None
    asyncio.run(remove_documents(store, [DOC2]))
    _add(tools, [DOC2])
    assert _resolution(tools) == before


def test_build_drops_links_of_a_document_that_is_gone() -> None:
    tools = make_tools()
    asyncio.run(store_documents(tools, DOCS))
    _build(tools)
    store = tools.triple_store_manager
    assert store is not None
    asyncio.run(store.drop_named_graph(DOC1, store="facts"))
    _build(tools)
    assert identity_links_graph_uri(DOC1) not in _resolution(tools)


def test_build_is_idempotent() -> None:
    tools = make_tools()
    asyncio.run(store_documents(tools, DOCS))
    _build(tools)
    once = _resolution(tools)
    _build(tools)
    assert _resolution(tools) == once


def test_linking_never_changes_a_document_graph() -> None:
    tools = make_tools()
    asyncio.run(store_documents(tools, DOCS))
    store = tools.triple_store_manager
    assert store is not None
    before = _stored(store, DOC1)
    _build(tools)
    after = _stored(store, DOC1)
    assert after == before


def test_a_declaration_change_needs_only_a_build() -> None:
    tools = make_tools()
    asyncio.run(store_documents(tools, DOCS))
    _build(tools)
    changed = stamped_ontology(
        catalog_graph(court_globally_named=False), ontology_id="lk2"
    )
    tools.ontology_manager.add_ontology(changed, skip_vector_index=True)
    _build(tools)

    fresh = make_tools()
    fresh.ontology_manager.add_ontology(changed, skip_vector_index=True)
    asyncio.run(store_documents(fresh, DOCS))
    _build(fresh)
    assert _resolution(tools) == _resolution(fresh)


def _state(doc: str) -> AgentState:
    return AgentState(
        current_domain=DOMAIN,
        doc_hid=doc.rstrip("/").rsplit("/", 1)[-1],
        render_mode=RenderMode.FACTS,
        aggregated_facts=document(doc),
    )


def test_storing_a_document_with_linking_on_is_an_update(monkeypatch) -> None:
    tools = make_tools(link_enabled=True)
    monkeypatch.setattr(
        type(tools.get_atomic_tools()), "value_classes", lambda self: VALUE_CLASSES
    )
    for doc in DOCS:
        asyncio.run(tools.aserialize(_state(doc)))
    assert _resolution(tools) == _built_from(DOCS)


def test_storing_a_document_with_linking_off_writes_no_links() -> None:
    tools = make_tools(link_enabled=False)
    state = _state(DOC1)
    asyncio.run(tools.aserialize(state))
    assert _resolution(tools) == {}
    store = tools.triple_store_manager
    assert store is not None
    stored = asyncio.run(store.afetch_named_graph(DOC1, store="facts"))
    assert stored is not None
    assert not list(stored.triples((None, SKOS.closeMatch, None)))
