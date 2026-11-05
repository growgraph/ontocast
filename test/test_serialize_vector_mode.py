"""Serialization must not call the sync catalog path from async code.

``ToolBox.aserialize`` used to call the sync ``add_ontology()``, whose guard
refuses to reindex the vector store while an event loop is running --
``aserialize`` is by definition inside one, so with vector retrieval
registered the first document that actually produced an ontology version to
serialize raised ``RuntimeError``. It never fired before because every
prior run used ``render_mode: facts``, which serializes no ontology artifacts.
"""

from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import RDFS, Literal, URIRef

from ontocast.onto.identity import OntologyIdentity
from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_header import mint_ontology, stamp_version
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.tool.ontology_manager import OntologyManager
from ontocast.tool.vector_store.patch_retriever import OntologyPatchRetriever
from ontocast.toolbox import ToolBox

pytestmark = pytest.mark.unit

_IRI = "https://example.com/onto"


def _artifact() -> Ontology:
    """An unstamped root artifact, as the reduce step hands it to serialization."""
    graph = RDFGraph()
    graph.add((URIRef(f"{_IRI}#Sample"), RDFS.label, Literal("Sample")))
    return mint_ontology(
        OntologyIdentity(ontology_id="sample-onto", iri=_IRI, title="Sample"), graph
    )


def _manager_with_retriever(reindexed: list[Ontology]) -> OntologyManager:
    manager = OntologyManager()
    retriever = cast(
        OntologyPatchRetriever,
        SimpleNamespace(
            vector_store=SimpleNamespace(
                reindex_ontology=lambda ontology: reindexed.append(ontology) or 1
            )
        ),
    )
    manager.register_vector_store(retriever)
    return manager


@pytest.mark.anyio
async def test_sync_add_ontology_refuses_to_reindex_inside_a_loop() -> None:
    """The guard this fix routes around: the failure shape, pinned."""
    manager = _manager_with_retriever([])
    version = stamp_version(_artifact())
    assert version is not None
    with pytest.raises(RuntimeError, match="aadd_ontology"):
        manager.add_ontology(version)


@pytest.mark.anyio
async def test_aserialize_registers_the_ontology_with_vector_store_live() -> None:
    reindexed: list[Ontology] = []
    manager = _manager_with_retriever(reindexed)
    artifact = _artifact()
    assert artifact.hash, "artifact must arrive hashed for serialization"

    state = AgentState()
    state.reduced_ontology_artifacts = [artifact]

    stub = SimpleNamespace(ontology_manager=manager, triple_store_manager=None)
    await ToolBox.aserialize(cast(ToolBox, stub), state)

    head = manager.get_head_by_iri(_IRI)
    assert head is not None
    assert head.version_iri == f"{_IRI}#{artifact.hash}"
    # What the index receives is the stamped version, not the bare artifact.
    assert [entry.version_iri for entry in reindexed] == [head.version_iri]
    assert [(r.iri, r.role) for r in state.ontology_versions_written] == [
        (_IRI, "created")
    ]


def _artifact_named(ontology_id: str, iri: str) -> Ontology:
    graph = RDFGraph()
    graph.add((URIRef(f"{iri}#Thing"), RDFS.label, Literal("Thing")))
    return mint_ontology(
        OntologyIdentity(ontology_id=ontology_id, iri=iri, title=ontology_id), graph
    )


@pytest.mark.anyio
async def test_an_identity_conflict_skips_the_version_and_still_writes_the_facts() -> (
    None
):
    """A colliding version is refused before any write; the job keeps its facts.

    The collision used to raise at registration, after the version was already
    in the store, and the facts write after it never ran.
    """
    from ontocast.onto.enum import RenderMode

    manager = OntologyManager()
    held = stamp_version(_artifact_named("fro", "https://example.com/fro"))
    assert held is not None
    manager.add_ontology(held)

    writes: list[object] = []

    async def record(graph: object, **_: object) -> None:
        writes.append(graph)

    state = AgentState(render_mode=RenderMode.ONTOLOGY_AND_FACTS)
    clash = _artifact_named("fro", "https://example.com/frobase")
    fine = _artifact_named("other", "https://example.com/other")
    twin = _artifact_named("other", "https://example.com/other-twin")
    state.reduced_ontology_artifacts = [clash, fine, twin]
    state.aggregated_facts.add(
        (URIRef("https://example.com/cd/x"), RDFS.label, Literal("x"))
    )

    stub = SimpleNamespace(
        ontology_manager=manager,
        triple_store_manager=SimpleNamespace(aserialize=record),
    )
    await ToolBox.aserialize(cast(ToolBox, stub), state)

    assert [r.iri for r in state.ontology_versions_written] == [
        "https://example.com/other"
    ]
    written_iris = [getattr(graph, "iri", None) for graph in writes]
    assert written_iris == ["https://example.com/other", None]
    assert writes[-1] is state.aggregated_facts
    assert state.budget_tracker.counters["serialize/ontology_identity_conflict"] == 2
