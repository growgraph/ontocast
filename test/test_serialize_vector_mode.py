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

from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.tool.ontology_manager import OntologyManager
from ontocast.tool.vector_store.patch_retriever import OntologyPatchRetriever
from ontocast.toolbox import ToolBox

pytestmark = pytest.mark.unit

_IRI = "https://example.com/onto"


def _ontology() -> Ontology:
    graph = RDFGraph()
    graph.add((URIRef(f"{_IRI}#Sample"), RDFS.label, Literal("Sample")))
    return Ontology(graph=graph, iri=_IRI, ontology_id="sample-onto")


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
    with pytest.raises(RuntimeError, match="aadd_ontology"):
        manager.add_ontology(_ontology())


@pytest.mark.anyio
async def test_aserialize_registers_the_ontology_with_vector_store_live() -> None:
    reindexed: list[Ontology] = []
    manager = _manager_with_retriever(reindexed)
    ontology = _ontology()
    assert ontology.hash, "artifact must arrive hashed for serialization"

    state = AgentState()
    state.reduced_ontology_artifacts = [ontology]

    stub = SimpleNamespace(ontology_manager=manager, triple_store_manager=None)
    await ToolBox.aserialize(cast(ToolBox, stub), state)

    assert manager.get_freshest_terminal_ontology_by_iri(_IRI) is not None
    assert [entry.iri for entry in reindexed] == [_IRI]
