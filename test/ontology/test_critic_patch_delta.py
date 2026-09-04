"""An ontology unit's product is its delta, not its scratchpad graph.

``build_delta`` replays the unit's recorded GraphUpdates onto a fresh copy of
the snapshot and diffs. A critic patch written straight into ``working_graph``
would therefore be reported by no delta and dropped at reduce time -- applied,
visible for the rest of the loop, and absent from the output. These tests pin
the channel that avoids that, and the rollback that has to undo both halves.
"""

import pytest
from rdflib import RDF, RDFS, Literal, Namespace, URIRef

from ontocast.onto.ontology_snapshot import OntologySnapshot
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import GraphUpdate, TripleOp
from ontocast.onto.unit_states import UnitOntologyState

pytestmark = pytest.mark.unit

ONTO = Namespace("https://growgraph.dev/ontologies/demo#")


def _snapshot_graph() -> RDFGraph:
    graph = RDFGraph()
    graph.bind("demo", ONTO)
    graph.add((ONTO.Material, RDF.type, RDFS.Class))
    graph.add((ONTO.Material, RDFS.label, Literal("Material")))
    return graph


def _state() -> UnitOntologyState:
    from ontocast.onto.content_unit import ContentUnit

    snapshot = OntologySnapshot(graph=_snapshot_graph())
    state = UnitOntologyState(
        content_unit=ContentUnit(
            text="demo", index=0, doc_iri=URIRef("https://example.org/doc")
        ),
        ontology_snapshot=snapshot,
    )
    state.working_graph = snapshot.graph.copy()
    return state


def _insert(*triples) -> GraphUpdate:
    graph = RDFGraph()
    for triple in triples:
        graph.add(triple)
    return GraphUpdate(triple_operations=[TripleOp(type="insert", graph=graph)])


def test_a_patch_reaches_the_delta_not_just_the_scratchpad() -> None:
    """The whole hazard: applied in the loop, present in the output."""
    state = _state()
    assert state.apply_patch(_insert((ONTO.Alloy, RDF.type, RDFS.Class))) is True

    assert (ONTO.Alloy, RDF.type, RDFS.Class) in state.working_graph
    assert (ONTO.Alloy, RDF.type, RDFS.Class) in state.build_delta().inserts
    assert state.all_updates, "the patch must be recorded for the reduce step"


def test_a_rolled_back_patch_leaves_neither_the_graph_nor_the_delta() -> None:
    """Restoring the graph alone would let the delta replay the patch back in."""
    state = _state()
    token = state.snapshot_for_rollback()
    state.apply_patch(_insert((ONTO.Alloy, RDF.type, RDFS.Class)))

    state.restore(token)

    assert (ONTO.Alloy, RDF.type, RDFS.Class) not in state.working_graph
    assert (ONTO.Alloy, RDF.type, RDFS.Class) not in state.build_delta().inserts
    assert state.all_updates == []


def test_a_patch_over_the_size_backstop_is_refused_with_the_graph_intact() -> None:
    """A refusal must not half-apply, and must not leave a pending update.

    Otherwise ``all_updates`` would replay into a delta the working graph never
    held -- the same silent divergence, arriving by the other direction.
    """
    state = _state()
    state.ontology_max_triples = 2
    before = set(state.working_graph)

    applied = state.apply_patch(
        _insert(
            (ONTO.Alloy, RDF.type, RDFS.Class),
            (ONTO.Ceramic, RDF.type, RDFS.Class),
            (ONTO.Polymer, RDF.type, RDFS.Class),
        )
    )

    assert applied is False
    assert set(state.working_graph) == before
    assert state.all_updates == []
    assert (
        state.build_delta().inserts == RDFGraph()
        or len(state.build_delta().inserts) == 0
    )


def test_the_product_count_measures_the_delta_not_the_working_graph() -> None:
    """The working graph is snapshot + delta, so its size barely moves.

    A rollback rule keyed on it would never fire.
    """
    state = _state()
    assert state.product_triple_count() == 0

    state.apply_patch(_insert((ONTO.Alloy, RDF.type, RDFS.Class)))

    assert state.product_triple_count() == 1
    assert len(state.working_graph) == len(_snapshot_graph()) + 1


def test_the_patch_target_is_the_scratchpad_graph() -> None:
    """Ids are resolved against what the critic was shown."""
    state = _state()
    assert state.patch_target_graph() is state.working_graph
