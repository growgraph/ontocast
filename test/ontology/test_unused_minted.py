"""Classes a document minted that none of its own facts instantiates."""

import pytest
from rdflib import RDF, URIRef

from ontocast.onto.enum import RenderMode
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.stategraph.node_factories import record_unused_minted_classes

pytestmark = pytest.mark.unit

EX = "https://example.com/onto#"


def _state(render_mode: RenderMode) -> AgentState:
    state = AgentState(render_mode=render_mode)
    state.ontology_minted_classes = [EX + "Furnace", EX + "Sample"]
    facts = RDFGraph()
    facts.add((URIRef("https://example.com/cd#s1"), RDF.type, URIRef(EX + "Sample")))
    state.aggregated_facts = facts
    return state


def test_unused_minted_classes_are_counted_when_both_lanes_ran() -> None:
    state = _state(RenderMode.ONTOLOGY_AND_FACTS)
    record_unused_minted_classes(state)
    assert state.ontology_reduce_metrics["unused_minted_classes"] == 1
    assert state.ontology_reduce_metrics["unused_minted_class_iris"] == [EX + "Furnace"]


def test_nothing_is_recorded_without_a_facts_lane() -> None:
    state = _state(RenderMode.ONTOLOGY)
    record_unused_minted_classes(state)
    assert "unused_minted_classes" not in state.ontology_reduce_metrics
