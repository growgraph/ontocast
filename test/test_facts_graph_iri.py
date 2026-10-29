"""Where a run's facts are written: the document namespace, or a named graph per run."""

import pytest

from ontocast.onto.state import AgentState

pytestmark = pytest.mark.unit


def _state(**kwargs) -> AgentState:
    return AgentState(current_domain="https://x.org", doc_hid="abc", **kwargs)


def test_facts_go_to_the_document_namespace_by_default() -> None:
    state = _state()
    assert state.graph_uri == state.doc_namespace


def test_a_named_graph_keeps_one_run_apart_and_leaves_entity_iris_alone() -> None:
    first = _state(facts_graph_iri="https://x.org/runs/1")
    second = _state(facts_graph_iri="urn:job:2")
    assert first.graph_uri == "https://x.org/runs/1"
    assert second.graph_uri == "urn:job:2"
    assert first.doc_namespace == second.doc_namespace == _state().doc_namespace


@pytest.mark.parametrize(
    "value", ["relative/graph", "https://", "https://x.org/a b", ""]
)
def test_a_graph_name_must_be_an_absolute_iri(value: str) -> None:
    with pytest.raises(ValueError):
        _state(facts_graph_iri=value)
