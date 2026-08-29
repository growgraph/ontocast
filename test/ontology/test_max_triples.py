"""``ONTOLOGY_MAX_TRIPLES`` had no test at all, and a lock-out bug behind it.

The guard skips a whole update batch when the result would exceed the cap. It
compared *absolute* post-apply size, so a working graph seeded above the cap
failed the check on every subsequent update -- discarding the LLM's work for the
rest of the run with only a WARNING, and with no way to shrink back under, since
deletions were rejected too. It now rejects only updates that grow the graph.
"""

from rdflib import Literal, URIRef
from rdflib.namespace import RDFS

from ontocast.onto.enum import LLMGraphFormat
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import GraphUpdate, TripleOp
from ontocast.onto.state import AgentState

EX = "https://example.org/onto#"
_TURTLE_CTX = {"llm_graph_format": LLMGraphFormat.TURTLE}
_PREFIX = f"@prefix ex: <{EX}> .\n"


def _seed(n: int) -> RDFGraph:
    graph = RDFGraph()
    for i in range(n):
        graph.add((URIRef(f"{EX}T{i}"), RDFS.label, Literal(f"term {i}")))
    return graph


def _op(op_type: str, turtle_body: str) -> GraphUpdate:
    op = TripleOp.model_validate(
        {"type": op_type, "graph": _PREFIX + turtle_body},
        context=_TURTLE_CTX,
    )
    return GraphUpdate(triple_operations=[op])


def _insert(i: int) -> GraphUpdate:
    return _op("insert", f'ex:New{i} rdfs:label "new {i}" .')


def _delete(i: int) -> GraphUpdate:
    return _op("delete", f'ex:T{i} rdfs:label "term {i}" .')


def test_no_limit_applies_any_update() -> None:
    graph = _seed(5)
    updated, applied = AgentState.render_updated_graph(
        graph, [_insert(0)], max_triples=None
    )

    assert applied is True
    assert len(updated) == 6


def test_update_under_the_limit_applies() -> None:
    graph = _seed(5)
    updated, applied = AgentState.render_updated_graph(
        graph, [_insert(0)], max_triples=10
    )

    assert applied is True
    assert len(updated) == 6


def test_growing_past_the_limit_is_skipped_and_leaves_the_graph_alone() -> None:
    graph = _seed(5)
    updated, applied = AgentState.render_updated_graph(
        graph, [_insert(0)], max_triples=5
    )

    assert applied is False
    assert updated is graph
    assert len(graph) == 5, (
        "the original must not be mutated when the update is skipped"
    )


def test_seed_already_over_the_limit_still_accepts_a_shrinking_update() -> None:
    """The lock-out: without this, an oversized seed rejects everything forever."""
    graph = _seed(10)
    updated, applied = AgentState.render_updated_graph(
        graph, [_delete(0)], max_triples=5
    )

    assert applied is True, "a delete cannot be refused for making the graph too big"
    assert len(updated) == 9


def test_seed_already_over_the_limit_still_refuses_to_grow() -> None:
    graph = _seed(10)
    updated, applied = AgentState.render_updated_graph(
        graph, [_insert(0)], max_triples=5
    )

    assert applied is False
    assert len(updated) == 10


def test_empty_update_list_is_a_no_op() -> None:
    graph = _seed(3)
    updated, applied = AgentState.render_updated_graph(graph, [], max_triples=1)

    assert applied is True
    assert updated is graph
