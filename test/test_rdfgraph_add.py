"""``RDFGraph +=`` adds in place, at the cost of the right-hand side."""

import pytest
from rdflib import Literal, URIRef

from ontocast.onto.rdfgraph import RDFGraph

pytestmark = pytest.mark.unit

EX = "https://ex.org/"


def _graph(n: int, *, prefix: str = "ex", namespace: str = EX) -> RDFGraph:
    graph = RDFGraph()
    graph.bind(prefix, namespace)
    for i in range(n):
        graph.add((URIRef(f"{namespace}s{i}"), URIRef(f"{namespace}p"), Literal(i)))
    return graph


def _counting_adds(graph: RDFGraph, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls = [0]
    add = graph.add

    def counted(triple):
        calls[0] += 1
        return add(triple)

    monkeypatch.setattr(graph, "add", counted)
    return calls


def test_iadd_keeps_identity_and_matches_add() -> None:
    left, right = _graph(3), _graph(2, namespace="https://other.org/")
    expected = left + right
    alias = left
    left += right
    assert left is alias
    assert set(left) == set(expected)


def test_iadd_costs_the_right_hand_side_only(monkeypatch: pytest.MonkeyPatch) -> None:
    big = _graph(1000)
    calls = _counting_adds(big, monkeypatch)
    big += _graph(1, namespace="https://other.org/")
    assert len(big) == 1001
    assert calls[0] == 1


def test_iadd_renames_a_clashing_prefix_like_add() -> None:
    left = _graph(1)
    right = _graph(1, namespace="https://clash.org/")  # same prefix "ex"
    expected = dict((p, str(u)) for p, u in (left + right).namespaces())
    left += right
    bound = dict((p, str(u)) for p, u in left.namespaces())
    assert bound["ex"] == EX
    assert "https://clash.org/" in bound.values()
    assert bound == expected


def test_iadd_with_itself_is_a_no_op() -> None:
    graph = _graph(3)
    graph += graph
    assert len(graph) == 3
