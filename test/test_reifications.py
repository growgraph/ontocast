"""RDF 1.2 provenance travels with its graph.

An :class:`RDFGraph` keeps its ``rdf:reifies`` statements beside the rdflib
triples (``graph.reifications``), so every path that copies, merges, dumps or
loads a graph carries both halves, and values keep their lexical form.
"""

import copy
from typing import cast

import pytest
from rdflib import BNode, Literal, Node, URIRef
from rdflib.namespace import XSD

from ontocast.onto.constants import PROV
from ontocast.onto.rdfgraph import (
    RDFGraph,
    drop_reifiers_mentioning,
    retarget_reifiers,
)
from ontocast.onto.reification import ProvenanceFormatError
from ontocast.onto.scaffolding import reifier_nodes
from ontocast.onto.sparql_models import GraphUpdate, TripleOp
from ontocast.onto.state import AgentState
from ontocast.tool.triple_manager.core import TripleStoreManager

pytestmark = pytest.mark.unit

EX = "http://example.org/"
S = URIRef(f"{EX}sample")
P = URIRef(f"{EX}thickness")
VALUE = Literal("2.0", datatype=XSD.decimal)
NAME = URIRef(f"{EX}name")
CHUNK = URIRef(f"{EX}chunk/1")

# ruff: noqa: SLF001 -- the pydantic hooks are what a model field calls


def _reified() -> tuple[RDFGraph, BNode]:
    """One fact with a verbatim decimal, one plain string, one reifier."""
    graph = RDFGraph()
    graph.bind("ex", EX)
    graph.add((S, P, VALUE))
    graph.add((S, NAME, Literal("sample 1")))
    reifier = BNode()
    graph.reifications.add(reifier, (S, P, VALUE))
    graph.add((reifier, PROV.wasDerivedFrom, CHUNK))
    return graph, reifier


def _assert_intact(graph: RDFGraph) -> None:
    assert isinstance(graph, RDFGraph)
    assert len(graph.reifications) == 1
    ((reifier, triple),) = list(graph.reifications)
    assert triple == (S, P, VALUE)
    assert str(triple[2]) == "2.0"
    assert (reifier, PROV.wasDerivedFrom, CHUNK) in graph
    assert (S, NAME, Literal("sample 1")) in graph


# ---------------------------------------------------------------------------
# Copies and merges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "duplicate",
    [
        RDFGraph.copy,
        copy.copy,
        copy.deepcopy,
        lambda graph: graph + RDFGraph(),
        lambda graph: RDFGraph() + graph,
    ],
    ids=["copy", "copy.copy", "deepcopy", "add-left", "add-right"],
)
def test_copies_carry_the_provenance(duplicate) -> None:
    graph, _ = _reified()

    duplicate_graph = duplicate(graph)

    _assert_intact(duplicate_graph)
    assert duplicate_graph.reifications is not graph.reifications


def test_a_copy_is_independent() -> None:
    graph, _ = _reified()
    duplicate = graph.copy()

    duplicate.reifications.clear()

    assert len(graph.reifications) == 1


def test_in_place_merge_takes_the_other_graphs_provenance() -> None:
    first, _ = _reified()
    second = RDFGraph()
    other = (S, P, Literal("3.5", datatype=XSD.decimal))
    second.add(other)
    second.reifications.add(BNode(), other)

    first += second

    assert {triple for _, triple in first.reifications} == {(S, P, VALUE), other}


def test_model_copy_carries_the_provenance() -> None:
    graph, _ = _reified()
    state = AgentState()
    state.aggregated_facts = graph

    _assert_intact(state.model_copy(deep=True).aggregated_facts)


# ---------------------------------------------------------------------------
# Dumps and loads
# ---------------------------------------------------------------------------


def test_pydantic_dump_and_load_round_trip() -> None:
    graph, _ = _reified()

    text = RDFGraph._to_turtle_str(graph)

    assert "<<(" in text
    _assert_intact(RDFGraph._from_str(text))


def test_agent_state_json_round_trip() -> None:
    graph, _ = _reified()
    state = AgentState()
    state.aggregated_facts = graph

    restored = AgentState.model_validate_json(state.model_dump_json())

    _assert_intact(restored.aggregated_facts)


@pytest.mark.parametrize("fmt", ["turtle", "nt", "nquads", "trig"])
def test_rdf12_formats_carry_the_provenance_verbatim(fmt: str) -> None:
    graph, _ = _reified()

    text = graph.serialize(format=fmt)

    assert "reifies" in text
    assert "2.0" in text
    assert '"sample 1"' in text


def test_canonical_turtle_carries_the_provenance() -> None:
    graph, _ = _reified()

    _assert_intact(RDFGraph._from_str(graph.serialize_canonical_turtle()))


@pytest.mark.parametrize("fmt", ["json-ld", "xml"])
def test_formats_without_triple_terms_refuse_provenance(fmt: str) -> None:
    graph, _ = _reified()

    with pytest.raises(ProvenanceFormatError, match="strip the provenance"):
        graph.serialize(format=fmt)


def test_stripped_graph_serializes_to_any_format() -> None:
    graph, _ = _reified()

    stripped = TripleStoreManager.strip_provenance(graph)

    assert not stripped.has_reifications
    assert (S, P, VALUE) in stripped
    assert not any(predicate == PROV.wasDerivedFrom for _, predicate, _ in stripped)
    assert stripped.serialize(format="json-ld")


def test_hash_covers_the_statements_not_the_provenance() -> None:
    graph, _ = _reified()
    plain = RDFGraph()
    for triple in graph:
        plain.add(triple)

    assert graph.hash() == plain.hash()


# ---------------------------------------------------------------------------
# Rewrites
# ---------------------------------------------------------------------------


def test_dropping_a_node_drops_its_reifiers_and_their_arcs() -> None:
    graph, reifier = _reified()

    removed = drop_reifiers_mentioning(graph, {S})

    assert removed == 2
    assert not graph.has_reifications
    assert (reifier, None, None) not in graph


def test_retargeting_moves_the_reifier_to_the_replacement() -> None:
    graph, reifier = _reified()
    replacement = (S, P, Literal("2.00", datatype=XSD.decimal))

    assert retarget_reifiers(graph, {(S, P, VALUE): replacement}) == 1

    assert list(graph.reifications) == [(reifier, replacement)]


def test_sweeps_leave_a_graph_without_provenance_alone() -> None:
    graph = RDFGraph()
    graph.add((S, P, VALUE))

    assert drop_reifiers_mentioning(graph, {S}) == 0
    assert retarget_reifiers(graph, {(S, P, VALUE): (S, P, Literal("x"))}) == 0
    assert len(graph) == 1


def test_namespace_remap_rewrites_the_reified_triples() -> None:
    graph, _ = _reified()

    graph.remap_namespaces(EX, "https://example.org/")

    moved = URIRef("https://example.org/sample")
    assert {triple[0] for _, triple in graph.reifications} == {moved}
    assert (moved, URIRef("https://example.org/thickness"), VALUE) in graph


def test_reifiers_are_scaffolding() -> None:
    graph, reifier = _reified()

    assert reifier in reifier_nodes(graph)


# ---------------------------------------------------------------------------
# SPARQL updates
# ---------------------------------------------------------------------------


def test_sparql_update_carries_the_statements_not_the_provenance() -> None:
    graph, _ = _reified()
    update = GraphUpdate(triple_operations=[TripleOp(type="insert", graph=graph)])

    target = RDFGraph()
    for query in update.generate_sparql_queries():
        assert "reifies" not in query
        target.update(query)

    assert (S, P, VALUE) in target


def test_serialize_rdf_term_raises_on_a_tuple() -> None:
    with pytest.raises(TypeError, match="Cannot serialize"):
        GraphUpdate()._serialize_rdf_term(cast(Node, (S, P, VALUE)))


def test_serialize_rdf_term_raises_on_an_iri_no_iriref_can_hold() -> None:
    with pytest.raises(ValueError, match="IRIREF"):
        GraphUpdate()._serialize_rdf_term(URIRef("https://ex.org/a b"))


def test_serialize_rdf_term_does_not_trim_an_iri_into_another() -> None:
    with pytest.raises(ValueError, match="IRIREF"):
        GraphUpdate()._serialize_rdf_term(URIRef("https://ex.org/a "))
