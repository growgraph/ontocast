"""RDF 1.2 triple terms must not reach rdflib graphs or SPARQL text.

Oxigraph-backed graphs (``tool/agg/rewriter``) carry triple terms, which the
oxrdflib iterator yields as plain tuples. rdflib's ``Graph.add`` asserts on them
and SPARQL has no syntax for them, so both copying and serialisation filter.
"""

import copy
from typing import cast

import pytest
from rdflib import Literal, Node, URIRef

from ontocast.onto.rdfgraph import (
    RDFGraph,
    copy_triples,
    drop_reifiers_mentioning,
    is_rdflib_triple,
    retarget_reifiers,
)
from ontocast.onto.sparql_models import GraphUpdate, TripleOp

pytestmark = pytest.mark.unit

EX = "http://example.org/"
S = URIRef(f"{EX}subject")
P = URIRef(f"{EX}predicate")
O = URIRef(f"{EX}object")  # noqa: E741 -- RDF term naming
REIFIES = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#reifies")

#: What oxrdflib yields for an RDF 1.2 triple term: the object is a bare tuple.
TRIPLE_TERM_TRIPLE = (URIRef(f"{EX}reifier"), REIFIES, (S, P, O))

# ruff: noqa: SLF001 -- these tests pin the behaviour of the private serialisers


def _oxigraph_graph_with_triple_term(*, with_plain_triple: bool = True) -> RDFGraph:
    """An oxigraph-backed graph holding a triple term, as the aggregator builds.

    Args:
        with_plain_triple: Also add an ordinary triple, so tests can show that
            the serialisable remainder survives.
    """
    pytest.importorskip("oxrdflib")
    import pyoxigraph as ox

    from ontocast.onto.rdfgraph import _oxigraph_inner_store

    graph = RDFGraph(store="oxigraph")
    if with_plain_triple:
        graph.add((S, P, O))
    # The quad must land in the graph's own context, otherwise rdflib never
    # iterates it -- ``RDFGraph`` identifies itself with a blank node.
    store = cast(ox.Store, _oxigraph_inner_store(graph.store))
    store.add(
        ox.Quad(
            ox.BlankNode(),
            ox.NamedNode(str(REIFIES)),
            ox.Triple(ox.NamedNode(str(S)), ox.NamedNode(str(P)), ox.NamedNode(str(O))),
            ox.BlankNode(str(graph.identifier)),
        )
    )
    # Guard the premise: without a tuple in the iteration these tests prove nothing.
    assert any(not is_rdflib_triple(triple) for triple in graph)
    return graph


def test_is_rdflib_triple_accepts_plain_triples() -> None:
    assert is_rdflib_triple((S, P, O))
    assert is_rdflib_triple((S, P, Literal("text")))


def test_is_rdflib_triple_rejects_triple_terms_and_junk() -> None:
    assert not is_rdflib_triple(TRIPLE_TERM_TRIPLE)
    assert not is_rdflib_triple((S, P))
    assert not is_rdflib_triple("not a triple")


def test_copy_triples_skips_triple_terms() -> None:
    target = RDFGraph()

    dropped = copy_triples([(S, P, O), TRIPLE_TERM_TRIPLE], target, origin="test")

    assert dropped == 1
    assert len(target) == 1
    assert (S, P, O) in target


def test_reifier_sweeps_are_no_ops_on_a_plain_rdflib_graph() -> None:
    """A plain store cannot hold a triple term, so neither sweep has work.

    Both go through pyoxigraph directly, so they must recognise a store they
    cannot address rather than reaching into it.
    """
    graph = RDFGraph()
    graph.add((S, P, O))

    assert drop_reifiers_mentioning(graph, {O}) == 0
    assert retarget_reifiers(graph, {(S, P, O): (S, P, Literal("replacement"))}) == 0
    assert len(graph) == 1


def test_retarget_reifiers_is_a_no_op_without_replacements() -> None:
    graph = _oxigraph_graph_with_triple_term()

    assert retarget_reifiers(graph, {}) == 0


def test_deepcopy_of_oxigraph_graph_with_triple_term_degrades() -> None:
    """Issue #48: used to raise TypeError (pickle), then AssertionError (add)."""
    graph = _oxigraph_graph_with_triple_term()

    duplicate = copy.deepcopy(graph)

    assert isinstance(duplicate, RDFGraph)
    assert (S, P, O) in duplicate


def test_serialize_rdf_term_raises_on_a_triple_term() -> None:
    """Issue #49: this used to return a Python repr and produce invalid SPARQL."""
    update = GraphUpdate()

    with pytest.raises(TypeError, match="Cannot serialize"):
        update._serialize_rdf_term(cast(Node, (S, P, O)))


def test_insert_query_skips_triple_terms_and_keeps_the_rest() -> None:
    update = GraphUpdate()

    query = update._generate_insert_query(
        _oxigraph_graph_with_triple_term(), f"PREFIX ex: <{EX}>"
    )

    assert "INSERT DATA {" in query
    assert f"<{S}> <{P}> <{O}> ." in query
    assert "rdflib.term.URIRef" not in query
    assert "reifies" not in query


def test_insert_query_is_empty_when_every_triple_is_a_triple_term() -> None:
    update = GraphUpdate()
    graph = _oxigraph_graph_with_triple_term(with_plain_triple=False)

    assert update._generate_insert_query(graph, "") == ""


def test_generated_queries_apply_cleanly_to_a_graph() -> None:
    """End to end: the compiled update parses and executes."""
    source = RDFGraph()
    source.bind("ex", URIRef(EX))
    source.add((S, P, Literal("value")))

    update = GraphUpdate(triple_operations=[TripleOp(type="insert", graph=source)])

    target = RDFGraph()
    for query in update.generate_sparql_queries():
        target.update(query)

    assert (S, P, Literal("value")) in target


def test_diff_summary_skips_triple_terms() -> None:
    update = GraphUpdate(
        triple_operations=[
            TripleOp(type="insert", graph=_oxigraph_graph_with_triple_term())
        ]
    )

    summary = update.generate_diff_summary()

    assert f"<{S}>" in summary
    assert "rdflib.term.URIRef" not in summary


def test_generated_queries_skip_triple_terms_end_to_end() -> None:
    """Issue #49: the compiled update used to fail to parse at apply time."""
    update = GraphUpdate(
        triple_operations=[
            TripleOp(type="insert", graph=_oxigraph_graph_with_triple_term())
        ]
    )

    target = RDFGraph()
    for query in update.generate_sparql_queries():
        target.update(query)

    assert (S, P, O) in target
