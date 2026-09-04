"""The triple index is a reference mechanism, so its ids must be reproducible.

The critic cites ids instead of requoting graph text. That only works if the id
a fix names is the id the resolver looks up -- which means the numbering must
not depend on store iteration order, and must be detectably stale once the loop
has patched the graph underneath it.
"""

import pytest
from rdflib import RDF, RDFS, XSD, BNode, Literal, Namespace

from ontocast.onto.enum import LLMGraphFormat
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.triple_index import build_triple_index, fingerprint_graph
from ontocast.prompt.graph_format import GraphFormatProfile

pytestmark = pytest.mark.unit

CD = Namespace("https://growgraph.dev/facts/")
MS = Namespace("https://growgraph.dev/ontologies/matsci#")


def _triples() -> list[tuple]:
    return [
        (CD.sample_1, RDF.type, MS.PerovskiteSample),
        (CD.sample_1, MS.hasEdgeLength, CD.len_1),
        (
            CD.len_1,
            MS.numericValue,
            Literal("0.30000000000000004", datatype=XSD.double),
        ),
        (CD.len_1, RDFS.label, Literal("edge length", lang="en")),
    ]


def _graph(triples=None) -> RDFGraph:
    graph = RDFGraph()
    graph.bind("cd", CD)
    graph.bind("ms", MS)
    for triple in triples if triples is not None else _triples():
        graph.add(triple)
    return graph


def test_ids_cover_every_triple_exactly_once() -> None:
    graph = _graph()
    index = build_triple_index(graph)
    assert sorted(index.by_id) == list(range(1, len(graph) + 1))
    assert set(index.by_id.values()) == set(graph)
    assert index.scope_size == len(graph)


def test_numbering_does_not_depend_on_insertion_order() -> None:
    """Iteration order is store-dependent, so the sort has to impose it.

    If this fails, two prompts built from equal graphs hand out different ids and
    a fix carried between them deletes the wrong statement.
    """
    forward = build_triple_index(_graph(_triples()))
    backward = build_triple_index(_graph(list(reversed(_triples()))))
    assert forward.by_id == backward.by_id
    assert forward.fingerprint == backward.fingerprint


def test_a_literal_and_an_iri_with_the_same_lexical_form_do_not_tie() -> None:
    same = "https://growgraph.dev/facts/x"
    graph = _graph([(CD.s, MS.p, CD.x), (CD.s, MS.p, Literal(same))])
    first = build_triple_index(graph)
    second = build_triple_index(
        _graph([(CD.s, MS.p, Literal(same)), (CD.s, MS.p, CD.x)])
    )
    assert first.by_id == second.by_id


def test_blank_node_statements_are_citable() -> None:
    """A bnode restriction cannot be requoted as text, but it does have an id.

    This is a large part of why ids exist: the old fix format could not address
    these statements at all.
    """
    bnode = BNode("b7")
    graph = _graph(
        [(CD.sample_1, RDFS.subClassOf, bnode), (bnode, RDF.type, MS.Restriction)]
    )
    index = build_triple_index(graph)
    subjects = [subject for subject, _ in index.order]
    assert bnode in subjects
    assert any(triple[0] == bnode for triple in index.by_id.values())


def test_the_listing_does_not_round_floating_point_literals() -> None:
    """rdflib's Turtle writer renders doubles through %e and loses digits.

    A critic shown the rounded value reports a rounding artifact as a defect, so
    the listing must use the lossless rendering.
    """
    chapter = GraphFormatProfile(
        format=LLMGraphFormat.TURTLE
    ).format_facts_chapter_indexed(_graph())
    assert "0.30000000000000004" in chapter.text


def test_turtle_carries_ids_inline_and_jsonld_carries_a_table() -> None:
    """JSON-LD's output contract demands strictly valid JSON, so no inline marker."""
    graph = _graph()
    turtle = GraphFormatProfile(
        format=LLMGraphFormat.TURTLE
    ).format_facts_chapter_indexed(graph)
    jsonld = GraphFormatProfile(
        format=LLMGraphFormat.JSONLD
    ).format_facts_chapter_indexed(graph)

    assert "[1]" in turtle.text
    assert "TRIPLE INDEX" not in turtle.text

    assert "TRIPLE INDEX (id | subject | predicate | object)" in jsonld.text
    body = jsonld.text.split("```json")[1].split("```")[0]
    assert "[1]" not in body
    assert turtle.index.by_id == jsonld.index.by_id


def test_rdf_type_is_written_as_the_turtle_shorthand() -> None:
    chapter = GraphFormatProfile(
        format=LLMGraphFormat.TURTLE
    ).format_facts_chapter_indexed(_graph())
    assert "] a ms:PerovskiteSample" in chapter.text


def test_a_patched_graph_no_longer_matches_its_index() -> None:
    """The loop mutates the graph between passes.

    A residual fix carried forward must not resolve against a later numbering,
    so staleness has to be detectable rather than assumed away.
    """
    graph = _graph()
    index = build_triple_index(graph)
    assert index.matches(graph)

    graph.add((CD.sample_1, MS.hasWidth, CD.width_1))
    assert not index.matches(graph)
    assert fingerprint_graph(graph) != index.fingerprint


def test_the_ontology_chapter_numbers_what_it_shows_not_what_it_was_given() -> None:
    """Condensing drops triples; an id on a dropped one is a blind delete."""
    graph = _graph()
    profile = GraphFormatProfile(format=LLMGraphFormat.TURTLE)
    chapter = profile.format_ontology_chapter_indexed(graph, max_triples=2)
    assert len(chapter.index) <= len(graph)
    for triple_id in chapter.index.by_id:
        assert f"[{triple_id}]" in chapter.text


def test_an_empty_graph_yields_an_empty_index_and_no_table() -> None:
    empty = RDFGraph()
    index = build_triple_index(empty)
    assert index.is_empty and len(index) == 0
    chapter = GraphFormatProfile(
        format=LLMGraphFormat.JSONLD
    ).format_facts_chapter_indexed(empty)
    assert "TRIPLE INDEX" not in chapter.text


def test_resolve_returns_none_for_an_id_that_was_never_issued() -> None:
    index = build_triple_index(_graph())
    assert index.resolve(len(index) + 1) is None
    assert index.resolve(0) is None
