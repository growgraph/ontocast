"""Bounding the text literals that reach an ontology chapter.

The triple budget is a count and says nothing about how long one literal may
be, so a catalog well inside it can still ship an unbounded chapter -- and does,
on every call of every unit. These pin the bound: what it does, what it refuses
to do, and that it is inert when unset.
"""

from __future__ import annotations

import pytest
from rdflib import OWL, RDF, RDFS, SKOS, Literal, Namespace, URIRef

from ontocast.onto.enum import LLMGraphFormat, OntologyChapterFormat
from ontocast.onto.ontology_condense import (
    CLIP_MARKER,
    TextCaps,
    clip_text,
    condense_graph_for_prompt,
)
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.prompt.graph_format import get_graph_format_profile

EX = Namespace("https://example.org/onto#")


def _catalog(*, terms: int = 12, prose: int = 400, note: int = 900) -> RDFGraph:
    """A catalog whose prose volume is a parameter rather than a constant."""
    graph = RDFGraph()
    graph.bind("ex", EX)
    for index in range(terms):
        term = EX[f"Term{index}"]
        graph.add((term, RDF.type, OWL.Class))
        graph.add((term, RDFS.label, Literal(f"term {index}")))
        graph.add((term, RDFS.comment, Literal(" ".join(["prose"] * (prose // 6)))))
        graph.add((term, SKOS.scopeNote, Literal(" ".join(["applies"] * (note // 8)))))
    return graph


def test_clip_cuts_on_a_word_boundary_and_marks_the_cut() -> None:
    clipped = clip_text("the quick brown fox jumps over the lazy dog", 20)

    assert clipped.endswith(CLIP_MARKER)
    assert not clipped[: -len(CLIP_MARKER)].endswith(" ")
    assert len(clipped) - len(CLIP_MARKER) <= 20
    assert "the quick brown fox" in clipped


def test_clip_falls_back_when_there_is_no_word_boundary() -> None:
    """A single long token still has to be bounded, marker included."""
    assert clip_text("x" * 40, 10) == "x" * 10 + CLIP_MARKER


def test_clip_leaves_short_text_byte_identical() -> None:
    assert clip_text("short", 80) == "short"
    assert clip_text("short", None) == "short"


def test_unset_caps_are_a_true_no_op() -> None:
    """An untouched deployment must not see its prompts -- or cache keys -- move."""
    graph = _catalog()
    condensed, report = condense_graph_for_prompt(graph, 4000, TextCaps())

    assert condensed is graph, "inactive caps must not even copy the graph"
    assert not report.changed
    assert report.literals_clipped == 0


def test_caps_clip_the_roles_they_govern_and_nothing_else() -> None:
    graph = _catalog()
    condensed, report = condense_graph_for_prompt(
        graph, 4000, TextCaps(contract=120, prose=80)
    )

    labels = [str(o) for o in condensed.objects(None, RDFS.label)]
    comments = [str(o) for o in condensed.objects(None, RDFS.comment)]
    notes = [str(o) for o in condensed.objects(None, SKOS.scopeNote)]

    assert all(CLIP_MARKER not in label for label in labels), "naming was uncapped"
    assert all(len(c) <= 80 + len(CLIP_MARKER) for c in comments)
    assert all(len(n) <= 120 + len(CLIP_MARKER) for n in notes)
    assert report.literals_clipped == len(comments) + len(notes)
    assert report.literals_dropped == 0
    assert len(condensed) == len(graph), "clipping must not remove statements"


def test_language_tag_survives_clipping() -> None:
    graph = RDFGraph()
    graph.add((EX.T, RDFS.comment, Literal("a " * 200, lang="en")))

    condensed, _ = condense_graph_for_prompt(graph, 4000, TextCaps(prose=40))

    (obj,) = list(condensed.objects(EX.T, RDFS.comment))
    assert isinstance(obj, Literal)
    assert obj.language == "en"


@pytest.mark.parametrize("prose", [400, 4_000, 40_000])
def test_total_budget_holds_however_verbose_the_catalog(prose: int) -> None:
    """The point of the backstop: chapter size stops tracking prose volume.

    Without it the chapter costs whatever the catalog's authors chose to write.
    This is the guarantee for a catalog nobody here has seen.
    """
    budget = 3_000
    graph = _catalog(prose=prose, note=prose)

    condensed, report = condense_graph_for_prompt(
        graph, 4000, TextCaps(total_budget=budget)
    )

    assert report.text_chars_after <= budget
    assert not report.text_over_budget
    assert report.text_chars_before > budget


def test_total_budget_never_removes_the_names() -> None:
    """A term the model cannot name is an invitation to invent one.

    So an impossible budget is reported and passed through, the same way the
    triple budget refuses to cut into load-bearing structure.
    """
    graph = _catalog(terms=60, prose=400, note=400)

    condensed, report = condense_graph_for_prompt(graph, 4000, TextCaps(total_budget=1))

    assert report.text_over_budget
    labels = list(condensed.objects(None, RDFS.label))
    assert len(labels) == 60, "every term must still have a name"


def test_caps_apply_to_the_turtle_chapter_too() -> None:
    """Not a term-sheet feature: any chapter the pipeline builds is bounded."""
    graph = _catalog()
    profile = get_graph_format_profile(
        LLMGraphFormat.JSONLD, ontology_chapter_format=OntologyChapterFormat.TURTLE
    )

    uncapped = profile.format_ontology_chapter(graph, max_triples=4000)
    capped = profile.format_ontology_chapter(
        graph, max_triples=4000, text_caps=TextCaps(contract=100, prose=100)
    )

    assert len(capped) < len(uncapped)
    assert CLIP_MARKER in capped
    assert "ttl" in capped, "still a Turtle chapter"


def test_caps_join_the_memoised_chapter_key() -> None:
    """Two cap settings are two chapters; the memo must not serve one as the other."""
    from ontocast.onto.ontology_snapshot import OntologySnapshot

    graph = _catalog()
    snapshot = OntologySnapshot(
        graph=graph, source_iris=[URIRef("https://example.org")]
    )
    profile = get_graph_format_profile(LLMGraphFormat.TURTLE)

    loose = snapshot.prompt_chapter(profile, text_caps=TextCaps(prose=400))
    tight = snapshot.prompt_chapter(profile, text_caps=TextCaps(prose=40))

    assert loose != tight
    assert snapshot.prompt_chapter(profile, text_caps=TextCaps(prose=40)) == tight


def test_chapter_report_reaches_the_caller() -> None:
    """A cap whose effect cannot be read back is a cap nobody can size."""
    from ontocast.onto.ontology_condense import CondenseReport

    graph = _catalog()
    profile = get_graph_format_profile(LLMGraphFormat.TURTLE)
    seen: list[CondenseReport] = []

    profile.format_ontology_chapter(
        graph,
        max_triples=4000,
        text_caps=TextCaps(prose=40),
        on_report=seen.append,
    )

    (report,) = seen
    assert report.text_chars_before > report.text_chars_after > 0
    assert report.literals_clipped > 0


def test_chapter_report_fires_once_per_chapter_not_once_per_reader() -> None:
    """The snapshot is shared across the fan-out; the chapter is built once.

    Counting it per reader would report the traffic, not the context.
    """
    from ontocast.onto.ontology_condense import CondenseReport
    from ontocast.onto.ontology_snapshot import OntologySnapshot

    snapshot = OntologySnapshot(
        graph=_catalog(), source_iris=[URIRef("https://example.org")]
    )
    profile = get_graph_format_profile(LLMGraphFormat.TURTLE)
    caps = TextCaps(prose=40)
    seen: list[CondenseReport] = []

    for _ in range(4):
        snapshot.prompt_chapter(profile, text_caps=caps, on_report=seen.append)

    assert len(seen) == 1


def test_report_is_published_even_when_the_caps_are_inert() -> None:
    """The inert case is the one a deployment most needs told.

    Sizing a cap is a property of the catalog, so "your caps did nothing here"
    has to be readable rather than indistinguishable from "no caps set".
    """
    from ontocast.onto.ontology_condense import CondenseReport

    graph = _catalog(prose=30, note=30)
    profile = get_graph_format_profile(LLMGraphFormat.TURTLE)
    seen: list[CondenseReport] = []

    profile.format_ontology_chapter(
        graph,
        max_triples=4000,
        text_caps=TextCaps(naming=80, contract=240, prose=160),
        on_report=seen.append,
    )

    (report,) = seen
    assert report.literals_clipped == 0
    assert report.text_chars_before == report.text_chars_after > 0
