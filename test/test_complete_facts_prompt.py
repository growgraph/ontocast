"""Chapters the completion pass sends, and the empty cases that skip the call.

The completion pass is a paid call per unit. Each of these builders returns
``""`` when it has nothing to say, and an empty MISSING MEASUREMENTS chapter is
what makes the pass skip a unit entirely -- so the empty cases are a cost
contract, not a formatting nicety. The module shipped untested.
"""

import pytest
from rdflib import RDF, RDFS, Literal, URIRef

from ontocast.onto.enum import LLMGraphFormat
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.prompt.complete_facts import (
    build_catalog_subjects_chapter,
    build_missing_measurements_chapter,
    output_instruction_for,
)
from ontocast.util.measurement_lexicon import Mention
from ontocast.util.numeric_inventory import NumericInventory

pytestmark = pytest.mark.unit

_SAMPLE = URIRef("https://growgraph.dev/facts/sample_1")
_CATALOG_CLASS = "https://growgraph.dev/ontologies/matsci#Sample"


def _mention(value: str, unit: str, context: str) -> Mention:
    return Mention(value=value, unit=unit, start=0, end=0, context=context)


def test_an_empty_inventory_yields_no_chapter() -> None:
    """No missing measurement means no chapter, which means no paid call."""
    assert build_missing_measurements_chapter(NumericInventory()) == ""


def test_unclassified_numbers_alone_do_not_open_the_chapter() -> None:
    """A bare number is not a stated measurement, so it must not buy a call."""
    inventory = NumericInventory(measurements=[], unclassified=["3", "2015"])
    assert build_missing_measurements_chapter(inventory) == ""


def test_each_missing_measurement_carries_its_value_unit_and_context() -> None:
    """Context is what lets the pass place a value; without it the ask is blind."""
    inventory = NumericInventory(
        measurements=[
            _mention("96", "meV", "the shift was 96 meV overall"),
            _mention("300", "K", "annealed at 300 K for an hour"),
        ]
    )
    chapter = build_missing_measurements_chapter(inventory)

    assert "# MISSING MEASUREMENTS" in chapter
    assert '"96 meV"' in chapter
    assert "the shift was 96 meV overall" in chapter
    assert '"300 K"' in chapter
    assert "annealed at 300 K for an hour" in chapter


def test_no_catalog_terms_means_no_subjects_chapter() -> None:
    """With nothing to reuse, listing subjects would be prompt noise."""
    graph = RDFGraph()
    graph.add((_SAMPLE, RDF.type, URIRef(_CATALOG_CLASS)))
    assert build_catalog_subjects_chapter(graph, []) == ""


def test_a_subject_typed_outside_the_catalog_is_not_offered() -> None:
    """Offering an off-catalog subject invites attaching a fact to the wrong node."""
    graph = RDFGraph()
    graph.add((_SAMPLE, RDF.type, URIRef("https://example.org/other#Thing")))
    assert build_catalog_subjects_chapter(graph, [_CATALOG_CLASS]) == ""


def test_a_catalog_typed_subject_is_offered_for_reuse() -> None:
    """The whole point of the chapter: attach, do not mint a duplicate."""
    graph = RDFGraph()
    graph.add((_SAMPLE, RDF.type, URIRef(_CATALOG_CLASS)))
    graph.add((_SAMPLE, RDFS.label, Literal("aged film A")))

    chapter = build_catalog_subjects_chapter(graph, [_CATALOG_CLASS])

    assert "# EXISTING SUBJECTS" in chapter
    assert "sample_1" in chapter
    assert "aged film A" in chapter


def test_the_subject_list_respects_its_cap() -> None:
    """A large graph must not silently become the largest chapter in the prompt."""
    graph = RDFGraph()
    for index in range(25):
        subject = URIRef(f"https://growgraph.dev/facts/sample_{index}")
        graph.add((subject, RDF.type, URIRef(_CATALOG_CLASS)))

    chapter = build_catalog_subjects_chapter(graph, [_CATALOG_CLASS], limit=5)

    listed = [line for line in chapter.splitlines() if "sample_" in line]
    assert len(listed) == 5


@pytest.mark.parametrize("fmt", list(LLMGraphFormat))
def test_every_wire_format_has_an_output_instruction(fmt: LLMGraphFormat) -> None:
    """The pass must not emit an empty output contract for a supported wire."""
    assert output_instruction_for(fmt).strip()
