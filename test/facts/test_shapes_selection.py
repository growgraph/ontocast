"""Per-unit conformance-chapter selection: the context join in the loop.

Selection happens where the snapshot exists -- inside the unit loop, after
context resolution -- because the fan-out builds unit states before any
retrieval has run. The flag on the state is what keeps the loop from
touching the shapes catalog for deployments (and test doubles) that never
asked for selection.
"""

from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import RDF, URIRef

from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import OntologyAssemblyMode
from ontocast.onto.ontology_snapshot import OntologySnapshot
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.unit_states import UnitFactsState
from ontocast.stategraph.atomic import _select_conformance_chapter
from ontocast.toolbox import ToolBox

pytestmark = pytest.mark.unit

_TARGET = URIRef("https://vocab.test/qqval#QualifiedQuantityValue")


def _unit_state(*, pending: bool, with_context: bool) -> UnitFactsState:
    graph = RDFGraph()
    if with_context:
        graph.add(
            (
                _TARGET,
                RDF.type,
                URIRef("http://www.w3.org/2002/07/owl#Class"),
            )
        )
    snapshot = OntologySnapshot.from_graph(
        graph,
        source_iris=[],
        assembly_mode=OntologyAssemblyMode.SELECTED_VECTOR_SEARCH_ENSEMBLE,
        title="t",
        description="d",
    )
    unit = ContentUnit(
        text="a value of 96 meV",
        index=0,
        doc_iri=URIRef("https://example.com/doc/d1"),
        graph=RDFGraph(),
    )
    return UnitFactsState(
        content_unit=unit,
        ontology_snapshot=snapshot,
        conformance_selection_pending=pending,
    )


def test_selection_joins_snapshot_terms() -> None:
    seen: dict = {}

    def chapter_for(context_terms):
        seen["terms"] = set(context_terms)
        return "# CONFORMANCE REQUIREMENTS\n- selected"

    tools = cast(ToolBox, SimpleNamespace(shapes_chapter_for_context=chapter_for))
    state = _unit_state(pending=True, with_context=True)
    _select_conformance_chapter(state, tools)

    assert str(_TARGET) in seen["terms"]
    assert state.conformance_chapter.startswith("# CONFORMANCE REQUIREMENTS")
    assert state.conformance_selection_pending is False


def test_empty_snapshot_selects_nothing() -> None:
    tools = cast(
        ToolBox,
        SimpleNamespace(shapes_chapter_for_context=lambda terms: ""),
    )
    state = _unit_state(pending=True, with_context=False)
    _select_conformance_chapter(state, tools)
    assert state.conformance_chapter == ""
    assert state.conformance_selection_pending is False


def _toolbox_stub(mode: str, *, oversized: bool) -> ToolBox:
    """The real ToolBox methods over stubbed config + catalog state."""
    catalog = SimpleNamespace(
        prompt_contract_terms=lambda *, max_lines: ("https://vocab.test/t",),
        needs_selection=lambda *, max_lines: oversized,
        conformance_chapter=lambda *, max_lines: "# FULL CHAPTER",
        selected_chapter=lambda terms, *, max_lines: "# SELECTED",
    )
    stub = SimpleNamespace(
        config=SimpleNamespace(
            tool_config=SimpleNamespace(
                facts_validation=SimpleNamespace(
                    shapes_prompt_contract=mode,
                    shapes_prompt_max_lines=60,
                )
            )
        ),
        shapes_catalog=catalog,
    )
    stub.shapes_prompt_contract = ToolBox.shapes_prompt_contract.__get__(stub)
    stub.shapes_chapter_for_context = ToolBox.shapes_chapter_for_context.__get__(stub)
    return cast(ToolBox, stub)


@pytest.mark.parametrize(
    ("mode", "oversized", "expect_chapter", "expect_pending"),
    [
        ("off", False, "", False),
        ("full", True, "# FULL CHAPTER", False),
        ("context", False, "", True),
        ("auto", False, "# FULL CHAPTER", False),
        ("auto", True, "", True),
    ],
)
def test_mode_resolution(mode, oversized, expect_chapter, expect_pending) -> None:
    tools = _toolbox_stub(mode, oversized=oversized)
    chapter, terms, pending = tools.shapes_prompt_contract()
    assert chapter == expect_chapter
    assert pending is expect_pending
    # Exemption terms are full-catalog in every mode that has shapes.
    assert terms == ((), ("https://vocab.test/t",))[mode != "off"]


def test_chapter_for_context_delegates() -> None:
    tools = _toolbox_stub("context", oversized=True)
    assert tools.shapes_chapter_for_context({"x"}) == "# SELECTED"
