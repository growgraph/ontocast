"""LLM_OUTPUT_LAYOUT and the parse-recovery counters."""

import pytest

import ontocast.onto.rdfgraph as rdfgraph_module
from ontocast.config.settings import ServerConfig
from ontocast.onto.enum import LLMGraphFormat, LLMOutputLayout
from ontocast.onto.model import FactsRenderReport
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import BudgetTracker
from ontocast.onto.unit_states import UnitFactsState
from ontocast.prompt.graph_format import get_graph_format_profile
from ontocast.tool.llm import use_budget_tracker

pytestmark = pytest.mark.unit


def test_layout_defaults_to_compact() -> None:
    assert ServerConfig.model_fields["llm_output_layout"].default == (
        LLMOutputLayout.COMPACT
    )
    assert UnitFactsState.model_fields["llm_output_layout"].default == (
        LLMOutputLayout.COMPACT
    )


@pytest.mark.parametrize("fmt", list(LLMGraphFormat))
def test_free_layout_adds_nothing_to_the_format_instructions(
    fmt: LLMGraphFormat,
) -> None:
    free = get_graph_format_profile(fmt, output_layout=LLMOutputLayout.FREE)
    default = get_graph_format_profile(fmt)
    assert free is default
    assert "RESPONSE LAYOUT" not in free.format_instructions(FactsRenderReport)


def test_compact_jsonld_asks_for_minified_json_only() -> None:
    profile = get_graph_format_profile(
        LLMGraphFormat.JSONLD, output_layout=LLMOutputLayout.COMPACT
    )
    text = profile.format_instructions(FactsRenderReport)
    assert "RESPONSE LAYOUT" in text
    assert "minified" in text
    assert "Turtle string" not in text


def test_compact_turtle_also_flattens_the_turtle_string() -> None:
    profile = get_graph_format_profile(
        LLMGraphFormat.TURTLE, output_layout=LLMOutputLayout.COMPACT
    )
    text = profile.format_instructions(FactsRenderReport)
    assert "minified" in text
    assert "Inside a Turtle string" in text


def test_layout_leaves_output_instructions_and_parsing_alone() -> None:
    free = get_graph_format_profile(LLMGraphFormat.JSONLD)
    compact = get_graph_format_profile(
        LLMGraphFormat.JSONLD, output_layout=LLMOutputLayout.COMPACT
    )
    assert free.render_fresh_output_instruction() == (
        compact.render_fresh_output_instruction()
    )
    assert free.critique_graph_instruction() == compact.critique_graph_instruction()
    minified = (
        '{"semantic_graph":{"@context":{"ex":"http://ex.org/"},'
        '"@graph":[{"@id":"ex:a","ex:p":{"@id":"ex:b"}}]},'
        '"ontology_relevance_score":90,"triples_generation_score":90}'
    )
    report = compact.parse_report(FactsRenderReport, minified)
    assert len(report.semantic_graph) == 1


def test_one_line_turtle_subjects_parse() -> None:
    graph = RDFGraph._from_turtle_str(
        '@prefix ex: <http://ex.org/> . ex:a ex:p ex:b ; ex:q "x" . ex:b ex:p ex:c .'
    )
    assert len(graph) == 3


def test_repaired_turtle_is_counted() -> None:
    tracker = BudgetTracker()
    with use_budget_tracker(tracker):
        graph = RDFGraph._from_turtle_str(
            "@prefix ex: <http://ex.org/> .\nex:a ex:p ex:b ;"
        )
    assert len(graph) == 1
    assert tracker.counters == {"rdf/turtle_repair": 1}


def test_clean_turtle_is_not_counted() -> None:
    tracker = BudgetTracker()
    with use_budget_tracker(tracker):
        RDFGraph._from_turtle_str("@prefix ex: <http://ex.org/> .\nex:a ex:p ex:b .")
    assert tracker.counters == {}


def test_jsonld_rdflib_fallback_is_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject(*args: object, **kwargs: object) -> str:
        raise ValueError("normalization rejected")

    monkeypatch.setattr(rdfgraph_module.jsonld, "normalize", reject)
    tracker = BudgetTracker()
    with use_budget_tracker(tracker):
        graph = RDFGraph._from_jsonld_obj(
            {
                "@context": {"ex": "http://ex.org/"},
                "@graph": [{"@id": "ex:a", "ex:p": {"@id": "ex:b"}}],
            }
        )
    assert len(graph) == 1
    assert tracker.counters == {"rdf/jsonld_rdflib_fallback": 1}


def test_recovery_without_a_tracker_is_a_no_op() -> None:
    graph = RDFGraph._from_turtle_str(
        "@prefix ex: <http://ex.org/> .\nex:a ex:p ex:b ;"
    )
    assert len(graph) == 1
