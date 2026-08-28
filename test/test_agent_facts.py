import importlib
from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import Literal, URIRef

from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import FailureStage, LLMGraphFormat, Status
from ontocast.onto.model import (
    FactsCritiqueReport,
    FactsRenderReport,
    GraphUpdateRenderReport,
    TripleFix,
)
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import GraphUpdate, TripleOp
from ontocast.onto.unit_states import UnitFactsState
from ontocast.tool.atomic import AtomicToolBox
from test.snapshot_helpers import snapshot_from_ontology

criticise_facts_module = importlib.import_module("ontocast.agent.criticise_facts")
render_facts_module = importlib.import_module("ontocast.agent.render_facts")


def _build_content_unit(with_graph: bool = False) -> ContentUnit:
    unit = ContentUnit(
        text="Alice works for ACME.",
        index=0,
        doc_iri=URIRef("https://example.com/doc/d1"),
    )
    if with_graph:
        unit.graph.parse(
            data="""
            @prefix ex: <https://example.com/ns#> .
            ex:alice ex:worksFor ex:acme .
            """,
            format="turtle",
        )
    return unit


def _build_ontology() -> Ontology:
    ontology_graph = RDFGraph()
    ontology_graph.parse(
        data="""
        @prefix onto: <https://example.com/onto#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        onto:CompanyOntology a owl:Ontology .
        """,
        format="turtle",
    )
    return Ontology(graph=ontology_graph, iri="https://example.com/onto")


def _build_tools() -> AtomicToolBox:
    async def get_llm_tool(_budget_tracker):
        return object()

    return cast(
        AtomicToolBox,
        SimpleNamespace(
            get_llm_tool=get_llm_tool,
            web_grounding_enabled_for_node=lambda _node: False,
            object_property_literal_check=True,
            # Mirrors AtomicToolBox: read directly, not via getattr with a
            # duplicated literal default, so a rename fails loudly here
            # instead of silently reverting the configured ratio.
            property_alias_min_ratio=0.85,
            code_predicates=(),
            citation_vocabulary={},
            quantity_fallback_vocabulary=None,
            acceptance_policy=None,
        ),
    )


@pytest.mark.anyio
async def test_render_facts_routes_to_fresh_when_graph_is_empty(monkeypatch) -> None:
    calls = {"fresh": 0, "update": 0}

    async def fake_fresh(state: UnitFactsState, tools, **kwargs) -> UnitFactsState:
        calls["fresh"] += 1
        return state

    async def fake_update(state: UnitFactsState, tools, **kwargs) -> UnitFactsState:
        calls["update"] += 1
        return state

    monkeypatch.setattr(render_facts_module, "render_facts_fresh", fake_fresh)
    monkeypatch.setattr(render_facts_module, "render_facts_update", fake_update)

    state = UnitFactsState(
        content_unit=_build_content_unit(with_graph=False),
        ontology_snapshot=snapshot_from_ontology(_build_ontology()),
    )
    result = await render_facts_module.render_facts(state, tools=_build_tools())

    assert result is state
    assert calls["fresh"] == 1
    assert calls["update"] == 0


@pytest.mark.anyio
async def test_render_facts_fresh_sets_success_and_budget(monkeypatch) -> None:
    async def fake_call_llm_with_retry(**kwargs):
        rendered_graph = RDFGraph()
        rendered_graph.parse(
            data="""
            @prefix ex: <https://example.com/ns#> .
            ex:alice ex:worksFor ex:acme .
            """,
            format="turtle",
        )
        return FactsRenderReport(
            semantic_graph=rendered_graph,
            ontology_relevance_score=95,
            triples_generation_score=94,
        )

    monkeypatch.setattr(
        render_facts_module, "call_llm_with_retry", fake_call_llm_with_retry
    )

    state = UnitFactsState(
        content_unit=_build_content_unit(with_graph=False),
        ontology_snapshot=snapshot_from_ontology(_build_ontology()),
    )
    result = await render_facts_module.render_facts_fresh(state, tools=_build_tools())

    assert result.status == Status.SUCCESS
    assert result.failure_stage is None
    assert len(result.content_unit.graph) == 1
    assert result.budget_tracker.facts_operations_count == 1
    assert result.budget_tracker.facts_triples_generated == 1


@pytest.mark.anyio
async def test_criticise_facts_marks_failed_and_sets_suggestions(monkeypatch) -> None:
    async def fake_call_llm_with_retry(**kwargs):
        return FactsCritiqueReport(
            success=False,
            score=35,
            actionable_triple_fixes=[
                TripleFix(
                    text_fragment="Alice works for ACME.",
                    action="ADD",
                    severity="critical",
                    explanation="Missing employment relation triple.",
                    correct_value="ex:alice ex:worksFor ex:acme .",
                )
            ],
            systemic_critique_summary="Misses key relations.",
        )

    monkeypatch.setattr(
        criticise_facts_module, "call_llm_with_retry", fake_call_llm_with_retry
    )

    state = UnitFactsState(
        content_unit=_build_content_unit(with_graph=True),
        ontology_snapshot=snapshot_from_ontology(_build_ontology()),
    )
    result = await criticise_facts_module.criticise_facts(state, tools=_build_tools())

    assert result.status == Status.FAILED
    assert result.failure_stage == FailureStage.FACTS_CRITIQUE
    assert len(result.suggestions.actionable_fixes) == 1
    assert result.failure_reason == "Facts unit has 1 material defect(s)"
    assert result.attempt_log[-1].accept_reason == "critic_critical"


@pytest.mark.anyio
async def test_render_facts_fresh_coerces_invalid_typed_literal_at_ingest(
    monkeypatch,
) -> None:
    async def fake_call_llm_with_retry(**kwargs):
        rendered_graph = RDFGraph._from_jsonld_obj(
            {
                "@context": {
                    "ex": "https://example.com/ns#",
                    "xsd": "http://www.w3.org/2001/XMLSchema#",
                },
                "@graph": [
                    {
                        "@id": "ex:item",
                        "ex:value": {"@value": "10-15", "@type": "xsd:decimal"},
                    }
                ],
            }
        )
        return FactsRenderReport(
            semantic_graph=rendered_graph,
            ontology_relevance_score=95,
            triples_generation_score=94,
        )

    monkeypatch.setattr(
        render_facts_module, "call_llm_with_retry", fake_call_llm_with_retry
    )

    state = UnitFactsState(
        content_unit=_build_content_unit(with_graph=False),
        ontology_snapshot=snapshot_from_ontology(_build_ontology()),
    )
    result = await render_facts_module.render_facts_fresh(state, tools=_build_tools())

    assert result.status == Status.SUCCESS
    assert len(result.content_unit.graph) == 1
    assert len(result.quarantined_literal_triples) == 0
    obj = next(result.content_unit.graph.objects())
    assert isinstance(obj, Literal)
    assert obj.datatype is None
    assert str(obj) == "10-15"


@pytest.mark.anyio
async def test_render_facts_update_coerces_invalid_literal_in_update_graph(
    monkeypatch,
) -> None:
    async def fake_call_llm_with_retry(**kwargs):
        bad_graph = RDFGraph._from_jsonld_obj(
            {
                "@context": {
                    "ex": "https://example.com/ns#",
                    "xsd": "http://www.w3.org/2001/XMLSchema#",
                },
                "@graph": [
                    {
                        "@id": "ex:item",
                        "ex:value": {"@value": "10-15", "@type": "xsd:decimal"},
                    }
                ],
            }
        )
        return GraphUpdateRenderReport(
            graph_update=GraphUpdate(
                triple_operations=[TripleOp(type="insert", graph=bad_graph)]
            )
        )

    monkeypatch.setattr(
        render_facts_module, "call_llm_with_retry", fake_call_llm_with_retry
    )

    unit = _build_content_unit(with_graph=True)
    state = UnitFactsState(
        content_unit=unit,
        ontology_snapshot=snapshot_from_ontology(_build_ontology()),
    )
    initial_len = len(state.content_unit.graph)
    result = await render_facts_module.render_facts_update(state, tools=_build_tools())

    assert result.status == Status.SUCCESS
    assert len(result.quarantined_literal_triples) == 0
    assert len(result.content_unit.graph) == initial_len + 1


@pytest.mark.anyio
async def test_criticise_facts_prompt_includes_graph_format_instruction(
    monkeypatch,
) -> None:
    captured_prompt_kwargs: dict[str, object] = {}

    async def fake_call_llm_with_retry(**kwargs):
        captured_prompt_kwargs.update(kwargs["prompt_kwargs"])
        return FactsCritiqueReport(
            success=True,
            score=95,
            actionable_triple_fixes=[],
            systemic_critique_summary="",
        )

    monkeypatch.setattr(
        criticise_facts_module, "call_llm_with_retry", fake_call_llm_with_retry
    )

    state = UnitFactsState(
        content_unit=_build_content_unit(with_graph=True),
        ontology_snapshot=snapshot_from_ontology(_build_ontology()),
        llm_graph_format=LLMGraphFormat.JSONLD,
    )
    await criticise_facts_module.criticise_facts(state, tools=_build_tools())

    instruction = str(captured_prompt_kwargs.get("graph_format_instruction", ""))
    assert "LLM_GRAPH_FORMAT=jsonld" in instruction
    assert "incorrect_value" in instruction


@pytest.mark.anyio
async def test_criticise_facts_records_the_score_but_does_not_gate_on_it(
    monkeypatch,
) -> None:
    """The score is telemetry now, not a verdict.

    This test used to pin the opposite rule -- that ``score > 90`` accepted a
    render regardless of ``success``. Measured over the 2026-08 matsci critic
    calls that threshold rejected 28 of 34 renders, with a median score of 79
    and no rubric anywhere in the prompt to anchor the number against. A low
    score with no material defect must now accept, and the score must still be
    recorded so the calibration stays auditable from the run's own artifacts.
    """

    async def fake_call_llm_with_retry(**kwargs):
        return FactsCritiqueReport(
            success=False,
            score=55,
            actionable_triple_fixes=[],
            systemic_critique_summary="could be tighter",
        )

    monkeypatch.setattr(
        criticise_facts_module, "call_llm_with_retry", fake_call_llm_with_retry
    )

    state = UnitFactsState(
        content_unit=_build_content_unit(with_graph=True),
        ontology_snapshot=snapshot_from_ontology(_build_ontology()),
    )
    result = await criticise_facts_module.criticise_facts(state, tools=_build_tools())

    assert result.status == Status.SUCCESS
    assert result.failure_stage is None
    assert result.attempt_log[-1].score == 55
    assert result.attempt_log[-1].accept_reason == "clean"


def _opaque_iri_ontology() -> Ontology:
    """Wikidata-style catalog: the IRIs carry no meaning without labels."""
    ontology_graph = RDFGraph()
    ontology_graph.parse(
        data="""
        @prefix wd: <http://www.wikidata.org/entity/> .
        @prefix wdt: <http://www.wikidata.org/prop/direct/> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        wd:Q2013 a owl:Ontology .
        wd:Q4830453 a owl:Class ; rdfs:label "business" .
        wd:Q43229 a owl:Class ; rdfs:label "organization" .
        wdt:P169 a owl:ObjectProperty ; rdfs:label "chief executive officer" .
        """,
        format="turtle",
    )
    return Ontology(graph=ontology_graph, iri="http://www.wikidata.org/entity/Q2013")


@pytest.mark.anyio
async def test_criticise_facts_prompt_carries_the_term_index(monkeypatch) -> None:
    """The critic must see the same TERM INDEX the renderer is told to use.

    Building the chapter without the index appendix left the critic judging
    opaque IRIs it could not resolve, while facts guideline 6a instructs the
    renderer to resolve them through exactly that index.
    """
    captured: dict[str, object] = {}

    async def fake_call_llm_with_retry(**kwargs):
        captured.update(kwargs["prompt_kwargs"])
        return FactsCritiqueReport(
            success=True,
            score=95,
            actionable_triple_fixes=[],
            systemic_critique_summary="",
        )

    monkeypatch.setattr(
        criticise_facts_module, "call_llm_with_retry", fake_call_llm_with_retry
    )

    state = UnitFactsState(
        content_unit=_build_content_unit(with_graph=True),
        ontology_snapshot=snapshot_from_ontology(_opaque_iri_ontology()),
    )
    await criticise_facts_module.criticise_facts(state, tools=_build_tools())

    chapter = str(captured.get("ontology_chapter", ""))
    assert "TERM INDEX" in chapter
    assert "chief executive officer" in chapter


@pytest.mark.anyio
async def test_criticise_facts_adds_no_index_for_a_readable_ontology(
    monkeypatch,
) -> None:
    """The appendix is empty when IRIs are self-describing, so it costs nothing."""
    captured: dict[str, object] = {}

    async def fake_call_llm_with_retry(**kwargs):
        captured.update(kwargs["prompt_kwargs"])
        return FactsCritiqueReport(
            success=True,
            score=95,
            actionable_triple_fixes=[],
            systemic_critique_summary="",
        )

    monkeypatch.setattr(
        criticise_facts_module, "call_llm_with_retry", fake_call_llm_with_retry
    )

    state = UnitFactsState(
        content_unit=_build_content_unit(with_graph=True),
        ontology_snapshot=snapshot_from_ontology(_build_ontology()),
    )
    await criticise_facts_module.criticise_facts(state, tools=_build_tools())

    assert "TERM INDEX" not in str(captured.get("ontology_chapter", ""))
