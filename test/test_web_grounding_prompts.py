"""Tests for web-search prompt/schema stripping when grounding is disabled."""

import importlib
from typing import cast

import pytest
from rdflib import URIRef

from ontocast.config import WebSearchConfig
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.model import (
    FactsCritiqueReport,
    FactsRenderReport,
    GraphUpdateRenderReport,
    OntologyCritiqueReport,
    OntologyRenderReport,
)
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.unit_states import UnitFactsState, UnitOntologyState
from ontocast.tool.atomic import AtomicToolBox
from ontocast.tool.llm import LLMTool
from test.snapshot_helpers import snapshot_from_ontology

render_ontology_module = importlib.import_module("ontocast.agent.render_ontology")
render_facts_module = importlib.import_module("ontocast.agent.render_facts")
criticise_ontology_module = importlib.import_module("ontocast.agent.criticise_ontology")
criticise_facts_module = importlib.import_module("ontocast.agent.criticise_facts")

_SEARCH_MARKERS = (
    "external_evidence_request",
    "ExternalEvidenceRequest",
    "initiate_search",
)


def _build_content_unit() -> ContentUnit:
    return ContentUnit(
        text="Alice works for ACME.",
        index=0,
        doc_iri=URIRef("https://example.com/doc/d1"),
    )


def _build_ontology() -> Ontology:
    graph = RDFGraph()
    graph.parse(
        data="""
        @prefix onto: <https://example.com/onto#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        onto:CompanyOntology a owl:Ontology .
        """,
        format="turtle",
    )
    return Ontology(iri="https://example.com/onto", graph=graph)


class _FakeLLMProvider:
    async def get_llm_tool(self, budget_tracker: object) -> LLMTool:
        return cast(LLMTool, object())


def _tools_with_web_search(
    *,
    web_search_enabled: bool = False,
    web_search_for_ontology_render: bool = True,
    web_search_for_facts_render: bool = False,
) -> AtomicToolBox:
    return AtomicToolBox(
        llm_provider=_FakeLLMProvider(),
        web_search_config=WebSearchConfig(
            enabled=web_search_enabled,
            ontology_render_enabled=web_search_for_ontology_render,
            facts_render_enabled=web_search_for_facts_render,
        ),
    )


def _assert_no_search_surface(prompt_kwargs: dict[str, object]) -> None:
    combined = "\n".join(str(value) for value in prompt_kwargs.values())
    for marker in _SEARCH_MARKERS:
        assert marker not in combined


# One contract -- "web_search_enabled=False strips every search surface from the
# prompt kwargs" -- applied to each agent that builds a prompt. These were four
# copy-pasted bodies differing only in module, report type, and state type.
_DISABLED_SEARCH_CASES = [
    pytest.param(
        lambda: render_ontology_module,
        lambda: OntologyRenderReport(ontology=_build_ontology()),
        lambda: UnitOntologyState(
            content_unit=_build_content_unit(),
            ontology_snapshot=snapshot_from_ontology(_build_ontology()),
        ),
        "render_ontology_fresh",
        id="render_ontology_fresh",
    ),
    pytest.param(
        lambda: criticise_ontology_module,
        lambda: OntologyCritiqueReport(
            success=True,
            score=95,
            systemic_critique_summary="Looks good.",
            actionable_ontology_fixes=[],
        ),
        lambda: UnitOntologyState(
            content_unit=_build_content_unit(),
            ontology_snapshot=snapshot_from_ontology(_build_ontology()),
        ),
        "criticise_ontology",
        id="criticise_ontology",
    ),
    pytest.param(
        lambda: render_facts_module,
        FactsRenderReport,
        lambda: UnitFactsState(
            content_unit=_build_content_unit(),
            ontology_snapshot=snapshot_from_ontology(_build_ontology()),
        ),
        "render_facts_fresh",
        id="render_facts_fresh",
    ),
    pytest.param(
        lambda: criticise_facts_module,
        lambda: FactsCritiqueReport(
            success=True,
            score=95,
            systemic_critique_summary="Looks good.",
            actionable_triple_fixes=[],
        ),
        lambda: UnitFactsState(
            content_unit=_build_content_unit(),
            ontology_snapshot=snapshot_from_ontology(_build_ontology()),
        ),
        "criticise_facts",
        id="criticise_facts",
    ),
]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("module_factory", "report_factory", "state_factory", "entrypoint"),
    _DISABLED_SEARCH_CASES,
)
async def test_agent_omits_search_surface_when_disabled(
    monkeypatch,
    module_factory,
    report_factory,
    state_factory,
    entrypoint,
) -> None:
    module = module_factory()
    captured: dict[str, object] = {}

    async def fake_call_llm_with_retry(**kwargs):
        captured.update(kwargs["prompt_kwargs"])
        return report_factory()

    monkeypatch.setattr(module, "call_llm_with_retry", fake_call_llm_with_retry)
    tools = _tools_with_web_search(web_search_enabled=False)

    await getattr(module, entrypoint)(state_factory(), tools=tools)

    _assert_no_search_surface(captured)


@pytest.mark.anyio
async def test_per_node_facts_render_stripped_while_ontology_render_keeps_search(
    monkeypatch,
) -> None:
    ontology_captured: dict[str, object] = {}
    facts_captured: dict[str, object] = {}

    async def fake_ontology_call(**kwargs):
        ontology_captured.update(kwargs["prompt_kwargs"])
        return GraphUpdateRenderReport()

    async def fake_facts_call(**kwargs):
        facts_captured.update(kwargs["prompt_kwargs"])
        return FactsRenderReport()

    monkeypatch.setattr(
        render_ontology_module, "call_llm_with_retry", fake_ontology_call
    )
    monkeypatch.setattr(render_facts_module, "call_llm_with_retry", fake_facts_call)

    tools = _tools_with_web_search(
        web_search_enabled=True,
        web_search_for_ontology_render=True,
        web_search_for_facts_render=False,
    )

    ontology_state = UnitOntologyState(
        content_unit=_build_content_unit(),
        ontology_snapshot=snapshot_from_ontology(_build_ontology()),
    )
    await render_ontology_module.render_ontology_update(ontology_state, tools=tools)

    facts_state = UnitFactsState(
        content_unit=_build_content_unit(),
        ontology_snapshot=snapshot_from_ontology(_build_ontology()),
    )
    await render_facts_module.render_facts_fresh(facts_state, tools=tools)

    ontology_combined = "\n".join(str(v) for v in ontology_captured.values())

    assert "external_evidence_request" in ontology_combined
    _assert_no_search_surface(facts_captured)
