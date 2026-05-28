import os
from pathlib import Path

import pytest
from rdflib import URIRef

from ontocast.agent.criticise_facts import criticise_facts
from ontocast.agent.criticise_ontology import criticise_ontology
from ontocast.agent.render_facts import render_facts
from ontocast.agent.render_ontology import render_ontology
from ontocast.config import Config, LLMProvider
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import Status
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.unit_states import UnitFactsState, UnitOntologyState
from ontocast.toolbox import ToolBox

RUN_MANUAL_TESTS = os.getenv("ONTOCAST_RUN_MANUAL_TESTS", "0") == "1"

pytestmark = [
    pytest.mark.skipif(
        not RUN_MANUAL_TESTS,
        reason="Set ONTOCAST_RUN_MANUAL_TESTS=1 to run live manual tests.",
    ),
]


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        pytest.fail(f"Missing required environment variable: {name}")
    return value


def _create_tools_from_env() -> ToolBox:
    _ = _require_env("LLM_PROVIDER")
    _ = _require_env("LLM_MODEL_NAME")
    provider = LLMProvider(_require_env("LLM_PROVIDER").lower())
    if provider in (
        LLMProvider.OPENAI,
        LLMProvider.ANTHROPIC,
        LLMProvider.GOOGLE,
    ):
        _ = _require_env("LLM_API_KEY")
    elif provider == LLMProvider.OLLAMA:
        _ = _require_env("LLM_BASE_URL")

    _ = _require_env("ONTOCAST_WORKING_DIRECTORY")

    config = Config()
    config.validate_llm_config()

    if config.tool_config.path_config.working_directory is None:
        pytest.fail("ONTOCAST_WORKING_DIRECTORY must be set to run manual agent tests.")

    config.tool_config.path_config.working_directory = Path(
        config.tool_config.path_config.working_directory
    ).expanduser()
    config.tool_config.path_config.working_directory.mkdir(parents=True, exist_ok=True)

    if config.tool_config.path_config.ontology_directory is not None:
        config.tool_config.path_config.ontology_directory = Path(
            config.tool_config.path_config.ontology_directory
        ).expanduser()

    return ToolBox(config)


@pytest.fixture(scope="module")
def live_tools() -> ToolBox:
    return _create_tools_from_env()


@pytest.fixture
def realistic_text() -> str:
    return (
        "ACME Robotics announced that it signed a three-year collaboration with "
        "North Valley Hospital in Berlin to deploy autonomous delivery carts across "
        "seven departments. The pilot starts in May 2026 and is co-funded by the "
        "hospital innovation office and the regional health authority. "
        "The agreement names Dr. Lena Fischer as clinical lead and ACME CTO "
        "Rahul Mehta as technical lead. Success metrics include a 20 percent "
        "reduction in nurse walking distance, fewer late medication rounds, and "
        "weekly safety audits."
    )


def _build_seed_ontology() -> Ontology:
    graph = RDFGraph()
    graph.parse(
        data="""
        @prefix ex: <https://example.com/health#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

        ex:health a owl:Ontology ;
            rdfs:label "Healthcare Collaboration Ontology" .

        ex:Organization a rdfs:Class .
        ex:Hospital a rdfs:Class ; rdfs:subClassOf ex:Organization .
        ex:Company a rdfs:Class ; rdfs:subClassOf ex:Organization .
        ex:Person a rdfs:Class .
        ex:collaboratesWith a owl:ObjectProperty .
        ex:hasLead a owl:ObjectProperty .
        ex:locatedIn a owl:ObjectProperty .
        """,
        format="turtle",
    )
    return Ontology(graph=graph, iri="https://example.com/health")


def _build_content_unit(text: str, with_seed_facts: bool = False) -> ContentUnit:
    unit = ContentUnit(
        text=text,
        index=0,
        doc_iri=URIRef("https://example.com/doc/manual-live"),
    )
    if with_seed_facts:
        unit.graph.parse(
            data="""
            @prefix ex: <https://example.com/health#> .
            @prefix facts: <https://example.com/facts/> .
            facts:acme ex:collaboratesWith facts:north_valley_hospital .
            """,
            format="turtle",
        )
    return unit


@pytest.mark.anyio
async def test_render_facts_live_llm(live_tools: ToolBox, realistic_text: str) -> None:
    state = UnitFactsState(
        content_unit=_build_content_unit(realistic_text),
        ontology_snapshot=_build_seed_ontology(),
        facts_user_instruction=(
            "Extract organizations, people, timeline details, and measurable targets."
        ),
    )

    result = await render_facts(state, live_tools.get_atomic_tools())

    assert result.failure_stage is None
    assert result.status == Status.SUCCESS
    assert len(result.content_unit.graph) > 0
    assert result.budget_tracker.calls_count > 0


@pytest.mark.anyio
async def test_criticise_facts_live_llm(
    live_tools: ToolBox, realistic_text: str
) -> None:
    state = UnitFactsState(
        content_unit=_build_content_unit(realistic_text),
        ontology_snapshot=_build_seed_ontology(),
        facts_user_instruction=(
            "Prioritize correct entities, relations, and measurable outcomes."
        ),
    )
    rendered = await render_facts(state, live_tools.get_atomic_tools())
    assert len(rendered.content_unit.graph) > 0

    critiqued = await criticise_facts(rendered, live_tools.get_atomic_tools())

    assert (
        critiqued.failure_stage is None
        or critiqued.failure_stage.name == "FACTS_CRITIQUE"
    )
    assert critiqued.status in (Status.SUCCESS, Status.FAILED)
    assert critiqued.budget_tracker.calls_count > 0


@pytest.mark.anyio
async def test_render_ontology_live_llm(
    live_tools: ToolBox, realistic_text: str
) -> None:
    null_ontology = Ontology()
    state = UnitOntologyState(
        content_unit=_build_content_unit(realistic_text),
        ontology_snapshot=null_ontology,
        ontology_user_instruction=(
            "Create a compact ontology for healthcare logistics collaboration."
        ),
    )

    result = await render_ontology(state, live_tools.get_atomic_tools())

    assert result.failure_stage is None
    assert result.status == Status.SUCCESS
    assert not result.current_ontology.is_null()
    assert len(result.current_ontology.graph) > 0
    assert result.budget_tracker.calls_count > 0


@pytest.mark.anyio
async def test_criticise_ontology_live_llm(
    live_tools: ToolBox, realistic_text: str
) -> None:
    null_ontology = Ontology()
    state = UnitOntologyState(
        content_unit=_build_content_unit(realistic_text),
        ontology_snapshot=null_ontology,
        ontology_user_instruction=(
            "Keep class hierarchy minimal and ensure relation naming consistency."
        ),
    )
    rendered = await render_ontology(state, live_tools.get_atomic_tools())
    assert not rendered.current_ontology.is_null()
    assert len(rendered.current_ontology.graph) > 0

    critiqued = await criticise_ontology(rendered, live_tools.get_atomic_tools())

    assert (
        critiqued.failure_stage is None
        or critiqued.failure_stage.name == "ONTOLOGY_CRITIQUE"
    )
    assert critiqued.status in (Status.SUCCESS, Status.FAILED)
    assert critiqued.budget_tracker.calls_count > 0
