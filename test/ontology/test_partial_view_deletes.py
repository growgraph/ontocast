"""Deletes and partial context: one flag on the snapshot decides both the
prompt notice and whether a unit's deletes need a redeclaration."""

import asyncio
import importlib

import pytest
from rdflib import OWL, RDF, RDFS, Literal, URIRef

from ontocast.config import (
    Config,
    LLMConfig,
    LLMProvider,
    OllamaModel,
    PathConfig,
    ToolConfig,
)
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import OntologyAssemblyMode, RenderMode, Status
from ontocast.onto.model import (
    GraphUpdateRenderReport,
    OntologyConsolidationReport,
    OntologyIdentityProposal,
    OntologyPlanReport,
)
from ontocast.onto.ontology_apply import OntologyDelta
from ontocast.onto.ontology_snapshot import OntologySnapshot
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState, InputDocument
from ontocast.onto.unit_states import UnitOntologyState
from ontocast.stategraph.context_resolver import UnitOntologyContext
from ontocast.stategraph.create import create_agent_graph
from ontocast.stategraph.helpers import enforce_redeclared_deletes
from ontocast.toolbox import ToolBox
from test.ontology_helpers import DOMAIN, stamped_ontology

pytestmark = pytest.mark.unit

resolve_module = importlib.import_module("ontocast.agent.resolve_home_ontology")
create_module = importlib.import_module("ontocast.stategraph.create")
plan_module = importlib.import_module("ontocast.agent.plan_ontology")
consolidate_module = importlib.import_module("ontocast.agent.consolidate_ontology")
unit_loops = importlib.import_module("ontocast.stategraph.atomic")

SPORTS = f"{DOMAIN}/sports"
NS = f"{SPORTS}#"
VENUE = URIRef(f"{NS}Venue")
LEAGUE = URIRef(f"{NS}League")


def _catalog_graph() -> RDFGraph:
    graph = RDFGraph()
    for term, name in ((VENUE, "Venue"), (LEAGUE, "League")):
        graph.add((term, RDF.type, OWL.Class))
        graph.add((term, RDFS.label, Literal(name)))
    return graph


def _graph(*triples) -> RDFGraph:
    graph = RDFGraph()
    for triple in triples:
        graph.add(triple)
    return graph


# --- the helper ----------------------------------------------------------------


def test_only_deletes_judged_on_a_partial_view_need_a_redeclaration() -> None:
    venue_label = (VENUE, RDFS.label, Literal("Venue"))
    league_label = (LEAGUE, RDFS.label, Literal("League"))
    delta = OntologyDelta(
        inserts=RDFGraph(),
        deletes=_graph(venue_label, league_label),
    )

    dropped = enforce_redeclared_deletes(delta, _graph(venue_label))

    assert dropped == 1
    assert venue_label not in delta.deletes
    assert league_label in delta.deletes


def test_a_redeclared_delete_survives_a_partial_view() -> None:
    old = (VENUE, RDFS.label, Literal("Venue"))
    delta = OntologyDelta(
        inserts=_graph((VENUE, RDFS.label, Literal("Sports venue"))),
        deletes=_graph(old),
    )

    assert enforce_redeclared_deletes(delta, _graph(old)) == 0
    assert old in delta.deletes


# --- a unit cannot delete what it was not shown ---------------------------------


def test_a_delete_of_a_statement_outside_the_snapshot_never_reaches_the_delta() -> None:
    unseen = (URIRef(f"{DOMAIN}/other#Thing"), RDFS.label, Literal("Thing"))
    shown = (VENUE, RDFS.label, Literal("Venue"))
    state = UnitOntologyState(
        content_unit=ContentUnit(
            text="text", index=0, doc_iri=URIRef(f"{DOMAIN}/doc/d1")
        ),
        ontology_snapshot=OntologySnapshot.from_graph(
            _catalog_graph(),
            source_iris=[SPORTS],
            assembly_mode=OntologyAssemblyMode.FIXED_SINGLE_ONTOLOGY,
        ),
    )
    report = GraphUpdateRenderReport(delete_graph=_graph(unseen, shown))
    state.ontology_updates.append(report.to_graph_update())

    deletes = state.build_delta().deletes

    assert shown in deletes
    assert unseen not in deletes


# --- through the graph: the flag decides, per unit -------------------------------


def _config() -> Config:
    return Config(
        tool_config=ToolConfig(
            path_config=PathConfig(),
            llm_config=LLMConfig(
                provider=LLMProvider.OLLAMA,
                model_name=OllamaModel.LLAMA3_1,
                base_url="http://localhost:11434",
            ),
        ),
    )


def _run_with_a_bare_delete(monkeypatch, *, partial: bool) -> dict:
    """One document whose single unit deletes a catalog label without
    restating it, against a snapshot marked ``partial`` or whole."""
    tools = ToolBox(_config())
    sports = stamped_ontology(
        _catalog_graph(), ontology_id="sports", iri=SPORTS, title="Sports"
    )
    tools.ontology_manager.add_ontology(sports, skip_vector_index=True)

    async def identity(**_kwargs) -> OntologyIdentityProposal:
        return OntologyIdentityProposal(
            ontology_id="sports", title="Sports", description="Sport venues."
        )

    def convert(state: AgentState, tools: ToolBox) -> AgentState:
        return state

    async def chunk(state: AgentState, tools: ToolBox) -> AgentState:
        assert state.input_document is not None
        state.content_units = [
            ContentUnit(
                text=state.input_document.data.decode("utf-8"),
                index=0,
                doc_iri=state.doc_iri,
            )
        ]
        state.status = Status.SUCCESS
        return state

    async def resolve(_state, _tools, _unit, **_kwargs) -> UnitOntologyContext:
        snapshot = OntologySnapshot.from_ontology(
            sports, assembly_mode=OntologyAssemblyMode.FIXED_SINGLE_ONTOLOGY
        )
        snapshot.partial = partial
        return UnitOntologyContext(
            snapshot=snapshot, writable_iris=[SPORTS], confidence=1.0
        )

    async def render(state: UnitOntologyState, tools, **kwargs):
        report = GraphUpdateRenderReport(
            delete_graph=_graph((VENUE, RDFS.label, Literal("Venue")))
        )
        state.ontology_updates.append(report.to_graph_update())
        state.update_ontology()
        state.writable_iris = [SPORTS]
        state.status = Status.SUCCESS
        return state

    async def critic(state: UnitOntologyState, tools) -> UnitOntologyState:
        state.status = Status.SUCCESS
        return state

    def passthrough_factory(_tools: ToolBox):
        async def node(state: AgentState) -> AgentState:
            return state

        return node

    async def passthrough_serialize(state: AgentState, tools: ToolBox) -> AgentState:
        return state

    async def empty_plan(**_kwargs) -> OntologyPlanReport:
        return OntologyPlanReport()

    async def keep_everything(**_kwargs) -> OntologyConsolidationReport:
        return OntologyConsolidationReport()

    monkeypatch.setattr(resolve_module, "call_llm_with_retry", identity)
    monkeypatch.setattr(create_module, "convert_document", convert)
    monkeypatch.setattr(create_module, "chunk_text", chunk)
    monkeypatch.setattr(create_module, "serialize", passthrough_serialize)
    for factory in (
        "make_normalize_ontology_node",
        "make_structural_check_node",
        "make_consistency_critic_node",
    ):
        monkeypatch.setattr(create_module, factory, passthrough_factory)
    monkeypatch.setattr(plan_module, "call_llm_with_retry", empty_plan)
    monkeypatch.setattr(consolidate_module, "call_llm_with_retry", keep_everything)
    monkeypatch.setattr(unit_loops, "render_ontology", render)
    monkeypatch.setattr(unit_loops, "criticise_ontology", critic)
    monkeypatch.setattr(unit_loops, "resolve_unit_ontology_context", resolve)

    document = AgentState(
        input_document=InputDocument(filename="d.txt", data=b"Venues of the league."),
        render_mode=RenderMode.ONTOLOGY,
        current_domain=DOMAIN,
    )
    return asyncio.run(create_agent_graph(tools).ainvoke(document))


def test_a_bare_delete_judged_on_a_partial_view_is_dropped(monkeypatch) -> None:
    final = _run_with_a_bare_delete(monkeypatch, partial=True)
    assert final["ontology_reduce_metrics"]["deletes_dropped_unredeclared"] == 1


def test_a_bare_delete_judged_on_the_whole_ontology_is_kept(monkeypatch) -> None:
    final = _run_with_a_bare_delete(monkeypatch, partial=False)
    assert final["ontology_reduce_metrics"]["deletes_dropped_unredeclared"] == 0
