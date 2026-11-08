"""Owned and external catalog ontologies, the extension ontology, and the
deterministic clean-up of a document's ontology delta."""

import asyncio
import importlib
from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import OWL, RDF, RDFS, XSD, Literal, URIRef

from ontocast.config import (
    Config,
    LLMConfig,
    LLMProvider,
    OllamaModel,
    PathConfig,
    ToolConfig,
)
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import RenderMode, Status
from ontocast.onto.identity import (
    OntologyIdentity,
    extension_ontology_id,
    is_owned_ontology,
)
from ontocast.onto.model import (
    GraphUpdateRenderReport,
    OntologyConsolidationReport,
    OntologyIdentityProposal,
    OntologyPlanReport,
)
from ontocast.onto.ontology_apply import partition_triples_by_namespace
from ontocast.onto.ontology_header import mint_ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState, InputDocument
from ontocast.onto.unit_states import UnitOntologyState
from ontocast.stategraph.context_resolver import UnitOntologyContext
from ontocast.stategraph.create import create_agent_graph
from ontocast.tool.ontology_manager import OntologyManager
from ontocast.tool.ontology_validation.axiom_hygiene import (
    drop_example_namespace_triples,
    drop_external_subjects,
    drop_vacuous_upper_terms,
    normalize_datatype_ranges,
)
from ontocast.toolbox import ToolBox
from test.ontology_helpers import DOMAIN, stamped_ontology
from test.snapshot_helpers import snapshot_from_ontology

pytestmark = pytest.mark.unit

resolve_module = importlib.import_module("ontocast.agent.resolve_home_ontology")
create_module = importlib.import_module("ontocast.stategraph.create")
plan_module = importlib.import_module("ontocast.agent.plan_ontology")
consolidate_module = importlib.import_module("ontocast.agent.consolidate_ontology")
unit_loops = importlib.import_module("ontocast.stategraph.atomic")

EXT_IRI = "http://ext.test/onto"
SCHEMA_THING = URIRef("https://schema.org/Thing")


def _classes(namespace: str, *names: str) -> RDFGraph:
    graph = RDFGraph()
    for name in names:
        graph.add((URIRef(f"{namespace}{name}"), RDF.type, OWL.Class))
        graph.add((URIRef(f"{namespace}{name}"), RDFS.label, Literal(name)))
    return graph


def _external_head():
    """An ontology outside ``DOMAIN``: a published vocabulary the run uses."""
    return stamped_ontology(
        _classes(f"{EXT_IRI}/", "Organisation", "SportsTeam", "Company", "League"),
        ontology_id="ext",
        iri=EXT_IRI,
        title="External vocabulary",
    )


def _manager(*heads) -> OntologyManager:
    manager = OntologyManager()
    for head in heads:
        manager.add_ontology(head, skip_vector_index=True)
    return manager


def _tools(manager: OntologyManager) -> ToolBox:
    async def get_llm_tool(_budget_tracker: object) -> object:
        return object()

    return cast(
        ToolBox, SimpleNamespace(ontology_manager=manager, get_llm_tool=get_llm_tool)
    )


def _answer_identity(monkeypatch, ontology_id: str) -> None:
    async def fake_call(**_kwargs) -> OntologyIdentityProposal:
        return OntologyIdentityProposal(
            ontology_id=ontology_id,
            title=f"{ontology_id} ontology",
            description="What the document is about.",
        )

    monkeypatch.setattr(resolve_module, "call_llm_with_retry", fake_call)


def _state(text: str) -> AgentState:
    return AgentState(
        render_mode=RenderMode.ONTOLOGY,
        current_domain=DOMAIN,
        content_units=[
            ContentUnit(text=text, index=0, doc_iri=URIRef(f"{DOMAIN}/doc/d1"))
        ],
    )


# --- ownership and the extension identity ---------------------------------


def test_an_ontology_is_owned_only_under_current_domain() -> None:
    assert is_owned_ontology(f"{DOMAIN}/tmi", DOMAIN)
    assert not is_owned_ontology(EXT_IRI, DOMAIN)
    # A shared prefix is not a path under the domain.
    assert not is_owned_ontology(f"{DOMAIN}x/tmi", DOMAIN)


def test_the_extension_id_is_fixed_by_the_external_id_and_fits() -> None:
    assert extension_ontology_id("dbo") == "dbo-ext"
    long_id = "a" * 32
    assert extension_ontology_id(long_id) == "a" * 28 + "-ext"


def test_a_minted_extension_imports_what_it_extends() -> None:
    identity = OntologyIdentity(
        ontology_id="ext-ext",
        iri=f"{DOMAIN}/ext-ext",
        title="Extension",
        extends=EXT_IRI,
    )
    ontology = mint_ontology(identity, RDFGraph())
    assert (URIRef(identity.iri), OWL.imports, URIRef(EXT_IRI)) in ontology.graph


@pytest.mark.anyio
async def test_an_external_home_becomes_its_extension(monkeypatch) -> None:
    _answer_identity(monkeypatch, "ext")
    state = _state("A league of sports teams.")

    await resolve_module.resolve_home_ontology(
        state, _tools(_manager(_external_head()))
    )

    home = state.home_ontology
    assert home is not None
    assert home.iri == f"{DOMAIN}/ext-ext"
    assert home.extends == EXT_IRI
    assert home.namespace.startswith(f"{DOMAIN}/ext-ext")
    assert state.ontology_reduce_metrics["home_extension"] == 1


@pytest.mark.anyio
async def test_a_stored_extension_is_reused(monkeypatch) -> None:
    _answer_identity(monkeypatch, "ext")
    stored = stamped_ontology(
        _classes(f"{DOMAIN}/ext-ext#", "Tournament"),
        ontology_id="ext-ext",
        title="Stored extension",
    )
    state = _state("A league of sports teams.")

    await resolve_module.resolve_home_ontology(
        state, _tools(_manager(_external_head(), stored))
    )

    assert state.home_ontology is not None
    assert state.home_ontology.iri == stored.iri
    assert state.home_ontology.title == "Stored extension"


@pytest.mark.anyio
async def test_an_owned_home_is_extended_in_place(monkeypatch) -> None:
    _answer_identity(monkeypatch, "tmi")
    owned = stamped_ontology(_classes(f"{DOMAIN}/tmi#", "Sample"), ontology_id="tmi")
    state = _state("Alice works for ACME.")

    await resolve_module.resolve_home_ontology(state, _tools(_manager(owned)))

    assert state.home_ontology is not None
    assert state.home_ontology.iri == owned.iri
    assert state.home_ontology.extends is None
    assert "home_extension" not in state.ontology_reduce_metrics


# --- the coverage gate on a new home -------------------------------------


@pytest.mark.anyio
async def test_a_new_home_is_overridden_when_the_catalog_covers_the_text(
    monkeypatch,
) -> None:
    _answer_identity(monkeypatch, "sports-tournament")
    text = (
        "The league is an organisation of sports teams. Each company that "
        "sponsors a sports team joins the league; the organisation rules it."
    )
    head = stamped_ontology(
        _classes(
            f"{DOMAIN}/sport#",
            "League",
            "Organisation",
            "SportsTeam",
            "Company",
            "Sponsor",
        ),
        ontology_id="sport",
    )
    state = _state(text)

    await resolve_module.resolve_home_ontology(state, _tools(_manager(head)))

    assert state.home_ontology is not None
    assert state.home_ontology.iri == head.iri
    assert state.ontology_reduce_metrics["home_new_overridden"] == 1


@pytest.mark.anyio
async def test_a_new_home_stands_when_the_catalog_does_not_cover_the_text(
    monkeypatch,
) -> None:
    _answer_identity(monkeypatch, "chemistry")
    head = stamped_ontology(
        _classes(f"{DOMAIN}/sport#", "League", "SportsTeam"), ontology_id="sport"
    )
    state = _state("A perovskite film degrades under humidity.")

    await resolve_module.resolve_home_ontology(state, _tools(_manager(head)))

    assert state.home_ontology is not None
    assert state.home_ontology.iri == f"{DOMAIN}/chemistry"
    assert "home_new_overridden" not in state.ontology_reduce_metrics


# --- the clean-up rules ----------------------------------------------------


def test_triples_about_external_terms_are_refused() -> None:
    graph = RDFGraph()
    ext_class = URIRef(f"{EXT_IRI}/Cyclist")
    mine = URIRef(f"{DOMAIN}/ext-ext#Cyclist")
    graph.add((ext_class, RDFS.subClassOf, URIRef(f"{EXT_IRI}/Agent")))
    graph.add((mine, RDFS.subClassOf, URIRef(f"{EXT_IRI}/Athlete")))
    graph.add((URIRef(f"{EXT_IRI}/sports-tournament/Edition"), RDF.type, OWL.Class))
    assert drop_external_subjects(graph, [EXT_IRI]) == 2
    assert set(graph) == {(mine, RDFS.subClassOf, URIRef(f"{EXT_IRI}/Athlete"))}


def test_example_namespace_triples_are_dropped_outside_the_own_domain() -> None:
    graph = RDFGraph()
    stray = URIRef("https://example.com/cycling-sport#CyclingRace")
    graph.add((stray, RDFS.subClassOf, URIRef(f"{EXT_IRI}/Race")))
    graph.add((URIRef(f"{DOMAIN}/x#Kept"), RDF.type, OWL.Class))
    # DOMAIN is an example domain: the deployment's own IRIs stay.
    assert drop_example_namespace_triples(graph, DOMAIN) == 1
    assert len(graph) == 1
    assert drop_example_namespace_triples(graph, "https://growgraph.dev") == 1
    assert len(graph) == 0


def test_one_datatype_range_is_kept() -> None:
    graph = RDFGraph()
    reason, seed, score = (URIRef(f"{DOMAIN}/x#{n}") for n in ("why", "seed", "n"))
    graph.add((reason, RDFS.range, XSD.string))
    graph.add((reason, RDFS.range, RDF.langString))
    graph.add((seed, RDFS.range, XSD.string))
    graph.add((seed, RDFS.range, RDFS.Literal))
    graph.add((score, RDFS.range, XSD.decimal))
    assert normalize_datatype_ranges(graph) == 2
    assert set(graph.objects(reason, RDFS.range)) == {RDF.langString}
    assert set(graph.objects(seed, RDFS.range)) == {XSD.string}
    assert set(graph.objects(score, RDFS.range)) == {XSD.decimal}


def test_thing_is_dropped_as_range_and_as_a_second_parent() -> None:
    graph = RDFGraph()
    record, alone, prop = (URIRef(f"{DOMAIN}/x#{n}") for n in ("Rec", "Alone", "p"))
    graph.add((record, RDFS.subClassOf, SCHEMA_THING))
    graph.add((record, RDFS.subClassOf, URIRef("https://schema.org/StructuredValue")))
    graph.add((alone, RDFS.subClassOf, OWL.Thing))
    graph.add((prop, RDFS.range, SCHEMA_THING))
    assert drop_vacuous_upper_terms(graph) == 2
    assert (record, RDFS.subClassOf, SCHEMA_THING) not in graph
    # A sole parent is kept: the class would lose its place in the hierarchy.
    assert (alone, RDFS.subClassOf, OWL.Thing) in graph
    assert (prop, RDFS.range, SCHEMA_THING) not in graph


def test_a_named_subject_is_not_filed_by_its_object() -> None:
    head = _external_head()
    stray = URIRef("https://elsewhere.test/x#CyclingRace")
    inserts = RDFGraph()
    inserts.add((stray, RDFS.subClassOf, URIRef(f"{EXT_IRI}/Organisation")))
    buckets, unattributed = partition_triples_by_namespace(
        inserts, writable_iris=[head.iri], ontology_manager=_manager(head)
    )
    assert buckets == {}
    assert unattributed == 1


# --- end to end: a document whose home is an external ontology -------------


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


def test_update_renders_land_in_the_extension_and_never_in_the_external(
    monkeypatch,
) -> None:
    """Units render updates against the external ontology's patch.

    Each unit mints a class under the home (the extension, not yet stored),
    re-declares an external term with a new parent, and writes a stray
    example.com axiom. Only the home terms are written, into a new extension
    ontology that imports the external one.
    """
    tools = ToolBox(_config())
    external = _external_head()
    tools.ontology_manager.add_ontology(external, skip_vector_index=True)

    _answer_identity(monkeypatch, "ext")

    def fake_convert(state: AgentState, tools: ToolBox) -> AgentState:
        return state

    async def fake_chunk(state: AgentState, tools: ToolBox) -> AgentState:
        assert state.input_document is not None
        parts = state.input_document.data.decode("utf-8").split("\n\n")
        state.content_units = [
            ContentUnit(text=part, index=index, doc_iri=state.doc_iri)
            for index, part in enumerate(parts)
        ]
        state.status = Status.SUCCESS
        return state

    async def fake_resolve(_state, _tools, _unit, **_kwargs) -> UnitOntologyContext:
        return UnitOntologyContext(
            snapshot=snapshot_from_ontology(external),
            writable_iris=[external.iri],
            confidence=1.0,
        )

    async def update_render(state: UnitOntologyState, tools, **kwargs):
        home = state.home_ontology
        assert home is not None
        name = f"Term{state.content_unit.index}"
        inserts = RDFGraph()
        term = URIRef(f"{home.namespace}{name}")
        inserts.add((term, RDF.type, OWL.Class))
        inserts.add((term, RDFS.label, Literal(name)))
        inserts.add((term, RDFS.subClassOf, URIRef(f"{EXT_IRI}/Organisation")))
        inserts.add(
            (URIRef(f"{EXT_IRI}/SportsTeam"), RDFS.subClassOf, URIRef(f"{EXT_IRI}/X"))
        )
        inserts.add(
            (
                URIRef("https://example.com/cycling-sport#Race"),
                RDFS.subClassOf,
                URIRef(f"{EXT_IRI}/League"),
            )
        )
        report = GraphUpdateRenderReport(insert_graph=inserts)
        state.ontology_updates.append(report.to_graph_update())
        state.update_ontology()
        state.writable_iris = [external.iri]
        state.status = Status.SUCCESS
        return state

    async def fake_critic(state: UnitOntologyState, tools) -> UnitOntologyState:
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

    monkeypatch.setattr(create_module, "convert_document", fake_convert)
    monkeypatch.setattr(create_module, "chunk_text", fake_chunk)
    monkeypatch.setattr(create_module, "serialize", passthrough_serialize)
    for factory in (
        "make_normalize_ontology_node",
        "make_structural_check_node",
        "make_consistency_critic_node",
    ):
        monkeypatch.setattr(create_module, factory, passthrough_factory)
    monkeypatch.setattr(plan_module, "call_llm_with_retry", empty_plan)
    monkeypatch.setattr(consolidate_module, "call_llm_with_retry", keep_everything)
    monkeypatch.setattr(unit_loops, "render_ontology", update_render)
    monkeypatch.setattr(unit_loops, "criticise_ontology", fake_critic)
    monkeypatch.setattr(unit_loops, "resolve_unit_ontology_context", fake_resolve)

    document = AgentState(
        input_document=InputDocument(
            filename="doc.txt", data=b"teams one\n\nteams two"
        ),
        render_mode=RenderMode.ONTOLOGY,
        current_domain=DOMAIN,
    )
    final = asyncio.run(create_agent_graph(tools).ainvoke(document))

    home = final["home_ontology"]
    assert home.iri == f"{DOMAIN}/ext-ext"
    artifacts = final["ontology_artifacts"]
    assert [a.iri for a in artifacts] == [home.iri], [
        (a.iri, len(a.graph)) for a in artifacts
    ]
    (artifact,) = artifacts
    assert artifact.iri == home.iri
    assert (URIRef(home.iri), OWL.imports, URIRef(EXT_IRI)) in artifact.graph
    declared = {str(s) for s in artifact.graph.subjects(RDF.type, OWL.Class)}
    assert declared == {f"{home.namespace}Term0", f"{home.namespace}Term1"}
    subjects = {str(s) for s in artifact.graph.subjects()}
    assert not any(s.startswith(EXT_IRI) for s in subjects)
    assert not any("example.com/cycling-sport" in s for s in subjects)
    metrics = final["ontology_reduce_metrics"]
    assert metrics["external_subject_refused"] >= 1
    assert metrics["example_namespace_dropped"] >= 1


def test_the_update_prompt_names_the_home_namespace() -> None:
    """Without it the model guesses a namespace for the home prefix."""
    render_module = importlib.import_module("ontocast.agent.render_ontology")
    home = OntologyIdentity(
        ontology_id="ext-ext",
        iri=f"{DOMAIN}/ext-ext",
        title="Extension",
        extends=EXT_IRI,
    )
    state = UnitOntologyState(
        content_unit=ContentUnit(
            text="Teams play.", index=0, doc_iri=URIRef(f"{DOMAIN}/doc/d1")
        ),
        ontology_snapshot=snapshot_from_ontology(_external_head()),
        home_ontology=home,
    )
    access = render_module.ontology_access_for_unit_ontology(state)
    intro = render_module._build_update_intro(state, access)
    assert f"<{home.namespace}>" in intro
    assert "`ext-ext:`" in intro
