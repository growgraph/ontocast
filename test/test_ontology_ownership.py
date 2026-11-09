"""Owned and external catalog ontologies, the extension ontology, and the
deterministic clean-up of a document's ontology delta."""

import asyncio
import importlib
from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import OWL, PROV, RDF, RDFS, XSD, BNode, Literal, URIRef
from rdflib.collection import Collection
from rdflib.term import Node

from ontocast.config import (
    Config,
    LLMConfig,
    LLMProvider,
    OllamaModel,
    PathConfig,
    ToolConfig,
)
from ontocast.onto.constants import DEFAULT_IRI
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
from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_apply import (
    partition_triples_by_namespace,
    refile_fresh_statements,
)
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
    drop_non_term_subjects,
    drop_terms_minted_outside_home,
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
        inserts,
        writable_iris=[head.iri],
        ontology_manager=_manager(head),
        current_domain=DOMAIN,
    )
    assert buckets == {}
    assert unattributed == 1


def test_a_named_subject_is_not_filed_by_its_predicate() -> None:
    """A fact using an external property never lands in the external ontology."""
    head = _external_head()
    inserts = RDFGraph()
    inserts.add(
        (
            URIRef(f"{DEFAULT_IRI}Acme"),
            URIRef(f"{EXT_IRI}/city"),
            URIRef(f"{DEFAULT_IRI}SantaClara"),
        )
    )
    buckets, unattributed = partition_triples_by_namespace(
        inserts,
        writable_iris=[head.iri],
        ontology_manager=_manager(head),
        current_domain=DOMAIN,
    )
    assert buckets == {}
    assert unattributed == 1


HOME_NS = f"{DOMAIN}/sports#"


def test_a_bound_prefix_never_takes_a_namespace_from_its_owner() -> None:
    """An extension that binds another owned ontology's prefix (it refers to
    its terms) used to claim that namespace when it came first, and statements
    about the other ontology's terms were written into the extension."""
    sports = stamped_ontology(
        _classes(f"{DOMAIN}/sports#", "Venue"),
        ontology_id="sports",
        iri=f"{DOMAIN}/sports",
        title="Sports",
    )
    ext_graph = _classes(f"{DOMAIN}/ext-ext#", "Club")
    ext_graph.bind("sp", f"{DOMAIN}/sports#")
    ext_graph.add(
        (
            URIRef(f"{DOMAIN}/ext-ext#Club"),
            RDFS.subClassOf,
            URIRef(f"{DOMAIN}/sports#Venue"),
        )
    )
    extension = stamped_ontology(
        ext_graph, ontology_id="ext-ext", iri=f"{DOMAIN}/ext-ext", title="Extension"
    )
    inserts = RDFGraph()
    venue = URIRef(f"{DOMAIN}/sports#Venue")
    inserts.add((venue, RDFS.comment, Literal("Where a match is played.")))

    buckets, unattributed = partition_triples_by_namespace(
        inserts,
        writable_iris=[extension.iri, sports.iri],
        ontology_manager=_manager(extension, sports),
        current_domain=DOMAIN,
    )

    assert unattributed == 0
    assert set(buckets) == {sports.iri}


def test_facts_individuals_and_undeclared_subjects_are_dropped() -> None:
    graph = RDFGraph()
    league = URIRef(f"{HOME_NS}League")
    graph.add((league, RDF.type, OWL.Class))
    graph.add((league, RDFS.label, Literal("league")))
    weekly = URIRef(f"{HOME_NS}Weekly")
    graph.add((weekly, RDF.type, URIRef("http://www.w3.org/2004/02/skos/core#Concept")))
    implicit = URIRef(f"{HOME_NS}Cup")
    graph.add((implicit, RDFS.subClassOf, league))
    catalog_term = URIRef(f"{DOMAIN}/sports#Team")
    graph.add((catalog_term, RDFS.comment, Literal("A team.")))
    fact = URIRef(f"{DEFAULT_IRI}Acme")
    graph.add((fact, URIRef(f"{EXT_IRI}/city"), URIRef(f"{DEFAULT_IRI}SantaClara")))
    graph.add((fact, RDF.type, OWL.Class))
    person = URIRef(f"{HOME_NS}TerryCampese")
    graph.add((person, RDF.type, URIRef("http://xmlns.com/foaf/0.1/Person")))
    match = URIRef(f"{HOME_NS}2022Match")
    graph.add((match, URIRef("https://schema.org/competitor"), league))

    dropped = drop_non_term_subjects(
        graph, declared={str(catalog_term)}, facts_namespace=DEFAULT_IRI
    )

    assert dropped == 4
    assert {s for s in graph.subjects()} == {league, weekly, implicit, catalog_term}


def test_a_term_minted_outside_the_home_goes_with_every_mention() -> None:
    graph = RDFGraph()
    home_term = URIRef(f"{HOME_NS}Edition")
    stray = URIRef(f"{DOMAIN}/ext-ext#Edition")
    catalog_term = URIRef(f"{DOMAIN}/ext-ext#Venue")
    for term in (home_term, stray, catalog_term):
        graph.add((term, RDF.type, OWL.Class))
    graph.add((home_term, RDFS.subClassOf, stray))
    graph.add((stray, RDFS.label, Literal("edition")))

    dropped = drop_terms_minted_outside_home(
        graph,
        home_namespace=HOME_NS,
        catalog_terms={str(catalog_term)},
        current_domain=DOMAIN,
    )

    assert dropped == 3
    assert set(graph.subjects()) == {home_term, catalog_term}
    assert (home_term, RDFS.subClassOf, stray) not in graph


def test_an_untyped_term_outside_the_home_is_a_stray_but_a_reuse_is_not() -> None:
    """A term declared by an axiom alone is still declared: outside the home it
    is a stray. An axiom pointing at a well-known term is a reuse."""
    graph = RDFGraph()
    home_term = URIRef(f"{HOME_NS}Edition")
    graph.add((home_term, RDF.type, OWL.Class))
    graph.add((home_term, RDFS.subClassOf, URIRef("https://schema.org/Event")))
    stray = URIRef(f"{DOMAIN}/other#Season")
    graph.add((stray, RDFS.subClassOf, home_term))
    graph.add((home_term, RDFS.seeAlso, stray))

    dropped = drop_terms_minted_outside_home(
        graph, home_namespace=HOME_NS, catalog_terms=set(), current_domain=DOMAIN
    )

    assert dropped == 2
    assert set(graph) == {
        (home_term, RDF.type, OWL.Class),
        (home_term, RDFS.subClassOf, URIRef("https://schema.org/Event")),
    }


def test_an_external_parent_the_render_declares_stays_a_parent() -> None:
    """A render that echoes the external parent it reuses (its type, its own
    parent) declares a term it may not write; the statements about it go, the
    home's axiom pointing at it stays, as when the parent is not declared."""
    graph = RDFGraph()
    home_term = URIRef(f"{HOME_NS}Perovskite")
    graph.add((home_term, RDF.type, OWL.Class))
    parent = URIRef("http://purl.test/obo/CHEBI_1")
    graph.add((home_term, RDFS.subClassOf, parent))
    graph.add((parent, RDFS.subClassOf, URIRef("http://purl.test/obo/CHEBI_2")))
    typed = URIRef("http://purl.test/obo/CHEBI_3")
    graph.add((home_term, RDFS.subClassOf, typed))
    graph.add((typed, RDF.type, OWL.Class))
    _restriction(graph, typed, URIRef(f"{HOME_NS}uses"), home_term)

    dropped = drop_terms_minted_outside_home(
        graph, home_namespace=HOME_NS, catalog_terms=set(), current_domain=DOMAIN
    )

    assert dropped == 6
    assert set(graph) == {
        (home_term, RDF.type, OWL.Class),
        (home_term, RDFS.subClassOf, parent),
        (home_term, RDFS.subClassOf, typed),
    }


# --- blank nodes travel as units -------------------------------------------


def _home_head():
    return stamped_ontology(
        _classes(HOME_NS, "Team"),
        ontology_id="sports",
        iri=f"{DOMAIN}/sports",
        title="Sports",
    )


def _restriction(graph: RDFGraph, owner: URIRef, prop: URIRef, filler: URIRef):
    restriction = BNode()
    graph.add((owner, RDFS.subClassOf, restriction))
    graph.add((restriction, RDF.type, OWL.Restriction))
    graph.add((restriction, OWL.onProperty, prop))
    graph.add((restriction, OWL.someValuesFrom, filler))
    return restriction


def test_a_restriction_is_filed_whole_with_the_class_that_uses_it() -> None:
    """Per-triple filing dropped the restriction's type (it names only OWL) and
    sent ``owl:onProperty ext:p`` to the external ontology."""
    home, external = _home_head(), _external_head()
    inserts = RDFGraph()
    team = URIRef(f"{HOME_NS}Team")
    _restriction(
        inserts, team, URIRef(f"{EXT_IRI}/league"), URIRef(f"{EXT_IRI}/League")
    )

    buckets, unattributed = partition_triples_by_namespace(
        inserts,
        writable_iris=[external.iri, home.iri],
        ontology_manager=_manager(external, home),
        current_domain=DOMAIN,
    )

    assert unattributed == 0
    assert set(buckets) == {home.iri}
    assert set(buckets[home.iri]) == set(inserts)


def test_a_list_is_filed_whole() -> None:
    home = _home_head()
    inserts = RDFGraph()
    team = URIRef(f"{HOME_NS}Team")
    members: list[Node] = [URIRef(f"{HOME_NS}Club"), URIRef(f"{HOME_NS}Squad")]
    union = BNode()
    inserts.add((team, OWL.equivalentClass, union))
    head = BNode()
    inserts.add((union, OWL.unionOf, head))
    Collection(inserts, head, members)

    buckets, unattributed = partition_triples_by_namespace(
        inserts,
        writable_iris=[home.iri],
        ontology_manager=_manager(home),
        current_domain=DOMAIN,
    )

    assert unattributed == 0
    assert set(buckets[home.iri]) == set(inserts)


def test_an_unanchored_axiom_goes_to_the_single_owner_of_its_terms() -> None:
    home, other = (
        _home_head(),
        stamped_ontology(
            _classes(f"{DOMAIN}/venues#", "Venue"),
            ontology_id="venues",
            iri=f"{DOMAIN}/venues",
            title="Venues",
        ),
    )
    manager = _manager(home, other)
    disjoint = RDFGraph()
    axiom, head = BNode(), BNode()
    disjoint.add((axiom, RDF.type, OWL.AllDisjointClasses))
    disjoint.add((axiom, OWL.members, head))
    Collection(disjoint, head, [URIRef(f"{HOME_NS}Club"), URIRef(f"{HOME_NS}Squad")])

    buckets, unattributed = partition_triples_by_namespace(
        disjoint,
        writable_iris=[home.iri, other.iri],
        ontology_manager=manager,
        current_domain=DOMAIN,
    )
    assert unattributed == 0
    assert set(buckets[home.iri]) == set(disjoint)

    mixed = RDFGraph()
    axiom, head = BNode(), BNode()
    mixed.add((axiom, RDF.type, OWL.AllDisjointClasses))
    mixed.add((axiom, OWL.members, head))
    Collection(
        mixed, head, [URIRef(f"{HOME_NS}Club"), URIRef(f"{DOMAIN}/venues#Venue")]
    )

    buckets, unattributed = partition_triples_by_namespace(
        mixed,
        writable_iris=[home.iri, other.iri],
        ontology_manager=manager,
        current_domain=DOMAIN,
    )
    assert buckets == {}
    assert unattributed == len(mixed)


def test_nothing_is_filed_into_an_external_ontology() -> None:
    """An external ontology is writable only to give its namespace an owner;
    an axiom over its terms alone is unattributed, never written into it."""
    home, external = _home_head(), _external_head()
    inserts = RDFGraph()
    axiom, head = BNode(), BNode()
    inserts.add((axiom, RDF.type, OWL.AllDisjointClasses))
    inserts.add((axiom, OWL.members, head))
    Collection(
        inserts, head, [URIRef(f"{EXT_IRI}/Company"), URIRef(f"{EXT_IRI}/League")]
    )
    inserts.add((URIRef(f"{EXT_IRI}/League"), RDFS.comment, Literal("A league.")))

    buckets, unattributed = partition_triples_by_namespace(
        inserts,
        writable_iris=[external.iri, home.iri],
        ontology_manager=_manager(external, home),
        current_domain=DOMAIN,
    )

    assert buckets == {}
    assert unattributed == len(inserts)


def test_an_annotated_axiom_goes_with_its_source() -> None:
    """An ``owl:Axiom`` names its source and its target; its owner is the
    source's, as for the axiom it annotates."""
    home = _home_head()
    venues = stamped_ontology(
        _classes(f"{DOMAIN}/venues#", "Venue"),
        ontology_id="venues",
        iri=f"{DOMAIN}/venues",
        title="Venues",
    )
    stadium, venue = URIRef(f"{HOME_NS}Stadium"), URIRef(f"{DOMAIN}/venues#Venue")
    inserts = RDFGraph()
    inserts.add((stadium, RDFS.subClassOf, venue))
    annotation = BNode()
    inserts.add((annotation, RDF.type, OWL.Axiom))
    inserts.add((annotation, OWL.annotatedSource, stadium))
    inserts.add((annotation, OWL.annotatedProperty, RDFS.subClassOf))
    inserts.add((annotation, OWL.annotatedTarget, venue))
    inserts.add((annotation, RDFS.comment, Literal("Every stadium is a venue.")))

    buckets, unattributed = partition_triples_by_namespace(
        inserts,
        writable_iris=[home.iri, venues.iri],
        ontology_manager=_manager(home, venues),
        current_domain=DOMAIN,
    )

    assert unattributed == 0
    assert set(buckets) == {home.iri}
    assert set(buckets[home.iri]) == set(inserts)


def test_a_dropped_subject_takes_its_restriction_with_it() -> None:
    graph = RDFGraph()
    external_class = URIRef(f"{EXT_IRI}/SportsTeam")
    _restriction(
        graph, external_class, URIRef(f"{HOME_NS}plays"), URIRef(f"{HOME_NS}Sport")
    )
    kept = URIRef(f"{HOME_NS}Team")
    _restriction(graph, kept, URIRef(f"{HOME_NS}plays"), URIRef(f"{HOME_NS}Sport"))
    before = len(graph)

    assert drop_external_subjects(graph, [EXT_IRI]) == 4
    assert len(graph) == before - 4
    assert all(s != external_class for s in graph.subjects())
    # The kept class's restriction is whole.
    (restriction,) = graph.objects(kept, RDFS.subClassOf)
    assert (restriction, RDF.type, OWL.Restriction) in graph


def test_a_restriction_naming_a_stray_goes_whole() -> None:
    """Removing only ``owl:someValuesFrom stray`` would leave a restriction that
    constrains nothing."""
    graph = RDFGraph()
    team = URIRef(f"{HOME_NS}Team")
    graph.add((team, RDF.type, OWL.Class))
    stray = URIRef(f"{DOMAIN}/other#Sport")
    graph.add((stray, RDF.type, OWL.Class))
    _restriction(graph, team, URIRef(f"{HOME_NS}plays"), stray)

    drop_terms_minted_outside_home(
        graph, home_namespace=HOME_NS, catalog_terms=set(), current_domain=DOMAIN
    )

    assert set(graph) == {(team, RDF.type, OWL.Class)}


def test_a_fresh_render_is_filed_by_subject() -> None:
    """A fresh ontology's statement about another ontology's term belongs to
    that ontology; one about a well-known term belongs to no writable one."""
    venues = stamped_ontology(
        _classes(f"{DOMAIN}/venues#", "Venue"),
        ontology_id="venues",
        iri=f"{DOMAIN}/venues",
        title="Venues",
    )
    home = OntologyIdentity(ontology_id="sports", iri=f"{DOMAIN}/sports", title="S")
    graph = _classes(HOME_NS, "Stadium")
    stadium = URIRef(f"{HOME_NS}Stadium")
    venue = URIRef(f"{DOMAIN}/venues#Venue")
    graph.add((venue, RDFS.comment, Literal("Where a match is played.")))
    graph.add((URIRef("https://schema.org/Place"), RDFS.subClassOf, stadium))
    graph.add((stadium, RDFS.subClassOf, venue))
    fresh = mint_ontology(home, graph)

    moved, unattributed = refile_fresh_statements(
        fresh,
        writable_iris=[venues.iri],
        ontology_manager=_manager(venues),
        current_domain=DOMAIN,
    )

    assert set(moved) == {(venue, RDFS.comment, Literal("Where a match is played."))}
    assert unattributed == 1
    subjects = set(fresh.graph.subjects())
    assert venue not in subjects
    assert (stadium, RDFS.subClassOf, venue) in fresh.graph


def test_the_critic_is_told_the_home_namespace() -> None:
    """Without it the critic reads the home prefix as undeclared and asks for a
    namespace repair, which re-mints the terms elsewhere."""
    critic_module = importlib.import_module("ontocast.agent.criticise_ontology")
    home = OntologyIdentity(
        ontology_id="ext-ext", iri=f"{DOMAIN}/ext-ext", title="Extension"
    )
    unit = ContentUnit(text="Teams.", index=0, doc_iri=URIRef(f"{DOMAIN}/doc/d1"))
    state = UnitOntologyState(content_unit=unit, home_ontology=home)
    assert f"<{home.namespace}>" in critic_module.critic_intro(state)
    bare = UnitOntologyState(content_unit=unit)
    assert critic_module.critic_intro(bare) == critic_module.intro_instruction


def test_a_written_ontology_is_dumped_as_its_stored_version(tmp_path) -> None:
    """The reduced artifact precedes stamping: a new ontology has no header."""
    from ontocast.api.process_helpers import dump_ontology_ttls
    from ontocast.onto.state import OntologyVersionRecord

    home = OntologyIdentity(ontology_id="sports", iri=f"{DOMAIN}/sports", title="S")
    reduced = mint_ontology(home, _classes(HOME_NS, "League"))
    stored = stamped_ontology(
        _classes(HOME_NS, "League"), ontology_id="sports", iri=home.iri, title="S"
    )
    assert stored.version_iri is not None
    state = AgentState(current_domain=DOMAIN)
    state.reduced_ontology_artifacts = [reduced]
    state.ontology_versions_written = [
        OntologyVersionRecord(
            iri=stored.iri,
            version_iri=stored.version_iri,
            ontology_id="sports",
            version=stored.version or "",
            role="created",
        )
    ]
    src = tmp_path / "doc.txt"
    src.write_text("x")
    (path,) = dump_ontology_ttls(
        state, src, output_dir=tmp_path, ontology_manager=_manager(stored)
    )
    dumped = RDFGraph._from_turtle_str(path.read_text())
    assert (URIRef(home.iri), RDF.type, OWL.Ontology) in dumped
    assert (URIRef(home.iri), OWL.versionIRI, URIRef(stored.version_iri)) in dumped


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


def _extension_catalog() -> tuple[ToolBox, Ontology, Ontology]:
    """A toolbox holding an external ontology and an owned one, ``sports``."""
    tools = ToolBox(_config())
    external = _external_head()
    tools.ontology_manager.add_ontology(external, skip_vector_index=True)
    sports = stamped_ontology(
        _classes(f"{DOMAIN}/sports#", "Venue"),
        ontology_id="sports",
        iri=f"{DOMAIN}/sports",
        title="Sports",
    )
    tools.ontology_manager.add_ontology(sports, skip_vector_index=True)
    return tools, external, sports


def _script_extension_run(
    monkeypatch, external: Ontology, sports: Ontology, *, serialize: bool
) -> None:
    """Script every model call of a document whose home is ``external``'s extension.

    Each unit mints a class named after its text under the home, with a
    restriction on an external property; it also re-declares an external
    term, states a disjointness over external terms alone, writes a stray
    example.com axiom, a fact using an external property, an individual
    under the home namespace, and a copy of its class under
    ``sports``. With ``serialize`` false the serialize node is a passthrough.
    """
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
            writable_iris=[external.iri, sports.iri],
            confidence=1.0,
        )

    async def update_render(state: UnitOntologyState, tools, **kwargs):
        home = state.home_ontology
        assert home is not None
        name = "".join(word.title() for word in state.content_unit.text.split())
        inserts = RDFGraph()
        term = URIRef(f"{home.namespace}{name}")
        inserts.add((term, RDF.type, OWL.Class))
        inserts.add((term, RDFS.label, Literal(name)))
        inserts.add((term, RDFS.subClassOf, URIRef(f"{EXT_IRI}/Organisation")))
        _restriction(
            inserts, term, URIRef(f"{EXT_IRI}/city"), URIRef(f"{EXT_IRI}/League")
        )
        inserts.add(
            (URIRef(f"{EXT_IRI}/SportsTeam"), RDFS.subClassOf, URIRef(f"{EXT_IRI}/X"))
        )
        disjoint, members = BNode(), BNode()
        inserts.add((disjoint, RDF.type, OWL.AllDisjointClasses))
        inserts.add((disjoint, OWL.members, members))
        Collection(
            inserts,
            members,
            [URIRef(f"{EXT_IRI}/Company"), URIRef(f"{EXT_IRI}/League")],
        )
        inserts.add(
            (
                URIRef("https://example.com/cycling-sport#Race"),
                RDFS.subClassOf,
                URIRef(f"{EXT_IRI}/League"),
            )
        )
        inserts.add(
            (
                URIRef(f"{DEFAULT_IRI}Club{state.content_unit.index}"),
                URIRef(f"{EXT_IRI}/city"),
                URIRef(f"{DEFAULT_IRI}Leeds"),
            )
        )
        inserts.add(
            (URIRef(f"{home.namespace}Leeds"), RDF.type, URIRef(f"{EXT_IRI}/League"))
        )
        copy = URIRef(f"{DOMAIN}/sports#{name}")
        inserts.add((copy, RDF.type, OWL.Class))
        inserts.add((copy, RDFS.label, Literal(name)))
        report = GraphUpdateRenderReport(insert_graph=inserts)
        state.ontology_updates.append(report.to_graph_update())
        state.update_ontology()
        state.writable_iris = [external.iri, sports.iri]
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
    if not serialize:
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


def _document(filename: str, text: str) -> AgentState:
    return AgentState(
        input_document=InputDocument(filename=filename, data=text.encode("utf-8")),
        render_mode=RenderMode.ONTOLOGY,
        current_domain=DOMAIN,
    )


def test_update_renders_land_in_the_extension_and_never_in_the_external(
    monkeypatch,
) -> None:
    """Units render updates against the external ontology's patch; only the
    home terms are written, into a new extension ontology that imports the
    external one, each with its restriction whole."""
    tools, external, sports = _extension_catalog()
    _script_extension_run(monkeypatch, external, sports, serialize=False)

    document = _document("doc.txt", "teams one\n\nteams two")
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
    assert declared == {f"{home.namespace}TeamsOne", f"{home.namespace}TeamsTwo"}
    subjects = {str(s) for s in artifact.graph.subjects() if isinstance(s, URIRef)}
    assert not any(s.startswith(EXT_IRI) for s in subjects)
    assert not any("example.com/cycling-sport" in s for s in subjects)
    assert not any(s.startswith(DEFAULT_IRI) for s in subjects)
    assert f"{home.namespace}Leeds" not in subjects
    restrictions = set(artifact.graph.subjects(RDF.type, OWL.Restriction))
    assert len(restrictions) == 2
    for restriction in restrictions:
        assert (
            restriction,
            OWL.onProperty,
            URIRef(f"{EXT_IRI}/city"),
        ) in artifact.graph
    metrics = final["ontology_reduce_metrics"]
    assert metrics["external_subject_refused"] >= 1
    assert metrics["example_namespace_dropped"] >= 1
    assert metrics["non_term_subjects_dropped"] >= 2
    assert metrics["minted_outside_home_dropped"] >= 2
    # Each unit's disjointness axiom over external terms has no owned owner.
    assert metrics["unattributed_insert_triples"] == 2 * 6


def test_stored_versions_name_their_document_and_leave_the_external_alone(
    monkeypatch,
) -> None:
    """Two documents through the whole graph, serialization included.

    The first creates the extension, the second updates it. Each stored
    version names its document and the second points at the first; the
    external ontology keeps the one version it was loaded with.
    """
    tools, external, sports = _extension_catalog()
    _script_extension_run(monkeypatch, external, sports, serialize=True)
    graph = create_agent_graph(tools)

    first_doc = _document("one.txt", "teams one\n\nteams two")
    second_doc = _document("two.txt", "clubs three")
    first = asyncio.run(graph.ainvoke(first_doc))
    second = asyncio.run(graph.ainvoke(second_doc))

    manager = tools.ontology_manager
    external_head = manager.get_head_by_iri(EXT_IRI)
    assert external_head is not None
    assert external_head.version_iri == external.version_iri
    home_iri = first["home_ontology"].iri
    assert second["home_ontology"].iri == home_iri

    (created,) = first["ontology_versions_written"]
    (updated,) = second["ontology_versions_written"]
    assert (created.iri, created.role) == (home_iri, "created")
    assert (updated.iri, updated.role) == (home_iri, "updated")
    assert created.version_iri in updated.prior_version_iris

    head = manager.get_head_by_iri(home_iri)
    assert head is not None
    assert head.version_iri == updated.version_iri
    header = URIRef(home_iri)
    assert (header, PROV.wasDerivedFrom, second_doc.doc_iri) in head.graph
    assert (header, OWL.priorVersion, URIRef(created.version_iri)) in head.graph
    for restriction in head.graph.subjects(RDF.type, OWL.Restriction):
        assert (restriction, OWL.onProperty, URIRef(f"{EXT_IRI}/city")) in head.graph


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


def test_a_structure_whose_anchors_disagree_goes_with_its_edges() -> None:
    """Two owners point at one blank node: neither may keep an edge to a
    node the other would hold, so the structure and its edges go together."""
    home = _home_head()
    other = stamped_ontology(
        _classes(f"{DOMAIN}/venues#", "Venue"),
        ontology_id="venues",
        iri=f"{DOMAIN}/venues",
        title="Venues",
    )
    inserts = RDFGraph()
    restriction = _restriction(
        inserts,
        URIRef(f"{HOME_NS}Team"),
        URIRef(f"{HOME_NS}playsAt"),
        URIRef(f"{DOMAIN}/venues#Venue"),
    )
    inserts.add((URIRef(f"{DOMAIN}/venues#Venue"), RDFS.subClassOf, restriction))

    buckets, unattributed = partition_triples_by_namespace(
        inserts,
        writable_iris=[home.iri, other.iri],
        ontology_manager=_manager(home, other),
        current_domain=DOMAIN,
    )

    assert buckets == {}
    assert unattributed == len(inserts)
