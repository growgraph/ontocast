"""Tests for ontocast.onto.ontology_access."""

from rdflib import Namespace, URIRef

from ontocast.onto.constants import ONTOLOGY_NULL_IRI, WELL_KNOWN_PREFIXES
from ontocast.onto.content_unit import ContentUnit, SourceUnit
from ontocast.onto.null import NULL_ONTOLOGY
from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_access import (
    DocumentOntologyAccess,
    UnitFactsOntologyAccess,
    UnitOntologyAccess,
    build_llm_prefix_map,
    document_ontology_access,
    ontology_access_for_unit_facts,
    ontology_access_for_unit_ontology,
)
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.onto.unit_states import UnitFactsState, UnitOntologyState


def _source_unit() -> SourceUnit:
    return SourceUnit(
        text="x",
        index=0,
        doc_iri=URIRef("https://example.org/doc"),
    )


def _real_ontology(iri: str = "https://example.org/o") -> Ontology:
    g = RDFGraph()
    g.add(
        (
            URIRef(iri),
            URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            URIRef("http://www.w3.org/2002/07/owl#Ontology"),
        )
    )
    return Ontology(graph=g, iri=iri)


def test_document_serialization_targets_prefers_artifacts() -> None:
    a1 = _real_ontology("https://example.org/a1")
    a2 = _real_ontology("https://example.org/a2")
    state = AgentState(ontology_artifacts=[a1, a2])
    access = document_ontology_access(state)
    targets = access.serialization_targets()
    assert targets == [a1, a2]
    assert access.ontology_by_anchor(a2.iri) is a2


def test_document_serialization_targets_returns_empty_without_artifacts() -> None:
    state = AgentState()
    access = DocumentOntologyAccess(state)
    assert access.serialization_targets() == []


def test_document_serialization_targets_with_single_reduced_artifact() -> None:
    primary = _real_ontology("https://example.org/reduced")
    state = AgentState(reduced_ontology_artifacts=[primary], ontology_artifacts=[])
    access = DocumentOntologyAccess(state)
    assert access.serialization_targets() == [primary]


def test_document_no_artifacts_flags() -> None:
    state = AgentState()
    access = document_ontology_access(state)
    assert not access.has_any_artifacts()
    assert not access.has_non_null_artifacts()


def test_document_artifact_presence_helpers() -> None:
    state = AgentState(ontology_artifacts=[_real_ontology("https://example.org/a1")])
    access = document_ontology_access(state)
    assert access.has_any_artifacts()
    assert access.has_non_null_artifacts()


def test_agent_state_ontology_ids_exposes_all_artifact_ids() -> None:
    a1 = _real_ontology("https://example.org/a1")
    a2 = _real_ontology("https://example.org/a2")
    state = AgentState(ontology_artifacts=[a1, a2])
    assert len(state.ontology_ids) == 2


def test_unit_ontology_access_seed_and_effective() -> None:
    snap = NULL_ONTOLOGY
    state = UnitOntologyState(
        content_unit=_source_unit(),
        ontology_snapshot=snap,
    )
    access = ontology_access_for_unit_ontology(state)
    assert not access.has_non_null_seed_snapshot()
    assert access.effective_ontology_for_prompt() is state.current_ontology
    assert access.ontology_for_prefixes() is access.effective_ontology_for_prompt()

    other = _real_ontology()
    state.current_ontology = other
    access2 = UnitOntologyAccess(state)
    assert access2.effective_ontology_for_prompt() is other


def test_unit_facts_access_uses_snapshot_only() -> None:
    snap = _real_ontology("https://example.org/ctx")
    unit = ContentUnit(
        text="t",
        index=0,
        doc_iri=URIRef("https://example.org/doc"),
    )
    state = UnitFactsState(content_unit=unit, ontology_snapshot=snap)
    access = ontology_access_for_unit_facts(state)
    assert access.effective_ontology_for_prompt() is snap
    assert access.ontology_for_prefixes() is snap
    assert access.has_non_null_seed_snapshot()


def test_build_llm_prefix_map_merges_supplemental_ontology() -> None:
    primary = _real_ontology("https://example.org/primary")
    supplemental = _real_ontology("https://example.org/supp")
    supplemental.graph.bind("custom", Namespace("https://example.org/custom#"))
    supplemental.graph.add(
        (
            URIRef("https://example.org/custom#Thing"),
            URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            URIRef("http://www.w3.org/2002/07/owl#Class"),
        )
    )

    merged = build_llm_prefix_map(primary, [supplemental])

    assert merged["qudt"] == WELL_KNOWN_PREFIXES["qudt"]
    assert merged["custom"] == "https://example.org/custom#"


def test_unit_facts_access_null_snapshot() -> None:
    unit = ContentUnit(
        text="t",
        index=0,
        doc_iri=URIRef("https://example.org/doc"),
    )
    state = UnitFactsState(
        content_unit=unit,
        ontology_snapshot=Ontology(iri=ONTOLOGY_NULL_IRI),
    )
    access = UnitFactsOntologyAccess(state)
    assert not access.has_non_null_seed_snapshot()
