"""The document ontology plan: the model-free check and the candidate split."""

import pytest
from rdflib import OWL, RDF, RDFS, Literal, URIRef

from ontocast.onto.identity import OntologyIdentity
from ontocast.onto.model import OntologyPlanReport, OntologyTermProposal
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.ontology_validation.plan import (
    check_plan,
    compact_iri,
    expand_reference,
    minted_terms,
    plan_candidates,
)
from ontocast.tool.ontology_validation.shape import build_catalog_surface_index

pytestmark = pytest.mark.unit

CAT = "https://example.com/cat#"
HOME = OntologyIdentity(
    ontology_id="home",
    iri="https://example.com/home",
    title="Home",
)
PREFIXES = {"cat": CAT, "home": HOME.namespace, "schema": "https://schema.org/"}


def _catalog():
    graph = RDFGraph()
    sample = URIRef(f"{CAT}Sample")
    graph.add((sample, RDF.type, OWL.Class))
    graph.add((sample, RDFS.label, Literal("Sample")))
    has_part = URIRef(f"{CAT}hasPart")
    graph.add((has_part, RDF.type, OWL.ObjectProperty))
    graph.add((has_part, RDFS.label, Literal("has part")))
    return build_catalog_surface_index([graph])


def _proposal(**kwargs) -> OntologyTermProposal:
    base = {"kind": "class", "label": "Thing", "local_name": "Thing"}
    return OntologyTermProposal.model_validate({**base, **kwargs})


def test_proposal_wire_is_lenient_about_kind_and_local_name():
    proposal = OntologyTermProposal.model_validate(
        {"kind": "Object Property", "label": "has film", "local_name": "has film"}
    )
    assert proposal.kind == "object_property"
    assert proposal.local_name == "hasFilm"


def test_a_proposal_naming_a_catalog_term_becomes_a_reuse():
    report = OntologyPlanReport(
        proposals=[
            _proposal(label="sample", local_name="Sample"),
            _proposal(label="Thin film", local_name="ThinFilm", reuse="cat:Nope"),
        ]
    )
    plan = check_plan(
        report, home=HOME, catalog_index=_catalog(), prefixes=PREFIXES, max_terms=10
    )
    assert [(t.decision, t.iri) for t in plan.terms] == [
        ("reuse", f"{CAT}Sample"),
        ("mint", f"{HOME.namespace}ThinFilm"),
    ]
    assert plan.resolved_to_catalog == 1


def test_an_explicit_reuse_must_name_a_known_term_of_compatible_role():
    report = OntologyPlanReport(
        proposals=[
            _proposal(
                kind="object_property",
                label="contains",
                local_name="contains",
                reuse="cat:hasPart",
            ),
            _proposal(label="Part", local_name="Part", reuse="cat:hasPart"),
        ]
    )
    plan = check_plan(
        report, home=HOME, catalog_index=_catalog(), prefixes=PREFIXES, max_terms=10
    )
    assert plan.terms[0].decision == "reuse"
    assert plan.terms[0].iri == f"{CAT}hasPart"
    # A class cannot reuse a property: it is minted instead.
    assert plan.terms[1].decision == "mint"


def test_duplicates_collapse_and_the_cap_holds():
    report = OntologyPlanReport(
        proposals=[
            _proposal(label="Film", local_name="Film"),
            _proposal(label="film", local_name="FilmLayer"),
            _proposal(label="Layer", local_name="Layer"),
            _proposal(label="Device", local_name="Device"),
        ]
    )
    plan = check_plan(
        report, home=HOME, catalog_index=_catalog(), prefixes=PREFIXES, max_terms=2
    )
    assert [t.local_name for t in plan.terms] == ["Film", "Layer"]
    assert plan.rejected == [
        ("FilmLayer", "duplicate in plan"),
        ("Device", "plan size cap"),
    ]


def test_parents_resolve_to_catalog_well_known_or_plan_terms():
    report = OntologyPlanReport(
        proposals=[
            _proposal(label="Thin film", local_name="ThinFilm", parent="cat:Sample"),
            _proposal(label="Device", local_name="Device", parent="schema:Product"),
            _proposal(label="Solar cell", local_name="SolarCell", parent="Device"),
            _proposal(label="Widget", local_name="Widget", parent="cat:Unknown"),
        ]
    )
    plan = check_plan(
        report, home=HOME, catalog_index=_catalog(), prefixes=PREFIXES, max_terms=10
    )
    parents = {t.local_name: t.parent_iri for t in plan.terms}
    assert parents == {
        "ThinFilm": f"{CAT}Sample",
        "Device": "https://schema.org/Product",
        "SolarCell": f"{HOME.namespace}Device",
        "Widget": None,
    }


def test_curies_expand_and_compact():
    assert expand_reference("cat:Sample", PREFIXES) == f"{CAT}Sample"
    assert expand_reference("<https://x.org/a>", PREFIXES) == "https://x.org/a"
    assert expand_reference("Sample", PREFIXES) is None
    assert compact_iri(f"{CAT}Sample", PREFIXES) == "cat:Sample"
    assert compact_iri("https://other.org/x", PREFIXES) == "https://other.org/x"


def _minted_graph(*locals_: str, namespace: str = HOME.namespace) -> RDFGraph:
    graph = RDFGraph()
    for local in locals_:
        term = URIRef(f"{namespace}{local}")
        graph.add((term, RDF.type, OWL.Class))
        graph.add((term, RDFS.label, Literal(local)))
    return graph


def test_candidates_are_minted_terms_the_plan_does_not_name():
    report = OntologyPlanReport(
        proposals=[
            _proposal(label="Thin film", local_name="ThinFilm"),
            _proposal(label="Quality factor", local_name="QualityFactor"),
        ]
    )
    catalog = _catalog()
    plan = check_plan(
        report, home=HOME, catalog_index=catalog, prefixes=PREFIXES, max_terms=10
    )
    # Matched by IRI, by local name in another namespace, by label.
    graph = _minted_graph("ThinFilm", "Superradiance")
    graph += _minted_graph("thinfilm", namespace="https://example.com/other#")
    labelled = URIRef(f"{HOME.namespace}Q")
    graph.add((labelled, RDF.type, OWL.Class))
    graph.add((labelled, RDFS.label, Literal("quality factor")))
    # Catalog terms are never minted.
    graph.add((URIRef(f"{CAT}Sample"), RDFS.subClassOf, OWL.Thing))
    assert set(minted_terms([graph], catalog)) == {
        f"{HOME.namespace}ThinFilm",
        f"{HOME.namespace}Superradiance",
        "https://example.com/other#thinfilm",
        str(labelled),
    }
    assert plan_candidates([graph], plan, catalog) == {f"{HOME.namespace}Superradiance"}
    assert plan_candidates([graph], None, catalog) == set()


def test_a_well_known_term_of_the_other_kind_is_neither_a_reuse_nor_a_parent():
    """``schema:Property`` is a class: a property never reuses or extends it."""
    report = OntologyPlanReport(
        proposals=[
            _proposal(
                kind="object_property",
                label="has season",
                local_name="hasSeason",
                parent="schema:Property",
            ),
            _proposal(
                kind="object_property",
                label="sub event",
                local_name="subEvent",
                reuse="schema:subEvent",
            ),
            _proposal(
                kind="object_property",
                label="person",
                local_name="person",
                reuse="schema:Person",
            ),
        ]
    )
    plan = check_plan(
        report, home=HOME, catalog_index=_catalog(), prefixes=PREFIXES, max_terms=10
    )
    season, sub_event, person = plan.terms
    assert season.decision == "mint" and season.parent_iri is None
    assert (sub_event.decision, sub_event.iri) == (
        "reuse",
        "https://schema.org/subEvent",
    )
    assert person.decision == "mint"
