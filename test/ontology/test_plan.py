"""The document ontology plan: the model-free check and the candidate split."""

import pytest
from rdflib import OWL, RDF, RDFS, XSD, Literal, URIRef

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


def test_a_name_the_well_known_vocabulary_lacks_is_minted_not_reused():
    """``schema:leader`` is no schema.org term: reusing it would serve an unknown term."""
    report = OntologyPlanReport(
        proposals=[
            _proposal(
                kind="object_property",
                label="leader",
                local_name="leader",
                reuse="schema:leader",
            ),
            _proposal(
                kind="object_property",
                label="team",
                local_name="team",
                parent="schema:affiliationOf",
            ),
        ]
    )
    plan = check_plan(
        report, home=HOME, catalog_index=_catalog(), prefixes=PREFIXES, max_terms=10
    )
    leader, team = plan.terms
    assert (leader.decision, leader.iri) == ("mint", f"{HOME.namespace}leader")
    assert team.parent_iri is None


SIDE_PREFIXES = {
    **PREFIXES,
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}


def test_a_planned_property_keeps_a_specific_domain_and_range():
    report = OntologyPlanReport(
        proposals=[
            _proposal(label="Fuel type", local_name="FuelType", parent="skos:Concept"),
            _proposal(
                kind="object_property",
                label="has fuel type",
                local_name="hasFuelType",
                domain="cat:Sample",
                range="FuelType",
            ),
            _proposal(
                kind="datatype_property",
                label="trade name",
                local_name="tradeName",
                domain="cat:Sample",
                range="string",
            ),
        ]
    )
    plan = check_plan(
        report,
        home=HOME,
        catalog_index=_catalog(),
        prefixes=SIDE_PREFIXES,
        max_terms=10,
    )
    by_local = {t.local_name: t for t in plan.terms}
    assert by_local["hasFuelType"].domain_iri == f"{CAT}Sample"
    assert by_local["hasFuelType"].range_iri == f"{HOME.namespace}FuelType"
    assert by_local["tradeName"].domain_iri == f"{CAT}Sample"
    assert by_local["tradeName"].range_iri == "http://www.w3.org/2001/XMLSchema#string"
    assert by_local["FuelType"].domain_iri is None
    assert plan.generic_sides == 0


def test_a_generic_planned_side_is_kept_and_counted_an_unknown_one_dropped():
    report = OntologyPlanReport(
        proposals=[
            _proposal(
                kind="object_property",
                label="has format",
                local_name="hasFormat",
                domain="owl:Thing",
                range="skos:Concept",
            ),
            _proposal(
                kind="object_property",
                label="has origin",
                local_name="hasOrigin",
                domain="cat:Nope",
                range="home:Agent",
            ),
            _proposal(
                kind="datatype_property",
                label="code",
                local_name="code",
                range="MadeUpType",
            ),
        ]
    )
    plan = check_plan(
        report,
        home=HOME,
        catalog_index=_catalog(),
        prefixes=SIDE_PREFIXES,
        max_terms=10,
    )
    by_local = {t.local_name: t for t in plan.terms}
    assert list(by_local) == ["hasFormat", "hasOrigin", "code"]
    assert by_local["hasFormat"].domain_iri == "http://www.w3.org/2002/07/owl#Thing"
    assert (
        by_local["hasFormat"].range_iri == "http://www.w3.org/2004/02/skos/core#Concept"
    )
    assert plan.generic_sides == 2
    assert by_local["hasOrigin"].domain_iri is None
    assert by_local["hasOrigin"].range_iri is None
    assert by_local["code"].range_iri is None


def test_the_plan_sides_replace_the_render_sides_of_a_planned_property():
    """A render saw one unit; the plan's sides were resolved against the catalog."""
    from ontocast.onto.model import OntologyPlan, OntologyPlanTerm
    from ontocast.onto.property_sides import DOMAIN_INCLUDES, RANGE_INCLUDES
    from ontocast.tool.ontology_validation.plan import apply_plan_axioms

    home = HOME.namespace
    plan = OntologyPlan(
        terms=[
            OntologyPlanTerm(
                kind="object_property",
                label="has part",
                local_name="hasComponent",
                iri=f"{home}hasComponent",
                decision="mint",
                domain_iri=f"{CAT}Sample",
                range_iri=f"{home}Component",
            ),
            OntologyPlanTerm(
                kind="class",
                label="component",
                local_name="Component",
                iri=f"{home}Component",
                decision="mint",
            ),
            OntologyPlanTerm(
                kind="datatype_property",
                label="code",
                local_name="code",
                iri=f"{home}code",
                decision="mint",
                range_iri=str(XSD.string),
            ),
        ]
    )
    graph = RDFGraph()
    prop = URIRef(f"{home}hasComponent")
    graph.add((prop, RDF.type, OWL.ObjectProperty))
    graph.add((prop, RDFS.domain, URIRef(f"{CAT}ThinSample")))
    graph.add((prop, RDFS.range, URIRef("https://example.com/made-up#Component")))
    # Declared under another namespace of the deployment: matched by local name.
    code = URIRef("https://example.com/other#code")
    graph.add((code, RDF.type, OWL.DatatypeProperty))
    graph.add((code, RDFS.domain, URIRef(f"{CAT}Sample")))
    minted = {str(prop): "property", str(code): "property"}

    # The planned class is not declared: its side would dangle, so it waits.
    # Class sides are written as expected classes; a datatype range stays rdfs.
    assert apply_plan_axioms([graph], plan, minted).sides == 2
    assert set(graph.objects(prop, RDFS.domain)) == set()
    assert set(graph.objects(prop, DOMAIN_INCLUDES)) == {URIRef(f"{CAT}Sample")}
    assert set(graph.objects(prop, RDFS.range)) == {
        URIRef("https://example.com/made-up#Component")
    }
    assert set(graph.objects(code, RDFS.range)) == {XSD.string}
    # The plan gives no domain for it: the render's stays as written.
    assert set(graph.objects(code, RDFS.domain)) == {URIRef(f"{CAT}Sample")}

    component = URIRef(f"{home}Component")
    graph.add((component, RDF.type, OWL.Class))
    minted[str(component)] = "class"
    assert apply_plan_axioms([graph], plan, minted).sides == 1
    assert set(graph.objects(prop, RDFS.range)) == set()
    assert set(graph.objects(prop, RANGE_INCLUDES)) == {component}
    assert apply_plan_axioms([graph], plan, minted).sides == 0


def test_the_plan_parent_replaces_the_render_parent_once_it_is_declared():
    from ontocast.onto.model import OntologyPlan, OntologyPlanTerm
    from ontocast.tool.ontology_validation.plan import apply_plan_axioms

    home = HOME.namespace
    plan = OntologyPlan(
        terms=[
            OntologyPlanTerm(
                kind="class",
                label="stratovolcano",
                local_name="Stratovolcano",
                iri=f"{home}Stratovolcano",
                decision="mint",
                parent_iri=f"{home}Volcano",
            ),
            OntologyPlanTerm(
                kind="class",
                label="volcano",
                local_name="Volcano",
                iri=f"{home}Volcano",
                decision="mint",
                parent_iri=f"{CAT}Sample",
            ),
        ]
    )
    graph = RDFGraph()
    strato = URIRef(f"{home}Stratovolcano")
    volcano = URIRef(f"{home}Volcano")
    graph.add((strato, RDF.type, OWL.Class))
    graph.add((strato, RDFS.subClassOf, OWL.Thing))
    graph.add((volcano, RDF.type, OWL.Class))
    graph.add((volcano, RDFS.subClassOf, URIRef(f"{CAT}Thing")))
    minted = {str(strato): "class"}

    # The planned parent is minted by the plan but not declared: it waits.
    written = apply_plan_axioms([graph], plan, minted)
    assert written.parents == 0
    assert set(graph.objects(strato, RDFS.subClassOf)) == {OWL.Thing}

    minted[str(volcano)] = "class"
    written = apply_plan_axioms([graph], plan, minted)
    assert written.parents == 2
    assert set(graph.objects(strato, RDFS.subClassOf)) == {volcano}
    assert set(graph.objects(volcano, RDFS.subClassOf)) == {URIRef(f"{CAT}Sample")}
    assert apply_plan_axioms([graph], plan, minted).parents == 0


def test_a_parent_at_the_top_of_a_hierarchy_counts_as_generic():
    from rdflib import SKOS

    from ontocast.tool.ontology_validation.axiom_hygiene import has_generic_parent

    home = HOME.namespace
    graph = RDFGraph()
    graph.add((URIRef(f"{home}Event"), RDFS.subClassOf, OWL.Thing))
    graph.add((URIRef(f"{home}Award"), RDFS.subClassOf, OWL.Thing))
    graph.add((URIRef(f"{home}Award"), RDFS.subClassOf, URIRef(f"{CAT}Sample")))
    graph.add((URIRef(f"{home}relatedVenue"), RDFS.subPropertyOf, SKOS.related))
    graph.add((URIRef(f"{home}Status"), RDFS.subClassOf, SKOS.Concept))

    assert has_generic_parent(URIRef(f"{home}Event"), [graph])
    assert has_generic_parent(URIRef(f"{home}relatedVenue"), [graph])
    assert not has_generic_parent(URIRef(f"{home}Award"), [graph])
    # A class of values from a fixed list belongs under skos:Concept.
    assert not has_generic_parent(URIRef(f"{home}Status"), [graph])
    assert not has_generic_parent(URIRef(f"{home}Orphan"), [graph])
