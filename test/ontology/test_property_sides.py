"""A property's sides, stated by ``rdfs`` or listed by ``schema:*Includes``.

Every reader takes both kinds through one accessor; only a reader that infers a
type keeps to the entailing ones. An owned property's class sides are written
as listed classes.
"""

import pytest
from rdflib import OWL, RDF, RDFS, XSD, Literal, URIRef

from ontocast.onto.property_sides import (
    DOMAIN_INCLUDES,
    RANGE_INCLUDES,
    properties_with_side,
    property_sides,
    sides_by_property,
)
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.prompt.term_sheet import term_line
from ontocast.tool.facts_validation.literal_repair import materialize_entailed_types
from ontocast.tool.facts_validation.unit_findings import domain_violation_findings
from ontocast.tool.ontology_validation.axiom_hygiene import (
    list_owned_sides,
    property_side_reasons,
)

pytestmark = pytest.mark.unit

H = "https://example.com/home#"
CAT = "https://example.com/cat#"
F = "https://example.com/facts/"


def u(local: str, ns: str = H) -> URIRef:
    return URIRef(f"{ns}{local}")


def _ontology() -> RDFGraph:
    graph = RDFGraph()
    graph.bind("home", H)
    for cls in ("Volcano", "Mountain", "Person", "Eruption"):
        graph.add((u(cls), RDF.type, OWL.Class))
        graph.add((u(cls), RDFS.subClassOf, OWL.Thing))
    graph.add((u("elevation"), RDF.type, OWL.DatatypeProperty))
    graph.add((u("elevation"), DOMAIN_INCLUDES, u("Volcano")))
    graph.add((u("elevation"), DOMAIN_INCLUDES, u("Mountain")))
    graph.add((u("elevation"), RDFS.range, XSD.decimal))
    graph.add((u("eruptedIn"), RDF.type, OWL.ObjectProperty))
    graph.add((u("eruptedIn"), RDFS.domain, u("Volcano")))
    graph.add((u("eruptedIn"), RANGE_INCLUDES, u("Eruption")))
    return graph


def test_the_accessor_reads_both_kinds_and_says_which() -> None:
    onto = _ontology()

    assert property_sides(u("eruptedIn"), [onto], "domain") == {u("Volcano"): "rdfs"}
    assert property_sides(u("eruptedIn"), [onto], "range") == {
        u("Eruption"): "includes"
    }
    assert sides_by_property(onto, "domain")[u("elevation")] == {
        u("Volcano"),
        u("Mountain"),
    }
    assert u("elevation") not in sides_by_property(onto, "domain", source="rdfs")
    assert properties_with_side(onto, u("Volcano")) == {
        u("elevation"),
        u("eruptedIn"),
    }
    # The http spelling of schema.org lists a class as well.
    onto.add((u("eruptedIn"), URIRef("http://schema.org/rangeIncludes"), u("Person")))
    assert set(property_sides(u("eruptedIn"), [onto], "range")) == {
        u("Eruption"),
        u("Person"),
    }


def test_a_subject_fitting_no_listed_class_is_a_violation() -> None:
    facts = RDFGraph()
    facts.add((URIRef(F + "etna"), RDF.type, u("Volcano")))
    facts.add((URIRef(F + "etna"), u("elevation"), Literal("3357")))
    facts.add((URIRef(F + "ada"), RDF.type, u("Person")))
    facts.add((URIRef(F + "ada"), u("elevation"), Literal("1")))

    findings = domain_violation_findings(facts, _ontology())

    assert [(f.subject, f.predicate) for f in findings] == [
        (F + "ada", str(u("elevation")))
    ]
    assert "whose domain is" in findings[0].message


def test_listed_classes_type_nothing_when_there_are_several() -> None:
    facts = RDFGraph()
    peak = URIRef(F + "peak")
    lava = URIRef(F + "lava_1991")
    facts.add((peak, u("elevation"), Literal("3357")))
    facts.add((peak, u("eruptedIn"), lava))

    materialize_entailed_types(facts, _ontology())

    # rdfs:domain entails; the one class the range lists types the object.
    assert (peak, RDF.type, u("Volcano")) in facts
    assert (lava, RDF.type, u("Eruption")) in facts
    # Two listed domains are alternatives: neither is inferred.
    assert (peak, RDF.type, u("Mountain")) not in facts


def test_side_reasons_read_listed_sides() -> None:
    onto = _ontology()

    assert property_side_reasons(u("eruptedIn"), [onto]) == []
    assert property_side_reasons(u("elevation"), [onto]) == []


def test_an_owned_property_s_class_sides_are_restated_as_listed() -> None:
    graph = RDFGraph()
    graph.add((u("eruptedIn"), RDFS.domain, u("Volcano")))
    graph.add((u("eruptedIn"), RDFS.range, u("Eruption")))
    graph.add((u("elevation"), RDFS.domain, u("Volcano")))
    graph.add((u("elevation"), RDFS.range, XSD.decimal))
    graph.add((u("locatedIn", CAT), RDFS.domain, u("Place", CAT)))

    changed = list_owned_sides(graph, lambda iri: iri.startswith(H))

    assert changed == 3
    assert (u("eruptedIn"), DOMAIN_INCLUDES, u("Volcano")) in graph
    assert (u("eruptedIn"), RANGE_INCLUDES, u("Eruption")) in graph
    assert (u("elevation"), DOMAIN_INCLUDES, u("Volcano")) in graph
    assert (u("elevation"), RDFS.range, XSD.decimal) in graph
    assert (u("eruptedIn"), RDFS.domain, None) not in graph
    # A catalog property keeps its author's axiom.
    assert (u("locatedIn", CAT), RDFS.domain, u("Place", CAT)) in graph


def test_the_term_sheet_shows_listed_sides() -> None:
    onto = _ontology()

    [line] = term_line(onto, u("elevation"), kind="property")[:1]

    assert "home:Mountain/home:Volcano -> xsd:decimal" in line


def test_the_delta_clean_up_lists_the_home_s_sides_only() -> None:
    from ontocast.stategraph.node_factories import clean_ontology_delta
    from ontocast.tool.ontology_validation.shape import build_catalog_surface_index

    catalog = RDFGraph()
    catalog.add((u("locatedIn", CAT), RDF.type, OWL.ObjectProperty))
    catalog.add((u("Place", CAT), RDF.type, OWL.Class))
    delta = RDFGraph()
    delta.add((u("eruptedIn"), RDF.type, OWL.ObjectProperty))
    delta.add((u("eruptedIn"), RDFS.label, Literal("erupted in")))
    delta.add((u("eruptedIn"), RDFS.domain, u("Place", CAT)))
    delta.add((u("locatedIn", CAT), RDFS.range, u("Place", CAT)))

    counts = clean_ontology_delta(
        [delta],
        external=[],
        current_domain="https://example.com/",
        catalog_index=build_catalog_surface_index([catalog]),
        ontology_iris=[],
        home_namespace=H,
    )

    assert counts["owned_sides_listed"] == 1
    assert (u("eruptedIn"), DOMAIN_INCLUDES, u("Place", CAT)) in delta


def test_retrieval_links_a_class_to_the_properties_listing_it() -> None:
    from ontocast.tool.sparql import _crosslink_property_seeds

    linked = _crosslink_property_seeds(
        _ontology(), [str(u("Mountain"))], [], frozenset()
    )

    assert linked == [str(u("elevation"))]
