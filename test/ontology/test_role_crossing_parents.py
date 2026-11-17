"""A parent is of its child's kind: a class above a class, a property above a property."""

import pytest
from rdflib import OWL, RDF, RDFS, URIRef

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.well_known import well_known_role
from ontocast.tool.ontology_validation.axiom_hygiene import drop_role_crossing_parents
from ontocast.tool.ontology_validation.shape import (
    CatalogSurfaceIndex,
    role_compatible,
)

pytestmark = pytest.mark.unit

SCHEMA = "https://schema.org/"
XSD = "http://www.w3.org/2001/XMLSchema#"
HOME = "https://example.com/home#"
CAT = "https://example.com/cat#"


@pytest.mark.parametrize(
    ("iri", "role"),
    [
        (f"{SCHEMA}Property", "class"),
        (f"{SCHEMA}Thing", "class"),
        (f"{SCHEMA}subEvent", "property"),
        (f"{XSD}string", "datatype"),
        (str(RDF.Property), "class"),
        (str(RDF.langString), "datatype"),
        ("http://qudt.org/schema/qudt/numericValue", "property"),
        ("http://qudt.org/vocab/unit/M", None),
        (str(RDF.nil), None),
        (f"{SCHEMA}noSuchTerm", None),
        ("https://example.com/x", None),
    ],
)
def test_well_known_role(iri: str, role: str | None) -> None:
    assert well_known_role(iri) == role


def test_role_compatible_judges_well_known_terms_outside_the_catalog() -> None:
    index = CatalogSurfaceIndex()
    assert not role_compatible("property", f"{SCHEMA}Property", index)
    assert role_compatible("property", f"{SCHEMA}subEvent", index)
    assert not role_compatible("class", f"{SCHEMA}subEvent", index)
    assert not role_compatible("class", f"{XSD}string", index)
    # Outside the catalog and the well-known vocabularies nothing is known.
    assert role_compatible("class", "https://example.com/x", index)


def test_role_compatible_keeps_a_term_the_catalog_lists_as_both() -> None:
    iri = f"{CAT}both"
    index = CatalogSurfaceIndex(terms={iri}, classes={iri}, properties={iri})
    assert role_compatible("class", iri, index)
    assert role_compatible("property", iri, index)


def test_parents_of_the_other_kind_are_dropped() -> None:
    graph = RDFGraph()
    has_season = URIRef(f"{HOME}hasSeason")
    season = URIRef(f"{HOME}Season")
    score = URIRef(f"{HOME}Score")
    host = URIRef(f"{HOME}host")
    graph.add((has_season, RDF.type, OWL.ObjectProperty))
    graph.add((has_season, RDFS.subPropertyOf, URIRef(f"{SCHEMA}Property")))
    graph.add((season, RDF.type, OWL.Class))
    graph.add((season, RDFS.subClassOf, URIRef(f"{CAT}hostCity")))
    graph.add((score, RDFS.subClassOf, URIRef(f"{XSD}integer")))
    graph.add((host, RDFS.subPropertyOf, has_season))
    graph.add((season, RDFS.subClassOf, URIRef(f"{SCHEMA}Event")))

    removed = drop_role_crossing_parents(
        graph, catalog_classes=set(), catalog_properties={f"{CAT}hostCity"}
    )

    assert removed == 3
    assert (host, RDFS.subPropertyOf, has_season) in graph
    assert (season, RDFS.subClassOf, URIRef(f"{SCHEMA}Event")) in graph
    assert (has_season, RDFS.subPropertyOf, None) not in graph
