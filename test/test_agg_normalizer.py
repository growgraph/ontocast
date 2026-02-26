from rdflib import RDF, RDFS, Literal, Namespace, URIRef

from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.agg.normalizer import EntityNormalizer


def test_normalize_string_camel_case(normalizer: EntityNormalizer) -> None:
    assert normalizer.normalize_string("PLRedShift") == "pl red shift"


def test_normalize_string_snake_case(normalizer: EntityNormalizer) -> None:
    assert normalizer.normalize_string("PL_red_shift_value") == "pl red shift value"


def test_normalize_string_diacritics(normalizer: EntityNormalizer) -> None:
    assert normalizer.normalize_string("Café") == "cafe"


def test_normalize_uri_variants(normalizer: EntityNormalizer) -> None:
    camel_uri = URIRef("http://example.org/PLRedShift")
    snake_uri = URIRef("http://example.org/PL_red_shift_value")
    assert normalizer.normalize_uri(camel_uri) == "pl red shift"
    assert normalizer.normalize_uri(snake_uri) == "pl red shift value"


def test_is_ontology_entity(normalizer: EntityNormalizer) -> None:
    assert normalizer.is_ontology_entity(URIRef("http://ontology.org/Thing")) is True
    assert normalizer.is_ontology_entity(URIRef(f"{DEFAULT_IRI}/entity")) is False


def test_create_representation_collects_metadata(normalizer: EntityNormalizer) -> None:
    graph = RDFGraph()
    ex = Namespace("http://example.org/")
    ont = Namespace("http://ontology.org/")

    entity = ex.TestEntity
    graph.add((entity, RDF.type, ont.Thing))
    graph.add((entity, RDFS.label, Literal("Test Entity")))
    graph.add((entity, ex.hasValue, Literal("123")))

    representation = normalizer.create_representation(entity, graph)

    assert representation.entity == entity
    assert "test entity" in representation.normal_form
    assert representation.types == [ont.Thing]
    assert "Test Entity" in representation.labels
    assert ex.hasValue in representation.properties
    assert "type" in representation.representation


def test_create_representation_marks_ontology_entity(
    normalizer: EntityNormalizer,
) -> None:
    graph = RDFGraph()
    ont = Namespace("http://ontology.org/")
    entity = ont.SomeClass
    graph.add((entity, RDF.type, RDFS.Class))

    representation = normalizer.create_representation(entity, graph)
    assert representation.is_ontology_entity is True
