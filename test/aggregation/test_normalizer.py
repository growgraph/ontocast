from rdflib import RDF, RDFS, Literal, Namespace, URIRef

from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.agg.normalizer import EntityNormalizer, EntityRepresentation


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

    assert representation.iri == entity
    assert "test entity" in representation.normal_form
    assert representation.types == [ont.Thing]
    assert "Test Entity" in representation.labels
    assert ex.hasValue in representation.properties
    assert "type" in representation.representation
    assert representation.core_representation.startswith("test entity")


def test_create_representation_uses_alt_labels_when_no_rdfs_label(
    normalizer: EntityNormalizer,
) -> None:
    graph = RDFGraph()
    ex = Namespace("http://example.org/")
    rel = Namespace("http://relations.example/")

    entity = ex.fact4
    graph.add((entity, RDF.type, ex.Person))
    graph.add((entity, rel.screenwriter, Literal("Maurice Noble")))

    representation = normalizer.create_representation(entity, graph)

    assert representation.labels == []
    assert "Maurice Noble" in representation.alt_labels
    assert representation.core_representation.startswith("maurice noble")


def test_create_representation_marks_ontology_entity(
    normalizer: EntityNormalizer,
) -> None:
    graph = RDFGraph()
    ont = Namespace("http://ontology.org/")
    entity = ont.SomeClass
    graph.add((entity, RDF.type, RDFS.Class))

    representation = normalizer.create_representation(entity, graph)
    assert representation.is_ontology_entity is True


def test_create_representation_builds_deterministic_neighborhood(
    normalizer: EntityNormalizer,
) -> None:
    ex = Namespace("http://example.org/")
    graph_a = RDFGraph()
    graph_b = RDFGraph()
    triples = [
        (ex.A, ex.relatesTo, ex.B),
        (ex.C, ex.relatesTo, ex.A),
        (ex.B, ex.A, ex.C),
    ]
    for triple in triples:
        graph_a.add(triple)
    for triple in reversed(triples):
        graph_b.add(triple)

    rep_a = normalizer.create_representation(ex.A, graph_a)
    rep_b = normalizer.create_representation(ex.A, graph_b)

    assert rep_a.neighborhood_representation
    assert rep_a.neighborhood_representation == rep_b.neighborhood_representation
    assert rep_a.representation.startswith(rep_a.core_representation)
    assert rep_a.neighborhood_representation in rep_a.representation


def test_entity_representation_backfills_core_for_legacy_constructor() -> None:
    entity = URIRef("http://example.org/Legacy")
    representation = EntityRepresentation(
        iri=entity,
        normal_form="legacy",
        types=[],
        properties=[],
        labels=[],
        representation="legacy representation",
        is_ontology_entity=False,
    )

    assert representation.core_representation == "legacy representation"
    assert representation.representation == "legacy representation"


def test_entity_representation_contract_iri_properties(
    normalizer: EntityNormalizer,
) -> None:
    graph = RDFGraph()
    ont = Namespace("http://ontology.org/")
    entity = ont.SomeClass
    graph.add((entity, RDF.type, RDFS.Class))

    representation = normalizer.create_representation(entity, graph)
    assert str(representation.iri) == str(entity)
    assert representation.ontology_iri is None
