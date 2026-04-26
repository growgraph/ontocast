from rdflib import OWL, RDF, RDFS, URIRef

from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.agg.normalizer import EntityRepresentation
from ontocast.tool.agg.uri_builder import (
    EntityRole,
    URIBuilder,
    detect_role,
    format_structured_id,
    has_structured_id,
    normalize_local_name,
    to_lower_camel_case,
    to_pascal_case,
)


def make_representation(uri: str, normal_form: str) -> EntityRepresentation:
    return EntityRepresentation(
        iri=URIRef(uri),
        normal_form=normal_form,
        types=[],
        properties=[],
        labels=[],
        representation=normal_form,
        is_ontology_entity=False,
    )


def test_pascal_case_and_lower_camel_helpers() -> None:
    assert to_pascal_case("judicial decision") == "JudicialDecision"
    assert to_pascal_case("case") == "Case"
    assert to_lower_camel_case("has decision") == "hasDecision"
    assert to_lower_camel_case("name") == "name"


def test_structured_id_helpers() -> None:
    assert has_structured_id(URIRef("http://ex.org/Case_2023_456")) is True
    assert has_structured_id(URIRef("http://ex.org/Person")) is False
    assert (
        format_structured_id(URIRef("http://ex.org/case_2023_456")) == "case_2023_456"
    )


def test_detect_role_for_class_property_and_instance() -> None:
    graph = RDFGraph()
    class_entity = URIRef("http://ex.org/Person")
    prop_entity = URIRef("http://ex.org/hasAge")
    instance_entity = URIRef("http://ex.org/Alice")

    graph.add((class_entity, RDF.type, RDFS.Class))
    graph.add((prop_entity, RDF.type, OWL.DatatypeProperty))
    graph.add((instance_entity, RDF.type, class_entity))

    assert detect_role(class_entity, graph) == EntityRole.CLASS
    assert detect_role(prop_entity, graph) == EntityRole.PROPERTY
    assert detect_role(instance_entity, graph) == EntityRole.INSTANCE


def test_normalize_local_name_uses_role_specific_formatting() -> None:
    class_rep = make_representation(
        "http://ex.org/JudicialDecision", "judicial decision"
    )
    prop_rep = make_representation("http://ex.org/hasDecision", "has decision")
    structured_rep = make_representation("http://ex.org/Case_2023_456", "case 2023 456")

    assert normalize_local_name(class_rep, EntityRole.CLASS) == "JudicialDecision"
    assert normalize_local_name(prop_rep, EntityRole.PROPERTY) == "hasDecision"
    assert normalize_local_name(structured_rep, EntityRole.INSTANCE) == "Case_2023_456"


def test_build_uri_preserves_ontology_entities(uri_builder: URIBuilder) -> None:
    entity = URIRef("http://ontology.org/Thing")
    rep = EntityRepresentation(
        iri=entity,
        normal_form="thing",
        types=[],
        properties=[],
        labels=[],
        representation="thing",
        is_ontology_entity=True,
    )
    assert uri_builder.build_uri(entity, rep, EntityRole.CLASS) == entity


def test_compose_mappings_flattens_two_stage_mapping() -> None:
    e1 = URIRef("http://chunk1.org/A")
    e2 = URIRef("http://chunk2.org/B")
    representative = URIRef("http://chunk1.org/A")
    final = URIRef(f"{DEFAULT_IRI}/SomeEntity")

    composed = URIBuilder.compose_mappings(
        {e1: representative, e2: representative},
        {representative: final},
    )

    assert composed[e1] == final
    assert composed[e2] == final


def test_create_entity_uri_mapping_uses_doc_namespace_and_avoids_collisions() -> None:
    builder = URIBuilder(base_iri=DEFAULT_IRI)
    doc_iri = URIRef("https://example.org/docs/case-1")
    left = URIRef("https://growgraph.dev/factsEntityA")
    right = URIRef("https://growgraph.dev/factsEntityB")
    left_canonical = URIRef("https://growgraph.dev/factsCanonicalA")
    right_canonical = URIRef("https://growgraph.dev/factsCanonicalB")

    shared_representation = EntityRepresentation(
        iri=left_canonical,
        normal_form="collision",
        types=[],
        properties=[],
        labels=[],
        representation="collision",
        is_ontology_entity=False,
    )

    representations = {
        left_canonical: shared_representation,
        right_canonical: EntityRepresentation(
            iri=right_canonical,
            normal_form="collision",
            types=[],
            properties=[],
            labels=[],
            representation="collision",
            is_ontology_entity=False,
        ),
    }

    mapping = builder.create_entity_uri_mapping(
        identity_mapping={left: left_canonical, right: right_canonical},
        representations=representations,
        entity_doc_iris={left: doc_iri, right: doc_iri},
        entity_is_ontology={left_canonical: False, right_canonical: False},
    )

    assert str(mapping[left]).startswith(f"{doc_iri}/")
    assert str(mapping[right]).startswith(f"{doc_iri}/")
    assert mapping[left] != mapping[right]
