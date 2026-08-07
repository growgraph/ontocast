"""Repro + regression for cross-chunk person-variant identity.

Benchmark case7 produced ``doc:baranovD`` ("Baranov, D.", from the reference
list) and ``doc:personDmitryBaranov`` ("Dmitry Baranov", from the author
block) with no ``owl:sameAs`` link. The merge gates blocked the pair twice:
name-part literals (given/family name — mandated by prompt rule 7 atomicity)
tripped the blanket data-literal bar, and initials-style labels share no
exact normalized token.

These tests exercise the validation gates directly (no embeddings needed) so
they run fast and deterministically.
"""

from rdflib import URIRef

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.agg.aggregate import EmbeddingBasedAggregator

SCHEMA = "https://schema.org/"
DOC = "https://growgraph.dev/doc/testdoc/"

PERSON_FULL = URIRef(f"{DOC}personDmitryBaranov")
PERSON_INITIAL = URIRef(f"{DOC}baranovD")


def _person_graph() -> RDFGraph:
    graph = RDFGraph()
    graph.parse(
        data=f"""
@prefix schema: <{SCHEMA}> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix doc: <{DOC}> .

doc:personDmitryBaranov a schema:Person ;
    rdfs:label "Dmitry Baranov"@en ;
    schema:givenName "Dmitry"@en ;
    schema:familyName "Baranov"@en ;
    schema:email "dmitry.baranov@example.org" .

doc:baranovD a schema:Person ;
    rdfs:label "Baranov, D."@en ;
    schema:familyName "Baranov"@en .
""",
        format="turtle",
    )
    return graph


def _representations(graph: RDFGraph):
    aggregator = EmbeddingBasedAggregator()
    reps = {
        entity: aggregator.normalizer.create_representation(entity, graph)
        for entity in (PERSON_FULL, PERSON_INITIAL)
    }
    return aggregator, reps


def test_person_initial_variant_is_lexical_alias() -> None:
    aggregator, reps = _representations(_person_graph())
    assert aggregator._are_lexical_aliases(PERSON_FULL, PERSON_INITIAL, reps)


def test_person_initial_variant_passes_all_merge_gates() -> None:
    aggregator, reps = _representations(_person_graph())
    failures = aggregator._merge_validation_failures(PERSON_FULL, PERSON_INITIAL, reps)
    assert failures == []


def test_name_literals_do_not_trip_guard_literal_bar() -> None:
    _, reps = _representations(_person_graph())
    # String name parts must not put persons behind the strict measurement bar.
    assert not reps[PERSON_FULL].has_guard_literal
    assert not reps[PERSON_INITIAL].has_guard_literal


def test_measurement_nodes_keep_strict_bar() -> None:
    graph = RDFGraph()
    graph.parse(
        data=f"""
@prefix qudt: <http://qudt.org/schema/qudt/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix doc: <{DOC}> .

doc:red_shift_sl1 a qudt:QuantityValue ;
    rdfs:label "PL red shift of SL1"@en ;
    qudt:numericValue "12.5"^^xsd:decimal .

doc:red_shift_sl2 a qudt:QuantityValue ;
    rdfs:label "PL red shift of SL2"@en ;
    qudt:numericValue "96"^^xsd:decimal .
""",
        format="turtle",
    )
    aggregator = EmbeddingBasedAggregator()
    left = URIRef(f"{DOC}red_shift_sl1")
    right = URIRef(f"{DOC}red_shift_sl2")
    reps = {
        entity: aggregator.normalizer.create_representation(entity, graph)
        for entity in (left, right)
    }
    assert reps[left].has_guard_literal
    assert reps[right].has_guard_literal
    # Near-identical labels with distinct payloads must NOT alias.
    assert not aggregator._are_lexical_aliases(left, right, reps)


def test_disjoint_identifier_strings_conflict() -> None:
    graph = RDFGraph()
    graph.parse(
        data=f"""
@prefix schema: <{SCHEMA}> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix doc: <{DOC}> .

doc:sample_a a schema:Thing ;
    rdfs:label "perovskite sample"@en ;
    schema:identifier "S-2024-001" .

doc:sample_b a schema:Thing ;
    rdfs:label "perovskite sample"@en ;
    schema:identifier "S-2024-002" .
""",
        format="turtle",
    )
    aggregator = EmbeddingBasedAggregator()
    left = URIRef(f"{DOC}sample_a")
    right = URIRef(f"{DOC}sample_b")
    reps = {
        entity: aggregator.normalizer.create_representation(entity, graph)
        for entity in (left, right)
    }
    assert aggregator._have_conflicting_literals(reps[left], reps[right])


def test_initials_do_not_conflict_as_strings() -> None:
    """Prefix/initial-compatible strings on a shared predicate are not a conflict."""
    _, reps = _representations(_person_graph())
    aggregator = EmbeddingBasedAggregator()
    assert not aggregator._have_conflicting_literals(
        reps[PERSON_FULL], reps[PERSON_INITIAL]
    )


def test_unrelated_persons_do_not_alias() -> None:
    graph = RDFGraph()
    graph.parse(
        data=f"""
@prefix schema: <{SCHEMA}> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix doc: <{DOC}> .

doc:baranovD a schema:Person ;
    rdfs:label "Baranov, D."@en .

doc:akkermanQ a schema:Person ;
    rdfs:label "Akkerman, Q."@en .
""",
        format="turtle",
    )
    aggregator = EmbeddingBasedAggregator()
    left = URIRef(f"{DOC}baranovD")
    right = URIRef(f"{DOC}akkermanQ")
    reps = {
        entity: aggregator.normalizer.create_representation(entity, graph)
        for entity in (left, right)
    }
    assert not aggregator._are_lexical_aliases(left, right, reps)
