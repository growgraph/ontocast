"""Merge-guard tests: literal conflicts, functional objects, siblings, strict lexical.

The guards encode one asymmetry: a false merge silently corrupts data
(30 vs 230 μJ/cm² becomes one node), a false split leaves visible,
recoverable redundancy. Fixtures mirror observed failure shapes.
"""

import pytest
from rdflib import Literal, URIRef
from rdflib.namespace import XSD

from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool import EmbeddingBasedAggregator
from ontocast.tool.agg.signatures import (
    MergeGuardContext,
    build_sibling_pairs,
    canonical_literal,
    empirically_functional_predicates,
    harvest_max_one_predicates,
)

pytestmark = pytest.mark.unit

QUDT = "http://qudt.org/schema/qudt/"
UNIT = "http://qudt.org/vocab/unit/"
QQVAL = "https://growgraph.dev/ontologies/qqval#"
NUMERIC_VALUE = URIRef(f"{QUDT}numericValue")
QUDT_UNIT = URIRef(f"{QUDT}unit")


def _facts_ttl(body: str) -> RDFGraph:
    graph = RDFGraph()
    graph.parse(
        data=f"""
        @prefix cd: <{DEFAULT_IRI}/> .
        @prefix qudt: <{QUDT}> .
        @prefix unit: <{UNIT}> .
        @prefix qqval: <{QQVAL}> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
        {body}
        """,
        format="turtle",
    )
    return graph


def _pair_representations(
    aggregator: EmbeddingBasedAggregator, graph: RDFGraph, left: URIRef, right: URIRef
):
    return {
        left: aggregator.normalizer.create_representation(left, graph),
        right: aggregator.normalizer.create_representation(right, graph),
    }


# --- canonical_literal -------------------------------------------------------


def test_canonical_literal_numeric_datatype_insensitive() -> None:
    assert canonical_literal(Literal("230", datatype=XSD.decimal)) == (
        "230",
        "numeric",
    )
    assert canonical_literal(Literal(230)) == ("230", "numeric")
    assert canonical_literal(Literal("230.0", datatype=XSD.double)) == (
        "230",
        "numeric",
    )
    assert canonical_literal(Literal("2019-01-24", datatype=XSD.date)) == (
        "2019-01-24",
        "temporal",
    )
    assert canonical_literal(Literal("meV", datatype=XSD.string)) is None
    assert canonical_literal(Literal("not a number")) is None


# --- schema harvesting -------------------------------------------------------


def test_harvest_max_one_predicates_functional_and_restrictions() -> None:
    ontology = RDFGraph()
    ontology.parse(
        data=f"""
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
        @prefix qqval: <{QQVAL}> .
        qqval:numericLowerBound a owl:DatatypeProperty, owl:FunctionalProperty .
        qqval:QuantityRange a owl:Class ;
            owl:equivalentClass [
                a owl:Restriction ;
                owl:onProperty qqval:hasLowerBound ;
                owl:maxCardinality "1"^^xsd:nonNegativeInteger
            ] .
        """,
        format="turtle",
    )
    harvested = harvest_max_one_predicates(ontology)
    assert URIRef(f"{QQVAL}numericLowerBound") in harvested
    assert URIRef(f"{QQVAL}hasLowerBound") in harvested
    assert harvest_max_one_predicates(None) == set()


def test_empirically_functional_predicates_requires_support_and_no_multi() -> None:
    subject_a = URIRef("http://x/a")
    subject_b = URIRef("http://x/b")
    single = URIRef("http://x/unitPred")
    multi = URIRef("http://x/inputMaterial")
    groups = {
        (subject_a, single): {URIRef("http://x/u1")},
        (subject_b, single): {URIRef("http://x/u2")},
        (subject_a, multi): {URIRef("http://x/m1"), URIRef("http://x/m2")},
        (subject_b, multi): {URIRef("http://x/m3")},
    }
    functional = empirically_functional_predicates(groups, min_support=2)
    assert single in functional
    assert multi not in functional
    # Below support the inference is not trusted.
    assert (
        empirically_functional_predicates(
            {(subject_a, single): {URIRef("http://x/u1")}}, min_support=2
        )
        == set()
    )


def test_build_sibling_pairs_subject_scope_covers_cross_predicate() -> None:
    range_node = URIRef("http://x/range1")
    lower = URIRef("http://x/lower")
    upper = URIRef("http://x/upper")
    groups = {
        (range_node, URIRef(f"{QQVAL}hasLowerBound")): {lower},
        (range_node, URIRef(f"{QQVAL}hasUpperBound")): {upper},
    }
    # Subject scope: the two bound endpoints hang off one subject via
    # *different* predicates and must never merge (degenerate-range shape).
    assert frozenset((lower, upper)) in build_sibling_pairs(groups, scope="subject")
    assert frozenset((lower, upper)) not in build_sibling_pairs(
        groups, scope="predicate"
    )


# --- guards through _can_merge_as_identity ----------------------------------


def test_literal_conflict_blocks_merge_case4_shape() -> None:
    """12.5 meV and 96 meV red shifts must never become one node."""
    aggregator = EmbeddingBasedAggregator()
    graph = _facts_ttl(
        """
        cd:pl_redshift_clean_sl_1 rdfs:label "PL redshift" ;
            qudt:numericValue "12.5"^^xsd:decimal ;
            qudt:unit unit:MilliEV .
        cd:pl_redshift_agg_sl_1 rdfs:label "PL redshift" ;
            qudt:numericValue "96"^^xsd:decimal ;
            qudt:unit unit:MilliEV .
        """
    )
    left = URIRef(f"{DEFAULT_IRI}/pl_redshift_clean_sl_1")
    right = URIRef(f"{DEFAULT_IRI}/pl_redshift_agg_sl_1")
    representations = _pair_representations(aggregator, graph, left, right)
    guard_context = MergeGuardContext()

    assert not aggregator._can_merge_as_identity(
        left, right, representations, guard_context=guard_context
    )
    assert "literal_conflict" in aggregator._merge_validation_failures(
        left, right, representations, guard_context=guard_context
    )
    # Without guard context (e.g. cross-graph aligner) the pair still fails on
    # the strict lexical bar for literal-bearing nodes... unless labels match
    # exactly — which they do here — so guard context is what blocks it.
    assert aggregator._can_merge_as_identity(left, right, representations)


def test_one_sided_and_overlapping_literals_still_merge() -> None:
    """Re-mention/enrichment must not be blocked."""
    aggregator = EmbeddingBasedAggregator()
    graph = _facts_ttl(
        """
        cd:val_a rdfs:label "superlattice dimension" ;
            qudt:numericValue "70"^^xsd:decimal .
        cd:val_b rdfs:label "superlattice dimension" .
        cd:val_c rdfs:label "superlattice dimension" ;
            qudt:numericValue "70.0"^^xsd:double .
        """
    )
    val_a = URIRef(f"{DEFAULT_IRI}/val_a")
    val_b = URIRef(f"{DEFAULT_IRI}/val_b")
    val_c = URIRef(f"{DEFAULT_IRI}/val_c")
    guard_context = MergeGuardContext()
    representations = {
        entity: aggregator.normalizer.create_representation(entity, graph)
        for entity in (val_a, val_b, val_c)
    }
    # one-sided: allowed
    assert aggregator._can_merge_as_identity(
        val_a, val_b, representations, guard_context=guard_context
    )
    # overlap after Decimal canonicalization ("70" == "70.0"): allowed
    assert aggregator._can_merge_as_identity(
        val_a, val_c, representations, guard_context=guard_context
    )


def test_functional_object_conflict_blocks_unit_mismatch() -> None:
    """10 °C and 10 kHz share the numeral — the unit object must block."""
    aggregator = EmbeddingBasedAggregator()
    graph = _facts_ttl(
        """
        cd:aging_temp rdfs:label "temperature condition" ;
            qudt:numericValue "10"^^xsd:decimal ;
            qudt:unit unit:DEG_C .
        cd:rep_rate rdfs:label "temperature condition" ;
            qudt:numericValue "10"^^xsd:decimal ;
            qudt:unit unit:KiloHZ .
        """
    )
    left = URIRef(f"{DEFAULT_IRI}/aging_temp")
    right = URIRef(f"{DEFAULT_IRI}/rep_rate")
    representations = _pair_representations(aggregator, graph, left, right)
    guard_context = MergeGuardContext(functional_predicates={QUDT_UNIT})

    # Values overlap ("10" == "10") so the literal guard alone allows it;
    # the functional-object guard on qudt:unit is what blocks.
    assert not aggregator._can_merge_as_identity(
        left, right, representations, guard_context=guard_context
    )
    assert "functional_iri_conflict" in aggregator._merge_validation_failures(
        left, right, representations, guard_context=guard_context
    )
    # Same predicate on a non-functional predicate set: allowed.
    assert aggregator._can_merge_as_identity(
        left, right, representations, guard_context=MergeGuardContext()
    )


def test_sibling_guard_blocks_co_listed_objects() -> None:
    aggregator = EmbeddingBasedAggregator()
    graph = _facts_ttl(
        """
        cd:sample_area_1 rdfs:label "superlattice sample in area 1" .
        cd:sample_area_2 rdfs:label "superlattice sample in area 2" .
        """
    )
    left = URIRef(f"{DEFAULT_IRI}/sample_area_1")
    right = URIRef(f"{DEFAULT_IRI}/sample_area_2")
    representations = _pair_representations(aggregator, graph, left, right)
    guard_context = MergeGuardContext(sibling_pairs={frozenset((left, right))})

    assert not aggregator._can_merge_as_identity(
        left, right, representations, guard_context=guard_context
    )
    assert "sibling" in aggregator._merge_validation_failures(
        left, right, representations, guard_context=guard_context
    )


def test_strict_lexical_bar_for_literal_bearing_nodes() -> None:
    """Distinct grants/persons with string payloads must not fuzzy-merge."""
    aggregator = EmbeddingBasedAggregator()
    graph = _facts_ttl(
        """
        @prefix schema: <https://schema.org/> .
        cd:grant_1 rdfs:label "Grant 11654005" ;
            schema:identifier "11654005"^^xsd:string .
        cd:grant_2 rdfs:label "Grant 61875256" ;
            schema:identifier "61875256"^^xsd:string .
        """
    )
    left = URIRef(f"{DEFAULT_IRI}/grant_1")
    right = URIRef(f"{DEFAULT_IRI}/grant_2")
    representations = _pair_representations(aggregator, graph, left, right)

    # Labels share the token "grant" (Jaccard 1/3 >= 0.2 old default) but both
    # nodes carry data literals, so only exact tiers apply: no merge.
    assert not aggregator._are_lexical_aliases(left, right, representations)


def test_exact_label_match_still_merges_literal_bearing_nodes() -> None:
    """Positive control: identical labels merge even with data literals."""
    aggregator = EmbeddingBasedAggregator()
    graph = _facts_ttl(
        """
        @prefix schema: <https://schema.org/> .
        cd:person_a rdfs:label "Hao Chang" ;
            schema:name "Hao Chang"^^xsd:string .
        cd:person_b rdfs:label "Hao Chang" ;
            schema:name "Hao Chang"^^xsd:string .
        """
    )
    left = URIRef(f"{DEFAULT_IRI}/person_a")
    right = URIRef(f"{DEFAULT_IRI}/person_b")
    representations = _pair_representations(aggregator, graph, left, right)

    assert aggregator._are_lexical_aliases(left, right, representations)


def test_aggregate_graphs_builds_guard_context_end_to_end(monkeypatch) -> None:
    """Full pipeline: co-listed quantity values with conflicting literals stay split."""
    from test.aggregation.test_aggregate_pipeline import make_fact_unit

    doc_iri = "https://example.org/docs/case-guards"
    ttl = f"""
    @prefix cd: <{DEFAULT_IRI}/> .
    @prefix qudt: <{QUDT}> .
    @prefix unit: <{UNIT}> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
    @prefix schema: <https://schema.org/> .
    cd:sl_sample schema:measurement cd:redshift_clean, cd:redshift_agg .
    cd:redshift_clean rdfs:label "PL red shift" ;
        qudt:numericValue "12.5"^^xsd:decimal ;
        qudt:unit unit:MilliEV .
    cd:redshift_agg rdfs:label "PL red shift" ;
        qudt:numericValue "96"^^xsd:decimal ;
        qudt:unit unit:MilliEV .
    """
    unit = make_fact_unit("guard e2e", 0, doc_iri, ttl)
    aggregator = EmbeddingBasedAggregator()

    def force_cluster(representations):
        return [list(representations.keys())], {}

    monkeypatch.setattr(aggregator.clusterer, "cluster_entities", force_cluster)
    result = aggregator.aggregate_graphs([unit], ontology_graph=RDFGraph()).graph

    value_subjects = {
        subject
        for subject, predicate, obj in result
        if predicate == NUMERIC_VALUE and isinstance(obj, Literal)
    }
    assert len(value_subjects) == 2
    for subject in value_subjects:
        values = {
            str(obj) for _, predicate, obj in result if predicate == NUMERIC_VALUE
        } & {str(o) for s, p, o in result if s == subject and p == NUMERIC_VALUE}
        assert len(values) == 1


def test_type_objects_excluded_from_sibling_groups() -> None:
    aggregator = EmbeddingBasedAggregator()
    from test.aggregation.test_aggregate_pipeline import make_fact_unit

    doc_iri = "https://example.org/docs/case-type-sibling"
    ttl = f"""
    @prefix cd: <{DEFAULT_IRI}/> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    cd:x rdf:type <http://onto/A>, <http://onto/B> .
    """
    unit = make_fact_unit("types", 0, doc_iri, ttl)
    (
        *_,
        object_groups,
    ) = aggregator._collect_all_entities([unit], known_ontology_entities=set())
    assert not object_groups
