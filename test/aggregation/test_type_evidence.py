"""Asserted types are evidence, not identity.

Units type one thing differently. Between two minted instances only a declared
``owl:disjointWith`` (through the superclass closure) keeps a same-named pair
apart; a shared name brings the pair into one candidate cluster even when the
embeddings split it. Clustering is stubbed to singletons, so every merge here
comes from the label pairs, and no embedding model loads.
"""

import pytest
from rdflib import URIRef

from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.content_unit import ContentUnit, OutputType
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.agg.aggregate import EmbeddingBasedAggregator

pytestmark = pytest.mark.unit

CD = f"{DEFAULT_IRI}/"
DOC = "https://x.org/doc/1"
PREFIXES = f"""
@prefix cd: <{CD}> .
@prefix ex: <https://x.org/onto#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
"""


def _graph(body: str) -> RDFGraph:
    graph = RDFGraph()
    graph.parse(data=PREFIXES + body, format="turtle")
    return graph


def _unit(index: int, body: str) -> ContentUnit:
    return ContentUnit(
        text=f"unit {index}",
        index=index,
        doc_iri=URIRef(DOC),
        graph=_graph(body),
        type=OutputType.FACTS,
    )


def _aggregate(monkeypatch, units: list[ContentUnit], ontology: str = ""):
    aggregator = EmbeddingBasedAggregator()
    monkeypatch.setattr(
        aggregator.clusterer,
        "cluster_entities",
        lambda representations: ([[entity] for entity in representations], {}),
    )
    return aggregator.postprocess_facts_units(
        units, _graph(ontology), doc_iri=URIRef(DOC)
    )


def _merged_with(result, local: str) -> bool:
    return any(
        any(member.startswith(f"{CD}{local}__u") for member in members)
        and len(members) > 1
        for members in result.merged_clusters.values()
    )


def _place_units(first_type: str, second_type: str) -> list[ContentUnit]:
    return [
        _unit(
            0,
            f'cd:riverton a {first_type} ; rdfs:label "Riverton" .\n'
            'cd:station a ex:Sensor ; rdfs:label "station" ; ex:at cd:riverton .',
        ),
        _unit(
            1,
            f'cd:riverton_town a {second_type} ; rdfs:label "Riverton" .\n'
            'cd:heat a ex:Event ; rdfs:label "heat" ; ex:hit cd:riverton_town .',
        ),
    ]


def test_same_name_with_unrelated_types_merges(monkeypatch) -> None:
    result = _aggregate(
        monkeypatch, _place_units("ex:Location", "ex:FeatureOfInterest")
    )

    assert _merged_with(result, "riverton")


def test_declared_disjointness_keeps_them_apart(monkeypatch) -> None:
    result = _aggregate(
        monkeypatch,
        _place_units("ex:Location", "ex:Person"),
        ontology="ex:Location owl:disjointWith ex:Person .",
    )

    assert not _merged_with(result, "riverton")


def test_disjointness_holds_through_the_superclass_closure(monkeypatch) -> None:
    result = _aggregate(
        monkeypatch,
        _place_units("ex:City", "ex:Author"),
        ontology="""
        ex:City rdfs:subClassOf ex:Place .
        ex:Author rdfs:subClassOf ex:Agent .
        [] a owl:AllDisjointClasses ; owl:members ( ex:Place ex:Agent ) .
        """,
    )

    assert not _merged_with(result, "riverton")


def test_a_name_shared_too_widely_brings_nothing_together(monkeypatch) -> None:
    units = [
        _unit(
            index, f'cd:item{index} a ex:Thing ; rdfs:label "item" ; ex:n "{index}" .'
        )
        for index in range(10)
    ]

    result = _aggregate(monkeypatch, units)

    assert all(len(members) == 1 for members in result.merged_clusters.values())


def test_equal_values_under_different_carriers_stay_two_values(monkeypatch) -> None:
    value = '[ a ex:Value ; rdfs:label "49.6 C" ; ex:number "49.6"^^xsd:decimal ]'
    result = _aggregate(
        monkeypatch,
        [
            _unit(
                0,
                f'cd:max a ex:Observation ; rdfs:label "max" ; ex:hasResult {value} .',
            ),
            _unit(
                1,
                f'cd:peak a ex:Observation ; rdfs:label "peak" ; ex:hasResult {value} .',
            ),
        ],
    )

    results = list(result.graph.objects(None, URIRef("https://x.org/onto#hasResult")))
    assert len(results) == 2
    assert len(set(results)) == 2


def _plausible_pair(first_type: str, second_type: str) -> list[ContentUnit]:
    return [
        _unit(
            0,
            f'cd:temp a {first_type} ; rdfs:label "temperature" .\n'
            'cd:obs_a a ex:Obs ; rdfs:label "obs a" ; ex:measures cd:temp .',
        ),
        _unit(
            1,
            f'cd:daily_temp a {second_type} ; rdfs:label "daily maximum temperature" .\n'
            'cd:obs_b a ex:Obs ; rdfs:label "obs b" ; ex:tracks cd:daily_temp .',
        ),
    ]


def _one_cluster(monkeypatch, units: list[ContentUnit], ontology: str = ""):
    aggregator = EmbeddingBasedAggregator()
    monkeypatch.setattr(
        aggregator.clusterer,
        "cluster_entities",
        lambda representations: ([list(representations)], {}),
    )
    return aggregator.postprocess_facts_units(
        units, _graph(ontology), doc_iri=URIRef(DOC)
    )


def test_a_plausible_name_merges_when_the_types_agree(monkeypatch) -> None:
    result = _one_cluster(
        monkeypatch,
        _plausible_pair("ex:DailyMax", "ex:Property"),
        ontology="ex:DailyMax rdfs:subClassOf ex:Property .",
    )

    assert _merged_with(result, "temp")


def test_a_plausible_name_does_not_merge_types_that_share_nothing(monkeypatch) -> None:
    result = _one_cluster(monkeypatch, _plausible_pair("ex:Property", "ex:Sensor"))

    assert not _merged_with(result, "temp")


def test_refusals_are_counted_by_reason(monkeypatch) -> None:
    result = _one_cluster(monkeypatch, _plausible_pair("ex:Property", "ex:Sensor"))

    assert result.rejected_by_reason.get("type", 0) >= 1
    assert sum(result.rejected_by_reason.values()) >= result.rejected_merge_count
