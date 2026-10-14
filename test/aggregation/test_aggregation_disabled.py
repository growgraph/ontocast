"""``AGG_ENABLED=false``: facts units are named and rewritten, never merged.

Off must be an honest arm: two units minting the same local name stay two
nodes, no candidate is clustered or anchored (so the aggregation encoder is
never loaded), and the merged graph still carries per-unit provenance.
"""

import pytest
from pydantic import ValidationError
from rdflib import RDF, URIRef

from ontocast.config import AggregationConfig, Config
from ontocast.onto.constants import DEFAULT_IRI, PROV
from ontocast.onto.content_unit import ContentUnit, OutputType
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool import EmbeddingBasedAggregator
from ontocast.tool.agg.unit_scope import unit_scope_index

pytestmark = pytest.mark.unit

CD = DEFAULT_IRI
DOC = "https://x.org/doc/1"
Q = "https://x.org/schema#"
PREFIXES = f"""
@prefix cd: <{CD}> .
@prefix q: <{Q}> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
"""


def _unit(index: int, body: str) -> ContentUnit:
    graph = RDFGraph()
    graph.parse(data=PREFIXES + body, format="turtle")
    return ContentUnit(
        text=f"unit {index}",
        index=index,
        doc_iri=URIRef(DOC),
        graph=graph,
        type=OutputType.FACTS,
    )


def _disabled_aggregator(monkeypatch: pytest.MonkeyPatch) -> EmbeddingBasedAggregator:
    aggregator = EmbeddingBasedAggregator(AggregationConfig(enabled=False))

    def must_not_run(*args, **kwargs):
        raise AssertionError("consolidation ran with AGG_ENABLED=false")

    monkeypatch.setattr(aggregator.clusterer, "cluster_entities", must_not_run)
    monkeypatch.setattr(aggregator, "_select_ontology_anchor_candidates", must_not_run)
    return aggregator


def test_same_local_name_in_two_units_stays_two_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregator = _disabled_aggregator(monkeypatch)
    units = [
        _unit(0, 'cd:sample_a a q:Sample ; rdfs:label "Sample A" ; q:thickness 100 .'),
        _unit(1, 'cd:sample_a a q:Sample ; rdfs:label "Sample A" ; q:growth "MBE" .'),
    ]

    result = aggregator.postprocess_facts_units(units, RDFGraph())

    samples = set(result.graph.subjects(RDF.type, URIRef(f"{Q}Sample")))
    assert len(samples) == 2
    assert all(unit_scope_index(str(sample)) is None for sample in samples)
    assert result.merged_clusters == {}
    assert result.rejected_merge_count == 0
    assert result.key_supported_clusters == []


def test_provenance_still_names_each_source_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregator = _disabled_aggregator(monkeypatch)
    units = [
        _unit(0, 'cd:sample_a a q:Sample ; rdfs:label "Sample A" .'),
        _unit(1, 'cd:sample_b a q:Sample ; rdfs:label "Sample B" .'),
    ]

    result = aggregator.postprocess_facts_units(units, RDFGraph())

    sources = set(result.graph.objects(None, PROV.wasDerivedFrom))
    assert len(sources) == 2


def test_known_ontology_terms_are_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregator = _disabled_aggregator(monkeypatch)
    ontology = RDFGraph()
    ontology.parse(
        data=PREFIXES
        + "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
        + 'q:Sample a owl:Class ; rdfs:label "Sample" .',
        format="turtle",
    )
    units = [_unit(0, 'cd:sample_a a q:Sample ; rdfs:label "Sample A" .')]

    result = aggregator.postprocess_facts_units(units, ontology)

    assert len(set(result.graph.subjects(RDF.type, URIRef(f"{Q}Sample")))) == 1
    assert result.merged_clusters == {}


def test_env_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGG_ENABLED", "false")

    assert Config().tool_config.aggregation.enabled is False


def test_off_requires_unit_scoping() -> None:
    with pytest.raises(ValidationError, match="AGG_UNIT_SCOPED_FACT_IRIS"):
        AggregationConfig(enabled=False, unit_scoped_fact_iris=False)
