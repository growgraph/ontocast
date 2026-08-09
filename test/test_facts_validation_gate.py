"""Tests for the post-aggregation facts validation gate.

The gate is defense-in-depth behind the merge guards: it detects invariant
violations in the aggregated graph (notably transitive union-find merges the
pairwise guards cannot block) and repairs them deterministically by vetoing
the offending cluster's pairs and re-aggregating.
"""

from types import SimpleNamespace
from typing import cast

from rdflib import OWL, RDF, RDFS, Literal, URIRef
from rdflib.namespace import XSD

from ontocast.api.process_helpers import validate_unit_pipeline_facts
from ontocast.config import FactsValidationConfig
from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.content_unit import ContentUnit, OutputType
from ontocast.onto.enum import RetrievalMetric
from ontocast.onto.model import FactsValidationFinding, FactsValidationFindingKind
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.stategraph.node_factories import (
    _vetoes_from_findings,
    make_merge_facts_node,
    make_validate_facts_node,
)
from ontocast.tool import EmbeddingBasedAggregator
from ontocast.tool.agg.aggregate import build_merged_clusters
from ontocast.tool.facts_invariants import validate_aggregated_facts
from ontocast.toolbox import ToolBox

CD = f"{DEFAULT_IRI}/"
Q = "https://x.org/schema#"


def _fact_unit(index: int, ttl: str, text: str = "text") -> ContentUnit:
    graph = RDFGraph()
    graph.parse(data=ttl, format="turtle")
    return ContentUnit(
        text=text,
        index=index,
        doc_iri=URIRef("https://x.org/doc/1"),
        graph=graph,
        type=OutputType.FACTS,
    )


def _normal_form_aggregator(monkeypatch) -> EmbeddingBasedAggregator:
    aggregator = EmbeddingBasedAggregator()

    def cluster_by_normal_form(representations):
        clusters: dict[str, list[URIRef]] = {}
        for entity, representation in representations.items():
            clusters.setdefault(representation.normal_form, []).append(entity)
        return list(clusters.values()), {}

    monkeypatch.setattr(
        aggregator.clusterer, "cluster_entities", cluster_by_normal_form
    )
    return aggregator


# --- validate_aggregated_facts -----------------------------------------------


def test_functional_violation_detected() -> None:
    ontology = RDFGraph()
    ontology.add((URIRef(Q + "unit"), RDF.type, OWL.FunctionalProperty))
    graph = RDFGraph()
    subject = URIRef(CD + "q1")
    graph.add((subject, URIRef(Q + "unit"), URIRef(Q + "MicroJ")))
    graph.add((subject, URIRef(Q + "unit"), URIRef(Q + "MicroM")))

    report = validate_aggregated_facts(graph, ontology, fact_namespaces=[CD])

    kinds = {finding.kind for finding in report.findings}
    assert FactsValidationFindingKind.FUNCTIONAL_VIOLATION in kinds
    assert report.error_findings[0].subject == str(subject)


def test_suspect_multi_numeric_detected_and_severity_configurable() -> None:
    graph = RDFGraph()
    subject = URIRef(CD + "q1")
    predicate = URIRef(Q + "numericValue")
    graph.add((subject, predicate, Literal("12.5", datatype=XSD.double)))
    graph.add((subject, predicate, Literal("96", datatype=XSD.double)))

    report = validate_aggregated_facts(graph, None, fact_namespaces=[CD])
    assert [f.kind for f in report.error_findings] == [
        FactsValidationFindingKind.SUSPECT_MULTI_VALUE
    ]

    relaxed = validate_aggregated_facts(
        graph, None, fact_namespaces=[CD], suspect_multi_value_severity="warning"
    )
    assert not relaxed.error_findings
    assert len(relaxed.findings) == 1


def test_equal_canonical_numerics_are_not_flagged() -> None:
    graph = RDFGraph()
    subject = URIRef(CD + "q1")
    predicate = URIRef(Q + "numericValue")
    graph.add((subject, predicate, Literal("230", datatype=XSD.integer)))
    graph.add((subject, predicate, Literal("230.0", datatype=XSD.double)))

    report = validate_aggregated_facts(graph, None, fact_namespaces=[CD])
    assert not report.findings


def test_dominant_single_valued_predicate_multi_object_flagged() -> None:
    graph = RDFGraph()
    predicate = URIRef(Q + "unit")
    for index in range(3):
        graph.add((URIRef(CD + f"q{index}"), predicate, URIRef(Q + f"Unit{index}")))
    offender = URIRef(CD + "q9")
    graph.add((offender, predicate, URIRef(Q + "UnitA")))
    graph.add((offender, predicate, URIRef(Q + "UnitB")))

    report = validate_aggregated_facts(graph, None, fact_namespaces=[CD])

    assert len(report.error_findings) == 1
    finding = report.error_findings[0]
    assert finding.kind == FactsValidationFindingKind.SUSPECT_MULTI_VALUE
    assert finding.subject == str(offender)


def test_degenerate_coreference_detected() -> None:
    ontology = RDFGraph()
    lower = URIRef(Q + "hasLowerBound")
    upper = URIRef(Q + "hasUpperBound")
    ontology.add((lower, RDF.type, OWL.FunctionalProperty))
    ontology.add((upper, RDF.type, OWL.FunctionalProperty))
    graph = RDFGraph()
    subject = URIRef(CD + "range1")
    endpoint = URIRef(CD + "value1")
    graph.add((subject, lower, endpoint))
    graph.add((subject, upper, endpoint))

    report = validate_aggregated_facts(graph, ontology, fact_namespaces=[CD])

    kinds = [finding.kind for finding in report.findings]
    assert FactsValidationFindingKind.DEGENERATE_COREFERENCE in kinds


def test_fact_namespace_scoping_excludes_ontology_subjects() -> None:
    graph = RDFGraph()
    outside = URIRef(Q + "SomeClassIndividual")
    predicate = URIRef(Q + "numericValue")
    graph.add((outside, predicate, Literal("1", datatype=XSD.integer)))
    graph.add((outside, predicate, Literal("2", datatype=XSD.integer)))

    report = validate_aggregated_facts(graph, None, fact_namespaces=[CD])
    assert not report.findings


def test_clean_graph_yields_no_findings() -> None:
    graph = RDFGraph()
    subject = URIRef(CD + "q1")
    graph.add((subject, RDF.type, URIRef(Q + "QuantityValue")))
    graph.add((subject, RDFS.label, Literal("redshift")))
    graph.add(
        (subject, URIRef(Q + "numericValue"), Literal("12.5", datatype=XSD.double))
    )
    graph.add((subject, URIRef(Q + "unit"), URIRef(Q + "MilliEV")))

    report = validate_aggregated_facts(graph, None, fact_namespaces=[CD])
    assert not report.findings


# --- merge vetoes ------------------------------------------------------------


def _conflicting_alias_units() -> list[ContentUnit]:
    """A/B/C share one label; A and C carry conflicting numeric literals.

    Pairwise guards reject A-C, but A-B and B-C are each mergeable, so the
    union-find transitively unites all three — the exact gap the gate closes.
    """
    ttl_a = f"""
    @prefix cd: <{CD}> .
    @prefix q: <{Q}> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
    cd:sample_x a q:Sample ; rdfs:label "sample x" ;
        q:numericValue "12.5"^^xsd:double .
    """
    ttl_b = f"""
    @prefix cd: <{CD}> .
    @prefix q: <{Q}> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    cd:SampleX a q:Sample ; rdfs:label "sample x" .
    """
    ttl_c = f"""
    @prefix cd: <{CD}> .
    @prefix q: <{Q}> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
    cd:Sample_X a q:Sample ; rdfs:label "sample x" ;
        q:numericValue "96"^^xsd:double .
    """
    return [_fact_unit(0, ttl_a), _fact_unit(1, ttl_b), _fact_unit(2, ttl_c)]


def test_transitive_merge_slips_guards_and_vetoes_split_it(monkeypatch) -> None:
    aggregator = _normal_form_aggregator(monkeypatch)
    result = aggregator.aggregate_graphs(
        _conflicting_alias_units(), ontology_graph=RDFGraph()
    )

    # The transitive union merged all three aliases into one node …
    assert len(result.merged_clusters) == 1
    members = next(iter(result.merged_clusters.values()))
    assert len(members) == 3
    # … which now carries both conflicting values. Final URIs live under the
    # document namespace, so scope the check there as the gate node does.
    scope = [CD, "https://x.org/doc/1"]
    report = validate_aggregated_facts(result.graph, None, fact_namespaces=scope)
    assert report.error_findings

    vetoes = {
        frozenset((URIRef(left), URIRef(right)))
        for index, left in enumerate(members)
        for right in members[index + 1 :]
    }
    repaired = aggregator.aggregate_graphs(
        _conflicting_alias_units(), ontology_graph=RDFGraph(), merge_vetoes=vetoes
    )
    assert not repaired.merged_clusters
    report = validate_aggregated_facts(repaired.graph, None, fact_namespaces=scope)
    assert not report.error_findings


def test_build_merged_clusters_spans_doc_bases() -> None:
    """One canonical rendered under two doc bases keys the FULL cluster twice.

    A veto on either document's flagged final URI must dissolve the whole
    merge decision, not just that document's half.
    """
    entity_doc1 = URIRef(CD + "sample_x")
    entity_doc2 = URIRef("https://y.org/doc/2/facts/sample_x")
    canonical = entity_doc1
    identity_mapping = {entity_doc1: canonical, entity_doc2: canonical}
    final_doc1 = URIRef(CD + "SampleX")
    final_doc2 = URIRef("https://y.org/doc/2/facts/SampleX")
    final_mapping = {entity_doc1: final_doc1, entity_doc2: final_doc2}

    clusters = build_merged_clusters(final_mapping, identity_mapping)

    full_cluster = sorted([str(entity_doc1), str(entity_doc2)])
    assert clusters[str(final_doc1)] == full_cluster
    assert clusters[str(final_doc2)] == full_cluster


def test_build_merged_clusters_skips_singletons() -> None:
    entity = URIRef(CD + "solo")
    clusters = build_merged_clusters({entity: URIRef(CD + "Solo")}, {entity: entity})
    assert clusters == {}


def test_vetoes_from_findings_builds_full_cluster_pairs() -> None:
    finding = FactsValidationFinding(
        kind=FactsValidationFindingKind.SUSPECT_MULTI_VALUE,
        message="conflict",
        subject=CD + "merged",
    )
    clusters = {
        CD + "merged": [CD + "a", CD + "b", CD + "c"],
        CD + "other": [CD + "d", CD + "e"],
    }
    vetoes = _vetoes_from_findings([finding], clusters)
    assert vetoes == {
        frozenset((URIRef(CD + "a"), URIRef(CD + "b"))),
        frozenset((URIRef(CD + "a"), URIRef(CD + "c"))),
        frozenset((URIRef(CD + "b"), URIRef(CD + "c"))),
    }


# --- VALIDATE_FACTS node -----------------------------------------------------


def _fake_tools(aggregator: EmbeddingBasedAggregator, **overrides) -> ToolBox:
    facts_validation = FactsValidationConfig(**overrides)
    return cast(
        ToolBox,
        SimpleNamespace(
            aggregator=aggregator,
            config=SimpleNamespace(
                get_tool_config=lambda: SimpleNamespace(
                    facts_validation=facts_validation
                )
            ),
        ),
    )


def test_validate_facts_node_repairs_transitive_bad_merge(monkeypatch) -> None:
    aggregator = _normal_form_aggregator(monkeypatch)
    tools = _fake_tools(aggregator)
    state = AgentState()
    state.current_domain = "https://x.org"
    state.doc_hid = "1"
    state.facts_units = _conflicting_alias_units()

    make_merge_facts_node(tools)(state)
    assert state.aggregation_clusters  # the bad merge happened

    make_validate_facts_node(tools)(state)

    assert state.retrieval_metrics["facts_merge_repair_passes"] == 1
    assert state.retrieval_metrics["facts_validation_errors"] == 0
    assert not [
        finding
        for finding in state.facts_validation_findings
        if finding.severity == "error"
    ]
    values = {
        float(str(obj))
        for obj in state.aggregated_facts.objects(None, URIRef(Q + "numericValue"))
    }
    assert values == {12.5, 96.0}


def test_validate_facts_node_zero_passes_records_findings_only(monkeypatch) -> None:
    aggregator = _normal_form_aggregator(monkeypatch)
    tools = _fake_tools(aggregator, merge_repair_passes=0)
    state = AgentState()
    state.current_domain = "https://x.org"
    state.doc_hid = "1"
    state.facts_units = _conflicting_alias_units()

    make_merge_facts_node(tools)(state)
    before = state.aggregated_facts
    make_validate_facts_node(tools)(state)

    assert state.retrieval_metrics["facts_merge_repair_passes"] == 0
    assert cast(int, state.retrieval_metrics["facts_validation_errors"]) >= 1
    assert state.aggregated_facts is before


def test_validate_facts_node_noop_on_empty_state() -> None:
    tools = _fake_tools(EmbeddingBasedAggregator())
    state = AgentState()
    make_validate_facts_node(tools)(state)
    assert state.retrieval_metrics.get("facts_validation_errors") is None


def test_repair_pass_republishes_the_guard_count_not_the_veto_count(
    monkeypatch,
) -> None:
    """``facts_rejected_merges`` must stay the aggregator's guard count.

    The un-merge pass used to overwrite it with ``len(vetoes)`` -- a different
    quantity, already published as ``facts_merge_vetoes``. The two coincide on
    a simple fixture (a vetoed pair is also a guard rejection), so the count is
    forced apart here: whenever the aggregator rejects pairs the vetoes did not
    name, the veto count under-reports the graph that is served.
    """
    aggregator = _normal_form_aggregator(monkeypatch)
    real_postprocess = aggregator.postprocess_facts_units
    sentinel = 41

    def counted(*args, **kwargs):
        result = real_postprocess(*args, **kwargs)
        if kwargs.get("merge_vetoes"):
            result.rejected_merge_count = sentinel
        return result

    monkeypatch.setattr(aggregator, "postprocess_facts_units", counted)

    tools = _fake_tools(aggregator)
    state = AgentState()
    state.current_domain = "https://x.org"
    state.doc_hid = "1"
    state.facts_units = _conflicting_alias_units()

    make_merge_facts_node(tools)(state)
    make_validate_facts_node(tools)(state)

    metrics = state.retrieval_metrics
    assert metrics[RetrievalMetric.FACTS_MERGE_REPAIR_PASSES] == 1
    assert cast(int, metrics[RetrievalMetric.FACTS_MERGE_VETOES]) > 0
    assert metrics[RetrievalMetric.FACTS_REJECTED_MERGES] == sentinel


def test_both_entry_paths_write_the_same_gate_metric_keys(monkeypatch) -> None:
    """The graph pipeline and the unit pipeline share one metric set.

    They ran the same gate through two hand-maintained metric blocks, so a
    counter added to one path was silently absent from the other and batch
    dumps stopped being comparable across entry paths. Merge counters are
    excluded: they have no meaning for a single unit.
    """
    merge_only = {
        RetrievalMetric.FACTS_MERGE_REPAIR_PASSES,
        RetrievalMetric.FACTS_MERGE_VETOES,
        RetrievalMetric.FACTS_MERGE_REPAIRS_REJECTED,
        RetrievalMetric.FACTS_REJECTED_MERGES,
    }

    def gate_keys(run) -> set[str]:
        aggregator = _normal_form_aggregator(monkeypatch)
        tools = _fake_tools(aggregator)
        state = AgentState()
        state.current_domain = "https://x.org"
        state.doc_hid = "1"
        state.facts_units = _conflicting_alias_units()
        make_merge_facts_node(tools)(state)
        state.retrieval_metrics.clear()
        run(state, tools)
        return set(state.retrieval_metrics) - merge_only

    graph_keys = gate_keys(lambda state, tools: make_validate_facts_node(tools)(state))
    unit_keys = gate_keys(
        lambda state, tools: validate_unit_pipeline_facts(state, RDFGraph(), tools)
    )
    assert graph_keys == unit_keys


def test_dangling_reference_reported_as_warning() -> None:
    graph = RDFGraph()
    graph.parse(
        data=f"""
@prefix cd: <{CD}> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix q: <{Q}> .

cd:observation_1 a q:Observation ;
    rdfs:label "obs"@en ;
    q:hasCondition cd:condition_never_declared .
""",
        format="turtle",
    )
    report = validate_aggregated_facts(graph, None, fact_namespaces=[DEFAULT_IRI])
    dangling = [
        finding
        for finding in report.findings
        if finding.kind is FactsValidationFindingKind.DANGLING_REFERENCE
    ]
    assert len(dangling) == 1
    assert dangling[0].severity == "warning"
    assert "condition_never_declared" in dangling[0].subject


def test_described_and_external_objects_are_not_dangling() -> None:
    graph = RDFGraph()
    graph.parse(
        data=f"""
@prefix cd: <{CD}> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix q: <{Q}> .

cd:observation_1 a q:Observation ;
    q:hasCondition cd:condition_1 ;
    q:hasUnit <http://qudt.org/vocab/unit/NanoM> .
cd:condition_1 rdfs:label "77 K"@en .
""",
        format="turtle",
    )
    report = validate_aggregated_facts(graph, None, fact_namespaces=[DEFAULT_IRI])
    assert not any(
        finding.kind is FactsValidationFindingKind.DANGLING_REFERENCE
        for finding in report.findings
    )
