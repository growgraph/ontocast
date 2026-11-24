"""Tests for the post-aggregation facts validation gate.

The gate is defense-in-depth behind the merge guards: it detects invariant
violations in the aggregated graph (notably transitive union-find merges the
pairwise guards cannot block) and repairs them deterministically by vetoing
the offending cluster's pairs and re-aggregating.
"""

from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import OWL, RDF, RDFS, Literal, URIRef
from rdflib.namespace import XSD

from ontocast.api.process_helpers import validate_unit_pipeline_facts
from ontocast.config import AggregationConfig, FactsValidationConfig
from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.content_unit import ContentUnit, OutputType
from ontocast.onto.enum import RetrievalMetric
from ontocast.onto.model import (
    FactsValidationFinding,
    FactsValidationFindingKind,
    MergeProvenance,
)
from ontocast.onto.prefixes import PrefixTable
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.stategraph.facts_gate import vetoes_from_findings
from ontocast.stategraph.node_factories import (
    make_merge_facts_node,
    make_validate_facts_node,
)
from ontocast.tool import EmbeddingBasedAggregator
from ontocast.tool.agg.aggregate import build_merged_clusters
from ontocast.tool.agg.merge_provenance import build_merge_provenance
from ontocast.tool.facts_validation import validate_aggregated_facts
from ontocast.toolbox import ToolBox

pytestmark = pytest.mark.unit

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


def _normal_form_aggregator(
    monkeypatch, *, ablate_literal_guard: bool = False
) -> EmbeddingBasedAggregator:
    """Cluster by normal form; ``ablate_literal_guard`` switches the
    literal-conflict guard off, so a bad merge reaches the gate."""
    aggregator = EmbeddingBasedAggregator(AggregationConfig())
    if ablate_literal_guard:
        monkeypatch.setattr(
            aggregator,
            "_have_conflicting_literals",
            lambda left, right, name_lists: False,
        )

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


def test_transitive_merge_blocked_cluster_wide_at_aggregation(monkeypatch) -> None:
    """A vetoed pair must not merge through a chain of accepted edges.

    Pairwise guards reject A-C (conflicting numeric literals); A-B and B-C
    are each mergeable. The union used to chain all three together anyway;
    the distinctness veto now holds cluster-wide, so no cluster unites the
    two conflicting value carriers and the served graph is clean.
    """
    aggregator = _normal_form_aggregator(monkeypatch)
    result = aggregator.aggregate_graphs(
        _conflicting_alias_units(), ontology_graph=RDFGraph()
    )

    for members in result.merged_clusters.values():
        assert not {str(URIRef(CD + "sample_x")), str(URIRef(CD + "Sample_X"))} <= set(
            members
        )
    assert result.rejected_merge_count >= 1
    scope = [CD, "https://x.org/doc/1"]
    report = validate_aggregated_facts(result.graph, None, fact_namespaces=scope)
    assert not report.error_findings


def test_transitive_merge_slips_disabled_guard_and_vetoes_split_it(
    monkeypatch,
) -> None:
    """With the literal-conflict guard ablated, the gate's vetoes still repair.

    The un-merge lever is the backstop for whatever the aggregation-time
    guards cannot see; ablating the guard reproduces the historical
    transitive over-merge so the lever stays covered.
    """
    aggregator = _normal_form_aggregator(monkeypatch, ablate_literal_guard=True)
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
    vetoes = vetoes_from_findings([finding], clusters)
    assert vetoes == {
        frozenset((URIRef(CD + "a"), URIRef(CD + "b"))),
        frozenset((URIRef(CD + "a"), URIRef(CD + "c"))),
        frozenset((URIRef(CD + "b"), URIRef(CD + "c"))),
    }


def test_a_multi_value_finding_leaves_the_objects_merges_alone() -> None:
    """A merged subject collects its members' objects; the objects are innocent."""
    finding = FactsValidationFinding(
        kind=FactsValidationFindingKind.SUSPECT_MULTI_VALUE,
        message="conflict",
        subject=CD + "event",
        predicate=Q + "place",
        values=[CD + "city_a", CD + "city_b"],
    )
    clusters = {
        CD + "event": [CD + "event__u1", CD + "event__u2"],
        CD + "city_a": [CD + "city_a__u1", CD + "city_a__u3"],
    }
    assert vetoes_from_findings([finding], clusters) == {
        frozenset((URIRef(CD + "event__u1"), URIRef(CD + "event__u2")))
    }


def test_a_degenerate_coreference_finding_vetoes_the_collapsed_endpoint() -> None:
    finding = FactsValidationFinding(
        kind=FactsValidationFindingKind.DEGENERATE_COREFERENCE,
        message="collapsed",
        subject=CD + "range",
        values=[CD + "bound"],
    )
    clusters = {CD + "bound": [CD + "low__u1", CD + "high__u1"]}
    assert vetoes_from_findings([finding], clusters) == {
        frozenset((URIRef(CD + "low__u1"), URIRef(CD + "high__u1")))
    }


def _unit(index: int, ttl: str) -> ContentUnit:
    graph = RDFGraph()
    graph.parse(
        data=f"@prefix cd: <{CD}> .\n@prefix q: <{Q}> .\n{ttl}", format="turtle"
    )
    return ContentUnit(
        text=f"unit {index}",
        index=index,
        doc_iri=URIRef(CD + "doc"),
        graph=graph,
        type=OutputType.FACTS,
    )


def test_a_veto_splits_only_the_members_that_disagree() -> None:
    """A member that carried no value on the predicate keeps its merges."""
    finding = FactsValidationFinding(
        kind=FactsValidationFindingKind.FUNCTIONAL_VIOLATION,
        message="two values",
        subject=CD + "Wave",
        predicate=Q + "peak",
        values=["40", "46"],
    )
    clusters = {CD + "Wave": [CD + "wave_a", CD + "wave_b", CD + "wave_c"]}
    units = [
        _unit(0, "cd:wave_a q:peak 40 ."),
        _unit(1, "cd:wave_b q:peak 46.0 ."),
        _unit(2, 'cd:wave_c q:name "blocking high" .'),
    ]
    assert vetoes_from_findings([finding], clusters, units) == {
        frozenset((URIRef(CD + "wave_a"), URIRef(CD + "wave_b")))
    }


def test_alias_compatible_string_values_are_not_split() -> None:
    finding = FactsValidationFinding(
        kind=FactsValidationFindingKind.SUSPECT_MULTI_VALUE,
        message="names",
        subject=CD + "Person",
        predicate=Q + "name",
    )
    clusters = {CD + "Person": [CD + "p1", CD + "p2", CD + "p3"]}
    units = [
        _unit(0, 'cd:p1 q:name "Mr Beer" .'),
        _unit(1, 'cd:p2 q:name "Mr Karlheinz Beer" .'),
        _unit(2, 'cd:p3 q:name "Mrs Palm" .'),
    ]
    assert vetoes_from_findings([finding], clusters, units) == {
        frozenset((URIRef(CD + "p1"), URIRef(CD + "p3"))),
        frozenset((URIRef(CD + "p2"), URIRef(CD + "p3"))),
    }


def test_a_degenerate_endpoint_is_split_by_the_predicate_reaching_it() -> None:
    finding = FactsValidationFinding(
        kind=FactsValidationFindingKind.DEGENERATE_COREFERENCE,
        message="collapsed",
        subject=CD + "Range",
        predicate=f"{Q}lower, {Q}upper",
        values=[CD + "Bound"],
    )
    clusters = {CD + "Bound": [CD + "low", CD + "high", CD + "low_again"]}
    units = [
        _unit(0, "cd:range q:lower cd:low ; q:upper cd:high ."),
        _unit(1, "cd:range2 q:lower cd:low_again ."),
    ]
    assert vetoes_from_findings([finding], clusters, units) == {
        frozenset((URIRef(CD + "low"), URIRef(CD + "high"))),
        frozenset((URIRef(CD + "low_again"), URIRef(CD + "high"))),
    }


def test_an_unattributable_finding_vetoes_the_whole_cluster() -> None:
    finding = FactsValidationFinding(
        kind=FactsValidationFindingKind.FUNCTIONAL_VIOLATION,
        message="two values",
        subject=CD + "Wave",
        predicate=Q + "peak",
    )
    clusters = {CD + "Wave": [CD + "wave_a", CD + "wave_b"]}
    units = [_unit(0, "cd:wave_a q:peak 40, 46 .")]
    assert vetoes_from_findings([finding], clusters, units) == {
        frozenset((URIRef(CD + "wave_a"), URIRef(CD + "wave_b")))
    }


# --- VALIDATE_FACTS node -----------------------------------------------------


def _fake_tools(
    aggregator: EmbeddingBasedAggregator, *, shapes=None, **overrides
) -> ToolBox:
    facts_validation = FactsValidationConfig(**overrides)
    return cast(
        ToolBox,
        SimpleNamespace(
            prefix_table=lambda: PrefixTable({}, {}),
            aggregator=aggregator,
            shapes_catalog=SimpleNamespace(graph=lambda: shapes),
            shapes_prompt_contract=lambda: ("", (), False),
            shapes_chapter_terms=lambda context_terms: frozenset(),
            ontology_manager=SimpleNamespace(
                catalog_terms=lambda: set(), get_head_by_iri=lambda iri: None
            ),
            get_atomic_tools=lambda: SimpleNamespace(value_classes=lambda: frozenset()),
            config=SimpleNamespace(
                get_tool_config=lambda: SimpleNamespace(
                    facts_validation=facts_validation
                )
            ),
        ),
    )


def test_validate_facts_node_repairs_transitive_bad_merge(monkeypatch) -> None:
    # Guard ablated so the historical over-merge reaches the gate at all —
    # with default guards the aggregator blocks it upstream.
    aggregator = _normal_form_aggregator(monkeypatch, ablate_literal_guard=True)
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
    aggregator = _normal_form_aggregator(monkeypatch, ablate_literal_guard=True)
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
    aggregator = _normal_form_aggregator(monkeypatch, ablate_literal_guard=True)
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


# ---------------------------------------------------------------------------
# Merge provenance for SUSPECT_MULTI_VALUE
# ---------------------------------------------------------------------------


def _joint_agents_graph() -> tuple[RDFGraph, URIRef, URIRef]:
    """One application with two agents, plus single-agent subjects around it.

    The neighbours are what make the predicate "dominantly single-valued", so
    the IRI branch fires on the two-object subject.
    """
    graph = RDFGraph()
    subject = URIRef(CD + "application1")
    predicate = URIRef("https://schema.org/agent")
    graph.add((subject, predicate, URIRef(CD + "personBeer")))
    graph.add((subject, predicate, URIRef(CD + "personRegan")))
    for index in range(3):
        graph.add(
            (
                URIRef(f"{CD}other{index}"),
                predicate,
                URIRef(f"{CD}soleAgent{index}"),
            )
        )
    return graph, subject, predicate


def _two_abbreviations_graph() -> tuple[RDFGraph, URIRef, URIRef]:
    """A league with an English and a French abbreviation, among single-abbreviation neighbours."""
    graph = RDFGraph()
    subject = URIRef(CD + "league")
    predicate = URIRef("http://dbpedia.org/ontology/abbreviation")
    graph.add((subject, predicate, Literal("CFL")))
    graph.add((subject, predicate, Literal("LCF")))
    for index, name in enumerate(("IRFU", "WIFU", "CRU")):
        graph.add((URIRef(f"{CD}union{index}"), predicate, Literal(name)))
    return graph, subject, predicate


def _two_quantities_graph() -> tuple[RDFGraph, URIRef, URIRef]:
    graph = RDFGraph()
    subject = URIRef(CD + "q1")
    predicate = URIRef(Q + "numericValue")
    graph.add((subject, predicate, Literal("12.5", datatype=XSD.double)))
    graph.add((subject, predicate, Literal("96", datatype=XSD.double)))
    return graph, subject, predicate


_BRANCHES = pytest.mark.parametrize(
    "build", [_joint_agents_graph, _two_abbreviations_graph, _two_quantities_graph]
)


def _suspects(report, subject: URIRef) -> list[FactsValidationFinding]:
    return [
        finding
        for finding in report.findings
        if finding.kind == FactsValidationFindingKind.SUSPECT_MULTI_VALUE
        and finding.subject == str(subject)
    ]


@_BRANCHES
def test_multi_value_is_an_error_without_provenance(build) -> None:
    """No evidence supplied: every branch reports an error."""
    graph, subject, _ = build()

    report = validate_aggregated_facts(graph, None, fact_namespaces=[CD])

    assert [finding.severity for finding in _suspects(report, subject)] == ["error"]


@_BRANCHES
def test_values_one_subject_was_written_with_are_only_a_warning(build) -> None:
    """No merge brought them together, so the un-merge repair must not act.

    An error here would dissolve a correct merge elsewhere in the cluster.
    """
    graph, subject, _ = build()

    report = validate_aggregated_facts(
        graph, None, fact_namespaces=[CD], merge_provenance=MergeProvenance()
    )

    assert [finding.severity for finding in _suspects(report, subject)] == ["warning"]
    assert not report.error_findings


@_BRANCHES
def test_values_a_merge_joined_stay_an_error(build) -> None:
    graph, subject, predicate = build()

    report = validate_aggregated_facts(
        graph,
        None,
        fact_namespaces=[CD],
        merge_provenance=MergeProvenance(
            joined_values=[(str(subject), str(predicate))]
        ),
    )

    assert [finding.severity for finding in _suspects(report, subject)] == ["error"]


def test_only_merge_joined_multi_values_are_errors_by_default(monkeypatch) -> None:
    monkeypatch.delenv("FACTS_SUSPECT_MULTI_VALUE_REQUIRE_MERGE", raising=False)
    assert FactsValidationConfig().suspect_multi_value_require_merge is True


def _winner_graph() -> tuple[RDFGraph, URIRef, URIRef]:
    """A final whose winner is also a competitor, among one-winner finals."""
    graph = RDFGraph()
    winner = URIRef("https://schema.org/winner")
    competitor = URIRef("https://schema.org/competitor")
    final = URIRef(CD + "final1950")
    team = URIRef(CD + "cityCollege")
    graph.add((final, winner, team))
    graph.add((final, competitor, team))
    for index in range(3):
        other = URIRef(f"{CD}final{index}")
        graph.add((other, winner, URIRef(f"{CD}champion{index}")))
        graph.add((other, competitor, URIRef(f"{CD}entrant{index}")))
    return graph, final, team


def _degenerate(report, subject: URIRef) -> list[FactsValidationFinding]:
    return [
        finding
        for finding in report.findings
        if finding.kind == FactsValidationFindingKind.DEGENERATE_COREFERENCE
        and finding.subject == str(subject)
    ]


def test_a_link_one_subject_wrote_twice_is_not_a_degenerate_merge() -> None:
    """The winner of a final is also its competitor: one reading, no merge."""
    graph, final, _ = _winner_graph()

    report = validate_aggregated_facts(
        graph, None, fact_namespaces=[CD], merge_provenance=MergeProvenance()
    )

    assert [f.severity for f in _degenerate(report, final)] == ["warning"]
    assert not report.error_findings


def test_a_link_a_merge_made_stays_a_degenerate_error() -> None:
    graph, final, team = _winner_graph()

    report = validate_aggregated_facts(
        graph,
        None,
        fact_namespaces=[CD],
        merge_provenance=MergeProvenance(joined_links=[(str(final), str(team))]),
    )

    assert [f.severity for f in _degenerate(report, final)] == ["error"]


def test_merge_provenance_records_links_a_merge_made() -> None:
    """Two bounds merged into one node: the link to it is merge-made."""
    lower, upper = "https://x.org/onto#lowerBound", "https://x.org/onto#upperBound"
    range_node = URIRef(CD + "range")
    unit = _fact_unit(
        0,
        f"<{range_node}> <{lower}> <{CD}low> ; <{upper}> <{CD}high> .\n"
        f"<{CD}final> <https://schema.org/winner> <{CD}team> ;"
        f" <https://schema.org/competitor> <{CD}team> .",
    )
    merged = URIRef(CD + "bound")

    links = build_merge_provenance(
        [unit], {URIRef(CD + "low"): merged, URIRef(CD + "high"): merged}
    ).joined_links

    assert (str(range_node), str(merged)) in links
    assert (CD + "final", CD + "team") not in links


def test_merge_provenance_separates_the_two_cases() -> None:
    subject = URIRef(CD + "application1")
    predicate = "https://schema.org/agent"
    shared = URIRef(CD + "sharedSubject")

    joint_unit = _fact_unit(
        0,
        f"""
        <{subject}> <{predicate}> <{CD}personBeer> , <{CD}personRegan> .
        """,
    )
    first = _fact_unit(1, f"<{shared}> <{predicate}> <{CD}alpha> .")
    second = _fact_unit(2, f"<{shared}> <{predicate}> <{CD}beta> .")

    pairs = build_merge_provenance([joint_unit, first, second], {}).joined_values

    # Two objects one subject was written with: not merge-created.
    assert (str(subject), predicate) not in pairs
    # One object each from two units: a merge could have produced this.
    assert (str(shared), predicate) in pairs


def test_merge_provenance_canonicalizes_subjects() -> None:
    """Keys must match the post-merge graph the gate actually validates."""
    predicate = "https://schema.org/agent"
    raw_left = URIRef(CD + "app_a")
    raw_right = URIRef(CD + "app_b")
    canonical = URIRef(CD + "application")

    left = _fact_unit(0, f"<{raw_left}> <{predicate}> <{CD}alpha> .")
    right = _fact_unit(1, f"<{raw_right}> <{predicate}> <{CD}beta> .")

    pairs = build_merge_provenance(
        [left, right], {raw_left: canonical, raw_right: canonical}
    ).joined_values
    assert (str(canonical), predicate) in pairs


def test_a_merge_inside_one_unit_joins_values() -> None:
    """Two subjects of one unit merged into one: their values were joined."""
    predicate = "http://dbpedia.org/ontology/abbreviation"
    raw_left = URIRef(CD + "cfl")
    raw_right = URIRef(CD + "league")
    canonical = URIRef(CD + "canadianFootballLeague")
    unit = _fact_unit(
        0, f'<{raw_left}> <{predicate}> "CFL" . <{raw_right}> <{predicate}> "AFL" .'
    )

    pairs = build_merge_provenance(
        [unit], {raw_left: canonical, raw_right: canonical}
    ).joined_values
    assert (str(canonical), predicate) in pairs


def test_a_source_repeating_one_value_does_not_join_the_others() -> None:
    """One unit wrote "CFL" and "LCF"; another wrote "CFL" again.

    Every distinct value is still held by the first source, so the conflict
    the gate sees predates the merge.
    """
    predicate = "http://dbpedia.org/ontology/abbreviation"
    raw_left = URIRef(CD + "cfl_u0")
    raw_right = URIRef(CD + "cfl_u1")
    canonical = URIRef(CD + "canadianFootballLeague")
    first = _fact_unit(0, f'<{raw_left}> <{predicate}> "CFL" , "LCF" .')
    second = _fact_unit(1, f'<{raw_right}> <{predicate}> "cfl"@en .')

    pairs = build_merge_provenance(
        [first, second], {raw_left: canonical, raw_right: canonical}
    ).joined_values
    assert (str(canonical), predicate) not in pairs


def test_numbers_are_compared_by_value() -> None:
    """``"16"`` and ``16.0`` from two sources are one value: nothing was joined."""
    predicate = Q + "numericValue"
    raw_left = URIRef(CD + "q_u0")
    raw_right = URIRef(CD + "q_u1")
    canonical = URIRef(CD + "q")
    first = _fact_unit(0, f'<{raw_left}> <{predicate}> "16" , 18 .')
    second = _fact_unit(1, f"<{raw_right}> <{predicate}> 16.0 .")

    pairs = build_merge_provenance(
        [first, second], {raw_left: canonical, raw_right: canonical}
    ).joined_values
    assert (str(canonical), predicate) not in pairs


def test_the_gate_lists_an_isolated_statement_it_keeps(monkeypatch) -> None:
    """An instance holding a structured value but linked to nothing is kept and listed."""
    tools = _fake_tools(_normal_form_aggregator(monkeypatch))
    state = AgentState()
    state.current_domain = "https://x.org"
    state.doc_hid = "1"
    state.facts_units = [
        _fact_unit(
            0,
            f"""
@prefix cd: <{CD}> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix q: <{Q}> .
cd:cup a q:Tournament ; rdfs:label "Spring Cup" ; q:heldAt cd:hall .
cd:hall a q:Venue ; rdfs:label "Harbour Hall" .
cd:limit a q:Rule ; rdfs:label "age limit" ; q:amount [ q:value 23 ] .
""",
        )
    ]

    make_merge_facts_node(tools)(state)
    make_validate_facts_node(tools)(state)

    isolated = state.facts_isolated_report
    assert [(item.iri.rsplit("/", 1)[-1], item.kind) for item in isolated] == [
        ("limit", "holds_value")
    ]
    assert len(isolated) == state.retrieval_metrics[RetrievalMetric.FACTS_ISOLATED]
