"""Unit-scoped fact IRIs: a cross-unit name collision is a merge decision.

Units mint instance IRIs independently, so two units that both wrote
``cd:temperature_value`` used to fuse into one node carrying both values
before any merge guard could look at them -- and the validation gate's
un-merge repair only dissolves clusters of two or more source IRIs, so a
singleton collision could never be split. Scoping suffixes each minted IRI
with its unit index: the pair reaches clustering and the guards as two
entities, and the served graph carries whichever decision they earn.
"""

from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import OWL, RDF, Literal, URIRef

from ontocast.config import AggregationConfig, FactsValidationConfig
from ontocast.onto.constants import DEFAULT_IRI, PROV, RDF_REIFIES
from ontocast.onto.content_unit import ContentUnit, OutputType
from ontocast.onto.iri_policy import split_namespace_local
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.stategraph.node_factories import (
    make_merge_facts_node,
    make_validate_facts_node,
)
from ontocast.tool import EmbeddingBasedAggregator
from ontocast.tool.agg.unit_scope import (
    UNIT_SCOPE_MARKER,
    scope_fact_iris,
    scope_local_name,
    strip_unit_scope,
    unit_scope_index,
    unscoped_iri,
)
from ontocast.toolbox import ToolBox

pytestmark = pytest.mark.unit

CD = DEFAULT_IRI
DOC = "https://x.org/doc/1"
Q = "https://x.org/schema#"
QUDT = "http://qudt.org/schema/qudt/"
UNIT = "http://qudt.org/vocab/unit/"
NUMERIC_VALUE = URIRef(f"{QUDT}numericValue")
QUDT_UNIT = URIRef(f"{QUDT}unit")
HAS_QUANTITY = URIRef(f"{Q}hasQuantity")
PREFIXES = f"""
@prefix cd: <{CD}> .
@prefix doc: <{DOC}/> .
@prefix q: <{Q}> .
@prefix qudt: <{QUDT}> .
@prefix unit: <{UNIT}> .
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


def _temperature_unit(index: int, value: str, unit_name: str) -> ContentUnit:
    """A unit that mints ``cd:temperature_value`` for its own measurement."""
    return _unit(
        index,
        f"""
        cd:sample_a a q:Sample ; rdfs:label "Sample A" ;
            q:hasQuantity cd:temperature_value .
        cd:temperature_value a qudt:QuantityValue ; rdfs:label "temperature" ;
            qudt:numericValue "{value}"^^xsd:decimal ;
            qudt:unit unit:{unit_name} .
        """,
    )


def _normal_form_aggregator(
    monkeypatch: pytest.MonkeyPatch, **overrides
) -> EmbeddingBasedAggregator:
    """Cluster by normal form -- what a model does with two identical mentions."""
    aggregator = EmbeddingBasedAggregator(AggregationConfig(**overrides))

    def cluster_by_normal_form(representations):
        clusters: dict[str, list[URIRef]] = {}
        for entity, representation in representations.items():
            clusters.setdefault(representation.normal_form, []).append(entity)
        return list(clusters.values()), {}

    monkeypatch.setattr(
        aggregator.clusterer, "cluster_entities", cluster_by_normal_form
    )
    return aggregator


def _iris(graph: RDFGraph) -> set[URIRef]:
    return {term for triple in graph for term in triple if isinstance(term, URIRef)}


def _scoped(graph: RDFGraph) -> set[URIRef]:
    return {iri for iri in _iris(graph) if unit_scope_index(str(iri)) is not None}


def _values_by_subject(graph: RDFGraph) -> dict[URIRef, set[float]]:
    values: dict[URIRef, set[float]] = {}
    for subject, _, obj in graph.triples((None, NUMERIC_VALUE, None)):
        if isinstance(subject, URIRef) and isinstance(obj, Literal):
            values.setdefault(subject, set()).add(float(obj))
    return values


# --- suffix helpers ----------------------------------------------------------


def test_scope_and_strip_round_trip() -> None:
    scoped = scope_local_name("temperature_value", 3)
    assert scoped == f"temperature_value{UNIT_SCOPE_MARKER}3"
    assert strip_unit_scope(scoped) == "temperature_value"
    assert unit_scope_index(scoped) == 3
    assert unit_scope_index("temperature_value") is None
    assert strip_unit_scope("temperature_value") == "temperature_value"


def test_rescoping_replaces_instead_of_stacking() -> None:
    assert scope_local_name(scope_local_name("x", 1), 2) == "x__u2"
    assert scope_local_name(scope_local_name("x", 1), 1) == "x__u1"


def test_strip_accepts_full_iris() -> None:
    scoped = URIRef(f"{CD}sample_a__u7")
    assert strip_unit_scope(str(scoped)) == f"{CD}sample_a"
    assert unscoped_iri(scoped) == URIRef(f"{CD}sample_a")
    plain = URIRef(f"{CD}sample_a")
    assert unscoped_iri(plain) is plain


def test_suffix_stays_inside_the_local_name() -> None:
    graph = _graph('cd:sample_a a q:Sample ; rdfs:label "Sample A" .')
    mapping = scope_fact_iris(graph, 4, [CD, DOC])

    scoped = mapping[URIRef(f"{CD}sample_a")]
    assert split_namespace_local(str(scoped)) == (CD, "sample_a__u4")


# --- scope_fact_iris ---------------------------------------------------------


def test_scope_fact_iris_rewrites_instances_and_keeps_vocabulary() -> None:
    graph = _graph(
        """
        cd:hasPart a owl:ObjectProperty .
        cd:sample_a a cd:Sample ; rdfs:label "Sample A" ;
            cd:hasPart cd:layer_1 ;
            q:hasQuantity cd:temperature_value ;
            q:seeAlso <https://other.org/thing> .
        cd:temperature_value qudt:numericValue "6"^^xsd:decimal .
        doc:figure_1 rdfs:label "Figure 1" .
        """
    )
    before = len(graph)

    mapping = scope_fact_iris(graph, 2, [CD, DOC])

    assert set(mapping) == {
        URIRef(f"{CD}sample_a"),
        URIRef(f"{CD}layer_1"),
        URIRef(f"{CD}temperature_value"),
        URIRef(f"{DOC}/figure_1"),
    }
    assert all(unit_scope_index(str(target)) == 2 for target in mapping.values())
    assert len(graph) == before
    iris = _iris(graph)
    # Vocabulary the unit refers to keeps its name in every position.
    assert URIRef(f"{CD}Sample") in iris
    assert URIRef(f"{CD}hasPart") in iris
    assert (URIRef(f"{CD}hasPart"), RDF.type, OWL.ObjectProperty) in graph
    assert URIRef("https://other.org/thing") in iris
    # Instances are rewritten in every position they occupy.
    assert not (set(mapping) & iris)
    assert (
        URIRef(f"{CD}sample_a__u2"),
        URIRef(f"{CD}hasPart"),
        URIRef(f"{CD}layer_1__u2"),
    ) in graph
    assert (
        URIRef(f"{CD}temperature_value__u2"),
        NUMERIC_VALUE,
        Literal("6", datatype=URIRef("http://www.w3.org/2001/XMLSchema#decimal")),
    ) in graph


def test_scope_fact_iris_is_idempotent_for_the_same_unit() -> None:
    graph = _graph('cd:sample_a rdfs:label "Sample A" .')
    scope_fact_iris(graph, 1, [CD])
    snapshot = set(graph)

    assert scope_fact_iris(graph, 1, [CD]) == {}
    assert set(graph) == snapshot


def test_scope_fact_iris_ignores_foreign_namespaces_and_empty_input() -> None:
    graph = _graph('<https://other.org/x> rdfs:label "x" .')
    assert scope_fact_iris(graph, 0, [CD, ""]) == {}
    assert scope_fact_iris(graph, 0, []) == {}
    assert scope_fact_iris(RDFGraph(), 0, [CD]) == {}


# --- aggregation -------------------------------------------------------------


def test_same_local_name_with_conflicting_literals_stays_apart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregator = _normal_form_aggregator(monkeypatch)
    units = [_temperature_unit(0, "6", "K"), _temperature_unit(1, "10", "DEG_C")]

    result = aggregator.postprocess_facts_units(units, RDFGraph())

    values = _values_by_subject(result.graph)
    assert sorted(values.values(), key=sorted) == [{6.0}, {10.0}]
    assert len(values) == 2
    assert result.rejected_merge_count >= 1
    assert not _scoped(result.graph)


def test_without_scoping_the_collision_fuses_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The knob off reproduces name-keyed fusion: one node, both values."""
    aggregator = _normal_form_aggregator(monkeypatch, unit_scoped_fact_iris=False)
    units = [_temperature_unit(0, "6", "K"), _temperature_unit(1, "10", "DEG_C")]

    result = aggregator.postprocess_facts_units(units, RDFGraph())

    assert list(_values_by_subject(result.graph).values()) == [{6.0, 10.0}]


def test_same_label_and_compatible_content_still_combine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The subject both units describe merges; its per-unit quantities do not.

    ``sample_a`` points at a different scoped quantity in each unit. That is
    the shape scoping creates for every subject mentioned more than once, so
    it must not read as a functional-object conflict on the subject.
    """
    aggregator = _normal_form_aggregator(monkeypatch)
    units = [_temperature_unit(0, "6", "K"), _temperature_unit(1, "10", "DEG_C")]

    result = aggregator.postprocess_facts_units(units, RDFGraph())

    samples = set(result.graph.subjects(RDF.type, URIRef(f"{Q}Sample")))
    assert len(samples) == 1
    sample = samples.pop()
    assert len(set(result.graph.objects(sample, HAS_QUANTITY))) == 2
    members = result.merged_clusters[str(sample)]
    assert {unit_scope_index(member) for member in members} == {0, 1}
    assert {strip_unit_scope(member) for member in members} == {f"{CD}sample_a"}


def test_same_local_name_enrichment_across_units_combines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregator = _normal_form_aggregator(monkeypatch)
    units = [
        _unit(0, 'cd:sample_a a q:Sample ; rdfs:label "Sample A" ; q:thickness 100 .'),
        _unit(1, 'cd:sample_a a q:Sample ; rdfs:label "Sample A" ; q:growth "MBE" .'),
    ]

    result = aggregator.postprocess_facts_units(units, RDFGraph())

    samples = set(result.graph.subjects(RDF.type, URIRef(f"{Q}Sample")))
    assert len(samples) == 1
    sample = samples.pop()
    assert (sample, URIRef(f"{Q}thickness"), Literal(100)) in result.graph
    assert (sample, URIRef(f"{Q}growth"), Literal("MBE")) in result.graph
    assert not _scoped(result.graph)


def test_final_iris_and_provenance_carry_no_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregator = _normal_form_aggregator(monkeypatch)
    units = [
        _temperature_unit(0, "6", "K"),
        _unit(1, 'doc:figure_1 rdfs:label "Figure 1" ; q:shows cd:sample_a .'),
        _temperature_unit(2, "10", "DEG_C"),
    ]

    result = aggregator.postprocess_facts_units(units, RDFGraph())
    merged = result.graph

    assert not _scoped(merged)
    subjects = set(merged.subjects())
    reifiers = list(merged.subjects(RDF_REIFIES, None))
    assert reifiers
    for reifier in reifiers:
        for quoted in merged.objects(reifier, RDF_REIFIES):
            assert isinstance(quoted, tuple)
            assert quoted[0] in subjects
            assert not any(
                unit_scope_index(str(term)) is not None
                for term in quoted
                if isinstance(term, URIRef)
            )
        sources = set(merged.objects(reifier, PROV.wasDerivedFrom))
        assert sources <= {URIRef(unit.iri_absolute) for unit in units}
    # Alias records name the source IRI a unit wrote, not its scoped form.
    for _, _, alias in merged.triples((None, OWL.sameAs, None)):
        assert unit_scope_index(str(alias)) is None
    # The unit graphs themselves are what aggregation saw: scoped, in place.
    assert all(_scoped(unit.graph) for unit in units)


def test_distinct_final_iris_are_minted_in_unit_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregator = _normal_form_aggregator(monkeypatch)
    units = [_temperature_unit(0, "6", "K"), _temperature_unit(1, "10", "DEG_C")]

    result = aggregator.postprocess_facts_units(units, RDFGraph())

    by_value = {
        next(iter(values)): subject
        for subject, values in _values_by_subject(result.graph).items()
    }
    assert str(by_value[6.0]).endswith("/temperatureValue")
    assert str(by_value[10.0]).endswith("/temperatureValue_1")


# --- the gate can now split what it could not see ----------------------------


def _fake_tools(aggregator: EmbeddingBasedAggregator) -> ToolBox:
    return cast(
        ToolBox,
        SimpleNamespace(
            aggregator=aggregator,
            shapes_catalog=SimpleNamespace(graph=lambda: None),
            config=SimpleNamespace(
                get_tool_config=lambda: SimpleNamespace(
                    facts_validation=FactsValidationConfig()
                )
            ),
        ),
    )


def test_gate_vetoes_split_a_cross_unit_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the literal guard ablated the pair fuses -- and the gate repairs it.

    Before scoping the fused node was a singleton cluster (one source IRI),
    which the veto repair skips; scoped, the cluster has two members and the
    un-merge pass dissolves it.
    """
    aggregator = _normal_form_aggregator(monkeypatch, literal_conflict_guard=False)
    tools = _fake_tools(aggregator)
    state = AgentState()
    state.current_domain = "https://x.org"
    state.doc_hid = "1"
    state.facts_units = [
        _temperature_unit(0, "6", "K"),
        _temperature_unit(1, "10", "DEG_C"),
    ]

    make_merge_facts_node(tools)(state)
    fused = _values_by_subject(state.aggregated_facts)
    assert list(fused.values()) == [{6.0, 10.0}]

    make_validate_facts_node(tools)(state)

    assert state.retrieval_metrics["facts_merge_repair_passes"] == 1
    repaired = _values_by_subject(state.aggregated_facts)
    assert sorted(repaired.values(), key=sorted) == [{6.0}, {10.0}]
    assert not _scoped(state.aggregated_facts)
