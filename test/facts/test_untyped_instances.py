"""Every instance in the facts is typed.

Three mechanisms share one notion of *instance* (a fact-namespace IRI in
subject or IRI-object position) and *typed* (an ``rdf:type`` naming a class
outside the fact namespaces that is neither a metaclass nor ``owl:Thing``):
the parse-time repair that materialises the types the ontology entails, the
mandatory ``UNTYPED_INSTANCE`` finding that drives the critic, and the gate
step that drops whatever is still untyped after aggregation.
"""

from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import RDF, RDFS, BNode, Literal, URIRef
from rdflib.namespace import PROV

from ontocast.config import FactsValidationConfig
from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.enum import RetrievalMetric
from ontocast.onto.model import FactsGateRepairKind, FactsUnitFindingKind
from ontocast.onto.rdfgraph import RDFGraph, is_rdflib_triple
from ontocast.onto.state import AgentState
from ontocast.stategraph.facts_gate import drop_untyped_instances, run_facts_gate
from ontocast.tool.facts_validation import (
    collect_unit_findings,
    materialize_entailed_types,
    untyped_instance_findings,
    untyped_instances,
)
from ontocast.toolbox import ToolBox

pytestmark = pytest.mark.unit

CD = f"{DEFAULT_IRI}/"
EX = "https://x.org/schema#"

PREFIXES = f"""
@prefix cd:   <{CD}> .
@prefix ex:   <{EX}> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix schema: <https://schema.org/> .
"""

ONTOLOGY = """
ex:Sample a owl:Class ; rdfs:label "Sample" .
ex:Device a owl:Class ; rdfs:label "Device" .
ex:Measurement a owl:Class .
ex:TimedMeasurement a owl:Class ; rdfs:subClassOf ex:Measurement .
ex:Process a owl:Class .
ex:measuredOn a owl:ObjectProperty ; rdfs:domain ex:Sample ; rdfs:range ex:Device .
ex:hasResult a owl:ObjectProperty ; rdfs:domain ex:TimedMeasurement .
ex:hasCode a owl:DatatypeProperty ; rdfs:range xsd:string .
ex:hasShade a owl:DatatypeProperty ; rdfs:range ex:Shade .
ex:Shade a rdfs:Datatype .
ex:unit_m a ex:Unit ; rdfs:label "m" .
"""


def _ontology() -> RDFGraph:
    graph = RDFGraph()
    graph.parse(data=PREFIXES + ONTOLOGY, format="turtle")
    return graph


def _facts(body: str) -> RDFGraph:
    graph = RDFGraph()
    graph.parse(data=PREFIXES + body, format="turtle")
    return graph


def _cd(name: str) -> URIRef:
    return URIRef(f"{CD}{name}")


def _untyped_findings(graph: RDFGraph, ontology: RDFGraph | None = None):
    return {
        finding.subject: finding
        for finding in collect_unit_findings(
            graph=graph,
            ontology_graph=ontology,
            quarantined=[],
            extraction_text="",
            fact_namespaces=[CD],
            coverage_limit=0,
        )
        if finding.kind == FactsUnitFindingKind.UNTYPED_INSTANCE
    }


# ---------------------------------------------------------------------------
# The finding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        'cd:s rdfs:label "s" ; ex:hasCode "A1" .',
        'cd:s a owl:Thing ; rdfs:label "s" .',
        'cd:s a schema:Thing ; rdfs:label "s" .',
        'cd:s a cd:LocalClass ; rdfs:label "s" .',
        'cd:s a owl:NamedIndividual ; rdfs:label "s" .',
        'cd:s a rdfs:Class ; rdfs:label "s" .',
    ],
    ids=["no type", "owl:Thing", "schema:Thing", "facts class", "individual", "class"],
)
def test_untyped_instance_is_reported(body: str) -> None:
    findings = _untyped_findings(_facts(body))

    assert set(findings) == {str(_cd("s"))}
    finding = findings[str(_cd("s"))]
    assert finding.mandatory is True
    assert "ONTOLOGY section" in finding.message
    assert "new class" not in finding.message.lower()


def test_an_object_only_instance_is_reported() -> None:
    graph = _facts("cd:s a ex:Sample ; ex:measuredOn cd:d .")

    assert set(_untyped_findings(graph)) == {str(_cd("d"))}


@pytest.mark.parametrize(
    "body",
    [
        'cd:s a ex:Sample ; rdfs:label "s" .',
        'cd:s a owl:Thing, ex:Sample ; rdfs:label "s" .',
        'cd:s a schema:Person ; rdfs:label "s" .',
        # Blank nodes are never instances.
        'cd:s a ex:Sample ; ex:hasValue [ ex:numericValue "3"^^xsd:decimal ] .',
        # Alias and lineage handles are not instances.
        "cd:s a ex:Sample ; owl:sameAs cd:s_alias ; prov:wasDerivedFrom cd:chunk_0 .",
    ],
    ids=["catalog class", "thing plus class", "generic class", "blank node", "handles"],
)
def test_typed_instances_blank_nodes_and_handles_are_not_reported(body: str) -> None:
    assert _untyped_findings(_facts(body)) == {}


def test_chunk_and_document_scaffolding_is_not_an_instance() -> None:
    graph = _facts(
        """
        cd:chunk_0 a prov:Entity, schema:Text ;
            schema:position "0"^^xsd:integer ;
            schema:identifier "h0" ;
            prov:generatedAtTime "2024-01-01T00:00:00"^^xsd:dateTime .
        cd:other_chunk schema:position "1"^^xsd:integer .
        cd:s a ex:Sample ; prov:wasDerivedFrom cd:chunk_0 .
        """
    )

    assert untyped_instances(graph, [CD]) == []


def test_suggestions_come_from_domain_range_and_name_tokens() -> None:
    graph = _facts(
        """
        cd:sample_1 rdfs:label "sample 1" ; ex:measuredOn cd:d .
        cd:d rdfs:label "the device" .
        """
    )

    findings = _untyped_findings(graph, _ontology())

    assert findings[str(_cd("sample_1"))].suggestions == [f"{EX}Sample"]
    assert findings[str(_cd("d"))].suggestions == [f"{EX}Device"]


def test_direct_call_without_context_offers_no_suggestions() -> None:
    findings = untyped_instance_findings(_facts('cd:s rdfs:label "s" .'), None, [CD])

    assert [finding.suggestions for finding in findings] == [[]]


# ---------------------------------------------------------------------------
# Entailed types
# ---------------------------------------------------------------------------


def test_domain_and_range_types_are_materialised_and_recorded() -> None:
    graph = _facts("cd:s ex:measuredOn cd:d .")

    added, records = materialize_entailed_types(graph, _ontology())

    assert added == 2
    assert (_cd("s"), RDF.type, URIRef(f"{EX}Sample")) in graph
    assert (_cd("d"), RDF.type, URIRef(f"{EX}Device")) in graph
    assert {record.kind for record in records} == {FactsUnitFindingKind.ENTAILED_TYPE}
    assert {(record.source, record.target) for record in records} == {
        (f"{_cd('s')} {EX}measuredOn", f"{EX}Sample"),
        (f"{_cd('d')} {EX}measuredOn", f"{EX}Device"),
    }
    assert _untyped_findings(graph, _ontology()) == {}


def test_entailment_specialises_an_asserted_supertype() -> None:
    graph = _facts("cd:m a ex:Measurement ; ex:hasResult cd:r .")

    added, _ = materialize_entailed_types(graph, _ontology())

    assert added == 1
    assert (_cd("m"), RDF.type, URIRef(f"{EX}TimedMeasurement")) in graph


def test_entailment_leaves_a_contradicting_type_to_the_domain_violation() -> None:
    graph = _facts("cd:m a ex:Process ; ex:hasResult cd:r .")

    added, _ = materialize_entailed_types(graph, _ontology())

    assert added == 0
    kinds = {
        finding.kind
        for finding in collect_unit_findings(
            graph=graph,
            ontology_graph=_ontology(),
            quarantined=[],
            extraction_text="",
            fact_namespaces=[CD],
            coverage_limit=0,
        )
    }
    assert FactsUnitFindingKind.DOMAIN_VIOLATION in kinds


def test_entailment_skips_redundant_datatype_and_catalog_targets() -> None:
    graph = _facts(
        """
        cd:m a ex:TimedMeasurement ; ex:hasResult cd:r ; ex:hasShade "dark" .
        cd:s ex:hasCode "A1" ; ex:measuredOn ex:unit_m .
        """
    )

    added, _ = materialize_entailed_types(graph, _ontology())

    # cd:m already is a TimedMeasurement; literal ranges type nothing; the
    # catalog individual is typed by the catalog; cd:s gains the domain.
    assert added == 1
    assert (_cd("s"), RDF.type, URIRef(f"{EX}Sample")) in graph
    assert (URIRef(f"{EX}unit_m"), RDF.type, None) not in graph


def test_entailment_without_context_is_a_no_op() -> None:
    graph = _facts("cd:s ex:measuredOn cd:d .")
    assert materialize_entailed_types(graph, None) == (0, [])
    assert materialize_entailed_types(graph, RDFGraph()) == (0, [])


# ---------------------------------------------------------------------------
# The drop step
# ---------------------------------------------------------------------------


def _reified_oxigraph_graph(body: str) -> RDFGraph:
    """An oxigraph-backed facts graph, every triple reified as the aggregator does."""
    pytest.importorskip("oxrdflib")
    import pyoxigraph as ox

    from ontocast.onto.constants import RDF_REIFIES
    from ontocast.onto.rdfgraph import _oxigraph_inner_store

    parsed = _facts(body)
    graph = RDFGraph(store="oxigraph")
    for triple in parsed:
        graph.add(triple)
    store = cast(ox.Store, _oxigraph_inner_store(graph.store))
    context = ox.BlankNode(str(graph.identifier))
    for subject, predicate, obj in parsed:
        if not isinstance(subject, URIRef) or not isinstance(obj, URIRef):
            continue
        reifier = ox.BlankNode()
        store.add(
            ox.Quad(
                reifier,
                ox.NamedNode(str(RDF_REIFIES)),
                ox.Triple(
                    ox.NamedNode(str(subject)),
                    ox.NamedNode(str(predicate)),
                    ox.NamedNode(str(obj)),
                ),
                context,
            )
        )
        store.add(
            ox.Quad(
                reifier,
                ox.NamedNode(str(PROV.wasDerivedFrom)),
                ox.NamedNode(f"{CD}chunk_0"),
                context,
            )
        )
    return graph


def _reifier_count(graph: RDFGraph) -> int:
    return sum(1 for triple in graph if not is_rdflib_triple(triple))


def _value_blank_nodes(graph: RDFGraph) -> set[BNode]:
    """Blank nodes of the facts themselves; reifiers are not counted."""
    reifiers = {triple[0] for triple in graph if not is_rdflib_triple(triple)}
    return {
        term
        for triple in graph
        if is_rdflib_triple(triple)
        for term in triple
        if isinstance(term, BNode) and term not in reifiers
    }


def test_drop_cascades_through_triples_blank_nodes_and_reifiers() -> None:
    graph = _reified_oxigraph_graph(
        """
        cd:typed a ex:Sample ; rdfs:label "typed" ;
            ex:measuredOn cd:ghost ;
            ex:hasValue [ ex:numericValue "3"^^xsd:decimal ] .
        cd:ghost rdfs:label "ghost" ;
            ex:relatesTo cd:typed ;
            ex:hasValue [ ex:numericValue "4"^^xsd:decimal ;
                          ex:hasUnit [ rdfs:label "nested" ] ] .
        cd:shared a ex:Device ; ex:owns cd:ghost ; ex:owns cd:typed .
        """
    )
    typed, ghost, shared = _cd("typed"), _cd("ghost"), _cd("shared")
    assert _reifier_count(graph) == 6
    blank_before = _value_blank_nodes(graph)
    assert len(blank_before) == 3

    _, dropped = drop_untyped_instances(graph, [CD])

    assert dropped == [str(ghost)]
    assert (ghost, None, None) not in graph
    assert (None, None, ghost) not in graph
    # The typed neighbours keep everything that does not mention the ghost.
    assert (typed, RDFS.label, Literal("typed")) in graph
    assert (shared, URIRef(f"{EX}owns"), typed) in graph
    assert (typed, RDF.type, URIRef(f"{EX}Sample")) in graph
    # The typed node's blank value survives; the ghost's, and the blank node
    # reachable only through it, are gone.
    blank_after = _value_blank_nodes(graph)
    assert len(blank_after) == 1
    assert (None, URIRef(f"{EX}hasUnit"), None) not in graph
    assert (None, RDFS.label, Literal("nested")) not in graph
    # Reifiers quoting removed statements are swept; the others stay.
    assert _reifier_count(graph) == 3


def test_drop_is_a_no_op_on_a_typed_graph() -> None:
    graph = _facts(
        'cd:s a ex:Sample ; rdfs:label "s" ; ex:measuredOn cd:d . cd:d a ex:Device .'
    )
    size = len(graph)

    _, dropped = drop_untyped_instances(graph, [CD])

    assert dropped == []
    assert len(graph) == size


def _fake_tools() -> ToolBox:
    facts_validation = FactsValidationConfig()
    return cast(
        ToolBox,
        SimpleNamespace(
            shapes_catalog=SimpleNamespace(graph=lambda: None),
            shapes_prompt_contract=lambda: ("", (), False),
            ontology_manager=SimpleNamespace(catalog_terms=lambda: set()),
            config=SimpleNamespace(
                get_tool_config=lambda: SimpleNamespace(
                    facts_validation=facts_validation
                )
            ),
        ),
    )


def test_gate_records_dropped_instances_on_the_state() -> None:
    state = AgentState()
    state.current_domain = "https://x.org"
    state.doc_hid = "1"
    state.aggregated_facts = _facts(
        """
        cd:s a ex:Sample ; rdfs:label "s" ; ex:relatesTo cd:ghost_b, cd:ghost_a .
        cd:ghost_a rdfs:label "a" .
        cd:ghost_b rdfs:label "b" .
        """
    )

    run_facts_gate(state, _ontology(), _fake_tools(), merge_repair=False)

    dropped = [f"{CD}ghost_a", f"{CD}ghost_b"]
    assert state.facts_untyped_dropped == dropped
    assert state.retrieval_metrics[RetrievalMetric.FACTS_UNTYPED_DROPPED] == 2
    assert [
        record.source
        for record in state.facts_gate_repairs
        if record.kind == FactsGateRepairKind.UNTYPED_DROPPED
    ] == dropped
    assert state.facts_conformance["repairs_applied"] == {"untyped_dropped": 2}
    assert (None, None, _cd("ghost_a")) not in state.aggregated_facts
    assert (_cd("s"), RDFS.label, Literal("s")) in state.aggregated_facts


def test_gate_metric_is_zero_when_nothing_is_dropped() -> None:
    state = AgentState()
    state.current_domain = "https://x.org"
    state.doc_hid = "1"
    state.aggregated_facts = _facts('cd:s a ex:Sample ; rdfs:label "s" .')

    run_facts_gate(state, _ontology(), _fake_tools(), merge_repair=False)

    assert state.retrieval_metrics[RetrievalMetric.FACTS_UNTYPED_DROPPED] == 0
    assert state.facts_untyped_dropped == []
    assert state.facts_gate_repairs == []


def test_gate_types_an_instance_its_link_entails_instead_of_dropping_it() -> None:
    # The unit's context lacked ex:measuredOn's range; the gate's whole context
    # declares it, so the device is typed and the sample keeps its link.
    state = AgentState()
    state.current_domain = "https://x.org"
    state.doc_hid = "1"
    state.aggregated_facts = _facts(
        """
        cd:s a ex:Sample ; rdfs:label "s" ; ex:measuredOn cd:rig .
        cd:rig rdfs:label "rig" .
        cd:loose rdfs:label "loose" ; ex:relatesTo cd:s .
        """
    )

    run_facts_gate(state, _ontology(), _fake_tools(), merge_repair=False)

    facts = state.aggregated_facts
    assert (_cd("rig"), RDF.type, URIRef(f"{EX}Device")) in facts
    assert (_cd("s"), URIRef(f"{EX}measuredOn"), _cd("rig")) in facts
    assert state.retrieval_metrics[RetrievalMetric.FACTS_UNTYPED_INFERRED] == 1
    assert state.facts_untyped_dropped == [f"{CD}loose"]
    # The dropped instance took its link to the sample with it.
    assert state.retrieval_metrics[RetrievalMetric.FACTS_LINKS_REMOVED] == 1
    assert state.retrieval_metrics[RetrievalMetric.FACTS_COMPONENTS] == 1
    assert state.retrieval_metrics[RetrievalMetric.FACTS_ISOLATED] == 0


def test_gate_counts_components_and_isolated_instances() -> None:
    state = AgentState()
    state.current_domain = "https://x.org"
    state.doc_hid = "1"
    state.aggregated_facts = _facts(
        """
        cd:s a ex:Sample ; ex:measuredOn cd:d .
        cd:d a ex:Device .
        cd:alone a ex:Process ; rdfs:label "alone" .
        """
    )

    run_facts_gate(state, _ontology(), _fake_tools(), merge_repair=False)

    assert state.retrieval_metrics[RetrievalMetric.FACTS_COMPONENTS] == 2
    assert state.retrieval_metrics[RetrievalMetric.FACTS_ISOLATED] == 1
    assert state.retrieval_metrics[RetrievalMetric.FACTS_UNTYPED_INFERRED] == 0
    assert state.retrieval_metrics[RetrievalMetric.FACTS_LINKS_REMOVED] == 0
