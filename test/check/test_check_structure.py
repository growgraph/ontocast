"""SHACL, structure findings and the findings cap of ``check_facts``."""

import pytest
from rdflib import Graph

from ontocast.check import FactsCheckKind, FactsCheckReport, check_facts
from ontocast.onto.rdfgraph import RDFGraph

pytestmark = pytest.mark.unit

ONTO = "http://example.com/onto#"
DATA = "http://example.com/data/"

CATALOG = """
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex:   <http://example.com/onto#> .

ex:Person a owl:Class .
ex:knows a owl:ObjectProperty .
ex:name a owl:DatatypeProperty .
"""

SHAPES = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.com/onto#> .

ex:PersonShape a sh:NodeShape ;
    sh:targetClass ex:Person ;
    sh:property [ sh:path ex:name ; sh:minCount 1 ; sh:severity %s ] .
"""

PREFIXES = f"""
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex:   <{ONTO}> .
@prefix d:    <{DATA}> .
"""


def _rdf(text: str) -> RDFGraph:
    graph = RDFGraph()
    graph.parse(data=text, format="turtle")
    return graph


def _catalog() -> RDFGraph:
    return _rdf(CATALOG)


def _facts(body: str) -> Graph:
    graph = Graph()
    graph.parse(data=PREFIXES + body, format="turtle")
    return graph


def _of_kind(report: FactsCheckReport, kind: FactsCheckKind):
    return [f for f in report.findings if f.kind == kind]


def test_shacl_violation_is_error():
    report = check_facts(
        _facts("d:a a ex:Person ."),
        _catalog(),
        _rdf(SHAPES % "sh:Violation"),
    )
    found = _of_kind(report, FactsCheckKind.SHACL)
    assert len(found) == 1
    assert found[0].severity == "error"
    assert found[0].source_shape is not None
    assert report.shacl_evaluated is True


def test_shacl_warning_severity_kept():
    report = check_facts(
        _facts("d:a a ex:Person ."),
        _catalog(),
        _rdf(SHAPES % "sh:Warning"),
    )
    found = _of_kind(report, FactsCheckKind.SHACL)
    assert [f.severity for f in found] == ["warning"]


def test_shacl_not_evaluated_reports_none():
    facts = _facts("d:a a ex:Person . d:b a ex:Person .")
    report = check_facts(
        facts, _catalog(), _rdf(SHAPES % "sh:Violation"), shacl_max_triples=1
    )
    assert report.shacl_evaluated is None
    assert not _of_kind(report, FactsCheckKind.SHACL)
    assert check_facts(facts, _catalog()).shacl_evaluated is None


def test_untyped_instance_warning():
    report = check_facts(
        _facts('d:a rdfs:label "A" . d:b a ex:Person ; ex:knows d:a .'), _catalog()
    )
    found = _of_kind(report, FactsCheckKind.UNTYPED_INSTANCE)
    assert [f.subject for f in found] == [f"{DATA}a"]
    assert found[0].severity == "warning"
    assert report.metrics.untyped == 1


def test_class_in_instance_namespace_is_a_type():
    facts = Graph()
    facts.parse(data=f"@prefix ex: <{ONTO}> . ex:alice a ex:Person .", format="turtle")
    report = check_facts(facts, _catalog())
    assert not _of_kind(report, FactsCheckKind.UNTYPED_INSTANCE)


def test_isolated_and_silent_warnings():
    report = check_facts(
        _facts(
            "d:a a ex:Person ; ex:knows d:b . d:b a ex:Person . d:c a ex:Person ."
            ' d:v a ex:Person ; ex:name "V" .'
        ),
        _catalog(),
    )
    isolated = {f.subject for f in _of_kind(report, FactsCheckKind.ISOLATED_INSTANCE)}
    silent = {f.subject for f in _of_kind(report, FactsCheckKind.SILENT_INSTANCE)}
    assert silent == {f"{DATA}c"}
    assert isolated == {f"{DATA}v"}
    assert report.metrics.isolated == 2


def test_dangling_reference_warning():
    report = check_facts(_facts("d:a a ex:Person ; ex:knows d:ghost ."), _catalog())
    found = _of_kind(report, FactsCheckKind.DANGLING_REFERENCE)
    assert [f.subject for f in found] == [f"{DATA}ghost"]
    assert found[0].severity == "warning"
    assert found[0].predicate == f"{ONTO}knows"


def test_twin_labels_metric():
    report = check_facts(
        _facts(
            'd:a a ex:Person ; rdfs:label "Ann Lee" .'
            ' d:b a ex:Person ; rdfs:label " ann  lee" .'
            ' d:c a ex:Person ; rdfs:label "Solo" .'
        ),
        _catalog(),
    )
    assert report.metrics.twin_labels == 1
    assert len(_of_kind(report, FactsCheckKind.TWIN_LABELS)) == 1
    assert _of_kind(report, FactsCheckKind.TWIN_LABELS)[0].severity == "warning"


def test_cap_keeps_counts_complete():
    lines = [f'd:p{i} a ex:Person ; ex:name "{i}" .' for i in range(300)]
    lines += [f"d:q{i} ex:badterm{i} d:p{i} ." for i in range(500)]
    report = check_facts(_facts("\n".join(lines)), _catalog(), max_findings=500)
    assert len(report.findings) == 500
    assert report.findings_total >= 800
    assert (
        sum(n for by in report.counts.values() for n in by.values())
        == report.findings_total
    )
    severities = [f.severity for f in report.findings]
    assert severities == sorted(severities, key=lambda s: s != "error")
    assert severities[0] == "error"


def test_input_graph_not_mutated_with_shapes():
    facts = _facts("d:a a ex:Person .")
    before = set(facts)
    check_facts(facts, _catalog(), _rdf(SHAPES % "sh:Violation"))
    assert set(facts) == before


@pytest.mark.parametrize("same_graph", [False, True])
def test_catalog_and_shapes_not_mutated(same_graph):
    catalog = _catalog()
    if same_graph:
        catalog.parse(data=SHAPES % "sh:Violation", format="turtle")
        shapes = catalog
    else:
        shapes = _rdf(SHAPES % "sh:Violation")
    catalog_before, shapes_before = set(catalog), set(shapes)
    report = check_facts(_facts("d:a a ex:Person ."), catalog, shapes)
    assert report.shacl_evaluated is True
    assert set(catalog) == catalog_before
    assert set(shapes) == shapes_before


@pytest.mark.parametrize("with_catalog", [True, False])
def test_named_individual_alone_is_untyped(with_catalog):
    graph = _facts(
        """
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        d:n a owl:NamedIndividual ; ex:knows d:m .
        d:m a ex:Person ; ex:knows d:n .
        """
    )
    report = check_facts(graph, _catalog() if with_catalog else None)
    assert [f.subject for f in _of_kind(report, FactsCheckKind.UNTYPED_INSTANCE)] == [
        f"{DATA}n"
    ]
