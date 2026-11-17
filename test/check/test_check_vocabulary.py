"""Vocabulary, domain and literal checks of ``check_facts``."""

import pytest
from rdflib import XSD, Graph, Literal, URIRef
from rdflib.compare import isomorphic

from ontocast.check import FactsCheckKind, FactsCheckReport, check_facts
from ontocast.onto.rdfgraph import RDFGraph

pytestmark = pytest.mark.unit

ONTO = "http://example.com/onto#"
DATA = "http://example.com/data/"

CATALOG = """
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
@prefix ex:   <http://example.com/onto#> .

ex:Station a owl:Class .
ex:City a owl:Class .
ex:operatedBy a owl:ObjectProperty ; rdfs:domain ex:Station .
ex:inspectedOn a owl:DatatypeProperty ; rdfs:range xsd:date .
ex:capacity a owl:DatatypeProperty ; rdfs:range xsd:decimal .
ex:code a owl:DatatypeProperty , owl:FunctionalProperty .
ex:name a owl:DatatypeProperty .
"""

PREFIXES = f"""
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
@prefix sosa: <http://www.w3.org/ns/sosa/> .
@prefix ex:   <{ONTO}> .
@prefix d:    <{DATA}> .
@prefix other: <http://example.com/other#> .
"""


def _catalog() -> RDFGraph:
    graph = RDFGraph()
    graph.parse(data=CATALOG, format="turtle")
    return graph


def _facts(body: str) -> Graph:
    graph = Graph()
    graph.parse(data=PREFIXES + body, format="turtle")
    return graph


def _of_kind(report: FactsCheckReport, kind: FactsCheckKind):
    return [finding for finding in report.findings if finding.kind == kind]


def test_check_facts_exported():
    from ontocast.check import check_facts as exported

    assert exported is check_facts


def test_non_catalog_vocabulary_strict_is_error():
    graph = _facts(
        """
        d:s1 a ex:Station ; other:worksFor d:x .
        d:s2 a ex:Station ; other:worksFor d:y .
        """
    )
    report = check_facts(graph, _catalog(), vocabulary="strict")
    found = _of_kind(report, FactsCheckKind.NON_CATALOG_VOCABULARY)
    assert len(found) == 1
    assert found[0].severity == "error"
    assert found[0].predicate == "http://example.com/other#worksFor"
    assert report.counts["non_catalog_vocabulary"] == {"error": 1}
    assert not _of_kind(report, FactsCheckKind.UNKNOWN_TERM)


def test_non_catalog_vocabulary_open_is_warning():
    graph = _facts("d:s1 a ex:Station ; other:worksFor d:x .")
    report = check_facts(graph, _catalog(), vocabulary="open")
    found = _of_kind(report, FactsCheckKind.NON_CATALOG_VOCABULARY)
    assert len(found) == 1
    assert found[0].severity == "warning"


@pytest.mark.parametrize("vocabulary", ["strict", "open"])
def test_unknown_well_known_term_error_in_both_modes(vocabulary):
    graph = _facts('d:s1 a ex:Station ; sosa:hasValue "5" .')
    report = check_facts(graph, _catalog(), vocabulary=vocabulary)
    found = _of_kind(report, FactsCheckKind.UNKNOWN_TERM)
    assert len(found) == 1
    assert found[0].severity == "error"
    assert found[0].predicate == "http://www.w3.org/ns/sosa/hasValue"
    assert found[0].suggestions
    assert not _of_kind(report, FactsCheckKind.NON_CATALOG_VOCABULARY)


@pytest.mark.parametrize(
    ("vocabulary", "severity"), [("strict", "error"), ("open", "warning")]
)
def test_undefined_term_in_catalog_namespace_is_non_catalog(vocabulary, severity):
    graph = _facts('d:s1 a ex:Station ; ex:nam "Central" .')
    report = check_facts(graph, _catalog(), vocabulary=vocabulary)
    found = _of_kind(report, FactsCheckKind.NON_CATALOG_VOCABULARY)
    assert [finding.predicate for finding in found] == [f"{ONTO}nam"]
    assert found[0].severity == severity
    assert f"{ONTO}name" in found[0].suggestions
    assert not _of_kind(report, FactsCheckKind.UNKNOWN_TERM)


@pytest.mark.parametrize(
    ("vocabulary", "severity"), [("strict", "error"), ("open", "warning")]
)
def test_undefined_term_in_borrowed_namespace_is_non_catalog(vocabulary, severity):
    catalog = _catalog()
    catalog.parse(
        data=f"""
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix ex: <{ONTO}> .
        @prefix q: <http://example.com/q#> .
        ex:Station rdfs:subClassOf q:Thing .
        """,
        format="turtle",
    )
    graph = _facts(
        """
        @prefix q: <http://example.com/q#> .
        d:s1 a ex:Station , q:Thing ; q:worksFor d:x .
        """
    )
    report = check_facts(graph, catalog, vocabulary=vocabulary)
    found = _of_kind(report, FactsCheckKind.NON_CATALOG_VOCABULARY)
    # The borrowed term itself is a catalog term; its namespace admits no other.
    assert [finding.predicate for finding in found] == ["http://example.com/q#worksFor"]
    assert found[0].severity == severity


def test_domain_violation():
    graph = _facts("d:c1 a ex:City ; ex:operatedBy d:o1 .")
    report = check_facts(graph, _catalog())
    found = _of_kind(report, FactsCheckKind.DOMAIN_VIOLATION)
    assert len(found) == 1
    assert found[0].severity == "error"
    assert found[0].subject == f"{DATA}c1"
    assert found[0].predicate == f"{ONTO}operatedBy"


def test_untyped_value_warning():
    graph = _facts('d:s1 a ex:Station ; ex:inspectedOn "2024-01-01" .')
    report = check_facts(graph, _catalog())
    found = _of_kind(report, FactsCheckKind.UNTYPED_VALUE)
    assert len(found) == 1
    assert found[0].severity == "warning"
    assert found[0].value == Literal("2024-01-01", datatype=XSD.date).n3()
    assert not _of_kind(report, FactsCheckKind.INVALID_LITERAL)
    assert not _of_kind(report, FactsCheckKind.TEMPORAL_RANGE_MISMATCH)


def test_invalid_literal_error():
    graph = _facts('d:s1 a ex:Station ; ex:inspectedOn "every 5 years" .')
    report = check_facts(graph, _catalog())
    found = _of_kind(report, FactsCheckKind.INVALID_LITERAL)
    assert len(found) == 1
    assert found[0].severity == "error"
    assert found[0].subject == f"{DATA}s1"
    assert found[0].predicate == f"{ONTO}inspectedOn"
    assert not _of_kind(report, FactsCheckKind.TEMPORAL_RANGE_MISMATCH)
    assert not _of_kind(report, FactsCheckKind.UNTYPED_VALUE)


def test_invalid_literal_in_its_own_datatype():
    graph = _facts('d:s1 a ex:Station ; ex:inspectedOn "every 5 years"^^xsd:date .')
    report = check_facts(graph, _catalog())
    found = _of_kind(report, FactsCheckKind.INVALID_LITERAL)
    assert len(found) == 1
    assert found[0].severity == "error"
    assert found[0].subject == f"{DATA}s1"


@pytest.mark.parametrize(
    "value", ['"about five"', '"about five"^^xsd:string', '"2024-01-01"^^xsd:date']
)
def test_invalid_literal_on_numeric_range(value):
    graph = _facts(f"d:s1 a ex:Station ; ex:capacity {value} .")
    report = check_facts(graph, _catalog())
    found = _of_kind(report, FactsCheckKind.INVALID_LITERAL)
    assert len(found) == 1
    assert found[0].severity == "error"
    assert found[0].predicate == f"{ONTO}capacity"
    assert not _of_kind(report, FactsCheckKind.TEMPORAL_RANGE_MISMATCH)


@pytest.mark.parametrize(
    ("predicate", "value"),
    [("ex:capacity", '"230"@en'), ("ex:inspectedOn", '"2024-01-01"@en')],
)
def test_language_tagged_literal_on_typed_range_is_invalid(predicate, value):
    graph = _facts(f"d:s1 a ex:Station ; {predicate} {value} .")
    report = check_facts(graph, _catalog())
    found = _of_kind(report, FactsCheckKind.INVALID_LITERAL)
    assert len(found) == 1
    assert found[0].severity == "error"
    literal_kinds = {
        FactsCheckKind.UNTYPED_VALUE,
        FactsCheckKind.TEMPORAL_RANGE_MISMATCH,
    }
    assert not [f for f in report.findings if f.kind in literal_kinds]


def test_temporal_range_mismatch():
    graph = _facts('d:s1 a ex:Station ; ex:inspectedOn "5"^^xsd:integer .')
    report = check_facts(graph, _catalog())
    found = _of_kind(report, FactsCheckKind.TEMPORAL_RANGE_MISMATCH)
    assert len(found) == 1
    assert found[0].severity == "error"
    assert found[0].predicate == f"{ONTO}inspectedOn"
    assert not _of_kind(report, FactsCheckKind.INVALID_LITERAL)
    assert not _of_kind(report, FactsCheckKind.UNTYPED_VALUE)


def test_literal_never_gets_two_findings():
    graph = _facts(
        """
        d:s1 a ex:Station ;
            ex:inspectedOn "2024-01-01" , "every 5 years" ,
                "never"^^xsd:date , "5"^^xsd:integer , "2024-01-02"^^xsd:date ,
                "29 June 2021"^^xsd:string ;
            ex:capacity "about five" , "7" , "8"^^xsd:integer ,
                "true"^^xsd:boolean , "x"^^xsd:string .
        """
    )
    report = check_facts(graph, _catalog())
    literal_kinds = {
        FactsCheckKind.UNTYPED_VALUE,
        FactsCheckKind.INVALID_LITERAL,
        FactsCheckKind.TEMPORAL_RANGE_MISMATCH,
    }
    keys = [
        (finding.subject, finding.predicate, finding.value)
        for finding in report.findings
        if finding.kind in literal_kinds
    ]
    assert len(keys) == len(set(keys))
    # Every value but the already-typed date and the promoted integer is reported.
    assert len(keys) == 9


def test_literal_checks_do_not_log_quarantine(caplog):
    graph = _facts('d:s1 a ex:Station ; ex:inspectedOn "never"^^xsd:date .')
    with caplog.at_level("WARNING"):
        check_facts(graph, _catalog())
    assert not any("Quarantined" in record.getMessage() for record in caplog.records)


def test_functional_violation():
    graph = _facts('d:s1 a ex:Station ; ex:code "A" , "B" .')
    report = check_facts(graph, _catalog())
    found = _of_kind(report, FactsCheckKind.FUNCTIONAL_VIOLATION)
    assert len(found) == 1
    assert found[0].severity == "error"
    assert found[0].subject == f"{DATA}s1"
    assert found[0].predicate == f"{ONTO}code"


def test_catalog_subject_warning():
    graph = _facts('ex:Station rdfs:comment "a stop on the line" .')
    report = check_facts(graph, _catalog())
    found = _of_kind(report, FactsCheckKind.CATALOG_SUBJECT)
    assert len(found) == 1
    assert found[0].severity == "warning"
    assert found[0].subject == f"{ONTO}Station"


def test_schema_in_facts_warning():
    graph = _facts(
        """
        other:Robot a owl:Class ; rdfs:label "Robot" .
        d:r1 a other:Robot .
        d:r2 a other:Robot .
        """
    )
    report = check_facts(graph, _catalog())
    found = _of_kind(report, FactsCheckKind.SCHEMA_IN_FACTS)
    assert len(found) == 1
    assert found[0].severity == "warning"
    assert found[0].subject == "http://example.com/other#Robot"
    # Its uses are undefined vocabulary like any other term outside the catalog.
    unknown = _of_kind(report, FactsCheckKind.NON_CATALOG_VOCABULARY)
    assert [finding.predicate for finding in unknown] == [
        "http://example.com/other#Robot"
    ]


@pytest.mark.parametrize("ontology", [None, RDFGraph()])
def test_empty_catalog_skips_catalog_checks(ontology):
    graph = _facts(
        """
        d:c1 a ex:City ; ex:operatedBy d:o1 ; other:worksFor d:x .
        d:s1 ex:code "A" , "B" ; ex:inspectedOn "2024-01-01" .
        d:s2 sosa:hasValue "5" ; ex:inspectedOn "soon"^^xsd:date .
        """
    )
    report = check_facts(graph, ontology)
    assert report.catalog_empty is True
    kinds = {finding.kind for finding in report.findings}
    for skipped in (
        FactsCheckKind.NON_CATALOG_VOCABULARY,
        FactsCheckKind.DOMAIN_VIOLATION,
        FactsCheckKind.FUNCTIONAL_VIOLATION,
        FactsCheckKind.UNTYPED_VALUE,
        FactsCheckKind.TEMPORAL_RANGE_MISMATCH,
    ):
        assert skipped not in kinds
    # The well-known vocabulary and literal syntax are judged without a catalog.
    assert FactsCheckKind.UNKNOWN_TERM in kinds
    assert FactsCheckKind.INVALID_LITERAL in kinds


def test_input_graph_not_mutated():
    graph = _facts(
        """
        d:s1 a ex:Station ;
            ex:inspectedOn "2024-01-01" ;
            ex:capacity "about five" .
        d:s2 a ex:Station ; ex:inspectedOn "every 5 years" .
        d:s3 a ex:Station ; ex:inspectedOn "never"^^xsd:date .
        d:s4 a ex:Station ; ex:inspectedOn "5"^^xsd:integer .
        """
    )
    before = Graph()
    for triple in graph:
        before.add(triple)
    report = check_facts(graph, _catalog())
    kinds = {finding.kind for finding in report.findings}
    assert {
        FactsCheckKind.UNTYPED_VALUE,
        FactsCheckKind.INVALID_LITERAL,
        FactsCheckKind.TEMPORAL_RANGE_MISMATCH,
    } <= kinds
    assert isomorphic(before, graph)
    assert len(before) == len(graph)


def test_report_counts_and_metrics():
    graph = _facts(
        """
        d:s1 a ex:Station ; other:worksFor d:x ; ex:inspectedOn "2024-01-01" .
        d:s2 a ex:Station ; other:worksFor d:y .
        """
    )
    report = check_facts(graph, _catalog())
    assert report.findings_total == len(report.findings)
    assert sum(sum(by.values()) for by in report.counts.values()) == len(
        report.findings
    )
    assert report.metrics.triples == len(graph)
    assert report.metrics.instances == 2
    assert report.catalog_empty is False
    for finding in report.findings:
        assert "IN PLACE" not in finding.message


PLACEHOLDER_CATALOG = """
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex:   <http://example.org/onto#> .

ex:Person a owl:Class .
ex:knows a owl:ObjectProperty ; rdfs:domain ex:Person .
"""


def test_catalog_in_placeholder_namespace_accepts_its_own_terms():
    catalog = RDFGraph()
    catalog.parse(data=PLACEHOLDER_CATALOG, format="turtle")
    graph = Graph()
    graph.parse(
        data="""
        @prefix ex: <http://example.org/onto#> .
        @prefix d:  <http://example.org/data/> .
        d:a a ex:Person ; ex:knows d:b .
        d:b a ex:Person ; ex:knows d:a .
        """,
        format="turtle",
    )
    report = check_facts(graph, catalog)
    assert not _of_kind(report, FactsCheckKind.UNKNOWN_TERM)
    assert not _of_kind(report, FactsCheckKind.NON_CATALOG_VOCABULARY)
    # A term the catalog does not define is still a placeholder term.
    graph.add(
        (
            URIRef("http://example.org/data/a"),
            URIRef("http://example.org/onto#invented"),
            Literal("x"),
        )
    )
    report = check_facts(graph, catalog)
    assert [f.predicate for f in _of_kind(report, FactsCheckKind.UNKNOWN_TERM)] == [
        "http://example.org/onto#invented"
    ]


def test_catalog_individual_as_subject_is_an_instance():
    catalog = _catalog()
    catalog.parse(data=f"@prefix ex: <{ONTO}> . ex:Paris a ex:City .", format="turtle")
    graph = _facts('ex:Paris a ex:City ; ex:operatedBy d:o1 ; ex:code "A" , "B" .')
    report = check_facts(graph, catalog)
    assert not _of_kind(report, FactsCheckKind.CATALOG_SUBJECT)
    assert [f.subject for f in _of_kind(report, FactsCheckKind.DOMAIN_VIOLATION)] == [
        f"{ONTO}Paris"
    ]
    assert [
        f.subject for f in _of_kind(report, FactsCheckKind.FUNCTIONAL_VIOLATION)
    ] == [f"{ONTO}Paris"]
    assert report.metrics.instances == 1


DERIVED_CATALOG = """
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
@prefix ex:   <http://example.com/onto#> .

ex:Station a owl:Class .
ex:count a owl:DatatypeProperty ; rdfs:range xsd:integer .
ex:capacity a owl:DatatypeProperty ; rdfs:range xsd:decimal .
ex:at a owl:DatatypeProperty ; rdfs:range xsd:dateTime .
"""

_LITERAL_KINDS = {
    FactsCheckKind.UNTYPED_VALUE,
    FactsCheckKind.INVALID_LITERAL,
    FactsCheckKind.TEMPORAL_RANGE_MISMATCH,
}


@pytest.mark.parametrize(
    ("predicate", "value"),
    [
        ("ex:count", '"5"^^xsd:int'),
        ("ex:count", '"7"^^xsd:long'),
        ("ex:count", '"3"^^xsd:nonNegativeInteger'),
        ("ex:count", '"-3"^^xsd:negativeInteger'),
        ("ex:count", '"9"^^xsd:unsignedByte'),
        ("ex:capacity", '"3"^^xsd:int'),
        ("ex:capacity", '"4"^^xsd:integer'),
        ("ex:capacity", '"2"^^xsd:positiveInteger'),
        ("ex:at", '"2024-01-01T10:00:00Z"^^xsd:dateTimeStamp'),
    ],
)
def test_derived_datatype_fits_its_base_range(predicate, value):
    catalog = RDFGraph()
    catalog.parse(data=DERIVED_CATALOG, format="turtle")
    graph = _facts(f"d:s1 a ex:Station ; {predicate} {value} .")
    report = check_facts(graph, catalog)
    assert not [f for f in report.findings if f.kind in _LITERAL_KINDS]


@pytest.mark.parametrize(
    ("predicate", "value"),
    [("ex:count", '"1.5"^^xsd:decimal'), ("ex:at", '"2024-01-01"^^xsd:date')],
)
def test_base_datatype_does_not_fit_a_narrower_range(predicate, value):
    catalog = RDFGraph()
    catalog.parse(data=DERIVED_CATALOG, format="turtle")
    graph = _facts(f"d:s1 a ex:Station ; {predicate} {value} .")
    report = check_facts(graph, catalog)
    assert len([f for f in report.findings if f.kind in _LITERAL_KINDS]) == 1


@pytest.mark.parametrize(
    "value",
    [
        '"maybe"^^xsd:boolean',
        '"abc"^^xsd:int',
        '"-1"^^xsd:nonNegativeInteger',
    ],
)
@pytest.mark.parametrize("with_catalog", [True, False])
def test_value_invalid_in_its_own_datatype(value, with_catalog):
    graph = _facts(f"d:s1 a ex:Station ; ex:name {value} .")
    report = check_facts(graph, _catalog() if with_catalog else None)
    found = _of_kind(report, FactsCheckKind.INVALID_LITERAL)
    assert len(found) == 1
    assert found[0].severity == "error"


def test_custom_datatype_is_not_judged():
    graph = _facts('d:s1 a ex:Station ; ex:name "x"^^<http://example.com/dt#c> .')
    report = check_facts(graph, _catalog())
    assert not _of_kind(report, FactsCheckKind.INVALID_LITERAL)


def test_untyped_value_message_fits_a_string_source():
    graph = _facts('d:s1 a ex:Station ; ex:inspectedOn "2024-01-01"^^xsd:string .')
    report = check_facts(graph, _catalog())
    (found,) = _of_kind(report, FactsCheckKind.UNTYPED_VALUE)
    assert "no datatype" not in found.message
    assert "xsd:date" in found.message


def test_invalid_literal_message_keeps_the_language_tag():
    graph = _facts('d:s1 a ex:Station ; ex:capacity "230"@en .')
    report = check_facts(graph, _catalog())
    (found,) = _of_kind(report, FactsCheckKind.INVALID_LITERAL)
    assert '"230"@en' in found.message


def test_suggestions_computed_once_per_reported_term(monkeypatch):
    from ontocast.check import facts as facts_module
    from ontocast.tool.facts_validation import terms as terms_module

    catalog = RDFGraph()
    for i in range(200):
        catalog.parse(
            data=f"""
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            <{ONTO}prop{i}Name> a owl:DatatypeProperty .
            <{ONTO}Class{i}> a owl:Class .
            """,
            format="turtle",
        )
    lines = [f'd:x{i} a ex:Class1 ; ex:prop{i}Nam "v" .' for i in range(200)]
    graph = _facts("\n".join(lines))

    full = check_facts(graph, catalog)
    role_calls = 0
    alias_calls = 0
    roles = terms_module._catalog_term_roles
    alias = facts_module._alias_candidates

    def counted_roles(*args, **kwargs):
        nonlocal role_calls
        role_calls += 1
        return roles(*args, **kwargs)

    def counted_alias(*args, **kwargs):
        nonlocal alias_calls
        alias_calls += 1
        return alias(*args, **kwargs)

    monkeypatch.setattr(terms_module, "_catalog_term_roles", counted_roles)
    monkeypatch.setattr(facts_module, "_alias_candidates", counted_alias)
    capped = check_facts(graph, catalog, max_findings=10)

    assert role_calls <= 1
    assert alias_calls <= 10
    assert capped.counts == full.counts
    assert capped.findings == full.findings[:10]
    assert any(f.suggestions for f in capped.findings)


def test_catalog_individual_keeps_its_catalog_type():
    catalog = _catalog()
    catalog.parse(data=f"@prefix ex: <{ONTO}> . ex:Paris a ex:City .", format="turtle")
    graph = _facts(
        "d:s1 a ex:Station ; ex:operatedBy ex:Paris . ex:Paris ex:name 'P' ."
    )
    report = check_facts(graph, catalog)
    assert not _of_kind(report, FactsCheckKind.UNTYPED_INSTANCE)
    assert not _of_kind(report, FactsCheckKind.CATALOG_SUBJECT)
