"""Deterministic facts normalization/repair/findings tests.

Fixtures reproduce the exact case5 failure shapes: plain-int literals next
to typed decimals, ``qudt:value`` for ``qudt:numericValue``,
``qqval:lowerBound`` for ``qqval:hasLowerBound``, ``ex:`` placeholder
predicates, doc-namespace predicates, and string epistemic qualifiers
against a closed individual range.
"""

from rdflib import Literal, URIRef

from ontocast.onto.model import FactsUnitFindingKind
from ontocast.onto.rdfgraph import RDFGraph, RejectedLiteralTriple
from ontocast.tool.facts_invariants import (
    collect_unit_findings,
    format_findings_for_prompt,
    normalize_literals_against_schema,
    repair_property_aliases,
)

QUDT = "http://qudt.org/schema/qudt/"
QQVAL = "https://growgraph.dev/ontologies/qqval#"
FACTS = "https://growgraph.dev/facts/"


def _ontology() -> RDFGraph:
    graph = RDFGraph()
    graph.parse(
        data=f"""
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
        @prefix qudt: <{QUDT}> .
        @prefix qqval: <{QQVAL}> .
        qudt:numericValue a owl:DatatypeProperty ; rdfs:range xsd:decimal .
        qudt:unit a owl:ObjectProperty ; rdfs:range qudt:Unit .
        qqval:hasLowerBound a owl:ObjectProperty .
        qqval:hasUpperBound a owl:ObjectProperty .
        qqval:epistemicQualifier a owl:ObjectProperty ;
            rdfs:range qqval:EpistemicQualifier .
        qqval:Approximate a owl:NamedIndividual, qqval:EpistemicQualifier ;
            rdfs:label "approximate" .
        qqval:Exact a owl:NamedIndividual, qqval:EpistemicQualifier .
        """,
        format="turtle",
    )
    return graph


def _facts(body: str) -> RDFGraph:
    graph = RDFGraph()
    graph.parse(
        data=f"""
        @prefix cd: <{FACTS}> .
        @prefix qudt: <{QUDT}> .
        @prefix qqval: <{QQVAL}> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
        @prefix ex: <http://example.org/> .
        {body}
        """,
        format="turtle",
    )
    return graph


def test_normalize_literals_retypes_untyped_numeric() -> None:
    graph = _facts('cd:v qudt:numericValue 230 , "96"^^xsd:decimal .')
    retyped = normalize_literals_against_schema(graph, _ontology())
    assert retyped == 1
    values = {
        (str(obj), obj.datatype)
        for obj in graph.objects(URIRef(f"{FACTS}v"), URIRef(f"{QUDT}numericValue"))
        if isinstance(obj, Literal)
    }
    assert all(str(dt).endswith("decimal") for _, dt in values)
    assert {value for value, _ in values} == {"230", "96"}


def test_repair_property_aliases_rewrites_qudt_value() -> None:
    # qudt:numericValue is dominant in the graph itself; qudt:value is the
    # near-miss (the case5 paper-3 shape: 8 qudt:value vs many numericValue).
    graph = _facts(
        """
        cd:a qudt:numericValue "1"^^xsd:decimal .
        cd:b qudt:numericValue "2"^^xsd:decimal .
        cd:c qudt:value "375"^^xsd:decimal .
        """
    )
    rewritten, findings = repair_property_aliases(graph, _ontology())
    assert rewritten == 1
    assert not findings
    assert (
        URIRef(f"{FACTS}c"),
        URIRef(f"{QUDT}numericValue"),
        None,
    ) in graph


def test_repair_property_aliases_rewrites_qqval_bound() -> None:
    graph = _facts('cd:r qqval:lowerBound "10"^^xsd:decimal .')
    rewritten, findings = repair_property_aliases(graph, _ontology())
    # qqval:lowerBound tokens are contained in hasLowerBound -> unique rewrite.
    assert rewritten == 1
    assert not findings
    assert (
        URIRef(f"{FACTS}r"),
        URIRef(f"{QQVAL}hasLowerBound"),
        None,
    ) in graph


def test_collect_unit_findings_flags_example_org_and_doc_predicates() -> None:
    doc_ns = "https://growgraph.dev/doc/abc/"
    graph = _facts(
        f"""
        @prefix doc: <{doc_ns}> .
        cd:x ex:redShiftContribution cd:y .
        cd:z doc:hasApplication cd:w .
        """
    )
    findings = collect_unit_findings(
        graph=graph,
        ontology_graph=_ontology(),
        quarantined=[],
        extraction_text="",
        fact_namespaces=[FACTS, doc_ns],
    )
    kinds = {(finding.kind, finding.predicate) for finding in findings}
    assert (
        FactsUnitFindingKind.UNKNOWN_TERM,
        "http://example.org/redShiftContribution",
    ) in kinds
    assert (
        FactsUnitFindingKind.UNKNOWN_TERM,
        f"{doc_ns}hasApplication",
    ) in kinds
    assert all(finding.mandatory for finding in findings)


def test_collect_unit_findings_quarantine_gets_closed_range_suggestions() -> None:
    rejected = RejectedLiteralTriple(
        subject=f"{FACTS}v1",
        predicate=f"{QQVAL}epistemicQualifier",
        object_lexical="approximate",
        datatype="",
        reason="object property expects IRI",
        expected_range=f"{QQVAL}EpistemicQualifier",
    )
    findings = collect_unit_findings(
        graph=_facts(""),
        ontology_graph=_ontology(),
        quarantined=[rejected],
        extraction_text="",
        fact_namespaces=[FACTS],
    )
    quarantine = [
        finding
        for finding in findings
        if finding.kind == FactsUnitFindingKind.QUARANTINED_LITERAL
    ]
    assert len(quarantine) == 1
    assert quarantine[0].suggestions == [f"{QQVAL}Approximate"]


def test_collect_unit_findings_numeric_coverage_is_advisory() -> None:
    graph = _facts('cd:v qudt:numericValue "96"^^xsd:decimal .')
    findings = collect_unit_findings(
        graph=graph,
        ontology_graph=_ontology(),
        quarantined=[],
        extraction_text="red shift of 96 meV and 12.5 meV at 77 K",
        fact_namespaces=[FACTS],
    )
    coverage = [
        finding
        for finding in findings
        if finding.kind == FactsUnitFindingKind.NUMERIC_COVERAGE
    ]
    assert len(coverage) == 1
    assert not coverage[0].mandatory
    assert "12.5" in coverage[0].message
    assert "77" in coverage[0].message
    assert "96" not in coverage[0].value


def test_format_findings_for_prompt_sections() -> None:
    graph = _facts("cd:x ex:p cd:y .")
    findings = collect_unit_findings(
        graph=graph,
        ontology_graph=_ontology(),
        quarantined=[],
        extraction_text="value 42.5 K",
        fact_namespaces=[FACTS],
    )
    prompt = format_findings_for_prompt(findings)
    assert "## MANDATORY fixes" in prompt
    assert "## Verify numeric coverage" in prompt
    assert "42.5" in prompt
