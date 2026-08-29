"""Deterministic facts normalization/repair/findings tests.

Fixtures reproduce observed failure shapes: plain-int literals next
to typed decimals, ``qudt:value`` for ``qudt:numericValue``,
``qqval:lowerBound`` for ``qqval:hasLowerBound``, ``ex:`` placeholder
predicates, doc-namespace predicates, and string epistemic qualifiers
against a closed individual range.
"""

import pytest
from rdflib import RDF, Literal, URIRef
from rdflib.namespace import XSD

from ontocast.onto.model import FactsUnitFindingKind, format_findings_for_prompt
from ontocast.onto.rdfgraph import RDFGraph, RejectedLiteralTriple
from ontocast.tool.facts_validation import (
    collect_unit_findings,
    normalize_literals_against_schema,
    repair_literal_type_objects,
    repair_property_aliases,
)

pytestmark = pytest.mark.unit

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
    # near-miss: a few qudt:value against many qudt:numericValue.
    graph = _facts(
        """
        cd:a qudt:numericValue "1"^^xsd:decimal .
        cd:b qudt:numericValue "2"^^xsd:decimal .
        cd:c qudt:value "375"^^xsd:decimal .
        """
    )
    rewritten, findings, _applied = repair_property_aliases(graph, _ontology())
    assert rewritten == 1
    assert not findings
    assert (
        URIRef(f"{FACTS}c"),
        URIRef(f"{QUDT}numericValue"),
        None,
    ) in graph


def test_repair_property_aliases_rewrites_qqval_bound() -> None:
    graph = _facts('cd:r qqval:lowerBound "10"^^xsd:decimal .')
    rewritten, findings, _applied = repair_property_aliases(graph, _ontology())
    # qqval:lowerBound tokens are contained in hasLowerBound -> unique rewrite.
    assert rewritten == 1
    assert not findings
    assert (
        URIRef(f"{FACTS}r"),
        URIRef(f"{QQVAL}hasLowerBound"),
        None,
    ) in graph


def test_repair_literal_type_objects_coerces_compact_and_absolute() -> None:
    # The failure shape: `a "domain-ontology:Material"^^xsd:string`
    # instead of `a domain-ontology:Material`.
    graph = _facts(
        f"""
        cd:s a "qqval:Approximate"^^xsd:string .
        cd:t a "{QQVAL}Exact" .
        """
    )
    rewritten, findings, applied = repair_literal_type_objects(graph)
    assert rewritten == 2
    assert not findings
    assert (URIRef(f"{FACTS}s"), RDF.type, URIRef(f"{QQVAL}Approximate")) in graph
    assert (URIRef(f"{FACTS}t"), RDF.type, URIRef(f"{QQVAL}Exact")) in graph
    assert {record.kind for record in applied} == {
        FactsUnitFindingKind.LITERAL_TYPE_OBJECT
    }
    assert not [
        obj for obj in graph.objects(None, RDF.type) if not isinstance(obj, URIRef)
    ]


def test_repair_literal_type_objects_unresolvable_becomes_finding() -> None:
    graph = _facts('cd:s a "unboundprefix:Material"^^xsd:string .')
    rewritten, findings, applied = repair_literal_type_objects(graph)
    assert rewritten == 0
    assert not applied
    assert len(findings) == 1
    assert findings[0].kind is FactsUnitFindingKind.LITERAL_TYPE_OBJECT
    assert findings[0].mandatory


def test_collect_unit_findings_flags_literal_type_objects() -> None:
    graph = _facts('cd:s a "not an iri" .')
    findings = collect_unit_findings(
        graph=graph,
        ontology_graph=_ontology(),
        quarantined=[],
        extraction_text="",
        fact_namespaces=[FACTS],
    )
    literal_type = [
        finding
        for finding in findings
        if finding.kind is FactsUnitFindingKind.LITERAL_TYPE_OBJECT
    ]
    assert len(literal_type) == 1
    assert literal_type[0].mandatory


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


def test_closed_range_suggestions_match_case_exactly() -> None:
    # The facts prompt mandates character-for-character symbol matching; a
    # case-mismatched surface must not produce a suggestion (`unit:M` vs "m").
    rejected = RejectedLiteralTriple(
        subject=f"{FACTS}v1",
        predicate=f"{QQVAL}epistemicQualifier",
        object_lexical="exact",  # individual is qqval:Exact, local name "Exact"
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
    assert quarantine[0].suggestions == []


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


def _bounds_ontology() -> RDFGraph:
    graph = _ontology()
    graph.parse(
        data=f"""
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix qudt: <{QUDT}> .
        @prefix qqval: <{QQVAL}> .
        qudt:numericValue a owl:FunctionalProperty .
        qqval:numericLowerBound a owl:DatatypeProperty, owl:FunctionalProperty .
        qqval:numericUpperBound a owl:DatatypeProperty, owl:FunctionalProperty .
        qqval:numericUncertainty a owl:DatatypeProperty .
        """,
        format="turtle",
    )
    return graph


def _scalar_as_bounds(findings):
    return [
        finding
        for finding in findings
        if finding.kind is FactsUnitFindingKind.SCALAR_AS_BOUNDS
    ]


def test_scalar_as_bounds_flags_equal_functional_bounds() -> None:
    graph = _facts(
        """
        cd:qv qqval:numericLowerBound "523"^^xsd:decimal ;
            qqval:numericUpperBound "523"^^xsd:decimal .
        """
    )
    findings = collect_unit_findings(
        graph=graph,
        ontology_graph=_bounds_ontology(),
        quarantined=[],
        extraction_text="",
        fact_namespaces=[FACTS],
    )
    flagged = _scalar_as_bounds(findings)
    assert len(flagged) == 1
    assert flagged[0].mandatory
    assert flagged[0].value == "523"
    assert "numericLowerBound" in flagged[0].message
    assert "numericUpperBound" in flagged[0].message


def test_scalar_as_bounds_matches_across_datatype_spellings() -> None:
    graph = _facts(
        """
        cd:qv qqval:numericLowerBound "5.0"^^xsd:double ;
            qqval:numericUpperBound "5"^^xsd:decimal .
        """
    )
    findings = collect_unit_findings(
        graph=graph,
        ontology_graph=_bounds_ontology(),
        quarantined=[],
        extraction_text="",
        fact_namespaces=[FACTS],
    )
    assert len(_scalar_as_bounds(findings)) == 1


def test_scalar_as_bounds_ignores_distinct_values() -> None:
    graph = _facts(
        """
        cd:qv qqval:numericLowerBound "500"^^xsd:decimal ;
            qqval:numericUpperBound "550"^^xsd:decimal .
        """
    )
    findings = collect_unit_findings(
        graph=graph,
        ontology_graph=_bounds_ontology(),
        quarantined=[],
        extraction_text="",
        fact_namespaces=[FACTS],
    )
    assert _scalar_as_bounds(findings) == []


def test_scalar_as_bounds_ignores_non_functional_predicates() -> None:
    # numericUncertainty is not functional in the fixture schema.
    graph = _facts(
        """
        cd:qv qudt:numericValue "8.5"^^xsd:decimal ;
            qqval:numericUncertainty "8.5"^^xsd:decimal .
        """
    )
    findings = collect_unit_findings(
        graph=graph,
        ontology_graph=_bounds_ontology(),
        quarantined=[],
        extraction_text="",
        fact_namespaces=[FACTS],
    )
    assert _scalar_as_bounds(findings) == []


def test_scalar_as_bounds_ignores_non_fact_namespace_subjects() -> None:
    graph = _facts(
        """
        qqval:someIndividual qqval:numericLowerBound "3"^^xsd:decimal ;
            qqval:numericUpperBound "3"^^xsd:decimal .
        """
    )
    findings = collect_unit_findings(
        graph=graph,
        ontology_graph=_bounds_ontology(),
        quarantined=[],
        extraction_text="",
        fact_namespaces=[FACTS],
    )
    assert _scalar_as_bounds(findings) == []


def test_scalar_as_bounds_no_ontology_graph_is_silent() -> None:
    graph = _facts(
        """
        cd:qv qqval:numericLowerBound "3"^^xsd:decimal ;
            qqval:numericUpperBound "3"^^xsd:decimal .
        """
    )
    findings = collect_unit_findings(
        graph=graph,
        ontology_graph=None,
        quarantined=[],
        extraction_text="",
        fact_namespaces=[FACTS],
    )
    assert _scalar_as_bounds(findings) == []


def test_fact_namespace_class_in_type_position_is_flagged() -> None:
    graph = _facts("cd:owlSameasLz a cd:Link .")
    findings = collect_unit_findings(
        graph=graph,
        ontology_graph=_ontology(),
        quarantined=[],
        extraction_text="",
        fact_namespaces=[FACTS],
    )
    flagged = [
        finding
        for finding in findings
        if finding.kind is FactsUnitFindingKind.UNKNOWN_TERM
        and finding.predicate == f"{FACTS}Link"
    ]
    assert len(flagged) == 1
    assert flagged[0].mandatory
    assert "class" in flagged[0].message


def test_catalog_class_in_type_position_not_flagged_as_fact_class() -> None:
    graph = _facts("cd:qv a qudt:Unit .")
    findings = collect_unit_findings(
        graph=graph,
        ontology_graph=_ontology(),
        quarantined=[],
        extraction_text="",
        fact_namespaces=[FACTS],
    )
    assert not any(
        finding.predicate == "http://qudt.org/schema/qudt/Unit"
        and "facts/document namespace" in finding.message
        for finding in findings
    )


def _dated_ontology() -> RDFGraph:
    """Ranges exercising the non-numeric and the unsafe-target paths."""
    graph = RDFGraph()
    graph.parse(
        data="""
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
        @prefix ex: <http://example.org/> .
        ex:publishedYear a owl:DatatypeProperty ; rdfs:range xsd:gYear .
        ex:measuredOn a owl:DatatypeProperty ; rdfs:range xsd:date .
        ex:note a owl:DatatypeProperty ; rdfs:range xsd:string .
        ex:reference a owl:DatatypeProperty ; rdfs:range xsd:anyURI .
        ex:amount a owl:DatatypeProperty ; rdfs:range xsd:decimal .
        """,
        format="turtle",
    )
    return graph


def test_normalize_literals_retypes_a_declared_non_numeric_range() -> None:
    """A gYear range receiving an xsd:string was left alone before."""
    graph = _facts(
        'cd:doc ex:publishedYear "2019"^^xsd:string ; ex:measuredOn "2021-04-05" .'
    )
    retyped = normalize_literals_against_schema(graph, _dated_ontology())

    assert retyped == 2
    assert (
        URIRef(f"{FACTS}doc"),
        URIRef("http://example.org/publishedYear"),
        Literal("2019", datatype=XSD.gYear),
    ) in graph
    assert (
        URIRef(f"{FACTS}doc"),
        URIRef("http://example.org/measuredOn"),
        Literal("2021-04-05", datatype=XSD.date),
    ) in graph


def test_normalize_literals_never_retypes_toward_string_or_anyuri() -> None:
    """Every lexical form parses as those, so a range declaring one must not fire.

    Admitting them would let a single sloppy ``rdfs:range`` rewrite correctly
    typed values across the whole graph.
    """
    graph = _facts(
        'cd:doc ex:note "42"^^xsd:decimal ; ex:reference "not-a-uri"^^xsd:decimal .'
    )
    retyped = normalize_literals_against_schema(graph, _dated_ontology())

    assert retyped == 0
    assert (
        URIRef(f"{FACTS}doc"),
        URIRef("http://example.org/note"),
        Literal("42", datatype=XSD.decimal),
    ) in graph


def test_normalize_literals_leaves_language_tagged_literals_alone() -> None:
    """Retyping an rdf:langString would silently discard the language tag."""
    graph = _facts('cd:doc ex:publishedYear "2019"@en .')
    retyped = normalize_literals_against_schema(graph, _dated_ontology())

    assert retyped == 0
    assert (
        URIRef(f"{FACTS}doc"),
        URIRef("http://example.org/publishedYear"),
        Literal("2019", lang="en"),
    ) in graph


def test_normalize_literals_still_promotes_numeric_to_numeric() -> None:
    """The pre-existing integer -> decimal promotion must survive the widening."""
    graph = _facts('cd:doc ex:amount "7"^^xsd:integer .')
    retyped = normalize_literals_against_schema(graph, _dated_ontology())

    assert retyped == 1
    assert (
        URIRef(f"{FACTS}doc"),
        URIRef("http://example.org/amount"),
        Literal("7", datatype=XSD.decimal),
    ) in graph


def test_normalize_literals_does_not_retype_an_unparseable_lexical_form() -> None:
    graph = _facts('cd:doc ex:publishedYear "sometime in 2019" .')
    assert normalize_literals_against_schema(graph, _dated_ontology()) == 0


def test_normalize_literals_never_retypes_toward_boolean_or_time() -> None:
    """Both accept nonsense, so a range declaring one must not drive a retype.

    ``Literal("2019", datatype=xsd:boolean).value`` is ``False`` rather than
    ``None``, and ``"2019"`` parses as ``xsd:time`` 20:19 -- so a parse check
    alone would happily rewrite a year into a boolean or a clock time.
    """
    ontology = RDFGraph()
    ontology.parse(
        data="""
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
        @prefix ex: <http://example.org/> .
        ex:flag a owl:DatatypeProperty ; rdfs:range xsd:boolean .
        ex:at a owl:DatatypeProperty ; rdfs:range xsd:time .
        """,
        format="turtle",
    )
    graph = _facts('cd:doc ex:flag "2019" ; ex:at "2019" .')

    assert normalize_literals_against_schema(graph, ontology) == 0
    for predicate in ("flag", "at"):
        objects = list(
            graph.objects(
                URIRef(f"{FACTS}doc"), URIRef(f"http://example.org/{predicate}")
            )
        )
        assert [str(obj) for obj in objects] == ["2019"]
        literal = objects[0]
        assert isinstance(literal, Literal)
        assert literal.datatype in (None, XSD.string)


def test_normalize_literals_rejects_a_malformed_gregorian_lexical_form() -> None:
    """gYear has no rdflib value parser, so its lexical space is checked here."""
    graph = _facts('cd:doc ex:publishedYear "19" .')
    assert normalize_literals_against_schema(graph, _dated_ontology()) == 0
