"""Tests for deterministic literal-object checks on rendered facts graphs."""

import pytest
from rdflib import Literal, Namespace
from rdflib.namespace import OWL, RDF, RDFS, XSD

from ontocast.onto.enum import LLMGraphFormat
from ontocast.onto.rdfgraph import RDFGraph, format_quarantine_for_prompt
from ontocast.tool.validate import (
    RDFGraphConnectivityValidator,
    partition_object_property_literal_triples,
)

pytestmark = pytest.mark.unit

QUDT = Namespace("http://qudt.org/schema/qudt/")
UNIT = Namespace("http://qudt.org/vocab/unit/")
CD = Namespace("https://growgraph.dev/facts/")
EX = Namespace("https://example.org/onto#")


def _unit_ontology() -> RDFGraph:
    onto = RDFGraph()
    onto.bind("qudt", QUDT)
    onto.add((QUDT.unit, RDF.type, OWL.ObjectProperty))
    onto.add((QUDT.unit, RDFS.range, QUDT.Unit))
    onto.add((QUDT.numericValue, RDF.type, OWL.DatatypeProperty))
    onto.add((EX.hasStatus, RDFS.range, EX.Status))
    onto.add((EX.hasCode, RDFS.range, XSD.string))
    onto.add((EX.hasNote, RDFS.range, RDFS.Literal))
    onto.add((EX.hasShade, RDFS.range, EX.Shade))
    onto.add((EX.Shade, RDF.type, RDFS.Datatype))
    return onto


def test_object_property_literal_is_quarantined_with_reason() -> None:
    facts = RDFGraph()
    facts.add((CD.v1, QUDT.unit, Literal("meV")))
    facts.add((CD.v1, QUDT.numericValue, Literal("30", datatype=XSD.decimal)))

    clean, rejected = partition_object_property_literal_triples(facts, _unit_ontology())
    assert len(rejected) == 1
    item = rejected[0]
    assert item.object_lexical == "meV"
    assert item.expected_range == str(QUDT.Unit)
    assert item.reason is not None
    assert (CD.v1, QUDT.numericValue, Literal("30", datatype=XSD.decimal)) in clean
    assert (CD.v1, QUDT.unit, Literal("meV")) not in clean


def test_class_range_without_object_property_declaration_also_quarantines() -> None:
    facts = RDFGraph()
    facts.add((CD.x, EX.hasStatus, Literal("active")))
    _, rejected = partition_object_property_literal_triples(facts, _unit_ontology())
    assert len(rejected) == 1
    assert rejected[0].expected_range == str(EX.Status)


def test_literal_compatible_ranges_and_unknown_predicates_pass() -> None:
    facts = RDFGraph()
    facts.add((CD.x, EX.hasCode, Literal("A1")))  # xsd:string range
    facts.add((CD.x, EX.hasNote, Literal("note")))  # rdfs:Literal range
    facts.add((CD.x, EX.hasShade, Literal("dark")))  # declared rdfs:Datatype range
    facts.add((CD.x, EX.undeclared, Literal("free")))  # unknown predicate
    facts.add((CD.x, RDFS.label, Literal("x")))  # annotation vocab

    clean, rejected = partition_object_property_literal_triples(facts, _unit_ontology())
    assert rejected == []
    assert len(clean) == 5


def test_iri_object_on_object_property_is_kept() -> None:
    facts = RDFGraph()
    facts.add((CD.v1, QUDT.unit, UNIT.MilliEV))
    clean, rejected = partition_object_property_literal_triples(facts, _unit_ontology())
    assert rejected == []
    assert (CD.v1, QUDT.unit, UNIT.MilliEV) in clean


def test_quarantine_prompt_includes_expected_range_hint() -> None:
    facts = RDFGraph()
    facts.add((CD.v1, QUDT.unit, Literal("meV")))
    _, rejected = partition_object_property_literal_triples(facts, _unit_ontology())

    turtle = format_quarantine_for_prompt(rejected, LLMGraphFormat.TURTLE)
    assert '"meV"' in turtle
    assert "^^" not in turtle  # no datatype on the plain literal
    assert str(QUDT.Unit) in turtle

    jsonld = format_quarantine_for_prompt(rejected, LLMGraphFormat.JSONLD)
    assert '"@value": "meV"' in jsonld
    assert "@type" not in jsonld
    assert str(QUDT.Unit) in jsonld


def test_validate_predicates_flags_literal_on_class_range() -> None:
    graph = RDFGraph()
    graph.add((EX.hasStatus, RDFS.label, Literal("has status")))
    graph.add((EX.hasStatus, RDFS.range, EX.Status))
    graph.add((CD.x, EX.hasStatus, Literal("active")))

    result = RDFGraphConnectivityValidator(graph).validate_predicates()
    assert not result.domain_range_consistent
    assert any("IRI expected" in v for v in result.domain_range_violations)


def test_validate_predicates_accepts_literal_on_datatype_range() -> None:
    graph = RDFGraph()
    graph.add((EX.hasCode, RDFS.label, Literal("has code")))
    graph.add((EX.hasCode, RDFS.range, XSD.string))
    graph.add((CD.x, EX.hasCode, Literal("A1")))

    result = RDFGraphConnectivityValidator(graph).validate_predicates()
    assert result.domain_range_consistent


# ---------------------------------------------------------------------------
# UNKNOWN_TERM against the full catalog, not the unit's retrieved snapshot
# ---------------------------------------------------------------------------


#: A catalog namespace of its own: ``example.org`` is the placeholder
#: namespace the validator forbids outright, which is a different finding.
MS = Namespace("https://vocab.test/matsci#")


def _site_snapshot() -> RDFGraph:
    """A retrieved snapshot that carries the B-site property but not the A-site one."""
    onto = RDFGraph()
    onto.bind("ms", MS)
    onto.add((MS.Perovskite, RDF.type, OWL.Class))
    onto.add((MS.hasBSiteComponent, RDF.type, OWL.ObjectProperty))
    onto.add((MS.hasBSiteComponent, RDFS.domain, MS.Perovskite))
    return onto


def _site_facts() -> RDFGraph:
    facts = RDFGraph()
    facts.bind("ms", MS)
    facts.add((CD.p1, RDF.type, MS.Perovskite))
    facts.add((CD.p1, MS.hasASiteComponent, CD.cs))
    facts.add((CD.p1, MS.hasBSiteComponent, CD.pb))
    return facts


def _unknown_terms(findings) -> dict[str, list[str]]:
    from ontocast.onto.model import FactsUnitFindingKind

    return {
        finding.predicate: finding.suggestions
        for finding in findings
        if finding.kind == FactsUnitFindingKind.UNKNOWN_TERM
    }


def test_a_term_the_full_catalog_declares_is_not_unknown() -> None:
    """The snapshot is a retrieved subset; a term it omitted is still a term.

    Reporting it unknown -- with the look-alike the snapshot does carry as the
    suggested replacement -- hands the critic the very substitution the
    LLM-free alias repair refuses to make.
    """
    from ontocast.tool.facts_validation import collect_unit_findings

    def unknown(full_catalog_terms):
        return _unknown_terms(
            collect_unit_findings(
                graph=_site_facts(),
                ontology_graph=_site_snapshot(),
                quarantined=[],
                extraction_text="",
                fact_namespaces=[str(CD)],
                coverage_limit=0,
                full_catalog_terms=full_catalog_terms,
            )
        )

    assert str(MS.hasASiteComponent) in unknown(None), (
        "without the catalog the snapshot is all there is to check against"
    )
    assert unknown({str(MS.hasASiteComponent)}) == {}


def test_a_genuinely_unknown_term_is_never_offered_a_token_swap() -> None:
    """Suggestions qualify by token containment, exactly as the repair does.

    ``hasASiteComponent`` and ``hasBSiteComponent`` differ in one token and
    score high on string similarity; neither contains the other, so the
    B-site property must not be suggested for the A-site one.
    """
    from ontocast.tool.facts_validation import collect_unit_findings

    unknown = _unknown_terms(
        collect_unit_findings(
            graph=_site_facts(),
            ontology_graph=_site_snapshot(),
            quarantined=[],
            extraction_text="",
            fact_namespaces=[str(CD)],
            coverage_limit=0,
        )
    )

    assert unknown[str(MS.hasASiteComponent)] == []


def test_a_containing_candidate_is_still_suggested() -> None:
    """``value`` is contained in ``numericValue``: that is a near-miss."""
    from ontocast.tool.facts_validation import collect_unit_findings

    onto = RDFGraph()
    onto.add((MS.numericValue, RDF.type, OWL.DatatypeProperty))
    facts = RDFGraph()
    facts.add((CD.v1, MS.value, Literal("3")))
    facts.add((CD.v2, MS.value, Literal("4")))

    unknown = _unknown_terms(
        collect_unit_findings(
            graph=facts,
            ontology_graph=onto,
            quarantined=[],
            extraction_text="",
            fact_namespaces=[str(CD)],
            coverage_limit=0,
        )
    )

    assert unknown[str(MS.value)] == [str(MS.numericValue)]


# ---------------------------------------------------------------------------
# Numeric coverage: measurements and bare numbers as separate findings
# ---------------------------------------------------------------------------


def _coverage(text: str, graph: RDFGraph, **kwargs):
    from ontocast.onto.model import FactsUnitFindingKind
    from ontocast.tool.facts_validation import collect_unit_findings

    return {
        finding.facet: finding
        for finding in collect_unit_findings(
            graph=graph,
            ontology_graph=None,
            quarantined=[],
            extraction_text=text,
            fact_namespaces=[str(CD)],
            **kwargs,
        )
        if finding.kind == FactsUnitFindingKind.NUMERIC_COVERAGE
    }


_TEXT = "a shift of 96 meV was seen in 3 samples at 77 K"


def test_coverage_is_split_into_measurements_and_bare_numbers() -> None:
    findings = _coverage(_TEXT, RDFGraph())

    assert set(findings) == {"measurements", "unclassified"}
    measurements = findings["measurements"]
    assert measurements.value == "96, 77"
    assert "96 meV" in measurements.message and "77 K" in measurements.message
    assert "a shift of 96 meV" in measurements.message, "context is carried"
    assert findings["unclassified"].value == "3"
    assert not measurements.mandatory and not findings["unclassified"].mandatory


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("off", (False, False)),
        ("measurements", (True, False)),
        ("all", (True, True)),
        (False, (False, False)),
        (True, (True, True)),
    ],
)
def test_coverage_mode_sets_which_finding_is_mandatory(mode, expected) -> None:
    findings = _coverage(_TEXT, RDFGraph(), coverage_mandatory=mode)

    assert (
        findings["measurements"].mandatory,
        findings["unclassified"].mandatory,
    ) == expected


def test_a_label_number_no_longer_silences_the_measurement_finding() -> None:
    """The placeholder-with-a-label answer to a coverage finding."""
    graph = RDFGraph()
    graph.add((CD.ignored_token_1, RDFS.label, Literal("96")))

    findings = _coverage(_TEXT, graph)

    assert "96" in findings["measurements"].value


def test_an_extracted_value_leaves_the_finding() -> None:
    graph = RDFGraph()
    graph.add((CD.v1, QUDT.numericValue, Literal("96", datatype=XSD.decimal)))
    graph.add((CD.v2, QUDT.numericValue, Literal("77", datatype=XSD.decimal)))

    findings = _coverage(_TEXT, graph)

    assert "measurements" not in findings
    assert findings["unclassified"].value == "3"
