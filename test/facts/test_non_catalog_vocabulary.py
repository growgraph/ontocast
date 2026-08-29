"""Detecting the silent fallback out of the catalog.

Reproduces the case6 shape. The catalog carried a qualified-quantity module
with ranges and epistemic qualifiers, retrieval admitted none of it, and the
renderer took the prompt's documented escape hatch: ``qudt:QuantityValue``
attached with ``schema:hasPart``. The result validated cleanly and scored in
the nineties while expressing every measurement in vocabulary no downstream
query against this catalog will match. A fallback firing is evidence about
*retrieval*, so it has to leave a trace instead of shipping silently.
"""

from __future__ import annotations

from ontocast.onto.model import FactsValidationFinding, FactsValidationFindingKind
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.facts_validation import validate_aggregated_facts

FACTS = "https://growgraph.dev/facts/"
MATSCI = "https://growgraph.dev/ontologies/matsci#"

_ONTOLOGY = f"""
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix matsci: <{MATSCI}> .

matsci:SuperlatticeSample a owl:Class ; rdfs:label "superlattice sample" .
matsci:describesMaterial a owl:ObjectProperty .
"""

# What the renderer actually emitted: catalog class, catalog property, then
# schema:hasPart + qudt:QuantityValue for the measurement it had no terms for.
_FACTS = f"""
@prefix rdfs:   <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:    <http://www.w3.org/2001/XMLSchema#> .
@prefix schema: <https://schema.org/> .
@prefix qudt:   <http://qudt.org/schema/qudt/> .
@prefix matsci: <{MATSCI}> .
@prefix cd:     <{FACTS}> .

cd:sl_sample_1 a matsci:SuperlatticeSample ;
    rdfs:label "CsPbBr3 nanocrystal superlattice" ;
    matsci:describesMaterial cd:cspbbr3 ;
    schema:hasPart cd:dimension_value_1 .

cd:dimension_value_1 a qudt:QuantityValue ;
    rdfs:label "SL dimension" ;
    qudt:numericValue "70"^^xsd:decimal .
"""


def _report(facts: str = _FACTS):
    graph = RDFGraph()
    graph.parse(data=facts, format="turtle")
    ontology = RDFGraph()
    ontology.parse(data=_ONTOLOGY, format="turtle")
    return validate_aggregated_facts(graph, ontology, fact_namespaces=[FACTS])


def _non_catalog(report) -> dict[str, FactsValidationFinding]:
    return {
        finding.predicate: finding
        for finding in report.findings
        if finding.kind == FactsValidationFindingKind.NON_CATALOG_VOCABULARY
    }


def test_fallback_vocabulary_is_reported() -> None:
    flagged = _non_catalog(_report())

    assert "https://schema.org/hasPart" in flagged
    assert "http://qudt.org/schema/qudt/QuantityValue" in flagged
    assert "http://qudt.org/schema/qudt/numericValue" in flagged


def test_catalog_terms_are_not_flagged() -> None:
    flagged = _non_catalog(_report())

    assert f"{MATSCI}describesMaterial" not in flagged
    assert f"{MATSCI}SuperlatticeSample" not in flagged


def test_scaffolding_is_not_flagged() -> None:
    """rdfs:label and rdf:type are in every facts graph; flagging them is noise."""
    flagged = _non_catalog(_report())

    assert not [term for term in flagged if "rdf-schema#" in term]
    assert not [term for term in flagged if "22-rdf-syntax-ns#" in term]


def test_chunk_metadata_predicates_are_not_flagged() -> None:
    """The pipeline's own provenance scaffolding is not renderer vocabulary.

    ``_add_unit_metadata`` mints ``schema:position``, ``schema:identifier`` and
    the section-label terms on every chunk node, and they are in the graph the
    gate validates. Only the ``prov:`` prefix used to be skipped, so any catalog
    without schema.org got a warning per predicate accusing the renderer of
    improvising terms it never emitted.
    """
    facts = f"""
    @prefix rdfs:   <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix xsd:    <http://www.w3.org/2001/XMLSchema#> .
    @prefix prov:   <http://www.w3.org/ns/prov#> .
    @prefix schema: <https://schema.org/> .
    @prefix oc:     <https://growgraph.dev/ontocast#> .
    @prefix matsci: <{MATSCI}> .
    @prefix cd:     <{FACTS}> .

    cd:sl_sample_1 a matsci:SuperlatticeSample ;
        rdfs:label "sample" ;
        matsci:describesMaterial cd:cspbbr3 .

    cd:chunk_0 a prov:Entity, schema:Text ;
        prov:generatedAtTime "2026-08-09T00:00:00"^^xsd:dateTime ;
        schema:position 0 ;
        schema:identifier "abc123" ;
        schema:articleSection "results" ;
        oc:sectionLabelSource "heading_keyword" ;
        oc:sectionLabelConfidence "0.9"^^xsd:decimal .
    """
    flagged = _non_catalog(_report(facts))

    assert flagged == {}


def test_findings_are_warnings_not_errors() -> None:
    """Telemetry about context assembly must not drive the un-merge repair."""
    for finding in _non_catalog(_report()).values():
        assert finding.severity == "warning"


def test_clean_extraction_reports_nothing() -> None:
    clean = f"""
    @prefix rdfs:   <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix matsci: <{MATSCI}> .
    @prefix cd:     <{FACTS}> .

    cd:sl_sample_1 a matsci:SuperlatticeSample ;
        rdfs:label "sample" ;
        matsci:describesMaterial cd:cspbbr3 .
    """
    assert _non_catalog(_report(clean)) == {}


def test_no_ontology_context_reports_nothing() -> None:
    """Without a context there is nothing to be outside of."""
    graph = RDFGraph()
    graph.parse(data=_FACTS, format="turtle")
    report = validate_aggregated_facts(graph, None, fact_namespaces=[FACTS])

    assert _non_catalog(report) == {}
