"""Tests for closed-vs-open namespace rules, term exemptions, and repairs.

Root cause: the catalog *referenced*
``qudt:QuantityValue``/``qudt:unit`` (making qudt a catalog namespace under the
old all-positions rule) while ``qudt:numericValue`` appeared only in prose — so
every unit graph carrying the canonical scalar property received a mandatory
UNKNOWN_TERM finding suggesting a **class** as the replacement, and repair
renders obeyed by deleting correct numeric values or re-encoding scalars as
degenerate bound pairs.
"""

import pytest
from rdflib import URIRef

from ontocast.onto.model import FactsUnitFindingKind, format_findings_for_prompt
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.facts_validation import (
    ValidationPolicy,
    collect_declared_namespaces,
    collect_unit_findings,
    expand_vocabulary_terms,
    promote_degenerate_bounds,
    promote_degenerate_bounds_from_vocabulary,
    shacl_catalog_contradictions,
)
from ontocast.util.graph_metrics import facts_graph_shape_metrics

pytestmark = pytest.mark.unit

QUDT = "http://qudt.org/schema/qudt/"
QQ = "https://x.org/qqval#"
CD = "https://growgraph.dev/facts#"

# The qqval pattern in miniature: qudt terms referenced, never declared.
ONTOLOGY = f"""
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
@prefix qudt: <{QUDT}> .
@prefix qq:   <{QQ}> .

qq:QualifiedValue a owl:Class ;
    rdfs:subClassOf qudt:QuantityValue ;
    rdfs:subClassOf [ a owl:Restriction ; owl:onProperty qudt:unit ;
                      owl:minCardinality 1 ] .
qq:numericLowerBound a owl:DatatypeProperty ; rdfs:range xsd:decimal .
qq:numericUpperBound a owl:DatatypeProperty ; rdfs:range xsd:decimal .
qq:lowerBoundInclusive a owl:DatatypeProperty ; rdfs:range xsd:boolean .
"""

FACTS = f"""
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
@prefix qudt: <{QUDT}> .
@prefix qq:   <{QQ}> .
@prefix cd:   <{CD}> .

cd:v1 a qq:QualifiedValue ;
    rdfs:label "PL peak: 2.25 eV"@en ;
    qudt:numericValue "2.25"^^xsd:decimal ;
    qudt:unit cd:ev .
"""

VOCAB = {
    "value_class": "qudt:QuantityValue",
    "numeric_value": "qudt:numericValue",
    "unit": "qudt:unit",
}


def _graph(data: str) -> RDFGraph:
    graph = RDFGraph()
    graph.parse(data=data, format="turtle")
    return graph


def _unit_findings(facts: RDFGraph, ontology: RDFGraph, **kwargs):
    kwargs.setdefault("extraction_text", "")
    return collect_unit_findings(
        graph=facts,
        ontology_graph=ontology,
        quarantined=[],
        fact_namespaces=[CD],
        coverage_limit=0,
        **kwargs,
    )


def _unknown_terms(findings) -> set[str]:
    return {
        finding.predicate
        for finding in findings
        if finding.kind == FactsUnitFindingKind.UNKNOWN_TERM
    }


class TestReferencedOnlyNamespacesAreOpen:
    def test_declared_namespaces_exclude_referenced_only(self) -> None:
        namespaces = collect_declared_namespaces(_graph(ONTOLOGY))
        assert QQ in namespaces
        assert QUDT not in namespaces

    def test_qudt_numeric_value_is_not_flagged(self) -> None:
        """The exact live pathology: the canonical property survives."""
        findings = _unit_findings(_graph(FACTS), _graph(ONTOLOGY))
        assert _unknown_terms(findings) == set()

    def test_near_miss_in_declared_namespace_still_flagged(self) -> None:
        facts = _graph(FACTS + f"\ncd:v1 <{QQ}lowerBond> '1.0'^^xsd:decimal .")
        findings = _unit_findings(facts, _graph(ONTOLOGY))
        assert _unknown_terms(findings) == {f"{QQ}lowerBond"}


class TestRoleAwareSuggestions:
    def test_predicate_never_gets_a_class_candidate(self) -> None:
        facts = _graph(FACTS + f"\ncd:v1 <{QQ}QualifiedValu> '1.0'^^xsd:decimal .")
        findings = _unit_findings(facts, _graph(ONTOLOGY))
        flagged = [
            finding
            for finding in findings
            if finding.kind == FactsUnitFindingKind.UNKNOWN_TERM
        ]
        assert flagged, "the near-miss predicate must be flagged"
        for finding in flagged:
            assert f"{QQ}QualifiedValue" not in finding.suggestions

    def test_unknown_term_message_forbids_deletion(self) -> None:
        facts = _graph(FACTS + f"\ncd:v1 <{QQ}lowerBond> '1.0'^^xsd:decimal .")
        findings = _unit_findings(facts, _graph(ONTOLOGY))
        (finding,) = [
            item for item in findings if item.kind == FactsUnitFindingKind.UNKNOWN_TERM
        ]
        assert "Do NOT delete" in finding.message

    def test_prompt_block_carries_rewrite_contract(self) -> None:
        facts = _graph(FACTS + f"\ncd:v1 <{QQ}lowerBond> '1.0'^^xsd:decimal .")
        findings = _unit_findings(facts, _graph(ONTOLOGY))
        block = format_findings_for_prompt(findings)
        assert "Never resolve a finding by deleting" in block


class TestFallbackVocabularyExemption:
    def test_fallback_terms_exempt_even_in_declared_namespace(self) -> None:
        # Declare one qudt term so the namespace closes, then confirm the
        # configured fallback vocabulary still passes.
        ontology = _graph(ONTOLOGY + "\nqudt:ucumCode a owl:DatatypeProperty .")
        assert QUDT in collect_declared_namespaces(ontology)
        findings = _unit_findings(
            _graph(FACTS),
            ontology,
            policy=ValidationPolicy(quantity_fallback_vocabulary=VOCAB),
        )
        assert _unknown_terms(findings) == set()

    def test_expand_vocabulary_terms_handles_curies_and_iris(self) -> None:
        terms = expand_vocabulary_terms(
            {"a": "qudt:numericValue", "b": f"{QQ}numericLowerBound", "c": ""},
            _graph(ONTOLOGY),
        )
        assert terms == {f"{QUDT}numericValue", f"{QQ}numericLowerBound"}


class TestShaclCatalogContradictions:
    SHAPES = f"""
    @prefix sh:   <http://www.w3.org/ns/shacl#> .
    @prefix qudt: <{QUDT}> .
    @prefix qq:   <{QQ}> .

    qq:Shape a sh:NodeShape ;
        sh:targetClass qq:QualifiedValue ;
        sh:property [ sh:path qudt:numericValue ; sh:minCount 1 ] ;
        sh:property [ sh:path qq:numericLowerBound ; sh:minCount 0 ] .
    """

    def test_contradiction_reported_when_namespace_closed(self) -> None:
        ontology = _graph(ONTOLOGY + "\nqudt:ucumCode a owl:DatatypeProperty .")
        contradictions = shacl_catalog_contradictions(_graph(self.SHAPES), ontology)
        assert contradictions == [f"{QUDT}numericValue"]

    def test_fallback_vocabulary_clears_the_contradiction(self) -> None:
        ontology = _graph(ONTOLOGY + "\nqudt:ucumCode a owl:DatatypeProperty .")
        contradictions = shacl_catalog_contradictions(
            _graph(self.SHAPES),
            ontology,
            policy=ValidationPolicy(quantity_fallback_vocabulary=VOCAB),
        )
        assert contradictions == []

    def test_open_namespace_is_no_contradiction(self) -> None:
        contradictions = shacl_catalog_contradictions(
            _graph(self.SHAPES), _graph(ONTOLOGY)
        )
        assert contradictions == []

    def test_a_term_the_snapshot_omitted_is_not_a_contradiction(self) -> None:
        """The gate sees the union of retrieved snapshots, not the catalog.

        A shape may require a property the catalog declares but no unit
        retrieved. The unit validator judges unknown terms against the whole
        catalog, so the gate's check must too, or it reports a contradiction
        no unit ever raised.
        """
        ontology = _graph(ONTOLOGY + "\nqudt:ucumCode a owl:DatatypeProperty .")
        contradictions = shacl_catalog_contradictions(
            _graph(self.SHAPES),
            ontology,
            catalog_terms={f"{QUDT}numericValue"},
        )
        assert contradictions == []

    def test_contract_exemption_clears_the_contradiction(self) -> None:
        """Shape-required terms are exempt from UNKNOWN_TERM per unit, so
        the same exemption must apply here."""
        ontology = _graph(ONTOLOGY + "\nqudt:ucumCode a owl:DatatypeProperty .")
        contradictions = shacl_catalog_contradictions(
            _graph(self.SHAPES),
            ontology,
            policy=ValidationPolicy(contract_exempt_terms=(f"{QUDT}numericValue",)),
        )
        assert contradictions == []


class TestLabelOnlyNumber:
    def test_unit_bearing_node_with_label_number_is_mandatory(self) -> None:
        facts = _graph(
            f"""
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix qudt: <{QUDT}> .
            @prefix cd:   <{CD}> .
            cd:v2 rdfs:label "Optical bandgap: ~2.25 eV"@en ; qudt:unit cd:ev .
            """
        )
        findings = _unit_findings(
            facts,
            _graph(ONTOLOGY),
            policy=ValidationPolicy(quantity_fallback_vocabulary=VOCAB),
        )
        (finding,) = [
            item
            for item in findings
            if item.kind == FactsUnitFindingKind.LABEL_ONLY_NUMBER
        ]
        assert finding.mandatory
        assert "2.25" in finding.value
        assert f"{QUDT}numericValue" in finding.suggestions

    def test_structured_value_silences_the_finding(self) -> None:
        findings = _unit_findings(
            _graph(FACTS),
            _graph(ONTOLOGY),
            policy=ValidationPolicy(quantity_fallback_vocabulary=VOCAB),
        )
        assert not [
            item
            for item in findings
            if item.kind == FactsUnitFindingKind.LABEL_ONLY_NUMBER
        ]

    def test_inactive_without_unit_vocabulary(self) -> None:
        facts = _graph(
            f"""
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix qudt: <{QUDT}> .
            @prefix cd:   <{CD}> .
            cd:v2 rdfs:label "2.25 eV"@en ; qudt:unit cd:ev .
            """
        )
        findings = _unit_findings(facts, _graph(ONTOLOGY))
        assert not [
            item
            for item in findings
            if item.kind == FactsUnitFindingKind.LABEL_ONLY_NUMBER
        ]


class TestPromoteDegenerateBounds:
    def _bounds_graph(self, low: str, high: str, extra: str = "") -> RDFGraph:
        return _graph(
            f"""
            @prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
            @prefix qudt: <{QUDT}> .
            @prefix qq:   <{QQ}> .
            @prefix cd:   <{CD}> .
            cd:v1 qq:numericLowerBound "{low}"^^xsd:decimal ;
                  qq:numericUpperBound "{high}"^^xsd:decimal {extra} .
            """
        )

    def _promote(self, graph: RDFGraph) -> int:
        return promote_degenerate_bounds(
            graph,
            numeric_value_property=f"{QUDT}numericValue",
            lower_bound_property=f"{QQ}numericLowerBound",
            upper_bound_property=f"{QQ}numericUpperBound",
            inclusive_flag_properties=[f"{QQ}lowerBoundInclusive"],
        )

    def test_equal_bounds_become_scalar(self) -> None:
        graph = self._bounds_graph("96", "96.0")
        assert self._promote(graph) == 1
        subject = URIRef(f"{CD}v1")
        assert (subject, URIRef(f"{QUDT}numericValue"), None) in graph
        assert (subject, URIRef(f"{QQ}numericLowerBound"), None) not in graph
        assert (subject, URIRef(f"{QQ}numericUpperBound"), None) not in graph

    def test_real_range_untouched(self) -> None:
        graph = self._bounds_graph("84", "85")
        assert self._promote(graph) == 0

    def test_exclusive_flag_blocks_promotion(self) -> None:
        graph = self._bounds_graph("96", "96", extra="; qq:lowerBoundInclusive false")
        assert self._promote(graph) == 0

    def test_existing_scalar_blocks_promotion(self) -> None:
        graph = self._bounds_graph(
            "96", "96", extra='; qudt:numericValue "96"^^xsd:decimal'
        )
        assert self._promote(graph) == 0

    def test_vocabulary_wiring_requires_all_three_roles(self) -> None:
        graph = self._bounds_graph("96", "96")
        assert promote_degenerate_bounds_from_vocabulary(graph, None, VOCAB) == 0, (
            "no bound roles configured -> off"
        )
        full = {
            **VOCAB,
            "lower_bound": f"{QQ}numericLowerBound",
            "upper_bound": f"{QQ}numericUpperBound",
        }
        assert promote_degenerate_bounds_from_vocabulary(graph, None, full) == 1


class TestGraphShapeMetrics:
    def test_components_and_isolated(self) -> None:
        graph = _graph(
            f"""
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix cd:   <{CD}> .
            @prefix ex:   <https://x.org/schema#> .
            cd:a ex:linksTo cd:b .
            cd:c ex:usesUnit ex:sharedIndividual .
            cd:d ex:usesUnit ex:sharedIndividual .
            cd:e rdfs:label "island"@en .
            cd:f a ex:Thing .
            """
        )
        metrics = facts_graph_shape_metrics(graph, [CD])
        assert metrics.nodes == 6
        # a-b directly; c-d through the shared catalog individual.
        assert metrics.largest_component == 2
        assert metrics.isolated_nodes == 2
        assert metrics.components == 4
        assert metrics.edges == 1


class TestUnitSymbolCaseMismatch:
    """Two catalog units whose symbols differ only by letter case."""

    ONTOLOGY = f"""
    @prefix owl:  <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
    @prefix qudt: <{QUDT}> .
    @prefix qq:   <{QQ}> .

    qq:QualifiedValue a owl:Class .
    qq:Unit a owl:Class .
    qudt:unit a owl:ObjectProperty ; rdfs:range qq:Unit .
    qudt:numericValue a owl:DatatypeProperty ; rdfs:range xsd:decimal .
    qq:small a qq:Unit ; rdfs:label "small unit" ; qudt:symbol "xU" .
    qq:big   a qq:Unit ; rdfs:label "big unit"   ; qudt:symbol "XU" .
    """

    def _facts(self, unit: str) -> RDFGraph:
        return _graph(
            f"""
            @prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
            @prefix qudt: <{QUDT}> .
            @prefix qq:   <{QQ}> .
            @prefix cd:   <{CD}> .
            cd:v1 a qq:QualifiedValue ;
                qudt:numericValue "96"^^xsd:decimal ;
                qudt:unit qq:{unit} .
            """
        )

    def _findings(self, unit: str, text: str):
        return [
            item
            for item in _unit_findings(
                self._facts(unit),
                _graph(self.ONTOLOGY),
                policy=ValidationPolicy(quantity_fallback_vocabulary=VOCAB),
                extraction_text=text,
            )
            if item.kind == FactsUnitFindingKind.UNIT_SYMBOL_CASE_MISMATCH
        ]

    def test_case_folded_symbol_match_is_mandatory_and_suggests_the_exact_unit(
        self,
    ) -> None:
        (finding,) = self._findings("big", "a shift of 96 xU was measured")
        assert finding.mandatory
        assert finding.value == "xU"
        assert finding.suggestions == [f"{QQ}small"]
        assert "case" in finding.message

    def test_exact_symbol_match_is_silent(self) -> None:
        assert self._findings("big", "a shift of 96 XU was measured") == []

    def test_value_absent_from_the_text_is_silent(self) -> None:
        assert self._findings("big", "a shift of 12 xU was measured") == []

    def test_a_case_variant_with_no_competing_unit_is_a_spelling_not_an_error(
        self,
    ) -> None:
        """Only one unit declares the symbol: the text merely spelt it in
        another case, and there is no other unit the value could belong to."""
        ontology = _graph(
            self.ONTOLOGY.replace(
                'qq:small a qq:Unit ; rdfs:label "small unit" ; qudt:symbol "xU" .', ""
            )
        )
        findings = [
            item
            for item in _unit_findings(
                self._facts("big"),
                ontology,
                policy=ValidationPolicy(quantity_fallback_vocabulary=VOCAB),
                extraction_text="a shift of 96 xU was measured",
            )
            if item.kind == FactsUnitFindingKind.UNIT_SYMBOL_CASE_MISMATCH
        ]
        assert findings == []

    def test_a_label_spelt_with_a_capital_is_not_a_symbol(self) -> None:
        ontology = _graph(
            self.ONTOLOGY.replace('qudt:symbol "XU"', 'rdfs:altLabel "XU"')
        )
        findings = [
            item
            for item in _unit_findings(
                self._facts("big"),
                ontology,
                policy=ValidationPolicy(quantity_fallback_vocabulary=VOCAB),
                extraction_text="a shift of 96 xU was measured",
            )
            if item.kind == FactsUnitFindingKind.UNIT_SYMBOL_CASE_MISMATCH
        ]
        assert findings == []
