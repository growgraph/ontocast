"""The two facts vocabularies and the well-known set they share.

Both modes use the catalog first, then the well-known vocabularies and the
declared namespaces; a well-known term must exist in its vocabulary.
``strict`` stops there: a term of any other vocabulary is ``unknown_term`` at
the unit and ``non_catalog_vocabulary`` is an error at the gate. ``open`` also
admits any other established vocabulary, as warnings.
"""

import pytest

from ontocast.config import FactsValidationConfig
from ontocast.onto.model import (
    FactsUnitFindingKind,
    FactsValidationFindingKind,
    OntologyUnitFindingKind,
)
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.well_known import WELL_KNOWN_VOCABULARIES, is_well_known_term
from ontocast.prompt.facts_guidelines import (
    format_facts_operational_guidelines,
    format_vocabulary_rules,
)
from ontocast.tool.facts_validation import (
    ValidationPolicy,
    collect_unit_findings,
    validate_aggregated_facts,
)
from ontocast.tool.ontology_validation.unit_findings import (
    collect_ontology_unit_findings,
)

pytestmark = pytest.mark.unit

CD = "https://growgraph.dev/facts/"
EX = "https://x.org/onto#"
SCHEMA = "https://schema.org/"
FOAF = "http://xmlns.com/foaf/0.1/"
DBO = "http://dbpedia.org/ontology/"

ONTOLOGY = f"""
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex:   <{EX}> .

ex:Sample a owl:Class ; rdfs:label "sample" .
ex:measuredBy a owl:ObjectProperty ; rdfs:label "measured by" .
"""

# Catalog terms; real well-known terms (FOAF, schema.org, an RDF container
# property); a vocabulary that is not well-known (DBpedia); and a schema.org
# term that does not exist.
FACTS = f"""
@prefix rdf:    <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs:   <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex:     <{EX}> .
@prefix foaf:   <{FOAF}> .
@prefix schema: <{SCHEMA}> .
@prefix dbo:    <{DBO}> .
@prefix cd:     <{CD}> .

cd:s1 a ex:Sample ; rdfs:label "sample 1" ; ex:measuredBy cd:p1 .
cd:p1 a foaf:Person ; rdfs:label "A. Author" ; schema:memberOf cd:lab ;
    rdf:_1 cd:s1 .
cd:lab a dbo:ResearchProject ; rdfs:label "lab" ; dbo:leader cd:p1 .
cd:elderly a schema:Audience ; rdfs:label "the elderly" ;
    schema:vulnerableTo cd:heat .
cd:heat a ex:Sample ; rdfs:label "heat" .
"""


def _graph(data: str) -> RDFGraph:
    graph = RDFGraph()
    graph.parse(data=data, format="turtle")
    return graph


def _unit_findings(policy: ValidationPolicy, facts: str = FACTS):
    return [
        finding
        for finding in collect_unit_findings(
            graph=_graph(facts),
            ontology_graph=_graph(ONTOLOGY),
            quarantined=[],
            extraction_text="",
            fact_namespaces=[CD],
            policy=policy,
        )
        if finding.kind == FactsUnitFindingKind.UNKNOWN_TERM
    ]


def _unknown_terms(policy: ValidationPolicy) -> set[str]:
    return {finding.predicate for finding in _unit_findings(policy)}


# ---------------------------------------------------------------------------
# Defaults and the well-known set
# ---------------------------------------------------------------------------


def test_strict_is_the_default_and_nothing_is_declared_by_default() -> None:
    config = FactsValidationConfig()

    assert config.vocabulary == "strict"
    assert config.additional_standard_namespaces == []
    assert ValidationPolicy().vocabulary == "strict"


def test_the_well_known_set() -> None:
    assert {vocabulary.prefix for vocabulary in WELL_KNOWN_VOCABULARIES} == {
        "rdf", "rdfs", "owl", "xsd", "skos", "dcterms", "dc", "prov", "sosa",
        "ssn", "time", "csvw", "dcat", "org", "vcard", "foaf", "schema", "qudt",
        "unit", "quantitykind",
    }  # fmt: skip


@pytest.mark.parametrize(
    ("iri", "expected"),
    [
        (f"{SCHEMA}memberOf", True),
        ("http://schema.org/memberOf", True),
        (f"{SCHEMA}vulnerableTo", False),
        ("http://www.w3.org/1999/02/22-rdf-syntax-ns#reifies", True),
        ("http://www.w3.org/1999/02/22-rdf-syntax-ns#_7", True),
        ("http://qudt.org/vocab/unit/NanoM", True),
        (f"{DBO}Person", None),
    ],
)
def test_membership(iri: str, expected: bool | None) -> None:
    assert is_well_known_term(iri) is expected


def test_an_unknown_mode_is_refused() -> None:
    with pytest.raises(ValueError):
        format_vocabulary_rules("lenient")


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------


def _guidelines(vocabulary: str, **kwargs) -> str:
    return format_facts_operational_guidelines(
        facts_namespace=CD,
        domain_ontologies_clause="`ex:`",
        jsonld=False,
        vocabulary=vocabulary,
        **kwargs,
    )


@pytest.mark.parametrize("vocabulary", ["strict", "open"])
def test_both_modes_name_the_well_known_vocabularies(vocabulary: str) -> None:
    text = _guidelines(vocabulary)

    for named in (f"`schema:` <{SCHEMA}>", f"`foaf:` <{FOAF}>", "`csvw:`", "`qudt:`"):
        assert named in text
    assert "never invent a term" in text


def test_strict_prompt_admits_no_other_vocabulary() -> None:
    text = _guidelines("strict")

    assert "No other vocabulary is available" in text
    assert "is not extracted" in text
    assert "LAST RESORT" not in text
    assert "another established vocabulary" not in text


def test_open_prompt_admits_other_established_vocabularies() -> None:
    text = _guidelines("open")

    assert "LAST RESORT" in text
    assert "another established vocabulary" in text
    assert "No other vocabulary is available" not in text


def test_declared_namespaces_and_the_carrier_are_named() -> None:
    text = _guidelines(
        "strict",
        additional_namespaces=(DBO,),
        quantity_fallback_vocabulary={"carrier": "ex:hasValue"},
    )

    assert f"<{DBO}>" in text
    assert "to carry it, use ex:hasValue." in text


def test_mode_changes_only_the_vocabulary_rules() -> None:
    strict, open_ = _guidelines("strict"), _guidelines("open")

    for shared in (
        "TWO-NAMESPACE CONTRACT",
        "3. EXTRACT STATEMENTS",
        "A STRUCTURED VALUE",
    ):
        assert shared in strict and shared in open_


# ---------------------------------------------------------------------------
# The unit validator
# ---------------------------------------------------------------------------


def test_strict_flags_other_vocabularies_and_missing_well_known_terms() -> None:
    assert _unknown_terms(ValidationPolicy(vocabulary="strict")) == {
        f"{DBO}ResearchProject",
        f"{DBO}leader",
        f"{SCHEMA}vulnerableTo",
    }


def test_open_flags_only_missing_well_known_terms() -> None:
    assert _unknown_terms(ValidationPolicy(vocabulary="open")) == {
        f"{SCHEMA}vulnerableTo"
    }


def test_a_declared_namespace_joins_the_strict_vocabulary() -> None:
    policy = ValidationPolicy(
        vocabulary="strict", additional_standard_namespaces=(DBO,)
    )

    assert _unknown_terms(policy) == {f"{SCHEMA}vulnerableTo"}


def test_a_missing_well_known_term_gets_the_vocabulary_near_misses() -> None:
    facts = f"""
    @prefix rdfs:   <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix schema: <{SCHEMA}> .
    @prefix cd:     <{CD}> .
    cd:lab a schema:Organization ; rdfs:label "lab" ; schema:hasMember cd:p1 .
    """

    (finding,) = _unit_findings(ValidationPolicy(), facts)

    assert finding.predicate == f"{SCHEMA}hasMember"
    assert "not a term of schema.org" in finding.message
    assert f"{SCHEMA}member" in finding.suggestions


def test_strict_finding_lets_the_renderer_leave_the_statement_out() -> None:
    messages = {
        finding.predicate: finding.message
        for finding in _unit_findings(ValidationPolicy(vocabulary="strict"))
    }

    assert "the vocabulary is strict" in messages[f"{DBO}leader"]


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def _non_catalog(**kwargs) -> dict[str, str]:
    report = validate_aggregated_facts(
        _graph(FACTS), _graph(ONTOLOGY), fact_namespaces=[CD], **kwargs
    )
    return {
        finding.predicate: finding.severity
        for finding in report.findings
        if finding.kind == FactsValidationFindingKind.NON_CATALOG_VOCABULARY
    }


def test_gate_under_strict() -> None:
    severities = _non_catalog(non_catalog_severity="error")

    assert severities[f"{FOAF}Person"] == "warning"
    assert severities[f"{SCHEMA}memberOf"] == "warning"
    assert severities[f"{DBO}leader"] == "error"
    assert severities[f"{SCHEMA}vulnerableTo"] == "error"


def test_gate_under_open_warns_only() -> None:
    assert set(_non_catalog().values()) == {"warning"}


def test_a_declared_namespace_stays_a_warning_under_strict() -> None:
    severities = _non_catalog(
        non_catalog_severity="error", additional_standard_namespaces=(DBO,)
    )

    assert severities[f"{DBO}leader"] == "warning"


# ---------------------------------------------------------------------------
# The ontology render
# ---------------------------------------------------------------------------


def test_an_ontology_render_may_not_mint_in_a_well_known_namespace() -> None:
    inserts = _graph(
        f"""
        @prefix owl:    <http://www.w3.org/2002/07/owl#> .
        @prefix rdfs:   <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix schema: <{SCHEMA}> .
        @prefix ex:     <{EX}> .
        schema:HeatVulnerableGroup a owl:Class ; rdfs:label "heat-vulnerable group" .
        schema:Person rdfs:subClassOf ex:Sample .
        """
    )

    findings = collect_ontology_unit_findings(
        inserts=inserts, deletes=RDFGraph(), snapshot_graph=_graph(ONTOLOGY)
    )

    assert {
        finding.subject
        for finding in findings
        if finding.kind == OntologyUnitFindingKind.FOREIGN_NAMESPACE
    } == {f"{SCHEMA}HeatVulnerableGroup"}
