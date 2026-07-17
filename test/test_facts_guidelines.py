"""Smoke tests for facts operational guidelines prompt content."""

from ontocast.prompt.criticise_facts import evaluation_instruction
from ontocast.prompt.facts_guidelines import format_facts_operational_guidelines


def test_facts_guidelines_include_specificity_rule() -> None:
    guidelines = format_facts_operational_guidelines(
        facts_namespace="https://example.com/facts/",
        domain_ontologies_clause="the domain ontologies",
        jsonld=False,
    )
    assert "1d. SPECIFICITY RULE" in guidelines
    assert "rdfs:subPropertyOf" in guidelines
    assert "ex:hasComponent" in guidelines


def test_criticise_facts_include_specificity_checklist() -> None:
    assert "6b. Specificity" in evaluation_instruction
    assert "rdfs:subPropertyOf" in evaluation_instruction
    assert "REPLACE with the narrower descendant" in evaluation_instruction
