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
    # Clause stated once in 1a; later items refer back without re-interpolating.
    assert guidelines.count("the domain ontologies") == 1
    assert "domain ontology namespace(s) above" in guidelines


def test_criticise_facts_include_specificity_checklist() -> None:
    assert "6b. Specificity" in evaluation_instruction
    assert "rdfs:subPropertyOf" in evaluation_instruction
    assert "REPLACE with the narrower descendant" in evaluation_instruction


def test_facts_guidelines_include_object_property_iri_rule() -> None:
    guidelines = format_facts_operational_guidelines(
        facts_namespace="https://example.com/facts/",
        domain_ontologies_clause="the domain ontologies",
        jsonld=False,
    )
    assert "8a. OBJECT PROPERTIES TAKE IRIs, NOT STRINGS" in guidelines
    assert "skos:notation" in guidelines
    # Case sensitivity is the rule's whole point (meV vs MeV).
    assert "character-for-character" in guidelines


def test_render_ontology_units_rule_is_iri_first() -> None:
    from ontocast.prompt.render_ontology import general_ontology_instruction

    assert "unit individual" in general_ontology_instruction
    assert 'schema:duration schema:unitCode "DAY"' not in general_ontology_instruction
