"""Deterministic validation of per-unit ontology deltas.

These checks run on the *delta*, never the working graph: the working graph is
snapshot + delta, and validating it would attribute every pre-existing catalog
defect to this unit. The finding kinds pinned here are the deterministic lane
the ontology critic gate will eventually rest on; until then they are recorded
as telemetry and injected into the critic prompt (shadow mode).
"""

import pytest
from rdflib import OWL, RDF, RDFS, BNode, Literal, URIRef

from ontocast.onto.model import OntologyUnitFindingKind, TripleFix
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.ontology_validation import (
    collect_ontology_unit_findings,
    count_fixes_targeting_snapshot,
)

pytestmark = pytest.mark.unit

ONTO = "https://example.com/onto#"
FACTS = "https://growgraph.dev/cd#"


def _graph(*triples, bind: dict[str, str] | None = None) -> RDFGraph:
    graph = RDFGraph()
    for prefix, namespace in (bind or {"onto": ONTO}).items():
        graph.bind(prefix, namespace)
    for triple in triples:
        graph.add(triple)
    return graph


def _snapshot() -> RDFGraph:
    sample = URIRef(f"{ONTO}Sample")
    device = URIRef(f"{ONTO}Device")
    has_part = URIRef(f"{ONTO}hasPart")
    return _graph(
        (sample, RDF.type, OWL.Class),
        (sample, RDFS.label, Literal("Sample")),
        (device, RDF.type, OWL.Class),
        (device, RDFS.label, Literal("Device")),
        (sample, RDFS.subClassOf, device),
        (has_part, RDF.type, OWL.ObjectProperty),
        (has_part, RDFS.label, Literal("has part")),
        (has_part, RDFS.domain, device),
    )


def _kinds(findings) -> set[OntologyUnitFindingKind]:
    return {finding.kind for finding in findings}


def _collect(inserts: RDFGraph, deletes: RDFGraph | None = None, **kwargs):
    return collect_ontology_unit_findings(
        inserts=inserts,
        deletes=deletes if deletes is not None else RDFGraph(),
        snapshot_graph=kwargs.pop("snapshot_graph", _snapshot()),
        fact_namespaces=kwargs.pop("fact_namespaces", [FACTS]),
        **kwargs,
    )


def test_clean_extension_yields_no_findings() -> None:
    new_class = URIRef(f"{ONTO}Detector")
    inserts = _graph(
        (new_class, RDF.type, OWL.Class),
        (new_class, RDFS.label, Literal("Detector")),
        (new_class, RDFS.subClassOf, URIRef(f"{ONTO}Device")),
    )
    assert _collect(inserts) == []


def test_term_minted_under_unowned_namespace_is_mandatory() -> None:
    """The reduce partition silently drops these; the finding predicts it."""
    foreign = URIRef("https://elsewhere.example/vocab#Widget")
    inserts = _graph(
        (foreign, RDF.type, OWL.Class),
        (foreign, RDFS.label, Literal("Widget")),
    )
    findings = _collect(inserts)
    assert _kinds(findings) == {OntologyUnitFindingKind.FOREIGN_NAMESPACE}
    assert all(finding.mandatory for finding in findings)


def test_fresh_create_path_has_no_namespace_authority() -> None:
    """An empty seed means any namespace is fair game — no false mandatory."""
    minted = URIRef("https://brand-new.example/onto#Thing")
    inserts = _graph(
        (minted, RDF.type, OWL.Class),
        (minted, RDFS.label, Literal("Thing")),
    )
    findings = _collect(inserts, snapshot_graph=None)
    assert findings == []


def test_example_org_placeholder_flagged_even_on_fresh_path() -> None:
    minted = URIRef("http://example.org/onto#Thing")
    inserts = _graph(
        (minted, RDF.type, OWL.Class),
        (minted, RDFS.label, Literal("Thing")),
    )
    findings = _collect(inserts, snapshot_graph=None)
    assert _kinds(findings) == {OntologyUnitFindingKind.FOREIGN_NAMESPACE}


def test_ontology_term_in_facts_namespace_is_mandatory() -> None:
    minted = URIRef(f"{FACTS}NewClass")
    inserts = _graph(
        (minted, RDF.type, OWL.Class),
        (minted, RDFS.label, Literal("New class")),
    )
    findings = _collect(inserts)
    assert _kinds(findings) == {OntologyUnitFindingKind.FOREIGN_NAMESPACE}
    assert "instances only" in findings[0].message


def test_degenerate_restriction_stub_flagged() -> None:
    stub = BNode()
    new_class = URIRef(f"{ONTO}Sensor")
    inserts = _graph(
        (new_class, RDF.type, OWL.Class),
        (new_class, RDFS.label, Literal("Sensor")),
        (new_class, RDFS.subClassOf, stub),
        (stub, RDF.type, OWL.Restriction),
    )
    findings = _collect(inserts)
    assert OntologyUnitFindingKind.DEGENERATE_RESTRICTION in _kinds(findings)


def test_complete_restriction_not_flagged() -> None:
    restriction = BNode()
    new_class = URIRef(f"{ONTO}Sensor")
    inserts = _graph(
        (new_class, RDF.type, OWL.Class),
        (new_class, RDFS.label, Literal("Sensor")),
        (new_class, RDFS.subClassOf, restriction),
        (restriction, RDF.type, OWL.Restriction),
        (restriction, OWL.onProperty, URIRef(f"{ONTO}hasPart")),
        (restriction, OWL.someValuesFrom, URIRef(f"{ONTO}Sample")),
    )
    findings = _collect(inserts)
    assert OntologyUnitFindingKind.DEGENERATE_RESTRICTION not in _kinds(findings)


def test_new_term_without_label_is_mandatory() -> None:
    unlabeled = URIRef(f"{ONTO}Mystery")
    inserts = _graph((unlabeled, RDF.type, OWL.Class))
    findings = _collect(inserts)
    assert _kinds(findings) == {OntologyUnitFindingKind.MISSING_LABEL}


def test_snapshot_label_satisfies_the_label_check() -> None:
    """Adding an axiom about an already-labeled catalog term is not a gap."""
    sample = URIRef(f"{ONTO}Sample")
    inserts = _graph((sample, RDFS.subClassOf, URIRef(f"{ONTO}Device")))
    findings = _collect(inserts)
    assert OntologyUnitFindingKind.MISSING_LABEL not in _kinds(findings)


def test_subclass_cycle_through_snapshot_is_mandatory() -> None:
    """Snapshot says Sample ⊑ Device; inserting Device ⊑ Sample closes a cycle."""
    inserts = _graph(
        (URIRef(f"{ONTO}Device"), RDFS.subClassOf, URIRef(f"{ONTO}Sample"))
    )
    findings = _collect(inserts)
    assert OntologyUnitFindingKind.SUBCLASS_CYCLE in _kinds(findings)


def test_self_subclass_is_mandatory() -> None:
    inserts = _graph(
        (URIRef(f"{ONTO}Sample"), RDFS.subClassOf, URIRef(f"{ONTO}Sample"))
    )
    findings = _collect(inserts)
    assert OntologyUnitFindingKind.SUBCLASS_CYCLE in _kinds(findings)


def test_catalog_class_used_as_predicate_is_role_confusion() -> None:
    inserts = _graph(
        (URIRef(f"{ONTO}Detector2"), URIRef(f"{ONTO}Sample"), Literal("x"))
    )
    findings = _collect(inserts)
    assert OntologyUnitFindingKind.ROLE_CONFUSION in _kinds(findings)


def test_catalog_property_used_as_class_is_role_confusion() -> None:
    inserts = _graph((URIRef(f"{ONTO}Widget"), RDF.type, URIRef(f"{ONTO}hasPart")))
    findings = _collect(inserts)
    assert OntologyUnitFindingKind.ROLE_CONFUSION in _kinds(findings)


def test_duplicate_label_is_advisory_with_reuse_suggestion() -> None:
    duplicate = URIRef(f"{ONTO}Sample2")
    inserts = _graph(
        (duplicate, RDF.type, OWL.Class),
        (duplicate, RDFS.label, Literal("Sample")),
    )
    findings = _collect(inserts)
    collisions = [
        finding
        for finding in findings
        if finding.kind == OntologyUnitFindingKind.LABEL_COLLISION
    ]
    assert len(collisions) == 1
    assert not collisions[0].mandatory
    assert collisions[0].suggestions == [f"{ONTO}Sample"]


def test_min_two_against_functional_is_cardinality_contradiction() -> None:
    prop = URIRef(f"{ONTO}hasExactlyOne")
    restriction = BNode()
    inserts = _graph(
        (prop, RDF.type, OWL.ObjectProperty),
        (prop, RDF.type, OWL.FunctionalProperty),
        (prop, RDFS.label, Literal("has exactly one")),
        (URIRef(f"{ONTO}Sample"), RDFS.subClassOf, restriction),
        (restriction, RDF.type, OWL.Restriction),
        (restriction, OWL.onProperty, prop),
        (restriction, OWL.minCardinality, Literal(2)),
    )
    findings = _collect(inserts)
    assert OntologyUnitFindingKind.CARDINALITY_CONTRADICTION in _kinds(findings)


def test_delete_of_unowned_catalog_term_is_mandatory() -> None:
    deletes = _graph((URIRef(f"{ONTO}Device"), RDFS.label, Literal("Device")))
    findings = _collect(_graph(), deletes=deletes)
    assert OntologyUnitFindingKind.FOREIGN_DELETE in _kinds(findings)


def test_delete_with_redeclaration_is_a_replace_not_a_foreign_delete() -> None:
    device = URIRef(f"{ONTO}Device")
    deletes = _graph((device, RDFS.label, Literal("Device")))
    inserts = _graph((device, RDFS.label, Literal("Measurement device")))
    findings = _collect(inserts, deletes=deletes)
    assert OntologyUnitFindingKind.FOREIGN_DELETE not in _kinds(findings)


def _fix(incorrect_value: str | None) -> TripleFix:
    return TripleFix(
        text_fragment="a sample",
        action="REPLACE",
        severity="important",
        explanation="rework the term",
        incorrect_value=incorrect_value,
    )


def test_count_fixes_targeting_snapshot_counts_untouched_catalog_terms() -> None:
    fixes = [
        _fix(f"<{ONTO}Device> rdfs:label 'Device' ."),  # snapshot-only, full IRI
        _fix("onto:Sample rdfs:comment 'x' ."),  # snapshot-only, qname
        _fix(f"<{ONTO}Detector> a owl:Class ."),  # the unit's own insert
        _fix(None),  # nothing to match
    ]
    count = count_fixes_targeting_snapshot(
        fixes, _snapshot(), insert_subjects={f"{ONTO}Detector"}
    )
    assert count == 2


def test_count_fixes_targeting_snapshot_without_snapshot_is_zero() -> None:
    assert (
        count_fixes_targeting_snapshot([_fix("onto:Sample a owl:Class .")], None, set())
        == 0
    )
