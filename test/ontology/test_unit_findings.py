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
    build_catalog_surface_index,
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


def _of_kind(findings, kind: OntologyUnitFindingKind) -> list:
    return [finding for finding in findings if finding.kind == kind]


def _class(name: str, label: str, parent: URIRef) -> list[tuple]:
    term = URIRef(f"{ONTO}{name}")
    return [
        (term, RDF.type, OWL.Class),
        (term, RDFS.label, Literal(label)),
        (term, RDFS.subClassOf, parent),
    ]


def _deep_chain(length: int) -> RDFGraph:
    """``L{length} ⊂ … ⊂ L1 ⊂ Sample``; Sample is one below Device, a root."""
    triples: list[tuple] = []
    parent = URIRef(f"{ONTO}Sample")
    for level in range(1, length + 1):
        triples += _class(f"L{level}", f"Level {chr(64 + level)}", parent)
        parent = URIRef(f"{ONTO}L{level}")
    return _graph(*triples)


def test_minted_class_beyond_the_depth_bound_is_mandatory() -> None:
    # L6 sits 7 steps below Device: L6 ⊂ … ⊂ L1 ⊂ Sample ⊂ Device.
    findings = _of_kind(
        _collect(_deep_chain(6), max_class_depth=6),
        OntologyUnitFindingKind.HIERARCHY_TOO_DEEP,
    )
    assert [finding.subject for finding in findings] == [f"{ONTO}L6"]
    assert findings[0].mandatory
    assert findings[0].value == "7"
    assert f"<{ONTO}Device>" in findings[0].message


def test_a_class_within_the_depth_bound_is_clean() -> None:
    findings = _collect(_deep_chain(6), max_class_depth=7)
    assert not _of_kind(findings, OntologyUnitFindingKind.HIERARCHY_TOO_DEEP)


def test_three_bare_siblings_one_word_apart_are_a_value_set() -> None:
    parent = URIRef(f"{ONTO}Sample")
    inserts = _graph(
        *_class("CesiumSample", "Cesium sample", parent),
        *_class("FormamidiniumSample", "Formamidinium sample", parent),
        *_class("MethylammoniumSample", "Methylammonium sample", parent),
    )
    findings = _of_kind(_collect(inserts), OntologyUnitFindingKind.VALUE_SET_SIBLINGS)
    assert len(findings) == 1
    assert findings[0].mandatory
    assert findings[0].subject == str(parent)
    assert findings[0].value == "* sample"
    assert findings[0].suggestions == [
        f"{ONTO}CesiumSample",
        f"{ONTO}FormamidiniumSample",
        f"{ONTO}MethylammoniumSample",
    ]


def test_siblings_with_structure_of_their_own_are_not_a_value_set() -> None:
    parent = URIRef(f"{ONTO}Sample")
    inserts = _graph(
        *_class("CesiumSample", "Cesium sample", parent),
        *_class("FormamidiniumSample", "Formamidinium sample", parent),
        *_class("MethylammoniumSample", "Methylammonium sample", parent),
        (URIRef(f"{ONTO}hasCation"), RDFS.domain, URIRef(f"{ONTO}CesiumSample")),
    )
    findings = _collect(inserts)
    assert not _of_kind(findings, OntologyUnitFindingKind.VALUE_SET_SIBLINGS)


def test_two_siblings_are_not_a_value_set() -> None:
    parent = URIRef(f"{ONTO}Sample")
    inserts = _graph(
        *_class("CesiumSample", "Cesium sample", parent),
        *_class("FormamidiniumSample", "Formamidinium sample", parent),
    )
    findings = _collect(inserts)
    assert not _of_kind(findings, OntologyUnitFindingKind.VALUE_SET_SIBLINGS)


def test_a_leaf_with_a_specific_label_is_advisory() -> None:
    inserts = _graph(*_class("Sample3", "Sample 3", URIRef(f"{ONTO}Sample")))
    findings = _of_kind(_collect(inserts), OntologyUnitFindingKind.SPECIFIC_LABEL_LEAF)
    assert len(findings) == 1
    assert not findings[0].mandatory
    assert findings[0].facet == "digit"


def test_a_specific_label_on_a_used_class_is_silent() -> None:
    term = URIRef(f"{ONTO}Sample3")
    inserts = _graph(
        *_class("Sample3", "Sample 3", URIRef(f"{ONTO}Sample")),
        (URIRef(f"{ONTO}hasPart"), RDFS.range, term),
    )
    findings = _collect(inserts)
    assert not _of_kind(findings, OntologyUnitFindingKind.SPECIFIC_LABEL_LEAF)


def _catalog_index():
    other = URIRef("https://example.com/other#plRedShiftProperty")
    holder = URIRef("https://example.com/other#SampleHolder")
    return build_catalog_surface_index(
        [
            _snapshot(),
            _graph(
                (other, RDF.type, OWL.ObjectProperty),
                (holder, RDF.type, OWL.Class),
                (holder, RDFS.label, Literal("Sample holder")),
            ),
        ]
    )


def test_a_term_named_like_an_unshown_catalog_term_is_a_duplicate() -> None:
    minted = URIRef(f"{ONTO}PLRedShiftProperty")
    holder = URIRef(f"{ONTO}Holder")
    inserts = _graph(
        (minted, RDF.type, OWL.ObjectProperty),
        (minted, RDFS.label, Literal("photoluminescence red shift")),
        (holder, RDF.type, OWL.Class),
        (holder, RDFS.label, Literal("sample HOLDER")),
        (holder, RDFS.subClassOf, URIRef(f"{ONTO}Device")),
    )
    findings = _of_kind(
        _collect(inserts, catalog_index=_catalog_index()),
        OntologyUnitFindingKind.CATALOG_DUPLICATE,
    )
    assert {(f.subject, f.facet, tuple(f.suggestions)) for f in findings} == {
        (
            str(minted),
            "local_name",
            ("https://example.com/other#plRedShiftProperty",),
        ),
        (str(holder), "label", ("https://example.com/other#SampleHolder",)),
    }
    assert all(finding.mandatory for finding in findings)


def test_an_exact_snapshot_label_is_left_to_label_collision() -> None:
    duplicate = URIRef(f"{ONTO}Sample2")
    inserts = _graph(
        (duplicate, RDF.type, OWL.Class),
        (duplicate, RDFS.label, Literal("Sample")),
    )
    kinds = _kinds(_collect(inserts, catalog_index=_catalog_index()))
    assert OntologyUnitFindingKind.LABEL_COLLISION in kinds
    assert OntologyUnitFindingKind.CATALOG_DUPLICATE not in kinds


def test_a_role_crossed_name_is_not_a_duplicate() -> None:
    minted = URIRef(f"{ONTO}SampleHolder")
    inserts = _graph(
        (minted, RDF.type, OWL.ObjectProperty),
        (minted, RDFS.label, Literal("holds sample")),
    )
    findings = _collect(inserts, catalog_index=_catalog_index())
    assert not _of_kind(findings, OntologyUnitFindingKind.CATALOG_DUPLICATE)
