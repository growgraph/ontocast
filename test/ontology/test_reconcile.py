"""Reduce-time reconciliation: full terminals are the authority, not snapshots.

Under vector-retrieval context the per-unit snapshot is a retrieved subset of
the catalog, so a unit can re-mint a term that exists, unretrieved, in the
remainder — and the per-unit label-collision check (indexed on the snapshot)
is structurally blind to exactly that duplicate. These tests pin the reduce
lane that closes the gap, plus the two write-safety policies that ship with
it: unredeclared deletes are dropped under partial context, and same-IRI
fresh ontologies are union-merged instead of silently last-wins.
"""

from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import OWL, RDF, RDFS, Literal, URIRef

from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_apply import OntologyDelta, apply_partitioned_updates
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.stategraph.helpers import (
    enforce_redeclared_deletes,
    reconcile_fresh_ontologies,
)
from ontocast.tool.ontology_manager import OntologyManager
from ontocast.tool.ontology_validation import (
    apply_minted_duplicate_rewrites,
    detect_minted_duplicates,
)

pytestmark = pytest.mark.unit

ONTO = "https://example.com/onto#"


def _graph(*triples) -> RDFGraph:
    graph = RDFGraph()
    graph.bind("onto", ONTO)
    for triple in triples:
        graph.add(triple)
    return graph


def _terminal() -> RDFGraph:
    sample = URIRef(f"{ONTO}Sample")
    has_part = URIRef(f"{ONTO}hasPart")
    return _graph(
        (sample, RDF.type, OWL.Class),
        (sample, RDFS.label, Literal("Sample")),
        (has_part, RDF.type, OWL.ObjectProperty),
        (has_part, RDFS.label, Literal("has part")),
    )


def test_minted_class_matching_catalog_label_is_detected() -> None:
    minted = URIRef(f"{ONTO}Sample2")
    inserts = _graph(
        (minted, RDF.type, OWL.Class),
        (minted, RDFS.label, Literal("Sample")),
    )
    duplicates = detect_minted_duplicates(inserts, {ONTO: _terminal()})
    assert [(d.minted_iri, d.catalog_iri, d.role) for d in duplicates] == [
        (f"{ONTO}Sample2", f"{ONTO}Sample", "class")
    ]


def test_term_already_in_terminal_is_not_minted() -> None:
    """Adding an axiom about an existing catalog term is not a re-mint."""
    existing = URIRef(f"{ONTO}Sample")
    inserts = _graph((existing, RDFS.comment, Literal("a material sample")))
    assert detect_minted_duplicates(inserts, {ONTO: _terminal()}) == []


def test_ambiguous_surface_is_refused() -> None:
    """A label two catalog terms share cannot identify either of them."""
    terminal = _terminal()
    twin = URIRef(f"{ONTO}SampleTwin")
    terminal.add((twin, RDF.type, OWL.Class))
    terminal.add((twin, RDFS.label, Literal("Sample")))
    minted = URIRef(f"{ONTO}Sample2")
    inserts = _graph(
        (minted, RDF.type, OWL.Class),
        (minted, RDFS.label, Literal("Sample")),
    )
    assert detect_minted_duplicates(inserts, {ONTO: terminal}) == []


def test_role_mismatch_is_refused() -> None:
    """A minted property never reconciles onto a catalog class."""
    minted = URIRef(f"{ONTO}sampleProp")
    inserts = _graph(
        (minted, RDF.type, OWL.ObjectProperty),
        (minted, RDFS.label, Literal("Sample")),  # collides with the CLASS label
    )
    assert detect_minted_duplicates(inserts, {ONTO: _terminal()}) == []


def test_rewrite_substitutes_subject_and_object_positions() -> None:
    minted = URIRef(f"{ONTO}Sample2")
    other = URIRef(f"{ONTO}Detector")
    inserts = _graph(
        (minted, RDF.type, OWL.Class),
        (minted, RDFS.label, Literal("Sample")),
        (other, RDF.type, OWL.Class),
        (other, RDFS.label, Literal("Detector")),
        (other, RDFS.subClassOf, minted),  # object position must follow
    )
    duplicates = detect_minted_duplicates(inserts, {ONTO: _terminal()})
    rewritten = apply_minted_duplicate_rewrites(inserts, duplicates)

    catalog = URIRef(f"{ONTO}Sample")
    assert rewritten == 3
    assert (other, RDFS.subClassOf, catalog) in inserts
    assert not list(inserts.triples((minted, None, None)))
    assert not list(inserts.triples((None, None, minted)))


def test_unredeclared_deletes_are_dropped_redeclared_kept() -> None:
    sample = URIRef(f"{ONTO}Sample")
    has_part = URIRef(f"{ONTO}hasPart")
    delta = OntologyDelta(
        inserts=_graph((sample, RDFS.label, Literal("Material sample"))),
        deletes=_graph(
            (sample, RDFS.label, Literal("Sample")),  # redeclared -> kept
            (has_part, RDFS.label, Literal("has part")),  # bare -> dropped
        ),
    )
    dropped = enforce_redeclared_deletes(delta)
    assert dropped == 1
    assert (sample, RDFS.label, Literal("Sample")) in delta.deletes
    assert not list(delta.deletes.triples((has_part, None, None)))


def _fresh(iri: str, *triples) -> Ontology:
    return Ontology(graph=_graph(*triples), iri=iri, ontology_id="fresh-onto")


def test_same_iri_fresh_ontologies_union_merge_instead_of_last_wins() -> None:
    a_term = URIRef(f"{ONTO}A")
    b_term = URIRef(f"{ONTO}B")
    first = _fresh(f"{ONTO.rstrip('#')}", (a_term, RDF.type, OWL.Class))
    second = _fresh(f"{ONTO.rstrip('#')}", (b_term, RDF.type, OWL.Class))

    merged, metrics = reconcile_fresh_ontologies([first, second])

    assert metrics["fresh_ontologies_merged"] == 1
    assert len(merged) == 1
    assert (a_term, RDF.type, OWL.Class) in merged[0].graph
    assert (b_term, RDF.type, OWL.Class) in merged[0].graph
    # A root version: the transient per-unit artifacts never enter the
    # catalog, so lineage must not point at their hashes.
    assert merged[0].parent_hashes == []
    assert merged[0].hash and merged[0].hash != first.hash


def test_cross_iri_fresh_overlap_is_counted_not_merged() -> None:
    shared = (URIRef(f"{ONTO}Sample"),)
    first = _fresh(
        "https://example.com/onto-a",
        (URIRef("https://example.com/onto-a#Sample"), RDF.type, OWL.Class),
        (URIRef("https://example.com/onto-a#Sample"), RDFS.label, Literal("Sample")),
    )
    second = _fresh(
        "https://example.com/onto-b",
        (URIRef("https://example.com/onto-b#Sample"), RDF.type, OWL.Class),
        (URIRef("https://example.com/onto-b#Sample"), RDFS.label, Literal("Sample")),
    )
    _ = shared

    merged, metrics = reconcile_fresh_ontologies([first, second])

    assert len(merged) == 2, "distinct IRIs stay separate artifacts"
    assert metrics["fresh_minted_duplicates"] == 1


def test_deletes_absent_from_the_terminal_are_counted() -> None:
    """Snapshot/terminal divergence (a stale vector index) must be visible.

    A delete names a triple the unit's snapshot contained; the apply target is
    the freshest terminal. When the terminal has moved on, DELETE DATA is a
    silent no-op -- the counter is the only trace.
    """
    sample = URIRef(f"{ONTO}Sample")
    base = Ontology(
        graph=_graph((sample, RDFS.label, Literal("Sample"))),
        iri=ONTO.rstrip("#"),
        ontology_id="onto",
    )

    def _normalize(units, tools, *, base_ontology, require_base, delete_graph=None):
        return base_ontology, [], None

    _, metrics, _ = apply_partitioned_updates(
        {},
        ontology_manager=cast(
            OntologyManager,
            SimpleNamespace(get_freshest_terminal_ontology_by_iri=lambda iri: base),
        ),
        normalize_units_fn=_normalize,
        tools=None,
        partitioned_deletes={
            base.iri: _graph(
                (sample, RDFS.label, Literal("Sample")),  # present -> matches
                (sample, RDFS.comment, Literal("gone")),  # absent -> no-op
            )
        },
    )

    assert metrics["apply_delete_triples"] == 2
    assert metrics["apply_deletes_no_match"] == 1
