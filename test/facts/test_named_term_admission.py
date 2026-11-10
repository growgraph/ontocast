"""A facts prompt defines every catalog term it names.

The conformance chapter prints all rules of a shape it selects, and a class
note may quote another term, so the prompt can tell the model to use a catalog
term its ``# ONTOLOGY`` chapter never defines. Those terms' definitions are
admitted into the snapshot; nothing outside the catalog is.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from rdflib import RDF, RDFS, Literal, URIRef

from ontocast.onto.enum import OntologyAssemblyMode
from ontocast.onto.ontology_snapshot import OntologySnapshot
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import BudgetTracker
from ontocast.stategraph.context_resolver import (
    UnitOntologyContext,
    admit_named_catalog_terms,
)
from ontocast.tool.ontology_manager import OntologyManager
from test.ontology_helpers import stamped_ontology

pytestmark = pytest.mark.unit

EX = "https://example.org/mat#"

CATALOG = f"""
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix mat:  <{EX}> .

<https://example.org/mat> a owl:Ontology .

mat:Portion a owl:Class ; rdfs:label "material portion" ;
    rdfs:comment "A measured amount of a material; see mat:hasSupplier." .
mat:hasAmount a owl:DatatypeProperty ; rdfs:label "has amount" ;
    rdfs:domain mat:Portion ;
    rdfs:comment "Amount of the portion, as stated in a recipe." .
mat:hasSupplier a owl:DatatypeProperty ; rdfs:label "has supplier" ;
    rdfs:domain mat:Portion .
mat:Unrelated a owl:Class ; rdfs:label "unrelated" .
"""


def _manager() -> OntologyManager:
    manager = OntologyManager()
    manager.add_ontology(stamped_ontology(CATALOG), skip_vector_index=True)
    return manager


def _snapshot_with_portion_only() -> UnitOntologyContext:
    graph = RDFGraph()
    graph.bind("mat", EX)
    portion = URIRef(EX + "Portion")
    graph.add((portion, RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class")))
    graph.add((portion, RDFS.label, Literal("material portion")))
    graph.add(
        (
            portion,
            RDFS.comment,
            Literal("A measured amount of a material; see mat:hasSupplier."),
        )
    )
    snapshot = OntologySnapshot.from_graph(
        graph,
        source_iris=["https://example.org/mat"],
        assembly_mode=OntologyAssemblyMode.SELECTED_VECTOR_SEARCH_ENSEMBLE,
        partial=True,
    )
    return UnitOntologyContext(snapshot=snapshot, writable_iris=[])


def _tools(manager: OntologyManager, chapter_terms: set[str]) -> Any:
    return SimpleNamespace(
        ontology_manager=manager,
        shapes_chapter_terms=lambda context_terms: frozenset(chapter_terms),
    )


def test_term_definitions_copy_what_defines_a_term() -> None:
    graph = _manager().term_definitions([EX + "hasAmount", "https://nowhere.org/x"])

    amount = URIRef(EX + "hasAmount")
    assert graph.value(amount, RDFS.domain) == URIRef(EX + "Portion")
    assert graph.value(amount, RDFS.label) == Literal("has amount")
    assert {str(s) for s in graph.subjects()} == {EX + "hasAmount"}
    assert dict(graph.namespaces()).get("mat") == URIRef(EX)


def test_a_shape_path_and_a_note_mention_are_admitted() -> None:
    ctx = _snapshot_with_portion_only()
    tracker = BudgetTracker()

    admitted = admit_named_catalog_terms(
        ctx,
        _tools(_manager(), {EX + "Portion", EX + "hasAmount"}),
        budget_tracker=tracker,
    )

    subjects = {str(s) for s in admitted.snapshot.graph.subjects()}
    # hasAmount: a path the chapter names. hasSupplier: quoted by Portion's note.
    assert {EX + "hasAmount", EX + "hasSupplier"} <= subjects
    assert EX + "Unrelated" not in subjects
    assert tracker.counters["ctx/admitted_named_terms"] == 2
    # The shared document context is never changed in place.
    assert {str(s) for s in ctx.snapshot.graph.subjects()} == {EX + "Portion"}
    assert admitted.snapshot.partial is True


def test_a_term_outside_the_catalog_is_not_admitted() -> None:
    ctx = _snapshot_with_portion_only()
    ctx.snapshot.graph.remove((None, RDFS.comment, None))

    admitted = admit_named_catalog_terms(
        ctx, _tools(_manager(), {"https://schema.org/name"})
    )

    assert admitted is ctx


def test_an_empty_snapshot_admits_nothing() -> None:
    ctx = UnitOntologyContext(
        snapshot=OntologySnapshot.empty(
            assembly_mode=OntologyAssemblyMode.SELECTED_VECTOR_SEARCH_ENSEMBLE
        )
    )

    assert admit_named_catalog_terms(ctx, _tools(_manager(), {EX + "hasAmount"})) is ctx
