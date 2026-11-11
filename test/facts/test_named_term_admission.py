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
from rdflib import OWL, RDF, RDFS, Literal, URIRef

from ontocast.onto.enum import OntologyAssemblyMode
from ontocast.onto.model import OntologyPlan, OntologyPlanTerm
from ontocast.onto.ontology_snapshot import OntologySnapshot
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import BudgetTracker
from ontocast.stategraph.context_resolver import (
    UnitOntologyContext,
    admit_named_catalog_terms,
    admit_planned_terms,
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


def _plan(*terms: OntologyPlanTerm) -> OntologyPlan:
    return OntologyPlan(
        terms=list(terms), prefixes={"mat": EX, "schema": "https://schema.org/"}
    )


def _reuse(iri: str, kind: str, label: str, definition: str = "") -> OntologyPlanTerm:
    return OntologyPlanTerm.model_validate(
        {
            "kind": kind,
            "label": label,
            "local_name": iri.rsplit("/", 1)[-1].rsplit("#", 1)[-1],
            "iri": iri,
            "decision": "reuse",
            "definition": definition,
        }
    )


def test_planned_reuses_are_admitted_from_the_catalog_and_the_plan() -> None:
    sub_event = "https://schema.org/subEvent"
    plan = _plan(
        _reuse(
            sub_event,
            "object_property",
            "sub event",
            "Relates a competition to an event that forms part of it.",
        ),
        _reuse(EX + "hasAmount", "datatype_property", "has amount"),
        _reuse(EX + "Portion", "class", "material portion"),
    )
    ctx = _snapshot_with_portion_only()
    tracker = BudgetTracker()

    admitted = admit_planned_terms(
        ctx, plan, _tools(_manager(), set()), budget_tracker=tracker
    )

    graph = admitted.snapshot.graph
    node = URIRef(sub_event)
    assert graph.value(node, RDF.type) == OWL.ObjectProperty
    assert graph.value(node, RDFS.label) == Literal("sub event")
    assert "forms part of it" in str(graph.value(node, RDFS.comment))
    # A catalog term comes with its catalog definition, not the plan's.
    assert graph.value(URIRef(EX + "hasAmount"), RDFS.domain) == URIRef(EX + "Portion")
    assert dict(graph.namespaces()).get("schema") == URIRef("https://schema.org/")
    # Portion was already defined; only the two missing terms are counted.
    assert tracker.counters["ctx/admitted_planned_terms"] == 2
    assert {str(s) for s in ctx.snapshot.graph.subjects()} == {EX + "Portion"}


def test_minted_terms_and_a_missing_plan_add_nothing() -> None:
    ctx = _snapshot_with_portion_only()
    minted = OntologyPlanTerm.model_validate(
        {
            "kind": "class",
            "label": "season",
            "local_name": "Season",
            "iri": EX + "Season",
            "decision": "mint",
        }
    )

    assert admit_planned_terms(ctx, None, _tools(_manager(), set())) is ctx
    assert admit_planned_terms(ctx, _plan(minted), _tools(_manager(), set())) is ctx


@pytest.mark.anyio
async def test_a_unit_resolving_its_own_context_admits_the_planned_reuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plan that mints nothing leaves no merged context; units still get its reuses."""
    from ontocast.onto.enum import OntologyContextMode
    from ontocast.stategraph import atomic
    from ontocast.stategraph.unit_context import UnitLoopContext

    resolved = _snapshot_with_portion_only()

    async def resolve(*_args: Any, **_kwargs: Any) -> UnitOntologyContext:
        return resolved

    monkeypatch.setattr(atomic, "resolve_unit_ontology_context", resolve)
    sub_event = "https://schema.org/subEvent"
    plan = _plan(_reuse(sub_event, "object_property", "sub event"))
    tracker = BudgetTracker()
    context = UnitLoopContext(
        ontology_context_mode=OntologyContextMode.SELECTED_VECTOR_SEARCH_ONTOLOGY,
        ontology_plan=plan,
        budget_tracker=tracker,
    )
    unit_state: Any = SimpleNamespace(content_unit=None)

    await atomic._apply_facts_ontology_context(
        unit_state, context, _tools(_manager(), set())
    )

    graph = unit_state.ontology_snapshot.graph
    assert graph.value(URIRef(sub_event), RDF.type) == OWL.ObjectProperty
    assert tracker.counters["ctx/admitted_planned_terms"] == 1
