"""An ontology critic's fixes are kept or undone one at a time.

The driver is shared with the facts loop; these tests run it with the ontology
phase and its real findings, so a fix that moves a term out of the home
namespace is caught by the finding it raises, not by a stub.
"""

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import OWL, RDF, RDFS, Literal, URIRef

from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import Status
from ontocast.onto.identity import OntologyIdentity
from ontocast.onto.model import TripleFix
from ontocast.onto.ontology_snapshot import OntologySnapshot
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.unit_states import UnitOntologyState
from ontocast.stategraph.atomic import ONTOLOGY_PHASE, _apply_critic_patch
from ontocast.tool.atomic import AtomicToolBox
from ontocast.tool.facts_validation import CriticPatchPolicy

pytestmark = pytest.mark.unit

HOME = OntologyIdentity(
    ontology_id="sports", iri="https://sports.test/onto", title="Sports"
)
CATALOG_NS = "https://catalog.test/onto#"


def _state() -> UnitOntologyState:
    snapshot_graph = RDFGraph()
    material = URIRef(f"{CATALOG_NS}Organisation")
    snapshot_graph.add((material, RDF.type, OWL.Class))
    snapshot_graph.add((material, RDFS.label, Literal("Organisation")))
    snapshot = OntologySnapshot(graph=snapshot_graph)
    state = UnitOntologyState(
        content_unit=ContentUnit(
            text="Clubs play in leagues.",
            index=0,
            doc_iri=URIRef("https://example.org/doc/d1"),
        ),
        ontology_snapshot=snapshot,
        home_ontology=HOME,
    )
    state.working_graph = snapshot.graph.copy()
    state.status = Status.SUCCESS
    return state


def _atomic() -> AtomicToolBox:
    return cast(
        AtomicToolBox,
        SimpleNamespace(
            ontology_critic_passes=1,
            catalog_terms=lambda: set(),
            ontology_patch_policy=CriticPatchPolicy(),
            ontology_acceptance_policy=None,
            ontology_max_class_depth=6,
            validation_policy=None,
            catalog_surface_index=lambda: None,
        ),
    )


def _add(subject: str, label: str) -> TripleFix:
    return TripleFix(
        text_fragment="Clubs",
        action="ADD",
        severity="critical",
        correct_value=(
            f"<{subject}> <{RDF.type}> <{OWL.Class}> .\n"
            f'<{subject}> <{RDFS.label}> "{label}" .'
        ),
        explanation="test fix",
    )


def test_a_fix_minting_outside_the_home_is_undone_and_the_others_stay() -> None:
    """The critic once re-minted a home vocabulary under another namespace; the
    namespace finding now makes that fix a regression on its own."""
    state = _state()
    home_term = URIRef(f"{HOME.namespace}Club")
    stray = URIRef("https://elsewhere.test/onto#League")
    state.suggestions.actionable_fixes = [
        _add(str(home_term), "Club"),
        _add(str(stray), "League"),
    ]

    outcome = _apply_critic_patch(
        state,
        _atomic(),
        replace(ONTOLOGY_PHASE),
        render_attempt=1,
        pass_index=1,
        mandatory_before=0,
    )

    assert outcome.applied == 1
    assert outcome.rolled_back == 1
    inserts = state.build_delta().inserts
    assert (home_term, RDF.type, OWL.Class) in inserts
    assert (stray, RDF.type, OWL.Class) not in inserts
    attempt = state.attempt_log[-1]
    assert attempt.rolled_back_fixes[0].reason == "new_mandatory"
