"""The prompt must not present a retrieved subset as the whole ontology.

Under vector retrieval the ontology chapter is a stitched induced subgraph;
a prompt that reads as "the domain ontology provided below" invites two
errors: re-minting terms the catalog defines but retrieval did not surface,
and deleting statements whose justification lives in the unretrieved
remainder. The intro is therefore keyed on HOW the snapshot was assembled —
not on how many IRIs are writable, which used to hand a vector-mode unit
whose retrieval hit one ontology the full-ontology phrasing verbatim.
"""

import importlib

import pytest
from rdflib import RDFS, Literal, URIRef

from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import OntologyAssemblyMode
from ontocast.onto.ontology_access import ontology_access_for_unit_ontology
from ontocast.onto.ontology_snapshot import OntologySnapshot
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.unit_states import UnitOntologyState

render_ontology_module = importlib.import_module("ontocast.agent.render_ontology")

pytestmark = pytest.mark.unit

ONTO = "https://example.com/onto"


def _state(
    assembly_mode: OntologyAssemblyMode, writable_iris: list[str]
) -> UnitOntologyState:
    graph = RDFGraph()
    graph.add((URIRef(f"{ONTO}#Sample"), RDFS.label, Literal("Sample")))
    state = UnitOntologyState(
        content_unit=ContentUnit(
            text="text",
            index=0,
            doc_iri=URIRef("https://example.com/doc/d1"),
            graph=RDFGraph(),
        ),
        ontology_snapshot=OntologySnapshot(
            graph=graph, source_iris=writable_iris, assembly_mode=assembly_mode
        ),
    )
    state.writable_iris = writable_iris
    return state


@pytest.mark.parametrize("writable", [[ONTO], [ONTO, f"{ONTO}-2"]])
def test_vector_assembled_context_declares_itself_partial(
    writable: list[str],
) -> None:
    state = _state(OntologyAssemblyMode.SELECTED_VECTOR_SEARCH_ENSEMBLE, writable)
    intro = render_ontology_module._build_update_intro(
        state, ontology_access_for_unit_ontology(state)
    )
    assert "PARTIAL CONTEXT" in intro
    assert "RETRIEVED SUBSET" in intro


def test_full_copy_context_keeps_the_whole_ontology_phrasing() -> None:
    state = _state(OntologyAssemblyMode.SELECTED_SINGLE_ONTOLOGY_LLM, [ONTO])
    intro = render_ontology_module._build_update_intro(
        state, ontology_access_for_unit_ontology(state)
    )
    assert "PARTIAL CONTEXT" not in intro
    assert "Complement the domain ontology" in intro
