import asyncio

import pytest
from rdflib import Literal, URIRef

from ontocast.agent import check_chunks_empty, chunk_text, select_ontology
from ontocast.onto.ontology import Ontology
from ontocast.onto.sparql_models import GraphUpdate, TripleOp
from ontocast.onto.state import AgentState


def test_agent_state_json():
    state = AgentState()
    state.current_ontology = Ontology(ontology_id="ex")

    # Add a custom triple
    custom_triple = (
        URIRef("http://example.com/subject"),
        URIRef("http://example.com/predicate"),
        Literal("object"),
    )
    state.current_ontology.graph.add(custom_triple)

    state_json = state.model_dump_json()

    loaded_state = AgentState.model_validate_json(state_json)

    # Check that the graph has at least our custom triple
    assert len(loaded_state.current_ontology.graph) >= 1
    assert custom_triple in loaded_state.current_ontology.graph


def test_chunks(apple_report: dict, tools, state_chunked_filename):
    state = AgentState()
    state.set_text(apple_report["text"])
    state = chunk_text(state, tools)
    assert len(state.chunks) == 12
    state.chunks = state.chunks[:2]
    state = check_chunks_empty(state)
    assert state.current_chunk is not None
    state.serialize(state_chunked_filename)


@pytest.mark.order(after="test_chunks")
def test_select_ontology_fsec(
    state_chunked: AgentState,
    tools,
    state_onto_selected_filename,
):
    state = state_chunked
    state = asyncio.run(select_ontology(state=state, tools=tools))
    assert state.current_ontology.ontology_id == "fsec"

    state.serialize(state_onto_selected_filename)


def test_select_ontology_null(
    random_report: dict,
    tools,
    state_onto_null_filename,
):
    state = AgentState()
    state.set_text(random_report["text"])
    state = chunk_text(state, tools)
    state = check_chunks_empty(state)
    state = asyncio.run(select_ontology(state=state, tools=tools))
    # TODO small investigation
    assert state.current_ontology.ontology_id is None

    state.serialize(state_onto_null_filename)


def test_render_updated_graph():
    """Test the generalized render_updated_graph function."""
    state = AgentState()

    # Create a simple graph with one triple
    original_graph = state.current_chunk.graph
    original_graph.add(
        (
            URIRef("http://example.com/subject"),
            URIRef("http://example.com/predicate"),
            Literal("original_value"),
        )
    )

    # Create a GraphUpdate that adds a new triple using Turtle format
    graph_update = GraphUpdate(
        triple_operations=[
            TripleOp(
                type="insert",
                graph="""@prefix ex: <http://example.com/> .

ex:subject ex:predicate "updated_value" .""",  # type: ignore[arg-type]
                prefixes={"ex": "http://example.com/"},
            ),
        ]
    )

    # Apply the update
    updated_graph, _ = state.render_updated_graph(
        original_graph, [graph_update], max_triples=None
    )

    # Check that the original graph is unchanged
    assert len(original_graph) == 1
    assert (
        URIRef("http://example.com/subject"),
        URIRef("http://example.com/predicate"),
        Literal("original_value"),
    ) in original_graph

    # Check that the updated graph has both triples
    assert len(updated_graph) == 2
    assert (
        URIRef("http://example.com/subject"),
        URIRef("http://example.com/predicate"),
        Literal("original_value"),
    ) in updated_graph
    assert (
        URIRef("http://example.com/subject"),
        URIRef("http://example.com/predicate"),
        Literal("updated_value"),
    ) in updated_graph


def test_render_uptodate_facts():
    """Test the render_uptodate_facts function."""
    state = AgentState()

    # Add a triple to the current chunk's graph
    state.current_chunk.graph.add(
        (
            URIRef("http://example.com/subject"),
            URIRef("http://example.com/predicate"),
            Literal("original_value"),
        )
    )

    # Create a facts update using Turtle format
    state.facts_updates = [
        GraphUpdate(
            triple_operations=[
                TripleOp(
                    type="insert",
                    graph="""@prefix ex: <http://example.com/> .

ex:subject ex:predicate "facts_value" .""",  # type: ignore[arg-type]
                    prefixes={"ex": "http://example.com/"},
                ),
            ]
        )
    ]

    # Get the updated facts graph
    updated_facts = state.render_uptodate_facts()

    # Check that the original chunk graph is unchanged
    assert len(state.current_chunk.graph) == 1

    # Check that the updated facts graph has both triples
    assert len(updated_facts) == 2
    assert (
        URIRef("http://example.com/subject"),
        URIRef("http://example.com/predicate"),
        Literal("original_value"),
    ) in updated_facts
    assert (
        URIRef("http://example.com/subject"),
        URIRef("http://example.com/predicate"),
        Literal("facts_value"),
    ) in updated_facts


def test_update_facts():
    """Test the update_facts function."""
    state = AgentState()

    # Add a triple to the current chunk's graph
    state.current_chunk.graph.add(
        (
            URIRef("http://example.com/subject"),
            URIRef("http://example.com/predicate"),
            Literal("original_value"),
        )
    )

    # Create a facts update using Turtle format
    state.facts_updates = [
        GraphUpdate(
            triple_operations=[
                TripleOp(
                    type="insert",
                    graph="""@prefix ex: <http://example.com/> .

ex:subject ex:predicate "facts_value" .""",  # type: ignore[arg-type]
                    prefixes={"ex": "http://example.com/"},
                ),
            ]
        )
    ]

    # Apply the update
    state.update_facts()

    # Check that the chunk graph is updated
    assert len(state.current_chunk.graph) == 2
    assert (
        URIRef("http://example.com/subject"),
        URIRef("http://example.com/predicate"),
        Literal("original_value"),
    ) in state.current_chunk.graph
    assert (
        URIRef("http://example.com/subject"),
        URIRef("http://example.com/predicate"),
        Literal("facts_value"),
    ) in state.current_chunk.graph

    # Check that facts_updates is cleared
    assert len(state.facts_updates) == 0
