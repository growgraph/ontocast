"""Ontology sublimation agent for OntoCast.

This module provides functionality for refining and enhancing ontologies through
a process of sublimation, which involves improving the structure, consistency,
and expressiveness of the ontological knowledge.
"""

import logging
from typing import Iterable, cast

from rdflib.term import Node

from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.enum import FailureStage
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def _sublimate_ontology(state: AgentState) -> tuple[RDFGraph, RDFGraph]:
    graph_onto_addendum = RDFGraph()
    graph_facts_pure = RDFGraph()

    # Copy all prefixes from the original graph to both new graphs
    for prefix, namespace in state.current_content_unit.graph.namespaces():
        graph_onto_addendum.bind(prefix, namespace)
        graph_facts_pure.bind(prefix, namespace)

    query_ontology = f"""
    PREFIX cd: <{DEFAULT_IRI}>

    SELECT ?s ?p ?o
    WHERE {{
    ?s ?p ?o .
    FILTER (
        !(
            STRSTARTS(STR(?s), STR(cd:)) ||
            STRSTARTS(STR(?p), STR(cd:)) ||
            (isIRI(?o) && STRSTARTS(STR(?o), STR(cd:)))
        )
    )
    }}
    """
    results = cast(
        Iterable[tuple[Node, Node, Node]],
        state.current_content_unit.graph.query(query_ontology),
    )

    # Add filtered triples to the new graph
    for s, p, o in results:
        graph_onto_addendum.add((s, p, o))

    query_facts = f"""
        PREFIX cd: <{DEFAULT_IRI}>

        SELECT ?s ?p ?o
        WHERE {{
        ?s ?p ?o .
        FILTER (
            STRSTARTS(STR(?s), STR(cd:)) ||
            STRSTARTS(STR(?p), STR(cd:)) ||
            (isIRI(?o) && STRSTARTS(STR(?o), STR(cd:)))
        )
        }}
    """

    results = cast(
        Iterable[tuple[Node, Node, Node]],
        state.current_content_unit.graph.query(query_facts),
    )

    # Add filtered triples to the new graph
    for s, p, o in results:
        graph_facts_pure.add((s, p, o))

    logger.info(
        f"Found triples: facts {len(graph_facts_pure)}; ontology {len(graph_onto_addendum)}"
    )
    return graph_onto_addendum, graph_facts_pure


def sublimate_ontology(state: AgentState, tools: ToolBox):
    logger.debug("Starting ontology sublimation")

    if state.current_ontology is None:
        return state
    try:
        state.update_facts()
        graph_onto_addendum, graph_facts = _sublimate_ontology(state=state)

        # Ensure ontology is not null and ontology_id is set before updating
        if len(graph_onto_addendum) > 0:
            logger.info("ontology seeped into facts:")
            logger.info(f"graph: {graph_onto_addendum.serialize()}")
            if state.current_ontology.is_null():
                logger.warning(
                    "Cannot update ontology: null ontology cannot be updated"
                )
            elif state.current_ontology.ontology_id:
                # Check if adding triples would exceed max_triples limit
                max_triples = state.ontology_max_triples
                if max_triples is not None:
                    current_size = len(state.current_ontology.graph)
                    addendum_size = len(graph_onto_addendum)
                    if current_size + addendum_size > max_triples:
                        logger.warning(
                            f"Ontology sublimation skipped: would exceed limit "
                            f"({current_size + addendum_size} > {max_triples} triples). "
                            f"Current size: {current_size} triples."
                        )
                    else:
                        # Only update state.current_ontology, not OntologyManager
                        # OntologyManager will be updated in serialize() during final serialization
                        state.current_ontology.graph += graph_onto_addendum
                        logger.debug(
                            f"Updated state.current_ontology with {len(graph_onto_addendum)} triples from sublimation"
                        )
                else:
                    # No limit set, proceed with update
                    state.current_ontology.graph += graph_onto_addendum
                    logger.debug(
                        f"Updated state.current_ontology with {len(graph_onto_addendum)} triples from sublimation"
                    )
            else:
                logger.warning("Cannot update ontology: ontology_id is None")

        # Ensure graph_facts is an RDFGraph instance
        if not isinstance(graph_facts, RDFGraph):
            logger.warning("received an rdflib.Graph rather than RDFGraph")
            new_graph = RDFGraph()
            graph_facts_rdflib = cast(Iterable[tuple[Node, Node, Node]], graph_facts)
            for triple in graph_facts_rdflib:
                new_graph.add(triple)
            graph_facts_namespaces = cast(
                Iterable[tuple[str, str]], graph_facts.namespaces()
            )
            for prefix, namespace in graph_facts_namespaces:
                new_graph.bind(prefix, namespace)
            graph_facts = new_graph

        state.current_content_unit.graph = graph_facts

        state.clear_failure()
    except Exception as e:
        logger.error(f"Error in sublimate_ontology: {str(e)}")
        state.set_failure(
            FailureStage.SUBLIMATE_ONTOLOGY,
            str(e),
        )

    return state
