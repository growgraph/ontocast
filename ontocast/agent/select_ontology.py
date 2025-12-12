"""Ontology selection agent for OntoCast.

This module provides functionality for selecting appropriate ontologies based on
the content of text chunks, ensuring that the chosen ontology best matches the
domain and requirements of the text.
"""

import logging

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

from ontocast.agent.common import call_llm_with_retry
from ontocast.onto.model import create_ontology_selector_report_model
from ontocast.onto.null import NULL_ONTOLOGY
from ontocast.onto.state import AgentState
from ontocast.prompt.select_ontology import template_prompt
from ontocast.tool import OntologyManager
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def _create_document_excerpt(state: AgentState, max_length: int = 3000) -> str:
    """Create a representative excerpt from the document for ontology selection.

    This function samples text from multiple chunks to provide a better
    representation of the document content than just the first chunk.

    Args:
        state: The current agent state.
        max_length: Maximum total length of the excerpt.

    Returns:
        str: A representative excerpt from the document.
    """
    excerpt_parts = []
    total_length = 0
    chunk_length = max_length // 3  # Aim for ~3 chunks, ~1000 chars each

    # Strategy: Sample from first, middle, and last chunks if available
    if state.chunks:
        num_chunks = len(state.chunks)
        indices_to_sample = []

        if num_chunks == 1:
            indices_to_sample = [0]
        elif num_chunks == 2:
            indices_to_sample = [0, 1]
        else:
            # Sample first, middle, and last
            indices_to_sample = [0, num_chunks // 2, num_chunks - 1]

        for idx in indices_to_sample:
            if idx < num_chunks and total_length < max_length:
                chunk_text = state.chunks[idx].text
                # Take a portion of this chunk
                remaining = max_length - total_length
                sample_length = min(chunk_length, remaining, len(chunk_text))

                if sample_length > 0:
                    if sample_length < len(chunk_text):
                        excerpt_parts.append(chunk_text[:sample_length] + " ...")
                    else:
                        excerpt_parts.append(chunk_text)
                    total_length += sample_length

        if excerpt_parts:
            return "\n\n[...]\n\n".join(excerpt_parts)

    # Fallback: Use input_text if available
    if state.input_text:
        if len(state.input_text) <= max_length:
            return state.input_text
        return state.input_text[:max_length] + " ..."

    # Last resort: Use current_chunk
    if state.current_chunk and state.current_chunk.text:
        chunk_text = state.current_chunk.text
        if len(chunk_text) <= max_length:
            return chunk_text
        return chunk_text[:max_length] + " ..."

    return ""


async def select_ontology(state: AgentState, tools: ToolBox) -> AgentState:
    """Select an appropriate ontology for the document.

    This function analyzes the document and selects the most appropriate
    ontology based on its content and requirements using a numbered list selection.
    If an ontology is already selected, it skips selection to ensure one ontology
    per document.

    Args:
        state: The current agent state containing the document to process.
        tools: The toolbox instance providing utility functions.

    Returns:
        AgentState: Updated state with selected ontology.
    """
    # Skip if ontology already selected (for subsequent chunks in the loop)
    if not state.current_ontology.is_null():
        logger.debug(
            f"Ontology already selected: {state.current_ontology.ontology_id}, "
            "skipping selection to maintain one ontology per document"
        )
        return state

    progress_info = state.get_chunk_progress_string()
    logger.info(f"Selecting ontology for document ({progress_info})")
    llm_tool = tools.llm
    om_tool: OntologyManager = tools.ontology_manager

    if om_tool.has_ontologies:
        ontologies = om_tool.ontologies
        num_ontologies = len(ontologies)

        # Create numbered list of ontologies
        ontologies_list_lines = []
        for i, ontology in enumerate(ontologies, start=1):
            ontologies_list_lines.append(f"{i}. {ontology.describe()}")
        ontologies_list_lines.append(
            f"{len(ontologies) + 1}. None of the ontologies matches the text"
        )
        ontologies_list = "\n\n".join(ontologies_list_lines)

        logger.info(f"Presenting {num_ontologies} ontologies for selection")

        # Create a better document excerpt using multiple chunks
        excerpt = _create_document_excerpt(state, max_length=3000)

        # Create dynamic model with correct constraint
        ontology_selector_report_model = create_ontology_selector_report_model(
            num_ontologies
        )
        parser = PydanticOutputParser(pydantic_object=ontology_selector_report_model)

        prompt = PromptTemplate(
            template=template_prompt,
            input_variables=[
                "excerpt",
                "ontologies_list",
                "num_ontologies",
                "format_instructions",
            ],
        )

        selector = await call_llm_with_retry(
            llm_tool=llm_tool,
            prompt=prompt,
            parser=parser,
            prompt_kwargs={
                "excerpt": excerpt,
                "ontologies_list": ontologies_list,
                "num_ontologies": num_ontologies,
                "format_instructions": parser.get_format_instructions(),
            },
        )

        # Map answer_index to ontology
        # answer_index: 1 to num_ontologies -> select ontology at (answer_index - 1)
        # answer_index: num_ontologies + 1 -> select None
        if selector.answer_index == num_ontologies + 1:
            # None selected
            logger.debug("LLM selected: None (no suitable ontology)")
            state.current_ontology = NULL_ONTOLOGY
        elif 1 <= selector.answer_index <= num_ontologies:
            # Select ontology at index (answer_index - 1) since list is 0-based
            selected_ontology = ontologies[selector.answer_index - 1]
            logger.debug(
                f"LLM selected ontology at index {selector.answer_index}: "
                f"{selected_ontology.ontology_id} ({selected_ontology.iri})"
            )
            state.current_ontology = selected_ontology
        else:
            # This should not happen due to Pydantic validation, but handle gracefully
            logger.warning(
                f"Invalid answer_index {selector.answer_index} "
                f"(expected 1-{num_ontologies + 1}), defaulting to NULL_ONTOLOGY"
            )
            state.current_ontology = NULL_ONTOLOGY
    else:
        state.current_ontology = NULL_ONTOLOGY

    # Set the initial version if not already set (tracks original version when ontology was selected)
    if state.current_ontology.initial_version is None:
        state.current_ontology.initial_version = state.current_ontology.version
        logger.debug(
            f"Set initial version for ontology {state.current_ontology.ontology_id}: {state.current_ontology.initial_version}"
        )

    logger.debug(f"Current ontology set to: {state.current_ontology.ontology_id}")
    return state
