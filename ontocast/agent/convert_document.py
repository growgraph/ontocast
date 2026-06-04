"""Document conversion agent for OntoCast.

This module provides functionality for converting various document formats into
structured data that can be processed by the OntoCast system.
"""

import json
import logging
import pathlib

from ontocast.onto.docling_helpers import plain_text_to_docling_doc
from ontocast.onto.enum import OntologyContextMode, Status
from ontocast.onto.state import AgentState
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def _get_json_text_field(result: dict[str, object]) -> str | None:
    """Extract text from JSON payload using a tiny top-level heuristic."""
    text_value = result.get("text")
    if isinstance(text_value, str):
        return text_value

    largest_text: str | None = None
    for value in result.values():
        if isinstance(value, str):
            if largest_text is None or len(value) > len(largest_text):
                largest_text = value

    return largest_text


def _extract_json_payload_text(
    payload: object, state: AgentState, source_name: str
) -> str | None:
    """Apply JSON extraction logic and return resolved text."""
    if not isinstance(payload, dict):
        logger.error(
            "Expected JSON object in %s, got %s", source_name, type(payload).__name__
        )
        return None

    json_payload: dict[str, object] = {
        str(key): value for key, value in payload.items()
    }

    ontology_user_instruction = json_payload.get("ontology_user_instruction", "")
    ontology_selection_user_instruction = json_payload.get(
        "ontology_selection_user_instruction", ""
    )
    facts_user_instruction = json_payload.get("facts_user_instruction", "")
    fixed_oid_raw = json_payload.get("ontology_context_fixed_ontology_id", "")

    if isinstance(ontology_user_instruction, str) and ontology_user_instruction:
        state.ontology_user_instruction = ontology_user_instruction
        logger.debug(f"Set ontology user instruction: {ontology_user_instruction}")
    if (
        isinstance(ontology_selection_user_instruction, str)
        and ontology_selection_user_instruction
    ):
        state.ontology_selection_user_instruction = ontology_selection_user_instruction
        logger.debug(
            "Set ontology selection user instruction: %s",
            ontology_selection_user_instruction,
        )
    if isinstance(facts_user_instruction, str) and facts_user_instruction:
        state.facts_user_instruction = facts_user_instruction
        logger.debug(f"Set facts user instruction: {facts_user_instruction}")
    if isinstance(fixed_oid_raw, str) and fixed_oid_raw.strip():
        state.ontology_context_fixed_ontology_id = fixed_oid_raw.strip()
        logger.debug(
            "Set ontology_context_fixed_ontology_id: %s", fixed_oid_raw.strip()
        )

    source_url = json_payload.get("url")
    if isinstance(source_url, str) and source_url:
        state.source_url = source_url
        logger.debug(f"Extracted source URL from JSON: {source_url}")

    json_text = _get_json_text_field(json_payload)
    if json_text is None:
        logger.error(
            "No string field found in JSON payload to use as text (%s)", source_name
        )
        return None
    return json_text


def _fail_when_fixed_catalog_ontology_missing(state: AgentState) -> AgentState | None:
    """If fixed catalog mode is active, require ontology_context_fixed_ontology_id."""
    if state.ontology_context_mode != OntologyContextMode.FIXED_SINGLE_ONTOLOGY:
        return None
    if state.ontology_context_fixed_ontology_id.strip():
        return None
    logger.error(
        "ontology_context_fixed_ontology_id required when ontology_context_mode is fixed_single_ontology"
    )
    state.status = Status.FAILED
    state.failure_reason = (
        "ontology_context_fixed_ontology_id is required when "
        "ontology_context_mode is fixed_single_ontology"
    )
    return state


def convert_document(state: AgentState, tools: ToolBox) -> AgentState:
    """Convert a single raw input payload on the state into a DoclingDocument.

    Strict 1-in / 1-out agent node: ``state.raw_input`` must contain exactly
    one ``{filename: bytes}`` entry. Batch fan-out (e.g. JSONL) is the caller's
    responsibility and must happen before this node is invoked.

    Args:
        state: The current agent state with a single raw input payload.
        tools: The toolbox instance providing utility functions.

    Returns:
        AgentState: Updated state with ``docling_doc`` populated, or with
        ``status == Status.FAILED`` if conversion could not be performed.
    """
    logger.debug("Converting document")

    state.status = Status.SUCCESS
    raw_input = state.raw_input

    if len(raw_input) != 1:
        logger.error(
            "convert_document expects exactly one raw input entry, received %d",
            len(raw_input),
        )
        state.status = Status.FAILED
        return state

    filename, file_content = next(iter(raw_input.items()))
    file_extension = pathlib.Path(filename).suffix.lower()
    logger.debug("Converting %s with extension %s", filename, file_extension)

    if file_extension in tools.converter.supported_extensions:
        doc = tools.converter(file_content)
        state.set_docling_doc(doc)
        blocked = _fail_when_fixed_catalog_ontology_missing(state)
        return blocked if blocked is not None else state

    if file_extension == ".json":
        result_json = json.loads(file_content.decode("utf-8"))
        json_text = _extract_json_payload_text(result_json, state, filename)
        if json_text is None:
            state.status = Status.FAILED
            return state
        state.set_docling_doc(plain_text_to_docling_doc(json_text, filename))
        blocked = _fail_when_fixed_catalog_ontology_missing(state)
        return blocked if blocked is not None else state

    if file_extension == ".txt":
        text = json.loads(file_content.decode("utf-8"))
        state.set_docling_doc(plain_text_to_docling_doc(text, filename))
        blocked = _fail_when_fixed_catalog_ontology_missing(state)
        return blocked if blocked is not None else state

    logger.error("Unsupported file extension %s for %s", file_extension, filename)
    state.status = Status.FAILED
    return state
