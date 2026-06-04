"""Text chunking agent for OntoCast.

Prepares content units via segment → tag → filter → size (see ``tool.chunk.prepare``).
"""

import logging

from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import Status
from ontocast.onto.state import AgentState
from ontocast.tool.chunk.prepare import PrepareOptions, prepare_content_units
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


async def chunk_text(state: AgentState, tools: ToolBox) -> AgentState:
    """Split document into manageable, optionally section-tagged content units."""
    logger.info("Chunking the text")
    if state.docling_doc is None:
        state.status = Status.FAILED
        return state

    state.content_units = []
    options = PrepareOptions(
        section_schema_id=state.section_schema_id,
        document_type_hint=state.document_type_hint,
        target_sections=state.target_sections,
        summarize_sections=state.summarize_sections,
    )
    prepared = await prepare_content_units(
        state.docling_doc,
        tools.chunker,
        tools.chunker.config,
        options,
        tools,
    )

    if state.max_chunks is not None:
        prepared = prepared[: state.max_chunks]

    logger.info(
        "Created %s chunks for processing: %s",
        len(prepared),
        [len(chunk.text) for chunk in prepared],
    )

    for i, chunk in enumerate(prepared):
        state.content_units.append(
            ContentUnit(
                text=chunk.text,
                index=i,
                doc_iri=state.doc_iri,
                headings=chunk.headings,
                doc_item_refs=list(chunk.doc_item_refs),
                section_label=chunk.section_label,
            )
        )

    logger.info(
        "Created %s content units: %s",
        len(state.content_units),
        [len(c.text) for c in state.content_units],
    )
    state.status = Status.SUCCESS
    return state
