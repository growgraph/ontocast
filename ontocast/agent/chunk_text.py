"""Text chunking agent for OntoCast.

Prepares content units via segment → tag → filter → size (see ``tool.chunk.prepare``).
"""

import logging

from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import Status
from ontocast.onto.state import AgentState
from ontocast.tool.chunk.bibliography import is_bibliography_unit
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
        exclude_sections=state.exclude_sections,
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

    bibliography_mode = tools.chunker.config.bibliography_mode
    skipped_bibliography = 0
    index = 0
    for chunk in prepared:
        is_bibliography = bibliography_mode != "domain_facts" and is_bibliography_unit(
            chunk.text, chunk.section_label
        )
        if is_bibliography:
            # A false positive silences a content section: the unit is routed to
            # citation-metadata extraction and no domain facts are minted from
            # it. That must never happen without a trace, so every routing
            # decision is logged, not only the 'skip' path.
            logger.info(
                "Chunk %d (%d chars, section_label=%r) routed as bibliography "
                "(CHUNK_BIBLIOGRAPHY_MODE=%s): %s",
                index,
                len(chunk.text),
                chunk.section_label,
                bibliography_mode,
                "dropped" if bibliography_mode == "skip" else "citation metadata only",
            )
        if is_bibliography and bibliography_mode == "skip":
            skipped_bibliography += 1
            continue
        state.content_units.append(
            ContentUnit(
                text=chunk.text,
                index=index,
                doc_iri=state.doc_iri,
                headings=chunk.headings,
                doc_item_refs=list(chunk.doc_item_refs),
                section_label=chunk.section_label,
                is_citation_metadata=is_bibliography,
            )
        )
        index += 1
    if skipped_bibliography:
        logger.info(
            "Dropped %d bibliography chunk(s) (CHUNK_BIBLIOGRAPHY_MODE=skip)",
            skipped_bibliography,
        )

    logger.info(
        "Created %s content units: %s",
        len(state.content_units),
        [len(c.text) for c in state.content_units],
    )
    state.status = Status.SUCCESS
    return state
