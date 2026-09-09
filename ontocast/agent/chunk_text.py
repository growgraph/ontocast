"""Text chunking agent for OntoCast.

Prepares content units via segment → tag → filter → size (see ``tool.chunk.prepare``).
"""

import logging
from collections import Counter

from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import Status
from ontocast.onto.state import AgentState
from ontocast.tool.chunk.bibliography import is_bibliography_unit
from ontocast.tool.chunk.non_content import first_line, is_non_content_unit
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
    non_content_mode = tools.chunker.config.non_content_mode
    min_unit_chars = tools.chunker.config.min_unit_chars
    skipped_bibliography = 0
    skipped_undersized = 0
    skipped_non_content = 0
    index = 0
    for chunk in prepared:
        if min_unit_chars and len(chunk.text) < min_unit_chars:
            # Same discipline as the bibliography route above: dropping a unit
            # changes what can be extracted, so each decision is logged rather
            # than only counted.
            logger.info(
                "Chunk (%d chars, section_label=%r) dropped as undersized "
                "(CHUNK_MIN_UNIT_CHARS=%d)",
                len(chunk.text),
                chunk.section_label,
                min_unit_chars,
            )
            skipped_undersized += 1
            continue
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
        # A reference list already has its route; only prose-shaped units are
        # tested for front/back matter. Same trace discipline: a false positive
        # drops a section, so the decision is logged in both modes.
        is_non_content = not is_bibliography and is_non_content_unit(
            chunk.text, chunk.headings, chunk.section_label
        )
        if is_non_content:
            logger.info(
                "Chunk %d (%d chars, section_label=%r, first line=%r) routed as "
                "non-content (CHUNK_NON_CONTENT_MODE=%s): %s",
                index,
                len(chunk.text),
                chunk.section_label,
                first_line(chunk.text)[:60],
                non_content_mode,
                "dropped" if non_content_mode == "skip" else "kept, flagged",
            )
        if is_non_content and non_content_mode == "skip":
            skipped_non_content += 1
            continue
        state.content_units.append(
            ContentUnit(
                text=chunk.text,
                index=index,
                doc_iri=state.doc_iri,
                headings=chunk.headings,
                doc_item_refs=list(chunk.doc_item_refs),
                section_label=chunk.section_label,
                section_label_source=chunk.section_label_source,
                section_label_confidence=chunk.section_label_confidence,
                is_citation_metadata=is_bibliography,
                is_non_content=is_non_content,
            )
        )
        index += 1
    if skipped_bibliography:
        logger.info(
            "Dropped %d bibliography chunk(s) (CHUNK_BIBLIOGRAPHY_MODE=skip)",
            skipped_bibliography,
        )
    if skipped_undersized:
        logger.info(
            "Dropped %d undersized chunk(s) (CHUNK_MIN_UNIT_CHARS=%d)",
            skipped_undersized,
            min_unit_chars,
        )
    if skipped_non_content:
        logger.info(
            "Dropped %d non-content chunk(s) (CHUNK_NON_CONTENT_MODE=skip)",
            skipped_non_content,
        )
    # Recorded on the state so the run manifest's selection block can say how
    # many units each routing knob removed, not only how many survived.
    state.bibliography_units_skipped = skipped_bibliography
    state.undersized_units_skipped = skipped_undersized
    state.non_content_units_skipped = skipped_non_content

    logger.info(
        "Created %s content units: %s",
        len(state.content_units),
        [len(c.text) for c in state.content_units],
    )
    histogram = Counter(
        unit.section_label or "(unlabeled)" for unit in state.content_units
    )
    logger.info(
        "Section labels: %s",
        ", ".join(f"{label}={count}" for label, count in sorted(histogram.items())),
    )
    state.status = Status.SUCCESS
    return state
