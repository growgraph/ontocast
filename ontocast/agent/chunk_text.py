"""Text chunking agent for OntoCast.

Prepares content units via segment → tag → select → size → trim → route (see
``tool.chunk.prepare``) and turns the kept units into ``ContentUnit``s.
"""

import logging
from collections import Counter

from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import RouteReason, Status, UnitRoute
from ontocast.onto.state import AgentState
from ontocast.tool.chunk.prepare import (
    PreparedChunk,
    PrepareOptions,
    prepare_content_units,
)
from ontocast.tool.llm import use_budget_tracker
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)

_CITATION_REASONS = frozenset({RouteReason.CITATION_LIST, RouteReason.CITATION_LABEL})


def route_histogram(chunks: list[PreparedChunk]) -> dict[str, dict[str, int]]:
    """``'<route>/<reason>' -> {'units', 'chars'}`` over prepared chunks."""
    histogram: dict[str, dict[str, int]] = {}
    for chunk in chunks:
        key = f"{chunk.route.value}/{chunk.route_reason.value}"
        entry = histogram.setdefault(key, {"units": 0, "chars": 0})
        entry["units"] += 1
        entry["chars"] += len(chunk.text)
    return histogram


async def chunk_text(state: AgentState, tools: ToolBox) -> AgentState:
    """Split document into manageable, section-tagged and routed content units."""
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
    # Bound here because no other tracker is bound this early in the run: LLM
    # section classification and its worker waits would be charged nowhere.
    with use_budget_tracker(state.budget_tracker):
        document = await prepare_content_units(
            state.docling_doc,
            tools.chunker,
            tools.chunker.config,
            options,
            tools,
        )

    kept = document.units
    if state.max_chunks is not None:
        kept = kept[: state.max_chunks]

    for index, chunk in enumerate(kept):
        state.content_units.append(
            ContentUnit(
                text=chunk.text,
                index=index,
                doc_iri=state.doc_iri,
                headings=chunk.headings,
                doc_item_refs=list(chunk.doc_item_refs),
                section_label=chunk.section_label,
                section_label_source=chunk.section_label_source,
                section_label_strength=chunk.section_label_strength,
                section_label_evidence=list(chunk.section_label_evidence),
                route_reason=chunk.route_reason,
                is_citation_metadata=chunk.route is UnitRoute.CITATIONS,
                is_non_content=chunk.route is UnitRoute.NON_CONTENT,
            )
        )

    dropped_by = Counter(chunk.route_reason for chunk in document.dropped)
    # Recorded on the state so the run manifest's selection block can say how
    # many units each routing rule removed, not only how many survived.
    state.bibliography_units_skipped = sum(
        dropped_by[reason] for reason in _CITATION_REASONS
    )
    state.undersized_units_skipped = dropped_by[RouteReason.UNDERSIZED]
    state.non_content_units_skipped = dropped_by[RouteReason.NON_CONTENT]
    state.route_histogram = route_histogram(document.routed)
    state.outline_labels = document.outline_labels
    state.allowlist_excluded_chars = document.allowlist_excluded_chars
    state.allowlist_excluded_unlabeled_chars = (
        document.allowlist_excluded_unlabeled_chars
    )
    if document.dropped:
        logger.info(
            "Dropped %d unit(s) before extraction: %s",
            len(document.dropped),
            ", ".join(
                f"{reason.value}={count}"
                for reason, count in sorted(dropped_by.items())
            ),
        )

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
