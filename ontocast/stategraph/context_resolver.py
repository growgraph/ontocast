import asyncio
import logging
import time
from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from ontocast.agent.select_ontology_catalog import select_catalog_ontology_for_excerpt
from ontocast.onto.content_unit import SourceUnit
from ontocast.onto.enum import (
    OntologyAssemblyMode,
    OntologyContextMode,
    RetrievalMetric,
)
from ontocast.onto.null import NULL_ONTOLOGY
from ontocast.onto.ontology_snapshot import OntologySnapshot
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.retrieval_capabilities import (
    EmptyOntologyContextError,
    require_vector_retrieval,
)
from ontocast.stategraph.unit_context import UnitLoopContext
from ontocast.tool.chunk.proposition import split_proposition_windows
from ontocast.tool.llm import use_budget_tracker
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


class UnitOntologyContext(BaseModel):
    """Assembled prompt context: snapshot view + writable catalog IRIs for apply."""

    snapshot: OntologySnapshot
    writable_iris: list[str] = Field(default_factory=list)
    confidence: float = 0.0

    @property
    def assembly_mode(self) -> OntologyAssemblyMode:
        return self.snapshot.assembly_mode

    @property
    def patch_sources(self) -> list[str]:
        return list(self.snapshot.source_iris)

    @property
    def primary_writable_iri(self) -> str:
        """Primary catalog IRI for metrics (first writable, else null)."""
        if self.writable_iris:
            return self.writable_iris[0]
        return NULL_ONTOLOGY.iri


def _unit_queries(unit: SourceUnit, tools: ToolBox) -> list[str]:
    vcfg = tools.config.tool_config.vector_store
    text = unit.text.strip()
    if not text:
        return []
    if not vcfg.proposition_retrieval_enabled:
        return [text]
    return split_proposition_windows(
        text,
        max_sentences=vcfg.proposition_window_sentences,
        max_windows=vcfg.proposition_max_windows,
    )


def build_merged_document_ontology_context(
    context: UnitLoopContext,
) -> UnitOntologyContext | None:
    """Build merged ontology context from reduced document artifacts.

    The result depends only on document-level state, so it should be computed
    once per document. ``"ctx/merge_document_ontology.calls"`` on the budget
    tracker exists to make a regression to per-unit calls visible.
    """
    started = time.perf_counter()
    context.budget_tracker.incr("ctx/merge_document_ontology.calls")
    artifacts = [
        ontology
        for ontology in context.reduced_artifacts()
        if not ontology.is_null() and len(ontology.graph) > 0
    ]
    if not artifacts:
        context.budget_tracker.add_duration(
            "ctx/merge_document_ontology", time.perf_counter() - started
        )
        return None

    sorted_artifacts = sorted(artifacts, key=lambda ontology: ontology.iri or "")
    merged_graph = RDFGraph()
    patch_sources: list[str] = []
    for ontology in sorted_artifacts:
        merged_graph += ontology.graph
        if ontology.iri:
            patch_sources.append(ontology.iri)
    merged_graph.sanitize_prefixes_namespaces()

    snapshot = OntologySnapshot.from_graph(
        merged_graph,
        source_iris=patch_sources,
        assembly_mode=OntologyAssemblyMode.DOCUMENT_MERGED_REDUCED,
        title="Merged document ontology context",
        description=(
            "Deterministic merge of reduced ontology artifacts used for facts context."
        ),
        strip_headers=True,
    )
    context.budget_tracker.add_duration(
        "ctx/merge_document_ontology", time.perf_counter() - started
    )
    context.retrieval_metrics[RetrievalMetric.ONTOLOGY_SNAPSHOT_TRIPLES] = len(
        snapshot.graph
    )
    return UnitOntologyContext(
        snapshot=snapshot,
        writable_iris=list(patch_sources),
        confidence=1.0,
    )


async def _resolve_selected_single_ontology_context(
    context: UnitLoopContext,
    tools: ToolBox,
    unit: SourceUnit,
) -> UnitOntologyContext:
    """One catalog ontology chosen by the LLM from the unit text."""
    # Scoped so the selection call is charged to the calling unit's budget
    # rather than to whichever tracker was bound last.
    with use_budget_tracker(context.budget_tracker):
        selected = await select_catalog_ontology_for_excerpt(
            tools.ontology_manager,
            tools.llm,
            unit.text,
            context.ontology_selection_user_instruction,
        )
    mode = OntologyAssemblyMode.SELECTED_SINGLE_ONTOLOGY_LLM
    if selected.is_null():
        return UnitOntologyContext(
            snapshot=OntologySnapshot.empty(
                assembly_mode=mode,
                title="Null ontology",
                description="No catalog ontology selected.",
            ),
            writable_iris=[],
            confidence=0.0,
        )
    return UnitOntologyContext(
        snapshot=OntologySnapshot.from_ontology(selected, assembly_mode=mode),
        writable_iris=[selected.iri] if selected.iri else [],
        confidence=0.5,
    )


async def _resolve_fixed_single_ontology_context(
    context: UnitLoopContext,
    tools: ToolBox,
    unit: SourceUnit,
) -> UnitOntologyContext:
    """Catalog ontology fixed by ontology_id (fresh terminal revision)."""
    _ = unit
    cleaned = context.ontology_context_fixed_ontology_id.strip()
    mode = OntologyAssemblyMode.FIXED_SINGLE_ONTOLOGY
    if not cleaned:
        return UnitOntologyContext(
            snapshot=OntologySnapshot.empty(
                assembly_mode=mode,
                title="Null ontology",
                description="No fixed ontology id provided.",
            ),
            writable_iris=[],
            confidence=0.0,
        )
    mgr = tools.ontology_manager
    selected = mgr.get_freshest_terminal_ontology(ontology_id=cleaned)
    if selected is None:
        logger.warning(
            "No catalog ontology match for ontology_context_fixed_ontology_id=%r; "
            "using empty snapshot",
            cleaned,
        )
        return UnitOntologyContext(
            snapshot=OntologySnapshot.empty(
                assembly_mode=mode,
                title="Null ontology",
                description=f"No catalog match for {cleaned!r}.",
            ),
            writable_iris=[],
            confidence=0.0,
        )
    return UnitOntologyContext(
        snapshot=OntologySnapshot.from_ontology(selected, assembly_mode=mode),
        writable_iris=[selected.iri] if selected.iri else [],
        confidence=1.0,
    )


async def _diagnose_empty_snapshot(
    tools: ToolBox, metrics: dict[str, Any] | None
) -> str:
    """Name the subsystem that produced an empty ontology snapshot.

    Catalog-first, because retrieval can select exactly the right atoms and
    still yield nothing when the triple store lists no graphs to expand them
    against -- the index and the store disagreeing is a deployment fault, not a
    tuning one, and reporting it as a threshold problem sends an operator to
    lower thresholds that were never involved.

    Two things this has to get right that reading the metrics alone cannot:

    - **A missing key is not a zero.** When retrieval short-circuits on zero
      atoms it never reaches the catalog, so ``catalog_context_triples`` is
      absent rather than ``0``. Testing it with ``== 0`` therefore skipped both
      catalog branches on exactly the run where the catalog was the cause, and
      reported the empty index instead -- true, but the symptom rather than the
      fault. The catalog is asked directly when the metrics cannot answer.
    - **``atoms_after_dedupe`` is counted after the score gate**, so a
      threshold rejection records zero of them. "Scored below the retrieval
      thresholds" was thus unreachable for the case it names; ``candidate_hits``
      and ``threshold_rejected``, taken before the gate, are what separate
      "search found nothing" from "a threshold ate everything".

    Args:
        tools: Toolbox, for inspecting the catalog and index as a last resort.
        metrics: ``last_retrieval_metrics`` from the patch retriever.

    Returns:
        str: A one-line cause, stored under ``empty_snapshot_reason``.
    """
    metrics = metrics or {}
    catalog_triples = metrics.get("catalog_context_triples")
    if catalog_triples is None:
        # Retrieval never consulted the catalog, so ask it.
        if not tools.ontology_manager.has_ontologies:
            return "the ontology catalog is empty (no ontologies stored)"
    else:
        graph_reads = (metrics.get("catalog_graph_cache_hits") or 0) + (
            metrics.get("catalog_graph_cache_misses") or 0
        )
        if catalog_triples == 0 and graph_reads == 0:
            return (
                "the ontology catalog resolved to zero graphs -- the vector "
                "index and the triple store disagree about which ontologies "
                "exist"
            )
        if catalog_triples == 0:
            return "the ontology catalog is empty (no ontologies stored)"
        if metrics.get("atoms_final"):
            return (
                "the catalog is populated but the induced subgraph over the "
                "selected atoms came back empty"
            )

    indexed_iris: set[str] = set()
    if tools.vector_store is not None:
        try:
            indexed_iris = await asyncio.to_thread(
                tools.vector_store.list_indexed_ontology_iris
            )
        except Exception as exc:
            logger.warning("Could not inspect the vector index: %s", exc)
    if not indexed_iris:
        return (
            "the vector index is empty or unreadable, though the catalog holds "
            "ontologies -- they were never indexed, or the index was wiped "
            "without a reindex"
        )
    if metrics.get("threshold_rejected"):
        return "all candidate atoms scored below the retrieval thresholds"
    if metrics.get("candidate_hits"):
        return "candidate atoms were filtered out after retrieval"
    return "no candidate atoms matched the unit's queries"


async def _resolve_ensemble_context(
    context: UnitLoopContext,
    tools: ToolBox,
    unit: SourceUnit,
) -> UnitOntologyContext:
    """Stitched induced subgraphs from vector retrieval."""
    mode = OntologyAssemblyMode.SELECTED_VECTOR_SEARCH_ENSEMBLE
    queries = _unit_queries(unit, tools)
    if not queries:
        return UnitOntologyContext(
            snapshot=OntologySnapshot.empty(
                assembly_mode=mode,
                title="Empty unit (no text queries for retrieval)",
                description="No proposition queries; ensemble graph is empty.",
            ),
            writable_iris=[],
            confidence=0.0,
        )
    retriever = tools.patch_retriever
    assert retriever is not None
    vcfg = tools.config.tool_config.vector_store
    patch_graph, source_iris = await retriever.aretrieve_ensemble(
        queries=queries,
        top_k=vcfg.top_k,
        expand_sparql=True,
        subgraph_depth=vcfg.induced_subgraph_depth,
        max_total_triples=vcfg.induced_subgraph_max_total_triples,
        estimated_triples_per_query=vcfg.induced_subgraph_estimated_triples_per_query,
        trigger_text=unit.text.strip(),
    )
    metrics = retriever.last_retrieval_metrics
    writable = list(source_iris)
    if metrics:
        context.retrieval_metrics[RetrievalMetric.PATCH_RETRIEVAL] = metrics
        expanded = metrics.get("expanded_ontology_iris") or []
        if isinstance(expanded, list):
            for iri in expanded:
                if isinstance(iri, str) and iri and iri not in writable:
                    writable.append(iri)
        logger.info(
            "Patch retrieval: queries=%s atoms_final=%s "
            "source_iris=%s expanded=%s triples=%s",
            metrics.get("query_count"),
            metrics.get("atoms_final"),
            metrics.get("source_ontology_iris"),
            metrics.get("expanded_ontology_iris"),
            metrics.get("snapshot_triple_count"),
        )
    if not len(patch_graph):
        # An empty snapshot reaching the renderer means it will extract with no
        # vocabulary at all, so name the subsystem at fault before deciding
        # whether to continue.
        reason = await _diagnose_empty_snapshot(tools, metrics)
        context.retrieval_metrics[RetrievalMetric.EMPTY_SNAPSHOT_REASON] = reason
        logger.warning("Ontology context for this unit is empty (%s)", reason)

    preferred = tools.ontology_manager.preferred_namespace_prefixes or None
    patch_graph.sanitize_prefixes_namespaces(preferred_namespace_prefixes=preferred)

    snapshot = OntologySnapshot.from_graph(
        patch_graph,
        source_iris=source_iris,
        assembly_mode=mode,
        title="Vector search ensemble context",
        description="Stitched induced subgraphs from hybrid retrieval.",
        strip_headers=True,
    )

    return UnitOntologyContext(
        snapshot=snapshot,
        writable_iris=writable,
        confidence=1.0 if source_iris else 0.5,
    )


async def resolve_unit_ontology_context(
    context: UnitLoopContext,
    tools: ToolBox,
    unit: SourceUnit,
    *,
    can_create_vocabulary: bool = False,
) -> UnitOntologyContext:
    """Assemble the ontology context one content unit is rendered against.

    Args:
        context: Document-level loop inputs.
        tools: Toolbox holding the catalog and retrieval.
        unit: The content unit being rendered.
        can_create_vocabulary: Whether the caller can act on an empty context
            by inventing vocabulary. True for the ontology loop, which answers
            an empty seed with ``render_ontology_fresh``; false for the facts
            loop, which can only fall back on generic terms.

    Returns:
        The resolved context, possibly empty.

    Raises:
        EmptyOntologyContextError: The context is empty, the caller cannot
            create vocabulary, and this deployment requires a context.
    """
    mode = context.ontology_context_mode
    context.retrieval_metrics[RetrievalMetric.ONTOLOGY_CONTEXT_MODE] = mode.value
    if mode == OntologyContextMode.SELECTED_SINGLE_ONTOLOGY:
        resolved = await _resolve_selected_single_ontology_context(context, tools, unit)
    elif mode == OntologyContextMode.FIXED_SINGLE_ONTOLOGY:
        resolved = await _resolve_fixed_single_ontology_context(context, tools, unit)
    elif mode == OntologyContextMode.SELECTED_VECTOR_SEARCH_ONTOLOGY:
        require_vector_retrieval(tools)
        resolved = await _resolve_ensemble_context(context, tools, unit)
    else:
        raise ValueError(f"Unknown ontology_context_mode: {mode!r}")
    # Recorded here rather than in each resolver so no mode can be added without
    # a size: the two that bound nothing were also the two that reported nothing.
    context.retrieval_metrics[RetrievalMetric.ONTOLOGY_SNAPSHOT_TRIPLES] = len(
        resolved.snapshot.graph
    )
    # Checked here, not per mode, for the same reason the size is recorded here:
    # every mode can return an empty context, and the two that bound nothing
    # were also the two that reported nothing.
    #
    # Two exemptions, both because an empty context is not a fault for them:
    #
    # * A unit with no retrievable text has nothing to extract either way.
    # * A caller that can create vocabulary. The ontology renderer branches on
    #   exactly this condition -- an empty seed sends it to
    #   ``render_ontology_fresh``, which mints a new catalog ontology from the
    #   text -- so raising here made the one path designed for an empty catalog
    #   unreachable, and turned "this corpus has no ontology yet" into a
    #   deployment error. It also stopped a populated-catalog run whenever the
    #   selector honestly reported that no catalog ontology fits.
    if (
        not len(resolved.snapshot.graph)
        and unit.text.strip()
        and not can_create_vocabulary
        and tools.config.server.ontology_context_required
    ):
        reason = context.retrieval_metrics.get(
            RetrievalMetric.EMPTY_SNAPSHOT_REASON, "no ontology context was assembled"
        )
        raise EmptyOntologyContextError(
            f"Ontology context for this content unit is empty: {reason}. "
            "Extraction would fall back on generic vocabulary and the "
            "conformance gate would then have no node to constrain, reporting "
            "a vacuous pass. Fix the catalog, or set "
            "ONTOLOGY_CONTEXT_REQUIRED=false to extract without one "
            "deliberately."
        )
    if not len(resolved.snapshot.graph) and can_create_vocabulary:
        logger.info(
            "No ontology context for this unit; rendering a fresh ontology from "
            "its text"
        )
    return resolved


def aggregate_writable_metrics(
    unit_contexts: dict[int, UnitOntologyContext]
    | dict[int, tuple[str, list[str], OntologyAssemblyMode]],
) -> tuple[
    dict[int, str],
    dict[int, list[str]],
    dict[int, OntologyAssemblyMode],
    dict[str, int],
]:
    """Aggregate per-unit writable IRI / source / mode metrics.

    Accepts either :class:`UnitOntologyContext` or legacy
    ``(primary_iri, patch_sources, mode)`` tuples for map-stage collect.
    """
    unit_primary_assignment: dict[int, str] = {}
    unit_patch_sources: dict[int, list[str]] = {}
    unit_context_mode_used: dict[int, OntologyAssemblyMode] = {}
    primary_counts: Counter[str] = Counter()
    for unit_index, context in unit_contexts.items():
        if isinstance(context, tuple):
            primary_iri, patch_sources, assembly_mode = context
        else:
            primary_iri = context.primary_writable_iri
            patch_sources = context.patch_sources
            assembly_mode = context.assembly_mode
        unit_primary_assignment[unit_index] = primary_iri
        unit_patch_sources[unit_index] = patch_sources
        unit_context_mode_used[unit_index] = assembly_mode
        primary_counts[primary_iri] += 1
    return (
        unit_primary_assignment,
        unit_patch_sources,
        unit_context_mode_used,
        dict(primary_counts),
    )
