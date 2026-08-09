import asyncio
import logging
import time
from collections import Counter

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
from ontocast.onto.retrieval_capabilities import require_vector_retrieval
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
        # vocabulary at all. Distinguish the causes: an empty index is a
        # deployment problem, everything-below-threshold is a tuning problem,
        # and neither should read as "this passage had no relevant terms".
        indexed_iris: set[str] = set()
        if tools.vector_store is not None:
            try:
                indexed_iris = await asyncio.to_thread(
                    tools.vector_store.list_indexed_ontology_iris
                )
            except Exception as exc:
                logger.warning("Could not inspect the vector index: %s", exc)
        if not indexed_iris:
            reason = "vector index is empty or unreadable"
        elif metrics and metrics.get("atoms_after_dedupe"):
            reason = "all candidate atoms scored below the retrieval thresholds"
        else:
            reason = "no candidate atoms matched the unit's queries"
        context.retrieval_metrics[RetrievalMetric.EMPTY_SNAPSHOT_REASON] = reason
        logger.warning(
            "Ontology context for this unit is empty (%s); extraction will "
            "proceed with no catalog vocabulary.",
            reason,
        )

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
) -> UnitOntologyContext:
    mode = context.ontology_context_mode
    context.retrieval_metrics[RetrievalMetric.ONTOLOGY_CONTEXT_MODE] = mode.value
    if mode == OntologyContextMode.SELECTED_SINGLE_ONTOLOGY:
        return await _resolve_selected_single_ontology_context(context, tools, unit)
    if mode == OntologyContextMode.FIXED_SINGLE_ONTOLOGY:
        return await _resolve_fixed_single_ontology_context(context, tools, unit)
    if mode == OntologyContextMode.SELECTED_VECTOR_SEARCH_ONTOLOGY:
        require_vector_retrieval(tools)
        return await _resolve_ensemble_context(context, tools, unit)
    raise ValueError(f"Unknown ontology_context_mode: {mode!r}")


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


# Back-compat alias used by older call sites / tests during migration.
aggregate_anchor_metrics = aggregate_writable_metrics
