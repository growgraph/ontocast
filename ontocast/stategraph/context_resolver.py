import logging
from collections import Counter

from pydantic import BaseModel, Field

from ontocast.agent.select_ontology_catalog import select_catalog_ontology_for_excerpt
from ontocast.onto.content_unit import SourceUnit
from ontocast.onto.enum import OntologyAssemblyMode, OntologyContextMode
from ontocast.onto.null import NULL_ONTOLOGY
from ontocast.onto.ontology_access import document_ontology_access
from ontocast.onto.ontology_snapshot import OntologySnapshot
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.retrieval_capabilities import require_vector_retrieval
from ontocast.onto.state import AgentState
from ontocast.tool.chunk.proposition import split_proposition_windows
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
    state: AgentState,
) -> UnitOntologyContext | None:
    """Build merged ontology context from reduced document artifacts."""
    artifacts = [
        ontology
        for ontology in document_ontology_access(state).reduced_artifacts()
        if not ontology.is_null() and len(ontology.graph) > 0
    ]
    if not artifacts:
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
    return UnitOntologyContext(
        snapshot=snapshot,
        writable_iris=list(patch_sources),
        confidence=1.0,
    )


async def _resolve_selected_single_ontology_context(
    state: AgentState,
    tools: ToolBox,
    unit: SourceUnit,
) -> UnitOntologyContext:
    """One catalog ontology chosen by the LLM from the unit text."""
    selected = await select_catalog_ontology_for_excerpt(
        tools.ontology_manager,
        tools.llm,
        unit.text,
        state.ontology_selection_user_instruction,
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
    state: AgentState,
    tools: ToolBox,
    unit: SourceUnit,
) -> UnitOntologyContext:
    """Catalog ontology fixed by ontology_id (fresh terminal revision)."""
    _ = unit
    cleaned = state.ontology_context_fixed_ontology_id.strip()
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
    state: AgentState,
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
        state.retrieval_metrics["patch_retrieval"] = metrics
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
    state: AgentState,
    tools: ToolBox,
    unit: SourceUnit,
) -> UnitOntologyContext:
    mode = state.ontology_context_mode
    state.retrieval_metrics["ontology_context_mode"] = mode.value
    if mode == OntologyContextMode.SELECTED_SINGLE_ONTOLOGY:
        return await _resolve_selected_single_ontology_context(state, tools, unit)
    if mode == OntologyContextMode.FIXED_SINGLE_ONTOLOGY:
        return await _resolve_fixed_single_ontology_context(state, tools, unit)
    if mode == OntologyContextMode.SELECTED_VECTOR_SEARCH_ONTOLOGY:
        require_vector_retrieval(tools)
        return await _resolve_ensemble_context(state, tools, unit)
    raise ValueError(f"Unknown ontology_context_mode: {mode!r}")


async def resolve_effective_facts_ontology_context(
    state: AgentState,
    tools: ToolBox,
    unit: SourceUnit,
) -> UnitOntologyContext:
    """Resolve facts context preferring merged document artifacts when available."""
    merged_context = build_merged_document_ontology_context(state)
    if merged_context is not None:
        return merged_context
    return await resolve_unit_ontology_context(state, tools, unit)


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
