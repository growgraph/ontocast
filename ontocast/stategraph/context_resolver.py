from collections import Counter

from pydantic import BaseModel, Field

from ontocast.onto.content_unit import SourceUnit
from ontocast.onto.enum import (
    OntologyAssemblyMode,
    OntologyContextMode,
    UnitContextStrategy,
)
from ontocast.onto.null import NULL_ONTOLOGY
from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_access import document_ontology_access
from ontocast.onto.state import AgentState
from ontocast.tool.chunk.util import split_proposition_windows
from ontocast.toolbox import ToolBox


class UnitOntologyContext(BaseModel):
    anchor_iri: str
    ontology_snapshot: Ontology
    patch_sources: list[str] = Field(default_factory=list)
    assembly_mode: OntologyAssemblyMode
    confidence: float = 0.0


def _primary_document_context(state: AgentState) -> UnitOntologyContext:
    """Use the document primary ontology with no vector or ensemble retrieval."""
    primary = document_ontology_access(state).primary_ontology()
    return UnitOntologyContext(
        anchor_iri=primary.iri,
        ontology_snapshot=primary,
        patch_sources=[],
        assembly_mode=OntologyAssemblyMode.PRIMARY_WITHOUT_RETRIEVAL,
        confidence=0.0,
    )


def _unit_queries(unit: SourceUnit, tools: ToolBox) -> list[str]:
    qcfg = tools.config.tool_config.qdrant
    text = unit.text.strip()
    if not text:
        return []
    if not qcfg.proposition_retrieval_enabled:
        return [text]
    return split_proposition_windows(
        text,
        max_sentences=qcfg.proposition_window_sentences,
        max_windows=qcfg.proposition_max_windows,
    )


def _majority_iri(counts: Counter[str]) -> tuple[str | None, float]:
    if not counts:
        return None, 0.0
    dominant_iri, dominant_count = counts.most_common(1)[0]
    total = sum(counts.values())
    confidence = (dominant_count / total) if total > 0 else 0.0
    return dominant_iri, confidence


async def _resolve_vote_majority_context(
    state: AgentState,
    tools: ToolBox,
    unit: SourceUnit,
) -> UnitOntologyContext:
    """Majority vote over vector patch hits; otherwise document primary."""
    queries = _unit_queries(unit, tools)
    if not queries or tools.vector_store is None:
        return _primary_document_context(state)

    qcfg = tools.config.tool_config.qdrant
    hits_by_query = await tools.vector_store.asearch_patch_hits_many(
        queries=queries,
        top_k=qcfg.top_k,
    )
    counts: Counter[str] = Counter()
    for hits in hits_by_query:
        for hit in hits:
            if hit.atom.ontology_iri:
                counts[hit.atom.ontology_iri] += 1

    dominant_iri, confidence = _majority_iri(counts)
    if dominant_iri:
        ontology = tools.ontology_manager.get_freshest_terminal_ontology_by_iri(
            dominant_iri
        )
        if ontology is None:
            ontology = tools.ontology_manager.get_ontology(ontology_iri=dominant_iri)
        if not ontology.is_null():
            return UnitOntologyContext(
                anchor_iri=ontology.iri,
                ontology_snapshot=ontology,
                patch_sources=[ontology.iri],
                assembly_mode=OntologyAssemblyMode.VOTE_MAJORITY_ONTOLOGY,
                confidence=confidence,
            )

    return _primary_document_context(state)


async def _resolve_ensemble_context(
    state: AgentState,
    tools: ToolBox,
    unit: SourceUnit,
) -> UnitOntologyContext:
    queries = _unit_queries(unit, tools)
    if not queries or tools.patch_retriever is None:
        return await _resolve_vote_majority_context(state, tools, unit)

    qcfg = tools.config.tool_config.qdrant
    patch_graph, source_iris = await tools.patch_retriever.aretrieve_ensemble(
        queries=queries,
        top_k=qcfg.top_k,
        expand_sparql=True,
        subgraph_depth=qcfg.induced_subgraph_depth,
        max_triples=qcfg.induced_subgraph_max_triples,
    )
    if len(patch_graph) == 0:
        return await _resolve_vote_majority_context(state, tools, unit)

    anchor_iri = source_iris[0] if source_iris else NULL_ONTOLOGY.iri
    ontology_snapshot = Ontology(
        ontology_id=None,
        title="Retrieved unit patch context",
        description="Composite ontology context assembled from unit-level retrieval.",
        graph=patch_graph,
        iri=anchor_iri,
        current_domain=state.current_domain,
    )
    return UnitOntologyContext(
        anchor_iri=anchor_iri,
        ontology_snapshot=ontology_snapshot,
        patch_sources=source_iris,
        assembly_mode=OntologyAssemblyMode.ENSEMBLE_STITCHED,
        confidence=1.0 if source_iris else 0.5,
    )


async def resolve_unit_ontology_context(
    state: AgentState,
    tools: ToolBox,
    unit: SourceUnit,
) -> UnitOntologyContext:
    if state.ontology_context_mode == OntologyContextMode.FULL_TTL:
        return _primary_document_context(state)
    strategy = state.unit_context_strategy
    if strategy == UnitContextStrategy.VOTE_FIRST:
        return await _resolve_vote_majority_context(state, tools, unit)
    if strategy == UnitContextStrategy.HYBRID_ADAPTIVE:
        ensemble = await _resolve_ensemble_context(state, tools, unit)
        if len(ensemble.ontology_snapshot.graph) > 0:
            return ensemble
        return await _resolve_vote_majority_context(state, tools, unit)
    return await _resolve_ensemble_context(state, tools, unit)


def aggregate_anchor_metrics(
    unit_contexts: dict[int, UnitOntologyContext]
    | dict[int, tuple[str, list[str], OntologyAssemblyMode]],
) -> tuple[
    dict[int, str],
    dict[int, list[str]],
    dict[int, OntologyAssemblyMode],
    dict[str, int],
]:
    unit_anchor_assignment: dict[int, str] = {}
    unit_patch_sources: dict[int, list[str]] = {}
    unit_context_mode_used: dict[int, OntologyAssemblyMode] = {}
    anchor_counts: Counter[str] = Counter()
    for unit_index, context in unit_contexts.items():
        if isinstance(context, tuple):
            anchor_iri, patch_sources, assembly_mode = context
        else:
            anchor_iri = context.anchor_iri
            patch_sources = context.patch_sources
            assembly_mode = context.assembly_mode
        unit_anchor_assignment[unit_index] = anchor_iri
        unit_patch_sources[unit_index] = patch_sources
        unit_context_mode_used[unit_index] = assembly_mode
        anchor_counts[anchor_iri] += 1
    return (
        unit_anchor_assignment,
        unit_patch_sources,
        unit_context_mode_used,
        dict(anchor_counts),
    )
