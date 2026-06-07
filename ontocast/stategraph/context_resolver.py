import logging
from collections import Counter

from pydantic import BaseModel, Field
from rdflib.namespace import OWL, RDF

from ontocast.agent.select_ontology_catalog import select_catalog_ontology_for_excerpt
from ontocast.onto.content_unit import SourceUnit
from ontocast.onto.enum import OntologyAssemblyMode, OntologyContextMode
from ontocast.onto.null import NULL_ONTOLOGY
from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_access import document_ontology_access
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.retrieval_capabilities import require_vector_retrieval
from ontocast.onto.state import AgentState
from ontocast.tool.chunk.util import split_proposition_windows
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


class UnitOntologyContext(BaseModel):
    anchor_iri: str
    ontology_snapshot: Ontology
    patch_sources: list[str] = Field(default_factory=list)
    assembly_mode: OntologyAssemblyMode
    confidence: float = 0.0


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
    """Build one deterministic merged ontology context from reduced document artifacts."""
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

    anchor_iri = patch_sources[0] if patch_sources else NULL_ONTOLOGY.iri
    snapshot = Ontology(
        ontology_id=None,
        title="Merged document ontology context",
        description=(
            "Deterministic merge of reduced ontology artifacts used for facts context."
        ),
        graph=merged_graph,
        iri=anchor_iri,
        current_domain=state.current_domain,
    )
    return UnitOntologyContext(
        anchor_iri=anchor_iri,
        ontology_snapshot=snapshot,
        patch_sources=patch_sources,
        assembly_mode=OntologyAssemblyMode.DOCUMENT_MERGED_REDUCED,
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
    if selected.is_null():
        return UnitOntologyContext(
            anchor_iri=NULL_ONTOLOGY.iri,
            ontology_snapshot=NULL_ONTOLOGY,
            patch_sources=[],
            assembly_mode=OntologyAssemblyMode.SELECTED_SINGLE_ONTOLOGY_LLM,
            confidence=0.0,
        )
    return UnitOntologyContext(
        anchor_iri=selected.iri,
        ontology_snapshot=selected,
        patch_sources=[selected.iri],
        assembly_mode=OntologyAssemblyMode.SELECTED_SINGLE_ONTOLOGY_LLM,
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
    if not cleaned:
        return UnitOntologyContext(
            anchor_iri=NULL_ONTOLOGY.iri,
            ontology_snapshot=NULL_ONTOLOGY,
            patch_sources=[],
            assembly_mode=OntologyAssemblyMode.FIXED_SINGLE_ONTOLOGY,
            confidence=0.0,
        )
    mgr = tools.ontology_manager
    selected = mgr.get_freshest_terminal_ontology(ontology_id=cleaned)
    if selected is None:
        logger.warning(
            "No catalog ontology match for ontology_context_fixed_ontology_id=%r; "
            "using NULL_ONTOLOGY",
            cleaned,
        )
        return UnitOntologyContext(
            anchor_iri=NULL_ONTOLOGY.iri,
            ontology_snapshot=NULL_ONTOLOGY,
            patch_sources=[],
            assembly_mode=OntologyAssemblyMode.FIXED_SINGLE_ONTOLOGY,
            confidence=0.0,
        )
    return UnitOntologyContext(
        anchor_iri=selected.iri,
        ontology_snapshot=selected,
        patch_sources=[selected.iri],
        assembly_mode=OntologyAssemblyMode.FIXED_SINGLE_ONTOLOGY,
        confidence=1.0,
    )


async def _resolve_ensemble_context(
    state: AgentState,
    tools: ToolBox,
    unit: SourceUnit,
) -> UnitOntologyContext:
    """Stitched induced subgraphs from vector retrieval."""
    queries = _unit_queries(unit, tools)
    if not queries:
        empty = Ontology(
            ontology_id=None,
            title="Empty unit (no text queries for retrieval)",
            description="No proposition queries; ensemble graph is empty.",
            graph=RDFGraph(),
            iri=NULL_ONTOLOGY.iri,
            current_domain=state.current_domain,
        )
        return UnitOntologyContext(
            anchor_iri=NULL_ONTOLOGY.iri,
            ontology_snapshot=empty,
            patch_sources=[],
            assembly_mode=OntologyAssemblyMode.SELECTED_VECTOR_SEARCH_ENSEMBLE,
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
    )
    metrics = retriever.last_retrieval_metrics
    if metrics:
        state.retrieval_metrics["patch_retrieval"] = metrics
        logger.info(
            "Patch retrieval: queries=%s atoms_final=%s source_iris=%s expanded=%s triples=%s",
            metrics.get("query_count"),
            metrics.get("atoms_final"),
            metrics.get("source_ontology_iris"),
            metrics.get("expanded_ontology_iris"),
            metrics.get("snapshot_triple_count"),
        )
    anchor_iri = source_iris[0] if source_iris else NULL_ONTOLOGY.iri
    for onto_subject in {
        s for s, _, _ in patch_graph.triples((None, RDF.type, OWL.Ontology))
    }:
        for triple in list(patch_graph.triples((onto_subject, None, None))):
            patch_graph.remove(triple)
    patch_graph.sanitize_prefixes_namespaces()

    ontology_snapshot = Ontology(
        ontology_id=None,
        title=None,
        description=None,
        graph=patch_graph,
        iri=anchor_iri,
        current_domain=state.current_domain,
    )
    for onto_subject in {
        s for s, _, _ in ontology_snapshot.graph.triples((None, RDF.type, OWL.Ontology))
    }:
        for triple in list(ontology_snapshot.graph.triples((onto_subject, None, None))):
            ontology_snapshot.graph.remove(triple)

    return UnitOntologyContext(
        anchor_iri=anchor_iri,
        ontology_snapshot=ontology_snapshot,
        patch_sources=source_iris,
        assembly_mode=OntologyAssemblyMode.SELECTED_VECTOR_SEARCH_ENSEMBLE,
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
