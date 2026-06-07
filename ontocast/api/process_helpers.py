"""Shared helpers for local batch processing and HTTP response assembly."""

import asyncio
import logging
import pathlib

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from ontocast.agent.serialize import serialize as serialize_agent_state
from ontocast.config import Config, ServerConfig
from ontocast.onto.enum import OntologyContextMode
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.stategraph.unit_pipeline import DocumentConversionError, run_unit_pipeline
from ontocast.tool.triple_manager.core import TripleStoreManager
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def get_supported_input_extensions(tools: ToolBox) -> tuple[str, ...]:
    """Return all input file suffixes handled by document conversion."""
    built_in_suffixes = {".json", ".jsonl", ".txt"}
    converter_suffixes = set(tools.converter.supported_extensions)
    return tuple(sorted(built_in_suffixes | converter_suffixes))


def turtle_from_graph(graph: RDFGraph, *, strip_provenance: bool) -> str:
    """Serialize ``graph`` to Turtle, optionally stripping reification/provenance."""
    out: RDFGraph = (
        TripleStoreManager.strip_provenance(graph) if strip_provenance else graph
    )
    return out.serialize_canonical_turtle()


async def flush_triple_configured_scope(tools: ToolBox) -> None:
    """Match POST /flush without tenant/project: triple store only, current scope."""
    if tools.triple_store_manager is not None:
        await tools.triple_store_manager.clean()


def calculate_recursion_limit(
    head_chunks: int | None,
    server_config: ServerConfig,
    *,
    max_visits_per_node: int | None = None,
) -> int:
    """Calculate the recursion limit based on max visits and head chunks."""
    visits = (
        max_visits_per_node
        if max_visits_per_node is not None
        else server_config.max_visits_per_node
    )
    if head_chunks is not None:
        return max(
            server_config.base_recursion_limit,
            visits * head_chunks * 10,
        )
    return max(
        server_config.base_recursion_limit,
        visits * server_config.estimated_chunks * 10,
    )


def expand_input_to_states(
    file_path: pathlib.Path,
    *,
    config: Config,
    head_chunks: int | None,
    ontology_context_mode_value: OntologyContextMode,
    tenant: str | None,
    project: str | None,
    target_sections: list[str] | None = None,
    summarize_sections: list[str] | None = None,
    summary_max_sentences: int = 5,
    document_type_hint: str | None = None,
    section_schema_id: str | None = None,
) -> list[AgentState]:
    """Expand a local input file into one ``AgentState`` per logical record."""
    file_bytes = file_path.read_bytes()
    base_state_kwargs = {
        "max_visits": config.server.max_visits_per_node,
        "max_chunks": head_chunks,
        "render_mode": config.server.render_mode,
        "llm_graph_format": config.server.llm_graph_format,
        "ontology_context_mode": ontology_context_mode_value,
        "ontology_context_fixed_ontology_id": (
            config.server.ontology_context_fixed_ontology_id
        ),
        "tenant": tenant,
        "project": project,
        "target_sections": target_sections,
        "summarize_sections": summarize_sections,
        "summary_max_sentences": summary_max_sentences,
        "document_type_hint": document_type_hint,
        "section_schema_id": section_schema_id,
    }

    if file_path.suffix.lower() != ".jsonl":
        return [
            AgentState(
                raw_input={file_path.as_posix(): file_bytes},
                **base_state_kwargs,
            )
        ]

    states: list[AgentState] = []
    for line_number, line in enumerate(
        file_bytes.decode("utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        virtual_path = f"{file_path.as_posix()}:{line_number}.json"
        states.append(
            AgentState(
                raw_input={virtual_path: line.encode("utf-8")},
                **base_state_kwargs,
            )
        )
    return states


def select_unit_facts_ontology_graph(onto_result, facts_result) -> RDFGraph:
    """Return ontology graph for facts post-processing in unit pipeline flows."""
    if facts_result is not None:
        return facts_result.ontology_snapshot.graph
    if (
        onto_result is not None
        and not onto_result.current_ontology.is_null()
        and len(onto_result.current_ontology.graph) > 0
    ):
        return onto_result.current_ontology.graph
    return RDFGraph()


async def persist_unit_pipeline_outputs(
    state: AgentState,
    onto_result,
    facts_result,
    tools: ToolBox,
) -> None:
    """Serialize unit-pipeline outputs using the standard document serializer."""
    if onto_result is not None and not onto_result.current_ontology.is_null():
        state.reduced_ontology_artifacts = [onto_result.current_ontology]
    if facts_result is not None:
        ontology_graph = select_unit_facts_ontology_graph(onto_result, facts_result)
        state.aggregated_facts = tools.aggregator.postprocess_facts_units(
            units=[facts_result.content_unit],
            ontology_graph=ontology_graph,
        )
    await asyncio.to_thread(serialize_agent_state, state, tools)


async def process_files_input(
    files: list[pathlib.Path],
    *,
    config: Config,
    head_chunks: int | None,
    use_unit_pipeline: bool,
    tools: ToolBox,
    workflow: CompiledStateGraph,
    ontology_context_mode_value: OntologyContextMode,
    tenant: str | None,
    project: str | None,
    target_sections: list[str] | None = None,
    summarize_sections: list[str] | None = None,
    summary_max_sentences: int = 5,
    document_type_hint: str | None = None,
    section_schema_id: str | None = None,
) -> None:
    recursion_limit = calculate_recursion_limit(head_chunks, config.server)
    for file_path in files:
        try:
            states = expand_input_to_states(
                file_path,
                config=config,
                head_chunks=head_chunks,
                ontology_context_mode_value=ontology_context_mode_value,
                tenant=tenant,
                project=project,
                target_sections=target_sections,
                summarize_sections=summarize_sections,
                summary_max_sentences=summary_max_sentences,
                document_type_hint=document_type_hint,
                section_schema_id=section_schema_id,
            )
            for state in states:
                if use_unit_pipeline:
                    try:
                        onto_result, facts_result = await run_unit_pipeline(
                            state, tools
                        )
                    except DocumentConversionError as exc:
                        logger.error("Error processing %s: %s", file_path, exc)
                        continue
                    await persist_unit_pipeline_outputs(
                        state, onto_result, facts_result, tools
                    )
                else:
                    async for _ in workflow.astream(
                        state,
                        stream_mode="values",
                        config=RunnableConfig(recursion_limit=recursion_limit),
                    ):
                        pass
        except Exception:
            logger.exception("Error processing %s", file_path)
