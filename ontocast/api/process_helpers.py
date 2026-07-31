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


def facts_ttl_output_path(
    file_path: pathlib.Path,
    *,
    line_number: int | None = None,
) -> pathlib.Path:
    """Return the sibling ``.facts.ttl`` path for a processed input file."""
    if line_number is not None:
        return file_path.with_name(f"{file_path.stem}.L{line_number}.facts.ttl")
    return file_path.with_name(f"{file_path.stem}.facts.ttl")


def dump_facts_ttl(
    state: AgentState,
    file_path: pathlib.Path,
    *,
    line_number: int | None = None,
) -> pathlib.Path | None:
    """Write chunk-stripped facts Turtle next to the input file when facts exist."""
    if state.aggregated_facts is None or len(state.aggregated_facts) == 0:
        return None
    ttl_content = turtle_from_graph(state.aggregated_facts, strip_provenance=True)
    output_path = facts_ttl_output_path(file_path, line_number=line_number)
    output_path.write_text(ttl_content, encoding="utf-8")
    logger.info(
        "Dumped facts graph with chunk-level provenance stripped to %s",
        output_path,
    )
    return output_path


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


def _resolve_document_metadata(
    file_path: pathlib.Path,
    document_metadata: dict[str, object] | None,
    *,
    line_number: int | None = None,
) -> dict[str, object]:
    """Return explicit metadata, or filename fallback when none was provided."""
    if document_metadata:
        return dict(document_metadata)
    title = file_path.name
    if line_number is not None:
        title = f"{file_path.name}:{line_number}"
    return {"title": title}


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
    max_visits: int | None = None,
    document_metadata: dict[str, object] | None = None,
) -> list[AgentState]:
    """Expand a local input file into one ``AgentState`` per logical record."""
    file_bytes = file_path.read_bytes()
    resolved_max_visits = (
        max_visits if max_visits is not None else config.server.max_visits_per_node
    )
    base_state_kwargs = {
        "max_visits": resolved_max_visits,
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
                document_metadata=_resolve_document_metadata(
                    file_path, document_metadata
                ),
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
                document_metadata=_resolve_document_metadata(
                    file_path,
                    document_metadata,
                    line_number=line_number,
                ),
                **base_state_kwargs,
            )
        )
    return states


def select_unit_facts_ontology_graph(onto_result, facts_result) -> RDFGraph:
    """Return ontology graph for facts post-processing in unit pipeline flows."""
    if facts_result is not None:
        return facts_result.ontology_snapshot.graph
    if onto_result is not None:
        if (
            onto_result.fresh_ontology is not None
            and not onto_result.fresh_ontology.is_null()
            and len(onto_result.fresh_ontology.graph) > 0
        ):
            return onto_result.fresh_ontology.graph
        if len(onto_result.working_graph) > 0:
            return onto_result.working_graph
        if not onto_result.ontology_snapshot.is_empty():
            return onto_result.ontology_snapshot.graph
    return RDFGraph()


def _effective_document_metadata(state: AgentState) -> dict[str, object]:
    metadata = dict(state.document_metadata)
    if (
        state.source_url
        and "source_url" not in metadata
        and "source_uri" not in metadata
    ):
        metadata["source_url"] = state.source_url
    return metadata


async def persist_unit_pipeline_outputs(
    state: AgentState,
    onto_result,
    facts_result,
    tools: ToolBox,
) -> None:
    """Serialize unit-pipeline outputs using the standard document serializer."""
    if onto_result is not None:
        if (
            onto_result.fresh_ontology is not None
            and not onto_result.fresh_ontology.is_null()
        ):
            state.reduced_ontology_artifacts = [onto_result.fresh_ontology]
    if facts_result is not None:
        ontology_graph = select_unit_facts_ontology_graph(onto_result, facts_result)
        state.aggregated_facts = tools.aggregator.postprocess_facts_units(
            units=[facts_result.content_unit],
            ontology_graph=ontology_graph,
            doc_iri=state.doc_iri,
            document_metadata=_effective_document_metadata(state),
            doc_namespace=state.doc_namespace,
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
    max_visits: int | None = None,
    document_metadata: dict[str, object] | None = None,
) -> None:
    resolved_max_visits = (
        max_visits if max_visits is not None else config.server.max_visits_per_node
    )
    recursion_limit = calculate_recursion_limit(
        head_chunks,
        config.server,
        max_visits_per_node=resolved_max_visits,
    )
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
                max_visits=resolved_max_visits,
                document_metadata=document_metadata,
            )
            for state_index, state in enumerate(states):
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
                    workflow_state: AgentState | dict | None = None
                    async for chunk in workflow.astream(
                        state,
                        stream_mode="values",
                        config=RunnableConfig(recursion_limit=recursion_limit),
                    ):
                        workflow_state = chunk
                    if isinstance(workflow_state, AgentState):
                        state = workflow_state
                    elif isinstance(workflow_state, dict):
                        facts = workflow_state.get("aggregated_facts")
                        if facts is not None:
                            state.aggregated_facts = facts
                        meta = workflow_state.get("document_metadata")
                        if meta:
                            state.document_metadata = dict(meta)
                        doc_hid = workflow_state.get("doc_hid")
                        if doc_hid:
                            state.doc_hid = doc_hid
                        current_domain = workflow_state.get("current_domain")
                        if current_domain:
                            state.current_domain = current_domain
                line_number: int | None = None
                if file_path.suffix.lower() == ".jsonl" and len(states) > 1:
                    # Recover line from virtual raw_input key "...:N.json"
                    raw_key = next(iter(state.raw_input), "")
                    marker = f"{file_path.as_posix()}:"
                    if raw_key.startswith(marker) and raw_key.endswith(".json"):
                        try:
                            line_number = int(raw_key[len(marker) : -len(".json")])
                        except ValueError:
                            line_number = state_index + 1
                    else:
                        line_number = state_index + 1
                dump_facts_ttl(state, file_path, line_number=line_number)
        except Exception:
            logger.exception("Error processing %s", file_path)
