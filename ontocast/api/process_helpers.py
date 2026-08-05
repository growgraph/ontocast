"""Shared helpers for local batch processing and HTTP response assembly."""

import asyncio
import logging
import pathlib
import re
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from ontocast.agent.serialize import serialize as serialize_agent_state
from ontocast.config import Config, ServerConfig
from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.enum import OntologyContextMode
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.stategraph.unit_pipeline import DocumentConversionError, run_unit_pipeline
from ontocast.tool.facts_invariants import (
    collect_shacl_shapes,
    validate_aggregated_facts,
)
from ontocast.tool.triple_manager.core import TripleStoreManager
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)

_SAFE_ONTOLOGY_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")


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


def resolve_batch_output_dirs(
    output_dir: pathlib.Path | None,
    facts_output_dir: pathlib.Path | None,
    ontology_output_dir: pathlib.Path | None,
) -> tuple[pathlib.Path | None, pathlib.Path | None]:
    """Resolve facts/ontology dump dirs from shared and override flags.

    Returns:
        ``(facts_dir, ontology_dir)``. ``None`` means sibling-of-input.
    """
    facts_dir = facts_output_dir or output_dir
    ontology_dir = ontology_output_dir or output_dir
    return facts_dir, ontology_dir


def _ttl_basename(
    file_path: pathlib.Path,
    *,
    line_number: int | None,
    kind: str,
    ontology_id: str | None = None,
) -> str:
    stem = file_path.stem
    line_part = f".L{line_number}" if line_number is not None else ""
    id_part = f".{ontology_id}" if ontology_id else ""
    return f"{stem}{line_part}{id_part}.{kind}.ttl"


def facts_ttl_output_path(
    file_path: pathlib.Path,
    *,
    line_number: int | None = None,
    output_dir: pathlib.Path | None = None,
) -> pathlib.Path:
    """Return the ``.facts.ttl`` path for a processed input file."""
    name = _ttl_basename(file_path, line_number=line_number, kind="facts")
    if output_dir is not None:
        return output_dir / name
    return file_path.with_name(name)


def safe_ontology_filename_id(ontology: Ontology) -> str | None:
    """Return a filesystem-safe ontology id fragment, or None if unavailable."""
    raw = ontology.ontology_id
    if not raw and ontology.iri:
        raw = ontology.iri.rstrip("/").rsplit("/", 1)[-1]
    if not raw:
        return None
    cleaned = _SAFE_ONTOLOGY_ID_RE.sub("_", raw).strip("._-")
    return cleaned or None


def ontology_ttl_output_path(
    file_path: pathlib.Path,
    *,
    line_number: int | None = None,
    output_dir: pathlib.Path | None = None,
    ontology_id: str | None = None,
) -> pathlib.Path:
    """Return the ``.ontology.ttl`` path for a processed input file."""
    name = _ttl_basename(
        file_path,
        line_number=line_number,
        kind="ontology",
        ontology_id=ontology_id,
    )
    if output_dir is not None:
        return output_dir / name
    return file_path.with_name(name)


def dump_facts_ttl(
    state: AgentState,
    file_path: pathlib.Path,
    *,
    line_number: int | None = None,
    output_dir: pathlib.Path | None = None,
) -> pathlib.Path | None:
    """Write chunk-stripped facts Turtle when facts exist."""
    if state.aggregated_facts is None or len(state.aggregated_facts) == 0:
        return None
    ttl_content = turtle_from_graph(state.aggregated_facts, strip_provenance=True)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    output_path = facts_ttl_output_path(
        file_path, line_number=line_number, output_dir=output_dir
    )
    output_path.write_text(ttl_content, encoding="utf-8")
    logger.info(
        "Dumped facts graph with chunk-level provenance stripped to %s",
        output_path,
    )
    return output_path


def _ontology_artifacts_for_dump(state: AgentState) -> list[Ontology]:
    artifacts = (
        state.reduced_ontology_artifacts
        if state.reduced_ontology_artifacts
        else state.ontology_artifacts
    )
    return [
        artifact
        for artifact in artifacts
        if artifact is not None and not artifact.is_null() and len(artifact.graph) > 0
    ]


def dump_ontology_ttls(
    state: AgentState,
    file_path: pathlib.Path,
    *,
    line_number: int | None = None,
    output_dir: pathlib.Path | None = None,
) -> list[pathlib.Path]:
    """Write provenance-stripped ontology Turtle dumps when artifacts exist."""
    artifacts = _ontology_artifacts_for_dump(state)
    if not artifacts:
        return []
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    include_ids = len(artifacts) > 1
    written: list[pathlib.Path] = []
    for artifact in artifacts:
        ontology_id = safe_ontology_filename_id(artifact) if include_ids else None
        ttl_content = turtle_from_graph(artifact.graph, strip_provenance=True)
        output_path = ontology_ttl_output_path(
            file_path,
            line_number=line_number,
            output_dir=output_dir,
            ontology_id=ontology_id,
        )
        output_path.write_text(ttl_content, encoding="utf-8")
        logger.info(
            "Dumped ontology graph with provenance stripped to %s",
            output_path,
        )
        written.append(output_path)
    return written


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
    # Explicitly typed: the splat below is only checkable if the mapping's
    # value type is known, and a heterogeneous literal infers as a union that
    # matches no single field.
    base_state_kwargs: dict[str, Any] = {
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
        ).graph
        _validate_unit_pipeline_facts(state, ontology_graph, tools)
    await asyncio.to_thread(serialize_agent_state, state, tools)


def _validate_unit_pipeline_facts(
    state: AgentState,
    ontology_graph: RDFGraph,
    tools: ToolBox,
) -> None:
    """Run the post-aggregation invariant gate for the single-unit path.

    The document graph reaches this gate at VALIDATE_FACTS; the unit pipeline
    does not run the graph, so without this call ``/process_unit`` would ship
    facts with no functional-violation, coreference, or SHACL check at all.
    Detection only: the un-merge repair re-aggregates *retained units against
    each other*, which has no meaning for a single unit.
    """
    facts_validation = tools.config.get_tool_config().facts_validation
    shapes_graph = collect_shacl_shapes(ontology_graph, facts_validation.shapes_dir)
    report = validate_aggregated_facts(
        state.aggregated_facts,
        ontology_graph,
        shapes_graph=shapes_graph,
        fact_namespaces=[DEFAULT_IRI, str(state.doc_iri), state.doc_namespace or ""],
        suspect_multi_value_severity=facts_validation.suspect_multi_value_severity,
        functional_min_single_support=facts_validation.functional_min_single_support,
        quantity_fallback_vocabulary=facts_validation.quantity_fallback_vocabulary,
    )
    state.facts_validation_findings = report.findings
    state.retrieval_metrics["facts_validation_findings"] = len(report.findings)
    state.retrieval_metrics["facts_validation_errors"] = len(report.error_findings)
    if report.error_findings:
        logger.warning(
            "Unit-pipeline facts validation: %d error finding(s) "
            "(no un-merge repair in single-unit mode)",
            len(report.error_findings),
        )


def _merge_workflow_state_into_agent_state(
    state: AgentState,
    workflow_state: AgentState | dict,
) -> AgentState:
    """Copy dump-relevant fields from an astream values chunk onto ``state``."""
    if isinstance(workflow_state, AgentState):
        return workflow_state
    if not isinstance(workflow_state, dict):
        return state
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
    reduced = workflow_state.get("reduced_ontology_artifacts")
    if reduced is not None:
        state.reduced_ontology_artifacts = list(reduced)
    artifacts = workflow_state.get("ontology_artifacts")
    if artifacts is not None:
        state.ontology_artifacts = list(artifacts)
    # Without these the post-aggregation validation gate is log-only on the
    # batch path: its findings and counters never reach the dumped state.
    findings = workflow_state.get("facts_validation_findings")
    if findings is not None:
        state.facts_validation_findings = list(findings)
    metrics = workflow_state.get("retrieval_metrics")
    if metrics:
        state.retrieval_metrics = dict(metrics)
    return state


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
    output_dir: pathlib.Path | None = None,
    facts_output_dir: pathlib.Path | None = None,
    ontology_output_dir: pathlib.Path | None = None,
) -> list[pathlib.Path]:
    """Process each input file, isolating per-file failures.

    Returns:
        The files that failed, in input order. Empty on full success. Callers
        use this to set a non-zero exit code -- previously every failure was
        logged and swallowed, so ``ontocast process`` exited 0 even when no
        file produced any output.
    """
    failed_files: list[pathlib.Path] = []
    resolved_max_visits = (
        max_visits if max_visits is not None else config.server.max_visits_per_node
    )
    recursion_limit = calculate_recursion_limit(
        head_chunks,
        config.server,
        max_visits_per_node=resolved_max_visits,
    )
    facts_dir, ontology_dir = resolve_batch_output_dirs(
        output_dir, facts_output_dir, ontology_output_dir
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
                        if file_path not in failed_files:
                            failed_files.append(file_path)
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
                    if workflow_state is not None:
                        state = _merge_workflow_state_into_agent_state(
                            state, workflow_state
                        )
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
                dump_facts_ttl(
                    state,
                    file_path,
                    line_number=line_number,
                    output_dir=facts_dir,
                )
                dump_ontology_ttls(
                    state,
                    file_path,
                    line_number=line_number,
                    output_dir=ontology_dir,
                )
        except Exception:
            logger.exception("Error processing %s", file_path)
            if file_path not in failed_files:
                failed_files.append(file_path)

    return failed_files
