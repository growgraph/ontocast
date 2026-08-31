"""Shared helpers for local batch processing and HTTP response assembly."""

import asyncio
import json
import logging
import pathlib
import re
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from ontocast._version import __version__
from ontocast.agent.serialize import serialize as serialize_agent_state
from ontocast.config import Config, ServerConfig
from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.enum import OntologyContextMode
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.run_manifest import (
    RunManifest,
    RunManifestLLM,
    RunManifestLoops,
    RunManifestSelection,
    RunManifestValidationConfig,
    summarize_loop,
)
from ontocast.onto.state import AgentState
from ontocast.stategraph.facts_gate import run_facts_gate
from ontocast.stategraph.unit_pipeline import DocumentConversionError, run_unit_pipeline
from ontocast.tool.chunk.prepare import SectionSelectionEmptyError
from ontocast.tool.triple_manager.core import TripleStoreManager
from ontocast.toolbox import ToolBox
from ontocast.util.graph_metrics import facts_graph_shape_metrics

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
    strip_provenance: bool = True,
) -> pathlib.Path | None:
    """Write the facts Turtle when facts exist.

    Args:
        state: Document state carrying ``aggregated_facts``.
        file_path: Source file the facts were extracted from.
        line_number: Record number for JSONL inputs.
        output_dir: Destination directory; defaults to the source's directory.
        strip_provenance: Drop chunk-level provenance from the dump. Keeping it
            is what lets a statement be traced back to its source span and
            re-verified against the document; stripping it stays the default so
            existing outputs are unchanged. Same meaning as the HTTP
            ``strip_provenance`` parameter.

    Returns:
        The path written, or None when there are no facts.
    """
    if state.aggregated_facts is None or len(state.aggregated_facts) == 0:
        return None
    ttl_content = turtle_from_graph(
        state.aggregated_facts, strip_provenance=strip_provenance
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    output_path = facts_ttl_output_path(
        file_path, line_number=line_number, output_dir=output_dir
    )
    output_path.write_text(ttl_content, encoding="utf-8")
    logger.info(
        "Dumped facts graph with chunk-level provenance %s to %s",
        "stripped" if strip_provenance else "retained",
        output_path,
    )
    return output_path


def dump_validation_report(
    state: AgentState,
    file_path: pathlib.Path,
    *,
    line_number: int | None = None,
    output_dir: pathlib.Path | None = None,
) -> pathlib.Path | None:
    """Write the conformance summary and residual findings beside the facts TTL.

    A batch run otherwise leaves no record of *why* a graph is non-conformant:
    the findings live on the state and are logged, and every downstream reader
    ends up re-running a validator to rebuild what the gate already computed.
    """
    if not state.facts_conformance and not state.facts_validation_findings:
        return None
    payload = {
        "source": file_path.name,
        "conformance": state.facts_conformance,
        "gate_repairs": [
            record.model_dump(mode="json") for record in state.facts_gate_repairs
        ],
        "findings": [
            finding.model_dump(mode="json")
            for finding in state.facts_validation_findings
        ],
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    facts_path = facts_ttl_output_path(
        file_path, line_number=line_number, output_dir=output_dir
    )
    output_path = facts_path.with_name(
        f"{facts_path.name.removesuffix('.ttl')}.validation.json"
    )
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Dumped facts validation report to %s", output_path)
    return output_path


def _selection_manifest(state: AgentState, config: Config) -> RunManifestSelection:
    """Selection settings plus the label census that says whether they acted."""
    histogram: dict[str, int] = {}
    for unit in state.content_units or []:
        label = unit.section_label or "(unlabeled)"
        histogram[label] = histogram.get(label, 0) + 1
    unlabeled = histogram.get("(unlabeled)", 0)
    return RunManifestSelection(
        target_sections=state.target_sections,
        exclude_sections=state.exclude_sections,
        summarize_sections=state.summarize_sections,
        summary_max_sentences=state.summary_max_sentences,
        bibliography_mode=str(config.get_tool_config().chunk_config.bibliography_mode),
        labeled_units=(sum(histogram.values()) - unlabeled) if histogram else None,
        unlabeled_units=unlabeled if histogram else None,
        section_label_histogram=histogram or None,
    )


def dump_run_manifest(
    state: AgentState,
    file_path: pathlib.Path,
    *,
    config: Config,
    line_number: int | None = None,
    output_dir: pathlib.Path | None = None,
    shapes_triples: int | None = None,
    shapes_prompt_selection: bool | None = None,
) -> pathlib.Path | None:
    """Write the run's cost and configuration beside the facts TTL.

    ``BudgetTracker`` is returned over HTTP and logged at INFO, then discarded,
    so a batch run left no record of the model, the settings, or the tokens
    behind its own output -- and no way to compare two dumps except by rerunning
    them. One small JSON per document closes that.
    """
    llm_config = config.tool_config.llm_config
    tool_config = config.get_tool_config()
    facts_validation = tool_config.facts_validation
    # The deprecated FACTS_LLM_REPAIR_VISITS names the same budget, so a run
    # configured the old way must not be recorded as having run no passes.
    facts_critic_passes = (
        facts_validation.llm_repair_visits
        if facts_validation.llm_repair_visits is not None
        else facts_validation.critic_passes
    )
    # The .facts.ttl dump strips provenance; count what the file will actually
    # hold, or the manifest is not comparable to its own TTL (1711 vs 557 on
    # observed runs).
    serialized_facts = (
        TripleStoreManager.strip_provenance(state.aggregated_facts)
        if state.aggregated_facts is not None
        else None
    )
    manifest = RunManifest(
        source=file_path.name,
        line_number=line_number,
        ontocast_version=__version__,
        render_mode=str(state.render_mode),
        loops=RunManifestLoops(
            max_visits=state.max_visits,
            max_critic_visits=config.server.max_critic_visits_per_node,
            facts_critic_passes=facts_critic_passes,
            ontology_critic_passes=tool_config.ontology_validation.critic_passes,
        ),
        critic=summarize_loop(state.facts_loop_telemetry),
        ontology_critic=summarize_loop(state.ontology_loop_telemetry),
        ontology_reduce_metrics=dict(state.ontology_reduce_metrics),
        selection=_selection_manifest(state, config),
        validation_config=RunManifestValidationConfig(
            context_from_units=facts_validation.context_from_units,
            json_mode=llm_config.json_mode,
            shapes_prompt_contract=facts_validation.shapes_prompt_contract,
            shapes_prompt_selection=shapes_prompt_selection,
            shapes_triples=shapes_triples,
            shacl_inference=str(facts_validation.shacl_inference),
            numeric_coverage_mandatory=facts_validation.numeric_coverage_mandatory,
            facts_user_instruction_chars=len(state.facts_user_instruction or ""),
        ),
        graph_metrics=(
            facts_graph_shape_metrics(
                serialized_facts,
                [DEFAULT_IRI, str(state.doc_iri), state.doc_namespace or ""],
            )
            if serialized_facts is not None
            else None
        ),
        current_domain=state.current_domain,
        doc_iri=str(state.doc_iri) if state.doc_hid else None,
        tenant=state.tenant,
        project=state.project,
        llm=RunManifestLLM(
            provider=str(llm_config.provider),
            model_name=str(llm_config.model_name),
            temperature=llm_config.temperature,
            think=llm_config.think,
            num_ctx=llm_config.num_ctx,
            num_predict=llm_config.num_predict,
        ),
        budget=state.budget_tracker,
        ontology_triples=sum(
            len(artifact.graph) for artifact in _ontology_artifacts_for_dump(state)
        ),
        facts_triples=(
            len(state.aggregated_facts) if state.aggregated_facts is not None else 0
        ),
        facts_triples_serialized=(
            len(serialized_facts) if serialized_facts is not None else 0
        ),
        retrieval_metrics=dict(state.retrieval_metrics),
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    facts_path = facts_ttl_output_path(
        file_path, line_number=line_number, output_dir=output_dir
    )
    output_path = facts_path.with_name(
        f"{facts_path.name.removesuffix('.facts.ttl')}.run.json"
    )
    output_path.write_text(
        manifest.model_dump_json(indent=2, exclude_none=True), encoding="utf-8"
    )
    logger.info("Dumped run manifest to %s", output_path)
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
    exclude_sections: list[str] | None = None,
    summarize_sections: list[str] | None = None,
    summary_max_sentences: int = 5,
    document_type_hint: str | None = None,
    section_schema_id: str | None = None,
    max_visits: int | None = None,
    document_metadata: dict[str, object] | None = None,
    facts_user_instruction: str = "",
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
        "exclude_sections": exclude_sections,
        "summarize_sections": summarize_sections,
        "summary_max_sentences": summary_max_sentences,
        "document_type_hint": document_type_hint,
        "section_schema_id": section_schema_id,
        "facts_user_instruction": facts_user_instruction,
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
        validate_unit_pipeline_facts(state, ontology_graph, tools)
    await asyncio.to_thread(serialize_agent_state, state, tools)


def validate_unit_pipeline_facts(
    state: AgentState,
    ontology_graph: RDFGraph,
    tools: ToolBox,
) -> None:
    """Run the post-aggregation invariant gate for the single-unit path.

    The document graph reaches this gate at VALIDATE_FACTS; the unit pipeline
    does not run the graph, so both single-unit callers -- the CLI
    ``--use-unit-pipeline`` batch path and the ``/process_unit`` route -- invoke
    it here after aggregation. Without it they would ship facts with no
    functional-violation, coreference, or SHACL check at all.

    ``merge_repair=False``: un-merging re-aggregates *retained units against
    each other*, which has no meaning for a single unit. Everything else is the
    document path verbatim, so batch dumps stay comparable across the two entry
    paths.
    """
    run_facts_gate(state, ontology_graph, tools, merge_repair=False)


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
    gate_repairs = workflow_state.get("facts_gate_repairs")
    if gate_repairs is not None:
        state.facts_gate_repairs = list(gate_repairs)
    conformance = workflow_state.get("facts_conformance")
    if conformance:
        state.facts_conformance = dict(conformance)
    metrics = workflow_state.get("retrieval_metrics")
    if metrics:
        state.retrieval_metrics = dict(metrics)
    # The manifest's critic blocks read these; leaving them off this copy list
    # is why case10's manifests reported `critic: {calls: 0}` while their own
    # retrieval_metrics recorded 20 facts-critic and 26 ontology-critic calls.
    facts_telemetry = workflow_state.get("facts_loop_telemetry")
    if facts_telemetry:
        state.facts_loop_telemetry = dict(facts_telemetry)
    ontology_telemetry = workflow_state.get("ontology_loop_telemetry")
    if ontology_telemetry:
        state.ontology_loop_telemetry = dict(ontology_telemetry)
    reduce_metrics = workflow_state.get("ontology_reduce_metrics")
    if reduce_metrics:
        state.ontology_reduce_metrics = dict(reduce_metrics)
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
    exclude_sections: list[str] | None = None,
    summarize_sections: list[str] | None = None,
    summary_max_sentences: int = 5,
    document_type_hint: str | None = None,
    section_schema_id: str | None = None,
    max_visits: int | None = None,
    document_metadata: dict[str, object] | None = None,
    facts_user_instruction: str = "",
    output_dir: pathlib.Path | None = None,
    facts_output_dir: pathlib.Path | None = None,
    ontology_output_dir: pathlib.Path | None = None,
    strip_provenance: bool = True,
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
                exclude_sections=exclude_sections,
                summarize_sections=summarize_sections,
                summary_max_sentences=summary_max_sentences,
                document_type_hint=document_type_hint,
                section_schema_id=section_schema_id,
                max_visits=resolved_max_visits,
                document_metadata=document_metadata,
                facts_user_instruction=facts_user_instruction,
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
                    try:
                        async for chunk in workflow.astream(
                            state,
                            stream_mode="values",
                            config=RunnableConfig(recursion_limit=recursion_limit),
                        ):
                            workflow_state = chunk
                    except SectionSelectionEmptyError as exc:
                        # Batch semantics: one unmatched selection must not kill
                        # the other files. cli/server.py turns a non-empty
                        # failed_files into a non-zero exit, so the `error` mode
                        # is scriptable without new CLI code.
                        logger.error("Error processing %s: %s", file_path, exc)
                        if file_path not in failed_files:
                            failed_files.append(file_path)
                        continue
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
                    strip_provenance=strip_provenance,
                )
                dump_ontology_ttls(
                    state,
                    file_path,
                    line_number=line_number,
                    output_dir=ontology_dir,
                )
                dump_validation_report(
                    state,
                    file_path,
                    line_number=line_number,
                    output_dir=facts_dir,
                )
                shapes_graph = tools.shapes_catalog.graph()
                _, _, selection_pending = tools.shapes_prompt_contract()
                dump_run_manifest(
                    state,
                    file_path,
                    config=config,
                    line_number=line_number,
                    output_dir=facts_dir,
                    shapes_triples=(
                        len(shapes_graph) if shapes_graph is not None else 0
                    ),
                    shapes_prompt_selection=selection_pending,
                )
        except Exception:
            logger.exception("Error processing %s", file_path)
            if file_path not in failed_files:
                failed_files.append(file_path)

    return failed_files
