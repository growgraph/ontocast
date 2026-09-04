"""Fact rendering agent for OntoCast.

This module provides functionality for rendering facts from RDF graphs into
human-readable formats, making the extracted knowledge more accessible and
understandable.
"""

import logging
import time
from collections.abc import Sequence

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from ontocast.agent.common import call_llm_with_retry
from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.enum import FailureStage, Status, WorkflowNode
from ontocast.onto.model import (
    FactsRenderReport,
    GraphRepairRecord,
)
from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_access import (
    UnitFactsOntologyAccess,
    build_llm_prefix_map,
    ontology_access_for_unit_facts,
)
from ontocast.onto.ontology_condense import CondenseReport
from ontocast.onto.rdfgraph import (
    RDFGraph,
    finalize_llm_graph,
)
from ontocast.onto.state import BudgetTracker
from ontocast.onto.unit_states import UnitFactsState
from ontocast.prompt.common import text_template, user_template
from ontocast.prompt.graph_format import get_graph_format_profile
from ontocast.prompt.ontology_context import (
    build_ontology_index,
    format_ontologies_clause,
)
from ontocast.prompt.render_facts import (
    build_citation_metadata_instruction,
    preamble,
    template_prompt,
)
from ontocast.prompt.web_grounding import persist_search_request, search_guidelines_for
from ontocast.tool.atomic import AtomicToolBox
from ontocast.tool.facts_validation import (
    expand_vocabulary_terms,
    normalize_literals_against_schema,
    promote_degenerate_bounds_from_vocabulary,
    repair_literal_type_objects,
    repair_property_aliases,
    resolve_code_literals,
)
from ontocast.tool.llm import LLMConfigurationError
from ontocast.tool.validate import partition_object_property_literal_triples

logger = logging.getLogger(__name__)


def _normalize_and_repair_graph(
    graph: RDFGraph,
    ontology_context_graph: RDFGraph,
    tools: AtomicToolBox,
    *,
    budget_tracker: BudgetTracker | None = None,
) -> tuple[RDFGraph, list[GraphRepairRecord]]:
    """Apply LLM-free parse-time fixes to a rendered graph in place.

    Retypes untyped numeric literals against declared numeric ranges, coerces
    literal ``rdf:type`` objects into IRIs, rewrites unambiguous near-miss
    predicates in catalog namespaces (e.g. ``qudt:value`` ->
    ``qudt:numericValue``), and links nodes to the catalog individual whose
    code they carry (``qudt:ucumCode "d"`` -> ``qudt:unit unit:DAY``).
    Ambiguous near-misses and unresolvable type literals are left for findings
    collection. The alias rewrite checks membership against the scope's *full*
    catalog, not only the unit's snapshot: a predicate the catalog declares
    but retrieval did not bring into this unit's context is a real term and
    is never rewritten toward a look-alike the snapshot happens to carry.

    Args:
        graph: Rendered facts graph, repaired in place.
        ontology_context_graph: Read-only schema the repairs are checked against.
        tools: Supplies the alias tie-break floor, the full-catalog term set,
            code predicates, and the quantity fallback vocabulary (alias
            exemptions and the degenerate-bound promotion roles).
        budget_tracker: Charged ``"repair/deterministic"``. Both scans here walk
            the whole ontology graph per call, so this is timed to show how much
            of it is per-unit-invariant work.

    Returns:
        Tuple of (repaired graph, applied-repair records for provenance).
    """
    started = time.perf_counter()
    retyped = normalize_literals_against_schema(graph, ontology_context_graph)
    type_repaired, _type_findings, type_records = repair_literal_type_objects(graph)
    rewritten, _alias_findings, alias_records = repair_property_aliases(
        graph,
        ontology_context_graph,
        min_ratio=tools.property_alias_min_ratio,
        exempt_terms=expand_vocabulary_terms(
            tools.quantity_fallback_vocabulary, graph, ontology_context_graph
        ),
        full_catalog_terms=tools.catalog_terms(),
    )
    resolved, code_records = resolve_code_literals(
        graph, ontology_context_graph, tools.code_predicates
    )
    promote_degenerate_bounds_from_vocabulary(
        graph, ontology_context_graph, tools.quantity_fallback_vocabulary
    )
    if budget_tracker is not None:
        budget_tracker.add_duration(
            "repair/deterministic", time.perf_counter() - started
        )
    if retyped or rewritten or type_repaired or resolved:
        logger.info(
            "Deterministic graph repair: retyped %d literal(s), coerced %d "
            "rdf:type literal(s), rewrote %d alias triple(s), resolved %d code(s)",
            retyped,
            type_repaired,
            rewritten,
            resolved,
        )
    return graph, [*type_records, *alias_records, *code_records]


async def render_facts(
    state: UnitFactsState,
    tools: AtomicToolBox,
    supplemental_ontologies: Sequence[Ontology] | None = None,
) -> UnitFactsState:
    """Extract a unit's facts from its text.

    There is one renderer because there is one render. The loop retries this
    only when it *fails*; a unit that rendered successfully is improved by the
    critic passes, which apply a compiled patch rather than asking for the graph
    to be written again. The update-mode renderer this used to dispatch to had
    become unreachable: it keyed on the unit graph being non-empty, and nothing
    populates that except a successful render, after which the render loop is
    already done.

    The ontology renderer still has both modes, and for a reason that does not
    apply here: it keys on the retrieved *snapshot*, a different field from the
    one it writes, so its update mode is the normal path whenever a catalog
    exists.

    Args:
        state: The current unit facts state.
        tools: The toolbox containing necessary tools.
        supplemental_ontologies: Extra ontologies for prefix resolution.

    Returns:
        UnitFactsState: Updated state with rendered facts.
    """
    logger.info(f"Render facts for {state.get_content_unit_progress_string()}")
    return await render_facts_fresh(
        state, tools, supplemental_ontologies=list(supplemental_ontologies or ())
    )


def _record_chapter_report(
    report: CondenseReport, budget_tracker: BudgetTracker | None
) -> None:
    """Publish what condensing and capping did to the ontology chapter.

    Fires once per chapter actually built, not once per call that reads it: the
    snapshot is shared across the fan-out and the chapter is memoised on it, so
    these describe the chapter, not the traffic.

    Sizing a text cap is not something a default can do in advance -- whether a
    catalog reaches one is a property of that catalog. So the counters carry the
    before/after even when nothing was clipped, which is the case that tells a
    deployment its caps are inert.
    """
    if budget_tracker is None or not report.text_chars_before:
        return
    budget_tracker.incr("chapter/text_chars_before", report.text_chars_before)
    budget_tracker.incr("chapter/text_chars_after", report.text_chars_after)
    budget_tracker.incr("chapter/literals_clipped", report.literals_clipped)
    budget_tracker.incr("chapter/literals_dropped", report.literals_dropped)
    if report.text_over_budget:
        budget_tracker.incr("chapter/text_over_budget")


def _prepare_prompt_data(
    state: UnitFactsState,
    access: UnitFactsOntologyAccess,
    profile,
    *,
    citation_vocabulary: dict[str, str] | None = None,
    quantity_fallback_vocabulary: dict[str, str] | None = None,
    search_guidelines: str = "",
) -> dict[str, str]:
    """Prepare common prompt data for both fresh and update rendering.

    Args:
        state: The current unit facts state
        access: Read-only ontology context (facts prompts use snapshot only).
        profile: Active graph format profile.

    Returns:
        Dictionary containing formatted prompt components
    """
    ctx = access.effective_ontology_for_prompt()
    # Normalise into a local graph rather than writing back onto ``ctx``: the
    # snapshot is shared by reference across every unit in the fan-out, so a
    # mutation here would leak into siblings mid-flight.
    ontology_graph = ctx.graph
    if not isinstance(ontology_graph, RDFGraph):
        normalized_graph = RDFGraph()
        for triple in ontology_graph:
            normalized_graph.add(triple)
        for prefix, namespace_uri in ontology_graph.namespaces():
            normalized_graph.bind(prefix, namespace_uri)
        ontology_graph = normalized_graph
    domain_pairs = access.domain_prefix_pairs()
    chapter_start = time.perf_counter()
    if ontology_graph is ctx.graph:
        # Memoised on the snapshot, which the whole fan-out shares: serialising
        # the same ontology once per unit dominated facts prompt construction.
        ontology_chapter = ctx.prompt_chapter(
            profile,
            max_triples=state.ontology_context_max_triples,
            text_caps=state.ontology_text_caps,
            on_report=lambda report: _record_chapter_report(
                report, state.budget_tracker
            ),
        )
    else:
        ontology_chapter = profile.format_ontology_chapter(
            ontology_graph,
            suffix=(
                ""
                if profile.renders_term_sheet
                else build_ontology_index(ontology_graph)
            ),
            max_triples=state.ontology_context_max_triples,
            text_caps=state.ontology_text_caps,
            on_report=lambda report: _record_chapter_report(
                report, state.budget_tracker
            ),
        )
    state.budget_tracker.add_duration(
        "prompt/ontology_chapter", time.perf_counter() - chapter_start
    )

    facts_instruction_str = profile.facts_operational_guidelines(
        facts_namespace=DEFAULT_IRI,
        domain_ontologies_clause=format_ontologies_clause(domain_pairs),
        quantity_fallback_vocabulary=quantity_fallback_vocabulary,
        search_guidelines=search_guidelines,
    )

    text_chapter = text_template.format(text=state.content_unit.extraction_text)

    fact_chapter = ""

    user_instruction = (
        user_template.format(user_instruction=state.facts_user_instruction)
        if state.facts_user_instruction
        else ""
    )
    if state.content_unit.is_citation_metadata:
        user_instruction = (
            build_citation_metadata_instruction(citation_vocabulary or {})
            + user_instruction
        )

    return {
        "ontology_chapter": ontology_chapter,
        # Rendered once per tenancy from the shapes partition; "" for
        # shape-less deployments, keeping their prompt byte-identical.
        "conformance_chapter": state.conformance_chapter,
        "user_instruction": user_instruction,
        "facts_instruction": facts_instruction_str,
        "text_chapter": text_chapter,
        "fact_chapter": fact_chapter,
    }


def _create_prompt_template() -> PromptTemplate:
    """Create the common prompt template used by both rendering functions.

    Returns:
        Configured PromptTemplate instance
    """
    return PromptTemplate(
        template=template_prompt,
        input_variables=[
            "preamble",
            "facts_instruction",
            "user_instruction",
            "ontology_chapter",
            "conformance_chapter",
            "text_chapter",
            "improvement_instruction",
            "output_instruction",
            "format_instructions",
        ],
    )


def _handle_rendering_error(
    state: UnitFactsState, error: Exception, stage: FailureStage
) -> UnitFactsState:
    """Handle rendering errors consistently.

    Args:
        state: The current agent state
        error: The exception that occurred
        stage: The failure stage to set

    Returns:
        Updated state with failure information
    """
    logger.error(f"Failed to generate triples: {str(error)}")
    state.set_failure(stage, str(error))
    state.set_node_status(WorkflowNode.TEXT_TO_FACTS, Status.FAILED)
    return state


async def render_facts_fresh(
    state: UnitFactsState,
    tools: AtomicToolBox,
    supplemental_ontologies: Sequence[Ontology] | None = None,
) -> UnitFactsState:
    """Render fresh facts from the current chunk into Turtle format.

    Args:
        state: The current unit facts state containing the chunk to render.
        tools: The toolbox instance providing utility functions.

    Returns:
        UnitFactsState: Updated state with rendered facts.
    """
    logger.info("Rendering fresh facts")
    state.quarantined_literal_triples = []
    llm_tool = await tools.get_llm_tool(state.budget_tracker)
    profile = get_graph_format_profile(
        state.llm_graph_format,
        ontology_chapter_format=state.ontology_chapter_format,
    )
    parser = PydanticOutputParser(pydantic_object=FactsRenderReport)

    access = ontology_access_for_unit_facts(state)

    known_prefixes = build_llm_prefix_map(
        access.ontology_for_prefixes(),
        supplemental_ontologies or (),
    )

    web_search_enabled = tools.web_grounding_enabled_for_node(
        WorkflowNode.TEXT_TO_FACTS
    )
    prompt_data = _prepare_prompt_data(
        state,
        access,
        profile,
        citation_vocabulary=tools.citation_vocabulary,
        quantity_fallback_vocabulary=tools.quantity_fallback_vocabulary,
        search_guidelines=search_guidelines_for(
            WorkflowNode.TEXT_TO_FACTS, web_search_enabled
        ),
    )
    prompt_data_fresh = {
        "preamble": preamble,
        "improvement_instruction": "",
        "output_instruction": profile.render_fresh_output_instruction(target="facts"),
    }
    prompt_data.update(prompt_data_fresh)

    prompt = _create_prompt_template()

    try:
        # Set known prefixes in context before parsing
        RDFGraph.set_known_prefixes(known_prefixes if known_prefixes else None)

        render_report: FactsRenderReport = await call_llm_with_retry(
            llm_tool=llm_tool,
            prompt=prompt,
            parser=parser,
            prompt_kwargs={
                "format_instructions": profile.format_instructions(
                    FactsRenderReport,
                    web_search_enabled=web_search_enabled,
                ),
                **prompt_data,
            },
            llm_graph_format=state.llm_graph_format,
        )
        persist_search_request(
            state,
            WorkflowNode.TEXT_TO_FACTS,
            render_report.external_evidence_request,
            web_search_enabled,
        )
        render_report.semantic_graph.sanitize_prefixes_namespaces()
        clean_graph, rejected = finalize_llm_graph(render_report.semantic_graph)
        ontology_context_graph = access.effective_ontology_for_prompt().graph
        clean_graph, repair_records = _normalize_and_repair_graph(
            clean_graph,
            ontology_context_graph,
            tools,
            budget_tracker=state.budget_tracker,
        )
        state.applied_repairs.extend(repair_records)
        if tools.object_property_literal_check:
            clean_graph, op_rejected = partition_object_property_literal_triples(
                clean_graph, ontology_context_graph
            )
            rejected = rejected + op_rejected
        state.content_unit.graph = clean_graph
        state.quarantined_literal_triples = rejected
        if rejected:
            logger.warning(
                "Fresh facts quarantined %d triple(s) with invalid literals",
                len(rejected),
            )

        # Track triples in budget tracker (fresh facts)
        num_triples = len(clean_graph)
        logger.info(f"Fresh facts generated with {num_triples} triple(s).")
        state.budget_tracker.add_facts_update(num_operations=1, num_triples=num_triples)

        state.clear_failure()
        state.set_node_status(WorkflowNode.TEXT_TO_FACTS, Status.SUCCESS)
        return state

    except LLMConfigurationError:
        # The provider rejects the request itself, not this attempt at
        # it: every other unit is about to be rejected identically.
        # Failing one unit here turns a configuration fault into an
        # empty run that reports success.
        raise
    except Exception as e:
        return _handle_rendering_error(state, e, FailureStage.GENERATE_TTL_FOR_FACTS)
    finally:
        # Clear the context after parsing
        RDFGraph.set_known_prefixes(None)
