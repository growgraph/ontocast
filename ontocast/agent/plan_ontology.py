"""Plan a document's ontology before its units render.

One LLM call per document sees the whole text (within a budget) and the
catalog's terms, and proposes the classes and properties the document adds to
its domain. The proposals are checked against the whole catalog without a
model (:func:`~ontocast.tool.ontology_validation.plan.check_plan`), and the
accepted plan is handed to every unit: units mint from it, and anything they
mint outside it is decided at reduce.
"""

from __future__ import annotations

import logging

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from ontocast.agent.common import call_llm_with_retry
from ontocast.onto.enum import LLMPhase, Status
from ontocast.onto.identity import OntologyIdentity
from ontocast.onto.model import OntologyPlan, OntologyPlanReport
from ontocast.onto.state import AgentState
from ontocast.prompt.common import system_preamble_ontology, text_template
from ontocast.prompt.plan_ontology import (
    catalog_empty,
    catalog_intro,
    plan_instruction,
    template_prompt,
)
from ontocast.tool.llm import (
    RUN_STOPPING_ERRORS,
    record_active_count,
    use_budget_tracker,
)
from ontocast.tool.ontology_validation.catalog_terms import (
    catalog_prefixes,
    catalog_terms,
    format_term_sheet,
)
from ontocast.tool.ontology_validation.plan import check_plan
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


async def attach_neighbours(
    plan: OntologyPlan,
    tools: ToolBox,
    labels: dict[str, str],
    top_k: int,
) -> None:
    """Add the vector index's nearest catalog terms to each minted plan term.

    Skipped when no vector store is ready. A failed lookup leaves the plan
    unchanged: neighbours are a hint, not a check.
    """
    store = tools.vector_store
    minted = plan.minted()
    if store is None or not tools.vector_store_ready or not minted or top_k <= 0:
        return
    queries = [f"{term.label}: {term.definition}".strip(": ") for term in minted]
    try:
        results = await store.asearch_patch_hits_many(queries, top_k=top_k)
    except Exception as exc:  # noqa: BLE001 - a hint, never fatal
        logger.warning("Plan neighbour lookup failed: %s", exc)
        return
    for term, by_channel in zip(minted, results, strict=False):
        found: list[tuple[str, str]] = []
        for hit in [*by_channel.core_hits, *by_channel.bm25_hits]:
            iri = hit.atom.iri
            if iri == term.iri or any(iri == seen for seen, _ in found):
                continue
            found.append((iri, labels.get(iri, "")))
            if len(found) >= top_k:
                break
        term.nearest = found


async def propose_plan(
    state: AgentState,
    tools: ToolBox,
    *,
    home: OntologyIdentity,
    text: str,
    term_sheet: str,
    max_terms: int,
) -> OntologyPlanReport:
    """One structured LLM call: the document's term proposals."""
    parser = PydanticOutputParser(pydantic_object=OntologyPlanReport)
    prompt = PromptTemplate(
        template=template_prompt,
        input_variables=[
            "preamble",
            "instruction",
            "catalog",
            "user_instruction",
            "text",
            "format_instructions",
        ],
    )
    record_active_count("llm/ontology_plan")
    with use_budget_tracker(state.budget_tracker):
        llm_tool = await tools.get_llm_tool(state.budget_tracker)
        return await call_llm_with_retry(
            llm_tool=llm_tool,
            phase=LLMPhase.ONTOLOGY_PLAN,
            prompt=prompt,
            parser=parser,
            prompt_kwargs={
                "preamble": system_preamble_ontology,
                "instruction": plan_instruction.format(
                    ontology_id=home.ontology_id,
                    title=home.title,
                    description=home.description,
                    max_terms=max_terms,
                ),
                "catalog": (
                    catalog_intro.format(term_sheet=term_sheet)
                    if term_sheet
                    else catalog_empty
                ),
                "user_instruction": state.ontology_user_instruction,
                "text": text_template.format(text=text),
                "format_instructions": parser.get_format_instructions(),
            },
        )


async def plan_ontology(state: AgentState, tools: ToolBox) -> AgentState:
    """Set ``state.ontology_plan`` for a run that renders an ontology.

    No plan (``None``) when planning is off, the run renders no ontology, the
    document has no home ontology or no text, or the call fails: units then
    mint as they would without one.
    """
    state.ontology_plan = None
    config = tools.config.get_tool_config().ontology_validation
    home = state.home_ontology
    if not config.plan_enabled or not state.render_ontology or home is None:
        return state
    # Imported here: the stategraph package imports this module.
    from ontocast.stategraph.helpers import build_document_text

    text = build_document_text(state, config.plan_max_chars).strip()
    if not text:
        return state

    heads = list(tools.ontology_manager.ontologies)
    prefixes = catalog_prefixes(heads, home, tools.prefix_table())
    terms = catalog_terms(heads, first_iri=home.iri)
    term_sheet = format_term_sheet(terms, prefixes, config.plan_catalog_max_terms)
    try:
        report = await propose_plan(
            state,
            tools,
            home=home,
            text=text,
            term_sheet=term_sheet,
            max_terms=config.plan_max_terms,
        )
    except RUN_STOPPING_ERRORS:
        raise
    except Exception as exc:  # noqa: BLE001 - units render without a plan
        logger.warning("Ontology plan failed; units render without one: %s", exc)
        state.ontology_reduce_metrics["plan_failed"] = 1
        return state

    plan = check_plan(
        report,
        home=home,
        catalog_index=tools.ontology_manager.catalog_surface_index(),
        prefixes=prefixes,
        max_terms=config.plan_max_terms,
    )
    if not plan.terms:
        # An empty plan would hold back every term the units mint; a model
        # that returned nothing has not planned, so units render without one.
        state.ontology_reduce_metrics["plan_empty"] = 1
        return state
    plan.prefixes = prefixes
    await attach_neighbours(
        plan, tools, {term.iri: term.label for term in terms}, config.plan_neighbours
    )
    state.ontology_plan = plan
    state.ontology_reduce_metrics.update(
        {
            "plan_proposed": len(report.proposals),
            "plan_minted": len(plan.minted()),
            "plan_reused": len(plan.reused()),
            "plan_resolved_to_catalog": plan.resolved_to_catalog,
            "plan_rejected": len(plan.rejected),
        }
    )
    logger.info(
        "Ontology plan: %d to mint, %d to reuse, %d refused",
        len(plan.minted()),
        len(plan.reused()),
        len(plan.rejected),
    )
    if state.status != Status.FAILED:
        state.status = Status.SUCCESS
    return state
