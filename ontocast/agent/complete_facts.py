"""Insert-only facts completion agent.

Runs after the facts render/critic loop, only when the numeric-coverage
inventory still lists measurements the render missed (see
:func:`ontocast.tool.facts_validation.unit_findings.unit_numeric_inventory`).
Mirrors :mod:`ontocast.agent.criticise_facts` in shape -- one LLM call parsed
into :class:`~ontocast.onto.model.TripleFix` fixes -- but proposes insertions
instead of judging what is already there.

The fixes returned here are not applied by this module: the unit loop
(``ontocast.stategraph.atomic``) compiles and applies them through
``compile_critic_fixes`` / ``_apply_patches``, the same per-subject
regression check a critic fix goes through, so a fix that leaves the unit
worse is rolled back exactly the way a bad critic fix would be.
"""

import logging

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from ontocast.agent.common import call_llm_with_retry
from ontocast.onto.model import FactsCompletionReport, TripleFix
from ontocast.onto.unit_states import UnitFactsState
from ontocast.prompt.common import text_template, user_template
from ontocast.prompt.complete_facts import (
    build_catalog_subjects_chapter,
    build_missing_measurements_chapter,
    build_term_sheet,
    completion_instruction,
    output_instruction_for,
    preamble,
    template_prompt,
)
from ontocast.prompt.graph_format import get_graph_format_profile
from ontocast.tool.atomic import AtomicToolBox
from ontocast.tool.facts_validation import (
    collect_catalog_terms,
    expand_vocabulary_terms,
)
from ontocast.util.numeric_inventory import NumericInventory

logger = logging.getLogger(__name__)


def _unit_role_property(atomic: AtomicToolBox, *graphs) -> set[str]:
    """The configured unit-role property, expanded to an IRI, or empty.

    ``atomic.quantity_fallback_vocabulary`` names the deployment's roles for
    bounded quantities (``value_class``, ``numeric_value``, ``unit``); only
    ``unit`` locates unit individuals, so the other roles are dropped before
    expansion rather than reused from ``expand_vocabulary_terms`` verbatim.
    """
    vocabulary = atomic.quantity_fallback_vocabulary or {}
    unit_term = vocabulary.get("unit")
    if not unit_term:
        return set()
    return expand_vocabulary_terms({"unit": unit_term}, *graphs)


def _build_prompt() -> PromptTemplate:
    return PromptTemplate(
        template=template_prompt,
        input_variables=[
            "preamble",
            "conformance_chapter",
            "term_sheet",
            "catalog_subjects_chapter",
            "completion_instruction",
            "user_instruction",
            "text_chapter",
            "missing_measurements_chapter",
            "output_instruction",
            "format_instructions",
        ],
    )


async def complete_facts(
    state: UnitFactsState,
    atomic: AtomicToolBox,
    inventory: NumericInventory,
) -> list[TripleFix]:
    """Propose insert-only fixes recovering measurements the render missed.

    Args:
        state: The unit's current facts state. Read-only here -- the unit
            loop applies whatever this returns.
        atomic: Toolbox for the LLM call and the unit's quantity vocabulary.
        inventory: The unit's missing-measurement inventory, already computed
            by the caller so the completion pass and the NUMERIC_COVERAGE
            finding agree on what is missing. An empty inventory short-
            circuits with no call.

    Returns:
        Proposed fixes, action ``ADD`` only. Anything else the model returns
        is dropped defensively rather than trusted -- this pass is
        insert-only by contract, not by the model's cooperation.
    """
    if not inventory.measurements:
        return []

    llm_tool = await atomic.get_llm_tool(state.budget_tracker)
    profile = get_graph_format_profile(
        state.llm_graph_format,
        ontology_chapter_format=state.ontology_chapter_format,
    )
    parser = PydanticOutputParser(pydantic_object=FactsCompletionReport)

    ontology_graph = state.ontology_snapshot.graph
    fact_graph = state.content_unit.graph

    unit_properties = _unit_role_property(atomic, ontology_graph, fact_graph)
    term_sheet = build_term_sheet(ontology_graph, unit_properties)
    catalog_subjects_chapter = build_catalog_subjects_chapter(
        fact_graph, collect_catalog_terms(ontology_graph)
    )
    missing_measurements_chapter = build_missing_measurements_chapter(inventory)

    user_instruction = (
        user_template.format(user_instruction=state.facts_user_instruction)
        if state.facts_user_instruction
        else ""
    )
    text_chapter = text_template.format(text=state.content_unit.extraction_text)

    prompt_data = {
        "preamble": preamble,
        "conformance_chapter": state.conformance_chapter,
        "term_sheet": term_sheet,
        "catalog_subjects_chapter": catalog_subjects_chapter,
        "completion_instruction": completion_instruction,
        "user_instruction": user_instruction,
        "text_chapter": text_chapter,
        "missing_measurements_chapter": missing_measurements_chapter,
        "output_instruction": output_instruction_for(profile.format),
        "format_instructions": profile.format_instructions(
            FactsCompletionReport, web_search_enabled=False
        ),
    }

    try:
        report: FactsCompletionReport = await call_llm_with_retry(
            llm_tool=llm_tool,
            prompt=_build_prompt(),
            parser=parser,
            prompt_kwargs=prompt_data,
            llm_graph_format=state.llm_graph_format,
        )
    except Exception as exc:
        # A failed completion call costs nothing beyond itself: the render
        # and critic loop already stand, accepted or not, and this pass only
        # ever adds to it. Propagating would fail the whole unit for a
        # best-effort improvement pass that never touched its graph.
        logger.warning("Facts completion pass call failed: %s", exc)
        return []

    fixes: list[TripleFix] = []
    for fix in report.proposed_fixes:
        if fix.action != "ADD":
            logger.warning(
                "Completion pass proposed a %s fix; dropping it -- this pass "
                "is insert-only and the model's action choice is not trusted",
                fix.action,
            )
            continue
        fixes.append(fix)
    return fixes
