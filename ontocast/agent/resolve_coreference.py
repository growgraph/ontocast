"""Coreference resolution across the units of one document: one LLM call.

Code proposes the pairs (:func:`ontocast.tool.agg.coreference.coreference_candidates`)
and shows each side as its unit stated it; the model returns a verdict per
pair. The caller turns verdicts into merge evidence and vetoes, which the
aggregator weighs against its guards.
"""

import logging
from collections.abc import Sequence

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from ontocast.agent.common import call_llm_with_retry
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import LLMPhase
from ontocast.onto.model import CoreferenceCandidate, CoreferenceReport
from ontocast.onto.state import AgentState
from ontocast.prompt.common import user_template
from ontocast.prompt.resolve_coreference import (
    instruction,
    pairs_template,
    preamble,
    template_prompt,
)
from ontocast.tool.agg.coreference import describe_entity
from ontocast.tool.llm import RUN_STOPPING_ERRORS, use_budget_tracker
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def render_pairs(
    candidates: Sequence[CoreferenceCandidate], units: Sequence[ContentUnit]
) -> str:
    """The numbered pairs, each side described from the unit that minted it."""
    blocks = [
        f"Pair {number}\n  A: {describe_entity(c.left, units)}\n"
        f"  B: {describe_entity(c.right, units)}"
        for number, c in enumerate(candidates, start=1)
    ]
    return pairs_template.format(pairs="\n\n".join(blocks))


async def judge_coreference(
    state: AgentState,
    tools: ToolBox,
    candidates: Sequence[CoreferenceCandidate],
) -> dict[int, str]:
    """Ask the model about ``candidates``; ``{pair number: verdict}``.

    Best effort: a failed call returns no verdicts and the aggregation stands.
    A request the provider will never accept stops the run instead.
    """
    if not candidates:
        return {}
    parser = PydanticOutputParser(pydantic_object=CoreferenceReport)
    prompt = PromptTemplate(
        template=template_prompt,
        input_variables=[
            "preamble",
            "instruction",
            "user_instruction",
            "pairs",
            "format_instructions",
        ],
    )
    user_instruction = (
        user_template.format(user_instruction=state.facts_user_instruction)
        if state.facts_user_instruction
        else ""
    )
    try:
        with use_budget_tracker(state.budget_tracker):
            llm_tool = await tools.get_llm_tool(state.budget_tracker)
            report: CoreferenceReport = await call_llm_with_retry(
                llm_tool=llm_tool,
                phase=LLMPhase.FACTS_COREFERENCE,
                prompt=prompt,
                parser=parser,
                prompt_kwargs={
                    "preamble": preamble,
                    "instruction": instruction,
                    "user_instruction": user_instruction,
                    "pairs": render_pairs(candidates, state.facts_units),
                    "format_instructions": parser.get_format_instructions(),
                },
            )
    except RUN_STOPPING_ERRORS:
        raise
    except Exception as exc:
        logger.warning("Coreference resolution call failed: %s", exc)
        return {}
    return {
        v.pair: v.verdict for v in report.verdicts if 1 <= v.pair <= len(candidates)
    }
