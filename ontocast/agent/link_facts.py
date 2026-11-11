"""The link pass's model call: links the text states for isolated instances.

Code chooses what the model sees
(:func:`ontocast.tool.facts_validation.linking.select_link_candidates`) and
checks everything it proposes
(:func:`ontocast.tool.facts_validation.linking.validate_link_proposals`).
"""

import logging

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from rdflib import URIRef

from ontocast.agent.common import call_llm_with_retry
from ontocast.onto.enum import LLMPhase
from ontocast.onto.model import FactsLinkProposal, FactsLinkReport
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.prompt.common import user_template
from ontocast.prompt.link_facts import (
    entities_template,
    instruction,
    preamble,
    properties_template,
    template_prompt,
)
from ontocast.prompt.term_sheet import qname_for, term_line
from ontocast.tool.agg.coreference import describe_node
from ontocast.tool.facts_validation.linking import LinkCandidates
from ontocast.tool.llm import RUN_STOPPING_ERRORS, use_budget_tracker
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def render_link_entities(candidates: LinkCandidates, graph: RDFGraph) -> str:
    """The numbered entities: each isolated one in full with its passage, then targets."""
    blocks: list[str] = []
    number = 0
    for iri in candidates.isolated:
        number += 1
        parts = describe_node(graph, URIRef(iri))
        parts.append(f'passage: "{candidates.passages[iri]}"')
        blocks.append(f"Entity {number} (ISOLATED)\n    " + "\n    ".join(parts))
    for iri in candidates.targets:
        number += 1
        parts = [
            part
            for part in describe_node(graph, URIRef(iri))
            if part.startswith(("labels:", "types:"))
        ]
        blocks.append(f"Entity {number}: " + "; ".join(parts))
    return entities_template.format(entities="\n\n".join(blocks))


def render_link_properties(candidates: LinkCandidates, ontology_graph: RDFGraph) -> str:
    """The properties as listed: name, label, ``domain -> range``.

    A property the context does not describe is listed by its name alone.
    """
    lines: list[str] = []
    for name, iri in candidates.properties.items():
        node = URIRef(iri)
        if (node, None, None) in ontology_graph:
            described = term_line(ontology_graph, node, kind="property")
            # Lead with the listed name, which the answer must repeat.
            first, *rest = described
            shown = qname_for(ontology_graph, node)
            lines.append("  " + name + first.strip()[len(shown) :])
            lines.extend(rest)
        else:
            lines.append(f"  {name}")
    return properties_template.format(properties="\n".join(lines))


async def propose_links(
    state: AgentState,
    tools: ToolBox,
    candidates: LinkCandidates,
    graph: RDFGraph,
    ontology_graph: RDFGraph,
) -> list[FactsLinkProposal]:
    """Ask the model for links the passages state.

    Best effort: a failed call proposes nothing and the graph stands. A
    request the provider will never accept stops the run instead.
    """
    parser = PydanticOutputParser(pydantic_object=FactsLinkReport)
    prompt = PromptTemplate(
        template=template_prompt,
        input_variables=[
            "preamble",
            "instruction",
            "user_instruction",
            "entities",
            "properties",
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
            report: FactsLinkReport = await call_llm_with_retry(
                llm_tool=llm_tool,
                phase=LLMPhase.FACTS_LINK,
                prompt=prompt,
                parser=parser,
                prompt_kwargs={
                    "preamble": preamble,
                    "instruction": instruction,
                    "user_instruction": user_instruction,
                    "entities": render_link_entities(candidates, graph),
                    "properties": render_link_properties(candidates, ontology_graph),
                    "format_instructions": parser.get_format_instructions(),
                },
            )
    except RUN_STOPPING_ERRORS:
        raise
    except Exception as exc:
        logger.warning("Link pass call failed: %s", exc)
        return []
    return list(report.links)
