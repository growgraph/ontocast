"""LLM pick of a single catalog ontology for a text excerpt (full-TTL per-unit path)."""

import logging

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from ontocast.agent.common import call_llm_with_retry
from ontocast.onto.model import create_ontology_selector_report_model
from ontocast.onto.null import NULL_ONTOLOGY
from ontocast.onto.ontology import Ontology
from ontocast.prompt.select_ontology import template_prompt
from ontocast.tool.llm import LLMTool
from ontocast.tool.ontology_manager import OntologyManager

logger = logging.getLogger(__name__)


async def select_catalog_ontology_for_excerpt(
    ontology_manager: OntologyManager,
    llm_tool: LLMTool,
    excerpt: str,
    ontology_selection_user_instruction: str = "",
) -> Ontology:
    """Use the LLM to select one catalog ontology, or :data:`NULL_ONTOLOGY` if none fit.

    The excerpt is usually a content unit's text. Empty excerpt or an empty
    catalog yields ``NULL_ONTOLOGY`` without calling the model.
    """
    text = excerpt.strip()
    if not text or not ontology_manager.has_ontologies:
        return NULL_ONTOLOGY

    ontologies = ontology_manager.ontologies
    num_ontologies = len(ontologies)
    if num_ontologies == 0:
        return NULL_ONTOLOGY

    lines: list[str] = []
    for i, o in enumerate(ontologies, start=1):
        lines.append(f"{i}. {o.describe()}")
    ontologies_list = "\n\n".join(lines)
    none_index = num_ontologies + 1

    model_cls = create_ontology_selector_report_model(num_ontologies)
    parser = PydanticOutputParser(pydantic_object=model_cls)
    prompt = PromptTemplate(
        template=template_prompt,
        input_variables=[
            "excerpt",
            "ontologies_list",
            "num_ontologies",
            "none_index",
            "ontology_selection_user_instruction",
            "format_instructions",
        ],
    )

    selector = await call_llm_with_retry(
        llm_tool=llm_tool,
        prompt=prompt,
        parser=parser,
        prompt_kwargs={
            "excerpt": text,
            "ontologies_list": ontologies_list,
            "num_ontologies": num_ontologies,
            "none_index": none_index,
            "ontology_selection_user_instruction": ontology_selection_user_instruction.strip(),
            "format_instructions": parser.get_format_instructions(),
        },
    )

    idx = selector.answer_index
    if idx == none_index:
        logger.debug("LLM selected: no suitable catalog ontology (none index)")
        return NULL_ONTOLOGY
    if 1 <= idx <= num_ontologies:
        return ontologies[idx - 1]
    logger.warning("Invalid answer_index %s from selector; using NULL_ONTOLOGY", idx)
    return NULL_ONTOLOGY
