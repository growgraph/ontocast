"""Resolve a document's home ontology: where its new vocabulary is minted.

One LLM call per document, before the ontology fan-out. The model proposes an
``ontology_id`` (plus title and description); code decides what that means: an
id already in the catalog makes the document extend that ontology, any other
id mints a new one under ``CURRENT_DOMAIN``. The model never writes an IRI.
"""

import logging

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from ontocast.agent.common import call_llm_with_retry
from ontocast.onto.enum import LLMPhase, Status
from ontocast.onto.identity import OntologyIdentity, mint_ontology_iri
from ontocast.onto.model import OntologyIdentityProposal
from ontocast.onto.state import AgentState
from ontocast.prompt.common import system_preamble_ontology, text_template
from ontocast.prompt.render_ontology import (
    home_ontology_catalog_clause,
    home_ontology_catalog_clause_empty,
    home_ontology_instruction,
)
from ontocast.stategraph.helpers import build_document_excerpt
from ontocast.tool.llm import record_active_count, use_budget_tracker
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)

_TEMPLATE = """
{preamble}

{instruction}

{user_instruction}

{text}

{format_instructions}
"""


def identity_for_existing(tools: ToolBox, ontology_id: str) -> OntologyIdentity | None:
    """The identity of the catalog ontology ``ontology_id`` names, if any.

    Raises:
        DivergedOntologyError: that ontology has several heads.
    """
    iri = tools.ontology_manager.resolve_ontology_ref(ontology_id)
    if iri is None:
        return None
    head = tools.ontology_manager.get_head_by_iri(iri)
    if head is None:
        return None
    return OntologyIdentity(
        ontology_id=head.ontology_id or ontology_id,
        iri=head.iri,
        title=head.title or head.ontology_id or ontology_id,
        description=head.description or "",
    )


def identity_for_new(
    tools: ToolBox, proposal: OntologyIdentityProposal, current_domain: str
) -> OntologyIdentity:
    """A new identity under ``current_domain``, refused if its IRI is taken."""
    iri = mint_ontology_iri(current_domain, proposal.ontology_id)
    if tools.ontology_manager.resolve_ontology_ref(iri) is not None:
        raise ValueError(
            f"ontology IRI {iri} is already in the catalog under another id"
        )
    return OntologyIdentity(
        ontology_id=proposal.ontology_id,
        iri=iri,
        title=proposal.title,
        description=proposal.description,
    )


async def propose_home_ontology(
    state: AgentState, tools: ToolBox
) -> OntologyIdentityProposal:
    """Ask the model for the document's home ontology id, title and description."""
    heads = tools.ontology_manager.ontologies
    if heads:
        catalog_clause = home_ontology_catalog_clause.format(
            catalog_list="\n".join(
                f"- {o.ontology_id}: {o.title or ''}".rstrip(": ") for o in heads
            )
        )
    else:
        catalog_clause = home_ontology_catalog_clause_empty
    parser = PydanticOutputParser(pydantic_object=OntologyIdentityProposal)
    prompt = PromptTemplate(
        template=_TEMPLATE,
        input_variables=[
            "preamble",
            "instruction",
            "user_instruction",
            "text",
            "format_instructions",
        ],
    )
    record_active_count("llm/ontology_identity")
    with use_budget_tracker(state.budget_tracker):
        llm_tool = await tools.get_llm_tool(state.budget_tracker)
        return await call_llm_with_retry(
            llm_tool=llm_tool,
            phase=LLMPhase.ONTOLOGY_IDENTITY,
            prompt=prompt,
            parser=parser,
            prompt_kwargs={
                "preamble": system_preamble_ontology,
                "instruction": home_ontology_instruction.format(
                    catalog_clause=catalog_clause
                ),
                "user_instruction": state.ontology_user_instruction,
                "text": text_template.format(text=build_document_excerpt(state)),
                "format_instructions": parser.get_format_instructions(),
            },
        )


async def resolve_home_ontology(state: AgentState, tools: ToolBox) -> AgentState:
    """Set ``state.home_ontology`` for a run that can create vocabulary.

    A facts-only run, or one with no text, gets none. An existing id resolves
    to that ontology's head; a new id is minted under ``state.current_domain``.

    Raises:
        DivergedOntologyError: the chosen catalog ontology has several heads.
        ValueError: the minted IRI is already in the catalog under another id.
    """
    state.home_ontology = None
    if not state.render_ontology or not build_document_excerpt(state).strip():
        return state
    proposal = await propose_home_ontology(state, tools)
    identity = identity_for_existing(tools, proposal.ontology_id)
    if identity is None:
        identity = identity_for_new(tools, proposal, state.current_domain)
        logger.info("Home ontology for this document: new %s", identity.iri)
    else:
        logger.info("Home ontology for this document: existing %s", identity.iri)
    state.home_ontology = identity
    if state.status != Status.FAILED:
        state.status = Status.SUCCESS
    return state
