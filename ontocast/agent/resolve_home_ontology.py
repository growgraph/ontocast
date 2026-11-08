"""Resolve a document's home ontology: where its new vocabulary is minted.

One LLM call per document, before the ontology fan-out. The model proposes an
``ontology_id`` (plus title and description); code decides what that means:

- an id already in the catalog makes the document extend that ontology;
- any other id mints a new one under ``CURRENT_DOMAIN``, unless the text names
  at least :data:`HOME_COVERAGE_MIN_TERMS` terms of an existing ontology: the
  proposal is then overridden to the best-covered one, so documents of one
  domain share a home instead of each opening its own;
- a home outside ``CURRENT_DOMAIN`` is external and is never written to: the
  document's terms go to its extension ontology, ``<id>-ext`` under
  ``CURRENT_DOMAIN``, which imports it.

The model never writes an IRI.
"""

import logging

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from ontocast.agent.common import call_llm_with_retry
from ontocast.onto.enum import LLMPhase, Status
from ontocast.onto.identity import (
    OntologyIdentity,
    extension_ontology_id,
    is_owned_ontology,
    mint_ontology_iri,
)
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
from ontocast.tool.ontology_validation.catalog_terms import catalog_coverage
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)

#: Catalog terms a text must name for its ontology to cover the document. A
#: handful of a domain's own terms in one text means the domain is there; a
#: smaller count is reached by common words alone.
HOME_COVERAGE_MIN_TERMS = 5

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
        namespace=head.namespace,
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


def identity_for_extension(
    tools: ToolBox, external: OntologyIdentity, current_domain: str
) -> OntologyIdentity:
    """The extension ontology of ``external``: the catalog's, else a new one."""
    ontology_id = extension_ontology_id(external.ontology_id)
    iri = mint_ontology_iri(current_domain, ontology_id)
    stored = identity_for_existing(tools, iri)
    if stored is not None:
        return stored.model_copy(update={"extends": external.iri})
    return OntologyIdentity(
        ontology_id=ontology_id,
        iri=iri,
        title=f"{external.title} (extension)"[:200],
        description=(
            f"Terms this deployment adds to {external.title} <{external.iri}>, "
            "which it imports."
        ),
        extends=external.iri,
    )


def covering_ontology(tools: ToolBox, text: str) -> OntologyIdentity | None:
    """The existing ontology the text covers best, if it names enough of its terms."""
    coverage = catalog_coverage(list(tools.ontology_manager.ontologies), text)
    if not coverage or coverage[0][0] < HOME_COVERAGE_MIN_TERMS:
        return None
    head = coverage[0][1]
    return identity_for_existing(tools, head.iri)


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
    to that ontology's head; a new id is minted under ``state.current_domain``
    unless an existing ontology covers the text; an external home is replaced
    by its extension ontology.

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
        identity = covering_ontology(tools, build_document_excerpt(state))
        if identity is not None:
            state.ontology_reduce_metrics["home_new_overridden"] = 1
            logger.info(
                "Home ontology: new %r proposed, but the text covers %s",
                proposal.ontology_id,
                identity.iri,
            )
    if identity is None:
        identity = identity_for_new(tools, proposal, state.current_domain)
        logger.info("Home ontology for this document: new %s", identity.iri)
    else:
        logger.info("Home ontology for this document: existing %s", identity.iri)
    if not is_owned_ontology(identity.iri, state.current_domain):
        identity = identity_for_extension(tools, identity, state.current_domain)
        state.ontology_reduce_metrics["home_extension"] = 1
        logger.info("External home: new terms go to its extension %s", identity.iri)
    state.home_ontology = identity
    if state.status != Status.FAILED:
        state.status = Status.SUCCESS
    return state
