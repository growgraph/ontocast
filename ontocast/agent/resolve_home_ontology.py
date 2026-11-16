"""Resolve a document's home ontology: where its new vocabulary is minted.

One LLM call per document, before the ontology fan-out. The catalog is listed
under each ontology's shown prefix (unique, unlike ids, which may repeat), with
its title and the start of its description. The
model proposes an ``ontology_id`` (plus title and description); code decides
what that means:

- a shown prefix (or an id one ontology alone holds) makes the document extend
  that ontology;
- any other id mints a new one under ``CURRENT_DOMAIN``, unless the text names
  at least :data:`HOME_COVERAGE_MIN_TERMS` terms of an existing ontology: the
  proposal is then overridden to the best-covered one, so documents of one
  domain share a home instead of each opening its own;
- a home outside ``CURRENT_DOMAIN`` is external and is never written to: the
  document's terms go to its extension ontology, ``<id>-ext-<hash>`` under
  ``CURRENT_DOMAIN`` (the hash is the external IRI's), which imports it.

The model never writes an IRI.
"""

import logging

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from ontocast.agent.common import call_llm_with_retry
from ontocast.onto.enum import LLMPhase, Status
from ontocast.onto.identity import (
    OntologyIdentity,
    extension_ontology_iri,
    is_owned_ontology,
    mint_ontology_iri,
)
from ontocast.onto.model import OntologyIdentityProposal
from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_header import AmbiguousOntologyRefError
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


def catalog_handles(tools: ToolBox) -> dict[str, str]:
    """Shown prefix -> IRI of each catalog ontology: the handles the model sees."""
    table = tools.prefix_table()
    handles: dict[str, str] = {}
    for head in tools.ontology_manager.ontologies:
        handle = table.prefix_for(head.namespace) if head.namespace else None
        handles[handle or head.iri] = head.iri
    return handles


def identity_for_handle(tools: ToolBox, handle: str) -> OntologyIdentity | None:
    """The identity of the catalog ontology ``handle`` names, if any.

    ``handle`` is a prefix the catalog's prefix table maps to an ontology's
    namespace (a shown prefix, or an alias only one namespace has), else an
    id or prefix one held ontology alone has (a diverged one, which the table
    leaves out, included). A name several ontologies share names none.

    Raises:
        DivergedOntologyError: that ontology has several heads.
    """
    handle = handle.strip()
    namespace = tools.prefix_table().namespace_for(handle)
    for head in tools.ontology_manager.ontologies:
        if namespace is not None and head.namespace == namespace:
            return identity_for_existing(tools, head.iri)
    try:
        iri = tools.ontology_manager.resolve_ontology_ref(handle)
    except AmbiguousOntologyRefError:
        return None
    return identity_for_existing(tools, iri) if iri is not None else None


def identity_for_existing(tools: ToolBox, iri: str) -> OntologyIdentity | None:
    """The identity of the catalog ontology ``iri``, if the catalog holds it.

    Raises:
        DivergedOntologyError: that ontology has several heads.
    """
    manager = tools.ontology_manager
    resolved = manager.resolve_ontology_ref(iri)
    head = manager.get_head_by_iri(resolved) if resolved is not None else None
    if head is None:
        return None
    return OntologyIdentity(
        ontology_id=head.ontology_id or iri.rsplit("/", 1)[-1],
        iri=head.iri,
        title=head.title or head.ontology_id or head.iri,
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
    iri = extension_ontology_iri(current_domain, external.iri, external.ontology_id)
    stored = identity_for_existing(tools, iri)
    if stored is not None:
        return stored.model_copy(update={"extends": external.iri})
    return OntologyIdentity(
        ontology_id=iri.rsplit("/", 1)[-1],
        iri=iri,
        title=f"{external.title} (extension)"[:200],
        description=(
            f"Terms this deployment adds to {external.title} <{external.iri}>, "
            "which it imports."
        ),
        extends=external.iri,
    )


def extended_ontology(tools: ToolBox, iri: str, current_domain: str) -> str | None:
    """The external catalog ontology whose extension ontology is ``iri``, if any."""
    for head in tools.ontology_manager.ontologies:
        if head.iri and not is_owned_ontology(head.iri, current_domain):
            stem = head.ontology_id or ""
            if extension_ontology_iri(current_domain, head.iri, stem) == iri:
                return head.iri
    return None


def covering_ontology(tools: ToolBox, text: str) -> OntologyIdentity | None:
    """The existing ontology the text covers best, if it names enough of its terms."""
    coverage = catalog_coverage(list(tools.ontology_manager.ontologies), text)
    if not coverage or coverage[0][0] < HOME_COVERAGE_MIN_TERMS:
        return None
    head = coverage[0][1]
    return identity_for_existing(tools, head.iri)


def _catalog_entry(handle: str, head: Ontology) -> str:
    """One list line: the handle, the title and IRI, then what the ontology covers."""
    name = " ".join(filter(None, (head.title or "", f"<{head.iri}>")))
    scope = head.description_preview()
    return f"- {handle}: {name}: {scope}" if scope else f"- {handle}: {name}"


async def propose_home_ontology(
    state: AgentState, tools: ToolBox
) -> OntologyIdentityProposal:
    """Ask the model for the document's home ontology id, title and description."""
    handles = catalog_handles(tools)
    if handles:
        heads = {o.iri: o for o in tools.ontology_manager.ontologies}
        catalog_clause = home_ontology_catalog_clause.format(
            catalog_list="\n".join(
                _catalog_entry(handle, heads[iri])
                for handle, iri in sorted(handles.items())
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
            validation_context={"catalog_handles": frozenset(handles)},
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
    identity = identity_for_handle(tools, proposal.ontology_id)
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
    elif identity.extends is None:
        # An extension picked by its own handle still imports its external.
        extends = extended_ontology(tools, identity.iri, state.current_domain)
        if extends is not None:
            identity = identity.model_copy(update={"extends": extends})
    state.home_ontology = identity
    if state.status != Status.FAILED:
        state.status = Status.SUCCESS
    return state
