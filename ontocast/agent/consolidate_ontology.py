"""Decide every term a document minted, before it is written.

After the ontology fan-out, the merged update holds every term the units
minted, the plan's and the candidates (terms a unit minted outside the plan).
One structured LLM call per batch of terms returns an action per term
(:class:`~ontocast.onto.model.OntologyTermAction`); the actions are compiled
into the update without a model
(:mod:`ontocast.tool.ontology_validation.consolidate`).

Each term is shown with what the decision needs and a unit cannot see: the
whole document's other terms, its catalog neighbours, the findings on it, and
short excerpts from every unit that minted it.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from rdflib import RDFS, SKOS, Literal, URIRef

from ontocast.agent.common import call_llm_with_retry
from ontocast.onto.enum import LLMPhase
from ontocast.onto.model import (
    OntologyConsolidationReport,
    OntologyPlan,
    OntologyTermAction,
    OntologyUnitFinding,
)
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.prompt.common import system_preamble_ontology
from ontocast.prompt.consolidate_ontology import (
    consolidation_instruction,
    other_terms_intro,
    template_prompt,
    terms_intro,
)
from ontocast.tool.facts_validation.terms import _local_name
from ontocast.tool.llm import (
    RUN_STOPPING_ERRORS,
    record_active_count,
    use_budget_tracker,
)
from ontocast.tool.ontology_validation.plan import compact_iri
from ontocast.tool.ontology_validation.reconcile import MintedDuplicate
from ontocast.tool.ontology_validation.shape import first_label
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)

#: Characters of document text shown on each side of a term's first mention.
EVIDENCE_WINDOW = 200
#: Excerpts shown per term, one per unit that minted it.
EVIDENCE_PER_TERM = 2


class TermContext:
    """What the consolidation prompt shows about one minted term."""

    def __init__(self, iri: str, role: str) -> None:
        self.iri = iri
        self.role = role
        self.status = "planned"
        #: Why the term is up for a decision (``candidate``, ``variant``, ...).
        self.reasons: list[str] = []
        self.label = ""
        self.definition = ""
        self.parents: list[str] = []
        self.nearest: list[tuple[str, str]] = []
        self.findings: list[str] = []
        self.excerpts: list[str] = []


def _local_key(iri: str) -> str:
    return _local_name(iri).casefold()


def _definition(term: URIRef, graphs: Sequence[RDFGraph]) -> str:
    for graph in graphs:
        for predicate in (SKOS.definition, RDFS.comment):
            for value in graph.objects(term, predicate):
                if isinstance(value, Literal) and str(value).strip():
                    return " ".join(str(value).split())
    return ""


def _excerpt(text: str, label: str) -> str | None:
    """``EVIDENCE_WINDOW`` characters either side of the label's first mention.

    Falls back to the label's longest word when the whole label does not occur.
    """
    if not text or not label:
        return None
    probes = [label] + sorted(
        (w for w in re.split(r"\W+", label) if len(w) >= 4), key=len, reverse=True
    )
    lowered = text.casefold()
    for probe in probes:
        position = lowered.find(probe.casefold())
        if position >= 0:
            start = max(0, position - EVIDENCE_WINDOW)
            end = min(len(text), position + len(probe) + EVIDENCE_WINDOW)
            snippet = " ".join(text[start:end].split())
            return f"...{snippet}..."
    return None


def build_term_contexts(
    state: AgentState,
    *,
    carriers: Sequence[RDFGraph],
    minted: Mapping[str, str],
    flagged: Mapping[str, set[str]],
    candidates: set[str],
    findings: Sequence[OntologyUnitFinding],
    term_units: Mapping[str, set[int]],
    duplicates: Sequence[MintedDuplicate],
    plan: OntologyPlan | None,
) -> list[TermContext]:
    """One context per flagged term, ordered by IRI."""
    plan_nearest: dict[str, list[tuple[str, str]]] = {}
    for term in plan.terms if plan is not None else []:
        plan_nearest[term.iri] = list(term.nearest)
        plan_nearest[term.local_name.casefold()] = list(term.nearest)
    by_subject: dict[str, list[str]] = {}
    for finding in findings:
        if finding.subject:
            by_subject.setdefault(finding.subject, []).append(
                f"{finding.kind}: {finding.message}"
            )
    duplicate_of = {d.minted_iri: d.catalog_iri for d in duplicates}

    contexts: list[TermContext] = []
    for iri in sorted(flagged):
        subject = URIRef(iri)
        context = TermContext(iri, minted[iri])
        context.status = "candidate" if iri in candidates else "planned"
        context.reasons = sorted(flagged[iri])
        context.label = first_label(subject, *carriers) or ""
        context.definition = _definition(subject, carriers)
        context.parents = sorted(
            {
                str(parent)
                for graph in carriers
                for predicate in (RDFS.subClassOf, RDFS.subPropertyOf)
                for parent in graph.objects(subject, predicate)
                if isinstance(parent, URIRef)
            }
        )
        context.nearest = list(
            plan_nearest.get(iri) or plan_nearest.get(_local_key(iri)) or []
        )
        if iri in duplicate_of:
            context.nearest.insert(0, (duplicate_of[iri], "same name in the catalog"))
        context.findings = by_subject.get(iri, [])[:3]
        for unit_index in sorted(term_units.get(iri, set())):
            if len(context.excerpts) >= EVIDENCE_PER_TERM:
                break
            if 0 <= unit_index < len(state.content_units):
                excerpt = _excerpt(
                    state.content_units[unit_index].text,
                    context.label or _local_name(iri),
                )
                if excerpt:
                    context.excerpts.append(excerpt)
        contexts.append(context)
    return contexts


def format_term_block(context: TermContext, prefixes: Mapping[str, str]) -> str:
    """The prompt lines for one term."""

    def curie(iri: str) -> str:
        return compact_iri(iri, prefixes)

    lines = [f"### `{curie(context.iri)}` ({context.role}, {context.status})"]
    if context.reasons:
        lines.append("review because: " + ", ".join(context.reasons))
    if context.label:
        lines.append(f"label: {context.label}")
    if context.definition:
        lines.append(f"definition: {context.definition}")
    if context.parents:
        lines.append("parents: " + ", ".join(f"`{curie(p)}`" for p in context.parents))
    if context.nearest:
        lines.append(
            "near: "
            + ", ".join(
                f"`{curie(iri)}`" + (f" ({label})" if label else "")
                for iri, label in context.nearest
            )
        )
    for finding in context.findings:
        lines.append(f"check: {finding}")
    for excerpt in context.excerpts:
        lines.append(f"text: {excerpt}")
    return "\n".join(lines)


def format_other_terms(
    minted: Mapping[str, str],
    flagged: Mapping[str, set[str]],
    carriers: Sequence[RDFGraph],
    prefixes: Mapping[str, str],
) -> str:
    """One line per minted term that needs no decision: possible merge targets."""
    lines = [
        f"- `{compact_iri(iri, prefixes)}` | {minted[iri]} | "
        f"{first_label(URIRef(iri), *carriers) or ''}".rstrip(" |")
        for iri in sorted(minted)
        if iri not in flagged
    ]
    if not lines:
        return ""
    return other_terms_intro.format(term_lines="\n".join(lines))


async def consolidate_terms(
    state: AgentState,
    tools: ToolBox,
    contexts: Sequence[TermContext],
    *,
    prefixes: Mapping[str, str],
    batch_size: int,
    other_terms: str = "",
) -> tuple[list[OntologyTermAction], int]:
    """Ask for an action per term, ``batch_size`` terms per call.

    Returns:
        Every action returned, and the number of calls that failed; a failed
        batch leaves its terms to their defaults.
    """
    parser = PydanticOutputParser(pydantic_object=OntologyConsolidationReport)
    prompt = PromptTemplate(
        template=template_prompt,
        input_variables=[
            "preamble",
            "instruction",
            "terms",
            "other_terms",
            "user_instruction",
            "format_instructions",
        ],
    )
    home = state.home_ontology
    instruction = consolidation_instruction.format(
        ontology_id=home.ontology_id if home is not None else "of this document",
        title=home.title if home is not None else "its domain",
    )
    actions: list[OntologyTermAction] = []
    failed = 0
    for start in range(0, len(contexts), max(1, batch_size)):
        batch = contexts[start : start + batch_size]
        blocks = "\n\n".join(format_term_block(context, prefixes) for context in batch)
        record_active_count("llm/ontology_consolidation")
        try:
            with use_budget_tracker(state.budget_tracker):
                llm_tool = await tools.get_llm_tool(state.budget_tracker)
                report = await call_llm_with_retry(
                    llm_tool=llm_tool,
                    phase=LLMPhase.ONTOLOGY_CONSOLIDATION,
                    prompt=prompt,
                    parser=parser,
                    prompt_kwargs={
                        "preamble": system_preamble_ontology,
                        "instruction": instruction,
                        "terms": terms_intro.format(term_blocks=blocks),
                        "other_terms": other_terms,
                        "user_instruction": state.ontology_user_instruction,
                        "format_instructions": parser.get_format_instructions(),
                    },
                )
        except RUN_STOPPING_ERRORS:
            raise
        except Exception as exc:  # noqa: BLE001 - the batch keeps its defaults
            logger.warning("Ontology consolidation call failed: %s", exc)
            failed += 1
            continue
        actions.extend(report.actions)
    return actions, failed
