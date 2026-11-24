"""Facts domain-repair agent.

Runs after the facts render/critic loop, only on a unit with a mandatory
``domain_violation`` finding: a statement whose property declares an
``rdfs:domain``, in the unit's snapshot or anywhere in the catalog, that its
subject's asserted types contradict. One LLM call per unit
is shown those statements under citable ids and may replace each one's
property; :func:`property_swaps` keeps only patches that do exactly that and
leave the subject fitting the new property's domain. The unit loop
(``ontocast.stategraph.atomic``) applies them through the same per-fix
regression check as a critic fix, and drops what is still in violation.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from rdflib import RDF, RDFS, URIRef

from ontocast.agent.common import call_llm_with_retry
from ontocast.onto.enum import LLMPhase
from ontocast.onto.model import (
    FactsDomainRepairReport,
    FactsUnitFinding,
    FactsUnitFindingKind,
    TripleFix,
)
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.triple_index import Triple, TripleIndex, build_triple_index
from ontocast.onto.unit_states import UnitFactsState
from ontocast.prompt.common import text_template, user_template
from ontocast.prompt.graph_format import get_graph_format_profile
from ontocast.prompt.repair_facts_domain import (
    build_violations_chapter,
    output_instruction_for,
    preamble,
    repair_instruction,
    template_prompt,
)
from ontocast.tool.atomic import AtomicToolBox
from ontocast.tool.facts_validation.critic_patch import FixPatch
from ontocast.tool.facts_validation.terms import (
    _declared_domains,
    domain_fits,
    domain_fitting_properties,
)
from ontocast.tool.llm import RUN_STOPPING_ERRORS

logger = logging.getLogger(__name__)


def violating_statements(
    graph: RDFGraph, findings: Sequence[FactsUnitFinding]
) -> list[Triple]:
    """The statements of the (subject, property) pairs mandatory domain violations name."""
    pairs = {
        (finding.subject, finding.predicate)
        for finding in findings
        if finding.kind == FactsUnitFindingKind.DOMAIN_VIOLATION
        and finding.mandatory
        and finding.subject
        and finding.predicate
    }
    return sorted(
        (
            triple
            for subject, predicate in pairs
            for triple in graph.triples((URIRef(subject), URIRef(predicate), None))
        ),
        key=lambda triple: tuple(str(term) for term in triple),
    )


def violations_index(graph: RDFGraph, statements: Sequence[Triple]) -> TripleIndex:
    """Ids over exactly ``statements``, listed with their ends' types and labels."""
    display = RDFGraph()
    for prefix, namespace in graph.namespaces():
        display.bind(prefix, namespace, override=True)
    scope = RDFGraph()
    for triple in statements:
        display.add(triple)
        scope.add(triple)
        for node in (triple[0], triple[2]):
            for predicate in (RDF.type, RDFS.label):
                for value in graph.objects(node, predicate):
                    display.add((node, predicate, value))
    return build_triple_index(display, scope=scope)


def property_swaps(
    patches: Sequence[FixPatch],
    index: TripleIndex,
    graph: RDFGraph,
    ontology_graph: RDFGraph,
) -> list[FixPatch]:
    """The patches that only replace the property of the statements they cite.

    Each cited statement ``s p o`` is replaced by ``s p' o`` with ``p'`` a
    different property whose declared domain ``s``'s types do not contradict,
    and nothing else is written. A patch doing anything more is refused: the
    call is shown only the cited statements, so it may change nothing else.
    """
    kept: list[FixPatch] = []
    for patch in patches:
        cited = [index.resolve(triple_id) for triple_id in patch.fix.triple_ids]
        inserts = patch.insert_triples
        if not cited or any(triple is None for triple in cited) or not inserts:
            continue
        ends = {(triple[0], triple[2]): triple[1] for triple in cited if triple}
        if len(inserts) != len(ends) or any(
            (subject, obj) not in ends
            or predicate == ends[(subject, obj)]
            or not isinstance(subject, URIRef)
            or not isinstance(predicate, URIRef)
            or not domain_fits(subject, predicate, graph, ontology_graph)
            for subject, predicate, obj in inserts
        ):
            logger.info(
                "Domain repair fix refused (not a property swap): %s",
                patch.fix.correct_value,
            )
            continue
        kept.append(patch)
    return kept


def _build_prompt() -> PromptTemplate:
    return PromptTemplate(
        template=template_prompt,
        input_variables=[
            "preamble",
            "repair_instruction",
            "user_instruction",
            "text_chapter",
            "violations_chapter",
            "output_instruction",
            "format_instructions",
        ],
    )


async def repair_domain_violations(
    state: UnitFactsState,
    atomic: AtomicToolBox,
    findings: Sequence[FactsUnitFinding],
) -> tuple[list[TripleFix], TripleIndex]:
    """Ask for a fitting property for each statement in domain violation.

    Args:
        state: The unit's current facts state. Read-only here.
        atomic: Toolbox for the LLM call.
        findings: The unit's findings; only mandatory ``domain_violation``
            ones are read. With none there is no call.

    Returns:
        ``REPLACE`` fixes citing ids of the returned index, and the index.
        Any other action is dropped.
    """
    graph = state.content_unit.graph
    # The catalog's axioms with the snapshot: a violation may rest on a domain
    # retrieval did not keep, and a fitting property may be one it left out.
    ontology_graph = atomic.catalog_roles().with_axioms(state.ontology_snapshot.graph)
    statements = violating_statements(graph, findings)
    index = violations_index(graph, statements)
    if not statements:
        return [], index

    domains = _declared_domains(ontology_graph, source="rdfs")
    chapter = build_violations_chapter(
        graph,
        index,
        {triple: domains.get(URIRef(str(triple[1])), set()) for triple in statements},
        {
            triple: domain_fitting_properties(
                URIRef(str(triple[0])),
                triple[2],
                graph,
                ontology_graph,
                replacing=URIRef(str(triple[1])),
            )
            for triple in statements
        },
    )
    profile = get_graph_format_profile(
        state.llm_graph_format,
        ontology_chapter_format=state.ontology_chapter_format,
        output_layout=state.llm_output_layout,
    )
    user_instruction = (
        user_template.format(user_instruction=state.facts_user_instruction)
        if state.facts_user_instruction
        else ""
    )
    prompt_data = {
        "preamble": preamble,
        "repair_instruction": repair_instruction,
        "user_instruction": user_instruction,
        "text_chapter": text_template.format(text=state.content_unit.extraction_text),
        "violations_chapter": chapter,
        "output_instruction": output_instruction_for(profile.format),
        "format_instructions": profile.format_instructions(
            FactsDomainRepairReport, web_search_enabled=False
        ),
    }
    llm_tool = await atomic.get_llm_tool(state.budget_tracker)
    try:
        report: FactsDomainRepairReport = await call_llm_with_retry(
            llm_tool=llm_tool,
            phase=LLMPhase.FACTS_COMPLETION,
            prompt=_build_prompt(),
            parser=PydanticOutputParser(pydantic_object=FactsDomainRepairReport),
            prompt_kwargs=prompt_data,
            llm_graph_format=state.llm_graph_format,
        )
    except RUN_STOPPING_ERRORS:
        raise
    except Exception as exc:
        # The statements are dropped after this call whatever it returns, so a
        # failed call costs the repair, not the unit.
        logger.warning("Facts domain-repair call failed: %s", exc)
        return [], index

    fixes = [
        fix
        for fix in report.proposed_fixes
        if fix.action == "REPLACE"
        and fix.triple_ids
        and all(triple_id in index.by_id for triple_id in fix.triple_ids)
    ]
    dropped = len(report.proposed_fixes) - len(fixes)
    if dropped:
        logger.info(
            "Domain repair: %d fix(es) dropped (not a REPLACE of a listed id)",
            dropped,
        )
    return fixes, index
