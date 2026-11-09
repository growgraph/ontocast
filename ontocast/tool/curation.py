"""Curate a catalog ontology across the documents that grew it.

A term one document added and no other document uses is the definition of a
document-level term, and only a pass over the catalog can see it. Curation:

1. attributes every term of the head to the version that introduced it, and
   that version to the document its header names (``prov:wasDerivedFrom``);
   terms of a version with no document (a seed file, an upload) are *authored*
   and never curated;
2. counts, per term, the documents whose facts use it, when the facts store
   answers SPARQL;
3. flags terms without a model: used by at most one document, unused, a name
   variant, a value-set family, or a duplicate of an authored term;
4. asks the model for an action on the flagged terms only, in batches;
5. compiles the actions into one update of the head, written as a new version
   on request.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from rdflib import OWL, RDFS, Literal, URIRef
from rdflib.namespace import DCTERMS, PROV

from ontocast.agent.common import call_llm_with_retry
from ontocast.agent.consolidate_ontology import (
    TermContext,
    format_other_terms,
    format_term_block,
)
from ontocast.onto.enum import LLMPhase
from ontocast.onto.model import OntologyConsolidationReport, OntologyTermAction
from ontocast.onto.ontology import Ontology, bump_version, increment_type_for
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import BudgetTracker
from ontocast.onto.well_known import well_known_vocabulary
from ontocast.prompt.common import system_preamble_ontology
from ontocast.prompt.consolidate_ontology import terms_intro
from ontocast.prompt.curate_catalog import curation_instruction, template_prompt
from ontocast.tool.llm import RUN_STOPPING_ERRORS, use_budget_tracker
from ontocast.tool.ontology_validation.catalog_terms import catalog_prefixes
from ontocast.tool.ontology_validation.consolidate import (
    compile_actions,
    resolve_actions,
    structural_flags,
)
from ontocast.tool.ontology_validation.reconcile import minted_roles
from ontocast.tool.ontology_validation.shape import (
    build_catalog_surface_index,
    first_label,
)
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)

#: Source IRIs a curation version records; never counted as a document.
CURATION_SOURCE_PREFIX = "urn:ontocast:curation:"


class CuratedTerm(BaseModel):
    """One term a document added, with what curation knows about it."""

    iri: str
    role: str
    minted_by: str
    #: Documents whose facts use the term; None when usage was not counted.
    used_by: list[str] | None = None
    flags: list[str] = Field(default_factory=list)


class CurationReport(BaseModel):
    """What curation found and decided for one ontology."""

    ontology_iri: str
    head_version_iri: str | None = None
    authored_terms: int = 0
    terms: list[CuratedTerm] = Field(default_factory=list)
    actions: dict[str, str] = Field(
        default_factory=dict,
        description="Decided action per reviewed term: 'merge_into <iri>' etc.",
    )
    ignored: list[tuple[str, str]] = Field(default_factory=list)
    calls: int = 0
    calls_failed: int = 0
    triples_deleted: int = 0
    triples_inserted: int = 0
    predicted_version: str | None = None
    written_version_iri: str | None = None


def version_source(version: Ontology, graph: RDFGraph) -> str | None:
    """The document a version's header says it was derived from, if any."""
    for source in graph.objects(URIRef(version.iri), PROV.wasDerivedFrom):
        text = str(source)
        if not text.startswith(CURATION_SOURCE_PREFIX):
            return text
    return None


def _declared(graph: RDFGraph) -> dict[str, str]:
    return {
        str(subject): role
        for subject, role in minted_roles(graph).items()
        if role in ("class", "property") and well_known_vocabulary(str(subject)) is None
    }


async def attribute_terms(
    tools: ToolBox, head: Ontology
) -> tuple[dict[str, str], dict[str, str]]:
    """Which document introduced each term of ``head``.

    A term is introduced by the version that declares it while none of that
    version's priors does. Each version's header names its own source
    (``prov:wasDerivedFrom`` is rewritten per version).

    The lineage is walked from the versions the manager holds through
    ``owl:priorVersion``: a manager loaded from a triple store holds only each
    ontology's head, and an unvisited prior would leave every term it
    introduced counted as authored. A version not held is read from its stored
    graph, whose header carries its priors, its date and its source.

    Returns:
        ``(minted_by, authored)``: term IRI to its document, and term IRI to
        role for terms of versions with no document.
    """
    held = {
        v.version_iri: v
        for v in tools.ontology_manager.get_ontology_versions_by_iri(head.iri)
        if v.version_iri
    }
    if head.version_iri:
        held.setdefault(head.version_iri, head)
    subject = URIRef(head.iri)
    graphs: dict[str, RDFGraph] = {}

    async def graph_of(version_iri: str) -> RDFGraph:
        if version_iri not in graphs:
            version = held.get(version_iri)
            graph = version.graph if version is not None else None
            if graph is None or len(graph) == 0:
                graph = await tools.aget_version_graph(version_iri) or RDFGraph()
            graphs[version_iri] = graph
        return graphs[version_iri]

    def priors_of(version_iri: str, graph: RDFGraph) -> list[str]:
        version = held.get(version_iri)
        if version is not None:
            return list(version.prior_version_iris)
        return sorted(str(p) for p in graph.objects(subject, OWL.priorVersion))

    def stamp_of(version_iri: str, graph: RDFGraph) -> datetime:
        version = held.get(version_iri)
        stamp = version.modified_at if version is not None else None
        if stamp is None:
            value = graph.value(subject, DCTERMS.modified)
            parsed = value.toPython() if isinstance(value, Literal) else None
            stamp = parsed if isinstance(parsed, datetime) else None
        return stamp or datetime.min.replace(tzinfo=timezone.utc)

    lineage: list[str] = list(held)
    seen = set(lineage)
    for version_iri in lineage:  # grows while walked: breadth-first over priors
        for prior in priors_of(version_iri, await graph_of(version_iri)):
            if prior not in seen:
                seen.add(prior)
                lineage.append(prior)

    introduced_by: dict[str, tuple[datetime, str | None]] = {}
    for version_iri in lineage:
        graph = await graph_of(version_iri)
        declared = set(_declared(graph))
        for prior in priors_of(version_iri, graph):
            declared -= set(_declared(await graph_of(prior)))
        stamp = stamp_of(version_iri, graph)
        source = version_source(head, graph)
        for term in declared:
            current = introduced_by.get(term)
            if current is None or stamp < current[0]:
                introduced_by[term] = (stamp, source)

    head_terms = _declared(head.graph)
    minted_by: dict[str, str] = {}
    authored: dict[str, str] = {}
    for term, role in head_terms.items():
        source = introduced_by.get(term, (None, None))[1]
        if source:
            minted_by[term] = source
        else:
            authored[term] = role
    return minted_by, authored


async def facts_usage(tools: ToolBox) -> dict[str, set[str]] | None:
    """Documents whose facts use each type or predicate, from the facts store.

    None when no store answers SPARQL. A document is one named graph of the
    facts partition.
    """
    store = tools.triple_store_manager
    if store is None or not store.supports_sparql_select():
        return None
    queries = (
        "SELECT DISTINCT ?g ?term WHERE { GRAPH ?g { ?s a ?term } "
        "FILTER(isIRI(?term)) }",
        "SELECT DISTINCT ?g ?term WHERE { GRAPH ?g { ?s ?term ?o } }",
    )
    usage: dict[str, set[str]] = {}
    for query in queries:
        for row in await store.aselect(query, store="facts"):
            graph, term = row.get("g"), row.get("term")
            if graph and term and not graph.endswith("/ontology-provenance"):
                usage.setdefault(term, set()).add(graph)
    return usage


def _contexts(
    head: Ontology,
    terms: Sequence[CuratedTerm],
    flagged: Mapping[str, set[str]],
) -> list[TermContext]:
    by_iri = {term.iri: term for term in terms}
    contexts = []
    for iri in sorted(flagged):
        term = by_iri[iri]
        context = TermContext(iri, term.role)
        uses = (
            f", used by {len(term.used_by)} document(s)"
            if term.used_by is not None
            else ""
        )
        context.status = f"added by {term.minted_by}{uses}"
        context.reasons = sorted(flagged[iri])
        subject = URIRef(iri)
        context.label = first_label(subject, head.graph) or ""
        comments = sorted(str(v) for v in head.graph.objects(subject, RDFS.comment))
        if comments:
            context.definition = " ".join(comments[0].split())
        context.parents = sorted(
            str(parent)
            for predicate in (RDFS.subClassOf, RDFS.subPropertyOf)
            for parent in head.graph.objects(subject, predicate)
            if isinstance(parent, URIRef)
        )
        contexts.append(context)
    return contexts


async def _decide(
    tools: ToolBox,
    head: Ontology,
    contexts: Sequence[TermContext],
    *,
    other_terms: str,
    prefixes: Mapping[str, str],
    batch_size: int,
    report: CurationReport,
) -> list[OntologyTermAction]:
    parser = PydanticOutputParser(pydantic_object=OntologyConsolidationReport)
    prompt = PromptTemplate(
        template=template_prompt,
        input_variables=[
            "preamble",
            "instruction",
            "terms",
            "other_terms",
            "format_instructions",
        ],
    )
    instruction = curation_instruction.format(
        ontology_id=head.ontology_id or head.iri, title=head.title or ""
    )
    tracker = BudgetTracker()
    actions: list[OntologyTermAction] = []
    for start in range(0, len(contexts), max(1, batch_size)):
        batch = contexts[start : start + batch_size]
        blocks = "\n\n".join(format_term_block(c, prefixes) for c in batch)
        report.calls += 1
        try:
            with use_budget_tracker(tracker):
                llm_tool = await tools.get_llm_tool(tracker)
                answer = await call_llm_with_retry(
                    llm_tool=llm_tool,
                    phase=LLMPhase.ONTOLOGY_CURATION,
                    prompt=prompt,
                    parser=parser,
                    prompt_kwargs={
                        "preamble": system_preamble_ontology,
                        "instruction": instruction,
                        "terms": terms_intro.format(term_blocks=blocks),
                        "other_terms": other_terms,
                        "format_instructions": parser.get_format_instructions(),
                    },
                )
        except RUN_STOPPING_ERRORS:
            raise
        except Exception as exc:  # noqa: BLE001 - the batch keeps its terms
            logger.warning("Curation call failed: %s", exc)
            report.calls_failed += 1
            continue
        actions.extend(answer.actions)
    return actions


async def curate_ontology(
    tools: ToolBox,
    head: Ontology,
    *,
    apply: bool = False,
    batch_size: int = 40,
    use_facts: bool = True,
) -> CurationReport:
    """Curate one catalog ontology; see the module docstring."""
    report = CurationReport(ontology_iri=head.iri, head_version_iri=head.version_iri)
    minted_by, authored = await attribute_terms(tools, head)
    report.authored_terms = len(authored)
    roles = _declared(head.graph)
    curated = {iri: roles[iri] for iri in minted_by}
    usage = await facts_usage(tools) if use_facts else None
    if use_facts and usage is None:
        logger.warning("Facts usage not counted: the facts store has no SPARQL")

    # Duplicates and variants are judged against the authored terms (and the
    # other ontologies): a curated term is in the head, so against the head
    # itself it would always match itself.
    authored_graph = RDFGraph()
    for triple in head.graph:
        if str(triple[0]) not in curated:
            authored_graph.add(triple)
    others = [o.graph for o in tools.ontology_manager.ontologies if o.iri != head.iri]
    authored_index = build_catalog_surface_index([authored_graph, *others])
    flagged = structural_flags(curated, [head.graph], authored_index)
    terms: list[CuratedTerm] = []
    for iri in sorted(curated):
        used = sorted(usage.get(iri, set())) if usage is not None else None
        if used is not None and len(used) == 0:
            flagged.setdefault(iri, set()).add("unused")
        elif used is not None and len(used) == 1:
            flagged.setdefault(iri, set()).add("single_document_use")
        terms.append(
            CuratedTerm(
                iri=iri,
                role=curated[iri],
                minted_by=minted_by[iri],
                used_by=used,
                flags=sorted(flagged.get(iri, set())),
            )
        )
    report.terms = terms
    if not flagged:
        return report

    prefixes = catalog_prefixes(list(tools.ontology_manager.ontologies), None)
    actions = await _decide(
        tools,
        head,
        _contexts(head, terms, flagged),
        other_terms=format_other_terms(curated, flagged, [head.graph], prefixes),
        prefixes=prefixes,
        batch_size=batch_size,
        report=report,
    )
    decided, ignored = resolve_actions(
        actions,
        minted=curated,
        candidates=set(),
        catalog_index=authored_index,
        prefixes=prefixes,
        reviewed=flagged,
    )
    report.ignored = ignored
    report.actions = {
        iri: f"{action.action} {action.target or ''}".strip()
        for iri, action in decided.items()
        if iri in flagged
    }
    compiled = compile_actions(decided, head.graph, minted=curated)
    report.triples_deleted = len(compiled.deletes)
    report.triples_inserted = len(compiled.inserts)
    if not compiled.deletes and not compiled.inserts:
        return report
    update = compiled.to_graph_update()
    report.predicted_version = bump_version(
        head.version or "1.0.0", increment_type_for([update])[0]
    )
    if apply:
        source = f"{CURATION_SOURCE_PREFIX}{datetime.now(timezone.utc).isoformat()}"
        version = await tools.acommit_ontology_update(head.iri, update, source=source)
        if version is not None:
            report.written_version_iri = version.version_iri
    return report


async def curate_catalog(
    tools: ToolBox,
    *,
    ontology: str | None = None,
    apply: bool = False,
    batch_size: int = 40,
    use_facts: bool = True,
) -> list[CurationReport]:
    """Curate one catalog ontology, or every one.

    Args:
        tools: An initialised toolbox; its catalog and facts store are read.
        ontology: An ontology id, alias or IRI; every head when None.
        apply: Write each changed ontology as a new version. Otherwise only
            report.
        batch_size: Flagged terms per model call.
        use_facts: Count, per term, the documents whose facts use it.

    Raises:
        ValueError: ``ontology`` names nothing in the catalog.
        DivergedOntologyError: the named ontology has several heads.
    """
    manager = tools.ontology_manager
    if ontology is not None:
        iri = manager.resolve_ontology_ref(ontology)
        head = manager.get_head_by_iri(iri) if iri is not None else None
        if head is None:
            raise ValueError(f"no catalog ontology named {ontology!r}")
        heads = [head]
    else:
        heads = list(manager.ontologies)
    return [
        await curate_ontology(
            tools, head, apply=apply, batch_size=batch_size, use_facts=use_facts
        )
        for head in heads
    ]


__all__ = [
    "CuratedTerm",
    "CurationReport",
    "attribute_terms",
    "curate_catalog",
    "curate_ontology",
    "facts_usage",
]
