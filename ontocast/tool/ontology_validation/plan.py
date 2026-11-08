"""The document ontology plan: model-free checks and the reduce-time split.

A plan lists the terms a document adds to its domain. Before any unit
renders, each proposal is checked against the whole catalog: a proposal that
names a catalog or well-known term becomes a reuse, a parent must exist, and
in-plan duplicates collapse. At reduce, a minted term the plan does not name is
a *candidate*: consolidation decides it, and without a decision it is dropped.

A minted term counts as planned when its IRI, its local name (case-insensitive)
or one of its labels matches a plan entry. The namespace is not compared:
which catalog ontology a unit writes into is the context resolver's choice.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from rdflib import URIRef

from ontocast.onto.identity import OntologyIdentity
from ontocast.onto.model import (
    OntologyPlan,
    OntologyPlanReport,
    OntologyPlanTerm,
    TermKind,
)
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.well_known import well_known_vocabulary
from ontocast.tool.facts_validation.terms import _local_name
from ontocast.tool.ontology_validation.reconcile import minted_roles, surfaces_of
from ontocast.tool.ontology_validation.shape import (
    CatalogSurfaceIndex,
    normalize_surface,
    resolve_unique,
    role_compatible,
)


def term_role(kind: TermKind) -> str:
    """The class/property role a term kind has in the catalog index."""
    return "class" if kind == "class" else "property"


def expand_reference(reference: str, prefixes: Mapping[str, str]) -> str | None:
    """An IRI for ``reference``: a full IRI, ``<IRI>``, or a known CURIE.

    Returns:
        The IRI, or ``None`` when ``reference`` is a bare name or an unknown
        prefix.
    """
    text = reference.strip().strip("<>").strip()
    if not text:
        return None
    if text.startswith(("http://", "https://", "urn:")):
        return text
    prefix, sep, local = text.partition(":")
    if sep and prefix in prefixes and local:
        return f"{prefixes[prefix]}{local}"
    return None


def compact_iri(iri: str, prefixes: Mapping[str, str]) -> str:
    """``prefix:local`` for ``iri`` under the longest matching namespace."""
    best: tuple[str, str] | None = None
    for prefix, namespace in prefixes.items():
        if prefix and namespace and iri.startswith(namespace):
            if best is None or len(namespace) > len(best[1]):
                best = (prefix, namespace)
    if best is None:
        return iri
    local = iri[len(best[1]) :]
    if not local or not local[0].isalpha():
        return iri
    return f"{best[0]}:{local}"


def _known_term(iri: str, catalog_index: CatalogSurfaceIndex) -> bool:
    return iri in catalog_index.terms or well_known_vocabulary(iri) is not None


def _resolve_known(
    reference: str | None,
    role: str,
    *,
    prefixes: Mapping[str, str],
    catalog_index: CatalogSurfaceIndex,
) -> str | None:
    """A catalog or well-known IRI ``reference`` names, role permitting."""
    if not reference:
        return None
    iri = expand_reference(reference, prefixes)
    if iri is not None:
        if _known_term(iri, catalog_index) and role_compatible(
            role, iri, catalog_index
        ):
            return iri
        return None
    match = resolve_unique(catalog_index, label=reference, local=reference)
    if match is not None and role_compatible(role, match[0], catalog_index):
        return match[0]
    return None


def check_plan(
    report: OntologyPlanReport,
    *,
    home: OntologyIdentity,
    catalog_index: CatalogSurfaceIndex,
    prefixes: Mapping[str, str],
    max_terms: int,
) -> OntologyPlan:
    """Accept, convert to reuse, or refuse each proposal, without a model.

    Rules, in order:

    - A proposal whose ``reuse`` names a known term of compatible role is a
      reuse of that term.
    - A proposal whose label or local name names exactly one catalog term of
      compatible role is a reuse of it.
    - A second proposal with the same local name or label is refused.
    - Past ``max_terms`` accepted entries, proposals are refused.
    - Anything else is minted under the home namespace.
    - A parent must be a known term or another accepted entry; an unknown
      parent is dropped and the term kept.

    Args:
        report: The model's proposals.
        home: The document's home ontology; minted IRIs live in its namespace.
        catalog_index: Names of every catalog term.
        prefixes: CURIE prefixes the prompt showed, prefix to namespace.
        max_terms: Largest number of accepted entries.

    Returns:
        The accepted plan, with refusals recorded.
    """
    plan = OntologyPlan()
    seen_local: set[str] = set()
    seen_label: set[str] = set()
    parents: list[str | None] = []
    for proposal in report.proposals:
        role = term_role(proposal.kind)
        local_key = proposal.local_name.casefold()
        label_key = normalize_surface(proposal.label)
        if local_key in seen_local or label_key in seen_label:
            plan.rejected.append((proposal.local_name, "duplicate in plan"))
            continue
        if len(plan.terms) >= max_terms:
            plan.rejected.append((proposal.local_name, "plan size cap"))
            continue
        reuse_iri = _resolve_known(
            proposal.reuse, role, prefixes=prefixes, catalog_index=catalog_index
        )
        resolved_by_name = False
        if reuse_iri is None:
            minted_iri = f"{home.namespace}{proposal.local_name}"
            if minted_iri in catalog_index.terms:
                reuse_iri = minted_iri
            else:
                match = resolve_unique(
                    catalog_index, label=proposal.label, local=proposal.local_name
                )
                if match is not None and role_compatible(role, match[0], catalog_index):
                    reuse_iri = match[0]
                    resolved_by_name = True
        seen_local.add(local_key)
        seen_label.add(label_key)
        if reuse_iri is not None:
            plan.resolved_to_catalog += int(resolved_by_name)
            plan.terms.append(
                OntologyPlanTerm(
                    kind=proposal.kind,
                    label=proposal.label,
                    local_name=proposal.local_name,
                    iri=reuse_iri,
                    decision="reuse",
                    definition=proposal.definition,
                )
            )
        else:
            plan.terms.append(
                OntologyPlanTerm(
                    kind=proposal.kind,
                    label=proposal.label,
                    local_name=proposal.local_name,
                    iri=f"{home.namespace}{proposal.local_name}",
                    decision="mint",
                    definition=proposal.definition,
                )
            )
        parents.append(proposal.parent)

    by_local = {term.local_name.casefold(): term.iri for term in plan.terms}
    plan_iris = {term.iri for term in plan.terms}
    for term, parent in zip(plan.terms, parents, strict=True):
        if term.decision != "mint" or not parent:
            continue
        role = term_role(term.kind)
        iri = _resolve_known(
            parent, role, prefixes=prefixes, catalog_index=catalog_index
        )
        if iri is None:
            expanded = expand_reference(parent, prefixes)
            if expanded in plan_iris:
                iri = expanded
            else:
                iri = by_local.get(parent.strip().casefold())
        if iri is not None and iri != term.iri:
            term.parent_iri = iri
    return plan


class PlanIndex:
    """What a minted term is matched against: plan IRIs, local names, labels."""

    def __init__(self, plan: OntologyPlan) -> None:
        self.iris = {term.iri for term in plan.terms}
        self.locals = {term.local_name.casefold() for term in plan.terms}
        self.labels = {normalize_surface(term.label) for term in plan.terms}

    def covers(self, iri: str, graph: RDFGraph) -> bool:
        """Whether the plan names the term ``iri`` declared in ``graph``."""
        if iri in self.iris or _local_name(iri).casefold() in self.locals:
            return True
        return any(
            normalize_surface(surface) in self.labels
            for surface in surfaces_of(URIRef(iri), graph)
        )


def minted_terms(
    graphs: Iterable[RDFGraph], catalog_index: CatalogSurfaceIndex
) -> dict[str, str]:
    """Terms the graphs declare that neither the catalog nor a well-known
    vocabulary holds, with their role (``class`` or ``property``)."""
    roles: dict[str, str] = {}
    for graph in graphs:
        for subject, role in minted_roles(graph).items():
            iri = str(subject)
            if role not in ("class", "property"):
                continue
            if iri in catalog_index.terms or well_known_vocabulary(iri) is not None:
                continue
            roles.setdefault(iri, role)
    return roles


def plan_candidates(
    graphs: Iterable[RDFGraph],
    plan: OntologyPlan | None,
    catalog_index: CatalogSurfaceIndex,
) -> set[str]:
    """Minted terms the plan does not name; none when there is no plan."""
    if plan is None:
        return set()
    graph_list = list(graphs)
    index = PlanIndex(plan)
    candidates: set[str] = set()
    for iri in minted_terms(graph_list, catalog_index):
        if not any(index.covers(iri, graph) for graph in graph_list):
            candidates.add(iri)
    return candidates


__all__ = [
    "PlanIndex",
    "check_plan",
    "compact_iri",
    "expand_reference",
    "minted_terms",
    "plan_candidates",
    "term_role",
]
