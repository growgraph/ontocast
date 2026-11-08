"""Compile term actions into a graph patch, deterministically.

Consolidation (per document) and curation (per catalog) ask a model for one
action per minted term; this module is the only place those actions become
triples. A model never writes a triple here, so a bad answer can at worst be
ignored, and every action is checked before it is applied:

- the term must be a minted term the caller listed -- a catalog term's own
  axioms are never edited;
- a ``merge_into`` / ``generalise_to`` target must be a listed minted term, a
  catalog term or a well-known term of compatible role, and a merge chain must
  end at a term that survives;
- ``demote_to_property_value`` applies to classes only.

An invalid action falls back to the term's default: ``keep``, or ``drop`` for a
candidate (a term the document plan did not name).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field
from rdflib import OWL, RDF, RDFS, SKOS, URIRef
from rdflib.graph import Graph
from rdflib.term import Node

from ontocast.onto.model import OntologyTermAction, TermActionKind
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import GraphUpdate, TripleOp
from ontocast.onto.well_known import well_known_vocabulary
from ontocast.tool.facts_validation.terms import _local_name
from ontocast.tool.ontology_validation.plan import expand_reference
from ontocast.tool.ontology_validation.reconcile import match_catalog_term
from ontocast.tool.ontology_validation.shape import (
    CatalogSurfaceIndex,
    first_label,
    name_tokens,
    value_set_families,
    variant_term_pairs,
)

_PARENT_PREDICATE = {"class": RDFS.subClassOf, "property": RDFS.subPropertyOf}


class ResolvedAction(BaseModel):
    """An action after validation: the term and target as IRIs."""

    term: str
    action: TermActionKind
    target: str | None = None


class CompiledActions(BaseModel):
    """What a set of term actions does to one graph."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    deletes: RDFGraph = Field(default_factory=RDFGraph)
    inserts: RDFGraph = Field(default_factory=RDFGraph)
    #: The action each listed term ends with, defaults included.
    decided: dict[str, ResolvedAction] = Field(default_factory=dict)
    #: Model actions that named no listed term or failed validation.
    ignored: list[tuple[str, str]] = Field(default_factory=list)

    def to_graph_update(self) -> GraphUpdate:
        """The patch as a delete-then-insert :class:`GraphUpdate`."""
        operations: list[TripleOp] = []
        if len(self.deletes) > 0:
            operations.append(TripleOp(type="delete", graph=self.deletes))
        if len(self.inserts) > 0:
            operations.append(TripleOp(type="insert", graph=self.inserts))
        return GraphUpdate(triple_operations=operations)

    def apply_to(self, graph: Graph) -> int:
        """Apply in place, deletes first; returns the triples changed."""
        changed = 0
        for triple in self.deletes:
            if triple in graph:
                graph.remove(triple)
                changed += 1
        for triple in self.inserts:
            if triple not in graph:
                graph.add(triple)
                changed += 1
        return changed

    def counts(self) -> dict[str, int]:
        """Decided actions by kind."""
        return dict(Counter(action.action for action in self.decided.values()))


def _target_role_ok(
    target: str,
    role: str,
    *,
    minted: Mapping[str, str],
    catalog_index: CatalogSurfaceIndex,
) -> bool:
    if target in minted:
        return minted[target] == role
    if target in catalog_index.terms:
        if role == "class":
            return target not in catalog_index.properties
        return target not in catalog_index.classes
    return well_known_vocabulary(target) is not None


def resolve_actions(
    actions: Iterable[OntologyTermAction],
    *,
    minted: Mapping[str, str],
    candidates: set[str],
    catalog_index: CatalogSurfaceIndex,
    prefixes: Mapping[str, str],
    reviewed: Iterable[str] | None = None,
) -> tuple[dict[str, ResolvedAction], list[tuple[str, str]]]:
    """Validate model actions and fill in defaults for every minted term.

    Args:
        actions: The model's actions, first one per term wins.
        minted: Every minted term the actions may name, with its role.
        candidates: Minted terms the plan did not name; their default is drop.
        catalog_index: Names and roles of every catalog term.
        prefixes: CURIE prefixes the prompt used.
        reviewed: The terms the model was asked about; an action on any other
            minted term is ignored. ``None`` accepts every minted term.

    Returns:
        The decided action per minted term, and the refused actions with a
        reason.
    """
    by_local: dict[str, list[str]] = {}
    for iri in minted:
        by_local.setdefault(_local_name(iri).casefold(), []).append(iri)

    def lookup(reference: str) -> str | None:
        iri = expand_reference(reference, prefixes)
        if iri is not None:
            return iri
        matches = by_local.get(reference.strip().casefold(), [])
        return matches[0] if len(matches) == 1 else None

    under_review = set(reviewed) if reviewed is not None else None
    ignored: list[tuple[str, str]] = []
    chosen: dict[str, ResolvedAction] = {}
    for action in actions:
        term = lookup(action.term)
        if term is None or term not in minted:
            ignored.append((action.term, "not a listed minted term"))
            continue
        if under_review is not None and term not in under_review:
            ignored.append((action.term, "not up for review"))
            continue
        if term in chosen:
            ignored.append((action.term, "second action for the term"))
            continue
        role = minted[term]
        target: str | None = None
        if action.action in ("merge_into", "generalise_to"):
            target = lookup(action.target) if action.target else None
            if target is None and action.target:
                # A catalog term named by its bare label or local name.
                target = _catalog_by_name(action.target, catalog_index)
            if (
                target is None
                or target == term
                or not _target_role_ok(
                    target, role, minted=minted, catalog_index=catalog_index
                )
            ):
                ignored.append((action.term, f"invalid target for {action.action}"))
                continue
        if action.action == "demote_to_property_value" and role != "class":
            ignored.append((action.term, "only a class can be demoted"))
            continue
        chosen[term] = ResolvedAction(term=term, action=action.action, target=target)

    decided: dict[str, ResolvedAction] = {}
    for term in minted:
        decided[term] = chosen.get(term) or ResolvedAction(
            term=term, action="drop" if term in candidates else "keep"
        )

    # A merge must end at a term that survives: follow chains, and fall back
    # to the default when one ends at a dropped or demoted term or loops.
    for term, resolved in list(decided.items()):
        if resolved.action != "merge_into":
            continue
        seen = {term}
        target = resolved.target
        while target in decided and decided[target].action == "merge_into":
            if target in seen:
                target = None
                break
            seen.add(target)
            target = decided[target].target
        if target is None or (
            target in decided
            and decided[target].action in ("drop", "demote_to_property_value")
        ):
            ignored.append((term, "merge target does not survive"))
            decided[term] = ResolvedAction(
                term=term, action="drop" if term in candidates else "keep"
            )
        else:
            decided[term] = ResolvedAction(
                term=term, action="merge_into", target=target
            )
    return decided, ignored


def _catalog_by_name(name: str, catalog_index: CatalogSurfaceIndex) -> str | None:
    key = name.strip().casefold()
    for table in (catalog_index.by_label, catalog_index.by_local):
        found = table.get(" ".join(key.split()), set())
        if len(found) == 1:
            return next(iter(found))
    return None


def compile_actions(
    decided: Mapping[str, ResolvedAction],
    graph: Graph,
    *,
    minted: Mapping[str, str],
) -> CompiledActions:
    """The deletes and inserts the decided actions make on ``graph``.

    - ``merge_into T``: the term's own triples go (``T`` carries its
      definition); every other triple naming it names ``T`` instead.
    - ``generalise_to T``: its named parents are replaced by ``T``.
    - ``demote_to_property_value``: its class typing and named superclasses
      go; it becomes a ``skos:Concept`` individual whose former parents are
      ``skos:broader``. Subclasses and domain/range uses move to its former
      parent, or go when it had none.
    - ``drop``: every triple naming it goes.
    """
    result = CompiledActions(decided=dict(decided))
    merge = {
        URIRef(term): URIRef(action.target)
        for term, action in decided.items()
        if action.action == "merge_into" and action.target
    }
    drop = {URIRef(term) for term, a in decided.items() if a.action == "drop"}
    demote = {
        URIRef(term)
        for term, a in decided.items()
        if a.action == "demote_to_property_value"
    }
    generalise = {
        URIRef(term): URIRef(a.target)
        for term, a in decided.items()
        if a.action == "generalise_to" and a.target
    }
    demoted_parents: dict[URIRef, list[URIRef]] = {
        term: sorted(
            (p for p in graph.objects(term, RDFS.subClassOf) if isinstance(p, URIRef)),
            key=str,
        )
        for term in demote
    }

    def rewrite(node: Node) -> Node:
        return merge.get(node, node) if isinstance(node, URIRef) else node

    for triple in list(graph):
        subject, predicate, obj = triple
        if subject in drop or predicate in drop or obj in drop:
            result.deletes.add(triple)
            continue
        if subject in merge:
            result.deletes.add(triple)
            continue
        if subject in generalise and predicate in (RDFS.subClassOf, RDFS.subPropertyOf):
            if isinstance(obj, URIRef):
                result.deletes.add(triple)
                continue
        if subject in demote:
            if (predicate == RDF.type and obj in (OWL.Class, RDFS.Class)) or (
                predicate == RDFS.subClassOf and isinstance(obj, URIRef)
            ):
                result.deletes.add(triple)
                continue
        if (
            isinstance(obj, URIRef)
            and obj in demote
            and predicate in (RDFS.subClassOf, RDFS.domain, RDFS.range)
        ):
            result.deletes.add(triple)
            for parent in demoted_parents[obj]:
                result.inserts.add((subject, predicate, parent))
            continue
        replaced = (rewrite(subject), rewrite(predicate), rewrite(obj))
        if replaced != triple:
            result.deletes.add(triple)
            result.inserts.add(replaced)  # type: ignore[arg-type]

    for term, target in generalise.items():
        if (term, None, None) in graph:
            role = minted.get(str(term), "class")
            result.inserts.add((term, _PARENT_PREDICATE[role], rewrite(target)))
    for term in demote:
        if (term, None, None) not in graph:
            continue
        result.inserts.add((term, RDF.type, SKOS.Concept))
        result.inserts.add((term, RDF.type, OWL.NamedIndividual))
        for parent in demoted_parents[term]:
            result.inserts.add((term, SKOS.broader, parent))
    # A triple both deleted and re-inserted is unchanged.
    for triple in list(result.inserts):
        if triple in result.deletes:
            result.deletes.remove(triple)
            result.inserts.remove(triple)
    return result


def structural_flags(
    minted: Mapping[str, str],
    graphs: Sequence[RDFGraph],
    catalog_index: CatalogSurfaceIndex,
) -> dict[str, set[str]]:
    """Minted terms whose shape says a decision is needed, with the reasons.

    - ``variant``: its name is another term's with one modifier added, or the
      same words, among minted and catalog terms of the same role;
    - ``value_set``: one of three or more minted sibling classes whose labels
      differ in one word;
    - ``catalog_duplicate``: its label or local name names exactly one catalog
      term of compatible role.

    Model-free; a term with no flag needs no model call.
    """
    flags: dict[str, set[str]] = {}

    def flag(iri: str, reason: str) -> None:
        if iri in minted:
            flags.setdefault(iri, set()).add(reason)

    pool: dict[str, str] = dict(minted)
    for iri in catalog_index.classes:
        pool.setdefault(iri, "class")
    for iri in catalog_index.properties:
        pool.setdefault(iri, "property")
    for broad, narrow in variant_term_pairs(minted, pool):
        flag(broad, "variant")
        flag(narrow, "variant")

    children: dict[str, list[tuple[str, str]]] = {}
    for iri, role in minted.items():
        if role != "class":
            continue
        subject = URIRef(iri)
        label = first_label(subject, *graphs) or " ".join(name_tokens(iri))
        for graph in graphs:
            for parent in graph.objects(subject, RDFS.subClassOf):
                if isinstance(parent, URIRef):
                    children.setdefault(str(parent), []).append((iri, label))
    for family in value_set_families(children):
        for iri, _ in family.members:
            flag(iri, "value_set")

    for iri, role in minted.items():
        for graph in graphs:
            if (URIRef(iri), None, None) not in graph:
                continue
            if match_catalog_term(URIRef(iri), graph, catalog_index, role) is not None:
                flag(iri, "catalog_duplicate")
            break
    return flags


__all__ = [
    "CompiledActions",
    "ResolvedAction",
    "compile_actions",
    "resolve_actions",
    "structural_flags",
]
