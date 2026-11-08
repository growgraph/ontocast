"""Model-free measures of an ontology's shape, and a whole-catalog name index.

The shape measures answer one question about a newly minted class: would a
second document in the same domain reuse it? Three signals say it would not:
a class sitting on a long chain of named superclasses, a family of sibling
classes that differ by one word (a value set modelled as subclasses), and a
leaf whose label names one document's case (a number, a chemical formula, a
long noun phrase).

The surface index maps names to catalog IRIs across every catalog ontology,
not the snapshot a unit was shown: a unit that re-mints a term the retrieval
did not surface can only be caught against the whole catalog.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, Field
from rdflib import OWL, RDF, RDFS, SKOS, BNode, URIRef
from rdflib import Literal as RDFLiteral
from rdflib.graph import Graph

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.well_known import well_known_vocabulary
from ontocast.tool.facts_validation.terms import _catalog_term_roles, _local_name

#: Predicates whose literal object names a term.
SURFACE_PREDICATES = (RDFS.label, SKOS.prefLabel, SKOS.notation)

#: Characters a label is split on. A hyphen joins one word
#: (``zero-dimensional``), so it does not split.
_TOKEN_SPLIT_RE = re.compile(r"[\s_/]+")

#: A token shaped like a chemical formula: two or more element symbols, each an
#: upper-case letter with an optional lower-case one and an optional count.
_FORMULA_TOKEN_RE = re.compile(r"^(?:[A-Z][a-z]?\d*){2,}$")

#: Labels with at least this many tokens name a case rather than a kind.
LONG_LABEL_TOKENS = 4

SpecificLabelReason = Literal["digit", "formula", "long"]


def is_depth_root(class_iri: URIRef, graph: Graph) -> bool:
    """Whether ``class_iri`` ends a named-superclass chain.

    ``owl:Thing``, any term of a well-known vocabulary, and a class with no
    named superclass in ``graph``.
    """
    if class_iri == OWL.Thing or well_known_vocabulary(str(class_iri)) is not None:
        return True
    return not any(
        isinstance(parent, URIRef)
        for parent in graph.objects(class_iri, RDFS.subClassOf)
    )


def named_class_depth(
    class_iri: URIRef,
    graph: Graph,
    memo: dict[URIRef, tuple[int, list[URIRef]]] | None = None,
) -> tuple[int, list[URIRef]]:
    """The longest chain of named superclasses from ``class_iri`` to a root.

    Only named ``rdfs:subClassOf`` objects are steps: a restriction or an
    ``owl:equivalentClass`` definition constrains a class without placing it
    deeper. A class already on the current path ends the walk, so a cycle
    terminates (reporting it is the subclass-cycle rule's job).

    Args:
        class_iri: The class to measure.
        graph: Snapshot plus delta.
        memo: Shared across calls on the same graph to make a batch linear.

    Returns:
        The number of steps (a root is 0) and the chain, ``class_iri`` first.
    """
    cache = memo if memo is not None else {}

    def walk(node: URIRef, path: frozenset[URIRef]) -> tuple[int, list[URIRef]]:
        cached = cache.get(node)
        if cached is not None:
            return cached
        if is_depth_root(node, graph):
            result = (0, [node])
            cache[node] = result
            return result
        best: tuple[int, list[URIRef]] = (0, [node])
        cyclic = False
        for parent in sorted(
            (p for p in graph.objects(node, RDFS.subClassOf) if isinstance(p, URIRef)),
            key=str,
        ):
            if parent in path or parent == node:
                cyclic = True
                continue
            depth, chain = walk(parent, path | {node})
            if depth + 1 > best[0]:
                best = (depth + 1, [node, *chain])
        if not cyclic:
            # A result reached through a cut cycle depends on where the walk
            # entered it; memoising it would leak that order into other calls.
            cache[node] = best
        return best

    return walk(class_iri, frozenset())


def label_tokens(text: str) -> list[str]:
    """Lower-case tokens of a label, split on whitespace, ``_`` and ``/``."""
    return [token for token in _TOKEN_SPLIT_RE.split(text.casefold()) if token]


def specific_label_reason(label: str) -> SpecificLabelReason | None:
    """Why ``label`` names one document's case rather than a reusable kind.

    ``digit``: it carries a number (``Sample 3``). ``formula``: a token is a
    chemical formula (``CsPbBr3 film``; an all-capitals acronym such as
    ``XRD`` is not one). ``long``: four or more tokens.
    """
    if any(character.isdigit() for character in label):
        return "digit"
    for token in label.split():
        if _FORMULA_TOKEN_RE.match(token) and any(c.islower() for c in token):
            return "formula"
    if len(label_tokens(label)) >= LONG_LABEL_TOKENS:
        return "long"
    return None


class ValueSetFamily(BaseModel):
    """Sibling classes whose labels differ in one token only."""

    parent: str
    #: The shared label with the varying token as ``*`` (``* perovskite``).
    template: str
    #: ``(iri, label)`` pairs, sorted by IRI.
    members: list[tuple[str, str]]


def value_set_families(
    parent_to_children: Mapping[str, Sequence[tuple[str, str]]],
    min_size: int = 3,
) -> list[ValueSetFamily]:
    """Groups of at least ``min_size`` siblings one token apart.

    Each child is matched against every template obtained by replacing one of
    its tokens with ``*``; siblings sharing a template form a family. A child
    belongs to at most one family: the largest is taken first, ties broken by
    template text, so the result does not depend on input order.

    Args:
        parent_to_children: Parent IRI to its ``(child iri, label)`` pairs.
        min_size: Smallest family reported.
    """
    families: list[ValueSetFamily] = []
    for parent in sorted(parent_to_children):
        children = sorted(set(parent_to_children[parent]))
        groups: dict[tuple[str, ...], set[tuple[str, str]]] = defaultdict(set)
        for iri, label in children:
            tokens = label_tokens(label)
            if len(tokens) < 2:
                continue
            for position in range(len(tokens)):
                template = (*tokens[:position], "*", *tokens[position + 1 :])
                groups[template].add((iri, label))
        taken: set[str] = set()
        ranked = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
        for template, members in ranked:
            free = sorted((iri, label) for iri, label in members if iri not in taken)
            if len(free) < min_size:
                continue
            taken.update(iri for iri, _ in free)
            families.append(
                ValueSetFamily(parent=parent, template=" ".join(template), members=free)
            )
    return families


def has_own_structure(class_iri: URIRef, graph: Graph) -> bool:
    """Whether ``class_iri`` carries something a property value could not.

    A subclass, a property whose domain it is, a restriction it is a subclass
    of, or an ``owl:equivalentClass`` definition. A sibling with any of these is
    a modelled kind, not a value in disguise.
    """
    if any(True for _ in graph.subjects(RDFS.subClassOf, class_iri)):
        return True
    if any(True for _ in graph.subjects(RDFS.domain, class_iri)):
        return True
    if any(isinstance(o, BNode) for o in graph.objects(class_iri, RDFS.subClassOf)):
        return True
    return any(True for _ in graph.objects(class_iri, OWL.equivalentClass))


def minted_class_iris(*graphs: Graph) -> set[str]:
    """Subjects the graphs declare as classes or place in a hierarchy."""
    classes: set[str] = set()
    for graph in graphs:
        for class_type in (OWL.Class, RDFS.Class):
            for subject in graph.subjects(RDF.type, class_type):
                if isinstance(subject, URIRef):
                    classes.add(str(subject))
        for subject in graph.subjects(RDFS.subClassOf, None):
            if isinstance(subject, URIRef):
                classes.add(str(subject))
    return classes


def first_label(subject: URIRef, *graphs: Graph) -> str | None:
    """The first ``rdfs:label`` or ``skos:prefLabel`` of ``subject``, sorted."""
    for graph in graphs:
        for predicate in (RDFS.label, SKOS.prefLabel):
            values = sorted(
                str(value)
                for value in graph.objects(subject, predicate)
                if isinstance(value, RDFLiteral) and str(value).strip()
            )
            if values:
                return values[0].strip()
    return None


def normalize_surface(text: str) -> str:
    """Case-folded, whitespace-collapsed form of a name."""
    return " ".join(text.casefold().split())


#: A case change inside a local name: ``hasPLRedShift`` splits before ``PL``,
#: ``Red`` and ``Shift``.
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def name_tokens(iri: str) -> tuple[str, ...]:
    """The local name split at case changes and separators, case-folded."""
    return tuple(label_tokens(_CAMEL_RE.sub(" ", _local_name(iri))))


def _narrows(broad: tuple[str, ...], narrow: tuple[str, ...]) -> bool:
    """``narrow`` is ``broad`` with one modifier added: a word before the last
    one, or a prefix on one word. The last word, the head, stays: a word added
    after it names something else (``opticalMode`` -> ``opticalModeEnergy``)."""
    if not broad or not narrow:
        return False
    if narrow[-1] != broad[-1] and not (
        len(narrow) == len(broad) and narrow[-1].endswith(broad[-1])
    ):
        return False
    if len(narrow) == len(broad) + 1:
        return any(narrow[:i] + narrow[i + 1 :] == broad for i in range(len(broad)))
    if len(narrow) != len(broad):
        return False
    differing = [
        i for i, (a, b) in enumerate(zip(broad, narrow, strict=True)) if a != b
    ]
    if len(differing) != 1:
        return False
    a, b = broad[differing[0]], narrow[differing[0]]
    return len(b) > len(a) and b.endswith(a)


def variant_term_pairs(
    terms: Mapping[str, str], pool: Mapping[str, str]
) -> list[tuple[str, str]]:
    """``(broader, narrower)`` pairs of near-identical names, ``narrower`` in ``terms``.

    Two terms of the same role (``class`` / ``property``) pair when their names
    have the same tokens (then in IRI order), or when the narrower adds one
    modifier to the broader (``hasOpticalCavity`` -> ``hasOpticalMicrocavity``,
    ``hasPhase`` -> ``hasInitialPhase``).

    Args:
        terms: Candidate narrower terms, IRI to role.
        pool: Every term a narrower one may vary, IRI to role; ``terms`` are
            included by the caller when pairs among them count.
    """
    tokens = {iri: name_tokens(iri) for iri in {*terms, *pool}}
    pairs: set[tuple[str, str]] = set()
    for narrow in sorted(terms):
        role = terms[narrow]
        for broad in sorted(pool):
            if broad == narrow or pool[broad] != role or not tokens[broad]:
                continue
            if tokens[broad] == tokens[narrow]:
                pairs.add((min(broad, narrow), max(broad, narrow)))
            elif _narrows(tokens[broad], tokens[narrow]):
                pairs.add((broad, narrow))
    return sorted(pairs)


class CatalogSurfaceIndex(BaseModel):
    """Names of every catalog term, case-insensitive, by label and local name."""

    by_label: dict[str, set[str]] = Field(default_factory=dict)
    by_local: dict[str, set[str]] = Field(default_factory=dict)
    #: Every named subject of the catalog.
    terms: set[str] = Field(default_factory=set)
    properties: set[str] = Field(default_factory=set)
    classes: set[str] = Field(default_factory=set)

    def is_empty(self) -> bool:
        """Whether the index holds no term."""
        return not self.terms


def build_catalog_surface_index(graphs: Iterable[RDFGraph]) -> CatalogSurfaceIndex:
    """Index the labels, notations and local names of every subject."""
    index = CatalogSurfaceIndex()
    for graph in graphs:
        for subject in graph.subjects():
            if not isinstance(subject, URIRef):
                continue
            iri = str(subject)
            if iri in index.terms:
                continue
            index.terms.add(iri)
            local = _local_name(iri)
            if local and local != iri:
                index.by_local.setdefault(local.casefold(), set()).add(iri)
        for predicate in SURFACE_PREDICATES:
            for subject, value in graph.subject_objects(predicate):
                if not isinstance(subject, URIRef) or not isinstance(value, RDFLiteral):
                    continue
                key = normalize_surface(str(value))
                if key:
                    index.by_label.setdefault(key, set()).add(str(subject))
        properties, classes = _catalog_term_roles(graph)
        index.properties |= properties
        index.classes |= classes
    index.classes -= index.properties & index.classes
    return index


def resolve_unique(
    index: CatalogSurfaceIndex,
    *,
    label: str | None = None,
    local: str | None = None,
) -> tuple[str, Literal["label", "local_name"]] | None:
    """The single catalog IRI a label, else a local name, names.

    The label key is tried first. A name claimed by more than one IRI is
    refused: an ambiguous name is not a duplicate of any one term.

    Returns:
        The IRI and the key that matched, or ``None``.
    """
    if label:
        candidates = index.by_label.get(normalize_surface(label), set())
        if len(candidates) == 1:
            return next(iter(candidates)), "label"
    if local:
        candidates = index.by_local.get(local.casefold(), set())
        if len(candidates) == 1:
            return next(iter(candidates)), "local_name"
    return None


def role_compatible(
    minted_role: str, catalog_iri: str, index: CatalogSurfaceIndex
) -> bool:
    """True when matching would not cross the class/property divide."""
    if minted_role == "property":
        return catalog_iri in index.properties or catalog_iri not in index.classes
    if minted_role == "class":
        return catalog_iri in index.classes or catalog_iri not in index.properties
    return catalog_iri not in index.properties and catalog_iri not in index.classes


__all__ = [
    "CatalogSurfaceIndex",
    "LONG_LABEL_TOKENS",
    "SURFACE_PREDICATES",
    "ValueSetFamily",
    "build_catalog_surface_index",
    "first_label",
    "has_own_structure",
    "is_depth_root",
    "label_tokens",
    "minted_class_iris",
    "name_tokens",
    "named_class_depth",
    "normalize_surface",
    "resolve_unique",
    "role_compatible",
    "specific_label_reason",
    "value_set_families",
    "variant_term_pairs",
]
