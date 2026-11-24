"""Whether an IRI the facts use is acceptable, decided without messages.

:func:`used_terms` yields every IRI a facts graph uses -- predicates, classes
and the IRIs in object position -- and :func:`term_verdict` judges one. The
unit validator, the document gate and ``check_facts`` read both, so the three
agree on what an unknown term is.
"""

from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Set as AbstractSet
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict
from rdflib import OWL, RDF, URIRef
from rdflib.term import Node

from ontocast.onto.scaffolding import FactScope
from ontocast.onto.well_known import well_known_vocabulary
from ontocast.tool.facts_validation.statements import (
    StatementContract,
    value_components,
)
from ontocast.tool.facts_validation.terms import (
    _FORBIDDEN_NAMESPACES,
    _local_name,
    _namespace_of,
)

TermPosition = Literal["predicate", "type", "object"]
"""Where a term is used: as a predicate, as an ``rdf:type`` object, or as
another IRI object (an individual)."""


class TermVerdictKind(StrEnum):
    """Why a term is not acceptable."""

    PLACEHOLDER_NAMESPACE = "placeholder_namespace"
    WELL_KNOWN_MISSING = "well_known_missing"
    CLOSED_NAMESPACE_MISSING = "closed_namespace_missing"
    #: An individual of a value whose property's range is a closed class,
    #: found neither in the catalog nor in a shipped term list.
    CLOSED_RANGE_MISSING = "closed_range_missing"
    NON_CATALOG_STRICT = "non_catalog_strict"


class TermVerdict(BaseModel):
    """A term judged unacceptable.

    Attributes:
        kind: The reason.
        term: The IRI judged.
        vocabulary_title: The well-known vocabulary's name, for
            ``WELL_KNOWN_MISSING``.
    """

    kind: TermVerdictKind
    term: str
    vocabulary_title: str | None = None


class UsedTerm(BaseModel):
    """One use of an IRI in a fact triple.

    Attributes:
        iri: The IRI.
        position: Where it is used.
        subject: The subject of the triple using it.
        closed_range: For an individual of a value, the closed classes of its
            property's range (:attr:`StatementContract.closed_ranges`).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    iri: str
    position: TermPosition
    subject: Node
    closed_range: frozenset[str] = frozenset()


#: Objects of these predicates are alias handles, not used terms.
_HANDLE_PREDICATES: frozenset[Node] = frozenset({OWL.sameAs})


def used_terms(
    scope: FactScope, contract: StatementContract | None = None
) -> Iterator[UsedTerm]:
    """Every IRI the fact triples of ``scope`` use, once per use.

    Predicates, ``rdf:type`` objects, and IRI objects of other predicates
    that are not instances or document nodes of the scope (an alias handle
    of ``owl:sameAs`` is none). With ``contract``, an individual of a value
    node (:func:`~ontocast.tool.facts_validation.statements.value_components`)
    carries the closed range of its property.
    """
    graph = scope.graph
    closed: dict[tuple[Node, str, Node], frozenset[str]] = {}
    if contract is not None and contract.value_classes:
        values = {
            node
            for node, cls in graph.subject_objects(RDF.type)
            if str(cls) in contract.value_classes and scope.holds(node)
        }
        for node in values:
            for component in value_components(graph, node, contract).individuals:
                if component.closed_range:
                    closed[(node, component.predicate, component.value)] = (
                        component.closed_range
                    )
    skipped = scope.instances | scope.documents
    for subject, predicate, obj in scope.triples():
        if isinstance(predicate, URIRef):
            yield UsedTerm(iri=str(predicate), position="predicate", subject=subject)
        if not isinstance(obj, URIRef):
            continue
        if predicate == RDF.type:
            yield UsedTerm(iri=str(obj), position="type", subject=subject)
            continue
        if predicate in _HANDLE_PREDICATES:
            continue
        # A fact instance is no term, except as a value's individual on a
        # property whose range is closed: there it must be a listed member.
        closed_range = closed.get((subject, str(predicate), obj), frozenset())
        if obj not in skipped or closed_range:
            yield UsedTerm(
                iri=str(obj),
                position="object",
                subject=subject,
                closed_range=closed_range,
            )


def term_verdict(
    text: str,
    *,
    catalog_terms: AbstractSet[str],
    known_terms: AbstractSet[str],
    fallback_terms: AbstractSet[str],
    declared_namespaces: AbstractSet[str],
    catalog_namespaces: AbstractSet[str],
    standard_namespaces: tuple[str, ...],
    vocabulary: Literal["strict", "open"],
    position: TermPosition = "predicate",
    closed_range: AbstractSet[str] = frozenset(),
) -> TermVerdict | None:
    """Judge one IRI against the catalog and the well-known vocabularies.

    A catalog, known or fallback term passes, and so does a term of a
    well-known vocabulary that has it; a term the vocabulary lacks is
    rejected even with an empty catalog. An individual whose property's range
    is a closed class (``closed_range``) must be one of those terms. The
    remaining verdicts need a catalog and leave standard namespaces alone: a
    member of a namespace the catalog declares terms in must be one of them,
    and a strict vocabulary closes every other namespace to predicates and
    classes. The placeholder namespace is rejected first for a predicate or
    a class; an individual elsewhere is judged by closed vocabularies only.

    Args:
        text: The IRI.
        catalog_terms: Terms of the retrieved catalog.
        known_terms: Terms of the whole catalog.
        fallback_terms: Sanctioned fallback vocabulary that always passes.
        declared_namespaces: Namespaces the catalog closes.
        catalog_namespaces: Namespaces the catalog declares or borrows from.
        standard_namespaces: Namespace prefixes of standard vocabularies.
        vocabulary: ``strict`` closes every namespace outside the catalog's.
        position: Where the IRI is used.
        closed_range: For an individual of a value, its property's closed
            range classes.

    Returns:
        The verdict, or ``None`` when the term is acceptable.
    """
    individual = position == "object"
    if not individual and text.startswith(_FORBIDDEN_NAMESPACES):
        return TermVerdict(kind=TermVerdictKind.PLACEHOLDER_NAMESPACE, term=text)
    if text in catalog_terms or text in known_terms or text in fallback_terms:
        return None
    known_vocabulary = well_known_vocabulary(text)
    if known_vocabulary is not None:
        if not known_vocabulary.has_term(_local_name(text)):
            return TermVerdict(
                kind=TermVerdictKind.WELL_KNOWN_MISSING,
                term=text,
                vocabulary_title=known_vocabulary.title,
            )
    elif closed_range:
        return TermVerdict(kind=TermVerdictKind.CLOSED_RANGE_MISSING, term=text)
    namespace = _namespace_of(text)
    if not catalog_terms or namespace.startswith(standard_namespaces):
        return None
    if namespace in declared_namespaces:
        return TermVerdict(kind=TermVerdictKind.CLOSED_NAMESPACE_MISSING, term=text)
    if (
        not individual
        and vocabulary == "strict"
        and namespace not in catalog_namespaces
    ):
        return TermVerdict(kind=TermVerdictKind.NON_CATALOG_STRICT, term=text)
    return None
