"""Whether a predicate or class IRI is acceptable, decided without messages."""

from __future__ import annotations

from collections.abc import Set as AbstractSet
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel

from ontocast.onto.well_known import well_known_vocabulary
from ontocast.tool.facts_validation.terms import (
    _FORBIDDEN_NAMESPACES,
    _local_name,
    _namespace_of,
)


class TermVerdictKind(StrEnum):
    """Why a term is not acceptable."""

    PLACEHOLDER_NAMESPACE = "placeholder_namespace"
    WELL_KNOWN_MISSING = "well_known_missing"
    CLOSED_NAMESPACE_MISSING = "closed_namespace_missing"
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
) -> TermVerdict | None:
    """Judge one predicate or class IRI against the catalog.

    Placeholder namespaces are rejected first. A catalog, known or fallback
    term passes, and so does a term of a well-known vocabulary that has it.
    A term the vocabulary lacks is rejected even with an empty catalog; the
    remaining verdicts need a catalog and leave standard namespaces alone.

    Args:
        text: The IRI.
        catalog_terms: Terms of the retrieved catalog.
        known_terms: Terms of the whole catalog.
        fallback_terms: Sanctioned fallback vocabulary that always passes.
        declared_namespaces: Namespaces the catalog closes.
        catalog_namespaces: Namespaces the catalog declares or borrows from.
        standard_namespaces: Namespace prefixes of standard vocabularies.
        vocabulary: ``strict`` closes every namespace outside the catalog's.

    Returns:
        The verdict, or ``None`` when the term is acceptable.
    """
    if text.startswith(_FORBIDDEN_NAMESPACES):
        return TermVerdict(kind=TermVerdictKind.PLACEHOLDER_NAMESPACE, term=text)
    if text in catalog_terms or text in known_terms or text in fallback_terms:
        return None
    known_vocabulary = well_known_vocabulary(text)
    if known_vocabulary is not None and not known_vocabulary.has_term(
        _local_name(text)
    ):
        return TermVerdict(
            kind=TermVerdictKind.WELL_KNOWN_MISSING,
            term=text,
            vocabulary_title=known_vocabulary.title,
        )
    namespace = _namespace_of(text)
    if not catalog_terms or namespace.startswith(standard_namespaces):
        return None
    if namespace in declared_namespaces:
        return TermVerdict(kind=TermVerdictKind.CLOSED_NAMESPACE_MISSING, term=text)
    if vocabulary == "strict" and namespace not in catalog_namespaces:
        return TermVerdict(kind=TermVerdictKind.NON_CATALOG_STRICT, term=text)
    return None
