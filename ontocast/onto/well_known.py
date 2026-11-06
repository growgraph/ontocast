"""The well-known vocabularies a facts render may use beside the catalog.

Every facts vocabulary mode admits them: the W3C and DCMI core, FOAF,
schema.org and QUDT. A term of one of them must exist in it. Term lists come
from rdflib's bundled namespace definitions; RDF 1.2 terms rdflib does not
list yet are added here. vCard and the QUDT vocabularies have no bundled list
and are admitted by namespace.
"""

from __future__ import annotations

import re
from functools import cache

from pydantic import BaseModel, ConfigDict
from rdflib.namespace import (
    CSVW,
    DC,
    DCAT,
    DCTERMS,
    FOAF,
    ORG,
    OWL,
    PROV,
    RDF,
    RDFS,
    SDO,
    SKOS,
    SOSA,
    SSN,
    TIME,
    XSD,
    DefinedNamespace,
)

_RDF_12_TERMS = ("reifies", "dirLangString", "CompoundLiteral", "language", "direction")
_CONTAINER_MEMBER = re.compile(r"_[1-9][0-9]*")


class WellKnownVocabulary(BaseModel):
    """One well-known vocabulary.

    Attributes:
        prefix: The prefix the prompt shows for it.
        title: Its name, for messages.
        namespaces: Its namespace IRIs; the first is canonical.
        terms: Local names of its terms, or ``None`` when it is admitted by
            namespace (no term list is bundled).
    """

    model_config = ConfigDict(frozen=True)

    prefix: str
    title: str
    namespaces: tuple[str, ...]
    terms: frozenset[str] | None

    def has_term(self, local_name: str) -> bool:
        """Whether ``local_name`` is a term of this vocabulary."""
        if self.terms is None:
            return True
        if self.prefix == "rdf" and _CONTAINER_MEMBER.fullmatch(local_name):
            return True
        return local_name in self.terms


def _defined_terms(namespace: type[DefinedNamespace], *extra: str) -> frozenset[str]:
    names = set(namespace.__annotations__) | set(getattr(namespace, "_extras", ()))
    return frozenset(name for name in names if not name.startswith("_")) | set(extra)


def _vocabulary(
    prefix: str,
    title: str,
    namespace: type[DefinedNamespace],
    *,
    also: tuple[str, ...] = (),
    extra: tuple[str, ...] = (),
) -> WellKnownVocabulary:
    return WellKnownVocabulary(
        prefix=prefix,
        title=title,
        namespaces=(str(namespace._NS), *also),
        terms=_defined_terms(namespace, *extra),
    )


def _by_namespace(prefix: str, title: str, namespace: str) -> WellKnownVocabulary:
    return WellKnownVocabulary(
        prefix=prefix, title=title, namespaces=(namespace,), terms=None
    )


WELL_KNOWN_VOCABULARIES: tuple[WellKnownVocabulary, ...] = (
    _vocabulary("rdf", "RDF", RDF, extra=_RDF_12_TERMS),
    _vocabulary("rdfs", "RDF Schema", RDFS),
    _vocabulary("owl", "OWL", OWL),
    _vocabulary("xsd", "XML Schema datatypes", XSD),
    _vocabulary("skos", "SKOS", SKOS),
    _vocabulary("dcterms", "DCMI Metadata Terms", DCTERMS),
    _vocabulary("dc", "Dublin Core elements", DC),
    _vocabulary("prov", "PROV-O", PROV),
    _vocabulary("sosa", "SOSA", SOSA),
    _vocabulary("ssn", "SSN", SSN),
    _vocabulary("time", "OWL-Time", TIME),
    _vocabulary("csvw", "CSVW", CSVW),
    _vocabulary("dcat", "DCAT", DCAT),
    _vocabulary("org", "the Organization Ontology", ORG),
    _by_namespace("vcard", "vCard", "http://www.w3.org/2006/vcard/ns#"),
    _vocabulary("foaf", "FOAF", FOAF),
    _vocabulary("schema", "schema.org", SDO, also=("http://schema.org/",)),
    _by_namespace("qudt", "QUDT", "http://qudt.org/schema/qudt/"),
    _by_namespace("unit", "QUDT units", "http://qudt.org/vocab/unit/"),
    _by_namespace(
        "quantitykind", "QUDT quantity kinds", "http://qudt.org/vocab/quantitykind/"
    ),
)


@cache
def _index() -> dict[str, WellKnownVocabulary]:
    return {
        namespace: vocabulary
        for vocabulary in WELL_KNOWN_VOCABULARIES
        for namespace in vocabulary.namespaces
    }


def _split(iri: str) -> tuple[str, str]:
    for separator in ("#", "/"):
        head, sep, local = iri.rpartition(separator)
        if sep and local:
            return head + sep, local
    return iri, ""


def well_known_namespaces() -> tuple[str, ...]:
    """Every namespace IRI of the well-known vocabularies."""
    return tuple(_index())


def well_known_vocabulary(iri: str) -> WellKnownVocabulary | None:
    """The well-known vocabulary ``iri`` belongs to, if any."""
    return _index().get(_split(iri)[0])


def is_well_known_term(iri: str) -> bool | None:
    """Whether ``iri`` is a term of its well-known vocabulary.

    Returns:
        ``None`` when ``iri`` is in no well-known namespace; otherwise whether
        the vocabulary has the term (always true for one admitted by
        namespace).
    """
    namespace, local_name = _split(iri)
    vocabulary = _index().get(namespace)
    if vocabulary is None:
        return None
    return vocabulary.has_term(local_name)


def well_known_terms(namespace: str) -> set[str]:
    """The term IRIs of the vocabulary at ``namespace``, spelled in it.

    Empty for a namespace that is not well-known or is admitted by namespace.
    """
    vocabulary = _index().get(namespace)
    if vocabulary is None or vocabulary.terms is None:
        return set()
    return {namespace + name for name in vocabulary.terms}


def well_known_prompt_list() -> str:
    """The vocabularies as the facts prompt names them: ``prefix:`` and IRI."""
    return ", ".join(
        f"`{vocabulary.prefix}:` <{vocabulary.namespaces[0]}>"
        for vocabulary in WELL_KNOWN_VOCABULARIES
    )


__all__ = [
    "WELL_KNOWN_VOCABULARIES",
    "WellKnownVocabulary",
    "is_well_known_term",
    "well_known_namespaces",
    "well_known_prompt_list",
    "well_known_terms",
    "well_known_vocabulary",
]
