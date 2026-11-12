"""The well-known vocabularies a facts render may use beside the catalog.

Every facts vocabulary mode admits them: the W3C and DCMI core, FOAF,
schema.org and QUDT. A term of one of them must exist in it. Term lists come
from rdflib's bundled namespace definitions; RDF 1.2 terms rdflib does not
list yet are added here. The QUDT schema and vCard lists ship as data in
``vocabularies/``. QUDT units and quantity kinds are individuals in their
thousands, so those two namespaces are admitted without a list.

Some vocabularies also name the classes of structured values (a quantity, an
amount): a node of such a class belongs to the statement that carries it.
"""

from __future__ import annotations

import re
from functools import cache
from importlib import resources
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
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
        value_classes: Local names of its classes of structured values.
        datatype_ranges: Local name of a property -> local name of the XSD
            datatype its published ``rdfs:range`` declares, for the date and
            time properties the facts repairs check literals against.
    """

    model_config = ConfigDict(frozen=True)

    prefix: str
    title: str
    namespaces: tuple[str, ...]
    terms: frozenset[str] | None
    value_classes: frozenset[str] = frozenset()
    datatype_ranges: dict[str, str] = Field(default_factory=dict)

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
    value_classes: tuple[str, ...] = (),
    datatype_ranges: dict[str, str] | None = None,
) -> WellKnownVocabulary:
    return WellKnownVocabulary(
        prefix=prefix,
        title=title,
        namespaces=(str(namespace._NS), *also),
        terms=_defined_terms(namespace, *extra),
        value_classes=frozenset(value_classes),
        datatype_ranges=datatype_ranges or {},
    )


def _shipped_terms(name: str) -> frozenset[str]:
    """Local names listed in ``vocabularies/<name>``, one per line, ``#`` comments."""
    text = resources.files(__package__).joinpath("vocabularies", name).read_text()
    return frozenset(
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def _listed(
    prefix: str,
    title: str,
    namespace: str,
    terms: str,
    *,
    value_classes: tuple[str, ...] = (),
) -> WellKnownVocabulary:
    return WellKnownVocabulary(
        prefix=prefix,
        title=title,
        namespaces=(namespace,),
        terms=_shipped_terms(terms),
        value_classes=frozenset(value_classes),
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
    _vocabulary(
        "prov",
        "PROV-O",
        PROV,
        datatype_ranges={
            name: "dateTime"
            for name in (
                "atTime",
                "startedAtTime",
                "endedAtTime",
                "generatedAtTime",
                "invalidatedAtTime",
            )
        },
    ),
    _vocabulary("sosa", "SOSA", SOSA, datatype_ranges={"resultTime": "dateTime"}),
    _vocabulary("ssn", "SSN", SSN),
    _vocabulary(
        "time",
        "OWL-Time",
        TIME,
        datatype_ranges={
            "inXSDDate": "date",
            "inXSDDateTime": "dateTime",
            "inXSDDateTimeStamp": "dateTimeStamp",
            "inXSDgYear": "gYear",
            "inXSDgYearMonth": "gYearMonth",
        },
    ),
    _vocabulary("csvw", "CSVW", CSVW),
    _vocabulary("dcat", "DCAT", DCAT),
    _vocabulary("org", "the Organization Ontology", ORG),
    _listed("vcard", "vCard", "http://www.w3.org/2006/vcard/ns#", "vcard.txt"),
    _vocabulary("foaf", "FOAF", FOAF),
    _vocabulary(
        "schema",
        "schema.org",
        SDO,
        also=("http://schema.org/",),
        value_classes=(
            "StructuredValue",
            "QuantitativeValue",
            "PropertyValue",
            "MonetaryAmount",
        ),
    ),
    _listed(
        "qudt",
        "QUDT",
        "http://qudt.org/schema/qudt/",
        "qudt_schema.txt",
        value_classes=("QuantityValue",),
    ),
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


#: Well-known datatypes outside XSD; their local names are not capitalised.
_DATATYPE_TERMS: frozenset[str] = frozenset(
    {
        str(RDF.langString),
        f"{RDF}dirLangString",
        str(RDF.HTML),
        str(RDF.XMLLiteral),
        str(RDF.JSON),
        str(RDF.PlainLiteral),
        str(OWL.real),
        str(OWL.rational),
    }
)

#: Well-known individuals, neither classes nor properties.
_INDIVIDUAL_TERMS: frozenset[str] = frozenset({str(RDF.nil)})

#: Vocabularies whose terms are individuals (units, quantity kinds).
_INDIVIDUAL_VOCABULARIES: frozenset[str] = frozenset({"unit", "quantitykind"})

WellKnownRole = Literal["class", "property", "datatype"]


def well_known_role(iri: str) -> WellKnownRole | None:
    """The kind of a well-known term, by the naming its vocabulary follows.

    The well-known vocabularies capitalise their classes and not their
    properties; XSD's terms and a few RDF and OWL literal types are
    datatypes.

    Returns:
        ``None`` for an IRI outside the well-known vocabularies, a term its
        vocabulary does not have, or an individual (QUDT units and quantity
        kinds, ``rdf:nil``).
    """
    namespace, local_name = _split(iri)
    vocabulary = _index().get(namespace)
    if (
        vocabulary is None
        or not local_name
        or vocabulary.prefix in _INDIVIDUAL_VOCABULARIES
        or not vocabulary.has_term(local_name)
        or iri in _INDIVIDUAL_TERMS
    ):
        return None
    if vocabulary.prefix == "xsd" or iri in _DATATYPE_TERMS:
        return "datatype"
    return "class" if local_name[0].isupper() else "property"


def well_known_terms(namespace: str) -> set[str]:
    """The term IRIs of the vocabulary at ``namespace``, spelled in it.

    Empty for a namespace that is not well-known or is admitted by namespace.
    """
    vocabulary = _index().get(namespace)
    if vocabulary is None or vocabulary.terms is None:
        return set()
    return {namespace + name for name in vocabulary.terms}


@cache
def well_known_value_classes() -> frozenset[str]:
    """The classes of structured values the well-known vocabularies name, as IRIs.

    A node of one of them (or of a subclass) is a value: it belongs to the
    statement that carries it and is never a fact on its own.
    """
    return frozenset(
        namespace + name
        for vocabulary in WELL_KNOWN_VOCABULARIES
        for namespace in vocabulary.namespaces
        for name in vocabulary.value_classes
    )


@cache
def well_known_datatype_ranges() -> dict[str, str]:
    """Property IRI -> XSD datatype IRI, as the well-known vocabularies publish them.

    Only the date and time properties are listed. Treat as read-only.
    """
    return {
        namespace + name: str(XSD) + datatype
        for vocabulary in WELL_KNOWN_VOCABULARIES
        for namespace in vocabulary.namespaces
        for name, datatype in vocabulary.datatype_ranges.items()
    }


def well_known_prompt_list() -> str:
    """The vocabularies as the facts prompt names them: ``prefix:`` and IRI."""
    return ", ".join(
        f"`{vocabulary.prefix}:` <{vocabulary.namespaces[0]}>"
        for vocabulary in WELL_KNOWN_VOCABULARIES
    )


__all__ = [
    "WELL_KNOWN_VOCABULARIES",
    "WellKnownRole",
    "WellKnownVocabulary",
    "is_well_known_term",
    "well_known_role",
    "well_known_namespaces",
    "well_known_prompt_list",
    "well_known_terms",
    "well_known_datatype_ranges",
    "well_known_value_classes",
    "well_known_vocabulary",
]
