"""The well-known vocabularies a facts render may use beside the catalog.

Every facts vocabulary mode admits them: the W3C and DCMI core, FOAF,
schema.org and QUDT. A term of one of them must exist in it. Term lists come
from rdflib's bundled namespace definitions; RDF 1.2 terms rdflib does not
list yet are added here. The QUDT schema, QUDT unit and vCard lists ship as
data in ``vocabularies/``, with the dimension of each QUDT unit. QUDT
quantity kinds are admitted by namespace, without a list.

Some vocabularies also name the classes of structured values (a quantity, an
amount): a node of such a class belongs to the statement that carries it. And
some name statements (an observation) with the roles that link one to what it
is about: its subject, the property it reports, its result and its context.
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

QUDT_UNIT_NAMESPACE = "http://qudt.org/vocab/unit/"

StatementRoleName = Literal["subject", "property", "result", "context"]
"""A role a statement link plays: what the statement is about (``subject``),
the property it reports, its result, or its context (a time, a procedure)."""


class StatementRole(BaseModel):
    """The role a property plays for a statement.

    Attributes:
        role: The role.
        direction: ``forward`` runs from the statement, ``inverse`` to it.
        label: How the prompt names the property beside a statement class;
            ``None`` for one it does not offer.
        classes: Local names of the statement classes its published domain
            (``forward``) or range (``inverse``) lists; empty admits every
            statement class.
        range_class: For a ``forward`` link, the local name of the class its
            published range names (what a statement's property is an
            instance of), if any.
    """

    model_config = ConfigDict(frozen=True)

    role: StatementRoleName
    direction: Literal["forward", "inverse"] = "forward"
    label: str | None = None
    classes: frozenset[str] = frozenset()
    range_class: str | None = None


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
        datatype_ranges: Local name of a property -> local names of the XSD
            datatypes its published range admits: one for an ``rdfs:range``,
            several for a union of datatypes.
        temporal_properties: Local names of its properties whose object is a
            time, a temporal entity or a date literal; those in
            ``datatype_ranges`` count without being listed here.
        statement_classes: Local name -> required roles of its classes whose
            instances are statements about something else (an observation of
            a feature of interest).
        statement_roles: Local name -> the role a property plays for a
            statement, with its direction.
        class_ranges: Local name of an object property -> local name of the
            class its published ``rdfs:range`` declares.
        member_class: For a vocabulary of individuals with a term list, the
            IRI of the class its terms are instances of.
        member_dimensions: Local name of a term -> the IRI of its dimension:
            two terms of different dimensions measure different kinds of
            quantity. A term absent here has no known dimension.
        object_pairs: Local name of a datatype property -> local name of the
            object property the vocabulary gives for the same relation when
            the object is a node.
        anchor_properties: Local names of its properties that declare what a
            property node measures: its quantity kind, or a measurement unit
            that applies to it.
        measurement: Whether it models statements and measurements (an
            observation, a quantity, a measurement unit): the facts prompt
            names it only where its statement rules apply.
        superseded: Whether its publisher deprecated it in favour of another
            vocabulary: its terms are accepted where used, and the prompt
            never offers it.
    """

    model_config = ConfigDict(frozen=True)

    prefix: str
    title: str
    namespaces: tuple[str, ...]
    terms: frozenset[str] | None
    value_classes: frozenset[str] = frozenset()
    datatype_ranges: dict[str, frozenset[str]] = Field(default_factory=dict)
    temporal_properties: frozenset[str] = frozenset()
    statement_classes: dict[str, frozenset[StatementRoleName]] = Field(
        default_factory=dict
    )
    statement_roles: dict[str, StatementRole] = Field(default_factory=dict)
    class_ranges: dict[str, str] = Field(default_factory=dict)
    member_class: str | None = None
    member_dimensions: dict[str, str] = Field(default_factory=dict)
    object_pairs: dict[str, str] = Field(default_factory=dict)
    anchor_properties: frozenset[str] = frozenset()
    measurement: bool = False
    superseded: bool = False

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
    datatype_ranges: dict[str, frozenset[str]] | None = None,
    temporal_properties: tuple[str, ...] = (),
    statement_classes: dict[str, frozenset[StatementRoleName]] | None = None,
    statement_roles: dict[str, StatementRole] | None = None,
    object_pairs: dict[str, str] | None = None,
    measurement: bool = False,
) -> WellKnownVocabulary:
    return WellKnownVocabulary(
        prefix=prefix,
        title=title,
        measurement=measurement,
        namespaces=(str(namespace._NS), *also),
        terms=_defined_terms(namespace, *extra),
        value_classes=frozenset(value_classes),
        datatype_ranges=datatype_ranges or {},
        temporal_properties=frozenset(temporal_properties),
        statement_classes=statement_classes or {},
        statement_roles=statement_roles or {},
        object_pairs=object_pairs or {},
    )


def _shipped_lines(name: str) -> list[str]:
    """The lines of ``vocabularies/<name>``, without blanks and ``#`` comments."""
    text = resources.files(__package__).joinpath("vocabularies", name).read_text()
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _shipped_terms(name: str) -> frozenset[str]:
    """Local names listed in ``vocabularies/<name>``, one per line."""
    return frozenset(_shipped_lines(name))


def _shipped_dimensions(name: str, namespace: str) -> dict[str, str]:
    """``vocabularies/<name>``: a local name, a tab, its dimension's local name
    in ``namespace``; as local name -> dimension IRI."""
    pairs = (line.split("\t") for line in _shipped_lines(name))
    return {term.strip(): namespace + dimension.strip() for term, dimension in pairs}


def _listed(
    prefix: str,
    title: str,
    namespace: str,
    terms: str,
    *,
    value_classes: tuple[str, ...] = (),
    datatype_ranges: dict[str, frozenset[str]] | None = None,
    class_ranges: dict[str, str] | None = None,
    member_class: str | None = None,
    member_dimensions: dict[str, str] | None = None,
    anchor_properties: tuple[str, ...] = (),
    measurement: bool = False,
    superseded: bool = False,
) -> WellKnownVocabulary:
    return WellKnownVocabulary(
        prefix=prefix,
        title=title,
        measurement=measurement,
        superseded=superseded,
        namespaces=(namespace,),
        terms=_shipped_terms(terms),
        value_classes=frozenset(value_classes),
        datatype_ranges=datatype_ranges or {},
        class_ranges=class_ranges or {},
        member_class=member_class,
        member_dimensions=member_dimensions or {},
        anchor_properties=frozenset(anchor_properties),
    )


def _by_namespace(
    prefix: str, title: str, namespace: str, *, measurement: bool = False
) -> WellKnownVocabulary:
    return WellKnownVocabulary(
        prefix=prefix,
        title=title,
        namespaces=(namespace,),
        terms=None,
        measurement=measurement,
    )


#: The SOSA classes whose instances act on a feature of interest: the classes
#: SOSA lists as the domain of its subject, result and context properties.
_SOSA_ACTS = frozenset({"Observation", "Actuation", "Sampling"})

#: The XSD datatypes the QUDT shapes admit for a numeric value, as a union.
_QUDT_NUMBERS = frozenset({"decimal", "double", "float", "integer"})

WELL_KNOWN_VOCABULARIES: tuple[WellKnownVocabulary, ...] = (
    _vocabulary("rdf", "RDF", RDF, extra=_RDF_12_TERMS),
    _vocabulary("rdfs", "RDF Schema", RDFS),
    _vocabulary("owl", "OWL", OWL),
    _vocabulary("xsd", "XML Schema datatypes", XSD),
    _vocabulary("skos", "SKOS", SKOS),
    _vocabulary(
        "dcterms",
        "DCMI Metadata Terms",
        DCTERMS,
        temporal_properties=("date", "created", "issued", "modified"),
    ),
    _vocabulary("dc", "Dublin Core elements", DC),
    _vocabulary(
        "prov",
        "PROV-O",
        PROV,
        datatype_ranges={
            name: frozenset({"dateTime"})
            for name in (
                "atTime",
                "startedAtTime",
                "endedAtTime",
                "generatedAtTime",
                "invalidatedAtTime",
            )
        },
    ),
    _vocabulary(
        "sosa",
        "SOSA",
        SOSA,
        datatype_ranges={"resultTime": frozenset({"dateTime"})},
        temporal_properties=("phenomenonTime",),
        statement_classes={
            "Observation": frozenset({"subject", "property"}),
            "Actuation": frozenset({"subject", "property"}),
            "Sampling": frozenset({"subject"}),
        },
        statement_roles={
            "hasFeatureOfInterest": StatementRole(
                role="subject", label="has feature of interest", classes=_SOSA_ACTS
            ),
            "isFeatureOfInterestOf": StatementRole(
                role="subject", direction="inverse", classes=_SOSA_ACTS
            ),
            "observedProperty": StatementRole(
                role="property",
                label="observed property",
                classes=frozenset({"Observation"}),
                range_class="ObservableProperty",
            ),
            "actsOnProperty": StatementRole(
                role="property",
                classes=frozenset({"Actuation"}),
                range_class="ActuatableProperty",
            ),
            "hasResult": StatementRole(
                role="result", label="has result", classes=_SOSA_ACTS
            ),
            "hasSimpleResult": StatementRole(role="result", classes=_SOSA_ACTS),
            "isResultOf": StatementRole(
                role="result", direction="inverse", classes=_SOSA_ACTS
            ),
            "phenomenonTime": StatementRole(role="context", classes=_SOSA_ACTS),
            "resultTime": StatementRole(role="context", classes=_SOSA_ACTS),
            "usedProcedure": StatementRole(role="context", classes=_SOSA_ACTS),
        },
        object_pairs={"hasSimpleResult": "hasResult"},
        measurement=True,
    ),
    _vocabulary("ssn", "SSN", SSN, measurement=True),
    _vocabulary(
        "time",
        "OWL-Time",
        TIME,
        datatype_ranges={
            "inXSDDate": frozenset({"date"}),
            "inXSDDateTime": frozenset({"dateTime"}),
            "inXSDDateTimeStamp": frozenset({"dateTimeStamp"}),
            "inXSDgYear": frozenset({"gYear"}),
            "inXSDgYearMonth": frozenset({"gYearMonth"}),
        },
        temporal_properties=("hasTime", "hasBeginning", "hasEnd", "inside"),
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
        temporal_properties=(
            "startDate",
            "endDate",
            "datePublished",
            "dateCreated",
            "dateModified",
        ),
    ),
    _listed(
        "qudt",
        "QUDT",
        "http://qudt.org/schema/qudt/",
        "qudt_schema.txt",
        value_classes=("QuantityValue",),
        datatype_ranges={"numericValue": _QUDT_NUMBERS},
        class_ranges={"hasUnit": "Unit", "unit": "Unit"},
        anchor_properties=("hasQuantityKind", "applicableUnit"),
        measurement=True,
    ),
    _listed(
        "unit",
        "QUDT units",
        QUDT_UNIT_NAMESPACE,
        "qudt_units.txt",
        member_class="http://qudt.org/schema/qudt/Unit",
        member_dimensions=_shipped_dimensions(
            "qudt_unit_dimensions.txt", "http://qudt.org/vocab/dimensionvector/"
        ),
        measurement=True,
    ),
    _listed(
        "cur",
        "QUDT currency units",
        "http://qudt.org/vocab/currency/",
        "qudt_currencies.txt",
        member_class="http://qudt.org/schema/qudt/Unit",
        measurement=True,
        superseded=True,
    ),
    _by_namespace(
        "quantitykind",
        "QUDT quantity kinds",
        "http://qudt.org/vocab/quantitykind/",
        measurement=True,
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


def well_known_dimension(iri: str) -> str | None:
    """The dimension of a listed individual (a measurement unit), as an IRI.

    ``None`` for an IRI outside the well-known vocabularies or one whose
    dimension is not shipped.
    """
    namespace, local_name = _split(iri)
    vocabulary = _index().get(namespace)
    if vocabulary is None:
        return None
    return vocabulary.member_dimensions.get(local_name)


def _variant_key(namespace: str) -> str | None:
    """``namespace`` without its ``http``/``https`` scheme and trailing separator."""
    for scheme in ("https:", "http:"):
        if namespace.startswith(scheme):
            rest = namespace[len(scheme) :]
            return rest[:-1] if rest.endswith(("/", "#")) else rest
    return None


@cache
def _canonical_by_variant() -> dict[str, str]:
    found: dict[str, set[str]] = {}
    for vocabulary in WELL_KNOWN_VOCABULARIES:
        for namespace in vocabulary.namespaces:
            key = _variant_key(namespace)
            if key is not None:
                found.setdefault(key, set()).add(vocabulary.namespaces[0])
    # A key two vocabularies share is no exact variant of either.
    return {key: next(iter(c)) for key, c in found.items() if len(c) == 1}


def canonical_namespace(namespace: str) -> str:
    """The canonical namespace of the well-known vocabulary ``namespace`` spells.

    ``namespace`` spells one when it differs from one of the vocabulary's
    namespaces only in its scheme (``http``/``https``) or its trailing
    separator (``/``, ``#``, none). Any other namespace is returned as given.
    """
    key = _variant_key(namespace)
    if key is None:
        return namespace
    return _canonical_by_variant().get(key, namespace)


def canonical_iri(iri: str) -> str:
    """``iri`` with its namespace made canonical (:func:`canonical_namespace`)."""
    namespace, local_name = _split(iri)
    if not local_name:
        return iri
    canonical = canonical_namespace(namespace)
    return iri if canonical == namespace else canonical + local_name


def is_listed_unit(iri: str) -> bool:
    """Whether ``iri`` is a unit of the shipped QUDT unit list."""
    return iri.startswith(QUDT_UNIT_NAMESPACE) and is_well_known_term(iri) is True


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


@cache
def _by_local_name() -> dict[str, tuple[str, ...]]:
    found: dict[str, list[str]] = {}
    for vocabulary in WELL_KNOWN_VOCABULARIES:
        if vocabulary.terms is None:
            continue
        namespace = vocabulary.namespaces[0]
        for name in sorted(vocabulary.terms):
            found.setdefault(name.casefold(), []).append(namespace + name)
    return {name: tuple(iris) for name, iris in found.items()}


def well_known_by_local_name(local_name: str, role: WellKnownRole) -> tuple[str, ...]:
    """Well-known terms of ``role`` named ``local_name``, ignoring case.

    In the order of :data:`WELL_KNOWN_VOCABULARIES`; vocabularies admitted by
    namespace have no term list and never match.
    """
    return tuple(
        iri
        for iri in _by_local_name().get(local_name.casefold(), ())
        if well_known_role(iri) == role
    )


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
def well_known_datatype_ranges() -> dict[str, frozenset[str]]:
    """Property IRI -> the XSD datatype IRIs its published range admits.

    A literal fits the range when it is a value of one of them. Treat as
    read-only.
    """
    return {
        namespace + name: frozenset(str(XSD) + datatype for datatype in datatypes)
        for vocabulary in WELL_KNOWN_VOCABULARIES
        for namespace in vocabulary.namespaces
        for name, datatypes in vocabulary.datatype_ranges.items()
    }


@cache
def well_known_statement_classes() -> dict[str, frozenset[StatementRoleName]]:
    """Statement class IRI -> the roles an instance of it must carry.

    An instance of one (or of a subclass) is about something else, its
    subject; without the required links it states a value of nothing.
    Treat as read-only.
    """
    return {
        namespace + name: roles
        for vocabulary in WELL_KNOWN_VOCABULARIES
        for namespace in vocabulary.namespaces
        for name, roles in vocabulary.statement_classes.items()
    }


@cache
def well_known_statement_roles() -> dict[str, StatementRole]:
    """Property IRI -> the role it plays for a well-known statement.

    ``classes`` are local names in the property's vocabulary. Treat as
    read-only.
    """
    return {
        vocabulary.namespaces[0] + name: role
        for vocabulary in WELL_KNOWN_VOCABULARIES
        for name, role in vocabulary.statement_roles.items()
    }


@cache
def well_known_closed_ranges() -> dict[str, frozenset[str]]:
    """Property IRI -> its published range classes whose members are listed.

    A class is listed when a well-known vocabulary ships its individuals
    (``member_class``). Treat as read-only.
    """
    listed = well_known_member_classes()
    found: dict[str, set[str]] = {}
    for vocabulary in WELL_KNOWN_VOCABULARIES:
        namespace = vocabulary.namespaces[0]
        for name, class_name in vocabulary.class_ranges.items():
            if namespace + class_name in listed:
                found.setdefault(namespace + name, set()).add(namespace + class_name)
    return {prop: frozenset(classes) for prop, classes in found.items()}


@cache
def well_known_member_classes() -> dict[str, tuple[WellKnownVocabulary, ...]]:
    """Class IRI -> the well-known vocabularies that list its individuals."""
    found: dict[str, list[WellKnownVocabulary]] = {}
    for vocabulary in WELL_KNOWN_VOCABULARIES:
        if vocabulary.member_class is not None:
            found.setdefault(vocabulary.member_class, []).append(vocabulary)
    return {cls: tuple(vocabularies) for cls, vocabularies in found.items()}


@cache
def well_known_object_pairs() -> dict[str, str]:
    """Datatype property IRI -> the object property IRI paired with it.

    The pair states the same relation with a node as its object, so a node
    written on the datatype property belongs on the object property.
    """
    return {
        namespace + datatype_property: namespace + object_property
        for vocabulary in WELL_KNOWN_VOCABULARIES
        for namespace in vocabulary.namespaces
        for datatype_property, object_property in vocabulary.object_pairs.items()
    }


@cache
def well_known_anchor_properties() -> frozenset[str]:
    """Property IRIs that declare what a property node measures.

    A property node carrying one (a quantity kind, an applicable measurement
    unit) is anchored: the catalog says which kind of quantity it is.
    """
    return frozenset(
        namespace + name
        for vocabulary in WELL_KNOWN_VOCABULARIES
        for namespace in vocabulary.namespaces
        for name in vocabulary.anchor_properties
    )


#: Local names of the XSD datatypes whose value is a time.
_TEMPORAL_DATATYPE_NAMES = frozenset(
    {"date", "dateTime", "dateTimeStamp", "gYear", "gYearMonth", "time"}
)


def well_known_temporal_properties() -> frozenset[str]:
    """Property IRIs of the well-known vocabularies whose object is a time."""
    return frozenset(
        namespace + name
        for vocabulary in WELL_KNOWN_VOCABULARIES
        for namespace in vocabulary.namespaces
        for name in (
            *vocabulary.temporal_properties,
            *(
                prop
                for prop, datatypes in vocabulary.datatype_ranges.items()
                if datatypes & _TEMPORAL_DATATYPE_NAMES
            ),
        )
    )


def well_known_prompt_list(*, measurement: bool = True) -> str:
    """The vocabularies as the facts prompt names them: ``prefix:`` and IRI.

    Args:
        measurement: Whether to name the vocabularies of statements and
            measurements (:attr:`WellKnownVocabulary.measurement`).
    """
    return ", ".join(
        f"`{vocabulary.prefix}:` <{vocabulary.namespaces[0]}>"
        for vocabulary in WELL_KNOWN_VOCABULARIES
        if not vocabulary.superseded and (measurement or not vocabulary.measurement)
    )


__all__ = [
    "QUDT_UNIT_NAMESPACE",
    "WELL_KNOWN_VOCABULARIES",
    "StatementRole",
    "StatementRoleName",
    "WellKnownRole",
    "WellKnownVocabulary",
    "canonical_iri",
    "canonical_namespace",
    "is_listed_unit",
    "is_well_known_term",
    "well_known_anchor_properties",
    "well_known_by_local_name",
    "well_known_role",
    "well_known_namespaces",
    "well_known_object_pairs",
    "well_known_prompt_list",
    "well_known_terms",
    "well_known_closed_ranges",
    "well_known_datatype_ranges",
    "well_known_dimension",
    "well_known_member_classes",
    "well_known_statement_classes",
    "well_known_statement_roles",
    "well_known_temporal_properties",
    "well_known_value_classes",
    "well_known_vocabulary",
]
