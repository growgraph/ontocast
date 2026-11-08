"""Ontology identity as types: the id, the IRI, and the pair.

An ontology's identity is minted by code, never written by a model: the model
proposes an ``ontology_id`` (and a title and description), and the IRI is
``<CURRENT_DOMAIN>/<ontology_id>``. Everything that would let two spellings of
one ontology coexist -- a fragment on the IRI, a standard prefix bound to
anything but its standard namespace, an id that collides with the document or
facts namespaces -- is refused here, at construction; an empty fragment is
dropped.

An id is also the prefix an ontology's terms are shown under, so a standard
prefix is a binding, not a name: ``sosa`` may name the ontology whose namespace
is ``http://www.w3.org/ns/sosa/`` (W3C SOSA, adopted from its file) and
nothing else, and never an ontology minted here.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from ontocast.onto.constants import COMMON_PREFIXES, WELL_KNOWN_PREFIXES
from ontocast.onto.iri_policy import normalize_namespace_iri
from ontocast.onto.util import CONVENTIONAL_MAPPINGS

#: A valid Turtle prefix name that is also a safe path segment.
ONTOLOGY_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{1,31}$")

#: Ids of the library's own namespaces (facts, documents, its vocabulary):
#: never an ontology's.
LIBRARY_ONTOLOGY_IDS: frozenset[str] = frozenset(
    {"cd", "doc", "facts", "oc", "ontocast"}
)


def _standard_prefix_namespaces() -> dict[str, frozenset[str]]:
    namespaces: dict[str, set[str]] = {}
    for prefix, namespace in COMMON_PREFIXES.items():
        namespaces.setdefault(prefix, set()).add(namespace.strip("<>"))
    for prefix, namespace in WELL_KNOWN_PREFIXES.items():
        namespaces.setdefault(prefix, set()).add(namespace)
    for namespace, prefix in CONVENTIONAL_MAPPINGS.items():
        namespaces.setdefault(prefix, set()).add(namespace)
    namespaces.setdefault("vann", set()).add("http://purl.org/vocab/vann/")
    return {prefix: frozenset(found) for prefix, found in namespaces.items()}


#: Standard prefixes and the namespaces each stands for. Such a prefix names
#: an ontology only when the ontology's namespace is one of these.
STANDARD_PREFIX_NAMESPACES: dict[str, frozenset[str]] = _standard_prefix_namespaces()


def validate_ontology_id(value: str) -> str:
    """Return ``value`` as an ontology id, or raise ``ValueError``.

    Only whitespace is stripped and case folded; any other repair would let two
    proposals that differ in spelling land on one id. A standard prefix passes
    here: whether it may name an ontology depends on the ontology's namespace
    (:func:`check_prefix_binding`).
    """
    cleaned = value.strip().lower()
    if not ONTOLOGY_ID_PATTERN.match(cleaned):
        raise ValueError(
            f"ontology_id {value!r} must match {ONTOLOGY_ID_PATTERN.pattern}: "
            "lowercase letters, digits, '_' or '-', 2-32 characters, starting "
            "with a letter"
        )
    if cleaned in LIBRARY_ONTOLOGY_IDS:
        raise ValueError(
            f"ontology_id {cleaned!r} is reserved (a namespace the library itself uses)"
        )
    return cleaned


def validate_minted_ontology_id(value: str) -> str:
    """An id for an ontology minted under ``CURRENT_DOMAIN``: no standard prefix.

    A minted ontology's namespace is never a standard one, so a standard prefix
    would show its terms under a prefix that already means something else.
    """
    cleaned = validate_ontology_id(value)
    if cleaned in STANDARD_PREFIX_NAMESPACES:
        raise ValueError(
            f"ontology_id {cleaned!r} is reserved (a standard prefix); choose "
            "another id"
        )
    return cleaned


def check_prefix_binding(ontology_id: str, namespace: str) -> None:
    """Refuse a standard prefix as the id of an ontology outside its namespace.

    Raises:
        ValueError: ``ontology_id`` is a standard prefix and ``namespace`` is not
            the namespace it stands for.
    """
    expected = STANDARD_PREFIX_NAMESPACES.get(ontology_id)
    if expected is not None and namespace not in expected:
        raise ValueError(
            f"ontology_id {ontology_id!r} is a standard prefix for "
            f"{' or '.join(sorted(expected))}; this ontology's namespace is "
            f"{namespace or '(none)'!s}, so it needs another id"
        )


def validate_ontology_iri(value: str) -> str:
    """Return ``value`` as a canonical ontology IRI, or raise ``ValueError``.

    An absolute ``http(s)`` IRI with no fragment: the version graph of each
    version is ``<iri>#<key>``, so the IRI itself cannot carry a ``#``. An
    empty fragment (``…/prov#``, common in published vocabularies) names the
    same document and is dropped; any other fragment is refused.
    """
    text = value.strip()
    if text.endswith("#"):
        text = text[:-1]
    parts = urlsplit(text)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(f"ontology IRI {value!r} must be an absolute http(s) IRI")
    if "#" in text or parts.query:
        raise ValueError(
            f"ontology IRI {value!r} must carry no fragment and no query: "
            "versions are addressed as <iri>#<key>"
        )
    if any(char.isspace() for char in text):
        raise ValueError(f"ontology IRI {value!r} must not contain whitespace")
    return text


def validate_current_domain(value: str) -> str:
    """The IRI stem ontology and document IRIs are minted under."""
    text = value.strip()
    parts = urlsplit(text)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(f"CURRENT_DOMAIN {value!r} must be an absolute http(s) IRI")
    if "#" in text or parts.query or text.endswith("/"):
        raise ValueError(
            f"CURRENT_DOMAIN {value!r} must carry no fragment, no query and no "
            "trailing '/'"
        )
    return text


OntologyId = Annotated[str, AfterValidator(validate_ontology_id)]
MintedOntologyId = Annotated[str, AfterValidator(validate_minted_ontology_id)]
OntologyIri = Annotated[str, AfterValidator(validate_ontology_iri)]


def mint_ontology_iri(current_domain: str, ontology_id: str) -> str:
    """The IRI of the ontology ``ontology_id`` under ``current_domain``."""
    return validate_ontology_iri(
        f"{validate_current_domain(current_domain)}/"
        f"{validate_minted_ontology_id(ontology_id)}"
    )


def is_owned_ontology(iri: str, current_domain: str) -> bool:
    """Whether the ontology ``iri`` is this deployment's, i.e. under ``current_domain``.

    An owned ontology is extended in place. Any other is external: its
    namespace is never written to, and new terms go to its extension ontology.
    """
    domain = validate_current_domain(current_domain)
    return iri == domain or iri.startswith(f"{domain}/")


#: Suffix of an extension ontology's id: ``<external id>-ext``.
EXTENSION_SUFFIX = "-ext"


def extension_ontology_id(external_id: str) -> str:
    """The id of the extension ontology of the external ontology ``external_id``.

    The id is fixed by the external id, so every document and every run that
    extends one external ontology writes to the same extension.
    """
    stem = external_id[: 32 - len(EXTENSION_SUFFIX)].rstrip("_-")
    return validate_minted_ontology_id(f"{stem}{EXTENSION_SUFFIX}")


def version_iri_for(iri: str, key: str) -> str:
    """The named graph holding version ``key`` of the ontology ``iri``."""
    return f"{iri}#{key}"


def split_version_iri(version_iri: str) -> tuple[str, str]:
    """``(ontology IRI, version key)`` of a version IRI.

    Raises:
        ValueError: ``version_iri`` has no fragment, or more than one.
    """
    if version_iri.count("#") != 1:
        raise ValueError(f"not a version IRI (expected <iri>#<key>): {version_iri!r}")
    iri, key = version_iri.split("#", 1)
    if not key:
        raise ValueError(f"version IRI has an empty key: {version_iri!r}")
    return validate_ontology_iri(iri), key


class OntologyIdentity(BaseModel):
    """What names an ontology: its id, its IRI, and how it describes itself.

    ``namespace`` is where its terms live: an existing ontology's own, else
    ``<iri>#``, where a new ontology's terms are minted. ``extends`` names the
    external ontology an extension ontology adds terms to.
    """

    model_config = ConfigDict(frozen=True)

    ontology_id: OntologyId
    iri: OntologyIri
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    namespace: str = Field(default="")
    #: The external ontology this one extends; minted with ``owl:imports`` of it.
    extends: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _default_namespace(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("namespace") and data.get("iri"):
            iri = validate_ontology_iri(str(data["iri"]))
            data = {
                **data,
                "namespace": normalize_namespace_iri(iri, context="ontology"),
            }
        return data

    @model_validator(mode="after")
    def _prefix_matches_namespace(self) -> OntologyIdentity:
        check_prefix_binding(self.ontology_id, self.namespace)
        return self


class VersionRef(BaseModel):
    """One stored version of one ontology."""

    model_config = ConfigDict(frozen=True)

    iri: str
    version_iri: str

    @property
    def key(self) -> str:
        return split_version_iri(self.version_iri)[1]


class OntologyVersionRecord(VersionRef):
    """A version a run wrote: created as a root, or updated from its priors."""

    ontology_id: str
    version: str
    prior_version_iris: list[str] = Field(default_factory=list)
    role: Literal["created", "updated"]
