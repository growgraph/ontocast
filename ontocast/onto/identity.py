"""Ontology identity as types: the id, the IRI, and the pair.

An ontology's identity is minted by code, never written by a model: the model
proposes an ``ontology_id`` (and a title and description), and the IRI is
``<CURRENT_DOMAIN>/<ontology_id>``. Everything that would let two spellings of
one ontology coexist -- a fragment on the IRI, an id that is also a standard
prefix, an id that collides with the document or facts namespaces -- is
refused here, at construction.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from ontocast.onto.constants import COMMON_PREFIXES, WELL_KNOWN_PREFIXES
from ontocast.onto.util import CONVENTIONAL_MAPPINGS

#: A valid Turtle prefix name that is also a safe path segment.
ONTOLOGY_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{1,31}$")

#: Ids that name something else already: standard-vocabulary prefixes, the
#: facts and document namespaces of the library itself, and the contract
#: vocabularies the header is written in.
RESERVED_ONTOLOGY_IDS: frozenset[str] = frozenset(
    {*COMMON_PREFIXES, *WELL_KNOWN_PREFIXES, *CONVENTIONAL_MAPPINGS.values()}
    | {"cd", "doc", "facts", "vann", "sh", "oc", "ontocast"}
)


def validate_ontology_id(value: str) -> str:
    """Return ``value`` as an ontology id, or raise ``ValueError``.

    Only whitespace is stripped and case folded; any other repair would let two
    proposals that differ in spelling land on one id.
    """
    cleaned = value.strip().lower()
    if not ONTOLOGY_ID_PATTERN.match(cleaned):
        raise ValueError(
            f"ontology_id {value!r} must match {ONTOLOGY_ID_PATTERN.pattern}: "
            "lowercase letters, digits, '_' or '-', 2-32 characters, starting "
            "with a letter"
        )
    if cleaned in RESERVED_ONTOLOGY_IDS:
        raise ValueError(
            f"ontology_id {cleaned!r} is reserved (a standard prefix or a "
            "namespace the library itself uses)"
        )
    return cleaned


def validate_ontology_iri(value: str) -> str:
    """Return ``value`` as an ontology IRI, or raise ``ValueError``.

    An absolute ``http(s)`` IRI with no fragment: the version graph of each
    version is ``<iri>#<key>``, so the IRI itself cannot carry a ``#``.
    """
    text = value.strip()
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
OntologyIri = Annotated[str, AfterValidator(validate_ontology_iri)]


def mint_ontology_iri(current_domain: str, ontology_id: str) -> str:
    """The IRI of the ontology ``ontology_id`` under ``current_domain``."""
    return validate_ontology_iri(
        f"{validate_current_domain(current_domain)}/{validate_ontology_id(ontology_id)}"
    )


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
    """What names an ontology: its id, its IRI, and how it describes itself."""

    model_config = ConfigDict(frozen=True)

    ontology_id: OntologyId
    iri: OntologyIri
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)

    @property
    def namespace(self) -> str:
        """The namespace its terms are minted under: ``<iri>#``, or the IRI
        itself when it already ends with a delimiter."""
        from ontocast.onto.iri_policy import normalize_namespace_iri

        return normalize_namespace_iri(self.iri, context="ontology")


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
