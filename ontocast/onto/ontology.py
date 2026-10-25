"""A catalog ontology: its graph, its identity and its place in a version DAG.

Identity (``ontology_id``, ``iri``, ``title``, ``description``) is set by code
when an ontology is minted, adopted or read from a store; nothing here guesses
it from the graph. A *version* of an ontology is one immutable named graph
``<iri>#<key>`` whose header carries its lineage; the header is written and
read by :mod:`ontocast.onto.ontology_header`, and only there.
"""

from __future__ import annotations

import logging
import pathlib
import re
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
from rdflib import DCTERMS, OWL, RDF, RDFS, SH, Namespace, URIRef

from ontocast.onto.constants import ONTOCAST, ONTOLOGY_NULL_IRI, PROV
from ontocast.onto.iri_policy import normalize_namespace_iri
from ontocast.onto.llm_graph_payload import LLMGraphWire
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import GraphUpdate

if TYPE_CHECKING:
    import networkx as nx

logger = logging.getLogger(__name__)

#: Semantic version label, MAJOR.MINOR.PATCH. A label, never an identity: the
#: identity of a version is its version IRI.
SemanticVersion = Annotated[
    str,
    Field(
        pattern=r"^\d+\.\d+\.\d+$",
        description="Semantic version in MAJOR.MINOR.PATCH format (e.g., 1.2.3)",
    ),
]

VersionIncrement = Literal["major", "minor", "patch"]

VANN = Namespace("http://purl.org/vocab/vann/")

#: Predicates the version header owns on the ontology subject. They are
#: rewritten on every write and left out of the content hash.
CONTRACT_PREDICATES: frozenset[URIRef] = frozenset(
    {
        OWL.versionIRI,
        OWL.versionInfo,
        OWL.priorVersion,
        VANN.preferredNamespacePrefix,
        VANN.preferredNamespaceUri,
        DCTERMS.title,
        RDFS.label,
        DCTERMS.description,
        RDFS.comment,
        DCTERMS.created,
        DCTERMS.modified,
        DCTERMS.identifier,
        PROV.wasDerivedFrom,
        SH.declare,
    }
)

#: Marks a version whose content was removed; its header stays so lineage
#: through it never dangles.
REMOVED_VERSION_TYPE = ONTOCAST.RemovedVersion

_SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def normalize_semantic_version(version: str) -> str | None:
    """``MAJOR.MINOR.PATCH`` for a version label an author wrote, or None.

    ``"3.5"`` becomes ``"3.5.0"`` and ``"3"`` becomes ``"3.0.0"``; any other
    text is not a version.
    """
    text = version.strip()
    if _SEMVER.match(text):
        return text
    if re.match(r"^\d+\.\d+$", text):
        return f"{text}.0"
    if re.match(r"^\d+$", text):
        return f"{text}.0.0"
    return None


def bump_version(version: str, increment: VersionIncrement) -> str:
    """``version`` bumped by ``increment``."""
    match = _SEMVER.match(version)
    if match is None:
        raise ValueError(f"not a semantic version: {version!r}")
    major, minor, patch = (int(part) for part in match.groups())
    if increment == "major":
        return f"{major + 1}.0.0"
    if increment == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


#: ``rdf:type`` objects counted as class-level changes by version bumps.
_CLASS_TYPES = frozenset({OWL.Class, RDFS.Class})
#: ``rdf:type`` objects counted as property-level changes by version bumps.
_PROPERTY_TYPES = frozenset(
    {OWL.ObjectProperty, OWL.DatatypeProperty, OWL.AnnotationProperty, RDF.Property}
)


def increment_type_for(updates: Sequence[GraphUpdate]) -> tuple[VersionIncrement, str]:
    """The version increment a set of applied updates earns, with its reason.

    Deletes of several classes or properties are a major change, any delete a
    minor one, a handful of new terms a patch.
    """
    if not updates:
        return ("patch", "no updates")
    total_deletes = total_inserts = class_changes = property_changes = 0
    for update in updates:
        for op in update.triple_operations:
            if op.type == "delete":
                total_deletes += len(op.graph)
            else:
                total_inserts += len(op.graph)
            for _, predicate, object_ in op.graph:
                if predicate != RDF.type:
                    continue
                if object_ in _CLASS_TYPES:
                    class_changes += 1
                elif object_ in _PROPERTY_TYPES:
                    property_changes += 1
    if total_deletes > 5 and (class_changes > 2 or property_changes > 3):
        return (
            "major",
            f"deleted {total_deletes} triples including {class_changes} classes "
            f"and {property_changes} properties",
        )
    if total_deletes > 0:
        return ("minor", f"deleted {total_deletes} triples")
    if class_changes >= 5 or property_changes >= 5:
        return (
            "minor",
            f"added {class_changes} classes and {property_changes} properties",
        )
    return ("patch", f"added {total_inserts} triples")


class OntologyProperties(BaseModel):
    """What names and describes an ontology.

    Attributes:
        ontology_id: Short lowercase handle; also its preferred prefix.
        title: Human-readable title.
        description: A few sentences on domain, purpose and applicability.
        version: Semantic version label of this version.
        iri: Ontology IRI, with no fragment.
    """

    ontology_id: str | None = Field(default=None)
    title: str | None = Field(default=None)
    description: str | None = Field(default=None)
    version: SemanticVersion | None = Field(default=None)
    iri: str = Field(default=ONTOLOGY_NULL_IRI)

    @property
    def namespace(self) -> str:
        """The namespace its terms live under (``<iri>#``)."""
        return normalize_namespace_iri(self.iri, context="ontology")


class OntologyPropertiesWithLineage(OntologyProperties):
    """Properties of one *version* of an ontology.

    Attributes:
        hash: Content digest of the graph, header excluded. Change detection
            only: it decides whether a derived artifact is a new version.
        version_iri: The named graph this version is stored as, ``<iri>#<key>``.
            The version's identity; None until the version is stamped.
        prior_version_iris: Version IRIs this version was derived from: none
            for a root, one for an update, several for a merge.
        created_at: When the root of this lineage was created.
        modified_at: When this version was created.
        removed: True for a tombstone, a version whose content was removed.
    """

    hash: str | None = Field(default=None)
    version_iri: str | None = Field(default=None)
    prior_version_iris: list[str] = Field(default_factory=list)
    created_at: datetime | None = Field(default=None)
    modified_at: datetime | None = Field(default=None)
    removed: bool = Field(default=False)

    @property
    def version_key(self) -> str | None:
        """The fragment of ``version_iri``."""
        if self.version_iri is None or "#" not in self.version_iri:
            return None
        return self.version_iri.rsplit("#", 1)[1]


class OntologyHeader(OntologyPropertiesWithLineage):
    """The header of one stored version, read without its graph.

    Attributes:
        graph_uri: The named graph it was read from; equals ``version_iri``.
    """

    graph_uri: str = Field(default="")

    @classmethod
    def from_ontology(cls, ontology: Ontology) -> OntologyHeader:
        """The header of a stamped ontology.

        Raises:
            ValueError: ``ontology`` has no ``version_iri`` (it was never stamped).
        """
        if not ontology.version_iri:
            raise ValueError(f"ontology {ontology.iri} is not stamped as a version")
        return cls(
            graph_uri=ontology.version_iri,
            iri=ontology.iri,
            ontology_id=ontology.ontology_id,
            title=ontology.title,
            description=ontology.description,
            version=ontology.version,
            hash=ontology.hash,
            version_iri=ontology.version_iri,
            prior_version_iris=list(ontology.prior_version_iris),
            created_at=ontology.created_at,
            modified_at=ontology.modified_at,
            removed=ontology.removed,
        )


class Ontology(OntologyPropertiesWithLineage):
    """An ontology's graph with its identity and lineage.

    Attributes:
        graph: The RDF graph. Between a read from the store and the next stamp
            it may carry the header of the version it was derived from; the
            header is rewritten when the artifact is stamped.
        bases: Stored versions this artifact derives from: none for a root,
            one for an update, two for a merge. Carried unchanged through
            in-run derivations so the stamped version's lineage points at
            stored versions only.
    """

    graph: LLMGraphWire = Field(default_factory=RDFGraph)
    bases: list[OntologyHeader] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def is_null(self) -> bool:
        """True for the null ontology (no identity, no graph)."""
        return self.iri == ONTOLOGY_NULL_IRI and self.ontology_id is None

    @property
    def prefix(self) -> str | None:
        """The prefix the graph binds to this ontology's namespace, if any."""
        namespace = URIRef(self.namespace)
        for prefix, iri in self.graph.namespaces():
            if iri == namespace and prefix:
                return prefix
        return None

    @property
    def base(self) -> OntologyHeader | None:
        """The single stored version this artifact updates, if exactly one."""
        return self.bases[0] if len(self.bases) == 1 else None

    def content_hash(self) -> str:
        """Compute, set and return the content digest of ``graph``.

        The digest covers every triple except the version header on the
        ontology subject and its prefix declarations, so two versions with the
        same terms hash alike whatever their labels or dates.
        """
        self.hash = content_hash(self.graph, self.iri)
        return self.hash

    def derive_updated_version(self, updated_graph: RDFGraph) -> Ontology:
        """A new unstamped artifact over ``updated_graph`` with the same bases."""
        derived = self.model_copy(deep=True)
        derived.graph = updated_graph
        derived.version_iri = None
        derived.prior_version_iris = []
        derived.modified_at = None
        derived.removed = False
        derived.content_hash()
        return derived

    def union_fresh(self, others: Sequence[Ontology]) -> Ontology:
        """One root over the union of this and ``others``' graphs.

        The fresh units of a document each render whole graphs under the
        document's home ontology; their union is the ontology.
        """
        merged = self.model_copy(deep=True)
        merged.graph = self.graph.copy()
        for other in others:
            for triple in other.graph:
                merged.graph.add(triple)
            for prefix, namespace_uri in other.graph.namespaces():
                if prefix:
                    merged.graph.bind(prefix, namespace_uri)
        merged.version_iri = None
        merged.prior_version_iris = []
        merged.modified_at = None
        merged.content_hash()
        return merged

    def __iadd__(self, other: Ontology | RDFGraph) -> Ontology:
        """Merge ``other``'s graph in; an ``Ontology`` also hands over its identity."""
        if isinstance(other, Ontology):
            self.graph += other.graph
            for name in ("ontology_id", "title", "description", "iri", "version"):
                setattr(self, name, getattr(other, name))
            self.bases = list(other.bases)
        else:
            self.graph += other
        self.hash = None
        self.version_iri = None
        return self

    @classmethod
    def from_file(
        cls,
        file_path: pathlib.Path,
        format: str = "turtle",
        *,
        ontology_id: str | None = None,
        now: datetime | None = None,
    ) -> Ontology:
        """Adopt the ontology in a file as a root version.

        Args:
            file_path: Path to the ontology file.
            format: rdflib parser name.
            ontology_id: The id to register it under; by default the file's
                ``vann:preferredNamespacePrefix``, else one derived from its IRI.
            now: The adoption time; defaults to the current UTC time.
        """
        from ontocast.onto.ontology_header import adopt_ontology

        graph = RDFGraph()
        graph.parse(file_path, format=format)
        return adopt_ontology(graph, ontology_id=ontology_id, now=now)

    @staticmethod
    def strip_ontology_header_triples(graph: RDFGraph) -> RDFGraph:
        """Remove every ``owl:Ontology`` subject and its triples, in place."""
        for onto_subject in {
            s for s, _, _ in graph.triples((None, RDF.type, OWL.Ontology))
        }:
            for triple in list(graph.triples((onto_subject, None, None))):
                graph.remove(triple)
        return graph

    def describe(self) -> str:
        """A short human-readable description."""
        return (
            f"Ontology id: {self.ontology_id}\n"
            f"Prefix: {self.prefix}\n"
            f"Description: {self.description}\n"
            f"Ontology IRI: {self.iri}\n"
        )

    @staticmethod
    def build_lineage_graph(
        versions: Sequence[OntologyPropertiesWithLineage],
    ) -> nx.DiGraph:
        """The version DAG of ``versions``: an edge from each version to each prior.

        Nodes are version IRIs; a prior not in ``versions`` is still a node.
        """
        import networkx as nx

        lineage = nx.DiGraph()
        for version in versions:
            if not version.version_iri:
                continue
            lineage.add_node(
                version.version_iri,
                iri=version.iri,
                title=version.title,
                ontology_id=version.ontology_id,
                version=version.version,
                modified_at=(
                    version.modified_at.isoformat() if version.modified_at else None
                ),
                removed=version.removed,
            )
            for prior in version.prior_version_iris:
                lineage.add_edge(version.version_iri, prior)
        return lineage


def content_hash(graph: RDFGraph, iri: str) -> str:
    """The content digest of ``graph`` as a version of the ontology ``iri``.

    Excludes the header predicates on the ontology subject and the prefix
    declaration nodes they point at; everything else, author extras on the
    subject included, is content.
    """
    subject = URIRef(iri)
    declaration_nodes = set(graph.objects(subject, SH.declare))
    content = RDFGraph()
    for s, p, o in graph:
        if s in declaration_nodes:
            continue
        if s == subject and (p in CONTRACT_PREDICATES or o == OWL.Ontology):
            continue
        content.add((s, p, o))
    for prefix, namespace in graph.namespaces():
        content.bind(prefix, namespace)
    return content.hash()


__all__: list[str] = [
    "CONTRACT_PREDICATES",
    "Ontology",
    "OntologyHeader",
    "OntologyProperties",
    "OntologyPropertiesWithLineage",
    "REMOVED_VERSION_TYPE",
    "SemanticVersion",
    "VANN",
    "VersionIncrement",
    "bump_version",
    "content_hash",
    "increment_type_for",
    "normalize_semantic_version",
]
