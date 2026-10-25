"""The version header: the one place it is written, and the one place it is read.

Every stored version of an ontology is a named graph ``<iri>#<key>`` whose
``owl:Ontology`` subject ``<iri>`` carries, on the version itself:

| predicate | value |
|---|---|
| ``owl:versionIRI`` | the named graph (``<iri>#<key>``) |
| ``owl:priorVersion`` | the version IRI(s) it was derived from; none on a root |
| ``owl:versionInfo`` | a semantic version label |
| ``vann:preferredNamespacePrefix`` / ``vann:preferredNamespaceUri`` | the id and ``<iri>#`` |
| ``dcterms:title`` and ``rdfs:label`` | the title |
| ``dcterms:description`` | the description, when there is one |
| ``dcterms:created`` / ``dcterms:modified`` | when the lineage began / when this version was made |
| ``prov:wasDerivedFrom`` | the source document, for versions a run wrote |

A removed version keeps its header and gets ``<graph> a oc:RemovedVersion``.
The key is a name, not a digest anyone recomputes: the library uses the content
hash, so identical content is never stored twice.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

from rdflib import DCTERMS, OWL, RDF, RDFS, SH, XSD, Literal, URIRef

from ontocast.onto.constants import DEFAULT_DOMAIN, PROV
from ontocast.onto.identity import (
    OntologyIdentity,
    mint_ontology_iri,
    split_version_iri,
    validate_ontology_id,
    validate_ontology_iri,
    version_iri_for,
)
from ontocast.onto.ontology import (
    CONTRACT_PREDICATES,
    REMOVED_VERSION_TYPE,
    VANN,
    Ontology,
    OntologyHeader,
    VersionIncrement,
    bump_version,
    increment_type_for,
    normalize_semantic_version,
)
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import GraphUpdate
from ontocast.onto.util import derive_ontology_id

logger = logging.getLogger(__name__)

#: Contract predicates the header rewrites on every write. Prefix declarations
#: are managed by ``RDFGraph.materialize_prefix_declarations`` and kept.
_REWRITTEN_PREDICATES: frozenset[URIRef] = CONTRACT_PREDICATES - {SH.declare}

__all__ = [
    "CatalogContractError",
    "DivergedOntologyError",
    "OntologyHeader",
    "VersionDiff",
    "adopt_ontology",
    "catalog_heads",
    "diff_versions",
    "header_from_fields",
    "heads_by_iri",
    "mint_ontology",
    "ontology_from_version_graph",
    "read_header",
    "stamp_version",
    "tombstone_graph",
    "without_version_header",
    "write_header",
]


class CatalogContractError(ValueError):
    """A stored graph does not carry the version header this library writes."""

    def __init__(self, graph_uri: str, missing: Sequence[str]) -> None:
        self.graph_uri = graph_uri
        self.missing = list(missing)
        super().__init__(
            f"graph {graph_uri} is not a version of an ontology: "
            f"missing {', '.join(self.missing)}"
        )


class DivergedOntologyError(RuntimeError):
    """An ontology has several heads; it cannot be extended until they are merged."""

    def __init__(self, iri: str, heads: Sequence[str]) -> None:
        self.iri = iri
        self.heads = list(heads)
        super().__init__(
            f"ontology {iri} has {len(self.heads)} heads ({', '.join(self.heads)}); "
            "merge them before extending it"
        )


def _semver_key(label: str) -> tuple[int, int, int]:
    """Numeric order of a semantic version label (``1.10.0`` > ``1.9.0``)."""
    major, minor, patch = label.split(".")
    return (int(major), int(minor), int(patch))


def _now(now: datetime | None) -> datetime:
    return now if now is not None else datetime.now(timezone.utc)


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def header_from_fields(
    graph_uri: str, fields: Mapping[str, Sequence[str]]
) -> OntologyHeader:
    """Build the header of ``graph_uri`` from its predicate values.

    ``fields`` maps a short name (``onto``, ``versionIri``, ``version``,
    ``prior``, ``prefix``, ``title``, ``description``, ``created``, ``modified``,
    ``removed``) to the lexical values found, so a SPARQL row and a parsed graph
    feed the same check.

    Raises:
        CatalogContractError: a required field is missing, or the version IRI
            does not name the graph it was read from.
    """

    def first(name: str) -> str | None:
        values = fields.get(name) or []
        return values[0] if values else None

    missing = [
        name
        for name in (
            "onto",
            "versionIri",
            "version",
            "prefix",
            "title",
            "created",
            "modified",
        )
        if first(name) is None
    ]
    if missing:
        raise CatalogContractError(graph_uri, missing)
    iri = first("onto")
    version_iri = first("versionIri")
    assert iri is not None and version_iri is not None
    if version_iri != graph_uri:
        raise CatalogContractError(
            graph_uri, [f"owl:versionIRI equal to the graph (found {version_iri})"]
        )
    version = normalize_semantic_version(first("version") or "")
    if version is None:
        raise CatalogContractError(graph_uri, ["a semantic owl:versionInfo"])
    try:
        validate_ontology_iri(iri)
        base_iri, _ = split_version_iri(version_iri)
        ontology_id = validate_ontology_id(first("prefix") or "")
        created_at = _parse_datetime(first("created") or "")
        modified_at = _parse_datetime(first("modified") or "")
    except ValueError as exc:
        raise CatalogContractError(graph_uri, [str(exc)]) from exc
    if base_iri != iri:
        raise CatalogContractError(
            graph_uri, [f"owl:versionIRI under the ontology IRI {iri}"]
        )
    return OntologyHeader(
        graph_uri=graph_uri,
        iri=iri,
        ontology_id=ontology_id,
        title=first("title"),
        description=first("description"),
        version=version,
        version_iri=version_iri,
        prior_version_iris=sorted(set(fields.get("prior") or [])),
        created_at=created_at,
        modified_at=modified_at,
        removed=bool(fields.get("removed")),
    )


def read_header(graph: RDFGraph, graph_uri: str) -> OntologyHeader:
    """The version header of the stored graph ``graph`` named ``graph_uri``.

    Raises:
        CatalogContractError: the graph has no single ``owl:Ontology`` subject,
            or its header is incomplete.
    """
    subjects = {
        s
        for s, _, _ in graph.triples((None, RDF.type, OWL.Ontology))
        if isinstance(s, URIRef)
    }
    if len(subjects) != 1:
        raise CatalogContractError(
            graph_uri, [f"exactly one owl:Ontology subject (found {len(subjects)})"]
        )
    subject = next(iter(subjects))

    def values(predicate: URIRef) -> list[str]:
        return sorted(str(o) for o in graph.objects(subject, predicate))

    fields: dict[str, list[str]] = {
        "onto": [str(subject)],
        "versionIri": values(OWL.versionIRI),
        "version": values(OWL.versionInfo),
        "prior": values(OWL.priorVersion),
        "prefix": values(VANN.preferredNamespacePrefix),
        "title": values(DCTERMS.title),
        "description": values(DCTERMS.description),
        "created": values(DCTERMS.created),
        "modified": values(DCTERMS.modified),
        "removed": (
            ["true"]
            if (URIRef(graph_uri), RDF.type, REMOVED_VERSION_TYPE) in graph
            else []
        ),
    }
    return header_from_fields(graph_uri, fields)


def write_header(graph: RDFGraph, version: Ontology, *, source_doc: str | None) -> None:
    """Write ``version``'s header onto its subject in ``graph``, in place.

    Every contract predicate on the subject except the prefix declarations is
    replaced; author extras (``owl:imports``, a licence, ...) are kept.
    """
    subject = URIRef(version.iri)
    if not version.version_iri or not version.version or not version.modified_at:
        raise ValueError(f"ontology {version.iri} is not stamped")
    for p in _REWRITTEN_PREDICATES:
        for o in list(graph.objects(subject, p)):
            graph.remove((subject, p, o))
    for s in {s for s, _, _ in graph.triples((None, RDF.type, OWL.Ontology))}:
        if s != subject:
            for triple in list(graph.triples((s, None, None))):
                graph.remove(triple)
    graph.add((subject, RDF.type, OWL.Ontology))
    graph.add((subject, OWL.versionIRI, URIRef(version.version_iri)))
    graph.add((subject, OWL.versionInfo, Literal(version.version)))
    for prior in version.prior_version_iris:
        graph.add((subject, OWL.priorVersion, URIRef(prior)))
    graph.add((subject, VANN.preferredNamespacePrefix, Literal(version.ontology_id)))
    graph.add((subject, VANN.preferredNamespaceUri, Literal(version.namespace)))
    graph.add((subject, DCTERMS.title, Literal(version.title)))
    graph.add((subject, RDFS.label, Literal(version.title)))
    if version.description:
        graph.add((subject, DCTERMS.description, Literal(version.description)))
    created = version.created_at or version.modified_at
    graph.add(
        (subject, DCTERMS.created, Literal(created.isoformat(), datatype=XSD.dateTime))
    )
    graph.add(
        (
            subject,
            DCTERMS.modified,
            Literal(version.modified_at.isoformat(), datatype=XSD.dateTime),
        )
    )
    if source_doc:
        graph.add((subject, PROV.wasDerivedFrom, URIRef(source_doc)))
    if version.ontology_id:
        graph.bind(version.ontology_id, version.namespace, override=True)
    graph.bind("vann", str(VANN))


def mint_ontology(identity: OntologyIdentity, graph: RDFGraph) -> Ontology:
    """A root artifact for the terms in ``graph`` under ``identity``.

    Any ``owl:Ontology`` subject the graph carries is dropped: the header is
    written when the artifact is stamped.
    """
    terms = graph.copy()
    Ontology.strip_ontology_header_triples(terms)
    terms.bind(identity.ontology_id, identity.namespace, override=True)
    artifact = Ontology(
        ontology_id=identity.ontology_id,
        iri=identity.iri,
        title=identity.title,
        description=identity.description or None,
        graph=terms,
    )
    artifact.content_hash()
    return artifact


def stamp_version(
    artifact: Ontology,
    *,
    updates: Sequence[GraphUpdate] = (),
    now: datetime | None = None,
    source_doc: str | None = None,
    version_label: str | None = None,
    increment: VersionIncrement | None = None,
) -> Ontology | None:
    """The stored version ``artifact`` becomes, or None if it changes nothing.

    A root (no bases) is version ``1.0.0`` unless the artifact carries a label
    already; an update of one base is that base's label bumped by ``updates``;
    a merge of several bases takes the highest label bumped once. The version
    IRI is ``<iri>#<content hash>``, so stamping the same content twice yields
    the same version.

    ``version_label`` wins when it parses and is strictly higher than every
    base's label (an upload that declares a newer ``owl:versionInfo``).
    ``increment`` overrides the bump derived from ``updates``.

    Returns:
        A deep copy with the lineage fields set and the header written, or
        None when the artifact's content equals its single base's.
    """
    if artifact.is_null() or len(artifact.graph) == 0:
        raise ValueError("cannot stamp an empty or null ontology")
    identity = OntologyIdentity(
        ontology_id=artifact.ontology_id or "",
        iri=artifact.iri,
        title=artifact.title or "",
        description=artifact.description or "",
    )
    digest = artifact.content_hash()
    stamped_at = _now(now)
    bases = list(artifact.bases)
    if len(bases) == 1 and bases[0].hash == digest:
        return None
    preferred = (
        normalize_semantic_version(version_label or "") if version_label else None
    )
    if not bases:
        label = preferred or artifact.version or "1.0.0"
        created_at = stamped_at
    else:
        bump = increment if increment is not None else increment_type_for(updates)[0]
        highest = max((base.version or "1.0.0" for base in bases), key=_semver_key)
        if preferred is not None and _semver_key(preferred) > _semver_key(highest):
            label = preferred
        else:
            label = bump_version(highest, bump)
        created_at = min(
            (base.created_at for base in bases if base.created_at is not None),
            default=stamped_at,
        )
    stamped = artifact.model_copy(deep=True)
    stamped.ontology_id = identity.ontology_id
    stamped.iri = identity.iri
    stamped.title = identity.title
    stamped.description = identity.description or None
    stamped.version = label
    stamped.hash = digest
    stamped.version_iri = version_iri_for(identity.iri, digest)
    stamped.prior_version_iris = [
        base.version_iri for base in bases if base.version_iri
    ]
    stamped.created_at = created_at
    stamped.modified_at = stamped_at
    stamped.removed = False
    stamped.graph.materialize_prefix_declarations(URIRef(identity.iri))
    write_header(stamped.graph, stamped, source_doc=source_doc)
    # A stamped artifact is a stored version: what derives from it next
    # derives from this version.
    stamped.bases = [OntologyHeader.from_ontology(stamped)]
    return stamped


def adopt_ontology(
    graph: RDFGraph,
    *,
    ontology_id: str | None = None,
    now: datetime | None = None,
    current_domain: str = DEFAULT_DOMAIN,
) -> Ontology:
    """Adopt an author's ontology as a root version.

    The graph must declare exactly one named ``owl:Ontology`` whose IRI has no
    fragment. The id comes from ``ontology_id``, else the file's
    ``vann:preferredNamespacePrefix``, else is derived from the IRI; the title
    from ``dcterms:title``, else ``rdfs:label``, else the id; the version label
    from ``owl:versionInfo`` when it parses, else ``1.0.0``.

    Args:
        current_domain: Unused for identity (the file's IRI is kept); accepted
            so adopters share one signature with minting.

    Raises:
        ValueError: the graph declares no, or several, ontologies, or its IRI
            or id is not acceptable.
    """
    _ = current_domain
    subjects = {
        s
        for s, _, _ in graph.triples((None, RDF.type, OWL.Ontology))
        if isinstance(s, URIRef)
    }
    if len(subjects) != 1:
        raise ValueError(
            f"an ontology declares exactly one owl:Ontology subject; found {len(subjects)}"
        )
    subject = next(iter(subjects))
    iri = validate_ontology_iri(str(subject))

    def first(predicate: URIRef) -> str | None:
        values = sorted(str(o) for o in graph.objects(subject, predicate))
        return values[0] if values else None

    candidate_id = (
        ontology_id
        or first(VANN.preferredNamespacePrefix)
        or derive_ontology_id(iri)
        or ""
    )
    identity = OntologyIdentity(
        ontology_id=candidate_id,
        iri=iri,
        title=first(DCTERMS.title) or first(RDFS.label) or candidate_id,
        description=first(DCTERMS.description) or first(RDFS.comment) or "",
    )
    label = normalize_semantic_version(first(OWL.versionInfo) or "") or "1.0.0"
    terms = graph.copy()
    for p in _REWRITTEN_PREDICATES:
        for o in list(terms.objects(subject, p)):
            terms.remove((subject, p, o))
    terms.bind(identity.ontology_id, identity.namespace, override=True)
    terms.bind_implicit_namespaces(prefix_base=identity.ontology_id)
    root = Ontology(
        ontology_id=identity.ontology_id,
        iri=identity.iri,
        title=identity.title,
        description=identity.description or None,
        version=label,
        graph=terms,
    )
    stamped = stamp_version(root, now=now)
    assert stamped is not None  # a root is never "unchanged"
    return stamped


def ontology_from_version_graph(graph_uri: str, graph: RDFGraph) -> Ontology:
    """The ontology stored as the version graph ``graph_uri``.

    Its ``bases`` is its own header, so an artifact derived from it is stamped
    as its child.

    Raises:
        CatalogContractError: the graph is not a version of an ontology.
    """
    header = read_header(graph, graph_uri)
    ontology = Ontology(
        ontology_id=header.ontology_id,
        iri=header.iri,
        title=header.title,
        description=header.description,
        version=header.version,
        version_iri=header.version_iri,
        prior_version_iris=list(header.prior_version_iris),
        created_at=header.created_at,
        modified_at=header.modified_at,
        removed=header.removed,
        graph=graph,
    )
    ontology.content_hash()
    header.hash = ontology.hash
    ontology.bases = [header]
    # Author prefix names only survive the store as sh:declare triples.
    ontology.graph.bind_declared_prefixes()
    ontology.graph.bind_implicit_namespaces(prefix_base=ontology.ontology_id)
    ontology.graph.bind(header.ontology_id or "", ontology.namespace, override=True)
    return ontology


def without_version_header(graph: RDFGraph) -> RDFGraph:
    """``graph`` without the version header of any ``owl:Ontology`` subject.

    Returns the same object when there is nothing to drop; otherwise a copy
    that keeps author extras (``owl:imports``, a licence) and prefix bindings.
    """
    subjects = {s for s, _, _ in graph.triples((None, RDF.type, OWL.Ontology))}
    doomed = [
        (s, p, o)
        for s in subjects
        for s_, p, o in graph.triples((s, None, None))
        if p in CONTRACT_PREDICATES or o == OWL.Ontology
    ]
    if not doomed:
        return graph
    declared = {o for _, p, o in doomed if p == SH.declare}
    stripped = graph.copy()
    for triple in doomed:
        stripped.remove(triple)
    for node in declared:
        for triple in list(stripped.triples((node, None, None))):
            stripped.remove(triple)
    return stripped


def tombstone_graph(graph: RDFGraph, graph_uri: str) -> RDFGraph:
    """Header-only copy of ``graph``, marked ``oc:RemovedVersion``.

    Content triples are dropped; the version header stays so lineage through
    this version never dangles. ``graph_uri`` is the named-graph IRI that
    carries the marker.
    """
    header = read_header(graph, graph_uri)
    if not header.version_iri or not header.version or not header.modified_at:
        raise CatalogContractError(graph_uri, ["version_iri", "version", "modified"])
    out = RDFGraph()
    for prefix, namespace in graph.namespaces():
        out.bind(prefix, namespace)
    version = Ontology(
        ontology_id=header.ontology_id,
        iri=header.iri,
        title=header.title,
        description=header.description,
        version=header.version,
        version_iri=header.version_iri,
        prior_version_iris=list(header.prior_version_iris),
        created_at=header.created_at,
        modified_at=header.modified_at,
        removed=True,
        graph=out,
    )
    write_header(out, version, source_doc=None)
    out.add((URIRef(graph_uri), RDF.type, REMOVED_VERSION_TYPE))
    return out


class VersionDiff:
    """Added and removed content triples between two versions.

    Attributes:
        from_iri: The ``from`` version IRI.
        to_iri: The ``to`` version IRI.
        added: Sorted N-Triples lines present only in ``to``.
        removed: Sorted N-Triples lines present only in ``from``.
    """

    __slots__ = ("from_iri", "to_iri", "added", "removed")

    def __init__(
        self,
        *,
        from_iri: str,
        to_iri: str,
        added: list[str],
        removed: list[str],
    ) -> None:
        self.from_iri = from_iri
        self.to_iri = to_iri
        self.added = added
        self.removed = removed

    @property
    def added_count(self) -> int:
        return len(self.added)

    @property
    def removed_count(self) -> int:
        return len(self.removed)


def _ntriples_set(graph: RDFGraph) -> set[str]:
    """Sorted-comparable N-Triples lines for the triples in ``graph``."""
    lines: set[str] = set()
    for subject, predicate, obj in without_version_header(graph):
        triple = RDFGraph()
        triple.add((subject, predicate, obj))
        for line in triple.serialize(format="nt").splitlines():
            text = line.strip()
            if text:
                lines.add(text)
    return lines


def diff_versions(
    from_graph: RDFGraph,
    to_graph: RDFGraph,
    *,
    from_iri: str,
    to_iri: str,
) -> VersionDiff:
    """Content diff of two version graphs, header triples excluded."""
    from_lines = _ntriples_set(from_graph)
    to_lines = _ntriples_set(to_graph)
    return VersionDiff(
        from_iri=from_iri,
        to_iri=to_iri,
        added=sorted(to_lines - from_lines),
        removed=sorted(from_lines - to_lines),
    )


def heads_by_iri(
    versions: Sequence[OntologyHeader],
) -> dict[str, list[OntologyHeader]]:
    """The heads of every ontology among ``versions``.

    A head is a live version that no live version descends from. Descent walks
    through tombstoned versions, so removing a head makes its nearest live
    ancestors the heads; an ontology whose versions are all tombstoned has no
    heads. Sorted by ``modified_at``, newest first.
    """
    by_version = {v.version_iri: v for v in versions if v.version_iri}
    children: dict[str, list[str]] = {}
    for version in versions:
        if not version.version_iri:
            continue
        for prior in version.prior_version_iris:
            children.setdefault(prior, []).append(version.version_iri)

    def has_live_descendant(version_iri: str) -> bool:
        stack = list(children.get(version_iri, []))
        seen: set[str] = set()
        while stack:
            child = stack.pop()
            if child in seen:
                continue
            seen.add(child)
            header = by_version.get(child)
            if header is None:
                continue
            if not header.removed:
                return True
            stack.extend(children.get(child, []))
        return False

    heads: dict[str, list[OntologyHeader]] = {}
    for version in versions:
        if version.removed or not version.version_iri:
            continue
        if has_live_descendant(version.version_iri):
            continue
        heads.setdefault(version.iri, []).append(version)
    for group in heads.values():
        group.sort(
            key=lambda v: v.modified_at or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
    return heads


def catalog_heads(
    versions: Sequence[OntologyHeader],
) -> tuple[list[OntologyHeader], dict[str, list[OntologyHeader]]]:
    """``(one head per ontology, the diverged ontologies)``.

    An ontology with several heads is diverged: it is listed, its versions
    can be read, and nothing extends it until a merge makes one head.
    """
    single: list[OntologyHeader] = []
    diverged: dict[str, list[OntologyHeader]] = {}
    for iri, group in sorted(heads_by_iri(versions).items()):
        if len(group) == 1:
            single.append(group[0])
        else:
            diverged[iri] = group
    return single, diverged


def mint_identity(
    current_domain: str, ontology_id: str, title: str, description: str = ""
) -> OntologyIdentity:
    """The identity of a new ontology ``ontology_id`` under ``current_domain``."""
    return OntologyIdentity(
        ontology_id=ontology_id,
        iri=mint_ontology_iri(current_domain, ontology_id),
        title=title,
        description=description,
    )
