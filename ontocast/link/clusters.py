"""Read-time clusters: which stored entities are one, from the resolution graph.

Links only propose: two entities linked to one match node are candidates.
A cluster joins candidates in a fixed order (stronger evidence first, then
by IRI) and never joins two clusters that a veto separates, so the answer
depends on the stored documents and declarations, not on their order.

Vetoes, all declared or structural:

- both entities are in one document graph (its aggregation kept them apart);
- their types are declared disjoint;
- a property the catalog caps at one value has disjoint values on each;
- a key that applies to both has different complete values.

An object IRI local to one document cannot be compared with another
document's, so it never counts as a conflicting value.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

from pydantic import BaseModel, Field
from rdflib import RDF, RDFS, SKOS, Literal, URIRef
from rdflib.term import Node

from ontocast.onto.constants import ONTOCAST, is_auxiliary_graph
from ontocast.tool.triple_manager.core import TripleStoreManager

from .declarations import IdentityDeclarations
from .links import LINK_KINDS, key_value

#: Cap on the candidates one cluster read collects; a larger component is
#: reported as truncated rather than read in full.
DEFAULT_MAX_CANDIDATES = 500

_VALUES_CHUNK = 200
_PREFIXES = f"PREFIX skos: <{SKOS}>\nPREFIX ontocast: <{ONTOCAST}>\n"


class ClusterMember(BaseModel):
    """One entity of a cluster."""

    iri: str
    graph: str = Field(description="The document graph that holds the entity.")
    labels: list[str] = Field(default_factory=list)
    types: list[str] = Field(default_factory=list)
    reason: str = Field(
        description="`self` for the entity asked about, else the strongest kind "
        "of link it shares with the cluster (key, ifp, identifier, label)."
    )
    via: list[str] = Field(
        default_factory=list, description="Match nodes it shares with the cluster."
    )


class KeptApart(BaseModel):
    """A candidate that shares a match node with the cluster but is not in it."""

    iri: str
    graph: str
    veto: str = Field(description="Why it is not one entity with the cluster.")


class EntityCluster(BaseModel):
    """The entities stored as one with ``iri``, and the candidates kept apart."""

    iri: str
    members: list[ClusterMember]
    kept_apart: list[KeptApart] = Field(default_factory=list)
    truncated: bool = Field(
        default=False,
        description="The candidate set reached its cap; some candidates were not read.",
    )


class _Entity:
    """What a cluster needs to know about one stored entity."""

    def __init__(self, iri: str) -> None:
        self.iri = iri
        self.graph = ""
        self.types: set[str] = set()
        self.labels: set[str] = set()
        self.values: dict[str, set[Node]] = defaultdict(set)


async def cluster_of(
    store: TripleStoreManager,
    declarations: IdentityDeclarations,
    iri: str,
    *,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> EntityCluster | None:
    """The cluster of the stored entity ``iri``, or None when no document holds it.

    Args:
        store: The triple store holding the documents and their links graphs.
        declarations: The catalog's identity declarations.
        iri: An entity of some document graph.
        max_candidates: Stop collecting candidates beyond this many.
    """
    nodes, edges, truncated = await _candidates(store, iri, max_candidates)
    candidates = sorted({iri} | {e for _, members in nodes.items() for e in members})
    entities = await _read_entities(store, candidates)
    if iri not in entities:
        return None
    ordered_edges = _ordered_edges(nodes, edges, entities)
    parent = {e: e for e in entities}
    members_of = {e: [e] for e in entities}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, _, left, right in ordered_edges:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            continue
        if any(
            _veto(entities[a], entities[b], declarations)
            for a in members_of[root_left]
            for b in members_of[root_right]
        ):
            continue
        keep, drop = sorted((root_left, root_right))
        parent[drop] = keep
        members_of[keep].extend(members_of.pop(drop))

    root = find(iri)
    cluster = sorted(members_of[root])
    in_cluster = set(cluster)
    shared = _shared_nodes(nodes, edges, in_cluster)
    members = [
        ClusterMember(
            iri=e,
            graph=entities[e].graph,
            labels=sorted(entities[e].labels),
            types=sorted(entities[e].types),
            reason="self" if e == iri else _strongest(shared.get(e, [])),
            via=sorted(node for node, _ in shared.get(e, [])),
        )
        for e in cluster
    ]
    kept_apart = [
        KeptApart(
            iri=e,
            graph=entities[e].graph,
            veto=next(
                (
                    reason
                    for other in cluster
                    if (reason := _veto(entities[e], entities[other], declarations))
                ),
                "joined another cluster",
            ),
        )
        for e in sorted(entities)
        if e not in in_cluster
    ]
    return EntityCluster(
        iri=iri, members=members, kept_apart=kept_apart, truncated=truncated
    )


async def _candidates(
    store: TripleStoreManager, iri: str, max_candidates: int
) -> tuple[dict[str, set[str]], dict[str, str], bool]:
    """Match nodes reachable from ``iri`` through links, with their members.

    Returns:
        ``(members by node, kind by node, truncated)``.
    """
    members: dict[str, set[str]] = {}
    kinds: dict[str, str] = {}
    seen: set[str] = {iri}
    frontier = [iri]
    while frontier:
        new_nodes: list[str] = []
        for chunk in _chunks(frontier):
            rows = await store.aselect(
                _PREFIXES
                + "SELECT DISTINCT ?e ?m ?kind WHERE { VALUES ?e { "
                + " ".join(f"<{e}>" for e in chunk)
                + " } ?e skos:closeMatch ?m . ?m ontocast:matchKind ?kind . }",
                store="facts",
                union=True,
            )
            for row in rows:
                node = row["m"]
                if node not in kinds:
                    kinds[node] = row["kind"]
                    new_nodes.append(node)
        frontier = []
        for chunk in _chunks(sorted(new_nodes)):
            rows = await store.aselect(
                _PREFIXES
                + "SELECT DISTINCT ?e ?m WHERE { VALUES ?m { "
                + " ".join(f"<{m}>" for m in chunk)
                + " } ?e skos:closeMatch ?m . }",
                store="facts",
                union=True,
            )
            for row in sorted(rows, key=lambda r: (r["m"], r["e"])):
                members.setdefault(row["m"], set()).add(row["e"])
                if row["e"] not in seen:
                    if len(seen) >= max_candidates:
                        return members, kinds, True
                    seen.add(row["e"])
                    frontier.append(row["e"])
        frontier.sort()
    return members, kinds, False


async def _read_entities(
    store: TripleStoreManager, iris: Sequence[str]
) -> dict[str, _Entity]:
    """Types, labels and values of each entity, from its document graph."""
    entities: dict[str, _Entity] = {}
    for chunk in _chunks(list(iris)):
        rows = await store.aselect(
            "SELECT ?e ?g ?p ?o (isIRI(?o) AS ?isiri) (STR(DATATYPE(?o)) AS ?dt) "
            "(LANG(?o) AS ?lang) WHERE { VALUES ?e { "
            + " ".join(f"<{e}>" for e in chunk)
            + " } GRAPH ?g { ?e ?p ?o } }",
            store="facts",
        )
        for row in rows:
            graph = row["g"]
            if is_auxiliary_graph(graph):
                continue
            entity = entities.setdefault(row["e"], _Entity(row["e"]))
            entity.graph = min(filter(None, (entity.graph, graph)))
            term = _term(row)
            if row["p"] == str(RDF.type) and isinstance(term, URIRef):
                entity.types.add(str(term))
            elif row["p"] == str(RDFS.label) and isinstance(term, Literal):
                entity.labels.add(str(term))
            entity.values[row["p"]].add(term)
    return {iri: e for iri, e in entities.items() if e.types}


def _term(row: dict[str, str]) -> Node:
    if row.get("isiri") == "true":
        return URIRef(row["o"])
    return Literal(
        row["o"],
        lang=row.get("lang") or None,
        datatype=None if row.get("lang") else (row.get("dt") or None),
    )


def _comparable(
    entity: _Entity, prop: str, declarations: IdentityDeclarations
) -> set[str]:
    """Values of ``prop`` that mean the same in every document."""
    return {
        value
        for term in entity.values.get(prop, ())
        if (value := key_value(term, declarations.catalog_terms)) is not None
    }


def _veto(
    left: _Entity, right: _Entity, declarations: IdentityDeclarations
) -> str | None:
    """Why ``left`` and ``right`` cannot be one entity, or None."""
    if left.graph == right.graph:
        return "same document"
    if declarations.are_disjoint(left.types, right.types):
        return "disjoint types"
    for prop in sorted(declarations.functional):
        left_values = _comparable(left, prop, declarations)
        right_values = _comparable(right, prop, declarations)
        if left_values and right_values and left_values.isdisjoint(right_values):
            return f"functional {prop}"
    left_keys = set(declarations.keys_for(left.types))
    for cls_iri, props in sorted(left_keys & set(declarations.keys_for(right.types))):
        left_key = [_comparable(left, p, declarations) for p in props]
        right_key = [_comparable(right, p, declarations) for p in props]
        if (
            all(left_key)
            and all(right_key)
            and any(a.isdisjoint(b) for a, b in zip(left_key, right_key, strict=True))
        ):
            return f"key {cls_iri}"
    return None


def _ordered_edges(
    nodes: dict[str, set[str]],
    kinds: dict[str, str],
    entities: dict[str, _Entity],
) -> list[tuple[int, str, str, str]]:
    """Candidate pairs, strongest kind first, then by node and IRIs."""
    edges: set[tuple[int, str, str, str]] = set()
    for node, members in nodes.items():
        rank = _rank(kinds.get(node, "label"))
        present = sorted(m for m in members if m in entities)
        for i, left in enumerate(present):
            for right in present[i + 1 :]:
                edges.add((rank, node, left, right))
    return sorted(edges)


def _shared_nodes(
    nodes: dict[str, set[str]], kinds: dict[str, str], cluster: set[str]
) -> dict[str, list[tuple[str, str]]]:
    """For each cluster member, the match nodes it shares with another member."""
    shared: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for node, members in nodes.items():
        inside = members & cluster
        if len(inside) < 2:
            continue
        for member in inside:
            shared[member].append((node, kinds.get(node, "label")))
    return shared


def _strongest(shared: Iterable[tuple[str, str]]) -> str:
    kinds = sorted((kind for _, kind in shared), key=_rank)
    return kinds[0] if kinds else "label"


def _rank(kind: str) -> int:
    return LINK_KINDS.index(kind) if kind in LINK_KINDS else len(LINK_KINDS)


def _chunks(items: Sequence[str]) -> list[Sequence[str]]:
    return [items[i : i + _VALUES_CHUNK] for i in range(0, len(items), _VALUES_CHUNK)]
