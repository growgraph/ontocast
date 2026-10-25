"""The version DAG: heads, divergence, removed versions, labels per IRI."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from rdflib import OWL, RDF, RDFS, Literal, URIRef

from ontocast.onto.ontology import Ontology, OntologyHeader
from ontocast.onto.ontology_header import (
    DivergedOntologyError,
    catalog_heads,
    heads_by_iri,
    stamp_version,
)
from ontocast.onto.ontology_operations import merge_ontologies
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import GraphUpdate, TripleOp
from ontocast.tool.ontology_manager import OntologyManager
from test.ontology_helpers import stamped_ontology

pytestmark = pytest.mark.unit

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
T1 = datetime(2024, 1, 2, tzinfo=timezone.utc)
T2 = datetime(2024, 1, 3, tzinfo=timezone.utc)
T3 = datetime(2024, 1, 4, tzinfo=timezone.utc)


def _root(ontology_id: str, now: datetime = T0) -> Ontology:
    graph = RDFGraph()
    iri = f"https://example.org/{ontology_id}"
    graph.add((URIRef(iri), RDF.type, OWL.Ontology))
    graph.add((URIRef(f"{iri}#Thing"), RDF.type, OWL.Class))
    return stamped_ontology(graph, ontology_id=ontology_id, now=now)


def _with_classes(base: Ontology, *locals_: str) -> tuple[RDFGraph, GraphUpdate]:
    """``base``'s graph plus the classes, and the update that inserts them."""
    delta = RDFGraph()
    for local in locals_:
        term = URIRef(f"{base.iri}#{local}")
        delta.add((term, RDF.type, OWL.Class))
        delta.add((term, RDFS.label, Literal(local)))
    graph = base.graph.copy()
    for triple in delta:
        graph.add(triple)
    return graph, GraphUpdate(triple_operations=[TripleOp(type="insert", graph=delta)])


def _child(base: Ontology, *locals_: str, now: datetime) -> Ontology:
    graph, update = _with_classes(base, *locals_)
    stamped = stamp_version(
        base.derive_updated_version(graph), updates=[update], now=now
    )
    assert stamped is not None
    return stamped


def _headers(*versions: Ontology) -> list[OntologyHeader]:
    return [OntologyHeader.from_ontology(v) for v in versions]


def test_two_children_of_one_version_are_two_heads() -> None:
    root = _root("tmi")
    left = _child(root, "Left", now=T1)
    right = _child(root, "Right", now=T2)

    heads = heads_by_iri(_headers(root, left, right))

    assert set(heads) == {root.iri}
    assert [h.version_iri for h in heads[root.iri]] == [
        right.version_iri,
        left.version_iri,
    ], "newest first"
    single, diverged = catalog_heads(_headers(root, left, right))
    assert single == []
    assert set(diverged) == {root.iri}


def test_a_diverged_ontology_cannot_be_extended_but_its_heads_are_listed() -> None:
    root = _root("tmi")
    left = _child(root, "Left", now=T1)
    right = _child(root, "Right", now=T2)
    manager = OntologyManager()
    for version in (root, left, right):
        manager.add_ontology(version)

    with pytest.raises(DivergedOntologyError) as info:
        manager.get_head_by_iri(root.iri)
    assert info.value.iri == root.iri
    assert set(info.value.heads) == {left.version_iri, right.version_iri}
    with pytest.raises(DivergedOntologyError):
        manager.get_head("tmi")

    assert {o.version_iri for o in manager.get_heads_by_iri(root.iri)} == {
        left.version_iri,
        right.version_iri,
    }
    assert manager.ontologies == []
    assert root.iri in manager.get_ontology_iris()


def test_a_diverged_ontology_does_not_hide_the_others() -> None:
    tmi = _root("tmi")
    manager = OntologyManager()
    for version in (tmi, _child(tmi, "Left", now=T1), _child(tmi, "Right", now=T2)):
        manager.add_ontology(version)
    other = _root("other")
    manager.add_ontology(other)

    assert [o.iri for o in manager.ontologies] == [other.iri]


def test_a_removed_middle_version_is_not_a_head_and_lineage_still_resolves() -> None:
    root = _root("tmi")
    middle = _child(root, "Middle", now=T1)
    tip = _child(middle, "Tip", now=T2)
    root_h, middle_h, tip_h = _headers(root, middle, tip)
    middle_h.removed = True

    heads = heads_by_iri([root_h, middle_h, tip_h])
    assert [h.version_iri for h in heads[root.iri]] == [tip.version_iri]
    assert root.version_iri not in {h.version_iri for h in heads[root.iri]}

    lineage = Ontology.build_lineage_graph([root_h, middle_h, tip_h])
    assert lineage.nodes[middle.version_iri]["removed"] is True
    assert list(lineage.successors(tip.version_iri)) == [middle.version_iri]
    assert list(lineage.successors(middle.version_iri)) == [root.version_iri]


def test_removing_a_head_falls_back_to_its_nearest_live_ancestors() -> None:
    root = _root("tmi")
    middle = _child(root, "Middle", now=T1)
    tip = _child(middle, "Tip", now=T2)
    root_h, middle_h, tip_h = _headers(root, middle, tip)

    tip_h.removed = True
    heads = heads_by_iri([root_h, middle_h, tip_h])
    assert [h.version_iri for h in heads[root.iri]] == [middle.version_iri]

    middle_h.removed = True
    heads = heads_by_iri([root_h, middle_h, tip_h])
    assert [h.version_iri for h in heads[root.iri]] == [root.version_iri]


def test_removing_every_version_leaves_no_heads() -> None:
    root = _root("tmi")
    tip = _child(root, "Tip", now=T1)
    root_h, tip_h = _headers(root, tip)
    root_h.removed = True
    tip_h.removed = True

    assert heads_by_iri([root_h, tip_h]) == {}


def test_removing_one_branch_of_a_diverged_ontology_leaves_a_single_head() -> None:
    root = _root("tmi")
    left = _child(root, "Left", now=T1)
    right = _child(root, "Right", now=T2)
    root_h, left_h, right_h = _headers(root, left, right)
    left_h.removed = True

    single, diverged = catalog_heads([root_h, left_h, right_h])
    assert [h.version_iri for h in single] == [right.version_iri]
    assert diverged == {}


def test_build_lineage_graph_edges_go_from_child_to_prior() -> None:
    root = _root("tmi")
    left = _child(root, "Left", now=T1)
    right = _child(root, "Right", now=T2)
    merged = stamp_version(merge_ontologies(left, right), now=T3)
    assert merged is not None

    lineage = Ontology.build_lineage_graph([root, left, right, merged])

    assert set(lineage.nodes) == {
        root.version_iri,
        left.version_iri,
        right.version_iri,
        merged.version_iri,
    }
    assert set(lineage.edges) == {
        (left.version_iri, root.version_iri),
        (right.version_iri, root.version_iri),
        (merged.version_iri, left.version_iri),
        (merged.version_iri, right.version_iri),
    }
    assert lineage.nodes[root.version_iri]["version"] == "1.0.0"
    assert lineage.nodes[merged.version_iri]["iri"] == root.iri


def test_a_prior_held_elsewhere_is_still_a_node() -> None:
    root = _root("tmi")
    child = _child(root, "Child", now=T1)

    lineage = Ontology.build_lineage_graph([child])

    assert set(lineage.nodes) == {child.version_iri, root.version_iri}


def test_labels_are_per_iri() -> None:
    a = _root("alpha")
    b = _root("beta")
    a_updated = _child(a, "C1", "C2", "C3", "C4", "C5", now=T1)

    assert a.version == "1.0.0"
    assert a_updated.version == "1.1.0", "five new classes is a minor bump"
    assert b.version == "1.0.0"
    manager = OntologyManager()
    for version in (a, a_updated, b):
        manager.add_ontology(version)
    assert {o.iri: o.version for o in manager.ontologies} == {
        a.iri: "1.1.0",
        b.iri: "1.0.0",
    }


def test_a_root_stamps_as_1_0_0() -> None:
    assert _root("tmi").version == "1.0.0"


def test_a_merge_stamps_with_two_priors_and_the_higher_label_bumped() -> None:
    root = _root("tmi")
    patch = _child(root, "One", now=T1)
    minor = _child(root, "C1", "C2", "C3", "C4", "C5", now=T2)
    assert (patch.version, minor.version) == ("1.0.1", "1.1.0")

    merged = stamp_version(merge_ontologies(patch, minor), now=T3)

    assert merged is not None
    assert merged.prior_version_iris == [patch.version_iri, minor.version_iri]
    assert merged.version == "1.1.1"
    assert merged.created_at == root.created_at
    manager = OntologyManager()
    for version in (root, patch, minor, merged):
        manager.add_ontology(version)
    assert manager.get_head_by_iri(root.iri) is manager.get_ontology(
        version_iri=merged.version_iri
    )


def test_merge_label_orders_versions_numerically() -> None:
    """``1.10.0`` is newer than ``1.9.0``; the merge bumps the numeric maximum."""
    from rdflib import OWL, RDF, URIRef

    from ontocast.onto.ontology import Ontology, OntologyHeader
    from ontocast.onto.ontology_header import stamp_version
    from ontocast.onto.rdfgraph import RDFGraph

    iri = "https://example.org/semver"
    graph = RDFGraph()
    graph.add((URIRef(f"{iri}#A"), RDF.type, OWL.Class))
    bases = [
        OntologyHeader(
            graph_uri=f"{iri}#{key}",
            iri=iri,
            ontology_id="semver",
            title="Semver",
            version=label,
            version_iri=f"{iri}#{key}",
        )
        for key, label in (("a", "1.10.0"), ("b", "1.9.0"))
    ]
    merged = Ontology(
        ontology_id="semver", iri=iri, title="Semver", graph=graph, bases=bases
    )
    stamped = stamp_version(merged)
    assert stamped is not None
    assert stamped.version == "1.10.1"
