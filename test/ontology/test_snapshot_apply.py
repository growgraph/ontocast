"""Tests for OntologySnapshot and namespace-ownership apply."""

from __future__ import annotations

from rdflib import OWL, RDF, URIRef

from ontocast.agent.normalize_ontology import normalize_ontology_units
from ontocast.onto.enum import OntologyAssemblyMode
from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_apply import (
    apply_partitioned_updates,
    complement_inserts,
    partition_triples_by_namespace,
)
from ontocast.onto.ontology_snapshot import OntologySnapshot
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool import OntologyManager


def _onto_graph(iri: str, local: str, prefix: str) -> RDFGraph:
    g = RDFGraph()
    ns = f"{iri}#"
    g.bind(prefix, ns)
    g.add((URIRef(iri), RDF.type, OWL.Ontology))
    g.add((URIRef(f"{ns}{local}"), RDF.type, OWL.Class))
    return g


def test_ontology_snapshot_from_ontology_copies_graph_and_source() -> None:
    iri = "https://example.org/a"
    ontology = Ontology(graph=_onto_graph(iri, "A", "exa"), iri=iri)
    snap = OntologySnapshot.from_ontology(
        ontology, assembly_mode=OntologyAssemblyMode.FIXED_SINGLE_ONTOLOGY
    )
    assert snap.source_iris == [iri]
    assert len(snap.graph) == len(ontology.graph)
    assert snap.graph is not ontology.graph
    assert "iri" not in OntologySnapshot.model_fields


def test_ontology_snapshot_from_graph_preserves_sources() -> None:
    g = RDFGraph()
    g.add(
        (
            URIRef("https://example.org/a#A"),
            RDF.type,
            OWL.Class,
        )
    )
    snap = OntologySnapshot.from_graph(
        g,
        source_iris=["https://example.org/a", "https://example.org/b"],
        assembly_mode=OntologyAssemblyMode.SELECTED_VECTOR_SEARCH_ENSEMBLE,
    )
    assert snap.source_iris == ["https://example.org/a", "https://example.org/b"]
    assert not snap.is_empty()
    assert "Source catalog IRIs" in snap.describe_for_prompt()


def test_complement_inserts_subtracts_snapshot_triples() -> None:
    snap = RDFGraph()
    s = URIRef("https://example.org/a#Existing")
    snap.add((s, RDF.type, OWL.Class))
    inserts = RDFGraph()
    inserts.add((s, RDF.type, OWL.Class))
    new = URIRef("https://example.org/a#New")
    inserts.add((new, RDF.type, OWL.Class))
    result = complement_inserts(inserts, snap)
    assert (new, RDF.type, OWL.Class) in result
    assert (s, RDF.type, OWL.Class) not in result


def test_partition_triples_by_namespace_routes_to_owner() -> None:
    iri_a = "https://example.org/a"
    iri_b = "https://example.org/b"
    mgr = OntologyManager()
    mgr.add_ontology(
        Ontology(graph=_onto_graph(iri_a, "A", "exa"), iri=iri_a),
        skip_vector_index=True,
    )
    mgr.add_ontology(
        Ontology(graph=_onto_graph(iri_b, "B", "exb"), iri=iri_b),
        skip_vector_index=True,
    )

    inserts = RDFGraph()
    inserts.add((URIRef(f"{iri_a}#NewA"), RDF.type, OWL.Class))
    inserts.add((URIRef(f"{iri_b}#NewB"), RDF.type, OWL.Class))
    inserts.add(
        (
            URIRef("http://www.w3.org/2000/01/rdf-schema#label"),
            RDF.type,
            OWL.DatatypeProperty,
        )
    )

    partitioned, unattributed = partition_triples_by_namespace(
        inserts,
        writable_iris=[iri_a, iri_b],
        ontology_manager=mgr,
    )
    assert iri_a in partitioned
    assert iri_b in partitioned
    assert (URIRef(f"{iri_a}#NewA"), RDF.type, OWL.Class) in partitioned[iri_a]
    assert (URIRef(f"{iri_b}#NewB"), RDF.type, OWL.Class) in partitioned[iri_b]
    assert unattributed >= 1


def test_apply_partitioned_updates_deletes_then_inserts_on_catalog_base() -> None:
    """Delete deltas reach the catalog artifact; applied ops carry true types."""
    iri = "https://example.org/a"
    ns = f"{iri}#"
    mgr = OntologyManager()
    base_graph = _onto_graph(iri, "A", "exa")
    obsolete = URIRef(f"{ns}Obsolete")
    base_graph.add((obsolete, RDF.type, OWL.Class))
    base = Ontology(graph=base_graph, iri=iri)
    mgr.add_ontology(base, skip_vector_index=True)

    new_class = URIRef(f"{ns}New")
    inserts = RDFGraph()
    inserts.bind("exa", ns)
    inserts.add((new_class, RDF.type, OWL.Class))
    deletes = RDFGraph()
    deletes.bind("exa", ns)
    deletes.add((obsolete, RDF.type, OWL.Class))

    artifacts, metrics, applied_updates = apply_partitioned_updates(
        {iri: inserts},
        ontology_manager=mgr,
        normalize_units_fn=normalize_ontology_units,
        tools=None,
        partitioned_deletes={iri: deletes},
    )

    assert len(artifacts) == 1
    result = artifacts[0]
    assert (new_class, RDF.type, OWL.Class) in result.graph
    assert (obsolete, RDF.type, OWL.Class) not in result.graph
    assert metrics["apply_delete_triples"] == 1
    assert metrics["apply_insert_triples"] == 1
    op_types = [
        op.type for update in applied_updates for op in update.triple_operations
    ]
    assert op_types == ["delete", "insert"]


def test_apply_partitioned_updates_delete_only_produces_artifact() -> None:
    """A pure-delete delta still derives an updated catalog artifact."""
    iri = "https://example.org/a"
    ns = f"{iri}#"
    mgr = OntologyManager()
    base_graph = _onto_graph(iri, "A", "exa")
    obsolete = URIRef(f"{ns}Obsolete")
    base_graph.add((obsolete, RDF.type, OWL.Class))
    mgr.add_ontology(Ontology(graph=base_graph, iri=iri), skip_vector_index=True)

    deletes = RDFGraph()
    deletes.bind("exa", ns)
    deletes.add((obsolete, RDF.type, OWL.Class))

    artifacts, metrics, applied_updates = apply_partitioned_updates(
        {},
        ontology_manager=mgr,
        normalize_units_fn=normalize_ontology_units,
        tools=None,
        partitioned_deletes={iri: deletes},
    )

    assert len(artifacts) == 1
    assert (obsolete, RDF.type, OWL.Class) not in artifacts[0].graph
    assert metrics["apply_delete_triples"] == 1
    assert [op.type for u in applied_updates for op in u.triple_operations] == [
        "delete"
    ]


def test_apply_partitioned_updates_base_override_wins_over_terminal() -> None:
    """base_overrides applies the delta onto the in-run artifact, not the catalog."""
    iri = "https://example.org/a"
    ns = f"{iri}#"
    mgr = OntologyManager()
    stale = Ontology(graph=_onto_graph(iri, "A", "exa"), iri=iri)
    mgr.add_ontology(stale, skip_vector_index=True)

    map_stage_class = URIRef(f"{ns}MapStage")
    primary_graph = _onto_graph(iri, "A", "exa")
    primary_graph.add((map_stage_class, RDF.type, OWL.Class))
    primary = stale.derive_updated_version(primary_graph)

    consolidation_class = URIRef(f"{ns}Consolidated")
    inserts = RDFGraph()
    inserts.bind("exa", ns)
    inserts.add((consolidation_class, RDF.type, OWL.Class))

    artifacts, _metrics, _applied = apply_partitioned_updates(
        {iri: inserts},
        ontology_manager=mgr,
        normalize_units_fn=normalize_ontology_units,
        tools=None,
        base_overrides={iri: primary},
    )

    assert len(artifacts) == 1
    result = artifacts[0]
    # Map-stage additions survive consolidation apply.
    assert (map_stage_class, RDF.type, OWL.Class) in result.graph
    assert (consolidation_class, RDF.type, OWL.Class) in result.graph
