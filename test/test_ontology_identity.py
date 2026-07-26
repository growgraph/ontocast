"""Tests for ontology identity, aliases, working-context, and prefix merge."""

from pathlib import Path

import pytest
from rdflib import OWL, RDF, URIRef

from ontocast.onto.namespace_merge import choose_best_prefix, merge_namespace_bindings
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.util import derive_ontology_id, normalize_ontology_iri
from ontocast.tool.ontology_manager import OntologyManager


def test_derive_ontology_id_skos_conventional() -> None:
    assert derive_ontology_id("http://www.w3.org/2004/02/skos/core#") == "skos"
    assert derive_ontology_id("http://www.w3.org/2004/02/skos/core") == "skos"


def test_derive_ontology_id_foaf_conventional() -> None:
    """rdflib conventional map wins over the opaque ``0.1`` path tail."""
    assert derive_ontology_id("http://xmlns.com/foaf/0.1/") == "foaf"


def test_derive_ontology_id_rejects_pure_numeric_tail() -> None:
    assert derive_ontology_id("https://example.org/ns/0.1/") is None


def test_derive_ontology_id_brick_conventional() -> None:
    assert derive_ontology_id("https://brickschema.org/schema/Brick#") == "brick"


def test_normalize_ontology_iri_strips_delimiters() -> None:
    assert (
        normalize_ontology_iri("https://example.org/ont#")
        == normalize_ontology_iri("https://example.org/ont/")
        == "https://example.org/ont"
    )
    assert (
        normalize_ontology_iri("<https://example.org/ont>") == "https://example.org/ont"
    )


def test_merge_namespace_bindings_renames_conflicts() -> None:
    existing = {"ex": "https://example.org/a#"}
    incoming = {"ex": "https://example.org/b#"}
    merged = merge_namespace_bindings(existing, incoming)
    assert merged["ex"] == "https://example.org/a#"
    renamed = [v for k, v in merged.items() if k != "ex"]
    assert "https://example.org/b#" in renamed


def test_choose_best_prefix_prefers_author() -> None:
    chosen = choose_best_prefix(
        "https://growgraph.dev/ontologies/observation#",
        ["observation", "obs"],
        preferred_namespace_prefixes={
            "https://growgraph.dev/ontologies/observation#": "obs"
        },
    )
    assert chosen == "obs"


def test_rdfgraph_add_preserves_conflicting_prefixes() -> None:
    left = RDFGraph()
    left.bind("ex", URIRef("https://example.org/a#"))
    left.add(
        (
            URIRef("https://example.org/a#A"),
            RDF.type,
            OWL.Class,
        )
    )
    right = RDFGraph()
    right.bind("ex", URIRef("https://example.org/b#"))
    right.add(
        (
            URIRef("https://example.org/b#B"),
            RDF.type,
            OWL.Class,
        )
    )
    merged = left + right
    bindings = {p: str(u) for p, u in merged.namespaces() if p and p.startswith("ex")}
    assert "https://example.org/a#" in bindings.values()
    assert "https://example.org/b#" in bindings.values()


def test_ontology_snapshot_preserves_sources_on_matsci_merge() -> None:
    from ontocast.onto.enum import OntologyAssemblyMode
    from ontocast.onto.ontology_snapshot import OntologySnapshot

    base = Path(
        "/home/alexander/work/codes/gg-core/ontocast/matsci-perovskite-ontologies/ontologies"
    )
    if not base.is_dir():
        pytest.skip("matsci ontology fixtures not available")

    arts = [Ontology.from_file(p) for p in sorted(base.glob("*.ttl"))]
    assert len(arts) >= 2
    merged = RDFGraph()
    sources: list[str] = []
    for ontology in sorted(arts, key=lambda o: o.iri or ""):
        merged += ontology.graph
        if ontology.iri:
            sources.append(ontology.iri)
    merged.sanitize_prefixes_namespaces()
    snap = OntologySnapshot.from_graph(
        merged,
        source_iris=sources,
        assembly_mode=OntologyAssemblyMode.DOCUMENT_MERGED_REDUCED,
        title="Merged",
    )
    assert snap.source_iris == sources
    assert len(snap.graph) > 0


def test_ontology_manager_aliases_prefix_and_id() -> None:
    graph = RDFGraph._from_turtle_str(
        """
        @prefix obs: <https://growgraph.dev/ontologies/observation#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        <https://growgraph.dev/ontologies/observation> a owl:Ontology .
        """
    )
    ontology = Ontology(
        graph=graph,
        iri="https://growgraph.dev/ontologies/observation",
        ontology_id="observation",
    )
    mgr = OntologyManager()
    mgr.add_ontology(ontology)

    assert mgr.resolve_ontology_ref("observation") == ontology.iri
    assert mgr.resolve_ontology_ref("obs") == ontology.iri
    assert mgr.resolve_ontology_ref(ontology.iri) == ontology.iri
    assert "obs" in mgr
    freshest = mgr.get_freshest_terminal_ontology(ontology_id="obs")
    assert freshest is not None
    assert freshest.iri == ontology.iri


def test_ontology_manager_allows_prefix_differing_from_ontology_id() -> None:
    graph = RDFGraph._from_turtle_str(
        """
        @prefix life: <https://growgraph.dev/ontologies/lifecycle#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        <https://growgraph.dev/ontologies/lifecycle> a owl:Ontology .
        """
    )
    ontology = Ontology(
        graph=graph,
        iri="https://growgraph.dev/ontologies/lifecycle",
        ontology_id="lifecycle",
    )
    assert ontology.prefix == "life"
    mgr = OntologyManager()
    mgr.add_ontology(ontology)  # must not raise
    assert mgr.resolve_ontology_ref("life") == ontology.iri
    assert mgr.resolve_ontology_ref("lifecycle") == ontology.iri


def test_ontology_manager_rejects_same_iri_different_ontology_id() -> None:
    base_graph = RDFGraph._from_turtle_str(
        """
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        <https://example.org/finance> a owl:Ontology .
        """
    )
    ontology_a = Ontology(
        graph=base_graph,
        iri="https://example.org/finance",
        ontology_id="finance",
    )
    mgr = OntologyManager()
    mgr.add_ontology(ontology_a)

    conflicting_graph = RDFGraph._from_turtle_str(
        """
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        <https://example.org/finance> a owl:Ontology .
        <https://example.org/finance#extra> a owl:Class .
        """
    )
    ontology_b = Ontology(
        graph=conflicting_graph,
        iri="https://example.org/finance",
        ontology_id="accounting",
    )
    with pytest.raises(ValueError, match="already bound to identity"):
        mgr.add_ontology(ontology_b)


def test_ontology_manager_rejects_alias_across_iris() -> None:
    graph_a = RDFGraph._from_turtle_str(
        """
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        <https://example.org/finance> a owl:Ontology .
        """
    )
    ontology_a = Ontology(
        graph=graph_a,
        iri="https://example.org/finance",
        ontology_id="finance",
    )
    mgr = OntologyManager()
    mgr.add_ontology(ontology_a)

    graph_b = RDFGraph._from_turtle_str(
        """
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        <https://example.com/finance> a owl:Ontology .
        """
    )
    ontology_b = Ontology(
        graph=graph_b,
        iri="https://example.com/finance",
        ontology_id="finance",
    )
    with pytest.raises(ValueError, match="already bound to IRI"):
        mgr.add_ontology(ontology_b)


def test_no_owl_ontology_does_not_promote_prefix() -> None:
    graph = RDFGraph()
    graph.bind("ex", URIRef("https://example.org/mystery#"))
    graph.add(
        (
            URIRef("https://example.org/mystery#Thing"),
            RDF.type,
            OWL.Class,
        )
    )
    ontology = Ontology(graph=graph)
    assert ontology.ontology_id is None
    assert list(ontology.graph.triples((None, RDF.type, OWL.Ontology))) == []
