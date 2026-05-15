from __future__ import annotations

from rdflib import SKOS

from ontocast.onto.facts import Facts
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.vector_store.atomizer import GraphAtomizer


def test_atomizer_filters_minimal_provenance_predicates() -> None:
    """Provenance/reification triples should not leak into embeddings."""
    facts_namespace = "https://example.org/facts"

    graph = RDFGraph._from_turtle_str(
        """
        @prefix cd: <https://example.org/facts/> .
        @prefix prov: <http://www.w3.org/ns/prov#> .
        @prefix dcterms: <http://purl.org/dc/terms/> .
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix ex: <https://example.org/test/> .

        cd:Alpha a ex:Thing ;
            ex:knows cd:Beta ;
            dcterms:source ex:Doc .

        cd:Beta a ex:Thing .
        cd:Alpha prov:wasDerivedFrom ex:Other .
        cd:Alpha rdf:reifies ex:Whatever .
        """
    )

    facts = Facts(
        graph=graph,
        iri="https://example.org/factsGraph#doc1",
        ontology_id="doc1",
        hash="hash1",
        version="1.0.0",
        facts_namespace=facts_namespace,
    )

    atoms = GraphAtomizer().atomize(source=facts, depth=1)
    assert atoms

    alpha_atom = next(a for a in atoms if a.iri.endswith("/Alpha"))
    assert "was derived from" not in alpha_atom.core_representation
    assert "reifies" not in alpha_atom.core_representation

    assert "was derived from" not in alpha_atom.neighborhood_representation
    assert "has relation source" not in alpha_atom.neighborhood_representation
    assert "reifies" not in alpha_atom.neighborhood_representation


def test_atomizer_facts_focal_entities_are_cd_only() -> None:
    """Facts atomization should only create atoms for `cd:`-namespaced entities."""
    facts_namespace = "https://example.org/facts"
    outside_ns = "https://example.org/outside#"

    graph = RDFGraph._from_turtle_str(
        f"""
        @prefix cd: <{facts_namespace}/> .
        @prefix ex: <https://example.org/test/> .
        @prefix out: <{outside_ns}> .

        cd:Alpha a ex:Thing ;
            ex:relatedTo out:Outside .
        out:Outside a ex:Thing .
        """
    )

    facts = Facts(
        graph=graph,
        iri="https://example.org/factsGraph#doc1",
        ontology_id="doc1",
        hash="hash1",
        version="1.0.0",
        facts_namespace=facts_namespace,
    )

    atoms = GraphAtomizer().atomize(source=facts, depth=1)
    assert atoms
    assert all(a.iri.startswith(facts_namespace) for a in atoms)
    assert not any(outside_ns in a.iri for a in atoms)


def test_atomizer_facts_core_representation_includes_skos_alt_label() -> None:
    facts_namespace = "https://example.org/facts"
    graph = RDFGraph._from_turtle_str(
        f"""
        @prefix cd: <{facts_namespace}/> .
        @prefix skos: <{SKOS}> .
        @prefix ex: <https://example.org/test/> .

        cd:Entity a ex:Thing ;
            skos:prefLabel "Main label"@en ;
            skos:altLabel "Alternate name"@en .
        """
    )
    facts = Facts(
        graph=graph,
        iri="https://example.org/factsGraph#doc1",
        ontology_id="doc1",
        hash="hash1",
        version="1.0.0",
        facts_namespace=facts_namespace,
    )
    atom = next(
        a
        for a in GraphAtomizer().atomize(source=facts, depth=0)
        if a.iri.endswith("/Entity")
    )
    core = atom.core_representation.lower()
    assert "main label" in core
    assert "alternate name" in core
