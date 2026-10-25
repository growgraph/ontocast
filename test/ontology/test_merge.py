"""Merging two versions of one ontology into an artifact with two bases."""

from datetime import datetime, timezone

import pytest
from rdflib import OWL, RDF, RDFS, Literal, URIRef

from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_header import (
    DivergedOntologyError,
    stamp_version,
    without_version_header,
)
from ontocast.onto.ontology_operations import merge_ontologies
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.ontology_manager import OntologyManager
from test.ontology_helpers import next_version, stamped_ontology

pytestmark = pytest.mark.unit

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
T1 = datetime(2024, 1, 2, tzinfo=timezone.utc)
T2 = datetime(2024, 1, 3, tzinfo=timezone.utc)
T3 = datetime(2024, 1, 4, tzinfo=timezone.utc)

IRI = "http://example.org/test"


def _class(graph: RDFGraph, local: str, label: str) -> URIRef:
    term = URIRef(f"{IRI}#{local}")
    graph.add((term, RDF.type, OWL.Class))
    graph.add((term, RDFS.label, Literal(label)))
    return term


def _root_graph() -> RDFGraph:
    graph = RDFGraph()
    graph.add((URIRef(IRI), RDF.type, OWL.Ontology))
    graph.add((URIRef(IRI), RDFS.label, Literal("Test Ontology")))
    _class(graph, "Class1", "Class 1")
    return graph


@pytest.fixture
def ontology_manager() -> OntologyManager:
    return OntologyManager()


@pytest.fixture
def base_ontology() -> Ontology:
    return stamped_ontology(_root_graph(), ontology_id="test", now=T0)


@pytest.fixture
def branch1_ontology(base_ontology: Ontology) -> Ontology:
    graph = base_ontology.graph.copy()
    _class(graph, "Class2", "Class 2")
    return next_version(base_ontology, graph, now=T1)


@pytest.fixture
def branch2_ontology(base_ontology: Ontology) -> Ontology:
    graph = base_ontology.graph.copy()
    _class(graph, "Class3", "Class 3")
    return next_version(base_ontology, graph, now=T2)


def test_merge_ontologies_has_both_bases_and_all_content(
    branch1_ontology: Ontology, branch2_ontology: Ontology
) -> None:
    merged = merge_ontologies(branch1_ontology, branch2_ontology)

    assert [base.version_iri for base in merged.bases] == [
        branch1_ontology.version_iri,
        branch2_ontology.version_iri,
    ]
    assert merged.version_iri is None, "a merge artifact is unstamped"
    assert merged.iri == branch1_ontology.iri
    assert merged.hash is not None

    for branch in (branch1_ontology, branch2_ontology):
        content = set(without_version_header(branch.graph))
        missing = content - set(merged.graph)
        assert not missing, f"missing triples from {branch.version_iri}: {missing}"


def test_stamped_merge_names_both_priors_and_bumps_the_higher_label(
    base_ontology: Ontology, branch1_ontology: Ontology, branch2_ontology: Ontology
) -> None:
    merged = stamp_version(merge_ontologies(branch1_ontology, branch2_ontology), now=T3)

    assert merged is not None
    assert merged.prior_version_iris == [
        branch1_ontology.version_iri,
        branch2_ontology.version_iri,
    ]
    assert branch1_ontology.version == branch2_ontology.version == "1.0.1"
    assert merged.version == "1.0.2"
    assert merged.created_at == base_ontology.created_at, "the lineage start"
    assert merged.modified_at == T3
    assert [base.version_iri for base in merged.bases] == [merged.version_iri]


def test_merge_ontologies_keeps_both_sides_of_a_contradiction() -> None:
    class1 = URIRef(f"{IRI}#Class1")

    graph1 = RDFGraph()
    graph1.add((URIRef(IRI), RDF.type, OWL.Ontology))
    graph1.add((class1, RDF.type, OWL.Class))
    graph1.add((class1, RDFS.label, Literal("Class One")))
    graph2 = RDFGraph()
    graph2.add((URIRef(IRI), RDF.type, OWL.Ontology))
    graph2.add((class1, RDF.type, OWL.Class))
    graph2.add((class1, RDFS.label, Literal("Class 1")))
    onto1 = stamped_ontology(graph1, ontology_id="test", now=T0)
    onto2 = stamped_ontology(graph2, ontology_id="test", now=T1)

    merged = merge_ontologies(onto1, onto2)

    labels = {str(o) for _, _, o in merged.graph.triples((class1, RDFS.label, None))}
    assert labels == {"Class One", "Class 1"}


def test_merging_the_heads_leaves_one_head(
    ontology_manager: OntologyManager,
    base_ontology: Ontology,
    branch1_ontology: Ontology,
    branch2_ontology: Ontology,
) -> None:
    ontology_manager.add_ontology(base_ontology)
    ontology_manager.add_ontology(branch1_ontology)
    ontology_manager.add_ontology(branch2_ontology)

    heads = ontology_manager.get_heads_by_iri(base_ontology.iri)
    assert {head.version_iri for head in heads} == {
        branch1_ontology.version_iri,
        branch2_ontology.version_iri,
    }
    with pytest.raises(DivergedOntologyError):
        ontology_manager.get_head_by_iri(base_ontology.iri)

    merged = stamp_version(merge_ontologies(heads[0], heads[1]), now=T3)
    assert merged is not None
    ontology_manager.add_ontology(merged)

    new_heads = ontology_manager.get_heads_by_iri(base_ontology.iri)
    assert [head.version_iri for head in new_heads] == [merged.version_iri]
    assert ontology_manager.get_head_by_iri(base_ontology.iri) is new_heads[0]


def test_merge_ontologies_preserves_namespaces() -> None:
    graph1 = RDFGraph()
    graph1.bind("exa", "http://example.org/a/")
    graph1.add((URIRef(IRI), RDF.type, OWL.Ontology))
    _class(graph1, "Class1", "Class 1")

    graph2 = RDFGraph()
    graph2.bind("exb", "http://example.org/b/")
    graph2.add((URIRef(IRI), RDF.type, OWL.Ontology))
    _class(graph2, "Class2", "Class 2")

    onto1 = stamped_ontology(graph1, ontology_id="test", now=T0)
    onto2 = stamped_ontology(graph2, ontology_id="test", now=T1)

    merged = merge_ontologies(onto1, onto2)

    namespaces = dict(merged.graph.namespaces())
    assert "exa" in namespaces
    assert "exb" in namespaces


def test_merge_ontologies_refuses_different_iris(base_ontology: Ontology) -> None:
    other = stamped_ontology(
        RDFGraph._from_turtle_str(
            """
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            <http://example.org/other> a owl:Ontology .
            <http://example.org/other#A> a owl:Class .
            """
        ),
        now=T0,
    )
    with pytest.raises(ValueError, match="different IRIs"):
        merge_ontologies(base_ontology, other)


def test_merge_ontologies_refuses_an_unstamped_input(
    base_ontology: Ontology, branch1_ontology: Ontology
) -> None:
    graph = base_ontology.graph.copy()
    _class(graph, "Loose", "Loose")
    unstamped = base_ontology.derive_updated_version(graph)

    with pytest.raises(ValueError, match="not stamped"):
        merge_ontologies(branch1_ontology, unstamped)
