"""Instance rule of the facts check."""

from rdflib import OWL, RDF, RDFS, BNode, Graph, Namespace, URIRef

from ontocast.check import FactsCheckKind, instances
from ontocast.tool.facts_validation.terms import ValidationPolicy

EX = Namespace("http://example.org/onto#")
STD = ValidationPolicy().standard_namespaces()
CATALOG = {str(EX.Person)}


def _run(graph: Graph, **kw) -> set[URIRef]:
    return instances(graph, catalog_terms=CATALOG, standard_namespaces=STD, **kw)


def test_individual_in_ontology_namespace_is_instance():
    g = Graph()
    g.add((EX.alice, RDF.type, EX.Person))
    assert _run(g) == {EX.alice}


def test_catalog_term_is_not_instance():
    g = Graph()
    g.add((EX.Person, RDFS.label, URIRef("http://example.org/x")))
    assert _run(g) == set()


def test_declared_in_upload_is_not_instance():
    g = Graph()
    g.add((EX.Robot, RDF.type, OWL.Class))
    g.add((EX.Robot, RDFS.label, EX.alice))
    assert _run(g) == set()


def test_well_known_subject_is_not_instance():
    g = Graph()
    g.add((RDFS.label, RDFS.comment, EX.alice))
    assert _run(g) == set()


def test_blank_node_subjects_ignored():
    g = Graph()
    g.add((BNode(), EX.p, EX.o))
    assert _run(g) == set()


def test_restrict_to_namespaces():
    g = Graph()
    g.add((EX.alice, EX.p, EX.o))
    g.add((URIRef("http://other.org/bob"), EX.p, EX.o))
    assert _run(g, restrict_to=["http://other.org/"]) == {
        URIRef("http://other.org/bob")
    }


def test_kind_values_are_snake_case():
    assert FactsCheckKind.NON_CATALOG_VOCABULARY.value == "non_catalog_vocabulary"
