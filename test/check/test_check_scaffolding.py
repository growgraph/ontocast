"""``check_facts`` judges the facts, not OntoCast's provenance around them."""

import pytest
from rdflib import FOAF, RDF, RDFS, BNode, Literal, URIRef
from rdflib.namespace import DCTERMS

from ontocast.check import FactsCheckKind, check_facts
from ontocast.onto.constants import ONTOCAST, PROV
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.scaffolding import FactScope
from ontocast.tool.agg.rewriter import add_reified_provenance

pytestmark = pytest.mark.unit

ONTO = "http://example.com/onto#"
DATA = "http://example.com/data/"
SCHEMA = "https://schema.org/"

CATALOG = f"""
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex:   <{ONTO}> .

ex:Item a owl:Class .
ex:Article a owl:Class .
ex:partOf a owl:ObjectProperty ; rdfs:domain ex:Item .
<{SCHEMA}articleSection> rdfs:domain ex:Article .
"""

ITEM = URIRef(DATA + "item_a")
OTHER = URIRef(DATA + "item_b")
CHUNK = URIRef(DATA + "chunk/0")
DOCUMENT = URIRef(DATA + "document")


def _reported(findings) -> str:
    return " ".join(
        f"{f.subject} {f.predicate} {f.value} {f.message}" for f in findings
    )


def _catalog() -> RDFGraph:
    graph = RDFGraph()
    graph.parse(data=CATALOG, format="turtle")
    return graph


def _with_provenance(profile: str) -> RDFGraph:
    """Two facts, one chunk node carrying OntoCast's own metadata, and a document node."""
    graph = RDFGraph()
    facts = [
        (ITEM, RDF.type, URIRef(ONTO + "Item")),
        (OTHER, RDF.type, URIRef(ONTO + "Item")),
        (ITEM, URIRef(ONTO + "partOf"), OTHER),
        (ITEM, RDFS.label, Literal("item a")),
        (OTHER, RDFS.label, Literal("item b")),
    ]
    for triple in facts:
        graph.add(triple)

    graph.add((CHUNK, RDF.type, PROV.Entity))
    graph.add((CHUNK, RDF.type, URIRef(SCHEMA + "Text")))
    graph.add((CHUNK, URIRef(SCHEMA + "articleSection"), Literal("Results")))
    graph.add((CHUNK, ONTOCAST.sectionLabelSource, Literal("heading")))
    graph.add((CHUNK, ONTOCAST.sectionLabelStrength, Literal("strong")))

    graph.add((DOCUMENT, RDF.type, PROV.Entity))
    graph.add((DOCUMENT, RDF.type, FOAF.Document))
    graph.add((DOCUMENT, DCTERMS.title, Literal("A document")))

    for s, p, o in facts:
        if profile == "rdf12":
            add_reified_provenance(graph, s, p, o, CHUNK)
        else:
            statement = BNode()
            graph.add((statement, RDF.type, RDF.Statement))
            graph.add((statement, RDF.subject, s))
            graph.add((statement, RDF.predicate, p))
            graph.add((statement, RDF.object, o))
            graph.add((statement, PROV.wasDerivedFrom, CHUNK))
    return graph


@pytest.mark.parametrize("profile", ["rdf12", "classic"])
def test_provenance_terms_are_not_reported(profile):
    report = check_facts(_with_provenance(profile), _catalog(), vocabulary="strict")

    assert "sectionLabel" not in _reported(report.findings)
    assert not [
        finding
        for finding in report.findings
        if finding.kind == FactsCheckKind.DOMAIN_VIOLATION
    ]


@pytest.mark.parametrize("profile", ["rdf12", "classic"])
def test_only_fact_instances_are_counted(profile):
    report = check_facts(_with_provenance(profile), _catalog())

    assert report.metrics.instances == 2


def test_the_same_term_on_a_fact_is_still_reported():
    graph = _with_provenance("rdf12")
    graph.add((ITEM, ONTOCAST.sectionLabelStrength, Literal("strong")))

    report = check_facts(graph, _catalog(), vocabulary="strict")

    vocabulary = [
        finding
        for finding in report.findings
        if finding.kind == FactsCheckKind.NON_CATALOG_VOCABULARY
    ]
    assert "sectionLabelStrength" in _reported(vocabulary)


def test_the_facts_view_keeps_facts_and_the_document_node():
    graph = _with_provenance("rdf12")

    facts = FactScope.of(graph, None).facts()

    assert (ITEM, URIRef(ONTO + "partOf"), OTHER) in facts
    assert (DOCUMENT, DCTERMS.title, Literal("A document")) in facts
    assert not list(facts.triples((CHUNK, None, None)))
    assert not facts.has_reifications
    assert len(graph.reifications) > 0
