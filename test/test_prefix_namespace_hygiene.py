"""Regression tests for domain-prefix / namespace hygiene in prompts and graphs."""

from rdflib import OWL, RDF, URIRef

from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.util import RDFLIB_DEFAULT_NAMESPACE_URIS
from ontocast.prompt.facts_guidelines import format_facts_operational_guidelines
from ontocast.prompt.ontology_context import extract_domain_prefix_pairs


def _matsci_turtle(*, prefix: str = "matsci") -> str:
    return f"""
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix {prefix}: <https://growgraph.dev/ontologies/matsci#> .

<https://growgraph.dev/ontologies/matsci> a owl:Ontology ;
    rdfs:label "Material Science Ontology"@en .

{prefix}:Material a owl:Class ;
    rdfs:label "Material"@en .
"""


def test_extract_domain_prefix_pairs_excludes_rdflib_defaults() -> None:
    graph = RDFGraph()
    graph.parse(data=_matsci_turtle(), format="turtle")
    # Force-bind a few rdflib defaults that previously leaked into the prompt.
    graph.bind("brick", URIRef("https://brickschema.org/schema/Brick#"))
    graph.bind("csvw", URIRef("http://www.w3.org/ns/csvw#"))
    graph.bind("geo", URIRef("http://www.opengis.net/ont/geosparql#"))
    ontology = Ontology(graph=graph)

    pairs = extract_domain_prefix_pairs(ontology)
    prefixes = {p for p, _ in pairs}
    namespaces = {ns for _, ns in pairs}

    assert "matsci" in prefixes
    assert "brick" not in prefixes
    assert "csvw" not in prefixes
    assert "geo" not in prefixes
    assert "xml" not in prefixes
    assert namespaces.isdisjoint(RDFLIB_DEFAULT_NAMESPACE_URIS)


def test_author_short_prefix_kept_not_rebound_to_ontology_id() -> None:
    graph = RDFGraph()
    graph.parse(data=_matsci_turtle(prefix="matsci"), format="turtle")
    ontology = Ontology(graph=graph)

    assert ontology.ontology_id == "matsci"
    domain_bindings = [
        (prefix, str(uri))
        for prefix, uri in ontology.graph.namespaces()
        if str(uri) == "https://growgraph.dev/ontologies/matsci#"
    ]
    assert domain_bindings == [("matsci", "https://growgraph.dev/ontologies/matsci#")]
    assert ontology.prefix == "matsci"


def test_degenerate_ns_prefix_is_rebound_and_old_binding_removed() -> None:
    graph = RDFGraph()
    ns = "https://growgraph.dev/ontologies/matsci#"
    graph.bind("ns11", URIRef(ns))
    graph.add(
        (
            URIRef("https://growgraph.dev/ontologies/matsci"),
            RDF.type,
            OWL.Ontology,
        )
    )
    ontology = Ontology(graph=graph)

    assert ontology.ontology_id == "matsci"
    domain_prefixes = [
        prefix for prefix, uri in ontology.graph.namespaces() if str(uri) == ns
    ]
    assert domain_prefixes == ["matsci"]
    assert "ns11" not in domain_prefixes


def test_sanitize_prefixes_namespaces_preserves_xml_binding() -> None:
    graph = RDFGraph()
    graph.parse(data=_matsci_turtle(), format="turtle")
    xml_uri = "http://www.w3.org/XML/1998/namespace"
    before = {prefix: str(uri) for prefix, uri in graph.namespaces()}
    assert before.get("xml") == xml_uri

    graph.sanitize_prefixes_namespaces()

    after = {prefix: str(uri) for prefix, uri in graph.namespaces()}
    assert after.get("xml") == xml_uri
    assert "xml1" not in after
    assert not any(uri == f"{xml_uri}/" for uri in after.values())


def test_facts_guidelines_domain_clause_appears_once() -> None:
    clause = (
        "domain ontologies `matsci:` (<https://growgraph.dev/ontologies/matsci#>), "
        "`qqval:` (<https://growgraph.dev/ontologies/qqval#>)"
    )
    guidelines = format_facts_operational_guidelines(
        facts_namespace="https://example.com/facts/",
        domain_ontologies_clause=clause,
        jsonld=False,
    )
    assert guidelines.count(clause) == 1
    assert "domain ontology namespace(s) above" in guidelines
