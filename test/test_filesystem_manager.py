from pathlib import Path

from rdflib import RDF, URIRef

from ontocast.onto.constants import DEFAULT_IRI, PROV, RDF_REIFIES, SCHEMA
from ontocast.onto.content_unit import ContentUnit, OutputType
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.agg.rewriter import GraphRewriter
from ontocast.tool.triple_manager.filesystem_manager import FilesystemTripleStoreManager


def test_filesystem_manager_serializes_clean_facts_graph(tmp_path: Path) -> None:
    fact_graph = RDFGraph()
    fact_triple = (
        URIRef(f"{DEFAULT_IRI}/Alice"),
        URIRef(f"{DEFAULT_IRI}/knows"),
        URIRef(f"{DEFAULT_IRI}/Bob"),
    )
    fact_graph.add(fact_triple)
    unit = ContentUnit(
        text="test",
        index=0,
        doc_iri=URIRef("https://example.org/doc"),
        graph=fact_graph,
        type=OutputType.FACTS,
    )

    merged_with_provenance = GraphRewriter().merge_graphs_with_provenance(
        [unit],
        mapping={},
    )

    manager = FilesystemTripleStoreManager(
        working_directory=tmp_path,
        ontology_path=tmp_path,
    )
    manager.serialize(
        merged_with_provenance, graph_uri="https://example.org/facts/main"
    )

    full_path = tmp_path / "facts_facts_main.ttl"
    clean_path = tmp_path / "facts_facts_main_clean.ttl"
    assert full_path.exists()
    assert clean_path.exists()

    clean_graph = RDFGraph()
    clean_graph.parse(clean_path, format="turtle")
    clean_ttl = clean_path.read_text()
    assert fact_triple in clean_graph
    assert not list(clean_graph.triples((None, RDF_REIFIES, None)))
    assert not list(clean_graph.triples((None, PROV.wasDerivedFrom, None)))
    assert "@prefix" in clean_ttl


def test_strip_provenance_removes_source_nodes() -> None:
    manager = FilesystemTripleStoreManager(
        working_directory=Path("/tmp"),
        ontology_path=Path("/tmp"),
    )
    graph = RDFGraph()
    source_node = URIRef(f"{DEFAULT_IRI}/Appeal1")
    domain_class = URIRef(f"{DEFAULT_IRI}/Appeal")
    statement = URIRef(f"{DEFAULT_IRI}/stmt-1")

    graph.add((statement, RDF_REIFIES, URIRef(f"{DEFAULT_IRI}/quoted")))
    graph.add((statement, PROV.wasDerivedFrom, source_node))
    graph.add((source_node, RDF.type, domain_class))
    graph.add((source_node, RDF.type, PROV.Entity))
    graph.add((source_node, SCHEMA.identifier, URIRef(f"{DEFAULT_IRI}/chunk-1")))

    clean_graph = manager.strip_provenance(graph)

    assert (source_node, RDF.type, domain_class) not in clean_graph
    assert (source_node, RDF.type, PROV.Entity) not in clean_graph
    assert (
        source_node,
        SCHEMA.identifier,
        URIRef(f"{DEFAULT_IRI}/chunk-1"),
    ) not in clean_graph
