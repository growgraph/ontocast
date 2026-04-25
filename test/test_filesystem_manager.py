from pathlib import Path

from rdflib import URIRef

from ontocast.onto.constants import DEFAULT_IRI, PROV, RDF_REIFIES
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
    assert fact_triple in clean_graph
    assert not list(clean_graph.triples((None, RDF_REIFIES, None)))
    assert not list(clean_graph.triples((None, PROV.wasDerivedFrom, None)))
