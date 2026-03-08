from rdflib import RDF, Literal, URIRef
from rdflib.namespace import XSD

from ontocast.onto.constants import DEFAULT_IRI, PROV, RDF_REIFIES, SCHEMA
from ontocast.onto.content_unit import ContentUnit, OutputType
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.agg.rewriter import GraphRewriter


def test_merge_graphs_with_provenance_adds_chunk_metadata(
    graph_rewriter: GraphRewriter,
) -> None:
    graph = RDFGraph()
    entity = URIRef(f"{DEFAULT_IRI}/Entity1")
    graph.add((entity, RDF.type, URIRef(f"{DEFAULT_IRI}/Thing")))

    unit = ContentUnit(
        text="test",
        index=5,
        doc_iri=URIRef("https://example.org/doc/abc123"),
        graph=graph,
        type=OutputType.FACTS,
    )
    merged = graph_rewriter.merge_graphs_with_provenance([unit], mapping={})
    unit_uri = URIRef(unit.iri_absolute)

    assert (unit_uri, RDF.type, PROV.Entity) in merged
    assert (unit_uri, SCHEMA.position, Literal(5, datatype=XSD.integer)) in merged
    assert (unit_uri, SCHEMA.identifier, Literal(unit.hid)) in merged

    namespaces = {prefix: str(namespace) for prefix, namespace in merged.namespaces()}

    assert namespaces["prov"] == str(PROV)
    assert namespaces["schema"] == str(SCHEMA)
    assert namespaces["doc"] == "https://example.org/doc/abc123/"


def test_merge_graphs_with_provenance_reifies_mapped_triple(
    graph_rewriter: GraphRewriter,
) -> None:
    graph = RDFGraph()
    old_subject = URIRef("http://chunk.org/OldEntity")
    old_predicate = URIRef("http://chunk.org/prop")
    value = Literal("value")
    graph.add((old_subject, old_predicate, value))

    new_subject = URIRef(f"{DEFAULT_IRI}/NewEntity")
    new_predicate = URIRef(f"{DEFAULT_IRI}/prop")
    unit = ContentUnit(
        text="test",
        index=0,
        doc_iri=URIRef("https://example.org/doc"),
        graph=graph,
        type=OutputType.FACTS,
    )

    merged = graph_rewriter.merge_graphs_with_provenance(
        [unit],
        {old_subject: new_subject, old_predicate: new_predicate},
    )
    stmt_nodes = list(merged.subjects(RDF_REIFIES, None))
    assert len(stmt_nodes) == 1

    reified = list(merged.objects(stmt_nodes[0], RDF_REIFIES))
    assert len(reified) == 1
    quoted = reified[0]
    assert isinstance(quoted, tuple)
    assert quoted[0] == new_subject
    assert quoted[1] == new_predicate
    assert str(quoted[2]) == str(value)


def test_shared_triple_accumulates_multiple_provenance_sources(
    graph_rewriter: GraphRewriter,
) -> None:
    triple = (
        URIRef(f"{DEFAULT_IRI}/Alice"),
        URIRef(f"{DEFAULT_IRI}/knows"),
        URIRef(f"{DEFAULT_IRI}/Bob"),
    )
    graph_a = RDFGraph()
    graph_b = RDFGraph()
    graph_a.add(triple)
    graph_b.add(triple)

    unit_a = ContentUnit(
        text="chunk 0",
        index=0,
        doc_iri=URIRef("https://example.org/doc"),
        graph=graph_a,
        type=OutputType.FACTS,
    )
    unit_b = ContentUnit(
        text="chunk 1",
        index=1,
        doc_iri=URIRef("https://example.org/doc"),
        graph=graph_b,
        type=OutputType.FACTS,
    )

    merged = graph_rewriter.merge_graphs_with_provenance([unit_a, unit_b], mapping={})
    statements = list(merged.subjects(RDF_REIFIES, None))
    assert len(statements) == 1

    sources = {str(src) for src in merged.objects(statements[0], PROV.wasDerivedFrom)}
    assert str(URIRef(unit_a.iri_absolute)) in sources
    assert str(URIRef(unit_b.iri_absolute)) in sources
