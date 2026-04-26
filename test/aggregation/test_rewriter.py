from rdflib import OWL, RDF, Literal, URIRef

from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.agg.rewriter import GraphRewriter


def test_apply_mapping_to_triple(graph_rewriter: GraphRewriter) -> None:
    e1 = URIRef("http://chunk1.org/e1")
    p1 = URIRef("http://chunk1.org/p1")
    e2 = URIRef("http://chunk1.org/e2")

    e1_new = URIRef(f"{DEFAULT_IRI}/Entity1")
    p1_new = URIRef(f"{DEFAULT_IRI}/property1")
    e2_new = URIRef(f"{DEFAULT_IRI}/Entity2")

    mapped = graph_rewriter.apply_mapping_to_triple(
        e1,
        p1,
        e2,
        {e1: e1_new, p1: p1_new, e2: e2_new},
    )
    assert mapped == (e1_new, p1_new, e2_new)


def test_apply_mapping_preserves_ontology_type_object(
    graph_rewriter: GraphRewriter,
) -> None:
    entity = URIRef("http://chunk1.org/entity")
    ontology_type = URIRef("http://ontology.org/Thing")
    mapped_entity = URIRef(f"{DEFAULT_IRI}/Entity")

    new_s, new_p, new_o = graph_rewriter.apply_mapping_to_triple(
        entity,
        RDF.type,
        ontology_type,
        {entity: mapped_entity},
    )
    assert (new_s, new_p, new_o) == (mapped_entity, RDF.type, ontology_type)


def test_rewrite_graph_applies_mapping(graph_rewriter: GraphRewriter) -> None:
    graph = RDFGraph()
    e1 = URIRef("http://chunk1.org/e1")
    e2 = URIRef("http://chunk1.org/e2")
    p = URIRef("http://chunk1.org/p")
    ont_type = URIRef("http://ontology.org/Thing")

    graph.add((e1, p, e2))
    graph.add((e1, RDF.type, ont_type))

    e1_new = URIRef(f"{DEFAULT_IRI}/Entity1")
    e2_new = URIRef(f"{DEFAULT_IRI}/Entity2")
    p_new = URIRef(f"{DEFAULT_IRI}/relatesTo")

    rewritten = graph_rewriter.rewrite_graph(graph, {e1: e1_new, e2: e2_new, p: p_new})
    assert (e1_new, p_new, e2_new) in rewritten
    assert (e1_new, RDF.type, ont_type) in rewritten


def test_merge_graphs_deduplicates_triples(graph_rewriter: GraphRewriter) -> None:
    graph1 = RDFGraph()
    graph2 = RDFGraph()
    e = URIRef("http://chunk1.org/e")
    p = URIRef("http://chunk1.org/p")
    value = Literal("value")

    graph1.add((e, p, value))
    graph2.add((e, p, value))

    merged = graph_rewriter.merge_graphs(
        [graph1, graph2],
        mapping={
            e: URIRef(f"{DEFAULT_IRI}/Entity"),
            p: URIRef(f"{DEFAULT_IRI}/hasValue"),
        },
        base_namespace=DEFAULT_IRI,
    )
    assert (
        len(list(merged.triples((URIRef(f"{DEFAULT_IRI}/Entity"), None, value)))) == 1
    )


def test_rewrite_graph_adds_sameas_for_merged_entities(
    graph_rewriter: GraphRewriter,
) -> None:
    graph_rewriter = GraphRewriter(add_sameas_links=True)
    graph = RDFGraph()
    e1 = URIRef("http://chunk1.org/e1")
    e2 = URIRef("http://chunk2.org/e2")
    p = URIRef("http://chunk1.org/p")
    canonical = URIRef(f"{DEFAULT_IRI}/Entity")

    graph.add((e1, p, Literal("a")))
    graph.add((e2, p, Literal("b")))

    rewritten = graph_rewriter.rewrite_graph(graph, {e1: canonical, e2: canonical})
    assert len(list(rewritten.triples((canonical, OWL.sameAs, None)))) >= 1


def test_rewriter_blocks_sameas_for_forbidden_namespace() -> None:
    base = "https://growgraph.dev/facts"
    graph_rewriter = GraphRewriter(
        add_sameas_links=True,
        blocked_sameas_namespaces=(base,),
    )
    graph = RDFGraph()
    original_fact = URIRef("https://growgraph.dev/facts/PersonA")
    original_doc = URIRef("https://example.org/docs/case-1/PersonA")
    canonical_doc = URIRef("https://example.org/docs/case-1/PersonCanonical")
    relation = URIRef("https://example.org/relation")
    graph.add((original_doc, relation, Literal("value")))

    rewritten = graph_rewriter.rewrite_graph(
        graph,
        {
            original_doc: canonical_doc,
            original_fact: canonical_doc,
        },
    )

    assert (canonical_doc, OWL.sameAs, original_doc) in rewritten
    assert (canonical_doc, OWL.sameAs, original_fact) not in rewritten
