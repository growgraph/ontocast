"""Test for RDFGraph __iadd__ method.

This test verifies that the __iadd__ method properly reuses __add__ and binds prefixes.
"""

from rdflib import Graph, Literal, Namespace, URIRef

from ontocast.onto.rdfgraph import RDFGraph


def test_rdfgraph_iadd_reuses_add_and_binds_prefixes():
    """Test that __iadd__ reuses __add__ and properly binds prefixes."""

    # Create two RDFGraph instances with different namespaces
    graph1 = RDFGraph()
    graph2 = RDFGraph()

    # Define namespacesЙ
    ns1 = Namespace("http://example.org/ns1/")
    ns2 = Namespace("http://example.org/ns2/")

    # Add some triples to graph1 with ns1 namespace
    graph1.add((ns1.subject1, ns1.predicate1, Literal("value1")))
    graph1.add((ns1.subject2, ns1.predicate2, Literal("value2")))
    graph1.bind("ns1", ns1)

    # Add some triples to graph2 with ns2 namespace
    graph2.add((ns2.subject1, ns2.predicate1, Literal("value3")))
    graph2.add((ns2.subject2, ns2.predicate2, Literal("value4")))
    graph2.bind("ns2", ns2)

    # Test __iadd__ method
    graph1 += graph2

    # Verify that all triples are present
    assert len(graph1) == 4
    assert (ns1.subject1, ns1.predicate1, Literal("value1")) in graph1
    assert (ns1.subject2, ns1.predicate2, Literal("value2")) in graph1
    assert (ns2.subject1, ns2.predicate1, Literal("value3")) in graph1
    assert (ns2.subject2, ns2.predicate2, Literal("value4")) in graph1

    # Verify that namespace bindings are preserved
    namespaces = dict(graph1.namespaces())
    assert "ns1" in namespaces
    assert "ns2" in namespaces
    assert str(namespaces["ns1"]) == "http://example.org/ns1/"
    assert str(namespaces["ns2"]) == "http://example.org/ns2/"


def test_rdfgraph_iadd_with_regular_graph():
    """Test that __iadd__ works with regular rdflib.Graph objects."""

    # Create RDFGraph and regular Graph
    rdf_graph = RDFGraph()
    regular_graph = Graph()

    # Define namespace
    ns = Namespace("http://example.org/test/")

    # Add triples to both graphs
    rdf_graph.add((ns.subject1, ns.predicate1, Literal("value1")))
    rdf_graph.bind("test", ns)

    regular_graph.add((ns.subject2, ns.predicate2, Literal("value2")))

    # Test __iadd__ method
    rdf_graph += regular_graph

    # Verify that all triples are present
    assert len(rdf_graph) == 2
    assert (ns.subject1, ns.predicate1, Literal("value1")) in rdf_graph
    assert (ns.subject2, ns.predicate2, Literal("value2")) in rdf_graph

    # Verify that namespace binding is preserved
    namespaces = dict(rdf_graph.namespaces())
    assert "test" in namespaces
    assert str(namespaces["test"]) == "http://example.org/test/"


def test_rdfgraph_iadd_returns_self():
    """Test that __iadd__ returns self for chaining."""

    graph1 = RDFGraph()
    graph2 = RDFGraph()

    # Add some triples
    graph1.add(
        (
            URIRef("http://example.org/subject1"),
            URIRef("http://example.org/predicate1"),
            Literal("value1"),
        )
    )
    graph2.add(
        (
            URIRef("http://example.org/subject2"),
            URIRef("http://example.org/predicate2"),
            Literal("value2"),
        )
    )

    # Test that __iadd__ returns self
    result = graph1.__iadd__(graph2)

    # Verify that result is the same object as graph1
    assert result is graph1
    assert len(graph1) == 2


def test_rdfgraph_iadd_equivalent_to_add():
    """Test that __iadd__ produces the same result as __add__."""

    # Create two graphs
    graph1 = RDFGraph()
    graph2 = RDFGraph()

    # Define namespaces
    ns1 = Namespace("http://example.org/ns1/")
    ns2 = Namespace("http://example.org/ns2/")

    # Add triples and bind namespaces
    graph1.add((ns1.subject1, ns1.predicate1, Literal("value1")))
    graph1.bind("ns1", ns1)

    graph2.add((ns2.subject1, ns2.predicate1, Literal("value2")))
    graph2.bind("ns2", ns2)

    # Create copies for comparison
    graph1_copy = RDFGraph()
    for triple in graph1:
        graph1_copy.add(triple)
    for prefix, uri in graph1.namespaces():
        graph1_copy.bind(prefix, uri)

    # Test __add__ method
    result_add = graph1_copy + graph2

    # Test __iadd__ method
    graph1 += graph2

    # Verify that both methods produce the same result
    assert len(graph1) == len(result_add)
    assert set(graph1) == set(result_add)

    # Verify namespace bindings are the same
    namespaces1 = dict(graph1.namespaces())
    namespaces_add = dict(result_add.namespaces())
    assert namespaces1 == namespaces_add
