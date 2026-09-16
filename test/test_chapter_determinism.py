"""The ontology chapter must be byte-identical for the same catalog, always.

The chapter text is a cache key twice over: the LLM disk cache is keyed on the
prompt string, and a provider's prefix cache on its prefix. A chapter that
differs run to run for the same input is one neither cache can ever serve, and
the difference is invisible -- same triples, same length, different bytes.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
from rdflib import OWL, RDF, RDFS, BNode, Literal, Namespace

from ontocast.onto.enum import LLMGraphFormat, OntologyChapterFormat
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.prompt.graph_format import get_graph_format_profile

EX = Namespace("https://example.org/onto#")


def _graph_with_restrictions() -> RDFGraph:
    """Blank nodes in structurally distinct positions, as an ontology has."""
    graph = RDFGraph()
    graph.bind("ex", EX)
    for index in range(6):
        cls = EX[f"C{index}"]
        graph.add((cls, RDF.type, OWL.Class))
        graph.add((cls, RDFS.label, Literal(f"class {index}")))
        restriction = BNode()
        graph.add((cls, RDFS.subClassOf, restriction))
        graph.add((restriction, RDF.type, OWL.Restriction))
        graph.add((restriction, OWL.onProperty, EX[f"p{index}"]))
        graph.add((restriction, OWL.someValuesFrom, EX[f"D{index}"]))
    return graph


@pytest.mark.parametrize(
    "chapter_format",
    [OntologyChapterFormat.INHERIT, OntologyChapterFormat.TURTLE],
)
def test_relabelling_bnodes_does_not_change_the_chapter(
    chapter_format: OntologyChapterFormat,
) -> None:
    """Two parses of the same source differ only in random blank-node labels."""
    profile = get_graph_format_profile(
        LLMGraphFormat.JSONLD, ontology_chapter_format=chapter_format
    )
    first = _graph_with_restrictions()
    turtle = first.serialize_canonical_turtle()
    second = RDFGraph()
    second.bind("ex", EX)
    second.parse(data=turtle, format="turtle")

    assert profile.format_ontology_chapter(
        first, max_triples=None
    ) == profile.format_ontology_chapter(second, max_triples=None)


def test_literals_differing_only_in_language_tag_order_stably() -> None:
    """The lexical form alone is not a sort key.

    Two literals can share it and differ only in language tag; sorting on the
    shared value leaves their order to the iteration, which is the exact
    nondeterminism this guards -- surviving where it is hardest to see.
    """
    graph = RDFGraph()
    graph.bind("ex", EX)
    graph.add((EX.T, RDF.type, OWL.Class))
    graph.add((EX.T, RDFS.comment, Literal("same text", lang="en")))
    graph.add((EX.T, RDFS.comment, Literal("same text", lang="en-US")))
    graph.add((EX.T, RDFS.label, Literal("same text")))
    profile = get_graph_format_profile(LLMGraphFormat.JSONLD)

    rendered = {
        profile.format_ontology_chapter(graph, max_triples=None) for _ in range(5)
    }

    assert len(rendered) == 1


def test_chapter_is_identical_across_processes() -> None:
    """The property that matters: two runs, two interpreters, same bytes.

    Python randomises string hashing per process, so rdflib's iteration order --
    and every serialization that follows it -- differs between runs unless it is
    made explicit. Only a subprocess can show that; within one process the seed
    is fixed and the bug is invisible.
    """
    script = textwrap.dedent(
        """
        import hashlib
        from rdflib import OWL, RDF, RDFS, BNode, Literal, Namespace
        from ontocast.onto.enum import LLMGraphFormat
        from ontocast.onto.rdfgraph import RDFGraph
        from ontocast.prompt.graph_format import get_graph_format_profile

        EX = Namespace("https://example.org/onto#")
        g = RDFGraph()
        g.bind("ex", EX)
        for i in range(6):
            c = EX[f"C{i}"]
            g.add((c, RDF.type, OWL.Class))
            g.add((c, RDFS.label, Literal(f"class {i}")))
            g.add((c, RDFS.comment, Literal("shared", lang="en")))
            g.add((c, RDFS.comment, Literal("shared", lang="en-US")))
            b = BNode()
            g.add((c, RDFS.subClassOf, b))
            g.add((b, RDF.type, OWL.Restriction))
            g.add((b, OWL.onProperty, EX[f"p{i}"]))
            g.add((b, OWL.someValuesFrom, EX[f"D{i}"]))
        p = get_graph_format_profile(LLMGraphFormat.JSONLD)
        print(hashlib.sha1(p.format_ontology_chapter(g, max_triples=None).encode()).hexdigest())
        """
    )
    digests = {
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            # A different hash seed per run is what makes this a real test.
            env={"PYTHONHASHSEED": str(seed), "PATH": "/usr/bin:/bin"},
        ).stdout.strip()
        for seed in (1, 2, 3)
    }

    assert len(digests) == 1, f"chapter differs between processes: {digests}"
