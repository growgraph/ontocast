"""``RDFGraph.from_file`` and the public ``check_facts`` export."""

from __future__ import annotations

from pathlib import Path

import pytest
from rdflib import Graph, Literal, URIRef
from rdflib.compare import isomorphic

from ontocast.onto.rdfgraph import RDFGraph

pytestmark = pytest.mark.unit

_S = URIRef("http://example.org/s")
_P = URIRef("http://example.org/p")


def _reference() -> Graph:
    g = Graph()
    g.add((_S, _P, Literal("v")))
    return g


@pytest.mark.parametrize(
    ("suffix", "fmt"),
    [
        (".ttl", "turtle"),
        (".turtle", "turtle"),
        (".nt", "nt"),
        (".jsonld", "json-ld"),
        (".json", "json-ld"),
        (".rdf", "xml"),
        (".owl", "xml"),
        (".xml", "xml"),
    ],
)
def test_from_file_formats(tmp_path: Path, suffix: str, fmt: str) -> None:
    path = tmp_path / f"g{suffix}"
    path.write_text(_reference().serialize(format=fmt), encoding="utf-8")
    graph = RDFGraph.from_file(path)
    assert isinstance(graph, RDFGraph)
    assert isomorphic(graph, _reference())


def test_from_file_explicit_format_overrides_suffix(tmp_path: Path) -> None:
    path = tmp_path / "g.dat"
    path.write_text(_reference().serialize(format="nt"), encoding="utf-8")
    assert isomorphic(RDFGraph.from_file(str(path), format="nt"), _reference())


def test_from_file_unknown_suffix(tmp_path: Path) -> None:
    path = tmp_path / "g.dat"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        RDFGraph.from_file(path)


def test_public_exports() -> None:
    from ontocast import FactsCheckReport, check_facts

    assert callable(check_facts)
    assert FactsCheckReport.__name__ == "FactsCheckReport"


def _reified() -> RDFGraph:
    from rdflib import BNode, Namespace

    prov = Namespace("http://www.w3.org/ns/prov#")
    graph = RDFGraph()
    graph.bind("ex", "http://example.org/")
    graph.add((_S, _P, Literal("v")))
    reifier = BNode()
    graph.reifications.add(reifier, (_S, _P, Literal("v")))
    graph.add((reifier, prov.wasDerivedFrom, URIRef("http://example.org/chunk/1")))
    return graph


def test_from_file_reads_the_library_turtle_with_reifications(tmp_path: Path) -> None:
    from ontocast.check import check_facts

    original = _reified()
    path = tmp_path / "facts.ttl"
    path.write_text(RDFGraph._to_turtle_str(original), encoding="utf-8")

    loaded = RDFGraph.from_file(path)

    assert isomorphic(loaded, original)
    assert len(loaded.reifications) == 1
    ((_, triple),) = list(loaded.reifications)
    assert triple == (_S, _P, Literal("v"))
    report = check_facts(loaded, None)
    assert report.metrics.triples == len(loaded)


def test_from_file_reads_ntriples_with_triple_terms(tmp_path: Path) -> None:
    path = tmp_path / "facts.nt"
    path.write_text(
        '<http://example.org/s> <http://example.org/p> "v" .\n'
        "_:r <http://www.w3.org/1999/02/22-rdf-syntax-ns#reifies> "
        '<<( <http://example.org/s> <http://example.org/p> "v" )>> .\n',
        encoding="utf-8",
    )
    loaded = RDFGraph.from_file(path)
    assert (_S, _P, Literal("v")) in loaded
    assert len(loaded.reifications) == 1


def test_from_file_turtle_with_shorthand_inside_a_literal(tmp_path: Path) -> None:
    path = tmp_path / "g.ttl"
    path.write_text(
        '<http://example.org/s> <http://example.org/p> "a <<( b {| c" .\n',
        encoding="utf-8",
    )
    loaded = RDFGraph.from_file(path)
    assert (_S, _P, Literal("a <<( b {| c")) in loaded
