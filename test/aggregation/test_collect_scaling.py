"""Entity collection is linear in the units, and an entity's context is its triples."""

import pytest
from rdflib import URIRef

from ontocast.onto.content_unit import ContentUnit, OutputType
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool import EmbeddingBasedAggregator

pytestmark = pytest.mark.unit

DOC = "https://ex.org/doc/1"
ONTO = "https://ex.org/onto#"


def _unit(i: int) -> ContentUnit:
    # Shared predicates and classes occur in every unit, as in a real document.
    lines = [
        f"@prefix f: <{DOC}/> .",
        f"@prefix o: <{ONTO}> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
    ]
    for j in range(20):
        e = f"f:e{(i * 7 + j) % 300}"
        lines.append(
            f'{e} a o:C{j % 10} ; rdfs:label "e{j}" ; o:value "{i}.{j}" ; '
            f"o:next f:e{(i * 7 + j + 1) % 300} ."
        )
    graph = RDFGraph()
    graph.parse(data="\n".join(lines), format="turtle")
    return ContentUnit(
        text=f"unit {i}", index=i, doc_iri=DOC, graph=graph, type=OutputType.FACTS
    )


def _collection_adds(n_units: int, monkeypatch: pytest.MonkeyPatch) -> int:
    calls = [0]
    add = RDFGraph.add

    def counted(self, triple):
        calls[0] += 1
        return add(self, triple)

    units = [_unit(i) for i in range(n_units)]
    monkeypatch.setattr(RDFGraph, "add", counted)
    EmbeddingBasedAggregator()._collect_all_entities(units, set())
    monkeypatch.undo()
    return calls[0]


def test_collection_work_grows_linearly_with_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    small = _collection_adds(10, monkeypatch)
    large = _collection_adds(40, monkeypatch)
    # Four times the units: about four times the work, never the square.
    assert large <= 5 * small


def test_entity_context_is_the_triples_that_mention_it() -> None:
    units = [_unit(0), _unit(1)]
    aggregator = EmbeddingBasedAggregator()
    _, _, graphs, *_ = aggregator._collect_all_entities(units, set())
    for entity in (URIRef(f"{ONTO}next"), URIRef(f"{DOC}/e7")):
        mentions = RDFGraph()
        for unit in units:
            for triple in unit.graph:
                if entity in triple:
                    mentions.add(triple)
        assert len(mentions)
        create = aggregator.normalizer.create_representation
        assert create(entity, graphs[entity]) == create(entity, mentions)
