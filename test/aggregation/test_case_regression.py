"""Regression suite for the measured case4/case5 aggregation damage.

``test/data/case4/pre_merge_unit.jsonld`` is the verbatim pre-merge LLM
output of the case4 run whose aggregation collapsed distinct quantity
values (12.5 vs 96 meV merged into one node; the photon-propagation
contribution absorbed into the impurity contribution). The tests assert
the merge guards keep every quantity distinct.

``test/data/case5/paper*.facts.ttl`` are damaged post-merge outputs of a
full-paper run; re-aggregating them must not add *new* damage.

Real sentence-transformer embeddings are exercised (marker: slow).
"""

from pathlib import Path

import pytest
from rdflib import Literal, URIRef

from ontocast.onto.content_unit import ContentUnit, OutputType
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool import EmbeddingBasedAggregator

pytestmark = pytest.mark.slow

DATA = Path(__file__).parent.parent / "data"
NUMERIC_VALUE = URIRef("http://qudt.org/schema/qudt/numericValue")
QUDT_UNIT = URIRef("http://qudt.org/schema/qudt/unit")


def _distinct_value_counts(graph: RDFGraph) -> dict[URIRef, int]:
    values: dict[URIRef, set[str]] = {}
    for subject, _, obj in graph.triples((None, NUMERIC_VALUE, None)):
        if isinstance(subject, URIRef) and isinstance(obj, Literal):
            values.setdefault(subject, set()).add(str(float(obj)))
    return {subject: len(vals) for subject, vals in values.items()}


def _distinct_unit_counts(graph: RDFGraph) -> dict[URIRef, int]:
    units: dict[URIRef, set[URIRef]] = {}
    for subject, _, obj in graph.triples((None, QUDT_UNIT, None)):
        if isinstance(subject, URIRef) and isinstance(obj, URIRef):
            units.setdefault(subject, set()).add(obj)
    return {subject: len(vals) for subject, vals in units.items()}


@pytest.fixture(scope="module")
def aggregator() -> EmbeddingBasedAggregator:
    return EmbeddingBasedAggregator()


def test_case4_pre_merge_quantities_stay_distinct(
    aggregator: EmbeddingBasedAggregator,
) -> None:
    graph = RDFGraph()
    graph.parse(DATA / "case4" / "pre_merge_unit.jsonld", format="json-ld")
    unit = ContentUnit(
        text="case4 excerpt",
        index=0,
        doc_iri=URIRef("https://growgraph.dev/doc/376c5e808804"),
        graph=graph,
        type=OutputType.FACTS,
    )

    merged = aggregator.aggregate_graphs([unit], ontology_graph=RDFGraph()).graph

    # No node may hold two distinct numeric values or two units.
    assert all(count == 1 for count in _distinct_value_counts(merged).values())
    assert all(count == 1 for count in _distinct_unit_counts(merged).values())

    # The measured collapse pairs stay distinct: 12.5 vs 96 meV red shifts,
    # and the two 30 meV contributions (impurity vs photon propagation).
    subjects_by_value: dict[str, set[URIRef]] = {}
    for subject, _, obj in merged.triples((None, NUMERIC_VALUE, None)):
        if isinstance(subject, URIRef) and isinstance(obj, Literal):
            subjects_by_value.setdefault(str(float(obj)), set()).add(subject)
    assert subjects_by_value["12.5"] != subjects_by_value["96.0"]
    assert len(subjects_by_value["30.0"]) == 2, (
        "impurity and photon-propagation 30 meV contributions must remain "
        "two distinct nodes"
    )


def test_case5_cross_paper_aggregation_adds_no_damage(
    aggregator: EmbeddingBasedAggregator,
) -> None:
    units = []
    input_value_damage: dict[URIRef, int] = {}
    input_unit_damage: dict[URIRef, int] = {}
    for index, name in enumerate(("paper1", "paper2")):
        graph = RDFGraph()
        graph.parse(DATA / "case5" / f"{name}.facts.ttl", format="turtle")
        input_value_damage.update(_distinct_value_counts(graph))
        input_unit_damage.update(_distinct_unit_counts(graph))
        units.append(
            ContentUnit(
                text=f"case5 {name}",
                index=index,
                doc_iri=URIRef(f"https://growgraph.dev/doc/case5-{name}"),
                graph=graph,
                type=OutputType.FACTS,
            )
        )

    merged = aggregator.aggregate_graphs(
        [units[0], units[1]], ontology_graph=RDFGraph()
    ).graph

    output_value_damage = _distinct_value_counts(merged)
    output_unit_damage = _distinct_unit_counts(merged)

    # Aggregation must not create nodes worse than the worst input node and
    # must not increase the count of multi-valued nodes.
    def worst(damage: dict[URIRef, int]) -> int:
        return max(damage.values(), default=1)

    def multi(damage: dict[URIRef, int]) -> int:
        return sum(1 for count in damage.values() if count > 1)

    assert worst(output_value_damage) <= worst(input_value_damage)
    assert multi(output_value_damage) <= multi(input_value_damage)
    assert worst(output_unit_damage) <= worst(input_unit_damage)
    assert multi(output_unit_damage) <= multi(input_unit_damage)
