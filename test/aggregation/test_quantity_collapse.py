"""Regression suite for aggregation collapsing distinct quantity values.

``test/data/pre_merge_unit.jsonld`` is a verbatim pre-merge LLM output
whose aggregation merged distinct quantities into one node (12.5 and
96 meV; a photon-propagation contribution absorbed into an impurity
contribution). The tests assert the merge guards keep every quantity
distinct.

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
    graph.parse(DATA / "pre_merge_unit.jsonld", format="json-ld")
    unit = ContentUnit(
        text="excerpt",
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
