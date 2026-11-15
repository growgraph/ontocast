"""A node written on a datatype property moves to the object property paired with it."""

import pytest
from rdflib import BNode, Literal, URIRef
from rdflib.namespace import SOSA

from ontocast.onto.model import FactsUnitFindingKind
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.well_known import well_known_object_pairs
from ontocast.tool.facts_validation import repair_paired_object_properties

pytestmark = pytest.mark.unit

_OBS = URIRef("http://example.org/observation_1")
_VALUE = URIRef("http://example.org/value_1")


def test_sosa_pairs_its_simple_result_with_its_result() -> None:
    assert well_known_object_pairs()[str(SOSA.hasSimpleResult)] == str(SOSA.hasResult)


def test_a_node_on_the_datatype_property_moves_to_the_object_property() -> None:
    graph = RDFGraph()
    blank = BNode()
    graph.add((_OBS, SOSA.hasSimpleResult, _VALUE))
    graph.add((_OBS, SOSA.hasSimpleResult, blank))

    moved, records = repair_paired_object_properties(graph)

    assert moved == 2
    assert set(graph) == {(_OBS, SOSA.hasResult, _VALUE), (_OBS, SOSA.hasResult, blank)}
    assert {record.kind for record in records} == {
        FactsUnitFindingKind.PAIRED_OBJECT_PROPERTY
    }


def test_a_literal_on_the_datatype_property_stays() -> None:
    graph = RDFGraph()
    graph.add((_OBS, SOSA.hasSimpleResult, Literal("22 °C")))

    assert repair_paired_object_properties(graph) == (0, [])
    assert (_OBS, SOSA.hasSimpleResult, Literal("22 °C")) in graph
