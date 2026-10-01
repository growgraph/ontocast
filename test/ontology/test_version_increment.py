"""Semantic-version bumps follow the kind of ontology change applied."""

import pytest

from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import GraphUpdate, TripleOp

pytestmark = pytest.mark.unit

_PREFIXES = """
@prefix ex: <https://example.org/onto#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
"""


def _update(kind: str, body: str) -> GraphUpdate:
    graph = RDFGraph._from_turtle_str(_PREFIXES + body)
    return GraphUpdate(triple_operations=[TripleOp(type=kind, graph=graph)])  # type: ignore[arg-type]


def _bump(update: GraphUpdate) -> str:
    ontology = Ontology(iri="https://example.org/onto", version="1.2.3")
    ontology.mark_as_updated([update])
    return str(ontology.version)


def test_adding_a_few_classes_is_a_patch() -> None:
    assert _bump(_update("insert", "ex:A a owl:Class . ex:B a owl:Class .")) == "1.2.4"


def test_adding_many_properties_is_a_minor() -> None:
    body = " ".join(f"ex:p{i} a owl:ObjectProperty ." for i in range(5))
    assert _bump(_update("insert", body)) == "1.3.0"


def test_deleting_a_label_is_a_minor() -> None:
    assert _bump(_update("delete", 'ex:A rdfs:label "A" .')) == "1.3.0"


def test_deleting_several_classes_is_a_major() -> None:
    body = " ".join(f'ex:C{i} a owl:Class ; rdfs:label "C{i}" .' for i in range(3))
    assert _bump(_update("delete", body)) == "2.0.0"


def test_deleting_several_properties_is_a_major() -> None:
    body = " ".join(
        f'ex:p{i} a owl:DatatypeProperty ; rdfs:label "p{i}" .' for i in range(4)
    )
    assert _bump(_update("delete", body)) == "2.0.0"
