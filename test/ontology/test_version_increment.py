"""Semantic-version bumps follow the kind of ontology change applied."""

import pytest

from ontocast.onto.ontology import bump_version, increment_type_for
from ontocast.onto.ontology_header import stamp_version
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import GraphUpdate, TripleOp
from test.ontology_helpers import stamped_ontology

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
    increment, _reason = increment_type_for([update])
    return bump_version("1.2.3", increment)


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


def test_no_updates_is_a_patch() -> None:
    assert increment_type_for([]) == ("patch", "no updates")


def test_bump_version_refuses_a_non_semantic_label() -> None:
    with pytest.raises(ValueError, match="not a semantic version"):
        bump_version("3.0", "patch")


def test_stamping_a_child_applies_the_increment_to_the_base_label() -> None:
    base = stamped_ontology(
        _PREFIXES
        + """
        <https://example.org/onto> a owl:Ontology ; owl:versionInfo "1.2.3" .
        ex:A a owl:Class .
        """
    )
    assert base.version == "1.2.3"
    updated_graph = base.graph.copy()
    body = " ".join(f"ex:p{i} a owl:ObjectProperty ." for i in range(5))
    update = _update("insert", body)
    for op in update.triple_operations:
        for triple in op.graph:
            updated_graph.add(triple)

    child = stamp_version(base.derive_updated_version(updated_graph), updates=[update])

    assert child is not None
    assert child.version == "1.3.0"
