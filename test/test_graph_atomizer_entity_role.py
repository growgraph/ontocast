"""Which atoms count as predicates.

Role was once read off predicate-position usage alone. A TBox-only module never
uses its own properties as predicates -- it asserts them as subjects
(``ex:hasResult a owl:ObjectProperty ; rdfs:domain ... ; rdfs:range ...``) --
so a catalog of pure schema modules classified nearly every property as a
resource. Resource-role atoms get a resource-shaped neighborhood
representation, which for a property with no outgoing domain assertions is
empty, and an empty string is what then goes into the neighborhood vector. The
channel meant to carry relational context went blind to the very terms that
carry the graph structure.
"""

from __future__ import annotations

from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.representation_text import ROLE_PREDICATE, ROLE_RESOURCE
from ontocast.tool.vector_store.atomizer import GraphAtomizer

# Pure schema: properties are declared and described, never used as predicates.
_TBOX_ONLY = """
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
@prefix ex:   <https://example.org/o#> .

ex:Measurement a owl:Class ;
    rdfs:label "measurement"@en .

ex:Bound a owl:Class ;
    rdfs:label "bound"@en .

ex:hasLowerBound a owl:ObjectProperty ;
    rdfs:label "has lower bound"@en ;
    rdfs:domain ex:Measurement ;
    rdfs:range ex:Bound .

ex:uncertainty a owl:DatatypeProperty ;
    rdfs:label "uncertainty"@en ;
    rdfs:domain ex:Measurement ;
    rdfs:range xsd:decimal .

ex:isExact a owl:FunctionalProperty ;
    rdfs:label "is exact"@en ;
    rdfs:domain ex:Measurement .
"""

# Same shape, plus one assertion that puts a property in predicate position.
_WITH_ABOX = """
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex:   <https://example.org/o#> .

ex:Sample a owl:Class ;
    rdfs:label "sample"@en .

ex:hasPhase a owl:ObjectProperty ;
    rdfs:label "has phase"@en ;
    rdfs:domain ex:Sample .

ex:sample_a a ex:Sample ;
    rdfs:label "sample a"@en ;
    ex:hasPhase ex:alpha .

ex:alpha a owl:NamedIndividual ;
    rdfs:label "alpha"@en .
"""


def _roles(turtle: str) -> dict[str, str | None]:
    graph = RDFGraph._from_turtle_str(turtle)
    atoms = GraphAtomizer().atomize(source=Ontology(graph=graph), depth=1)
    return {atom.iri: atom.entity_role for atom in atoms}


def _atoms(turtle: str) -> dict[str, list]:
    graph = RDFGraph._from_turtle_str(turtle)
    by_iri: dict[str, list] = {}
    for atom in GraphAtomizer().atomize(source=Ontology(graph=graph), depth=1):
        by_iri.setdefault(atom.iri, []).append(atom)
    return by_iri


def test_declared_properties_are_predicates_without_any_abox() -> None:
    """Declaration is enough; a schema module need not exemplify its own terms."""
    roles = _roles(_TBOX_ONLY)

    assert roles["https://example.org/o#hasLowerBound"] == ROLE_PREDICATE
    assert roles["https://example.org/o#uncertainty"] == ROLE_PREDICATE
    assert roles["https://example.org/o#isExact"] == ROLE_PREDICATE


def test_classes_stay_resources() -> None:
    """The widened rule must not sweep classes into the predicate role."""
    roles = _roles(_TBOX_ONLY)

    assert roles["https://example.org/o#Measurement"] == ROLE_RESOURCE
    assert roles["https://example.org/o#Bound"] == ROLE_RESOURCE


def test_predicate_position_usage_still_counts() -> None:
    """The original signal is kept, not replaced."""
    roles = _roles(_WITH_ABOX)

    assert roles["https://example.org/o#hasPhase"] == ROLE_PREDICATE
    assert roles["https://example.org/o#sample_a"] == ROLE_RESOURCE
    assert roles["https://example.org/o#alpha"] == ROLE_RESOURCE


def test_predicate_atoms_carry_a_neighborhood_representation() -> None:
    """The point of the role: domain/range text reaches the neighborhood vector."""
    atoms = _atoms(_TBOX_ONLY)

    variants = atoms["https://example.org/o#hasLowerBound"]
    texts = [(atom.neighborhood_representation or "").strip() for atom in variants]
    assert any(texts), "declared property embedded an empty neighborhood string"
    joined = " ".join(texts)
    assert "measurement" in joined
    assert "bound" in joined
