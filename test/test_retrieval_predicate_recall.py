"""Getting predicates into the prompt.

Two independent mechanisms, both aimed at the same failure: a snapshot full of
classes the renderer cannot link, because no property connecting them survived
retrieval. The renderer's recourse is to improvise a predicate from outside the
catalog, which reads as a plausible result and is not one.

- ``per_role_atom_floor`` reserves seed slots for predicate-role atoms, which
  lose a shared ranking because prose reads as noun phrases.
- the schema closure admits, deterministically, the properties whose declared
  domain/range names an admitted class, and the classes an admitted property
  declares.
"""

from __future__ import annotations

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.representation_text import ROLE_PREDICATE, ROLE_RESOURCE
from ontocast.tool.vector_store.core import GraphAtom, OntologySearchHit
from ontocast.tool.vector_store.patch_retriever import (
    _schema_closure_entities,
    _select_hits_round_robin_by_ontology,
)

ONTO = "https://x.org/o"


def _hit(iri: str, score: float, role: str = ROLE_RESOURCE) -> OntologySearchHit:
    atom = GraphAtom(
        atom_id=f"atom-{iri}",
        ontology_iri=ONTO,
        ontology_id="o",
        ontology_hash="h",
        ontology_version="1",
        iri=iri,
        entity_role=role,
        core_representation=iri,
        minimal_representation=iri,
        neighborhood_representation="",
        score=score,
    )
    return OntologySearchHit(atom=atom, score=score)


def _mixed_ranking() -> list[OntologySearchHit]:
    """Six classes outranking every property — the shape measured on case6."""
    hits = [_hit(f"{ONTO}#Class{i}", 0.90 - i * 0.01) for i in range(6)]
    hits.append(_hit(f"{ONTO}#hasResult", 0.40, role=ROLE_PREDICATE))
    hits.append(_hit(f"{ONTO}#hasBound", 0.35, role=ROLE_PREDICATE))
    return hits


def test_without_a_role_floor_properties_are_crowded_out() -> None:
    selected = _select_hits_round_robin_by_ontology(
        _mixed_ranking(), per_ontology_seed_quota=0, max_atoms=4
    )

    assert all(hit.atom.entity_role == ROLE_RESOURCE for hit in selected)


def test_role_floor_reserves_slots_for_predicates() -> None:
    selected = _select_hits_round_robin_by_ontology(
        _mixed_ranking(),
        per_ontology_seed_quota=0,
        max_atoms=4,
        per_role_atom_floor=2,
    )

    predicates = [hit for hit in selected if hit.atom.entity_role == ROLE_PREDICATE]
    assert len(predicates) == 2
    assert len(selected) == 4
    # Reserved hits are not handed out twice.
    assert len({hit.atom.iri for hit in selected}) == 4
    # The leftover slots still go to the best-scoring classes.
    assert f"{ONTO}#Class0" in {hit.atom.iri for hit in selected}


def test_role_floor_is_a_floor_not_a_quota() -> None:
    """A floor larger than the predicate supply must not starve the fill."""
    selected = _select_hits_round_robin_by_ontology(
        _mixed_ranking(),
        per_ontology_seed_quota=0,
        max_atoms=6,
        per_role_atom_floor=99,
    )

    assert len(selected) == 6
    assert len([h for h in selected if h.atom.entity_role == ROLE_PREDICATE]) == 2


_SCHEMA = """
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex:   <https://x.org/o#> .

ex:Observation a owl:Class .
ex:QuantitativeObservation a owl:Class ; rdfs:subClassOf ex:Observation .
ex:QuantityValue a owl:Class .
ex:Unrelated a owl:Class .

ex:hasQuantityResult a owl:ObjectProperty ;
    rdfs:domain ex:QuantitativeObservation ;
    rdfs:range ex:QuantityValue .

ex:hasObservation a owl:ObjectProperty ;
    rdfs:domain ex:Observation ;
    rdfs:range ex:Observation .

ex:unrelatedLink a owl:ObjectProperty ;
    rdfs:domain ex:Unrelated ;
    rdfs:range ex:Unrelated .
"""


def _closure(
    seeds: list[str], *, max_entities: int = 32, ancestor_depth: int = 2
) -> dict[str, str]:
    graph = RDFGraph._from_turtle_str(_SCHEMA)
    return _schema_closure_entities(
        graph, seeds, max_entities=max_entities, ancestor_depth=ancestor_depth
    )


def test_class_seed_pulls_in_properties_declared_on_it() -> None:
    added = _closure(["https://x.org/o#Observation"])

    assert added["https://x.org/o#hasObservation"] == ROLE_PREDICATE
    assert "https://x.org/o#unrelatedLink" not in added


def test_class_seed_reaches_properties_declared_on_an_ancestor() -> None:
    """Properties are usually declared on an ancestor of the mentioned class."""
    added = _closure(["https://x.org/o#QuantitativeObservation"])

    assert "https://x.org/o#hasQuantityResult" in added
    assert "https://x.org/o#hasObservation" in added


def test_ancestor_depth_zero_stops_at_the_seed() -> None:
    added = _closure(["https://x.org/o#QuantitativeObservation"], ancestor_depth=0)

    assert "https://x.org/o#hasQuantityResult" in added
    assert "https://x.org/o#hasObservation" not in added


def test_property_seed_pulls_in_its_domain_and_range() -> None:
    added = _closure(["https://x.org/o#hasQuantityResult"])

    assert added["https://x.org/o#QuantitativeObservation"] == ROLE_RESOURCE
    assert added["https://x.org/o#QuantityValue"] == ROLE_RESOURCE


def test_seeds_are_never_returned_as_additions() -> None:
    seed = "https://x.org/o#Observation"
    assert seed not in _closure([seed])


def test_closure_is_disabled_at_zero() -> None:
    assert _closure(["https://x.org/o#Observation"], max_entities=0) == {}


def test_closure_respects_its_cap() -> None:
    added = _closure(["https://x.org/o#QuantitativeObservation"], max_entities=1)

    assert len(added) == 1
