"""Which IRIs an ontology is allowed to atomize.

An ontology graph mentions far more IRIs than it defines. Every object of every
``qudt:hasDimensionVector``, ``qudt:applicableSystem`` and ``owl:versionIRI`` triple is
an IRI the ontology *references* without saying anything about. Atomizing those mints
terms whose only text is a mangled local name, and such strings sit near the corpus
centroid in embedding space -- they rank against every query rather than none.
"""

from __future__ import annotations

from ontocast.config import EmbeddingConfig, VectorStoreConfig
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.vector_store.atomizer import GraphAtomizer
from ontocast.tool.vector_store.util import (
    atom_scope_fingerprint,
    embedding_model_fingerprint,
    sync_atomizer_from_store_config,
)

# Mirrors the shape of matsci-units: one described unit whose axioms point at a QUDT
# dimension vector, a system of units and a prefix -- none of which this graph defines.
_UNITS = """
@prefix qudt:  <http://qudt.org/schema/qudt/> .
@prefix unit:  <http://qudt.org/vocab/unit/> .
@prefix dv:    <http://qudt.org/vocab/dimensionvector/> .
@prefix sou:   <http://qudt.org/vocab/sou/> .
@prefix prefix: <http://qudt.org/vocab/prefix/> .
@prefix owl:   <http://www.w3.org/2002/07/owl#> .
@prefix rdfs:  <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex:    <https://example.org/units#> .

<https://example.org/units> a owl:Ontology ;
    owl:versionIRI <https://example.org/units/3.0.0> .

ex:millielectronvolt a qudt:Unit ;
    rdfs:label "millielectronvolt"@en ;
    qudt:symbol "meV" ;
    qudt:prefix prefix:Milli ;
    qudt:scalingOf unit:EV ;
    qudt:hasDimensionVector dv:A0E0L2I0M1H0T-2D0 ;
    qudt:applicableSystem sou:SI .
"""

# A term this graph only names, without describing it further. A vocabulary may label a
# term it otherwise reuses, and that label is exactly what retrieval needs.
_LABEL_ONLY = """
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex:   <https://example.org/d#> .

ex:Described a rdfs:Class ;
    rdfs:label "described"@en ;
    rdfs:seeAlso ex:NamedOnly .

ex:NamedOnly rdfs:label "named only"@en .
"""


def _iris(turtle: str, *, index_undescribed_iris: bool = False) -> set[str]:
    graph = RDFGraph._from_turtle_str(turtle)
    atomizer = GraphAtomizer(index_undescribed_iris=index_undescribed_iris)
    atoms = atomizer.atomize(source=Ontology(graph=graph), depth=1)
    return {atom.iri for atom in atoms}


def test_referenced_only_iris_do_not_become_atoms() -> None:
    """Objects the ontology never describes are references, not terms."""
    iris = _iris(_UNITS)

    assert "https://example.org/units#millielectronvolt" in iris
    for referenced in (
        "http://qudt.org/vocab/dimensionvector/A0E0L2I0M1H0T-2D0",
        "http://qudt.org/vocab/sou/SI",
        "http://qudt.org/vocab/prefix/Milli",
        "http://qudt.org/vocab/unit/EV",
    ):
        assert referenced not in iris


def test_version_iri_does_not_become_an_atom() -> None:
    """``owl:versionIRI`` objects would otherwise embed as bare version strings."""
    assert "https://example.org/units/3.0.0" not in _iris(_UNITS)


def test_a_label_alone_is_enough_to_be_atomized() -> None:
    """Naming a term counts as describing it, even with no other triple about it."""
    iris = _iris(_LABEL_ONLY)

    assert "https://example.org/d#Described" in iris
    assert "https://example.org/d#NamedOnly" in iris


def test_index_undescribed_iris_restores_the_previous_behaviour() -> None:
    """The old scope stays reachable for anyone relying on it."""
    iris = _iris(_UNITS, index_undescribed_iris=True)

    assert "http://qudt.org/vocab/dimensionvector/A0E0L2I0M1H0T-2D0" in iris
    assert "http://qudt.org/vocab/sou/SI" in iris


def test_store_config_drives_the_atomizer_scope() -> None:
    """The knob is reachable from settings, not only from the constructor."""
    atomizer = GraphAtomizer()
    sync_atomizer_from_store_config(
        atomizer, VectorStoreConfig(index_undescribed_iris=True)
    )
    assert atomizer.index_undescribed_iris is True

    sync_atomizer_from_store_config(atomizer, VectorStoreConfig())
    assert atomizer.index_undescribed_iris is False


def test_atom_scope_only_fingerprints_a_divergence() -> None:
    """Defaults must not change the fingerprint, or every store needs a reindex."""
    assert atom_scope_fingerprint(VectorStoreConfig()) is None
    assert atom_scope_fingerprint(VectorStoreConfig(index_undescribed_iris=True))
    assert atom_scope_fingerprint(VectorStoreConfig(embed_standard_vocab_iris=True))
    assert atom_scope_fingerprint(
        VectorStoreConfig(extra_excluded_namespace_prefixes=["http://qudt.org/"])
    )


def test_diverging_atom_scope_changes_the_embedding_fingerprint() -> None:
    """A store built under a different atom scope must not validate as compatible."""
    embedding = EmbeddingConfig(dimension=8, model_name="scope-test")
    default = embedding_model_fingerprint(embedding)
    widened = embedding_model_fingerprint(
        embedding,
        atom_scope=atom_scope_fingerprint(
            VectorStoreConfig(index_undescribed_iris=True)
        ),
    )

    assert default != widened
