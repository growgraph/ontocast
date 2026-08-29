"""The graph atomizer: what becomes an embeddable atom, and how it is typed.

Four modules over one ~200-line component, each re-importing the same
fixtures, are one module with four sections. Every original module docstring
is preserved above its section -- they carry the defect each group of tests
was written for, which is the reason these tests exist at all.
"""

from __future__ import annotations

from rdflib import OWL, RDF, RDFS, SKOS, Literal, URIRef

from ontocast.config import EmbeddingConfig, VectorStoreConfig
from ontocast.onto.facts import Facts
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.representation_text import ROLE_PREDICATE, ROLE_RESOURCE
from ontocast.tool.vector_store.atomizer import GraphAtomizer
from ontocast.tool.vector_store.core import GraphAtom
from ontocast.tool.vector_store.util import (
    atom_scope_fingerprint,
    embedding_model_fingerprint,
    sync_atomizer_from_store_config,
)

# -------------------------------------------------------------------------
# facts
# -------------------------------------------------------------------------


def test_atomizer_filters_minimal_provenance_predicates() -> None:
    """Provenance/reification triples should not leak into embeddings."""
    facts_namespace = "https://example.org/facts"

    graph = RDFGraph._from_turtle_str(
        """
        @prefix cd: <https://example.org/facts/> .
        @prefix prov: <http://www.w3.org/ns/prov#> .
        @prefix dcterms: <http://purl.org/dc/terms/> .
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix ex: <https://example.org/test/> .

        cd:Alpha a ex:Thing ;
            ex:knows cd:Beta ;
            dcterms:source ex:Doc .

        cd:Beta a ex:Thing .
        cd:Alpha prov:wasDerivedFrom ex:Other .
        cd:Alpha rdf:reifies ex:Whatever .
        """
    )

    facts = Facts(
        graph=graph,
        iri="https://example.org/factsGraph#doc1",
        ontology_id="doc1",
        hash="hash1",
        version="1.0.0",
        facts_namespace=facts_namespace,
    )

    atoms = GraphAtomizer().atomize(source=facts, depth=1)
    assert atoms

    alpha_atom = next(a for a in atoms if a.iri.endswith("/Alpha"))
    assert "was derived from" not in alpha_atom.core_representation
    assert "reifies" not in alpha_atom.core_representation

    assert "was derived from" not in alpha_atom.neighborhood_representation
    assert "has relation source" not in alpha_atom.neighborhood_representation
    assert "reifies" not in alpha_atom.neighborhood_representation


def test_atomizer_facts_focal_entities_are_cd_only() -> None:
    """Facts atomization should only create atoms for `cd:`-namespaced entities."""
    facts_namespace = "https://example.org/facts"
    outside_ns = "https://example.org/outside#"

    graph = RDFGraph._from_turtle_str(
        f"""
        @prefix cd: <{facts_namespace}/> .
        @prefix ex: <https://example.org/test/> .
        @prefix out: <{outside_ns}> .

        cd:Alpha a ex:Thing ;
            ex:relatedTo out:Outside .
        out:Outside a ex:Thing .
        """
    )

    facts = Facts(
        graph=graph,
        iri="https://example.org/factsGraph#doc1",
        ontology_id="doc1",
        hash="hash1",
        version="1.0.0",
        facts_namespace=facts_namespace,
    )

    atoms = GraphAtomizer().atomize(source=facts, depth=1)
    assert atoms
    assert all(a.iri.startswith(facts_namespace) for a in atoms)
    assert not any(outside_ns in a.iri for a in atoms)


def test_atomizer_facts_core_representation_includes_skos_alt_label() -> None:
    facts_namespace = "https://example.org/facts"
    graph = RDFGraph._from_turtle_str(
        f"""
        @prefix cd: <{facts_namespace}/> .
        @prefix skos: <{SKOS}> .
        @prefix ex: <https://example.org/test/> .

        cd:Entity a ex:Thing ;
            skos:prefLabel "Main label"@en ;
            skos:altLabel "Alternate name"@en .
        """
    )
    facts = Facts(
        graph=graph,
        iri="https://example.org/factsGraph#doc1",
        ontology_id="doc1",
        hash="hash1",
        version="1.0.0",
        facts_namespace=facts_namespace,
    )
    atom = next(
        a
        for a in GraphAtomizer().atomize(source=facts, depth=0)
        if a.iri.endswith("/Entity")
    )
    core = atom.core_representation.lower()
    assert "main label" in core
    assert "alternate name" in core


# -------------------------------------------------------------------------
# symbols
# -------------------------------------------------------------------------

# Mirrors the shape of a real QUDT unit: an authoritative qudt:symbol and
# qudt:ucumCode alongside one rdfs:label per language. unit:DEG_C ships 23 labels.
_QUDT_UNITS = """
@prefix qudt: <http://qudt.org/schema/qudt/> .
@prefix unit: <http://qudt.org/vocab/unit/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

unit:DEG_C a qudt:Unit ;
    rdfs:label "Degree Celsius"@en ,
        "Grad Celsius"@de ,
        "Degre Celsius"@fr ,
        "Grado Celsius"@es ,
        "Celsius Fok"@hu ,
        "Darjah Celsius"@ms ,
        "Stupen Celsia"@cs ;
    qudt:symbol "degC" ;
    qudt:ucumCode "Cel" .

unit:Plain a qudt:Unit ;
    rdfs:label "Plain Unit"@en .
"""


def _atom_for(local_name: str) -> GraphAtom:
    graph = RDFGraph._from_turtle_str(_QUDT_UNITS)
    atoms = GraphAtomizer().atomize(source=Ontology(graph=graph), depth=1)
    return next(a for a in atoms if a.iri.endswith(local_name))


def test_qudt_symbol_survives_the_surface_form_cap() -> None:
    """A unit's symbol must reach the sparse lane despite many language labels.

    Surface forms are capped (``minimal_representation_label_limit``, default 5) and
    collected in predicate priority order. Were the symbol predicates simply appended
    after ``rdfs:label``, a unit declaring one label per language would exhaust the cap
    before any symbol was reached -- leaving it unfindable by "degC" or "Cel", which is
    the form measurement prose actually uses.
    """
    atom = _atom_for("DEG_C")

    assert "degc" in atom.minimal_representation
    assert "cel" in atom.minimal_representation


def test_qudt_symbol_does_not_displace_the_readable_name() -> None:
    """The core representation still leads with the human label, not the symbol."""
    atom = _atom_for("DEG_C")

    assert atom.core_representation.startswith("degree celsius")
    assert "degc" in atom.core_representation


def test_english_labels_outrank_other_languages() -> None:
    """A multi-language term is named in English, not by alphabetical accident.

    Literals are sorted for reproducibility, which also decides the display name.
    Without a language rank, ``unit:DEG_C`` sorts to the Hungarian "Celsius Fok".
    Other languages are demoted rather than dropped, so a non-English corpus keeps
    its aliases.
    """
    atom = _atom_for("DEG_C")
    text = atom.minimal_representation

    assert text.index("degree celsius") < text.index("celsius fok")


def test_entities_without_symbols_are_unaffected() -> None:
    """Terms declaring no symbol keep their previous label-only surface forms."""
    atom = _atom_for("Plain")

    assert "plain unit" in atom.minimal_representation


def test_symbol_predicates_are_configurable() -> None:
    """The indexing half of the symbol contract must follow config.

    Retrieval has always read its symbol predicates from config; the atomizer
    hardcoded them, so overriding the knob changed what surfaced without
    changing what was indexed.
    """
    custom = URIRef("http://example.com/vocab#code")
    graph = RDFGraph()
    subject = URIRef("http://example.com/onto#Widget")
    graph.add((subject, RDF.type, OWL.Class))
    graph.add((subject, RDFS.label, Literal("Widget")))
    graph.add((subject, custom, Literal("WDG")))

    default_atomizer = GraphAtomizer()
    configured = GraphAtomizer(symbol_predicates=[str(custom)])

    assert "WDG" not in default_atomizer._collect_raw_literals(
        graph, subject, default_atomizer._symbol_predicate_refs(), max_items=8
    )
    assert "WDG" in configured._collect_raw_literals(
        graph, subject, configured._symbol_predicate_refs(), max_items=8
    )


def test_label_predicates_are_configurable() -> None:
    custom = URIRef("http://example.com/vocab#displayName")
    graph = RDFGraph()
    subject = URIRef("http://example.com/onto#Widget")
    graph.add((subject, RDF.type, OWL.Class))
    graph.add((subject, custom, Literal("Widget display")))

    configured = GraphAtomizer(label_predicates=[str(custom)])

    # _collect_literals normalizes case.
    assert "widget display" in configured._collect_literals(
        graph, subject, configured._label_predicate_refs(), 5
    )
    assert (
        GraphAtomizer()._collect_literals(
            graph, subject, GraphAtomizer()._label_predicate_refs(), 5
        )
        == []
    )


# -------------------------------------------------------------------------
# entity_role
# -------------------------------------------------------------------------
# Which atoms count as predicates.
#
# Role was once read off predicate-position usage alone. A TBox-only module never
# uses its own properties as predicates -- it asserts them as subjects
# (``ex:hasResult a owl:ObjectProperty ; rdfs:domain ... ; rdfs:range ...``) --
# so a catalog of pure schema modules classified nearly every property as a
# resource. Resource-role atoms get a resource-shaped neighborhood
# representation, which for a property with no outgoing domain assertions is
# empty, and an empty string is what then goes into the neighborhood vector. The
# channel meant to carry relational context went blind to the very terms that
# carry the graph structure.

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


# -------------------------------------------------------------------------
# focal_scope
# -------------------------------------------------------------------------
# Which IRIs an ontology is allowed to atomize.
#
# An ontology graph mentions far more IRIs than it defines. Every object of every
# ``qudt:hasDimensionVector``, ``qudt:applicableSystem`` and ``owl:versionIRI`` triple is
# an IRI the ontology *references* without saying anything about. Atomizing those mints
# terms whose only text is a mangled local name, and such strings sit near the corpus
# centroid in embedding space -- they rank against every query rather than none.

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
