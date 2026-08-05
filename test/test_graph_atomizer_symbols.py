from __future__ import annotations

from rdflib import OWL, RDF, RDFS, Literal, URIRef

from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.vector_store.atomizer import GraphAtomizer
from ontocast.tool.vector_store.core import GraphAtom

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
