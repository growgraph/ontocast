"""The ontology chapter as a term sheet.

The chapter is most of a facts prompt, and most of the chapter is RDF
scaffolding and prose no extractor reads. Rendering it as a listing is only
admissible if nothing the model needs to *use* a term is lost -- so these pin
what must survive, and what must refuse to happen on the ontology path.
"""

from __future__ import annotations

import pytest
from rdflib import OWL, RDF, RDFS, SKOS, Literal, Namespace, URIRef

from ontocast.config.settings import RenderMode, ServerConfig
from ontocast.onto.enum import LLMGraphFormat, OntologyChapterFormat
from ontocast.onto.ontology_condense import TextCaps
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.prompt.graph_format import get_graph_format_profile
from ontocast.prompt.term_sheet import build_ontology_term_sheet

EX = Namespace("https://example.org/onto#")


@pytest.fixture
def catalog() -> RDFGraph:
    """One of each shape: a class hierarchy, a property, and a typed individual."""
    graph = RDFGraph()
    graph.bind("ex", EX)
    graph.add((EX.Sample, RDF.type, OWL.Class))
    graph.add((EX.Sample, RDFS.label, Literal("Sample")))
    graph.add((EX.Sample, RDFS.comment, Literal("A portion of material.")))
    graph.add((EX.Powder, RDF.type, OWL.Class))
    graph.add((EX.Powder, RDFS.label, Literal("Powder")))
    graph.add((EX.Powder, RDFS.subClassOf, EX.Sample))
    graph.add((EX.Powder, SKOS.altLabel, Literal("powdered solid")))
    graph.add((EX.Powder, SKOS.scopeNote, Literal("Only for a milled sample.")))
    graph.add((EX.hasMass, RDF.type, OWL.ObjectProperty))
    graph.add((EX.hasMass, RDFS.label, Literal("has mass")))
    graph.add((EX.hasMass, RDFS.domain, EX.Sample))
    graph.add((EX.hasMass, RDFS.range, EX.Mass))
    graph.add((EX.gram, RDF.type, EX.Unit))
    graph.add((EX.gram, RDF.type, OWL.NamedIndividual))
    graph.add((EX.gram, RDFS.label, Literal("gram")))
    graph.add((EX.gram, SKOS.altLabel, Literal("g")))
    return graph


def test_every_load_bearing_term_survives(catalog: RDFGraph) -> None:
    """The one property that makes the representation admissible at all.

    A term absent from the sheet cannot be used, and the model is told to use
    only what the sheet lists -- so an omission is not a smaller prompt, it is a
    term silently withdrawn from the vocabulary.
    """
    sheet = build_ontology_term_sheet(catalog)

    named = {
        subject
        for predicate in (
            RDFS.label,
            RDF.type,
            RDFS.subClassOf,
            RDFS.domain,
            RDFS.range,
            RDFS.subPropertyOf,
        )
        for subject in catalog.subjects(predicate, None)
        if isinstance(subject, URIRef)
    }

    missing = [
        catalog.namespace_manager.qname(term)
        for term in named
        if catalog.namespace_manager.qname(term) not in sheet
    ]
    assert not missing


def test_relations_and_signatures_are_stated(catalog: RDFGraph) -> None:
    sheet = build_ontology_term_sheet(catalog)

    assert "ex:Powder" in sheet and "< ex:Sample" in sheet
    assert "ex:hasMass" in sheet and "ex:Sample -> ex:Mass" in sheet
    assert "ex:gram" in sheet and ": ex:Unit" in sheet
    assert "owl:NamedIndividual" not in sheet, "a type saying nothing is noise"


def test_surface_forms_and_usage_contracts_survive(catalog: RDFGraph) -> None:
    """Alternative labels are what a document match has to work with."""
    sheet = build_ontology_term_sheet(catalog)

    assert "powdered surface" not in sheet
    assert "~ powdered solid" in sheet
    assert "~ g" in sheet
    assert "note: Only for a milled sample." in sheet


def test_prose_is_dropped(catalog: RDFGraph) -> None:
    """rdfs:comment is written for a human browsing the ontology."""
    assert "A portion of material." not in build_ontology_term_sheet(catalog)


def test_rendering_is_deterministic(catalog: RDFGraph) -> None:
    """A stable chapter is what a provider's prefix cache can serve twice."""
    assert build_ontology_term_sheet(catalog) == build_ontology_term_sheet(catalog)


def test_blank_nodes_are_never_named(catalog: RDFGraph) -> None:
    """rdflib mints bnode labels at random; a sheet that printed them would
    differ between processes for the same catalog."""
    from rdflib import BNode

    restriction = RDFGraph()
    restriction.bind("ex", EX)
    for triple in catalog:
        restriction.add(triple)

    bnode = BNode()
    restriction.add((EX.Powder, RDFS.subClassOf, bnode))
    restriction.add((bnode, RDF.type, OWL.Restriction))
    restriction.add((bnode, OWL.onProperty, EX.hasMass))

    sheet = build_ontology_term_sheet(restriction)

    assert str(bnode) not in sheet
    assert "ex:Powder" in sheet and "< ex:Sample" in sheet


def test_empty_snapshot_renders_nothing() -> None:
    assert build_ontology_term_sheet(RDFGraph()) == ""


def test_term_sheet_chapter_is_far_cheaper_than_the_graph(catalog: RDFGraph) -> None:
    jsonld = get_graph_format_profile(LLMGraphFormat.JSONLD)
    sheet = get_graph_format_profile(
        LLMGraphFormat.JSONLD,
        ontology_chapter_format=OntologyChapterFormat.TERM_SHEET,
    )

    graph_chapter = jsonld.format_ontology_chapter(catalog, max_triples=None)
    sheet_chapter = sheet.format_ontology_chapter(catalog, max_triples=None)

    assert len(sheet_chapter) < len(graph_chapter)
    assert "```" not in sheet_chapter, "a listing is not a fenced serialization"
    assert sheet_chapter.lstrip().startswith("# ONTOLOGY")


def test_term_sheet_does_not_collide_with_turtle_in_the_memo(
    catalog: RDFGraph,
) -> None:
    """Both report a Turtle wire; only the discriminator tells them apart."""
    turtle = get_graph_format_profile(
        LLMGraphFormat.JSONLD, ontology_chapter_format=OntologyChapterFormat.TURTLE
    )
    sheet = get_graph_format_profile(
        LLMGraphFormat.JSONLD,
        ontology_chapter_format=OntologyChapterFormat.TERM_SHEET,
    )

    assert turtle.ontology_chapter_wire == sheet.ontology_chapter_wire
    assert turtle.ontology_chapter_discriminator != sheet.ontology_chapter_discriminator


def test_text_caps_reach_the_term_sheet(catalog: RDFGraph) -> None:
    profile = get_graph_format_profile(
        LLMGraphFormat.JSONLD,
        ontology_chapter_format=OntologyChapterFormat.TERM_SHEET,
    )

    chapter = profile.format_ontology_chapter(
        catalog, max_triples=None, text_caps=TextCaps(contract=10)
    )

    assert "Only for a milled sample." not in chapter
    assert "note: Only" in chapter


@pytest.mark.parametrize("mode", [RenderMode.ONTOLOGY, RenderMode.ONTOLOGY_AND_FACTS])
def test_term_sheet_is_rejected_on_the_ontology_path(mode: RenderMode) -> None:
    """The ontology loop patches the statements in its chapter.

    A listing has nothing to insert into or delete from, and a silent fallback
    would spend an ontology pass producing patches nobody could apply while the
    manifest recorded a setting that never took effect.
    """
    with pytest.raises(ValueError, match="RENDER_MODE=facts"):
        ServerConfig(
            render_mode=mode,
            ontology_chapter_format=OntologyChapterFormat.TERM_SHEET,
        )


def test_term_sheet_is_allowed_on_a_facts_run() -> None:
    config = ServerConfig(
        render_mode=RenderMode.FACTS,
        ontology_chapter_format=OntologyChapterFormat.TERM_SHEET,
    )
    assert config.ontology_chapter_format == OntologyChapterFormat.TERM_SHEET


def test_text_caps_property_is_inactive_by_default() -> None:
    assert not ServerConfig(render_mode=RenderMode.FACTS).ontology_text_caps.active
