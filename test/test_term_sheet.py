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


def test_one_prose_field_per_term_the_contract_winning(catalog: RDFGraph) -> None:
    """Each term gets exactly one note, and a scope note outranks a comment.

    ``ex:Sample`` has only a comment, so the comment is its note -- dropping it
    would leave the term a name and a parent. ``ex:Powder`` has both, and there
    the comment restates for a human what the scope note states as a contract,
    so only the contract is rendered.
    """
    sheet = build_ontology_term_sheet(catalog)
    assert "note: A portion of material." in sheet
    assert "note: Only for a milled sample." in sheet
    # The indented form: the legend in the header also spells "note:".
    assert sheet.count("      note:") == 2


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


def test_a_term_with_several_names_keeps_all_of_them() -> None:
    """Catalogs spell terms more than one way; the sheet must not pick one.

    QUDT names its units in both British and American English, and the spelling
    a source document used is exactly the one a match needs. Choosing a display
    name and discarding the rest throws that away silently.
    """
    graph = RDFGraph()
    graph.bind("ex", EX)
    graph.add((EX.Metre, RDF.type, EX.Unit))
    graph.add((EX.Metre, RDFS.label, Literal("Metre")))
    graph.add((EX.Metre, RDFS.label, Literal("Meter")))
    graph.add((EX.Metre, SKOS.altLabel, Literal("m")))

    sheet = build_ontology_term_sheet(graph)

    assert "Metre" in sheet and "Meter" in sheet and "~ m" in sheet


def test_the_display_name_does_not_depend_on_iteration_order() -> None:
    """rdflib yields objects in hash order, randomised per process.

    Reading "the first label" from that picks a different name for the same
    term on a different run, drifting the chapter with no visible cause and
    defeating every cache keyed on it.
    """
    graph = RDFGraph()
    graph.bind("ex", EX)
    graph.add((EX.T, RDF.type, OWL.Class))
    for name in ("Zeta", "Alpha", "Mu", "Beta"):
        graph.add((EX.T, RDFS.label, Literal(name)))

    line = next(
        row for row in build_ontology_term_sheet(graph).splitlines() if "ex:T" in row
    )

    # Shortest, then alphabetical -- a total order, not an arrival order.
    assert line.split('"')[1] == "Mu"
    assert all(name in line for name in ("Zeta", "Alpha", "Beta"))


def test_the_display_name_is_never_repeated_as_a_surface_form() -> None:
    graph = RDFGraph()
    graph.bind("ex", EX)
    graph.add((EX.T, RDF.type, OWL.Class))
    graph.add((EX.T, RDFS.label, Literal("Sample")))
    graph.add((EX.T, SKOS.altLabel, Literal("Sample")))

    line = next(
        row for row in build_ontology_term_sheet(graph).splitlines() if "ex:T" in row
    )

    assert line.count("Sample") == 1


def test_comment_is_the_note_when_a_term_has_no_scope_note() -> None:
    """A term's only description must survive the change of representation.

    Most catalog terms carry an ``rdfs:comment`` and no ``skos:scopeNote``.
    Dropping the comment as "prose the extractor does not read" would leave
    those terms as a name and a parent -- a loss of content, not a cheaper
    encoding of it, and nothing downstream can recover it.
    """
    graph = RDFGraph()
    graph.parse(
        data="""
        @prefix ex: <https://example.org/o#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        ex:Anneal a owl:Class ;
            rdfs:label "Anneal" ;
            rdfs:comment "Heating held below the melting point." .
        """,
        format="turtle",
    )
    sheet = build_ontology_term_sheet(graph)
    assert "note: Heating held below the melting point." in sheet


def test_scope_note_outranks_comment() -> None:
    """Where both exist the contract wins: the comment restates it for a human."""
    graph = RDFGraph()
    graph.parse(
        data="""
        @prefix ex: <https://example.org/o#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix skos: <http://www.w3.org/2004/02/skos/core#> .
        ex:Anneal a owl:Class ;
            rdfs:label "Anneal" ;
            skos:scopeNote "Use for a stated hold; not for a ramp." ;
            rdfs:comment "Heating held below the melting point." .
        """,
        format="turtle",
    )
    sheet = build_ontology_term_sheet(graph)
    assert "note: Use for a stated hold; not for a ramp." in sheet
    assert "Heating held below the melting point" not in sheet


def test_auto_is_the_shipped_default() -> None:
    """The default must be the mode-aware member, not a chapter.

    Pinning it here is what makes the two resolution tests below a statement
    about what a deployment gets rather than about a value they set themselves.
    """
    assert (
        ServerConfig.model_fields["ontology_chapter_format"].default
        == OntologyChapterFormat.AUTO
    )


def test_auto_takes_the_term_sheet_on_a_facts_run() -> None:
    """The cheapest chapter a facts run can legally read."""
    config = ServerConfig(
        render_mode=RenderMode.FACTS,
        ontology_chapter_format=OntologyChapterFormat.AUTO,
    )
    assert config.ontology_chapter_format == OntologyChapterFormat.TERM_SHEET


@pytest.mark.parametrize("mode", [RenderMode.ONTOLOGY, RenderMode.ONTOLOGY_AND_FACTS])
def test_auto_falls_back_to_a_graph_where_a_listing_is_illegal(
    mode: RenderMode,
) -> None:
    """`auto` never fails: where a listing cannot be patched, it yields a graph.

    That is the whole difference from an explicit `term_sheet`, which is
    rejected on these modes -- asking for something impossible is an error,
    while asking for the best available is not.
    """
    config = ServerConfig(
        render_mode=mode, ontology_chapter_format=OntologyChapterFormat.AUTO
    )
    assert config.ontology_chapter_format == OntologyChapterFormat.INHERIT


def test_auto_is_resolved_before_anything_downstream_sees_it() -> None:
    """Resolution happens once, in the config -- not at each point of use.

    Every consumer reads this field, so a value still meaning "decide later"
    would reach the prompt profile, the LLM cache key and the run manifest as a
    name for no chapter in particular. The profile lookup refuses it, which is
    the backstop for that.
    """
    for mode in RenderMode:
        assert (
            ServerConfig(render_mode=mode).ontology_chapter_format
            != OntologyChapterFormat.AUTO
        )
    with pytest.raises(ValueError, match="resolved against the render mode"):
        get_graph_format_profile(
            LLMGraphFormat.JSONLD,
            ontology_chapter_format=OntologyChapterFormat.AUTO,
        )
