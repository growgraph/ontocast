"""The shapes-derived conformance chapter: domain knowledge in, no domain
strings of our own.

Every line of the chapter must come from the deployment's shapes graph --
``sh:message`` verbatim where the author wrote one, a synthesized structural
line otherwise. A SPARQL constraint without a message cannot be summarized
and is omitted (warned), never guessed at.
"""

import pytest
from rdflib import Graph

from ontocast.prompt.shapes_contract import (
    CHAPTER_HEADING,
    contract_terms,
    derive_shape_requirements,
    format_conformance_chapter,
)

pytestmark = pytest.mark.unit

SHAPES_TTL = """
@prefix sh:    <http://www.w3.org/ns/shacl#> .
@prefix xsd:   <http://www.w3.org/2001/XMLSchema#> .
@prefix qqval: <https://example.org/qqval#> .
@prefix qudt:  <https://example.org/qudt#> .

qqval:ValueShape a sh:NodeShape ;
    sh:targetClass qqval:QualifiedQuantityValue ;
    sh:property [
        sh:path qqval:epistemicQualifier ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class qqval:EpistemicQualifier ;
        sh:message "QualifiedQuantityValue requires exactly one epistemicQualifier."
    ] ,
    [
        sh:path qudt:unit ;
        sh:minCount 1 ;
    ] ;
    sh:sparql [
        sh:message "Exact values forbid bounds." ;
        sh:select "SELECT $this WHERE {}"
    ] .

qqval:SilentSparqlShape a sh:NodeShape ;
    sh:targetClass qqval:QuantityRange ;
    sh:sparql [
        sh:select "SELECT $this WHERE {}"
    ] .

qqval:EdgeShape a sh:NodeShape ;
    sh:targetSubjectsOf qudt:hasResult ;
    sh:property [
        sh:path qudt:hasResult ;
        sh:class qqval:QualifiedQuantityValue ;
        sh:message "hasResult must point at a QualifiedQuantityValue."
    ] .
"""


def _shapes() -> Graph:
    graph = Graph()
    graph.parse(data=SHAPES_TTL, format="turtle")
    return graph


def test_messages_render_verbatim_and_synthesis_fills_gaps() -> None:
    requirements = derive_shape_requirements(_shapes())
    by_anchor = {r.anchor: r for r in requirements}
    value = by_anchor["qqval:QualifiedQuantityValue"]
    assert (
        "QualifiedQuantityValue requires exactly one epistemicQualifier." in value.lines
    )
    # The message-less qudt:unit property gets a synthesized structural line.
    assert any("qudt:unit" in line and "at least 1" in line for line in value.lines)
    # The SPARQL constraint's message is rendered.
    assert "Exact values forbid bounds." in value.lines


def test_message_less_sparql_constraint_is_omitted() -> None:
    requirements = derive_shape_requirements(_shapes())
    anchors = [r.anchor for r in requirements]
    # QuantityRange's only constraint is an unrenderable SPARQL one.
    assert "qqval:QuantityRange" not in anchors


def test_target_subjects_of_shapes_render() -> None:
    requirements = derive_shape_requirements(_shapes())
    edge = next(r for r in requirements if "subjects of" in r.anchor)
    assert "hasResult must point at a QualifiedQuantityValue." in edge.lines


def test_chapter_caps_and_notes_truncation() -> None:
    requirements = derive_shape_requirements(_shapes())
    full = format_conformance_chapter(requirements, max_lines=60)
    assert full.startswith(CHAPTER_HEADING)
    assert "Never mint a placeholder node" in full
    capped = format_conformance_chapter(requirements, max_lines=1)
    assert "(Further rules exist" in capped


def test_empty_shapes_render_nothing() -> None:
    assert format_conformance_chapter([]) == ""
    assert derive_shape_requirements(Graph()) == []


def test_contract_terms_cover_targets_paths_and_classes() -> None:
    terms = set(contract_terms(_shapes()))
    assert "https://example.org/qqval#QualifiedQuantityValue" in terms
    assert "https://example.org/qqval#epistemicQualifier" in terms
    assert "https://example.org/qqval#EpistemicQualifier" in terms
    assert "https://example.org/qudt#hasResult" in terms


def test_contract_terms_are_exempt_from_unknown_term() -> None:
    """The exemption guardrail: the validator must never order removal of a
    term the conformance chapter required."""
    from rdflib import Literal, URIRef

    from ontocast.onto.model import FactsUnitFindingKind
    from ontocast.onto.rdfgraph import RDFGraph
    from ontocast.tool.facts_validation import (
        ValidationPolicy,
        collect_unit_findings,
    )

    subject = URIRef("https://example.com/doc/d1/value_1")
    qualifier = URIRef("https://vocab.test/qqval#epistemicQualifier")
    graph = RDFGraph()
    graph.add((subject, qualifier, URIRef("https://vocab.test/qqval#Exact")))
    graph.add(
        (
            subject,
            URIRef("http://www.w3.org/2000/01/rdf-schema#label"),
            Literal("v"),
        )
    )
    # The retrieved context declares the qqval namespace (another term of it
    # was retrieved) but NOT epistemicQualifier -- the exact situation where
    # UNKNOWN_TERM would order removal of what the chapter required.
    ontology = RDFGraph()
    ontology.add(
        (
            URIRef("https://vocab.test/qqval#QualifiedQuantityValue"),
            URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            URIRef("http://www.w3.org/2002/07/owl#Class"),
        )
    )

    def unknown_terms(policy):
        return {
            f.predicate
            for f in collect_unit_findings(
                graph=graph,
                ontology_graph=ontology,
                quarantined=[],
                extraction_text="a value",
                fact_namespaces=["https://example.com/doc/d1/"],
                policy=policy,
            )
            if f.kind == FactsUnitFindingKind.UNKNOWN_TERM
        }

    assert str(qualifier) in unknown_terms(ValidationPolicy())
    assert str(qualifier) not in unknown_terms(
        ValidationPolicy(contract_exempt_terms=(str(qualifier),))
    )


def test_per_shape_terms_are_extracted() -> None:
    requirements = derive_shape_requirements(_shapes())
    by_anchor = {r.anchor: r for r in requirements}
    value_terms = set(by_anchor["qqval:QualifiedQuantityValue"].terms)
    assert "https://example.org/qqval#QualifiedQuantityValue" in value_terms
    assert "https://example.org/qqval#epistemicQualifier" in value_terms
    assert "https://example.org/qudt#unit" in value_terms
    edge = next(r for r in requirements if "subjects of" in r.anchor)
    assert "https://example.org/qudt#hasResult" in set(edge.terms)


def test_select_requirements_joins_on_context_terms() -> None:
    from ontocast.prompt.shapes_contract import select_requirements

    requirements = derive_shape_requirements(_shapes())
    # A context carrying only the edge predicate selects only the edge shape.
    selected = select_requirements(requirements, {"https://example.org/qudt#hasResult"})
    assert [r.anchor for r in selected] == ["subjects of qudt:hasResult"]
    # A superclass IRI appearing as a snapshot *object* (schema closure)
    # joins the shape targeting it.
    selected = select_requirements(
        requirements, {"https://example.org/qqval#QualifiedQuantityValue"}
    )
    anchors = {r.anchor for r in selected}
    assert "qqval:QualifiedQuantityValue" in anchors
    # Order is preserved and the empty context selects nothing.
    assert select_requirements(requirements, set()) == []
    assert select_requirements(requirements, {"https://elsewhere/#X"}) == []


def test_shapes_catalog_selection_modes() -> None:
    """needs_selection flips on the cap; selected chapters are cached."""
    from ontocast.onto.rdfgraph import RDFGraph
    from ontocast.tool.shapes_catalog import ShapesCatalog

    catalog = ShapesCatalog()
    graph = RDFGraph()
    graph.parse(data=SHAPES_TTL, format="turtle")
    catalog._graph = graph  # materialized state, unit-test shortcut

    # 5 renderable rule lines in the fixture: under a cap of 60, over 2.
    assert catalog.needs_selection(max_lines=60) is False
    assert catalog.needs_selection(max_lines=2) is True

    full = catalog.conformance_chapter(max_lines=60)
    assert "qqval:QualifiedQuantityValue" in full

    context = {"https://example.org/qudt#hasResult"}
    selected = catalog.selected_chapter(context, max_lines=60)
    assert "hasResult must point at a QualifiedQuantityValue." in selected
    assert "epistemicQualifier" not in selected
    # Identical selection -> cached object.
    assert catalog.selected_chapter(set(context), max_lines=60) is selected
    # Empty context -> empty chapter.
    assert catalog.selected_chapter(set(), max_lines=60) == ""
