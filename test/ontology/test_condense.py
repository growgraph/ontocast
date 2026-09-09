"""The prompt budget must bind without removing the schema the model needs.

Condensing is the only thing bounding ontology context in
``selected_single_ontology`` and ``fixed_single_ontology``, so the interesting
assertions are not "it got smaller" but "the parts a model cannot work without
survived". A condenser that hits its number by dropping ``rdfs:label`` or
``rdfs:domain`` produces an extraction failure that reads as a bad model.
"""

import logging

import pytest
from rdflib import Literal, URIRef
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS

from ontocast.config.settings import ServerConfig
from ontocast.onto.ontology_condense import (
    GLOSS_PREDICATES,
    LOAD_BEARING_PREDICATES,
    condense_graph_for_prompt,
)
from ontocast.onto.rdfgraph import RDFGraph

pytestmark = pytest.mark.unit

EX = "https://example.org/onto#"


def _term(name: str) -> URIRef:
    return URIRef(f"{EX}{name}")


def _graph(n_classes: int = 20) -> RDFGraph:
    """A schema with labels, hierarchy, property links, glosses and header noise."""
    graph = RDFGraph()
    for i in range(n_classes):
        cls = _term(f"Class{i}")
        graph.add((cls, RDF.type, OWL.Class))
        graph.add((cls, RDFS.label, Literal(f"class {i}")))
        graph.add((cls, RDFS.comment, Literal(f"A long description of class {i}.")))
        graph.add((cls, SKOS.definition, Literal(f"Definition of class {i}.")))
        if i:
            graph.add((cls, RDFS.subClassOf, _term(f"Class{i - 1}")))
        prop = _term(f"prop{i}")
        graph.add((prop, RDF.type, OWL.ObjectProperty))
        graph.add((prop, RDFS.label, Literal(f"prop {i}")))
        graph.add((prop, RDFS.domain, cls))
        graph.add((prop, RDFS.range, _term(f"Class{(i + 1) % n_classes}")))
    onto = _term("")
    graph.add((onto, OWL.versionInfo, Literal("1.0.0")))
    graph.add((onto, DCTERMS.creator, Literal("someone")))
    graph.add((onto, DCTERMS.license, Literal("CC0")))
    return graph


def _predicate_counts(graph: RDFGraph, predicates) -> dict:
    return {p: sum(1 for _ in graph.triples((None, p, None))) for p in predicates}


def test_no_budget_is_a_no_op() -> None:
    graph = _graph()
    out, report = condense_graph_for_prompt(graph, None)

    assert out is graph
    assert not report.changed
    assert report.triples_before == report.triples_after


def test_graph_already_under_budget_is_returned_untouched() -> None:
    graph = _graph()
    out, report = condense_graph_for_prompt(graph, len(graph) + 1)

    assert out is graph, "no copy should be made when nothing needs doing"
    assert not report.changed
    assert not report.over_budget


def test_noise_goes_before_anything_load_bearing() -> None:
    graph = _graph()
    # A budget one below the graph size: only the cheapest pass should fire.
    out, report = condense_graph_for_prompt(graph, len(graph) - 1)

    assert report.dropped_noise >= 3  # versionInfo, creator, license
    assert report.dropped_glosses == 0, "glosses are a later resort than header noise"
    assert not report.over_budget
    for predicate in (OWL.versionInfo, DCTERMS.creator, DCTERMS.license):
        assert not list(out.triples((None, predicate, None)))


def test_glosses_are_dropped_but_structure_survives() -> None:
    graph = _graph()
    before = _predicate_counts(graph, LOAD_BEARING_PREDICATES)

    # Tight enough that noise and structural passes cannot get there alone.
    out, report = condense_graph_for_prompt(graph, len(graph) // 2)

    assert report.dropped_glosses > 0
    assert len(out) < len(graph)
    for predicate in GLOSS_PREDICATES:
        assert not list(out.triples((None, predicate, None))), (
            f"{predicate} should have been dropped"
        )

    after = _predicate_counts(out, LOAD_BEARING_PREDICATES)
    for predicate in (RDFS.label, RDFS.subClassOf, RDFS.domain, RDFS.range):
        assert after[predicate] == before[predicate], (
            f"{predicate} is load-bearing and must survive condensing"
        )


def test_unreachable_budget_warns_and_passes_through_rather_than_gutting() -> None:
    graph = _graph()
    before = _predicate_counts(graph, LOAD_BEARING_PREDICATES)

    out, report = condense_graph_for_prompt(graph, 5)

    assert report.over_budget
    assert report.triples_after > 5, "the graph is passed through, not truncated"
    after = _predicate_counts(out, LOAD_BEARING_PREDICATES)
    for predicate in (RDFS.label, RDFS.subClassOf, RDFS.domain, RDFS.range):
        assert after[predicate] == before[predicate]


def test_unreachable_budget_says_what_to_do_about_it(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="ontocast.onto.ontology_condense"):
        condense_graph_for_prompt(_graph(), 5)

    message = caplog.text
    assert "exceeds the prompt budget" in message
    # The warning has to name the way out, or it is just noise in a log.
    assert "selected_vector_search_ontology" in message


def test_input_graph_is_never_mutated() -> None:
    graph = _graph()
    original = set(graph)

    condense_graph_for_prompt(graph, len(graph) // 2)

    assert set(graph) == original


def test_report_counts_reconcile_with_the_triples_removed() -> None:
    graph = _graph()
    _, report = condense_graph_for_prompt(graph, len(graph) // 2)

    removed = report.triples_before - report.triples_after
    assert removed == (
        report.dropped_noise + report.dropped_structural + report.dropped_glosses
    )


def test_declared_defaults() -> None:
    """The write-path backstop is off; the prompt budget is on."""
    fields = ServerConfig.model_fields
    assert fields["ontology_max_triples"].default is None
    assert fields["ontology_context_max_triples"].default == 4000
