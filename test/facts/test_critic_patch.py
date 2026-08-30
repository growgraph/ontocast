"""Tier-1 compilation of critic fixes: what may be applied without an LLM call."""

from typing import Literal as TypingLiteral

from rdflib import Literal, URIRef

from ontocast.onto.model import TripleFix
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.facts_validation.critic_patch import (
    apply_compiled_patch,
    compile_critic_fixes,
)

CD = "https://growgraph.dev/facts/"
MATSCI = "https://growgraph.dev/ontologies/matsci#"

_TTL = f"""
@prefix cd: <{CD}> .
@prefix matsci: <{MATSCI}> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

cd:sample_1 a matsci:NanocrystalSample ;
    rdfs:label "sample 1" ;
    matsci:hasAmount cd:amount_1 .
"""


def _graph() -> RDFGraph:
    graph = RDFGraph()
    graph.parse(data=_TTL, format="turtle")
    return graph


def _fix(
    action: TypingLiteral["ADD", "REMOVE", "REPLACE"],
    incorrect: str = "",
    correct: str = "",
) -> TripleFix:
    return TripleFix(
        text_fragment="sample 1",
        action=action,
        severity="important",
        incorrect_value=incorrect,
        correct_value=correct,
        explanation="test fix",
    )


def test_remove_matching_a_present_triple_compiles() -> None:
    graph = _graph()
    compiled = compile_critic_fixes(
        [_fix("REMOVE", incorrect="cd:sample_1 matsci:hasAmount cd:amount_1 .")],
        graph,
    )

    assert compiled.applied and not compiled.residual
    assert compiled.update is not None
    apply_compiled_patch(graph, compiled.update)
    assert (
        URIRef(f"{CD}sample_1"),
        URIRef(f"{MATSCI}hasAmount"),
        URIRef(f"{CD}amount_1"),
    ) not in graph
    assert (URIRef(f"{CD}sample_1"), None, None) in graph, "only the quoted triple goes"


def test_remove_quoting_a_triple_the_graph_does_not_hold_is_residual() -> None:
    """A misquoted fix has misunderstood the graph; acting on it deletes blind."""
    graph = _graph()
    compiled = compile_critic_fixes(
        [_fix("REMOVE", incorrect="cd:sample_1 matsci:hasAmount cd:amount_99 .")],
        graph,
    )

    assert compiled.update is None
    assert compiled.residual and not compiled.applied
    assert len(graph) == len(_graph()), "nothing may be deleted on a miss"


def test_add_compiles_and_skips_triples_already_present() -> None:
    graph = _graph()
    compiled = compile_critic_fixes(
        [
            _fix("ADD", correct='cd:sample_1 rdfs:comment "grown in hexane" .'),
            _fix("ADD", correct='cd:sample_1 rdfs:label "sample 1" .'),
        ],
        graph,
    )

    assert len(compiled.applied) == 1
    assert len(compiled.residual) == 1, "an ADD that adds nothing is not applied"
    assert compiled.update is not None
    apply_compiled_patch(graph, compiled.update)
    assert (
        URIRef(f"{CD}sample_1"),
        URIRef("http://www.w3.org/2000/01/rdf-schema#comment"),
        Literal("grown in hexane"),
    ) in graph


def test_replace_needs_both_sides_to_resolve() -> None:
    graph = _graph()
    compiled = compile_critic_fixes(
        [
            _fix(
                "REPLACE",
                incorrect='cd:sample_1 rdfs:label "sample 1" .',
                correct='cd:sample_1 rdfs:label "CsPbBr3 sample 1" .',
            ),
            _fix("REPLACE", incorrect='cd:sample_1 rdfs:label "sample 1" .'),
        ],
        graph,
    )

    assert len(compiled.applied) == 1
    assert len(compiled.residual) == 1, "a REPLACE with no replacement is residual"
    assert compiled.update is not None
    apply_compiled_patch(graph, compiled.update)
    labels = set(graph.objects(URIRef(f"{CD}sample_1"), None))
    assert Literal("CsPbBr3 sample 1") in labels
    assert Literal("sample 1") not in labels


def test_unparseable_payloads_fall_through_rather_than_raising() -> None:
    """Real critic output is frequently truncated mid-payload."""
    compiled = compile_critic_fixes(
        [
            _fix("REMOVE", incorrect='{"@context": {"cd": "https://growgraph'),
            _fix("ADD", correct="cd:sample_1 matsci:hasAmount"),
            _fix("ADD", correct=""),
        ],
        _graph(),
    )

    assert compiled.update is None
    assert len(compiled.residual) == 3


def test_a_truncated_turtle_fragment_still_compiles() -> None:
    """Fragments lifted from a predicate list keep their trailing separator."""
    compiled = compile_critic_fixes(
        [_fix("REMOVE", incorrect='cd:sample_1 rdfs:label "sample 1" ;')],
        _graph(),
    )

    assert compiled.applied and compiled.update is not None


def test_fixes_are_never_silently_dropped() -> None:
    """Every fix lands in exactly one of applied/residual -- the whole point."""
    fixes = [
        _fix("REMOVE", incorrect='cd:sample_1 rdfs:label "sample 1" .'),
        _fix("ADD", correct='cd:sample_1 rdfs:comment "note" .'),
        _fix("REPLACE", incorrect="garbage"),
        _fix("ADD", correct="also garbage {"),
    ]
    compiled = compile_critic_fixes(fixes, _graph())

    assert len(compiled.applied) + len(compiled.residual) == len(fixes)
