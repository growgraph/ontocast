"""Tier-1 compilation of critic fixes: what may be applied without an LLM call."""

from typing import Literal as TypingLiteral

from rdflib import Literal, URIRef

from ontocast.onto.model import TripleFix
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.triple_index import build_triple_index
from ontocast.tool.facts_validation.critic_patch import (
    apply_compiled_patch,
    compile_critic_fixes,
)

CD = "https://growgraph.dev/facts/"
MATSCI = "https://growgraph.dev/ontologies/matsci#"
RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"

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
    triple_ids: list[int] | None = None,
) -> TripleFix:
    return TripleFix(
        text_fragment="sample 1",
        action=action,
        severity="important",
        incorrect_value=incorrect,
        correct_value=correct,
        triple_ids=triple_ids or [],
        explanation="test fix",
    )


def _id_of(index, predicate_suffix: str) -> int:
    """The id of the single statement whose predicate ends with ``suffix``."""
    matches = [
        triple_id
        for triple_id, (_, predicate, _) in index.by_id.items()
        if str(predicate).endswith(predicate_suffix)
    ]
    assert len(matches) == 1, f"expected one {predicate_suffix}, got {matches}"
    return matches[0]


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


def test_an_applied_fix_stops_being_a_request() -> None:
    """Tier 1 and the repair prompt must not both act on the same fix.

    ``state.suggestions`` feeds the render prompt's improvement block, a second
    channel alongside the findings block. A fix already compiled into the graph
    left in that channel asks a repair render to fix what was just fixed --
    against a graph that no longer matches the ``incorrect_value`` it quotes,
    so the model is being asked to locate something that is gone.
    """
    from ontocast.onto.model import Suggestions

    graph = _graph()
    fixes = [
        _fix("REMOVE", incorrect='cd:sample_1 rdfs:label "sample 1" .'),
        _fix("REPLACE", incorrect="not a triple"),
    ]
    compiled = compile_critic_fixes(fixes, graph)
    assert len(compiled.applied) == 1 and len(compiled.residual) == 1

    # What the loop does with the split.
    remaining = Suggestions(actionable_fixes=list(compiled.residual))

    assert compiled.applied[0] not in remaining.actionable_fixes
    assert remaining.actionable_fixes == compiled.residual


# --- addressing statements by id ---------------------------------------------


def test_a_removal_cited_by_id_compiles_without_any_requoting() -> None:
    """The whole point: no payload to match, so nothing to get wrong."""
    graph = _graph()
    index = build_triple_index(graph)
    label_id = _id_of(index, "label")

    compiled = compile_critic_fixes(
        [_fix("REMOVE", triple_ids=[label_id])], graph, index=index
    )

    assert compiled.update is not None
    apply_compiled_patch(graph, compiled.update)
    assert not list(graph.triples((None, URIRef(f"{RDFS_NS}label"), None)))
    assert len(compiled.applied) == 1


def test_an_id_beats_a_misquoted_incorrect_value() -> None:
    """A fix may carry both; the id is the one that cannot be wrong."""
    graph = _graph()
    index = build_triple_index(graph)
    label_id = _id_of(index, "label")
    fix = _fix(
        "REMOVE",
        incorrect='cd:sample_1 rdfs:label "totally the wrong text" .',
        triple_ids=[label_id],
    )

    compiled = compile_critic_fixes([fix], graph, index=index)

    assert compiled.residual == []
    assert compiled.update is not None


def test_an_id_the_index_never_issued_sends_the_whole_fix_back() -> None:
    """Acting on the half that resolved would delete something unreviewed."""
    graph = _graph()
    index = build_triple_index(graph)
    label_id = _id_of(index, "label")

    compiled = compile_critic_fixes(
        [_fix("REMOVE", triple_ids=[label_id, len(index) + 99])], graph, index=index
    )

    assert compiled.update is None
    assert len(compiled.residual) == 1
    assert compiled.bad_index_refs == 1


def test_a_statement_already_gone_is_not_deleted_again() -> None:
    graph = _graph()
    index = build_triple_index(graph)
    label_id = _id_of(index, "label")
    graph.remove(index.by_id[label_id])

    compiled = compile_critic_fixes(
        [_fix("REMOVE", triple_ids=[label_id])], graph, index=index
    )

    assert compiled.update is None
    assert len(compiled.residual) == 1


def test_a_replace_by_id_swaps_the_statement() -> None:
    graph = _graph()
    index = build_triple_index(graph)
    label_id = _id_of(index, "label")
    fix = _fix(
        "REPLACE",
        correct=f'<{CD}sample_1> <{RDFS_NS}label> "corrected sample 1" .',
        triple_ids=[label_id],
    )

    compiled = compile_critic_fixes([fix], graph, index=index)
    assert compiled.update is not None
    apply_compiled_patch(graph, compiled.update)

    labels = {
        str(o) for _, _, o in graph.triples((None, URIRef(f"{RDFS_NS}label"), None))
    }
    assert labels == {"corrected sample 1"}


def test_a_fix_that_re_adds_exactly_what_it_removes_is_recorded_as_no_change() -> None:
    """Measured on real critiques, a third of applicable REPLACEs are this.

    They arrive as a reordering of the same statement. Counting them as fixes
    that landed overstates what the critic bought; returning them as residual
    would ask the next pass to redo nothing.
    """
    graph = _graph()
    index = build_triple_index(graph)
    label_id = _id_of(index, "label")
    fix = _fix(
        "REPLACE",
        correct=f'<{CD}sample_1> <{RDFS_NS}label> "sample 1" .',
        triple_ids=[label_id],
    )

    compiled = compile_critic_fixes([fix], graph, index=index)

    assert compiled.noop == [fix]
    assert compiled.applied == [] and compiled.residual == []
    assert compiled.update is None


def test_no_fix_is_lost_between_the_three_outcomes() -> None:
    graph = _graph()
    index = build_triple_index(graph)
    label_id = _id_of(index, "label")
    fixes = [
        _fix("REMOVE", triple_ids=[label_id]),
        _fix("ADD", correct=f'<{CD}sample_1> <{MATSCI}note> "extra" .'),
        _fix("REMOVE", incorrect="not a triple at all"),
        _fix(
            "REPLACE",
            correct=f'<{CD}sample_1> <{RDFS_NS}label> "sample 1" .',
            triple_ids=[label_id],
        ),
    ]

    compiled = compile_critic_fixes(fixes, graph, index=index)

    accounted = compiled.applied + compiled.residual + compiled.noop
    assert sorted(map(id, accounted)) == sorted(map(id, fixes))


def test_without_an_index_the_requoting_path_still_works() -> None:
    """Cached critiques and any caller that cannot build an index keep working."""
    graph = _graph()
    compiled = compile_critic_fixes(
        [_fix("REMOVE", incorrect=f'<{CD}sample_1> <{RDFS_NS}label> "sample 1" .')],
        graph,
    )
    assert compiled.update is not None


# --- format-tolerant payload parsing -----------------------------------------
#
# The deployment names one syntax for fix payloads and the model does not
# reliably use it, so the parser accepts what is actually emitted rather than
# what was asked for. None of these rules guesses at meaning: each maps a
# JSON-LD shape onto the Turtle form that means the same thing.


def test_a_jsonld_language_object_inside_turtle_is_accepted() -> None:
    """The single most common malformed payload in real critic output."""
    graph = _graph()
    fix = _fix(
        "ADD",
        correct=f'<{CD}sample_1> <{RDFS_NS}comment> {{"@value": "a note", "@language": "en"}} .',
    )

    compiled = compile_critic_fixes([fix], graph)

    assert compiled.update is not None
    apply_compiled_patch(graph, compiled.update)
    assert (
        URIRef(f"{CD}sample_1"),
        URIRef(f"{RDFS_NS}comment"),
        Literal("a note", lang="en"),
    ) in graph


def test_a_jsonld_typed_value_inside_turtle_is_accepted() -> None:
    graph = _graph()
    fix = _fix(
        "ADD",
        correct=(
            f"<{CD}sample_1> <{MATSCI}measuredOn> "
            '{"@value": "2024-01-15", "@type": "<http://www.w3.org/2001/XMLSchema#date>"} .'
        ),
    )

    compiled = compile_critic_fixes([fix], graph)

    assert compiled.update is not None


def test_a_bare_absolute_iri_is_bracketed() -> None:
    """Turtle needs angle brackets; models routinely omit them."""
    graph = _graph()
    fix = _fix("ADD", correct=f"<{CD}sample_1> <{MATSCI}seeAlso> {MATSCI}Other ;")

    compiled = compile_critic_fixes([fix], graph)

    assert compiled.update is not None
    apply_compiled_patch(graph, compiled.update)
    assert (
        URIRef(f"{CD}sample_1"),
        URIRef(f"{MATSCI}seeAlso"),
        URIRef(f"{MATSCI}Other"),
    ) in graph


def test_an_iri_inside_a_quoted_literal_is_left_alone() -> None:
    """Bracketing must not reach inside a literal and corrupt its lexical form."""
    graph = _graph()
    fix = _fix(
        "ADD",
        correct=f'<{CD}sample_1> <{RDFS_NS}comment> "see https://example.org/x" .',
    )

    compiled = compile_critic_fixes([fix], graph)

    assert compiled.update is not None
    apply_compiled_patch(graph, compiled.update)
    assert (
        URIRef(f"{CD}sample_1"),
        URIRef(f"{RDFS_NS}comment"),
        Literal("see https://example.org/x"),
    ) in graph


def test_a_json_payload_carrying_no_statement_stays_residual() -> None:
    """A bare literal or a node reference is not a patch, whatever it meant."""
    graph = _graph()
    for payload in (
        '{"@value": "2023-10-23", "@type": "xsd:date"}',
        '{"@id": "cd:dallas"}',
    ):
        compiled = compile_critic_fixes([_fix("ADD", correct=payload)], graph)
        assert compiled.update is None, payload
        assert len(compiled.residual) == 1
