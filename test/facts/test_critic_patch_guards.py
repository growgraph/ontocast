"""What a critic pass is not allowed to destroy.

A compiled patch is transparent -- both halves are known before anything is
touched -- so these limits are enforced by withholding the delete half, not by
inspecting the damage afterwards. Every rule drops something and counts it.
"""

from typing import Literal as TypingLiteral

import pytest
from rdflib import RDF, RDFS, BNode, Literal, Namespace, URIRef

from ontocast.onto.model import TripleFix
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.triple_index import build_triple_index
from ontocast.tool.facts_validation.critic_patch import (
    CriticPatchPolicy,
    compile_critic_fixes,
)

pytestmark = pytest.mark.unit

CD = Namespace("https://growgraph.dev/facts/")
MS = Namespace("https://growgraph.dev/ontologies/matsci#")


def _graph(extra: int = 0) -> RDFGraph:
    graph = RDFGraph()
    graph.bind("cd", CD)
    graph.bind("ms", MS)
    graph.add((CD.sample_1, RDF.type, MS.NanocrystalSample))
    graph.add((CD.sample_1, RDFS.label, Literal("sample 1")))
    graph.add((CD.sample_1, MS.hasAmount, CD.amount_1))
    graph.add((CD.amount_1, RDF.type, MS.Amount))
    graph.add((CD.amount_1, MS.numericValue, Literal("1.5")))
    for n in range(extra):
        graph.add((CD.sample_1, URIRef(f"{MS}note{n}"), Literal(f"n{n}")))
    return graph


def _fix(
    action: TypingLiteral["ADD", "REMOVE", "REPLACE"],
    *,
    triple_ids=None,
    correct: str = "",
) -> TripleFix:
    return TripleFix(
        text_fragment="sample 1",
        action=action,
        severity="important",
        triple_ids=triple_ids or [],
        correct_value=correct,
        explanation="test fix",
    )


def _ids_for(index, subject) -> list[int]:
    return [tid for tid, (s, _, _) in index.by_id.items() if s == subject]


def test_a_removal_that_would_empty_a_subject_is_refused() -> None:
    """The contract is to correct a statement, not to make a node disappear."""
    graph = _graph()
    index = build_triple_index(graph)
    amount_ids = _ids_for(index, CD.amount_1)

    compiled = compile_critic_fixes(
        [_fix("REMOVE", triple_ids=amount_ids)], graph, index=index
    )

    assert compiled.update is None
    assert compiled.deletes_refused == 1
    assert len(compiled.residual) == 1


def test_a_removal_leaving_the_subject_described_is_allowed() -> None:
    graph = _graph()
    index = build_triple_index(graph)
    label_id = _ids_for(index, CD.sample_1)
    label_id = [tid for tid in label_id if index.by_id[tid][1] == RDFS.label]

    compiled = compile_critic_fixes(
        [_fix("REMOVE", triple_ids=label_id)], graph, index=index
    )

    assert compiled.update is not None
    assert compiled.deletes_refused == 0


def test_a_replace_that_writes_about_a_different_subject_keeps_only_its_insert() -> (
    None
):
    """A cross-subject REPLACE is a rename.

    Carried out literally it orphans the old node and leaves the new one with
    nothing but the one statement the fix happened to write.
    """
    graph = _graph()
    index = build_triple_index(graph)
    label_id = [
        tid
        for tid, (s, p, _) in index.by_id.items()
        if s == CD.sample_1 and p == RDFS.label
    ]

    compiled = compile_critic_fixes(
        [
            _fix(
                "REPLACE",
                triple_ids=label_id,
                correct=(
                    f"<{CD}sample_2> <{RDF.type}> <{MS}NanocrystalSample> ; "
                    f'<{RDFS.label}> "sample two" .'
                ),
            )
        ],
        graph,
        index=index,
    )

    assert compiled.deletes_refused == 1
    assert compiled.update is not None
    assert all(op.type == "insert" for op in compiled.update.triple_operations)
    assert (CD.sample_1, RDFS.label, Literal("sample 1")) in graph


def test_a_rename_is_allowed_when_the_deployment_says_so() -> None:
    graph = _graph()
    index = build_triple_index(graph)
    label_id = [
        tid
        for tid, (s, p, _) in index.by_id.items()
        if s == CD.sample_1 and p == RDFS.label
    ]

    compiled = compile_critic_fixes(
        [
            _fix(
                "REPLACE",
                triple_ids=label_id,
                correct=(
                    f"<{CD}sample_2> <{RDF.type}> <{MS}NanocrystalSample> ; "
                    f'<{RDFS.label}> "sample two" .'
                ),
            )
        ],
        graph,
        index=index,
        policy=CriticPatchPolicy(allow_subject_rename=True),
    )

    assert compiled.deletes_refused == 0
    assert compiled.update is not None
    assert any(op.type == "delete" for op in compiled.update.triple_operations)


def test_deleting_part_of_a_blank_node_takes_the_whole_node() -> None:
    """A partial delete leaves the degenerate stub the validator exists to flag."""
    graph = _graph()
    restriction = BNode("r1")
    graph.add((CD.sample_1, RDFS.subClassOf, restriction))
    graph.add((restriction, RDF.type, MS.Restriction))
    graph.add((restriction, MS.onProperty, MS.hasAmount))
    index = build_triple_index(graph)
    one_of_them = [
        tid
        for tid, (s, p, _) in index.by_id.items()
        if s == restriction and p == MS.onProperty
    ]

    compiled = compile_critic_fixes(
        [_fix("REMOVE", triple_ids=one_of_them)], graph, index=index
    )

    assert compiled.update is not None
    deleted = {
        triple
        for op in compiled.update.triple_operations
        if op.type == "delete"
        for triple in op.graph
    }
    assert (restriction, RDF.type, MS.Restriction) in deleted
    assert (CD.sample_1, RDFS.subClassOf, restriction) in deleted


def test_a_pass_that_wants_to_remove_most_of_the_graph_sends_its_removals_back() -> (
    None
):
    """Past the cap the critique has stopped correcting and started rewriting.

    The removals go back whole rather than being stripped to their insert
    halves: a REPLACE without its delete is an ADD of the new value beside
    the old one, which is not the edit that was proposed. Pure additions
    still land.
    """
    graph = _graph(extra=20)
    index = build_triple_index(graph)
    doomed = [
        tid
        for tid, (s, p, _) in index.by_id.items()
        if s == CD.sample_1 and p != RDF.type
    ]
    removals = [_fix("REMOVE", triple_ids=[tid]) for tid in doomed]
    replacement = _fix(
        "REPLACE",
        triple_ids=[doomed[0]],
        correct=f'<{CD}sample_1> <{RDFS.label}> "renamed" .',
    )
    addition = _fix("ADD", correct=f'<{CD}sample_1> <{MS}note> "kept" .')

    compiled = compile_critic_fixes(
        [*removals, replacement, addition],
        graph,
        index=index,
        policy=CriticPatchPolicy(max_delete_share=0.25, min_deletes=2),
    )

    assert compiled.delete_capped is True
    assert compiled.applied == [addition]
    assert set(map(id, compiled.residual)) == set(map(id, [*removals, replacement]))
    assert compiled.update is not None
    assert all(op.type == "insert" for op in compiled.update.triple_operations)
    assert [patch.fix for patch in compiled.patches] == [addition]


def test_the_cap_has_a_floor_so_a_short_unit_stays_correctable() -> None:
    """Share alone is strictest where it should be loosest.

    On a five-statement unit a single legitimate correction is already 20% of
    the graph, so a bare share cap would block exactly the small, safe fixes.
    """
    graph = _graph()
    index = build_triple_index(graph)
    label_id = [
        tid
        for tid, (s, p, _) in index.by_id.items()
        if s == CD.sample_1 and p == RDFS.label
    ]

    compiled = compile_critic_fixes(
        [_fix("REMOVE", triple_ids=label_id)],
        graph,
        index=index,
        policy=CriticPatchPolicy(max_delete_share=0.01, min_deletes=5),
    )

    assert compiled.delete_capped is False
    assert compiled.update is not None
