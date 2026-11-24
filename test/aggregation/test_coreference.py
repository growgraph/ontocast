"""Coreference resolution: the pairs put to the model, and what verdicts do."""

from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import URIRef

from ontocast.config import AggregationConfig
from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.content_unit import ContentUnit, OutputType
from ontocast.onto.enum import RetrievalMetric
from ontocast.onto.model import CoreferenceCandidate
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.stategraph import node_factories
from ontocast.stategraph.node_factories import (
    make_merge_facts_node,
    make_resolve_coreference_node,
)
from ontocast.tool import EmbeddingBasedAggregator
from ontocast.tool.agg.coreference import (
    coreference_candidates,
    describe_entity,
    verdict_pairs,
)
from ontocast.toolbox import ToolBox

pytestmark = pytest.mark.unit

CD = f"{DEFAULT_IRI}/"
Q = "https://x.org/schema#"


def _unit(index: int, ttl: str, text: str = "") -> ContentUnit:
    graph = RDFGraph()
    graph.parse(
        data=(
            f"@prefix cd: <{CD}> .\n@prefix q: <{Q}> .\n"
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n" + ttl
        ),
        format="turtle",
    )
    return ContentUnit(
        text=text or f"unit {index}",
        index=index,
        doc_iri=URIRef("https://x.org/doc/1"),
        graph=graph,
        type=OutputType.FACTS,
    )


def _applicant_units(*extra: str) -> list[ContentUnit]:
    return [
        _unit(0, "cd:applicant_1__u0 a q:Person ; q:applicantIn cd:case__u0 ."),
        _unit(
            1,
            'cd:rusi__u1 a q:Person ; rdfs:label "Rusi Stanev" ;'
            " q:applicantIn cd:case__u1 .",
        ),
        *(_unit(2 + i, ttl) for i, ttl in enumerate(extra)),
    ]


_CASE = {CD + f"case__u{i}": CD + "Case" for i in range(4)}


def test_shared_link_pairs_the_only_two_entities_holding_a_link() -> None:
    """A role phrase and a name for one referent: no name or embedding evidence."""
    candidates = coreference_candidates(
        lexical_near_misses=[],
        units=_applicant_units("cd:court__u2 a q:Court ; q:heardIn cd:case__u2 ."),
        final_uri=_CASE,
        max_pairs=10,
    )

    assert candidates == [
        CoreferenceCandidate(
            left=CD + "applicant_1__u0", right=CD + "rusi__u1", kind="shared_link"
        )
    ]


def test_a_link_many_entities_hold_is_no_evidence() -> None:
    candidates = coreference_candidates(
        lexical_near_misses=[],
        units=_applicant_units("cd:other__u2 a q:Person ; q:applicantIn cd:case__u2 ."),
        final_uri=_CASE,
        max_pairs=10,
    )

    assert candidates == []


def test_candidate_kinds_take_turns_under_the_cap() -> None:
    near = [
        (CD + f"n{i}__u0", CD + f"n{i}__u1", 0.5 + i / 10, "lexical") for i in range(3)
    ]
    catalog = [(CD + "x__u0", Q + "Term", 0.99, "overlap")]

    candidates = coreference_candidates(
        lexical_near_misses=catalog + near,
        units=_applicant_units("cd:court__u2 a q:Court ; q:heardIn cd:case__u2 ."),
        final_uri=_CASE,
        max_pairs=3,
    )

    assert [(c.kind, c.left) for c in candidates] == [
        ("near_miss", CD + "n2__u0"),  # most similar near miss first
        ("shared_link", CD + "applicant_1__u0"),
        ("near_miss", CD + "n1__u0"),
    ]


def test_overlap_pairs_are_asked_first_and_the_cap_holds() -> None:
    """Pairs named alike but not equally come before every other kind."""
    near = [
        (CD + f"n{i}__u0", CD + f"n{i}__u1", 0.9 - i / 10, "lexical") for i in range(2)
    ]
    overlap = [
        (CD + f"o{i}__u0", CD + f"o{i}__u1", 0.5 + i / 10, "overlap") for i in range(3)
    ]

    candidates = coreference_candidates(
        lexical_near_misses=near + overlap,
        units=_applicant_units("cd:court__u2 a q:Court ; q:heardIn cd:case__u2 ."),
        final_uri=_CASE,
        max_pairs=5,
    )

    assert [(c.kind, c.left) for c in candidates] == [
        ("overlap", CD + "o2__u0"),  # most similar overlap first
        ("overlap", CD + "o1__u0"),
        ("overlap", CD + "o0__u0"),
        ("near_miss", CD + "n0__u0"),
        ("shared_link", CD + "applicant_1__u0"),
    ]
    assert (
        len(
            coreference_candidates(
                lexical_near_misses=near + overlap,
                units=_applicant_units(),
                final_uri=_CASE,
                max_pairs=2,
            )
        )
        == 2
    )


def test_the_default_cap_is_eighty_pairs() -> None:
    assert AggregationConfig.model_fields["llm_coreference_max_pairs"].default == 80


def test_verdicts_become_vetoes_and_confirmations() -> None:
    candidates = [
        CoreferenceCandidate(left="a", right="b", kind="near_miss"),
        CoreferenceCandidate(left="c", right="d", kind="near_miss"),
        CoreferenceCandidate(left="e", right="f", kind="shared_link"),
    ]
    vetoes, confirmations = verdict_pairs(
        candidates, {1: "same", 2: "different", 3: "unsure"}
    )
    assert confirmations == {frozenset((URIRef("a"), URIRef("b")))}
    assert vetoes == {frozenset((URIRef("c"), URIRef("d")))}


def test_an_entity_is_described_as_its_unit_stated_it() -> None:
    units = [
        _unit(
            3,
            'cd:peak__u3 a q:Observation ; rdfs:label "daily maximum temperature" ;'
            ' q:value "46.6" ; q:at cd:lytton__u3 .\n'
            'cd:lytton__u3 rdfs:label "Lytton" .',
            text="The record fell. The daily maximum temperature reached 46.6 °C.",
        )
    ]
    description = describe_entity(CD + "peak__u3", units)

    assert 'labels: "daily maximum temperature"' in description
    assert "types: Observation" in description
    assert "value = 46.6" in description
    assert "at -> Lytton" in description
    assert 'source: "The daily maximum temperature reached 46.6 °C."' in description


def test_an_entity_is_never_shown_by_its_iri() -> None:
    """Two units can mint one local name for unrelated things; never evidence."""
    units = [
        _unit(3, 'cd:weather_observations__u3 rdfs:label "weather observations" .'),
    ]
    description = describe_entity(CD + "weather_observations__u3", units)

    assert description.startswith('labels: "weather observations"')
    assert "weather_observations" not in description
    assert describe_entity(CD + "unknown", units) == "(not described)"


# --- the RESOLVE_COREFERENCE node --------------------------------------------


def _one_cluster_aggregator(monkeypatch) -> EmbeddingBasedAggregator:
    aggregator = EmbeddingBasedAggregator(AggregationConfig())

    def one_cluster(representations):
        return [list(representations)], {}

    monkeypatch.setattr(aggregator.clusterer, "cluster_entities", one_cluster)
    return aggregator


def _state(units: list[ContentUnit]) -> AgentState:
    state = AgentState()
    state.current_domain = "https://x.org"
    state.doc_hid = "1"
    state.facts_units = units
    return state


def _subjects(state: AgentState) -> set[str]:
    """The entities served, one per labelled subject."""
    label = URIRef("http://www.w3.org/2000/01/rdf-schema#label")
    return {str(s) for s in state.aggregated_facts.subjects(label, None)}


def _judge(verdicts: dict[int, str]):
    async def judge(state, tools, candidates):
        return verdicts

    return judge


def _overlapping_names() -> list[ContentUnit]:
    """One name inside the other: identity is a question, not evidence."""
    return [
        _unit(0, 'cd:batch a q:Sample ; rdfs:label "sample batch" .'),
        _unit(1, 'cd:large_batch a q:Sample ; rdfs:label "large sample batch" .'),
    ]


def _tools(aggregator: EmbeddingBasedAggregator) -> ToolBox:
    return cast(
        ToolBox,
        SimpleNamespace(
            aggregator=aggregator,
            ontology_manager=SimpleNamespace(get_head_by_iri=lambda iri: None),
        ),
    )


@pytest.mark.anyio
async def test_an_overlapping_name_merges_only_on_a_same_verdict(monkeypatch) -> None:
    tools = _tools(_one_cluster_aggregator(monkeypatch))
    state = _state(_overlapping_names())
    make_merge_facts_node(tools)(state)
    assert len(_subjects(state)) == 2
    assert [c.kind for c in state.coreference_candidates] == ["overlap"]

    monkeypatch.setattr(node_factories, "judge_coreference", _judge({1: "same"}))
    await make_resolve_coreference_node(tools)(state)

    assert len(_subjects(state)) == 1
    assert len(state.coreference_confirmations) == 1


@pytest.mark.anyio
async def test_a_same_verdict_not_repeated_alone_neither_merges_nor_vetoes(
    monkeypatch,
) -> None:
    tools = _tools(_one_cluster_aggregator(monkeypatch))
    state = _state(_overlapping_names())
    make_merge_facts_node(tools)(state)
    calls: list[int] = []

    async def judge(state, tools, candidates):
        calls.append(len(candidates))
        return {1: "same"} if len(calls) == 1 else {1: "different"}

    monkeypatch.setattr(node_factories, "judge_coreference", judge)
    await make_resolve_coreference_node(tools)(state)

    assert calls == [1, 1]
    assert len(_subjects(state)) == 2
    assert not state.coreference_confirmations
    assert not state.coreference_vetoes
    metrics = state.retrieval_metrics
    assert metrics[RetrievalMetric.FACTS_COREFERENCE_SAME] == 1
    assert metrics[RetrievalMetric.FACTS_COREFERENCE_RETRACTED] == 1


@pytest.mark.anyio
async def test_only_same_verdicts_are_asked_again_one_pair_per_call(
    monkeypatch,
) -> None:
    tools = _tools(_one_cluster_aggregator(monkeypatch))
    state = _state(_overlapping_names())
    make_merge_facts_node(tools)(state)
    pair = state.coreference_candidates[0]
    state.coreference_candidates = [pair, pair.model_copy(), pair.model_copy()]
    asked: list[int] = []

    async def judge(state, tools, candidates):
        asked.append(len(candidates))
        return (
            {1: "same", 2: "different", 3: "unsure"} if len(asked) == 1 else {1: "same"}
        )

    monkeypatch.setattr(node_factories, "judge_coreference", judge)
    await make_resolve_coreference_node(tools)(state)

    assert asked == [3, 1]
    assert state.retrieval_metrics[RetrievalMetric.FACTS_COREFERENCE_RETRACTED] == 0
    assert len(state.coreference_confirmations) == 1


@pytest.mark.anyio
async def test_a_different_verdict_keeps_an_overlapping_name_apart(monkeypatch) -> None:
    tools = _tools(_one_cluster_aggregator(monkeypatch))
    state = _state(_overlapping_names())
    make_merge_facts_node(tools)(state)

    monkeypatch.setattr(node_factories, "judge_coreference", _judge({1: "different"}))
    await make_resolve_coreference_node(tools)(state)

    assert len(_subjects(state)) == 2
    assert len(state.coreference_vetoes) == 1
    metrics = state.retrieval_metrics
    assert metrics[RetrievalMetric.FACTS_COREFERENCE_PAIRS] == 1
    assert metrics[RetrievalMetric.FACTS_COREFERENCE_DIFFERENT] == 1


@pytest.mark.anyio
async def test_a_same_verdict_merges_what_wording_kept_apart(monkeypatch) -> None:
    aggregator = _one_cluster_aggregator(monkeypatch)
    tools = cast(
        ToolBox,
        SimpleNamespace(
            aggregator=aggregator,
            ontology_manager=SimpleNamespace(get_head_by_iri=lambda iri: None),
        ),
    )
    state = _state(
        [
            _unit(0, 'cd:applicant a q:Person ; rdfs:label "the first applicant" .'),
            _unit(1, 'cd:rusi a q:Person ; rdfs:label "Rusi Kosev Stanev" .'),
        ]
    )
    make_merge_facts_node(tools)(state)
    assert len(_subjects(state)) == 2
    assert [c.kind for c in state.coreference_candidates] == ["near_miss"]

    monkeypatch.setattr(node_factories, "judge_coreference", _judge({1: "same"}))
    await make_resolve_coreference_node(tools)(state)

    assert len(_subjects(state)) == 1
    assert len(state.coreference_confirmations) == 1


@pytest.mark.anyio
async def test_a_same_verdict_never_overrides_a_guard(monkeypatch) -> None:
    """Two entities one unit minted stay apart whatever the model says."""
    aggregator = _one_cluster_aggregator(monkeypatch)
    tools = cast(
        ToolBox,
        SimpleNamespace(
            aggregator=aggregator,
            ontology_manager=SimpleNamespace(get_head_by_iri=lambda iri: None),
        ),
    )
    state = _state(
        [
            _unit(
                0,
                'cd:oregon a q:Observation ; rdfs:label "extra deaths in Oregon" .\n'
                'cd:wash a q:Observation ; rdfs:label "extra deaths in Washington" .',
            ),
            _unit(1, 'cd:other a q:Observation ; rdfs:label "heat deaths" .'),
        ]
    )
    make_merge_facts_node(tools)(state)
    state.coreference_candidates = [
        CoreferenceCandidate(
            left=CD + "oregon__u0", right=CD + "wash__u0", kind="near_miss"
        )
    ]

    monkeypatch.setattr(node_factories, "judge_coreference", _judge({1: "same"}))
    await make_resolve_coreference_node(tools)(state)

    label = URIRef("http://www.w3.org/2000/01/rdf-schema#label")
    holders = {
        s
        for s, o in state.aggregated_facts.subject_objects(label)
        if "deaths in" in str(o)
    }
    assert len(holders) == 2


def test_one_unit_has_no_coreference_to_resolve(monkeypatch) -> None:
    aggregator = _one_cluster_aggregator(monkeypatch)
    tools = cast(
        ToolBox,
        SimpleNamespace(
            aggregator=aggregator,
            ontology_manager=SimpleNamespace(get_head_by_iri=lambda iri: None),
        ),
    )
    state = _state([_unit(0, 'cd:a a q:Person ; rdfs:label "A" .')])
    make_merge_facts_node(tools)(state)
    assert state.coreference_candidates == []
