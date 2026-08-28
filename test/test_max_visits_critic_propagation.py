"""``--max-visits`` must observably reach the unit loops.

The 2026-08 matsci ablation compared ``--max-visits 1`` against
``--max-visits 2``, but LLM-call accounting later showed the critic never ran
in the second arm — the two runs were an A/A comparison, and nothing recorded
the effective setting. These tests pin the two ends of the chain that make a
future arm auditable: the batch entry path writes the flag into
``AgentState``, and a unit loop at ``max_visits=2`` actually spends a critic
call.
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import URIRef

from ontocast.api.process_helpers import expand_input_to_states
from ontocast.config import Config
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import OntologyContextMode, RenderMode, Status
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.onto.unit_states import UnitFactsState
from ontocast.stategraph import atomic as atomic_module
from ontocast.stategraph.atomic import facts_loop
from ontocast.stategraph.context_resolver import UnitOntologyContext
from ontocast.stategraph.unit_context import UnitLoopContext
from ontocast.tool.atomic import AtomicToolBox
from ontocast.toolbox import ToolBox
from test.snapshot_helpers import empty_snapshot

pytestmark = pytest.mark.unit


def test_batch_entry_path_writes_max_visits_onto_state(
    tmp_path: pathlib.Path,
) -> None:
    document = tmp_path / "doc.txt"
    document.write_text("Alice works for ACME.")
    states = expand_input_to_states(
        document,
        config=Config(),
        head_chunks=None,
        ontology_context_mode_value=OntologyContextMode.SELECTED_SINGLE_ONTOLOGY,
        tenant=None,
        project=None,
        max_visits=2,
    )
    assert [state.max_visits for state in states] == [2]


def _unit_state(**kwargs) -> UnitFactsState:
    unit = ContentUnit(
        text="Alice works for ACME.",
        index=0,
        doc_iri=URIRef("https://example.com/doc/d1"),
        graph=RDFGraph(),
    )
    return UnitFactsState(
        content_unit=unit, ontology_snapshot=empty_snapshot(), **kwargs
    )


def _tools() -> ToolBox:
    return cast(
        ToolBox,
        SimpleNamespace(
            get_atomic_tools=lambda: cast(
                AtomicToolBox,
                SimpleNamespace(
                    facts_llm_repair_visits=0,
                    additional_standard_namespaces=(),
                    validation_policy=None,
                ),
            ),
        ),
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("max_visits", "expected_critic_calls"),
    [(1, 0), (2, 1)],
)
async def test_critic_spends_a_call_exactly_when_max_visits_allows(
    monkeypatch, max_visits: int, expected_critic_calls: int
) -> None:
    """At 1 the critic is skipped; at 2 a successful render is criticised.

    This is billing-visible in production (``criticise_facts`` is a provider
    call), which is how the silent A/A run was eventually detected — so the
    loop-level guarantee is asserted on call count, not on log output.
    """
    critic_calls = 0

    async def ok_render(state, tools, **kwargs):
        state.status = Status.SUCCESS
        return state

    async def converging_critic(state, tools):
        nonlocal critic_calls
        critic_calls += 1
        state.status = Status.SUCCESS
        return state

    monkeypatch.setattr(atomic_module, "render_facts", ok_render)
    monkeypatch.setattr(atomic_module, "criticise_facts", converging_critic)

    context = UnitLoopContext.from_agent_state(AgentState(render_mode=RenderMode.FACTS))
    resolved = UnitOntologyContext(
        snapshot=empty_snapshot(), writable_iris=[], confidence=1.0
    )
    result = await facts_loop(
        _unit_state(max_visits_per_node=max_visits),
        _tools(),
        context,
        pre_resolved_context=resolved,
    )

    assert result.status == Status.SUCCESS
    assert critic_calls == expected_critic_calls
