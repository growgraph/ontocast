"""Tokens by phase: every LLM call is charged to the step that made it.

``BudgetTracker`` reported one flat token total, so the split between render,
critic and completion could only be reconstructed offline from prompt caches.
These tests pin the label (task-local, outer step wins), the bookkeeping (phases
add up to the flat totals and survive merging), and the call sites (no agent
calls the LLM outside a phase).
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from ontocast.onto.enum import LLMPhase
from ontocast.onto.state import BudgetTracker
from ontocast.onto.token_usage import TokenUsage
from ontocast.tool.llm import (
    _active_budget_tracker,
    current_llm_phase,
    use_budget_tracker,
    use_llm_phase,
)
from ontocast.toolbox import ToolBox

pytestmark = pytest.mark.unit

PACKAGE = Path(__file__).resolve().parents[1] / "ontocast"


def _usage(inp: int, out: int) -> TokenUsage:
    return TokenUsage(input_tokens=inp, output_tokens=out)


def test_phases_add_up_to_the_flat_totals() -> None:
    tracker = BudgetTracker()
    tracker.add_usage(10, 5, usage=_usage(100, 20), phase=LLMPhase.FACTS_RENDER)
    tracker.add_usage(10, 5, usage=_usage(40, 8), phase=LLMPhase.FACTS_CRITIC)
    tracker.add_usage(10, 0, phase=LLMPhase.FACTS_CRITIC)  # a timed-out call
    tracker.add_cache_hit(10, 5, usage=_usage(7, 3), phase=LLMPhase.FACTS_RENDER)
    tracker.add_usage(1, 1, usage=_usage(1, 1))

    phases = tracker.tokens_by_phase
    assert set(phases) == {"facts_render", "facts_critic", "unlabeled"}
    assert phases["facts_critic"].calls == 2
    assert phases["facts_render"].replays == 1
    assert phases["facts_render"].replayed_input == 7
    assert sum(t.calls for t in phases.values()) == tracker.calls_count
    assert sum(t.replays for t in phases.values()) == tracker.cache_hits
    assert sum(t.input for t in phases.values()) == tracker.input_tokens
    assert sum(t.output for t in phases.values()) == tracker.output_tokens
    assert sum(t.replayed_input for t in phases.values()) == tracker.cached_input_tokens


def test_merge_sums_phases_without_aliasing() -> None:
    unit = BudgetTracker()
    unit.add_usage(1, 1, usage=_usage(5, 2), phase=LLMPhase.FACTS_RENDER)
    document = BudgetTracker()
    document.merge_from(unit)
    document.merge_from(unit)

    assert document.tokens_by_phase["facts_render"].input == 10
    assert unit.tokens_by_phase["facts_render"].input == 5
    assert "tokens_by_phase" in document.model_dump()


def test_an_enclosing_phase_wins_and_is_restored() -> None:
    assert current_llm_phase() is None
    with use_llm_phase(LLMPhase.ONTOLOGY_CONSOLIDATION):
        with use_llm_phase(LLMPhase.ONTOLOGY_RENDER):
            assert current_llm_phase() == "ontology_consolidation"
        assert current_llm_phase() == "ontology_consolidation"
    assert current_llm_phase() is None


@pytest.mark.anyio
async def test_concurrent_units_keep_their_own_phase() -> None:
    observed: dict[int, str | None] = {}
    phases = [LLMPhase.FACTS_RENDER, LLMPhase.FACTS_CRITIC, LLMPhase.UNIT_SUMMARY]

    async def worker(index: int) -> None:
        with use_llm_phase(phases[index]):
            await asyncio.sleep(0.01 * (3 - index))
            observed[index] = current_llm_phase()

    await asyncio.gather(*(worker(i) for i in range(3)))

    assert observed == {i: str(p) for i, p in enumerate(phases)}


def test_the_llm_tool_charges_the_bound_phase() -> None:
    from ontocast.config import LLMConfig
    from ontocast.tool.llm import LLMTool

    tool = LLMTool(config=LLMConfig())
    tracker = BudgetTracker()
    with use_budget_tracker(tracker), use_llm_phase(LLMPhase.FACTS_COMPLETION):
        tool._record_api_usage("prompt", SimpleNamespace(content="x"), _usage(3, 1))
        tool._record_cache_hit("prompt", "response", _usage(2, 1))

    totals = tracker.tokens_by_phase["facts_completion"]
    assert (totals.calls, totals.input, totals.replays) == (1, 3, 1)


@pytest.mark.anyio
async def test_chunking_charges_the_document_tracker(monkeypatch) -> None:
    """LLM section classification runs before any unit binds a tracker."""
    import importlib

    from ontocast.onto.state import AgentState

    chunk_module = importlib.import_module("ontocast.agent.chunk_text")

    seen: list[object] = []

    async def fake_prepare(*_args, **_kwargs):
        seen.append(_active_budget_tracker.get())
        return []

    monkeypatch.setattr(chunk_module, "prepare_content_units", fake_prepare)
    state = AgentState()
    state.docling_doc = object()  # type: ignore[assignment]
    tools = SimpleNamespace(
        chunker=SimpleNamespace(
            config=SimpleNamespace(
                bibliography_mode="skip", non_content_mode="extract", min_unit_chars=0
            )
        )
    )

    await chunk_module.chunk_text(state, cast(ToolBox, tools))

    assert seen == [state.budget_tracker]


def _llm_call_sites() -> list[tuple[Path, ast.Call]]:
    sites = []
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                sites.append((path, node))
    return sites


def test_every_retrying_call_names_its_phase() -> None:
    unlabeled = [
        f"{path.relative_to(PACKAGE)}:{call.lineno}"
        for path, call in _llm_call_sites()
        if isinstance(call.func, ast.Name)
        and call.func.id == "call_llm_with_retry"
        and path.name != "common.py"  # its own re-entry under the phase it set
        and not any(kw.arg == "phase" for kw in call.keywords)
    ]
    assert unlabeled == []


#: Direct ``tools.llm(...)`` calls that belong to no document, so no phase.
_UNPHASED_DIRECT_CALLS = {"toolbox.py"}


def test_every_direct_llm_call_sits_in_a_phase() -> None:
    """A module calling the LLM tool directly must also open a phase."""
    for path in PACKAGE.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        calls_directly = "await tools.llm(" in source or "await llm_tool(" in source
        if not calls_directly or path.name in _UNPHASED_DIRECT_CALLS:
            continue
        if path.name == "common.py":  # call_llm_with_retry itself
            continue
        assert "use_llm_phase(" in source, path.relative_to(PACKAGE)
