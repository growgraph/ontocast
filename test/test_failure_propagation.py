"""Failure must be distinguishable from an empty-but-successful extraction.

Both map nodes used to end with an unconditional ``Status.SUCCESS``, so a
document whose conversion failed -- or whose every unit failed -- returned
HTTP 200 with empty facts.
"""

from __future__ import annotations

import pytest

from ontocast.config import (
    Config,
    LLMConfig,
    LLMProvider,
    OllamaModel,
    PathConfig,
    ToolConfig,
)
from ontocast.onto.enum import Status
from ontocast.onto.model import UnitFailure
from ontocast.onto.state import AgentState
from ontocast.stategraph.node_factories import (
    _map_stage_status,
    make_render_facts_node,
    make_render_ontology_node,
)
from ontocast.toolbox import ToolBox

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("failed", "total", "expected"),
    [
        (0, 3, Status.SUCCESS),
        (1, 3, Status.SUCCESS),
        (2, 3, Status.SUCCESS),
        (3, 3, Status.FAILED),
        (1, 1, Status.FAILED),
        (0, 0, Status.SUCCESS),
    ],
)
def test_map_stage_status(failed: int, total: int, expected: Status) -> None:
    """Total failure is FAILED; partial failure keeps the surviving output."""
    assert _map_stage_status(failed, total) == expected


@pytest.fixture
def toolbox() -> ToolBox:
    """A ToolBox with no external services; the map nodes short-circuit before use."""
    return ToolBox(
        Config(
            tool_config=ToolConfig(
                path_config=PathConfig(),
                llm_config=LLMConfig(
                    provider=LLMProvider.OLLAMA,
                    model_name=OllamaModel.LLAMA3_1,
                    base_url="http://localhost:11434",
                ),
            ),
        )
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "make_node", [make_render_ontology_node, make_render_facts_node]
)
async def test_map_node_preserves_upstream_failure(toolbox, make_node) -> None:
    """An empty unit list after a failed conversion must not become SUCCESS."""
    state = AgentState()
    state.content_units = []
    state.status = Status.FAILED

    result = await make_node(toolbox)(state)

    assert result.status == Status.FAILED


@pytest.mark.anyio
@pytest.mark.parametrize(
    "make_node", [make_render_ontology_node, make_render_facts_node]
)
async def test_map_node_reports_success_when_there_is_nothing_to_do(
    toolbox, make_node
) -> None:
    state = AgentState()
    state.content_units = []

    result = await make_node(toolbox)(state)

    assert result.status == Status.SUCCESS


def test_unit_failure_record_round_trips() -> None:
    failure = UnitFailure(
        unit_index=2, phase="facts", stage="facts_critique", reason="boom"
    )

    assert failure.model_dump(mode="json") == {
        "unit_index": 2,
        "phase": "facts",
        "stage": "facts_critique",
        "reason": "boom",
    }


def test_agent_state_starts_with_no_unit_failures() -> None:
    assert AgentState().unit_failures == []
