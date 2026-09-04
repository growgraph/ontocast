"""How the per-unit fan-out separates unit faults from deployment faults."""

from __future__ import annotations

import pytest

from ontocast.onto.enum import WorkflowNode
from ontocast.onto.retrieval_capabilities import EmptyOntologyContextError
from ontocast.onto.state import AgentState
from ontocast.stategraph.node_factories import _gather_units
from ontocast.tool.llm import LLMConfigurationError

pytestmark = pytest.mark.unit


async def _ok(value: int) -> int:
    return value


async def _boom(error: Exception) -> int:
    raise error


@pytest.mark.anyio
async def test_a_unit_fault_is_isolated_to_its_own_unit() -> None:
    state = AgentState()

    results, failures = await _gather_units(
        WorkflowNode.RENDER_FACTS,
        state,
        [_ok(1), _boom(RuntimeError("provider exploded")), _ok(3)],
    )

    assert results == [1, 3]
    assert failures == 1


@pytest.mark.anyio
async def test_an_ontology_context_fault_is_not_isolated() -> None:
    """It describes the deployment, so every sibling has it too.

    Isolating it turned one configuration fault into N unit failures and a
    successful run over an empty graph. Re-raising after the gather has drained
    orphans nothing: every sibling has already been awaited.
    """
    state = AgentState()

    with pytest.raises(EmptyOntologyContextError):
        await _gather_units(
            WorkflowNode.RENDER_FACTS,
            state,
            [_ok(1), _boom(EmptyOntologyContextError("catalog is empty")), _ok(3)],
        )


@pytest.mark.anyio
async def test_a_rejected_request_is_not_isolated() -> None:
    """The provider refuses the request as configured, not this unit's prompt.

    Isolating it is how a run whose every call was rejected still finished,
    uploaded an empty graph, dumped a manifest and exited 0.

    Note that making the error a ``BaseException`` would not achieve this:
    ``asyncio.gather(..., return_exceptions=True)`` captures those as values
    too, so the re-raise has to be explicit.
    """
    state = AgentState()

    with pytest.raises(LLMConfigurationError):
        await _gather_units(
            WorkflowNode.RENDER_FACTS,
            state,
            [
                _ok(1),
                _boom(LLMConfigurationError("openai/gpt-x rejected the request")),
                _ok(3),
            ],
        )
