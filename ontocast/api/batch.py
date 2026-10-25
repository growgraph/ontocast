"""Run a request's documents one after another, each to its own result."""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence

from ontocast.api.responses import (
    budget_failure,
    conversion_failure,
    section_selection_failure,
)
from ontocast.api.schemas import (
    ProcessBatchResponse,
    ProcessErrorResponse,
    ProcessItemResult,
    ProcessOkResponse,
)
from ontocast.onto.run_meter import BudgetExceededError
from ontocast.onto.state import AgentState
from ontocast.stategraph.unit_pipeline import DocumentConversionError
from ontocast.tool.chunk.prepare import SectionSelectionEmptyError

logger = logging.getLogger(__name__)

RunDocument = Callable[
    [AgentState], Awaitable[ProcessOkResponse | ProcessErrorResponse]
]


async def run_documents(
    states: Sequence[AgentState],
    run_one: RunDocument,
    semaphore: asyncio.Semaphore | None,
) -> ProcessBatchResponse:
    """Run ``run_one`` on each state in order; one document's failure stops nothing.

    Each run holds ``semaphore`` on its own, so a long request does not keep a
    slot between its documents.

    Args:
        states: One state per document, each with ``input_document`` set.
        run_one: Extracts one document; returns its result or its error.
        semaphore: The server's concurrency cap, if any.

    Returns:
        ProcessBatchResponse: One result per state, in order.
    """
    results: list[ProcessItemResult] = []
    for state in states:
        if semaphore is None:
            results.append(await _run(state, run_one))
            continue
        async with semaphore:
            results.append(await _run(state, run_one))
    return ProcessBatchResponse(results=results)


async def _run(state: AgentState, run_one: RunDocument) -> ProcessItemResult:
    assert state.input_document is not None
    filename = state.input_document.filename
    try:
        outcome = await run_one(state)
    except DocumentConversionError as exc:
        outcome = conversion_failure(exc, exc.stage)
    except SectionSelectionEmptyError as exc:
        outcome = section_selection_failure(exc)
    except BudgetExceededError as exc:
        outcome = budget_failure(exc)
    except Exception as exc:
        logger.error("Processing %s failed", filename, exc_info=True)
        error = ProcessErrorResponse(error=str(exc), error_type=type(exc).__name__)
        return ProcessItemResult(filename=filename, status_code=500, outcome=error)
    status_code = 200 if isinstance(outcome, ProcessOkResponse) else 422
    return ProcessItemResult(
        filename=filename, status_code=status_code, outcome=outcome
    )
