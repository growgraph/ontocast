"""A request's documents run one after another, each to its own result."""

from __future__ import annotations

import asyncio

import pytest

from ontocast.api.batch import run_documents
from ontocast.api.schemas import (
    ProcessErrorResponse,
    ProcessOkResponse,
    ProcessResultData,
    ProcessResultMetadata,
)
from ontocast.onto.run_meter import BudgetExceededError, RunMeterSnapshot
from ontocast.onto.state import AgentState, InputDocument
from ontocast.stategraph.unit_pipeline import DocumentConversionError

pytestmark = pytest.mark.unit


def _state(filename: str) -> AgentState:
    return AgentState(input_document=InputDocument(filename=filename, data=b"x"))


def _ok() -> ProcessOkResponse:
    return ProcessOkResponse(
        data=ProcessResultData(facts="", ontology=None, ontology_artifacts=[]),
        metadata=ProcessResultMetadata(
            status="success", chunks_processed=1, chunks_remaining=0, budget={}
        ),
    )


def test_each_document_gets_its_own_status_and_failures_stop_nothing() -> None:
    async def run_one(state: AgentState):
        assert state.input_document is not None
        name = state.input_document.filename
        if name == "empty.txt":
            return ProcessErrorResponse(
                error="nothing", error_code="no_units_extracted"
            )
        if name == "scan.pdf":
            raise DocumentConversionError("unreadable", stage="convert")
        if name == "big.txt":
            raise BudgetExceededError(
                "cap",
                snapshot=RunMeterSnapshot(
                    usage_by_model={}, totals={}, token_caps={}, exceeded="input"
                ),
            )
        if name == "bug.txt":
            raise KeyError("boom")
        return _ok()

    names = ["a.txt", "empty.txt", "scan.pdf", "big.txt", "bug.txt", "b.txt"]
    batch = asyncio.run(run_documents([_state(n) for n in names], run_one, None))

    assert [r.filename for r in batch.results] == names
    assert [r.status_code for r in batch.results] == [200, 422, 422, 422, 500, 200]
    codes = [getattr(r.outcome, "error_code", None) for r in batch.results]
    assert str(codes[2]).startswith("conversion_failed")
    assert codes[3] == "budget_exceeded"
    assert batch.results[4].outcome.status == "error"


def test_each_document_holds_the_semaphore_on_its_own() -> None:
    semaphore = asyncio.Semaphore(1)
    held: list[bool] = []

    async def run_one(state: AgentState):
        held.append(semaphore.locked())
        return _ok()

    async def main() -> None:
        await run_documents([_state("a.txt"), _state("b.txt")], run_one, semaphore)
        # Released between and after documents, not held for the request.
        assert not semaphore.locked()

    asyncio.run(main())
    assert held == [True, True]


def test_the_wire_shape_discriminates_success_from_error() -> None:
    async def run_one(state: AgentState):
        return ProcessErrorResponse(error="nope")

    batch = asyncio.run(run_documents([_state("a.txt")], run_one, None))
    wire = batch.model_dump(mode="json")

    assert wire == {
        "results": [
            {
                "filename": "a.txt",
                "status_code": 422,
                "outcome": {
                    "status": "error",
                    "error": "nope",
                    "error_type": None,
                    "error_code": None,
                    "error_details": None,
                },
            }
        ]
    }
    assert type(batch).model_validate(wire) == batch
