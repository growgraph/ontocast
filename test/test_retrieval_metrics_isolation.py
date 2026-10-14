"""Retrieval metrics are per task, not per retriever.

The patch retriever is one shared instance, and content units retrieve
concurrently. The resolver reads ``last_retrieval_metrics`` back after its own
retrieval and makes the expanded ontologies writable, so metrics held on the
instance let a unit pick up another unit's expansion.
"""

from __future__ import annotations

import asyncio

import pytest

from ontocast.tool.vector_store.patch_retriever import OntologyPatchRetriever

pytestmark = pytest.mark.unit


@pytest.mark.anyio
async def test_concurrent_units_read_their_own_retrieval_metrics() -> None:
    retriever = OntologyPatchRetriever.model_construct()
    observed: dict[int, object] = {}

    async def unit(index: int) -> None:
        retriever._last_retrieval_metrics = {}
        retriever._last_retrieval_metrics["expanded_ontology_iris"] = [f"o{index}"]
        # Every other unit writes its own metrics while this one waits.
        await asyncio.sleep(0.01 * (3 - index))
        observed[index] = retriever.last_retrieval_metrics["expanded_ontology_iris"]

    await asyncio.gather(*(unit(i) for i in range(3)))

    assert observed == {0: ["o0"], 1: ["o1"], 2: ["o2"]}
