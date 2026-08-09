"""Tests for the event-loop lag sampler.

The sampler's whole value is that it separates *awaited* latency from
*blocking* latency. These tests pin exactly that: slow-but-awaited work must
read as zero lag, and a synchronous block of the same duration must not.
"""

import asyncio
import time

import pytest

from ontocast.util.loop_lag import loop_lag

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_awaited_io_produces_no_meaningful_lag() -> None:
    async with loop_lag(interval=0.01) as lag:
        # Stands in for a slow provider call: it yields, so every other task on
        # the loop keeps running and the loop is never blocked.
        await asyncio.sleep(0.3)

    assert lag.samples > 0
    assert lag.peak < 0.05, f"awaited sleep should not register as a stall: {lag.peak}"


async def test_synchronous_block_is_attributed_as_lag() -> None:
    async with loop_lag(interval=0.01) as lag:
        # Stands in for the rdflib merge/copy/serialize work inside a unit task.
        deadline = time.perf_counter() + 0.3
        while time.perf_counter() < deadline:
            pass

    assert lag.peak >= 0.2, f"a 0.3s block should surface as lag: {lag.peak}"
    assert lag.total >= 0.2


async def test_sampler_stops_and_reports_when_the_body_raises() -> None:
    with pytest.raises(ValueError):
        async with loop_lag(interval=0.01) as lag:
            await asyncio.sleep(0.05)
            raise ValueError("boom")

    # The accumulator survives the exception, and the sampler task is finished
    # rather than left running against a dead scope.
    assert lag.samples > 0
    await asyncio.sleep(0.05)
    samples_after = lag.samples
    await asyncio.sleep(0.05)
    assert lag.samples == samples_after
