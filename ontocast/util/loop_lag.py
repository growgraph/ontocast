"""Event-loop lag sampling, for telling provider latency apart from CPU stalls.

A fan-out node that awaits N provider calls concurrently and a fan-out node that
serialises N synchronous rdflib merges look identical in a wall-clock
measurement: both take a long time and both report the same summed per-unit
duration. They are not the same problem, and the fixes are opposite -- the first
wants more concurrency, the second wants the CPU work hoisted out of the loop.

Awaited I/O yields control, so it produces *zero* lag no matter how slow the
provider is. A synchronous block does not yield, so every task on the loop --
including the callbacks that would read the other units' sockets -- waits for it.
Sampling how late a fixed-interval sleep actually wakes up therefore measures
exactly the on-loop CPU stall and nothing else.

Usage::

    async with loop_lag() as lag:
        await run_the_fan_out()
    tracker.add_duration(f"{node}/loop_lag_total", lag.total)
    tracker.add_duration(f"{node}/loop_lag_max", lag.peak)
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

#: How often to probe the loop. Short enough to catch a single ~100ms stall,
#: long enough that the sampler itself is free (one wakeup per 50ms).
DEFAULT_SAMPLE_INTERVAL = 0.05


@dataclass
class LoopLag:
    """Accumulated event-loop delay observed while sampling.

    Attributes:
        total: Sum of all observed delays, in seconds. Roughly the time the loop
            spent unable to service ready callbacks.
        peak: Longest single observed delay, in seconds. A peak above ~0.3s is
            an unambiguous fingerprint of one long synchronous block.
        samples: Number of probes taken, for sanity-checking coverage.
    """

    total: float = 0.0
    peak: float = 0.0
    samples: int = 0


@asynccontextmanager
async def loop_lag(
    interval: float = DEFAULT_SAMPLE_INTERVAL,
) -> AsyncIterator[LoopLag]:
    """Sample event-loop lag for the duration of the block.

    Args:
        interval: Seconds between probes.

    Yields:
        LoopLag: Live accumulator; read it after the block exits.
    """
    lag = LoopLag()
    stop = asyncio.Event()
    started_sampling = asyncio.Event()

    async def sample() -> None:
        started_sampling.set()
        while not stop.is_set():
            started = time.perf_counter()
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
            # Anything beyond the interval is time the loop could not hand
            # control back -- i.e. somebody blocked it.
            delay = time.perf_counter() - started - interval
            if delay > 0:
                lag.total += delay
                lag.peak = max(lag.peak, delay)
            lag.samples += 1

    sampler = asyncio.create_task(sample())
    # Hand control to the sampler before the body runs. Without this the body is
    # free to block the loop immediately, and the sampler -- merely scheduled,
    # never started -- reports a stall of zero for the one case it exists to
    # catch.
    await started_sampling.wait()
    try:
        yield lag
    finally:
        stop.set()
        # The sampler only awaits ``stop``, so this cannot block on the work
        # that just finished; shield the caller from a sampler-side error.
        try:
            await sampler
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - defensive
            pass
