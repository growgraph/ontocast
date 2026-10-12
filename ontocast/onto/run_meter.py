"""Per-run token meter with enforceable limits.

A :class:`RunMeter` counts tokens for one run as they are spent, across every
concurrent content unit, and refuses the next provider call once a limit is
reached: a cap on a token count, or a caller's ``limit`` policy. The meter knows
nothing about money; a caller that prices tokens does so in its ``limit``.

The meter is bound to the running task with :func:`use_run_meter`. ``asyncio``
tasks (and LangGraph nodes) copy the context they start in, so every unit of a
fan-out records into the same object.

A refusal is an exception, not a state field: :class:`BudgetExceededError`
propagates out of the unit loops and best-effort steps, the fan-out re-raises it
once every sibling unit has finished, and it leaves the graph run (``astream`` /
``ainvoke``) carrying the meter's snapshot. The meter itself also stays readable
after the run, so a caller can settle from ``meter.snapshot()`` either way.

Deliberately a leaf module with no OntoCast imports, so :mod:`ontocast.tool.llm`
can import it without reaching back into :mod:`ontocast.tool`.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TypedDict

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TokenTotals(BaseModel):
    """Token counts for one model, split by where each response came from.

    ``input`` to ``reasoning`` count provider calls made in this run, as the
    provider reports them: ``cache_read`` and ``cache_write`` are counted inside
    ``input``, ``reasoning`` inside ``output``. ``replayed_*`` count responses
    served by the OntoCast disk cache, for which no provider call was made.
    """

    input: int = Field(
        default=0, description="Input tokens of provider calls, cache reads included."
    )
    cache_read: int = Field(
        default=0, description="Input tokens served from the provider's prompt cache."
    )
    cache_write: int = Field(
        default=0, description="Input tokens written to the provider's prompt cache."
    )
    output: int = Field(
        default=0, description="Output tokens of provider calls, reasoning included."
    )
    reasoning: int = Field(default=0, description="Reasoning tokens of provider calls.")
    replayed_input: int = Field(
        default=0, description="Input tokens of responses replayed from disk cache."
    )
    replayed_output: int = Field(
        default=0, description="Output tokens of responses replayed from disk cache."
    )
    calls: int = Field(default=0, description="Provider calls that returned.")
    replays: int = Field(default=0, description="Responses replayed from disk cache.")

    def add(self, other: TokenTotals) -> None:
        """Add every count of ``other`` to this one."""
        for name in TokenTotals.model_fields:
            setattr(self, name, getattr(self, name) + getattr(other, name))


#: Field names a token cap may be keyed by.
TOKEN_FIELDS: tuple[str, ...] = tuple(TokenTotals.model_fields)

#: A caller's policy: given the counts per model so far, a reason to stop the
#: run, or None to let the next provider call go ahead.
type RunLimit = Callable[[Mapping[str, TokenTotals]], str | None]


class RunMeterSnapshot(TypedDict):
    """JSON-serialisable state of a :class:`RunMeter`."""

    usage_by_model: dict[str, dict[str, int]]
    totals: dict[str, int]
    token_caps: dict[str, int]
    exceeded: str | None


class BudgetExceededError(RuntimeError):
    """A run reached a token cap or its meter's ``limit``.

    Raised before the provider call that would have exceeded it, so that call is
    never made. Calls already in flight finish and are recorded, so the final
    usage can pass the limit by what they spent.

    Attributes:
        snapshot: The meter's state when the call was refused.
    """

    def __init__(self, message: str, snapshot: RunMeterSnapshot) -> None:
        super().__init__(message)
        self.snapshot = snapshot


class RunMeter:
    """Thread- and task-safe token meter for one run.

    ``check()``, called before every provider call, raises
    :class:`BudgetExceededError` once any field summed over models reaches its
    ``token_caps`` entry, or once ``limit`` returns a reason. After the first
    refusal every later ``check()`` refuses too.

    Args:
        limit: The caller's policy, given a copy of the counts per model before
            each provider call; returns a reason to stop, or None. Called
            outside the meter's lock, so it may read the meter; an exception it
            raises propagates to the call it was guarding.
        token_caps: Limits on token counts summed over models, keyed by field
            name.
        on_usage: Called with a snapshot after usage is recorded, at most once
            every ``min_interval_s`` seconds, and on :meth:`flush`. Called
            outside the lock; an exception it raises is logged, not raised.
        min_interval_s: Minimum seconds between two ``on_usage`` calls.
        clock: Monotonic time source for the throttle.

    Raises:
        ValueError: On an unknown field name in ``token_caps``.
    """

    def __init__(
        self,
        *,
        limit: RunLimit | None = None,
        token_caps: dict[str, int] | None = None,
        on_usage: Callable[[RunMeterSnapshot], None] | None = None,
        min_interval_s: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        for name in token_caps or {}:
            if name not in TokenTotals.model_fields:
                raise ValueError(
                    f"Unknown token field {name!r}; expected one of {TOKEN_FIELDS}"
                )
        self.limit = limit
        self.token_caps: dict[str, int] = dict(token_caps or {})
        self.on_usage = on_usage
        self.min_interval_s = min_interval_s
        self.usage_by_model: dict[str, TokenTotals] = {}
        self._clock = clock
        self._lock = threading.Lock()
        self._exceeded: str | None = None
        self._last_emit: float | None = None

    @classmethod
    def from_token_limits(
        cls,
        *,
        max_input_tokens: int | None = None,
        max_output_tokens: int | None = None,
    ) -> RunMeter | None:
        """A meter capping ``input`` and ``output``, or None if unlimited.

        Args:
            max_input_tokens: Cap on provider input tokens over all models.
            max_output_tokens: Cap on provider output tokens over all models.
        """
        caps: dict[str, int] = {}
        if max_input_tokens is not None:
            caps["input"] = max_input_tokens
        if max_output_tokens is not None:
            caps["output"] = max_output_tokens
        return cls(token_caps=caps) if caps else None

    @property
    def exceeded(self) -> str | None:
        """Why the meter refused a call, or None if it has not."""
        return self._exceeded

    def record(self, model: str, usage: TokenTotals) -> None:
        """Add ``usage`` to ``model``'s totals and notify ``on_usage`` if due."""
        with self._lock:
            self.usage_by_model.setdefault(model, TokenTotals()).add(usage)
        self._maybe_emit()

    def totals(self) -> TokenTotals:
        """Counts summed over every model."""
        with self._lock:
            return self._totals_locked()

    def usage(self) -> dict[str, TokenTotals]:
        """A copy of the counts per model."""
        with self._lock:
            return self._usage_copy_locked()

    def check(self) -> None:
        """Raise :class:`BudgetExceededError` if a limit has been reached."""
        with self._lock:
            reason = self._exceeded or self._capped_locked()
            usage = self._usage_copy_locked() if reason is None else None
        if reason is None and self.limit is not None and usage is not None:
            reason = self.limit(usage)
        if reason is None:
            return
        with self._lock:
            if self._exceeded is None:
                self._exceeded = reason
            reason = self._exceeded
            snapshot = self._snapshot_locked()
        raise BudgetExceededError(f"Run budget exceeded: {reason}", snapshot)

    def snapshot(self) -> RunMeterSnapshot:
        """Current state as a JSON-serialisable dict."""
        with self._lock:
            return self._snapshot_locked()

    def flush(self) -> None:
        """Call ``on_usage`` now, whatever the throttle."""
        self._maybe_emit(force=True)

    def _maybe_emit(self, *, force: bool = False) -> None:
        callback = self.on_usage
        if callback is None:
            return
        with self._lock:
            now = self._clock()
            due = (
                force
                or self._last_emit is None
                or now - self._last_emit >= self.min_interval_s
            )
            if not due:
                return
            self._last_emit = now
            snapshot = self._snapshot_locked()
        try:
            callback(snapshot)
        except Exception:
            logger.exception("RunMeter on_usage callback failed")

    def _totals_locked(self) -> TokenTotals:
        total = TokenTotals()
        for usage in self.usage_by_model.values():
            total.add(usage)
        return total

    def _usage_copy_locked(self) -> dict[str, TokenTotals]:
        return {
            model: usage.model_copy() for model, usage in self.usage_by_model.items()
        }

    def _capped_locked(self) -> str | None:
        totals = self._totals_locked()
        for name, cap in self.token_caps.items():
            used = getattr(totals, name)
            if used >= cap:
                return f"{name} {used} reached its cap of {cap}"
        return None

    def _snapshot_locked(self) -> RunMeterSnapshot:
        totals = self._totals_locked()
        return RunMeterSnapshot(
            usage_by_model={
                model: usage.model_dump()
                for model, usage in self.usage_by_model.items()
            },
            totals=totals.model_dump(),
            token_caps=dict(self.token_caps),
            exceeded=self._exceeded,
        )


# Task-local, like the budget tracker in ontocast.tool.llm: asyncio.gather and
# LangGraph copy the context into each task, and the copy holds the same meter.
_active_run_meter: ContextVar[RunMeter | None] = ContextVar(
    "ontocast_active_run_meter", default=None
)


def active_run_meter() -> RunMeter | None:
    """The meter bound to the running task, if any."""
    return _active_run_meter.get()


@contextmanager
def use_run_meter(meter: RunMeter) -> Iterator[RunMeter]:
    """Meter every provider call made inside this block with ``meter``.

    On exit the meter is flushed. If the meter refused a call inside the block
    and the block itself raised nothing, :class:`BudgetExceededError` is raised
    here, so a refusal that a best-effort step swallowed still stops the run.
    """
    token = _active_run_meter.set(meter)
    try:
        yield meter
    finally:
        _active_run_meter.reset(token)
        meter.flush()
    if meter.exceeded is not None:
        raise BudgetExceededError(
            f"Run budget exceeded: {meter.exceeded}", meter.snapshot()
        )


@contextmanager
def run_meter_scope(
    *,
    max_input_tokens: int | None = None,
    max_output_tokens: int | None = None,
) -> Iterator[RunMeter | None]:
    """Bind a token-capped meter for one document unless one is already bound.

    An already-bound meter is yielded unchanged and the caps are ignored: the
    caller that bound it owns the limits. With no meter bound and no cap set,
    yields None and meters nothing.

    Args:
        max_input_tokens: Cap on provider input tokens for the document.
        max_output_tokens: Cap on provider output tokens for the document.
    """
    outer = _active_run_meter.get()
    if outer is not None:
        yield outer
        return
    meter = RunMeter.from_token_limits(
        max_input_tokens=max_input_tokens, max_output_tokens=max_output_tokens
    )
    if meter is None:
        yield None
        return
    with use_run_meter(meter):
        yield meter
