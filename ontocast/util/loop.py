"""Guards for helpers that drive async work through :func:`asyncio.run`.

Several construction paths are synchronous wrappers around coroutines. They are
correct from a script or the CLI, and illegal from inside a running event loop
-- which is exactly where an embedder calls them from. Python's own error for
that case (``asyncio.run() cannot be called from a running event loop``) names
neither the offending call nor the fix, so these helpers raise a directive one.
"""

from __future__ import annotations

import asyncio


def require_no_running_loop(sync_name: str, async_name: str) -> None:
    """Raise if called from inside a running event loop.

    Args:
        sync_name: The synchronous entry point being guarded.
        async_name: The coroutine the caller should await instead.

    Raises:
        RuntimeError: If an event loop is already running on this thread.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(
        f"{sync_name}() drives async setup through asyncio.run() and cannot be "
        f"called from inside a running event loop. Await {async_name}() instead."
    )
