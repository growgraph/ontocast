"""``POST /flush`` must leave the vector store in a state it can serve from.

``clean_tenancy_data`` drops the collections and their embedding metadata.
Leaving the toolbox marked ready afterwards pointed every later search in the
process at a collection that no longer existed, and the failure surfaced as an
empty retrieval rather than an error. The startup path already wipes, recreates
and only then reports ready; the flush path now follows it.
"""

import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from ontocast.config import Config
from ontocast.toolbox import ToolBox

pytestmark = pytest.mark.unit


def _tools(*, ready: bool, recreate_fails: bool = False) -> Any:
    tools = cast(Any, ToolBox.__new__(ToolBox))
    tools.config = Config()
    tools.triple_store_manager = None
    tools.vector_store = MagicMock()
    tools.vector_store.supports_tenancy_partition = MagicMock(return_value=True)
    tools.vector_store.clean_tenancy = AsyncMock()
    tools.vector_store.initialize = AsyncMock(
        side_effect=RuntimeError("collection gone") if recreate_fails else None
    )
    tools.vector_store_ready = ready
    tools.vector_store_last_error = None
    tools._tenancy_lock = None
    tools._tenancy_lock_loop = None
    tools._active_tenancy = ("other", "scope")
    tools.shapes_catalog = MagicMock()
    return tools


def test_flush_recreates_the_collections_it_dropped() -> None:
    tools = _tools(ready=True)

    asyncio.run(tools.clean_tenancy_data("acme", "p1"))

    tools.vector_store.clean_tenancy.assert_awaited_once_with("acme", "p1")
    tools.vector_store.initialize.assert_awaited_once()
    assert tools.vector_store_ready is True


def test_a_failed_recreate_leaves_the_store_marked_not_ready() -> None:
    tools = _tools(ready=True, recreate_fails=True)

    asyncio.run(tools.clean_tenancy_data("acme", "p1"))

    assert tools.vector_store_ready is False
    assert isinstance(tools.vector_store_last_error, RuntimeError)


def test_a_store_that_was_never_ready_is_not_brought_up_by_a_flush() -> None:
    """Flush drops data; it is not the place a deployment gains retrieval."""
    tools = _tools(ready=False)

    asyncio.run(tools.clean_tenancy_data("acme", "p1"))

    tools.vector_store.initialize.assert_not_awaited()
    assert tools.vector_store_ready is False
