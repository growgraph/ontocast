"""An LRU of scope-bound ToolBoxes over one shared runtime.

Tenancy used to be applied by mutating the single process-wide ToolBox: a
``?tenant=`` query parameter retargeted its Fuseki datasets, Qdrant collections
and ontology catalog in place. A lock made that safe, at the cost of serializing
*all* multi-tenant traffic behind one mutex and rebuilding the catalog on every
switch.

Here each scope gets its own ToolBox over its own deep-copied ``Config``, so
isolation is structural rather than disciplinary -- no lock, no interleaving, no
catalog rebuild on switch. The expensive tools stay shared through
:class:`~ontocast.runtime.ToolBoxRuntime`, and a compiled graph is cached
alongside each entry because graph nodes close over their ToolBox.

The LRU is bounded because scopes come from request parameters: without a cap, a
caller iterating tenant names would grow the process without limit.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from ontocast.config import Config
from ontocast.onto.tenancy import TenancyScope

if TYPE_CHECKING:
    from ontocast.runtime import ToolBoxRuntime
    from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


class ToolBoxRegistry:
    """Resolves a tenancy scope to a ToolBox, constructing on miss."""

    def __init__(
        self,
        base_config: Config,
        runtime: "ToolBoxRuntime",
        *,
        max_scopes: int = 16,
    ):
        """Create a registry.

        Args:
            base_config: Config to derive each scope's copy from.
            runtime: Shared tools every scoped ToolBox reuses.
            max_scopes: How many scoped ToolBoxes to keep. Evicting closes the
                entry's backend connections.
        """
        if max_scopes < 1:
            raise ValueError("max_scopes must be at least 1")
        self.base_config = base_config
        self.runtime = runtime
        self.max_scopes = max_scopes
        self._entries: OrderedDict[tuple[str, str], "ToolBox"] = OrderedDict()
        self._graphs: dict[tuple[str, str], Any] = {}
        # Per-key rather than global: two concurrent first-requests for the same
        # tenant must construct once, but requests for *different* tenants have
        # nothing to serialize and a shared lock would reintroduce exactly the
        # bottleneck this class removes.
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._locks_loop: asyncio.AbstractEventLoop | None = None

    def _lock_for(self, key: tuple[str, str]) -> asyncio.Lock:
        """Return the construction lock for ``key``, bound to the running loop.

        Locks are dropped when the loop changes: the CLI bootstrap makes several
        ``asyncio.run`` calls, and a Lock created on a closed loop cannot be
        awaited on a later one.
        """
        loop = asyncio.get_running_loop()
        if self._locks_loop is not loop:
            self._locks.clear()
            self._locks_loop = loop
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    async def get(
        self,
        scope: TenancyScope,
        *,
        ontology_context_mode: Any = None,
        fail_on_vector_store_error: bool = False,
    ) -> "ToolBox":
        """Return the ToolBox for ``scope``, building and initializing on miss.

        Args:
            scope: The partition to resolve.
            ontology_context_mode: Mode to initialize for; decides whether the
                vector store is prepared (``ToolBox.should_initialize_vector_store``).
            fail_on_vector_store_error: Raise rather than log when vector store
                preparation fails. Defaults to false so one tenant's broken
                collection does not fail the request that touched it.

        Returns:
            A ToolBox bound to ``scope``.
        """
        key = scope.key
        existing = self._entries.get(key)
        if existing is not None:
            self._entries.move_to_end(key)
            return existing

        async with self._lock_for(key):
            # Re-check: another task may have built this scope while we waited.
            existing = self._entries.get(key)
            if existing is not None:
                self._entries.move_to_end(key)
                return existing

            from ontocast.toolbox import ToolBox

            logger.info(
                "Building ToolBox for tenancy scope %s/%s", scope.tenant, scope.project
            )
            tools = ToolBox(
                self.base_config.for_tenancy(scope.tenant, scope.project),
                runtime=self.runtime,
            )
            await tools.initialize(
                ontology_context_mode=ontology_context_mode,
                fail_on_vector_store_error=fail_on_vector_store_error,
                wipe_vector_store=False,
            )
            self._entries[key] = tools
            await self._evict_if_needed()
            return tools

    async def _evict_if_needed(self) -> None:
        while len(self._entries) > self.max_scopes:
            evicted_key, evicted = self._entries.popitem(last=False)
            self._graphs.pop(evicted_key, None)
            logger.info("Evicting ToolBox for tenancy scope %s/%s", *evicted_key)
            await evicted.aclose()

    def graph_for(self, scope: TenancyScope, build: Any) -> Any:
        """Return the compiled graph for ``scope``, compiling once per scope.

        Graph nodes are ``partial(fn, tools=tools)`` and ``make_*_node(tools)``
        closures, so a graph is bound to one ToolBox and a scoped ToolBox needs
        its own. Compilation is pure in-memory topology work -- no I/O -- so
        caching it per scope costs far less than the ToolBox it belongs to.

        Args:
            scope: The partition whose graph is wanted.
            build: Zero-argument callable compiling a graph for that scope.

        Returns:
            The cached compiled graph.
        """
        key = scope.key
        graph = self._graphs.get(key)
        if graph is None:
            graph = build()
            self._graphs[key] = graph
        return graph

    async def aclose(self) -> None:
        """Close every scoped ToolBox. Safe to call more than once."""
        while self._entries:
            _, tools = self._entries.popitem(last=False)
            try:
                await tools.aclose()
            except Exception as exc:
                logger.warning("Error closing scoped ToolBox: %s", exc)
        self._graphs.clear()

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def scopes(self) -> list[tuple[str, str]]:
        """Currently resident scopes, least recently used first."""
        return list(self._entries)
