"""Tests for per-scope ToolBoxes and the tenancy registry.

Tenancy used to be applied by mutating the single process-wide ToolBox. These
tests pin the properties that replaced it: scopes are isolated by construction,
the expensive tools are still shared, and the LRU is bounded and closes what it
evicts.
"""

import asyncio
from typing import Any, cast

import pytest
from pydantic import ValidationError

from ontocast.config import Config, ToolConfig
from ontocast.config.settings import PathConfig
from ontocast.onto.enum import VectorStoreBackend
from ontocast.onto.tenancy import TenancyScope
from ontocast.registry import ToolBoxRegistry
from ontocast.runtime import ToolBoxRuntime
from ontocast.tool.agg.aggregate import EmbeddingBasedAggregator
from ontocast.tool.llm import LLMTool
from ontocast.toolbox import ToolBox

pytestmark = pytest.mark.unit

STUB_LLM = cast(LLMTool, object())


def _config(tmp_path) -> Config:
    config = Config.in_memory(tool_config=ToolConfig(path_config=PathConfig()))
    config.tool_config.vector_store.backend = VectorStoreBackend.NONE
    return config


def _toolbox(tmp_path) -> ToolBox:
    return ToolBox(_config(tmp_path), llm=STUB_LLM)


# -- TenancyScope ----------------------------------------------------------


def test_scope_resolves_backend_names() -> None:
    scope = TenancyScope.build("acme", "p1")
    assert scope.facts_name == "acme--p1--facts"
    assert scope.ontologies_name == "acme--p1--ontologies"
    assert scope.key == ("acme", "p1")


def test_scope_strips_and_rejects_blank() -> None:
    assert TenancyScope.build("  acme ", " p1 ").key == ("acme", "p1")
    with pytest.raises(ValueError, match="non-empty"):
        TenancyScope.build("acme", "   ")


def test_scope_is_frozen() -> None:
    """A scope keys the registry; a mutable one could alias two entries."""
    scope = TenancyScope.build("acme", "p1")
    with pytest.raises(ValidationError):
        cast(Any, scope).tenant = "other"


# -- Config.for_tenancy ----------------------------------------------------


def test_for_tenancy_applies_names_everywhere(tmp_path) -> None:
    scoped = _config(tmp_path).for_tenancy("acme", "p1")
    tool_config = scoped.tool_config
    assert tool_config.fuseki.dataset == "acme--p1--facts"
    assert tool_config.fuseki.ontologies_dataset == "acme--p1--ontologies"
    assert tool_config.qdrant.facts_collection == "acme--p1--facts"
    assert tool_config.qdrant.ontology_collection == "acme--p1--ontologies"
    assert tool_config.vector_store.facts_table == "acme--p1--facts"
    assert tool_config.lancedb.ontology_table == "acme--p1--ontologies"


def test_for_tenancy_deep_copies_every_mutated_section(tmp_path) -> None:
    """The load-bearing check.

    Vector store managers receive ``tool_config.vector_store`` and
    ``tool_config.qdrant`` *by reference* and rewrite them when tenancy is
    applied. If two scopes shared those objects, applying tenancy to one would
    silently retarget the other -- a cross-tenant data leak, not just a bug.
    """
    base = _config(tmp_path)
    a = base.for_tenancy("acme", "p1")
    b = base.for_tenancy("globex", "p2")

    for section in ("vector_store", "qdrant", "lancedb", "fuseki"):
        assert getattr(a.tool_config, section) is not getattr(b.tool_config, section)
        assert getattr(a.tool_config, section) is not getattr(base.tool_config, section)

    assert (
        a.tool_config.vector_store.facts_table != b.tool_config.vector_store.facts_table
    )


def test_for_tenancy_leaves_the_original_untouched(tmp_path) -> None:
    base = _config(tmp_path)
    before = base.tool_config.fuseki.dataset
    base.for_tenancy("acme", "p1")
    assert base.tool_config.fuseki.dataset == before


def test_for_tenancy_rejects_blank(tmp_path) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _config(tmp_path).for_tenancy("acme", "")


# -- runtime sharing -------------------------------------------------------


def test_runtime_delegates_are_readable_and_writable(tmp_path) -> None:
    """Existing code reads and substitutes these as ToolBox attributes."""
    tools = _toolbox(tmp_path)
    assert tools.converter is tools.runtime.converter
    assert tools.chunker is tools.runtime.chunker
    assert tools.embedding_tool is tools.runtime.embedding_tool

    replacement = EmbeddingBasedAggregator()
    tools.aggregator = replacement
    assert tools.runtime.aggregator is replacement


def test_scoped_toolboxes_share_one_runtime(tmp_path) -> None:
    """Sharing is the whole point: otherwise N scopes are N embedding models."""
    runtime = ToolBoxRuntime(_config(tmp_path), llm=STUB_LLM)
    a = ToolBox(_config(tmp_path).for_tenancy("acme", "p1"), runtime=runtime)
    b = ToolBox(_config(tmp_path).for_tenancy("globex", "p2"), runtime=runtime)

    assert a.embedding_tool is b.embedding_tool
    assert a.converter is b.converter
    assert a.llm is b.llm
    # ... while the partition-scoped halves stay separate.
    assert a.triple_store_manager is not b.triple_store_manager
    assert a.ontology_manager is not b.ontology_manager


def test_toolbox_new_still_supports_bare_attribute_use() -> None:
    """`ToolBox.__new__` is an established idiom in this suite."""
    tools = ToolBox.__new__(ToolBox)
    sentinel = EmbeddingBasedAggregator()
    tools.aggregator = sentinel
    assert tools.aggregator is sentinel


# -- registry --------------------------------------------------------------


@pytest.mark.anyio
async def test_registry_builds_and_caches_a_scope(tmp_path) -> None:
    base = _config(tmp_path)
    registry = ToolBoxRegistry(base, ToolBoxRuntime(base, llm=STUB_LLM))

    first = await registry.get(TenancyScope.build("acme", "p1"))
    second = await registry.get(TenancyScope.build("acme", "p1"))
    assert first is second
    assert len(registry) == 1


@pytest.mark.anyio
async def test_registry_isolates_scopes(tmp_path) -> None:
    base = _config(tmp_path)
    registry = ToolBoxRegistry(base, ToolBoxRuntime(base, llm=STUB_LLM))

    a = await registry.get(TenancyScope.build("acme", "p1"))
    b = await registry.get(TenancyScope.build("globex", "p2"))

    assert a is not b
    assert a.config is not b.config
    assert a.config.tool_config.fuseki.dataset == "acme--p1--facts"
    assert b.config.tool_config.fuseki.dataset == "globex--p2--facts"
    assert a.runtime is b.runtime


@pytest.mark.anyio
async def test_registry_evicts_least_recently_used_and_closes_it(
    tmp_path, monkeypatch
) -> None:
    base = _config(tmp_path)
    registry = ToolBoxRegistry(base, ToolBoxRuntime(base, llm=STUB_LLM), max_scopes=2)

    first = await registry.get(TenancyScope.build("a", "p"))
    await registry.get(TenancyScope.build("b", "p"))

    closed: list[bool] = []
    original = first.aclose

    async def record_close() -> None:
        closed.append(True)
        await original()

    monkeypatch.setattr(first, "aclose", record_close)

    await registry.get(TenancyScope.build("c", "p"))

    assert len(registry) == 2
    assert closed == [True], "evicted scope must be closed, not dropped"
    assert ("a", "p") not in registry.scopes


@pytest.mark.anyio
async def test_registry_touch_updates_recency(tmp_path) -> None:
    base = _config(tmp_path)
    registry = ToolBoxRegistry(base, ToolBoxRuntime(base, llm=STUB_LLM), max_scopes=2)

    await registry.get(TenancyScope.build("a", "p"))
    await registry.get(TenancyScope.build("b", "p"))
    await registry.get(TenancyScope.build("a", "p"))  # touch 'a'
    await registry.get(TenancyScope.build("c", "p"))  # evicts 'b', not 'a'

    assert ("a", "p") in registry.scopes
    assert ("b", "p") not in registry.scopes


@pytest.mark.anyio
async def test_concurrent_first_requests_build_one_toolbox(tmp_path) -> None:
    """Two requests racing for a cold scope must not each build it."""
    base = _config(tmp_path)
    registry = ToolBoxRegistry(base, ToolBoxRuntime(base, llm=STUB_LLM))
    scope = TenancyScope.build("acme", "p1")

    results = await asyncio.gather(*(registry.get(scope) for _ in range(5)))

    assert len({id(r) for r in results}) == 1
    assert len(registry) == 1


@pytest.mark.anyio
async def test_registry_rejects_a_zero_cap(tmp_path) -> None:
    base = _config(tmp_path)
    with pytest.raises(ValueError, match="at least 1"):
        ToolBoxRegistry(base, ToolBoxRuntime(base, llm=STUB_LLM), max_scopes=0)


@pytest.mark.anyio
async def test_graph_is_compiled_once_per_scope(tmp_path) -> None:
    base = _config(tmp_path)
    registry = ToolBoxRegistry(base, ToolBoxRuntime(base, llm=STUB_LLM))
    scope = TenancyScope.build("acme", "p1")

    calls: list[int] = []

    def build() -> object:
        calls.append(1)
        return object()

    first = registry.graph_for(scope, build)
    second = registry.graph_for(scope, build)

    assert first is second
    assert len(calls) == 1


# -- ToolBox.for_scope -----------------------------------------------------


@pytest.mark.anyio
async def test_for_scope_returns_self_for_the_active_partition(tmp_path) -> None:
    tools = _toolbox(tmp_path)
    await tools.update_tenancy_with_vector_mode(
        "acme", "p1", initialize_vector_store=False, fail_on_vector_store_error=False
    )
    assert await tools.for_scope("acme", "p1") is tools


@pytest.mark.anyio
async def test_for_scope_builds_a_registry_on_demand(tmp_path) -> None:
    """A single-tenant embedder should never allocate a registry."""
    tools = _toolbox(tmp_path)
    assert tools._registry is None

    other = await tools.for_scope("acme", "p1")

    assert other is not tools
    assert tools._registry is not None
    assert other.runtime is tools.runtime
    await tools.aclose()


@pytest.mark.anyio
async def test_aclose_closes_scoped_toolboxes(tmp_path) -> None:
    tools = _toolbox(tmp_path)
    await tools.for_scope("acme", "p1")
    assert len(tools.ensure_tenancy_registry()) == 1

    await tools.aclose()

    assert tools._registry is None
