"""Tenancy-independent tools shared across every :class:`~ontocast.toolbox.ToolBox`.

A ToolBox is bound to one tenant/project partition: its triple store, ontology
catalog and vector store all describe that partition. Serving several tenants
therefore means several ToolBoxes.

Most of what a ToolBox holds does not vary by tenant, though, and some of it is
expensive: :class:`~ontocast.tool.vector_store.embedding.EmbeddingTool` loads
model weights, :class:`~ontocast.tool.converter.ConverterTool` pulls docling,
and the LLM tool owns a provider client and the response cache. Duplicating
those per tenant would make a sixteen-scope registry sixteen copies of an
embedding model.

:class:`ToolBoxRuntime` holds exactly that shared half. ToolBox exposes every
one of its members as a property, so ``tools.llm`` and ``tools.converter`` mean
what they always did.
"""

from __future__ import annotations

import logging

from ontocast.config import Config, EmbeddingConfig, WebSearchProvider
from ontocast.onto.state import BudgetTracker
from ontocast.tool import AtomicToolBox, ChunkerTool, ConverterTool
from ontocast.tool.agg.aggregate import EmbeddingBasedAggregator
from ontocast.tool.agg.entity_aligner import EntityAligner
from ontocast.tool.cache import Cacher
from ontocast.tool.llm import LLMTool, _active_budget_tracker
from ontocast.tool.vector_store import EmbeddingTool, FastembedBm25SparseTool
from ontocast.tool.web_search import DuckDuckGoSearchProvider

logger = logging.getLogger(__name__)


class ToolBoxRuntime:
    """Shared, tenancy-independent tools.

    Satisfies :class:`~ontocast.tool.atomic.AtomicLLMProvider` so it can back
    the ``AtomicToolBox`` directly rather than routing through a ToolBox, which
    would tie the shared half to one scope.
    """

    def __init__(
        self,
        config: Config,
        *,
        llm: LLMTool | None = None,
        cache: Cacher | None = None,
        prune_cache: bool = True,
    ):
        """Build the shared tools.

        Args:
            config: Configuration to read tool settings from. Only
                tenancy-independent sections are consulted.
            llm: Pre-built LLM tool. Supply one from
                :meth:`~ontocast.toolbox.ToolBox.acreate`; otherwise
                ``LLMTool.create`` runs, which cannot be called inside a running
                event loop.
            cache: Pre-built shared cache. Supply the same instance that
                ``llm`` was built with; otherwise the two would end up with
                separate ``Cacher`` objects and separate prune counters.
            prune_cache: Bound the cache on construction. :meth:`acreate` sets
                this False because it has already pruned off the event loop.
        """
        tool_config = config.get_tool_config()

        self.shared_cache = cache or Cacher(config=config)
        # Bound the cache once per process start, so short CLI runs are covered
        # too; steady-state trimming rides on the write counter inside Cacher.
        if prune_cache:
            self.shared_cache.prune()
        self.llm_provider = tool_config.llm_config.provider
        self.llm: LLMTool = llm or LLMTool.create(
            config=tool_config.llm_config, cache=self.shared_cache
        )

        self.search_provider = None
        if tool_config.web_search.enabled:
            if tool_config.web_search.provider == WebSearchProvider.DUCKDUCKGO:
                self.search_provider = DuckDuckGoSearchProvider(
                    timeout_seconds=tool_config.web_search.timeout_seconds,
                    region=tool_config.web_search.region,
                    safesearch=tool_config.web_search.safesearch,
                )
            else:
                raise ValueError(
                    f"Unsupported web-search provider: {tool_config.web_search.provider}"
                )

        self.atomic_tools = AtomicToolBox(
            llm_provider=self,
            search_provider=self.search_provider,
            web_search_config=tool_config.web_search,
            facts_validation_config=tool_config.facts_validation,
            citation_vocabulary=tool_config.chunk_config.citation_vocabulary,
        )

        self.converter: ConverterTool = ConverterTool(
            cache=self.shared_cache,
            converter_config=tool_config.converter_config,
        )
        self.chunker: ChunkerTool = ChunkerTool(
            chunk_config=tool_config.chunk_config, cache=self.shared_cache
        )
        self.aggregator: EmbeddingBasedAggregator = EmbeddingBasedAggregator(
            tool_config.aggregation
        )
        self.embedding_tool: EmbeddingTool = EmbeddingTool.create(tool_config.embedding)
        # Built lazily and kept here rather than on each ToolBox: the BM25 model
        # is tenancy-independent, so a per-scope instance meant one ONNX model
        # per resident tenancy scope.
        self._sparse_embedding_tool: FastembedBm25SparseTool | None = None
        self.entity_aligners: dict[tuple[str, float], EntityAligner] = {}

    def sparse_embedding_tool(self, config: EmbeddingConfig) -> FastembedBm25SparseTool:
        """Shared BM25 sparse encoder for the external vector backends.

        Args:
            config: Embedding config; only consulted on first construction,
                since the encoder is tenancy-independent.

        Returns:
            FastembedBm25SparseTool: The process-shared sparse encoder.
        """
        # __dict__ rather than attribute access: tests build a runtime via
        # __new__ to exercise one tool, so __init__ may not have run.
        tool = self.__dict__.get("_sparse_embedding_tool")
        if tool is None:
            tool = FastembedBm25SparseTool(config=config)
            self.__dict__["_sparse_embedding_tool"] = tool
        return tool

    @classmethod
    async def acreate(cls, config: Config) -> "ToolBoxRuntime":
        """Build the shared tools from inside a running event loop."""
        cache = Cacher(config=config)
        await cache.aprune()
        llm = await LLMTool.acreate(
            config=config.get_tool_config().llm_config, cache=cache
        )
        return cls(config, llm=llm, cache=cache, prune_cache=False)

    async def get_llm_tool(self, budget_tracker: BudgetTracker) -> LLMTool:
        """Return the shared LLM tool, charging usage to ``budget_tracker``.

        The tracker is bound to the *calling task* rather than to the shared tool
        instance. Assigning it to the instance -- as this did once -- meant that
        with ``PARALLEL_WORKERS`` unit workers in flight, whichever bound last
        collected every concurrent call's usage; document totals still summed
        correctly, but per-unit attribution was arbitrary.

        Args:
            budget_tracker: The budget tracker to charge for this task's calls.

        Returns:
            LLMTool: The shared LLM tool.
        """
        _active_budget_tracker.set(budget_tracker)
        return self.llm

    def get_entity_aligner(
        self,
        embedding_model: str,
        similarity_threshold: float,
    ) -> EntityAligner:
        """Return a cached entity aligner for the given embedding settings."""
        cache_key = (embedding_model, similarity_threshold)
        aligner = self.entity_aligners.get(cache_key)
        if aligner is None:
            aligner = EntityAligner(
                embedding_model=embedding_model,
                similarity_threshold=similarity_threshold,
            )
            self.entity_aligners[cache_key] = aligner
        return aligner
