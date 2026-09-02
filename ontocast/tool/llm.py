"""Language Model (LLM) integration tool for OntoCast.

This module provides integration with various language models through LangChain,
supporting OpenAI, Ollama, Anthropic (Claude), and Google (Gemini) providers.
It enables text generation and
structured data extraction capabilities with optional caching support.

Cache Usage:
    The LLM tool supports caching of responses to avoid redundant API calls.
    Caching uses a shared Cacher instance that manages cache directories for all tools.
    The cache directory is managed by the shared Cacher class and follows these rules:

    ```python
    from ontocast.tool.llm import LLMTool
    from ontocast.config import LLMConfig
    from ontocast.tool.cache import Cacher

    # Create shared cache instance
    shared_cache = Cacher()

    # Create LLM tool with shared cache
    llm_tool = await LLMTool.acreate(
        config=LLMConfig(...),
        cache=shared_cache
    )
    ```

    Default cache locations:
    - Tests: .test_cache/llm/ in the current working directory
    - Windows: %USERPROFILE%\\AppData\\Local\\ontocast\\llm\
    - Unix/Linux: ~/.cache/ontocast/llm/ (or $XDG_CACHE_HOME/ontocast/llm/)

    Cache files are stored as JSON files with filenames based on SHA256 hashes
    of the prompt and LLM configuration. This ensures that identical prompts
    with the same configuration will return cached responses.

    The shared Cacher automatically manages subdirectories for different tools,
    ensuring organized cache storage while maintaining a single cache instance.
"""

from __future__ import annotations

import asyncio
import logging
import time
import weakref
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Type, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages.ai import AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, PrivateAttr, SecretStr

from ontocast.config import LLMConfig, LLMProvider
from ontocast.onto.token_usage import TokenUsage
from ontocast.util.loop import require_no_running_loop
from ontocast.util.optional import require

from .cache import LLM_CACHE_SUBDIR, Cacher, ToolCacher
from .onto import Tool

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)

# Bumped whenever the cache entry shape or the set of key inputs changes, so
# stale entries miss instead of being deserialised under the wrong assumptions.
# Version 2 added the Ollama generation knobs to the key.
LLM_CACHE_FORMAT_VERSION = 2

# Shared across all LLMTool instances with the same max_inflight setting, but
# partitioned per event loop: asyncio.Semaphore binds to the running loop the
# first time it has to wait, so a single process-wide instance breaks the second
# ``asyncio.run`` in a process as soon as it sees contention. Keyed weakly on the
# loop itself rather than on id(), which is recycled once a loop is collected.
_inflight_semaphores: "weakref.WeakKeyDictionary[Any, dict[int, asyncio.Semaphore]]" = (
    weakref.WeakKeyDictionary()
)

# The budget tracker usage should be charged to, scoped to the running task.
#
# The LLM tool is a singleton owned by the ToolBox, so binding a per-unit
# tracker to an instance attribute meant concurrent unit workers overwrote each
# other: whichever bound last collected every in-flight call's usage. Document
# totals still summed correctly (the reduce merges all per-unit trackers) but
# per-unit attribution was arbitrary, and it is reported to API clients.
#
# A ContextVar is task-local -- asyncio.gather copies the current context into
# each task -- so parallel units no longer share one slot.
_active_budget_tracker: ContextVar[Any | None] = ContextVar(
    "ontocast_active_budget_tracker", default=None
)


class LLMRequestTimeoutError(RuntimeError):
    """A provider call exceeded ``LLM_REQUEST_TIMEOUT_SECONDS``.

    Deliberately not an :class:`asyncio.TimeoutError`: the unit loops catch
    ``Exception`` to fail a single unit gracefully, and a cancellation-flavoured
    error escaping ``asyncio.gather`` would take the whole fan-out down with it.
    """


@contextmanager
def use_budget_tracker(budget_tracker: Any):
    """Charge LLM usage inside this block to ``budget_tracker``."""
    token = _active_budget_tracker.set(budget_tracker)
    try:
        yield
    finally:
        _active_budget_tracker.reset(token)


def record_active_span(name: str, seconds: float) -> None:
    """Charge a latency span to the running task's budget tracker, if any.

    A no-op when no tracker is bound. This reads the context variable directly
    rather than going through an :class:`LLMTool`, so stages that fan out
    *around* the LLM (e.g. chunk section classification) can report queue waits
    without holding a real tool instance -- which also keeps test stubs that
    substitute a plain callable for the LLM working.

    Args:
        name: Duration key, e.g. ``"chunk section classify/worker_wait"``.
        seconds: Elapsed seconds to accumulate.
    """
    bt = _active_budget_tracker.get()
    if bt is not None:
        bt.add_duration(name, seconds)


def record_active_count(name: str, n: int = 1) -> None:
    """Charge a named event count to the running task's budget tracker, if any.

    The counting sibling of :func:`record_active_span`, and a no-op when no
    tracker is bound -- so the parse layer can report how often it repaired or
    abandoned a response without holding an :class:`LLMTool`, and test stubs
    that substitute a plain callable keep working.

    Args:
        name: Counter key, e.g. ``"llm/json_bracket_repair"``.
        n: Amount to add.
    """
    bt = _active_budget_tracker.get()
    if bt is not None:
        bt.incr(name, n)


def llm_cache_config(
    config: LLMConfig, **extra: Any
) -> dict[str, str | int | float | bool | None]:
    """Cache-key inputs for a given LLM configuration.

    Every field here changes the provider's response, so it must take part in
    the key. This is the single definition: :class:`LLMTool` and the batch
    import in :mod:`ontocast.tool.llm_batch` both call it, and any divergence
    between them silently produces entries that are written but never read.

    Args:
        config: The LLM configuration a response would be produced under.
        **extra: Additional discriminators (e.g. an output schema name).

    Returns:
        dict: JSON-serialisable mapping used as the cache key's config part.
    """
    config_dict: dict[str, str | int | float | bool | None] = {
        "cache_format_version": LLM_CACHE_FORMAT_VERSION,
        "provider": config.provider,
        "model_name": config.model_name,
        "temperature": config.temperature,
        "base_url": config.base_url,
        # Ollama generation knobs: these bound reasoning and output length, so
        # the same prompt under a different num_ctx is a different response.
        "think": config.think,
        "num_predict": config.num_predict,
        "num_ctx": config.num_ctx,
    }
    config_dict.update(extra)
    return config_dict


def _inflight_semaphore(max_inflight: int) -> asyncio.Semaphore:
    """Return the in-flight limiter for ``max_inflight`` on the running loop."""
    loop = asyncio.get_running_loop()
    per_loop = _inflight_semaphores.setdefault(loop, {})
    if max_inflight not in per_loop:
        per_loop[max_inflight] = asyncio.Semaphore(max_inflight)
    return per_loop[max_inflight]


#: Exception class names the providers raise on throttling. Matched by name
#: so no provider SDK is imported here: openai.RateLimitError, anthropic's
#: RateLimitError, and Google's ResourceExhausted all identify themselves.
_RATE_LIMIT_ERROR_NAMES = ("RateLimitError", "ResourceExhausted")


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Whether an exception (or its cause chain) is a provider throttle."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ in _RATE_LIMIT_ERROR_NAMES:
            return True
        if "429" in str(current) and "rate" in str(current).lower():
            return True
        current = current.__cause__ or current.__context__
    return False


def _opt_int(source: Any, key: str) -> int | None:
    """Read ``key`` from a mapping as an int, or None when absent/unusable."""
    if not isinstance(source, dict):
        return None
    value = source.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _usage_from_llm_result(result: Any) -> TokenUsage:
    """Extract token usage from an LLM response when the provider reports it.

    Two tiers, because providers disagree: LangChain normalises everything it
    can into ``usage_metadata``, but OpenAI-compatible endpoints that predate
    (or ignore) that convention only populate ``response_metadata["token_usage"]``.
    An all-``None`` :class:`TokenUsage` means the provider said nothing.
    """
    if not isinstance(result, AIMessage):
        return TokenUsage()

    usage = result.usage_metadata
    if usage is not None:
        input_tokens = _opt_int(usage, "input_tokens")
        output_tokens = _opt_int(usage, "output_tokens")
        if input_tokens is not None and output_tokens is not None:
            input_details = usage.get("input_token_details")
            output_details = usage.get("output_token_details")
            return TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=_opt_int(output_details, "reasoning"),
                cache_read_input_tokens=_opt_int(input_details, "cache_read"),
                cache_creation_input_tokens=_opt_int(input_details, "cache_creation"),
            )

    return token_usage_from_openai_payload(
        (result.response_metadata or {}).get("token_usage")
    )


def token_usage_from_openai_payload(payload: Any) -> TokenUsage:
    """Parse an OpenAI-shaped ``usage`` object into a :class:`TokenUsage`.

    Shared with the Batch-API prefill in :mod:`ontocast.tool.llm_batch`, whose
    JSONL carries the same object under ``response.body.usage`` -- so a
    prewarmed cache entry accounts for tokens exactly like a live one.
    """
    prompt_tokens = _opt_int(payload, "prompt_tokens")
    completion_tokens = _opt_int(payload, "completion_tokens")
    if prompt_tokens is None or completion_tokens is None:
        return TokenUsage()
    completion_details = payload.get("completion_tokens_details")
    prompt_details = payload.get("prompt_tokens_details")
    return TokenUsage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        reasoning_tokens=_opt_int(completion_details, "reasoning_tokens"),
        cache_read_input_tokens=_opt_int(prompt_details, "cached_tokens"),
    )


def _usage_metadata_from(usage: TokenUsage) -> dict[str, Any] | None:
    """Render a :class:`TokenUsage` back into LangChain's ``usage_metadata`` shape.

    Replayed onto a cached ``AIMessage`` so a hit stays behaviourally identical
    to a fresh call for anything reading usage off the message -- notably
    LangChain-native tracers.
    """
    if usage.input_tokens is None or usage.output_tokens is None:
        return None
    metadata: dict[str, Any] = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.input_tokens + usage.output_tokens,
    }
    input_details = {
        key: value
        for key, value in (
            ("cache_read", usage.cache_read_input_tokens),
            ("cache_creation", usage.cache_creation_input_tokens),
        )
        if value is not None
    }
    if input_details:
        metadata["input_token_details"] = input_details
    if usage.reasoning_tokens is not None:
        metadata["output_token_details"] = {"reasoning": usage.reasoning_tokens}
    return metadata


def _chars_received_from_result(result: Any) -> int:
    if isinstance(result, AIMessage) and result.content:
        return len(result.content)
    return len(str(result))


def _content_to_str(content: Any) -> str:
    """Normalise an LLM response content value to a plain string.

    Some providers (Google Gemini, Anthropic) return a list of typed content
    blocks instead of a bare string, e.g.:
        [{'type': 'text', 'text': '...', ...}, ...]
    This function extracts and concatenates all ``text`` blocks so that
    downstream string-based parsers always receive a plain string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        if parts:
            return "".join(parts)
    return str(content)


class CachedResponse(BaseModel):
    """A stored LLM response.

    ``cache_format_version`` in the key guarantees entries were written by this
    version of the code, so the shape is known rather than sniffed.
    """

    content: str = Field(description="Response text, already normalised.")
    prompt: str = Field(default="", description="Prompt that produced it.")
    response_metadata: dict[str, Any] = Field(
        default_factory=dict,
        # Replaying this keeps a cache hit behaviourally identical to a fresh
        # call; without it a caller inspecting finish_reason would silently
        # branch differently on a cached run.
        description="Provider metadata, replayed on a hit.",
    )
    kwargs: dict[str, Any] = Field(default_factory=dict, description="Invoke kwargs.")
    usage: TokenUsage | None = Field(
        default=None,
        # Optional rather than a cache_format_version bump: it is purely
        # additive, and bumping would evict every existing entry and force a
        # paid re-run before any cache replay works again. Entries written
        # before this field report usage as unknown, which is the truth.
        #
        # usage_metadata is a separate AIMessage attribute, not part of
        # response_metadata, so persisting the latter never captured it -- which
        # is why replayed runs used to report zero tokens.
        description="Token counts, replayed on a hit. None for older entries.",
    )


class LLMTool(Tool):
    """Tool for interacting with language models.

    This class provides a unified interface for working with different language model
    providers (OpenAI, Ollama, Anthropic, Google) through LangChain. It supports both
    synchronous and
    asynchronous operations.

    Attributes:
        config: LLMConfig object containing all LLM settings.
        cache: Cacher instance for caching LLM responses.
    """

    config: LLMConfig = Field(default_factory=LLMConfig)
    cache: Any = Field(default=None, exclude=True)
    budget_tracker: Any = Field(default=None, exclude=True)
    _cache_hits: int = PrivateAttr(default=0)
    _cache_misses: int = PrivateAttr(default=0)

    def __init__(
        self,
        cache: Cacher | None = None,
        budget_tracker: Any = None,
        **kwargs: Any,
    ):
        """Initialize the LLM tool.

        Args:
            cache: Optional shared Cacher instance. If None, creates a new one.
            budget_tracker: Optional budget tracker instance for usage statistics.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._llm = None
        self.budget_tracker = budget_tracker

        # Initialize cache - use shared cacher or create new one
        if cache is not None:
            self.cache = ToolCacher(cache, LLM_CACHE_SUBDIR)
        else:
            # Standalone use (CLI helpers, direct library use): fall back to a
            # private Cacher on the configured/default directory.
            shared_cache = Cacher()
            self.cache = ToolCacher(shared_cache, LLM_CACHE_SUBDIR)

    @classmethod
    def create(
        cls,
        config: LLMConfig,
        cache: Cacher | None = None,
        budget_tracker: Any = None,
        **kwargs: Any,
    ) -> "LLMTool":
        """Create a new LLM tool instance synchronously.

        Args:
            config: LLMConfig object containing LLM settings.
            cache: Optional shared Cacher instance.
            budget_tracker: Optional budget tracker instance for usage statistics.
            **kwargs: Additional keyword arguments for initialization.

        Returns:
            LLMTool: A new instance of the LLM tool.

        Raises:
            RuntimeError: If called from inside a running event loop; use
                :meth:`acreate` there.
        """
        require_no_running_loop("LLMTool.create", "LLMTool.acreate")
        return asyncio.run(
            cls.acreate(
                config=config, cache=cache, budget_tracker=budget_tracker, **kwargs
            )
        )

    @classmethod
    async def acreate(
        cls,
        config: LLMConfig,
        cache: Cacher | None = None,
        budget_tracker: Any = None,
        **kwargs: Any,
    ) -> "LLMTool":
        """Create a new LLM tool instance asynchronously.

        Args:
            config: LLMConfig object containing LLM settings.
            cache: Optional shared Cacher instance.
            budget_tracker: Optional budget tracker instance for usage statistics.
            **kwargs: Additional keyword arguments for initialization.

        Returns:
            LLMTool: A new instance of the LLM tool.
        """
        # Create and initialize the instance with the config
        self = cls(config=config, cache=cache, budget_tracker=budget_tracker, **kwargs)
        await self.setup()
        return self

    async def setup(self):
        """Set up the language model based on the configured provider.

        Raises:
            ValueError: If the provider is not supported.
        """
        # Cross-provider pacing and retry kwargs. The rate limiter is a
        # per-process token bucket on request *starts* (langchain-core
        # InMemoryRateLimiter): the inflight semaphore caps concurrency, this
        # paces the sustained rate underneath it -- set it from the provider
        # tier. `max_retries` tunes the provider SDK's own 429/backoff
        # retries; there is deliberately no retry loop at this layer (see
        # agent/common.py -- retrying here multiplies request rate exactly
        # when the provider asks for less).
        pacing_kwargs: dict[str, Any] = {}
        if self.config.requests_per_second is not None:
            from langchain_core.rate_limiters import InMemoryRateLimiter

            pacing_kwargs["rate_limiter"] = InMemoryRateLimiter(
                requests_per_second=self.config.requests_per_second,
                check_every_n_seconds=0.1,
                max_bucket_size=max(1.0, self.config.requests_per_second),
            )
        retry_kwargs: dict[str, Any] = {}
        if self.config.max_retries is not None:
            retry_kwargs["max_retries"] = self.config.max_retries

        if self.config.provider == LLMProvider.OPENAI:
            if self.config.model_name.startswith("gpt-5"):
                self.config.temperature = 1.0
                logger.warning(
                    f"Setting temperature to {self.config.temperature} for gpt-5 class "
                    f"model {self.config.model_name}"
                )
            ChatOpenAI = require(
                "langchain_openai", feature="The OpenAI LLM provider"
            ).ChatOpenAI
            openai_kwargs: dict[str, Any] = {}
            if self.config.json_mode:
                # Constrains decoding to valid JSON at the provider, so a
                # truncated or bracket-swapped envelope cannot be produced in
                # the first place. Requires the word "JSON" in the prompt --
                # test_prompt_json_mode_precondition holds the prompt set to
                # that.
                openai_kwargs["response_format"] = {"type": "json_object"}
            self._llm = ChatOpenAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                base_url=self.config.base_url,
                api_key=(
                    SecretStr(self.config.api_key) if self.config.api_key else None
                ),
                model_kwargs=openai_kwargs,
                **pacing_kwargs,
                **retry_kwargs,
            )
        elif self.config.provider == LLMProvider.OLLAMA:
            ollama_kwargs: dict[str, Any] = {
                "model": self.config.model_name,
                "base_url": self.config.base_url,
                "temperature": self.config.temperature,
            }
            if self.config.think is not None:
                ollama_kwargs["reasoning"] = self.config.think
            if self.config.num_predict is not None:
                ollama_kwargs["num_predict"] = self.config.num_predict
            if self.config.num_ctx is not None:
                ollama_kwargs["num_ctx"] = self.config.num_ctx
            ChatOllama = require(
                "langchain_ollama", feature="The Ollama LLM provider"
            ).ChatOllama
            self._llm = ChatOllama(**ollama_kwargs, **pacing_kwargs)
        elif self.config.provider == LLMProvider.ANTHROPIC:
            anthropic_kwargs: dict[str, Any] = {
                "model": self.config.model_name,
                "temperature": self.config.temperature,
            }
            if self.config.api_key:
                anthropic_kwargs["anthropic_api_key"] = SecretStr(self.config.api_key)
            if self.config.base_url:
                anthropic_kwargs["anthropic_api_url"] = self.config.base_url
            ChatAnthropic = require(
                "langchain_anthropic", feature="The Anthropic LLM provider"
            ).ChatAnthropic
            self._llm = ChatAnthropic(
                **anthropic_kwargs, **pacing_kwargs, **retry_kwargs
            )
        elif self.config.provider == LLMProvider.GOOGLE:
            ChatGoogleGenerativeAI = require(
                "langchain_google_genai", feature="The Google LLM provider"
            ).ChatGoogleGenerativeAI
            self._llm = ChatGoogleGenerativeAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                google_api_key=self.config.api_key,
                **pacing_kwargs,
                **retry_kwargs,
            )
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    def _cache_config_dict(self, **extra: Any) -> dict[str, Any]:
        """Cache-key config for this tool's settings; see :func:`llm_cache_config`."""
        return dict(llm_cache_config(self.config, **extra))

    def _cache_key_content(self, *args: Any) -> str:
        """Stable string for disk cache keys from invoke arguments."""
        if not args:
            return ""
        primary = self._prompt_to_string(args[0])
        if len(args) == 1:
            return primary
        extra = [self._prompt_to_string(arg) for arg in args[1:]]
        return primary + "\n---\n" + "\n---\n".join(extra)

    def _current_budget_tracker(self) -> Any:
        """Tracker for the running task, falling back to the instance default.

        The context-local tracker wins so parallel unit workers charge their own
        budgets; ``self.budget_tracker`` remains for direct library use of a
        single ``LLMTool``.
        """
        scoped = _active_budget_tracker.get()
        return scoped if scoped is not None else self.budget_tracker

    def _record_cache_hit(
        self, prompt_str: str, content_str: str, usage: TokenUsage | None
    ) -> None:
        self._cache_hits += 1
        bt = self._current_budget_tracker()
        if bt is not None:
            bt.add_cache_hit(len(prompt_str), len(content_str), usage=usage)

    def record_span(self, name: str, seconds: float) -> None:
        """Charge a latency span to this call's budget tracker.

        Uses the same context-local tracker as usage accounting, so per-unit
        attribution under ``asyncio.gather`` is correct for free, and falls back
        to this tool's own tracker for direct library use. Callers without an
        :class:`LLMTool` instance should use :func:`record_active_span`.

        Args:
            name: Duration key, e.g. ``"llm/provider"``.
            seconds: Elapsed seconds to accumulate.
        """
        bt = self._current_budget_tracker()
        if bt is not None:
            bt.add_duration(name, seconds)

    def _record_api_usage(self, prompt_str: str, result: Any) -> None:
        self._cache_misses += 1
        bt = self._current_budget_tracker()
        if bt is None:
            return
        bt.add_usage(
            len(prompt_str),
            _chars_received_from_result(result),
            usage=_usage_from_llm_result(result),
        )

    def get_cache_stats(
        self, include_disk: bool = True
    ) -> dict[str, int | dict[str, int | dict[str, int] | dict[str, dict[str, int]]]]:
        """Return in-memory hit/miss counters and, optionally, on-disk file stats.

        Args:
            include_disk: Whether to walk the cache directory. The walk stats
                every file, so callers on a hot path (or on an event loop)
                should pass False or use :meth:`aget_cache_stats`.
        """
        stats: dict[
            str, int | dict[str, int | dict[str, int] | dict[str, dict[str, int]]]
        ] = {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
        }
        if include_disk:
            stats["disk"] = self.cache.get_cache_stats()
        return stats

    async def aget_cache_stats(
        self,
    ) -> dict[str, int | dict[str, int | dict[str, int] | dict[str, dict[str, int]]]]:
        """Async :meth:`get_cache_stats`, with the directory walk off the loop."""
        stats = self.get_cache_stats(include_disk=False)
        stats["disk"] = await asyncio.to_thread(self.cache.get_cache_stats)
        return stats

    async def _invoke_cached(
        self,
        *args: Any,
        cache_config_extra: dict[str, Any] | None = None,
        **kwds: Any,
    ) -> AIMessage:
        """Invoke the LLM with optional disk cache and global in-flight limiting.

        This is the single cache-aware entry point; :meth:`__call__`,
        :meth:`acall`, :meth:`complete`, and :meth:`extract` all route through
        it so that content normalisation, key construction, budget accounting,
        and in-flight limiting cannot drift apart between them.

        Args:
            *args: Positional arguments forwarded to the provider's ``ainvoke``.
                The first is treated as the prompt for keying and accounting.
            cache_config_extra: Extra cache-key discriminators beyond the LLM
                config (e.g. the structured-output schema name).
            **kwds: Keyword arguments forwarded to ``ainvoke`` and folded into
                the cache key.

        Returns:
            AIMessage: Response with content normalised to a plain string.
        """
        prompt_key = self._cache_key_content(*args)
        prompt_str = self._prompt_to_string(args[0]) if args else ""
        config_dict = self._cache_config_dict(**(cache_config_extra or {}))

        if self.config.cache_enabled:
            lookup_start = time.perf_counter()
            cached_response = await self.cache.aget(
                prompt_key, config=config_dict, **kwds
            )
            self.record_span("llm/cache_lookup", time.perf_counter() - lookup_start)
            if cached_response is not None:
                logger.debug("Cache hit: %s...", prompt_str[:50])
                entry = CachedResponse.model_validate(cached_response)
                self._record_cache_hit(prompt_str, entry.content, entry.usage)
                return AIMessage(
                    content=entry.content,
                    response_metadata=entry.response_metadata,
                    usage_metadata=(
                        _usage_metadata_from(entry.usage)
                        if entry.usage is not None
                        else None
                    ),
                )

        logger.debug("Cache miss, calling LLM: %s...", prompt_str[:50])

        # Three spans, because they have three different fixes: queueing behind
        # llm_max_inflight wants a higher cap, provider time wants a faster
        # model or fewer calls, and neither is visible in the node's wall clock.
        max_inflight = max(1, self.config.llm_max_inflight)
        wait_start = time.perf_counter()
        async with _inflight_semaphore(max_inflight):
            provider_start = time.perf_counter()
            self.record_span("llm/inflight_wait", provider_start - wait_start)
            timeout = self.config.request_timeout_seconds
            try:
                if timeout is None:
                    response = await self.llm.ainvoke(*args, **kwds)
                else:
                    response = await asyncio.wait_for(
                        self.llm.ainvoke(*args, **kwds), timeout=timeout
                    )
            except asyncio.TimeoutError as exc:
                bt = self._current_budget_tracker()
                if bt is not None:
                    bt.incr("llm/timeouts")
                # Re-raised as a plain error so the unit loop's handler treats
                # it as a failed render rather than a cancellation: letting a
                # bare TimeoutError escape asyncio.gather would abort the whole
                # fan-out and orphan its siblings.
                raise LLMRequestTimeoutError(
                    f"LLM request exceeded {timeout}s "
                    f"({self.config.provider}/{self.config.model_name})"
                ) from exc
            except Exception as exc:
                # A provider throttle that survived the SDK's own retries
                # surfaces as a failed render; without a counter it is
                # indistinguishable from a model failure in the telemetry,
                # which is how a throttled arm once read as a quality
                # regression. Detected by exception shape rather than type so
                # no provider SDK is imported here. Re-raised unchanged --
                # this layer deliberately does not retry (see
                # agent/common.py): raise LLM_MAX_RETRIES or lower
                # LLM_REQUESTS_PER_SECOND instead.
                if _is_rate_limit_error(exc):
                    bt = self._current_budget_tracker()
                    if bt is not None:
                        bt.incr("llm/rate_limited")
                    logger.warning(
                        "Provider rate limit hit (%s/%s): %s -- pace with "
                        "LLM_REQUESTS_PER_SECOND / LLM_MAX_INFLIGHT, or raise "
                        "LLM_MAX_RETRIES",
                        self.config.provider,
                        self.config.model_name,
                        exc,
                    )
                raise
            finally:
                self.record_span("llm/provider", time.perf_counter() - provider_start)

        bt = self._current_budget_tracker()
        if bt is not None:
            bt.incr("llm/calls_timed")
        self._record_api_usage(prompt_str, response)

        content_str = _content_to_str(response.content)
        response_metadata = getattr(response, "response_metadata", {}) or {}
        usage = _usage_from_llm_result(response)
        if self.config.cache_enabled and not self.config.cache_read_only:
            entry = CachedResponse(
                content=content_str,
                prompt=prompt_str,
                response_metadata=response_metadata,
                kwargs=kwds,
                usage=None if usage.is_empty() else usage,
            )
            await self.cache.aset(
                prompt_key, entry.model_dump(), config=config_dict, **kwds
            )

        return AIMessage(
            content=content_str,
            response_metadata=response_metadata,
            usage_metadata=_usage_metadata_from(usage),
        )

    async def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Call the language model directly (asynchronous)."""
        return await self._invoke_cached(*args, **kwds)

    async def acall(self, *args: Any, **kwds: Any) -> Any:
        """Alias for :meth:`__call__`."""
        return await self._invoke_cached(*args, **kwds)

    @property
    def llm(self) -> BaseChatModel:
        """Get the underlying language model instance.

        Returns:
            BaseChatModel: The configured language model.

        Raises:
            RuntimeError: If the LLM has not been properly initialized.
        """
        if self._llm is None:
            raise RuntimeError(
                "LLM resource not properly initialized. Call setup() first."
            )
        return self._llm

    def _prompt_to_string(self, prompt) -> str:
        """Convert various prompt types to string for caching.

        Args:
            prompt: The prompt object (string, StringPromptValue, etc.)

        Returns:
            str: String representation of the prompt.
        """
        if isinstance(prompt, str):
            return prompt
        to_string = getattr(prompt, "to_string", None)
        if callable(to_string):
            return str(to_string())
        text_attr = getattr(prompt, "text", None)
        if isinstance(text_attr, str):
            return text_attr
        content_attr = getattr(prompt, "content", None)
        if content_attr is not None:
            return str(content_attr)
        return str(prompt)

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        """Generate a completion for the given prompt.

        Args:
            prompt: The prompt to complete.
            **kwargs: Forwarded to the provider and folded into the cache key.

        Returns:
            str: The response text, normalised from provider content blocks.
        """
        response = await self._invoke_cached(prompt, **kwargs)
        return _content_to_str(response.content)

    async def extract(self, prompt: str, output_schema: Type[T], **kwargs: Any) -> T:
        """Extract structured data from the prompt according to a schema.

        Args:
            prompt: The prompt describing what to extract.
            output_schema: Pydantic model the response is parsed into.
            **kwargs: Forwarded to the provider and folded into the cache key.

        Returns:
            T: The parsed model instance.
        """
        parser = PydanticOutputParser(pydantic_object=output_schema)
        format_instructions = parser.get_format_instructions()

        # The format instructions embed the full JSON schema, so schema changes
        # already alter the key; the name is carried as an explicit
        # discriminator so entries stay attributable when inspected on disk.
        full_prompt = f"{prompt}\n\n{format_instructions}"
        response = await self._invoke_cached(
            full_prompt,
            cache_config_extra={"output_schema": output_schema.__name__},
            **kwargs,
        )
        return parser.parse(_content_to_str(response.content))
