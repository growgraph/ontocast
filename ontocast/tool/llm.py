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
from functools import wraps
from typing import Any, Callable, Type, TypeVar

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages.ai import AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, PrivateAttr, SecretStr

from ontocast.config import LLMConfig, LLMProvider

from .cache import Cacher, ToolCacher
from .onto import Tool

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)

# Shared across all LLMTool instances with the same max_inflight setting.
_inflight_semaphores: dict[int, asyncio.Semaphore] = {}


def _inflight_semaphore(max_inflight: int) -> asyncio.Semaphore:
    if max_inflight not in _inflight_semaphores:
        _inflight_semaphores[max_inflight] = asyncio.Semaphore(max_inflight)
    return _inflight_semaphores[max_inflight]


def _usage_from_llm_result(result: Any) -> tuple[int | None, int | None]:
    """Extract token usage from an LLM response when the provider reports it."""
    if isinstance(result, AIMessage):
        usage = result.usage_metadata
        if usage is not None:
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            if input_tokens is not None and output_tokens is not None:
                return int(input_tokens), int(output_tokens)

        response_metadata = result.response_metadata or {}
        token_usage = response_metadata.get("token_usage")
        if isinstance(token_usage, dict):
            prompt_tokens = token_usage.get("prompt_tokens")
            completion_tokens = token_usage.get("completion_tokens")
            if prompt_tokens is not None and completion_tokens is not None:
                return int(prompt_tokens), int(completion_tokens)

    return None, None


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


def track_llm_usage(func: Callable) -> Callable:
    """Decorator to track LLM usage for methods that always hit the provider."""

    def _record_usage(tool: LLMTool, prompt_str: str, result: Any) -> None:
        bt = tool.budget_tracker
        if bt is None:
            return
        input_tokens, output_tokens = _usage_from_llm_result(result)
        bt.add_usage(
            len(prompt_str),
            _chars_received_from_result(result),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    @wraps(func)
    async def async_wrapper(self: LLMTool, *args, **kwargs):
        prompt = args[0] if args else ""
        prompt_str = self._prompt_to_string(prompt)
        result = await func(self, *args, **kwargs)
        _record_usage(self, prompt_str, result)
        return result

    return async_wrapper


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
        **kwargs,
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
            self.cache = ToolCacher(cache, "llm")
        else:
            # Fallback for backward compatibility
            shared_cache = Cacher()
            self.cache = ToolCacher(shared_cache, "llm")

    @classmethod
    def create(
        cls,
        config: LLMConfig,
        cache: Cacher | None = None,
        budget_tracker: Any = None,
        **kwargs,
    ):
        """Create a new LLM tool instance synchronously.

        Args:
            config: LLMConfig object containing LLM settings.
            cache: Optional shared Cacher instance.
            budget_tracker: Optional budget tracker instance for usage statistics.
            **kwargs: Additional keyword arguments for initialization.

        Returns:
            LLMTool: A new instance of the LLM tool.
        """
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
        **kwargs,
    ):
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
        if self.config.provider == LLMProvider.OPENAI:
            if self.config.model_name.startswith("gpt-5"):
                self.config.temperature = 1.0
                logger.warning(
                    f"Setting temperature to {self.config.temperature} for gpt-5 class "
                    f"model {self.config.model_name}"
                )
            self._llm = ChatOpenAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                base_url=self.config.base_url,
                api_key=(
                    SecretStr(self.config.api_key) if self.config.api_key else None
                ),
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
            self._llm = ChatOllama(**ollama_kwargs)
        elif self.config.provider == LLMProvider.ANTHROPIC:
            anthropic_kwargs: dict[str, Any] = {
                "model": self.config.model_name,
                "temperature": self.config.temperature,
            }
            if self.config.api_key:
                anthropic_kwargs["anthropic_api_key"] = SecretStr(self.config.api_key)
            if self.config.base_url:
                anthropic_kwargs["anthropic_api_url"] = self.config.base_url
            self._llm = ChatAnthropic(**anthropic_kwargs)
        elif self.config.provider == LLMProvider.GOOGLE:
            self._llm = ChatGoogleGenerativeAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                google_api_key=self.config.api_key,
            )
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    def _cache_config_dict(self, **extra: Any) -> dict[str, Any]:
        config_dict: dict[str, Any] = {
            "provider": self.config.provider,
            "model_name": self.config.model_name,
            "temperature": self.config.temperature,
            "base_url": self.config.base_url,
        }
        config_dict.update(extra)
        return config_dict

    def _cache_key_content(self, *args: Any) -> str:
        """Stable string for disk cache keys from invoke arguments."""
        if not args:
            return ""
        primary = self._prompt_to_string(args[0])
        if len(args) == 1:
            return primary
        extra = [self._prompt_to_string(arg) for arg in args[1:]]
        return primary + "\n---\n" + "\n---\n".join(extra)

    def _record_cache_hit(self, prompt_str: str, content_str: str) -> None:
        self._cache_hits += 1
        bt = self.budget_tracker
        if bt is not None:
            bt.add_cache_hit(len(prompt_str), len(content_str))

    def _record_api_usage(self, prompt_str: str, result: Any) -> None:
        self._cache_misses += 1
        bt = self.budget_tracker
        if bt is None:
            return
        input_tokens, output_tokens = _usage_from_llm_result(result)
        bt.add_usage(
            len(prompt_str),
            _chars_received_from_result(result),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def get_cache_stats(
        self,
    ) -> dict[str, int | dict[str, int | dict[str, int] | dict[str, dict[str, int]]]]:
        """Return in-memory hit/miss counters and on-disk cache file stats."""
        disk_stats = self.cache.get_cache_stats()
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "disk": disk_stats,
        }

    async def _invoke_cached(self, *args: Any, **kwds: Any) -> Any:
        """Invoke the LLM with optional disk cache and global in-flight limiting."""
        prompt_key = self._cache_key_content(*args)
        prompt_str = self._prompt_to_string(args[0]) if args else ""
        config_dict = self._cache_config_dict()

        if self.config.cache_enabled:
            cached_response = self.cache.get(prompt_key, config=config_dict, **kwds)
            if cached_response is not None:
                logger.debug("Cache hit: %s...", prompt_str[:50])
                content_str = _content_to_str(cached_response["content"])
                self._record_cache_hit(prompt_str, content_str)
                return AIMessage(content=content_str)

        logger.debug("Cache miss, calling LLM: %s...", prompt_str[:50])

        max_inflight = max(1, self.config.llm_max_inflight)
        async with _inflight_semaphore(max_inflight):
            response = await self.llm.ainvoke(*args, **kwds)

        self._record_api_usage(prompt_str, response)

        content_str = _content_to_str(response.content)
        if self.config.cache_enabled and not self.config.cache_read_only:
            response_data = {
                "content": content_str,
                "prompt": prompt_str,
                "kwargs": kwds,
            }
            self.cache.set(prompt_key, response_data, config=config_dict, **kwds)

        return AIMessage(
            content=content_str,
            response_metadata=getattr(response, "response_metadata", {}),
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

    async def complete(self, prompt: str, **kwargs) -> Any:
        """Generate a completion for the given prompt."""
        config_dict = self._cache_config_dict()
        prompt_key = self._prompt_to_string(prompt)

        if self.config.cache_enabled:
            cached_response = self.cache.get(prompt_key, config=config_dict, **kwargs)
            if cached_response is not None:
                logger.debug("Cache hit for prompt: %s...", prompt_key[:50])
                content = cached_response["content"]
                content_str = content if isinstance(content, str) else str(content)
                self._record_cache_hit(prompt_key, content_str)
                return content_str

        logger.debug("Cache miss, calling LLM for prompt: %s...", prompt_key[:50])

        max_inflight = max(1, self.config.llm_max_inflight)
        async with _inflight_semaphore(max_inflight):
            response = await self.llm.ainvoke(prompt, **kwargs)

        self._record_api_usage(prompt_key, response)

        if self.config.cache_enabled and not self.config.cache_read_only:
            response_data = {
                "content": response.content,
                "prompt": prompt_key,
                "kwargs": kwargs,
            }
            self.cache.set(prompt_key, response_data, config=config_dict, **kwargs)

        content = response.content
        return content if isinstance(content, str) else str(content)

    async def extract(self, prompt: str, output_schema: Type[T], **kwargs) -> T:
        """Extract structured data from the prompt according to a schema."""
        parser = PydanticOutputParser(pydantic_object=output_schema)
        format_instructions = parser.get_format_instructions()

        full_prompt = f"{prompt}\n\n{format_instructions}"
        config_dict = self._cache_config_dict(
            output_schema=output_schema.__name__,
        )

        if self.config.cache_enabled:
            cached_response = self.cache.get(full_prompt, config=config_dict, **kwargs)
            if cached_response is not None:
                logger.debug("Cache hit for extraction: %s...", prompt[:50])
                content = cached_response["content"]
                content_str = content if isinstance(content, str) else str(content)
                self._record_cache_hit(full_prompt, content_str)
                if isinstance(content, str):
                    return parser.parse(content)
                return parser.parse(str(content))

        logger.debug("Cache miss, calling LLM for extraction: %s...", prompt[:50])

        max_inflight = max(1, self.config.llm_max_inflight)
        async with _inflight_semaphore(max_inflight):
            response = await self.llm.ainvoke(full_prompt, **kwargs)

        self._record_api_usage(full_prompt, response)

        if self.config.cache_enabled and not self.config.cache_read_only:
            response_data = {
                "content": response.content,
                "prompt": full_prompt,
                "output_schema": output_schema.__name__,
                "kwargs": kwargs,
            }
            self.cache.set(full_prompt, response_data, config=config_dict, **kwargs)

        content = response.content
        return parser.parse(content if isinstance(content, str) else str(content))
