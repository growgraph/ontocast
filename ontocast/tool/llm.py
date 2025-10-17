"""Language Model (LLM) integration tool for OntoCast.

This module provides integration with various language models through LangChain,
supporting both OpenAI and Ollama providers. It enables text generation and
structured data extraction capabilities with optional caching support.

Cache Usage:
    The LLM tool supports caching of responses to avoid redundant API calls.
    To enable caching, pass a cache_dir parameter when creating the tool:

    ```python
    from pathlib import Path
    from ontocast.tool.llm import LLMTool
    from ontocast.config import LLMConfig

    # Create LLM tool with cache
    cache_dir = Path("/tmp/llm_cache")
    llm_tool = await LLMTool.acreate(
        config=LLMConfig(...),
        cache_dir=cache_dir
    )
    ```

    Cache files are stored as JSON files with filenames based on SHA256 hashes
    of the prompt and LLM configuration. This ensures that identical prompts
    with the same configuration will return cached responses.
"""

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Type, TypeVar

from langchain.output_parsers import PydanticOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.messages.ai import AIMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr, field_validator

from ontocast.config import LLMConfig

from .onto import Tool

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


class LLMTool(Tool):
    """Tool for interacting with language models.

    This class provides a unified interface for working with different language model
    providers (OpenAI, Ollama) through LangChain. It supports both synchronous and
    asynchronous operations.

    Attributes:
        config: LLMConfig object containing all LLM settings.
        cache_dir: Optional directory path for caching LLM responses.
    """

    config: LLMConfig = Field(default_factory=LLMConfig)
    cache_dir: Path | None = Field(
        default=None, description="Directory for caching LLM responses"
    )

    @field_validator("cache_dir", mode="before")
    @classmethod
    def validate_cache_dir(cls, v):
        """Convert cache_dir to Path and expand user if provided."""
        if v is not None:
            return Path(v).expanduser()
        return v

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        **kwargs,
    ):
        """Initialize the LLM tool.

        Args:
            cache_dir: Optional directory path for caching LLM responses.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(cache_dir=cache_dir, **kwargs)
        self._llm = None

    @classmethod
    def create(cls, config: LLMConfig, cache_dir: Path | str | None = None, **kwargs):
        """Create a new LLM tool instance synchronously.

        Args:
            config: LLMConfig object containing LLM settings.
            cache_dir: Optional directory path for caching LLM responses.
            **kwargs: Additional keyword arguments for initialization.

        Returns:
            LLMTool: A new instance of the LLM tool.
        """
        return asyncio.run(cls.acreate(config=config, cache_dir=cache_dir, **kwargs))

    @classmethod
    async def acreate(
        cls, config: LLMConfig, cache_dir: Path | str | None = None, **kwargs
    ):
        """Create a new LLM tool instance asynchronously.

        Args:
            config: LLMConfig object containing LLM settings.
            cache_dir: Optional directory path for caching LLM responses.
            **kwargs: Additional keyword arguments for initialization.

        Returns:
            LLMTool: A new instance of the LLM tool.
        """
        # Create and initialize the instance with the config
        self = cls(config=config, cache_dir=cache_dir, **kwargs)
        await self.setup()
        return self

    async def setup(self):
        """Set up the language model based on the configured provider.

        Raises:
            ValueError: If the provider is not supported.
        """
        # Set up cache directory if provided
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"LLM cache directory set to: {self.cache_dir}")

        if self.config.provider == "openai":
            if self.config.model_name.startswith("gpt-5"):
                self.config.temperature = 1.0
                logger.warning(
                    f"Setting temperature to {self.config.temperature} for gpt-5 class model {self.config.model_name}"
                )
            self._llm = ChatOpenAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                base_url=self.config.base_url,
                api_key=SecretStr(self.config.api_key) if self.config.api_key else None,
            )
        elif self.config.provider == "ollama":
            self._llm = ChatOllama(
                model=self.config.model_name,
                base_url=self.config.base_url,
                temperature=self.config.temperature,
            )
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Call the language model directly.

        Args:
            *args: Positional arguments passed to the LLM.
            **kwds: Keyword arguments passed to the LLM.

        Returns:
            Any: The LLM's response.
        """
        # Extract prompt from args (first argument is typically the prompt)
        prompt = args[0] if args else ""

        # Check cache first
        cache_key = self._generate_cache_key(prompt, **kwds)
        cached_response = self._read_from_cache(cache_key)

        if cached_response is not None:
            prompt_str = self._prompt_to_string(prompt)
            logger.debug(f"Cache hit for __call__: {prompt_str[:50]}...")
            # Return a mock BaseMessage object with the cached content
            content = cached_response["content"]
            content_str = content if isinstance(content, str) else str(content)
            return AIMessage(content=content_str)

        # Generate new response
        prompt_str = self._prompt_to_string(prompt)
        logger.debug(f"Cache miss, calling LLM for __call__: {prompt_str[:50]}...")
        response = self.llm.invoke(*args, **kwds)

        # Cache the response
        response_data = {
            "content": response.content,
            "prompt": self._prompt_to_string(prompt),
            "kwargs": kwds,
        }
        self._write_to_cache(cache_key, response_data)

        return response

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
        elif hasattr(prompt, "to_string"):
            return prompt.to_string()
        elif hasattr(prompt, "text"):
            return prompt.text
        elif hasattr(prompt, "content"):
            return prompt.content
        else:
            return str(prompt)

    def _generate_cache_key(self, prompt: str, **kwargs) -> str:
        """Generate a cache key based on prompt and LLM configuration.

        Args:
            prompt: The input prompt.
            **kwargs: Additional parameters that affect the response.

        Returns:
            str: A hash string to use as cache key.
        """
        # Convert prompt to string if it's a LangChain prompt object
        prompt_str = self._prompt_to_string(prompt)

        # Create a dictionary with all relevant parameters
        cache_data = {
            "prompt": prompt_str,
            "provider": self.config.provider,
            "model_name": self.config.model_name,
            "temperature": self.config.temperature,
            "base_url": self.config.base_url,
            "kwargs": kwargs,
        }

        # Convert to JSON string and hash it
        cache_string = json.dumps(cache_data, sort_keys=True, default=str)
        return hashlib.sha256(cache_string.encode()).hexdigest()

    def _get_cache_file_path(self, cache_key: str) -> Path:
        """Get the cache file path for a given cache key.

        Args:
            cache_key: The cache key.

        Returns:
            Path: The path to the cache file.

        Raises:
            RuntimeError: If cache_dir is None.
        """
        if self.cache_dir is None:
            raise RuntimeError("Cache directory not set")
        return self.cache_dir / f"{cache_key}.json"

    def _read_from_cache(self, cache_key: str) -> dict | None:
        """Read cached response from file.

        Args:
            cache_key: The cache key.

        Returns:
            dict | None: The cached response data or None if not found.
        """
        if self.cache_dir is None:
            return None

        cache_file = self._get_cache_file_path(cache_key)

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read cache file {cache_file}: {e}")
            return None

    def _write_to_cache(self, cache_key: str, response_data: dict) -> None:
        """Write response to cache file.

        Args:
            cache_key: The cache key.
            response_data: The response data to cache.
        """
        if self.cache_dir is None:
            return

        cache_file = self._get_cache_file_path(cache_key)

        try:
            with open(cache_file, "w") as f:
                json.dump(response_data, f, indent=2)
            logger.debug(f"Cached response to {cache_file}")
        except IOError as e:
            logger.warning(f"Failed to write cache file {cache_file}: {e}")

    async def complete(self, prompt: str, **kwargs) -> Any:
        """Generate a completion for the given prompt.

        Args:
            prompt: The input prompt for generation.
            **kwargs: Additional keyword arguments for generation.

        Returns:
            Any: The generated completion.
        """
        # Check cache first
        cache_key = self._generate_cache_key(prompt, **kwargs)
        cached_response = self._read_from_cache(cache_key)

        if cached_response is not None:
            logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
            content = cached_response["content"]
            return content if isinstance(content, str) else str(content)

        # Generate new response
        logger.debug(f"Cache miss, calling LLM for prompt: {prompt[:50]}...")
        response = await self.llm.ainvoke(prompt)

        # Cache the response
        response_data = {
            "content": response.content,
            "prompt": self._prompt_to_string(prompt),
            "kwargs": kwargs,
        }
        self._write_to_cache(cache_key, response_data)

        return response.content

    async def extract(self, prompt: str, output_schema: Type[T], **kwargs) -> T:
        """Extract structured data from the prompt according to a schema.

        Args:
            prompt: The input prompt for extraction.
            output_schema: The Pydantic model class defining the output structure.
            **kwargs: Additional keyword arguments for extraction.

        Returns:
            T: The extracted data conforming to the output schema.
        """
        parser = PydanticOutputParser(pydantic_object=output_schema)
        format_instructions = parser.get_format_instructions()

        full_prompt = f"{prompt}\n\n{format_instructions}"

        # Check cache first - include output_schema in cache key
        cache_key = self._generate_cache_key(
            full_prompt, output_schema=output_schema.__name__, **kwargs
        )
        cached_response = self._read_from_cache(cache_key)

        if cached_response is not None:
            logger.debug(f"Cache hit for extraction: {prompt[:50]}...")
            # Parse the cached content
            content = cached_response["content"]
            if isinstance(content, str):
                return parser.parse(content)
            else:
                # Fallback: convert to string if it's not already
                return parser.parse(str(content))

        # Generate new response
        logger.debug(f"Cache miss, calling LLM for extraction: {prompt[:50]}...")
        response = await self.llm.ainvoke(full_prompt)

        # Cache the response
        response_data = {
            "content": response.content,
            "prompt": self._prompt_to_string(full_prompt),
            "output_schema": output_schema.__name__,
            "kwargs": kwargs,
        }
        self._write_to_cache(cache_key, response_data)

        content = response.content
        return parser.parse(content if isinstance(content, str) else str(content))
