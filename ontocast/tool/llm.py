"""Language Model (LLM) integration tool for OntoCast.

This module provides integration with various language models through LangChain,
supporting both OpenAI and Ollama providers. It enables text generation and
structured data extraction capabilities.
"""

import asyncio
import logging
from typing import Any, Type, TypeVar

from langchain.output_parsers import PydanticOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

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
    """

    config: LLMConfig = Field(default_factory=LLMConfig)

    def __init__(
        self,
        **kwargs,
    ):
        """Initialize the LLM tool.

        Args:
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._llm = None

    @classmethod
    def create(cls, config: LLMConfig, **kwargs):
        """Create a new LLM tool instance synchronously.

        Args:
            config: LLMConfig object containing LLM settings.
            **kwargs: Additional keyword arguments for initialization.

        Returns:
            LLMTool: A new instance of the LLM tool.
        """
        return asyncio.run(cls.acreate(config=config, **kwargs))

    @classmethod
    async def acreate(cls, config: LLMConfig, **kwargs):
        """Create a new LLM tool instance asynchronously.

        Args:
            config: LLMConfig object containing LLM settings.
            **kwargs: Additional keyword arguments for initialization.

        Returns:
            LLMTool: A new instance of the LLM tool.
        """
        # Create and initialize the instance with the config
        self = cls(config=config, **kwargs)
        await self.setup()
        return self

    async def setup(self):
        """Set up the language model based on the configured provider.

        Raises:
            ValueError: If the provider is not supported.
        """
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
        return self.llm.invoke(*args, **kwds)

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

    async def complete(self, prompt: str, **kwargs) -> Any:
        """Generate a completion for the given prompt.

        Args:
            prompt: The input prompt for generation.
            **kwargs: Additional keyword arguments for generation.

        Returns:
            Any: The generated completion.
        """
        response = await self.llm.ainvoke(prompt)
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
        response = await self.llm.ainvoke(full_prompt)

        return parser.parse(response.content)
