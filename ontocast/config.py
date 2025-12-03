"""Configuration management for OntoCast.

This module provides hierarchical configuration classes that map to the
environment variables and usage patterns in the OntoCast system.
"""

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ontocast.onto.constants import DEFAULT_DATASET, DEFAULT_ONTOLOGIES_DATASET


class LLMProvider(StrEnum):
    """Supported LLM providers."""

    OPENAI = "openai"
    OLLAMA = "ollama"


class LLMModelNameAbstract(StrEnum):
    """Abstract base class for all model names."""


class OpenAIModel(LLMModelNameAbstract):
    """OpenAI model names."""

    GPT4_O = "gpt-4o"
    GPT4_O_MINI = "gpt-4o-mini"
    GPT4_1 = "gpt-41"
    GPT4_1_MINI = "gpt-41-mini"
    GPT5 = "gpt-5"
    GPT5_MINI = "gpt-5-mini"
    GPT5_NANO = "gpt-5-nano"


class OllamaModel(LLMModelNameAbstract):
    """Ollama model names."""

    QWEN2_5 = "qwen2.5"
    QWEN2_5_72B = "qwen2.5:72b"
    LLAMA3_1 = "llama3.1"
    LLAMA3_1_70B = "llama3.1:70b"
    GRANITE3_3_2B = "granite3.3:2b"
    GRANITE3_3_8B = "granite3.3:8b"


LLMModelName = OpenAIModel | OllamaModel


class LLMConfig(BaseSettings):
    """LLM configuration settings."""

    provider: LLMProvider = Field(
        default=LLMProvider.OPENAI, description="LLM provider"
    )
    model_name: LLMModelName = Field(
        default=OpenAIModel.GPT4_O_MINI, description="LLM model name"
    )
    temperature: float = Field(default=0.0, description="LLM temperature setting")
    base_url: str | None = Field(
        default=None, description="LLM base URL (for ollama, etc.)"
    )
    api_key: str | None = Field(default=None, description="API key for LLM provider")

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        case_sensitive=False,
    )

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: LLMModelName, info) -> LLMModelName:
        """Validate that model_name is compatible with the provider."""
        if "provider" not in info.data:
            return v

        provider = info.data["provider"]

        if provider == LLMProvider.OPENAI and not isinstance(v, OpenAIModel):
            raise ValueError(
                f"Model {v} is not compatible with OpenAI provider. Use OpenAIModel values."
            )

        if provider == LLMProvider.OLLAMA and not isinstance(v, OllamaModel):
            raise ValueError(
                f"Model {v} is not compatible with Ollama provider. Use OllamaModel values."
            )

        return v


class ChunkConfig(BaseSettings):
    """Chunking configuration settings."""

    breakpoint_threshold_type: Literal[
        "percentile", "standard_deviation", "interquartile", "gradient"
    ] = Field(
        default="percentile", description="Type of threshold calculation for chunking"
    )
    breakpoint_threshold_amount: float = Field(
        default=95.0, description="Threshold amount for breakpoint detection"
    )
    buffer_size: int = Field(default=5, description="Buffer size for semantic chunking")
    min_size: int = Field(default=3000, description="Minimum chunk size in characters")
    max_size: int = Field(default=12000, description="Maximum chunk size in characters")

    model_config = SettingsConfigDict(
        env_prefix="CHUNK_",
        case_sensitive=False,
    )


class ServerConfig(BaseSettings):
    """Server configuration settings."""

    port: int = Field(default=8999, description="Server port")
    base_recursion_limit: int = Field(
        default=1000, description="Recursion limit for workflow"
    )
    estimated_chunks: int = Field(default=30, description="Estimated number of chunks")
    max_visits: int = Field(
        default=3, description="Maximum number of visits allowed per node"
    )
    skip_ontology_development: bool = Field(
        default=False, description="Skip ontology critique step"
    )
    skip_facts_rendering: bool = Field(
        default=False, description="Skip facts rendering and go straight to aggregation"
    )
    ontology_max_triples: int | None = Field(
        default=50000,
        description="Maximum number of triples allowed in ontology graph. "
        "Updates that would exceed this limit are skipped with a warning. "
        "Set to None for unlimited.",
    )

    model_config = SettingsConfigDict(
        case_sensitive=False,
    )


class Neo4jConfig(BaseSettings):
    """Neo4j triple store configuration."""

    uri: str | None = Field(default=None, description="Neo4j URI")
    auth: str | None = Field(default=None, description="Neo4j authentication")
    port: int = Field(default=7476, description="Neo4j HTTP port")
    bolt_port: int = Field(default=7689, description="Neo4j Bolt port")

    model_config = SettingsConfigDict(
        env_prefix="NEO4J_",
        case_sensitive=False,
    )


class FusekiConfig(BaseSettings):
    """Fuseki triple store configuration."""

    uri: str | None = Field(default=None, description="Fuseki URI")
    auth: str | None = Field(default=None, description="Fuseki authentication")
    dataset: str = Field(default=DEFAULT_DATASET, description="Fuseki dataset name")
    ontologies_dataset: str = Field(
        default=DEFAULT_ONTOLOGIES_DATASET,
        description="Fuseki dataset name for ontologies",
    )

    model_config = SettingsConfigDict(
        env_prefix="FUSEKI_",
        case_sensitive=False,
    )


class DomainConfig(BaseSettings):
    """Domain and URI configuration."""

    current_domain: str = Field(
        default="https://example.com", description="Current domain for URI generation"
    )

    model_config = SettingsConfigDict(
        case_sensitive=False,
    )


class PathConfig(BaseSettings):
    """Path and directory configuration."""

    working_directory: Path | None = Field(
        default=None,
        description="Working directory for OntoCast (required if filesystem_manager is enabled)",
    )
    ontology_directory: Path | None = Field(
        default=None, description="Directory containing ontology files"
    )
    cache_dir: Path | None = Field(
        default=None, description="Cache directory for LLM responses and tool outputs"
    )

    model_config = SettingsConfigDict(
        env_prefix="ONTOCAST_",
        case_sensitive=False,
    )


class ToolConfig(BaseSettings):
    """Configuration for tools (LLM, triple stores, paths, chunking)."""

    llm_config: LLMConfig = Field(default_factory=LLMConfig)
    chunk_config: ChunkConfig = Field(default_factory=ChunkConfig)
    path_config: PathConfig = Field(default_factory=PathConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    fuseki: FusekiConfig = Field(default_factory=FusekiConfig)
    domain: DomainConfig = Field(default_factory=DomainConfig)


class Config(BaseSettings):
    """Main OntoCast configuration.

    This class aggregates all configuration sections and provides
    a unified interface for accessing configuration values.
    """

    # Tool configuration (for ToolBox)
    tool_config: ToolConfig = Field(default_factory=ToolConfig)

    # Server configuration (for serve.py)
    server: ServerConfig = Field(default_factory=ServerConfig)

    # Additional settings
    logging_level: str | None = Field(default=None, description="Logging level")

    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore",
    )

    def get_tool_config(self) -> ToolConfig:
        """Get tool configuration.

        Returns:
            ToolConfig: Configuration for tools
        """
        return self.tool_config

    def validate_llm_config(self) -> None:
        """Validate LLM configuration and raise errors for missing required settings."""
        if (
            self.tool_config.llm_config.provider == LLMProvider.OPENAI
            and not self.tool_config.llm_config.api_key
        ):
            raise ValueError(
                "LLM_API_KEY environment variable is required for OpenAI provider"
            )
