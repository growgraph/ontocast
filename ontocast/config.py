"""Configuration management for OntoCast.

This module provides hierarchical configuration classes that map to the
environment variables and usage patterns in the OntoCast system.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseSettings):
    """LLM configuration settings."""

    provider: str = Field(
        default="openai", description="LLM provider (openai, ollama, etc.)"
    )
    model_name: str = Field(default="gpt-4.1-mini", description="LLM model name")
    temperature: float = Field(default=0.0, description="LLM temperature setting")
    base_url: str | None = Field(
        default=None, description="LLM base URL (for ollama, etc.)"
    )
    api_key: str | None = Field(default=None, description="API key for LLM provider")

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
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
    clean: bool = Field(default=False, description="Clean triple store on startup")
    skip_ontology_development: bool = Field(
        default=False, description="Skip ontology critique step"
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
        default=None, description="Working directory for OntoCast"
    )
    ontology_directory: Path | None = Field(
        default=None, description="Directory containing ontology files"
    )

    model_config = SettingsConfigDict(
        case_sensitive=False,
    )


class ToolConfig(BaseSettings):
    """Configuration for tools (LLM, triple stores, paths)."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    fuseki: FusekiConfig = Field(default_factory=FusekiConfig)
    domain: DomainConfig = Field(default_factory=DomainConfig)
    paths: PathConfig = Field(default_factory=PathConfig)


class Config(BaseSettings):
    """Main OntoCast configuration.

    This class aggregates all configuration sections and provides
    a unified interface for accessing configuration values.
    """

    # Tool configuration (for ToolBox)
    tools: ToolConfig = Field(default_factory=ToolConfig)

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
        return self.tools

    def validate_llm_config(self) -> None:
        """Validate LLM configuration and raise errors for missing required settings."""
        if self.tools.llm.provider == "openai" and not self.tools.llm.api_key:
            raise ValueError(
                "LLM_API_KEY environment variable is required for OpenAI provider"
            )
