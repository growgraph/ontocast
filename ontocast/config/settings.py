"""Configuration management for OntoCast.

This module provides hierarchical configuration classes that map to the
environment variables and usage patterns in the OntoCast system.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ontocast.onto.constants import DEFAULT_DOMAIN
from ontocast.onto.enum import (
    LLMGraphFormat,
    OntologyContextMode,
    RenderMode,
    VectorDistance,
    VectorStoreBackend,
)
from ontocast.onto.tenancy import (
    DEFAULT_PROJECT,
    DEFAULT_TENANT,
    TenancyScope,
    tenant_project_facts_name,
    tenant_project_ontologies_name,
)

# Explicit public surface. ``ontocast.config`` re-exports this module with a
# star import, so without __all__ every imported third-party name (Field,
# BaseSettings, SettingsConfigDict, AliasChoices, Path, Literal, StrEnum)
# became part of the package's public API.
__all__ = [
    "AggregationConfig",
    "ChunkConfig",
    "ClaudeModel",
    "Config",
    "ConverterConfig",
    "CrossQueryMergeMode",
    "DomainConfig",
    "EmbeddingConfig",
    "EmbeddingProvider",
    "FactsValidationConfig",
    "FusekiConfig",
    "GeminiModel",
    "InducedSubgraphSeedOrder",
    "LLMConfig",
    "LLMModelName",
    "LLMModelNameAbstract",
    "LLMProvider",
    "LanceDBConfig",
    "LexicalTriggerFusion",
    "OllamaModel",
    "OpenAIModel",
    "PatchRetrievalConfig",
    "PathConfig",
    "QdrantConfig",
    "ServerConfig",
    "SiblingGuardScope",
    "SymbolCaseMismatchPolicy",
    "ToolConfig",
    "VectorStoreConfig",
    "VectorStoreDedupMode",
    "WebSearchConfig",
    "WebSearchProvider",
]


class LLMProvider(StrEnum):
    """Supported LLM providers."""

    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class LLMModelNameAbstract(StrEnum):
    """Abstract base class for all model names."""


class OpenAIModel(LLMModelNameAbstract):
    """OpenAI model names"""

    # Flagship & Specialized Reasoning
    GPT5_4_PRO = "gpt-5.4-pro"
    GPT5_4_THINKING = "gpt-5.4-thinking"
    GPT5_4 = "gpt-5.4"

    # Cost-Optimized Lineup
    GPT5_4_MINI = "gpt-5.4-mini"
    GPT5_4_NANO = "gpt-5.4-nano"

    GPT4_O = "gpt-4o"
    GPT4_O_MINI = "gpt-4o-mini"
    GPT4_1 = "gpt-41"
    GPT4_1_MINI = "gpt-41-mini"
    GPT5 = "gpt-5"
    GPT5_MINI = "gpt-5-mini"
    GPT5_NANO = "gpt-5-nano"


class OllamaModel(LLMModelNameAbstract):
    """Ollama model names"""

    # Meta
    LLAMA4_SCOUT = "llama4-scout:17b"
    LLAMA3_3 = "llama3.3"
    LLAMA3_3_70B = "llama3.3:70b"
    LLAMA3_1 = "llama3.1"
    LLAMA3_1_70B = "llama3.1:70b"

    # Alibaba Qwen
    QWEN3_6_LATEST = "qwen3.6:latest"
    QWEN3_6_27B = "qwen3.6:27b"
    QWEN3_6_35B = "qwen3.6:35b"
    QWEN2_5_72B = "qwen2.5:72b"

    # IBM Granite
    GRANITE4_1_3B = "granite4.1:3b"
    GRANITE4_1_8B = "granite4.1:8b"
    GRANITE4_1_30B = "granite4.1:30b"

    # Moonshot / DeepSeek
    DEEPSEEK_R1 = "deepseek-r1"
    DEEPSEEK_V3 = "deepseek-v3"
    KIMI_K2_6_CLOUD = "kimi-k2.6:cloud"


class ClaudeModel(LLMModelNameAbstract):
    """Anthropic Claude model names"""

    CLAUDE_SONNET_4 = "claude-sonnet-4-20250514"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-latest"
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku-latest"

    # Frontier Flagships (High Intelligence / Reasoning)
    CLAUDE_4_7_OPUS = "claude-4.7-opus-latest"
    CLAUDE_4_6_OPUS = "claude-4.6-opus-latest"

    # Balanced Production Sweet Spot
    CLAUDE_4_6_SONNET = "claude-4.6-sonnet-latest"
    CLAUDE_4_5_SONNET = "claude-4.5-sonnet-latest"

    # Ultra-Fast / Cost-Effective
    CLAUDE_4_5_HAIKU = "claude-4.5-haiku-latest"


class GeminiModel(LLMModelNameAbstract):
    """Google Gemini model names"""

    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_1_5_PRO = "gemini-1.5-pro"

    # Frontier Intelligence & Reasoning
    GEMINI_3_1_PRO = "gemini-3.1-pro"
    GEMINI_2_5_PRO = "gemini-2.5-pro"

    # Speed & Multimodal Agents
    GEMINI_3_5_FLASH = "gemini-3.5-flash"
    GEMINI_3_FLASH = "gemini-3-flash"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"

    # Ultra Budget & Low-Latency
    GEMINI_3_1_FLASH_LITE = "gemini-3.1-flash-lite"
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"


LLMModelName = OpenAIModel | OllamaModel | ClaudeModel | GeminiModel


class WebSearchProvider(StrEnum):
    """Supported web-search providers."""

    DUCKDUCKGO = "duckduckgo"


class EmbeddingProvider(StrEnum):
    """Supported embedding providers."""

    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"


class CrossQueryMergeMode(StrEnum):
    """How per-query fused hits are merged across proposition windows."""

    HYBRID = "hybrid"
    MAX_SCORE = "max_score"
    SUM_SCORE = "sum_score"


class VectorStoreDedupMode(StrEnum):
    """How vector-store row/point identity is derived during upsert."""

    ATOM_ID = "atom_id"
    IRI = "iri"


class InducedSubgraphSeedOrder(StrEnum):
    """Seed expansion order for induced-subgraph triple budgeting."""

    SCORE = "score"
    ONTOLOGY_ROUND_ROBIN = "ontology_round_robin"


class LexicalTriggerFusion(StrEnum):
    """How lexical-trigger hits combine with semantic retrieval hits."""

    MAX_MERGE = "max_merge"
    APPEND = "append"


class SymbolCaseMismatchPolicy(StrEnum):
    """Treatment of hits whose only symbol evidence is case-mismatched."""

    OFF = "off"
    DEMOTE = "demote"
    DROP = "drop"


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
    cache_enabled: bool = Field(
        default=True,
        description="When true, read and write LLM response disk cache entries.",
    )
    cache_read_only: bool = Field(
        default=False,
        description="When true, use cached responses but do not write new entries.",
    )
    llm_max_inflight: int = Field(
        default=16,
        ge=1,
        # Documented as LLM_MAX_INFLIGHT, but the LLM_ env_prefix would otherwise
        # make the real variable LLM_LLM_MAX_INFLIGHT -- the documented name was a
        # silent no-op. "max_inflight" resolves to LLM_MAX_INFLIGHT under the prefix.
        validation_alias=AliasChoices("llm_max_inflight", "max_inflight"),
        description=(
            "Maximum concurrent provider LLM requests shared across all documents. "
            "Set via LLM_MAX_INFLIGHT."
        ),
    )
    think: bool | None = Field(
        default=None,
        description=(
            "Controls thinking/reasoning mode for Ollama thinking models "
            "(e.g. qwen3, deepseek-r1). "
            "False disables thinking and ensures a non-empty content response. "
            "True enables thinking and captures it separately in reasoning_content. "
            "None uses the model's default behaviour (thinking tags may appear "
            "inline in content, or the response may be empty if all tokens are "
            "consumed during reasoning)."
        ),
    )
    num_predict: int | None = Field(
        default=None,
        description=(
            "Maximum number of tokens to generate (Ollama only). "
            "None uses Ollama's default (unlimited). "
            "Increase this when using thinking models to ensure enough tokens "
            "remain for the actual response after the reasoning phase."
        ),
    )
    num_ctx: int | None = Field(
        default=None,
        description=(
            "Context window size in tokens (Ollama only). "
            "Controls the total KV-cache window: prompt tokens + output tokens must "
            "fit within this budget. Ollama's default is model-dependent (often "
            "2048–4096). For large prompts set this to 16384 or higher. "
            "Directly affects VRAM usage on the inference server."
        ),
    )

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

        if provider == LLMProvider.ANTHROPIC and not isinstance(v, ClaudeModel):
            raise ValueError(
                f"Model {v} is not compatible with Anthropic provider. Use ClaudeModel values."
            )

        if provider == LLMProvider.GOOGLE and not isinstance(v, GeminiModel):
            raise ValueError(
                f"Model {v} is not compatible with Google provider. Use GeminiModel values."
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
    min_size: int = Field(default=3000, description="Minimum chunk size in characters")
    max_size: int = Field(default=12000, description="Maximum chunk size in characters")
    segmenter: Literal["semantic", "docling"] = Field(
        default="semantic",
        description=(
            "Primary segmenter: 'semantic' splits the markdown export inside "
            "detected section boundaries with the built-in semantic chunker "
            "(naive fallback without torch extras); 'docling' uses docling's "
            "HybridChunker structural segments."
        ),
    )
    section_classifier: Literal["llm", "heading", "off"] = Field(
        default="llm",
        description=(
            "Chunk section classification: 'llm' = deterministic heading/span "
            "labels plus LLM backfill for unheaded chunks; 'heading' = "
            "deterministic only (no LLM cost); 'off' = no section tagging "
            "(disables section filters and schema default exclusions)."
        ),
    )
    section_tag_min_chars: int = Field(
        default=80,
        description=(
            "Min stripped length for LLM section tagging; smaller segments merge "
            "into neighbors before tagging"
        ),
    )
    bibliography_mode: Literal["domain_facts", "citations_only", "skip"] = Field(
        default="skip",
        description=(
            "Routing for chunks detected as bibliography/reference lists "
            "(section label or citation-density heuristics): 'skip' (default) "
            "drops the chunks before extraction, 'citations_only' extracts "
            "bibliographic metadata only, 'domain_facts' disables special "
            "handling."
        ),
    )
    citation_vocabulary: dict[str, str] = Field(
        default_factory=lambda: {
            "work_class": "schema:ScholarlyArticle",
            "fallback_class": "schema:CreativeWork",
            "title": "schema:name",
            "author": "schema:author",
            "author_name": "schema:name",
            "date_published": "schema:datePublished",
            "venue": "schema:isPartOf",
            "identifier": "schema:identifier",
            "cites": "schema:citation",
        },
        description=(
            "Terms the citation-metadata prompt uses in 'citations_only' mode, "
            "by role. Bibliographic entries are not domain facts, so unlike the "
            "rest of the pipeline these terms are not retrieved from the "
            "catalog -- they default to schema.org and are overridden here for "
            "catalogs that model citations with another vocabulary (e.g. "
            "bibo, FaBiO, DCMI). Keys are fixed roles; values are CURIEs or "
            "IRIs. Setting an empty mapping drops the vocabulary guidance."
        ),
    )

    model_config = SettingsConfigDict(
        env_prefix="CHUNK_",
        case_sensitive=False,
    )


class ConverterConfig(BaseSettings):
    """Document-conversion settings for Docling-backed inputs."""

    profile: Literal["default", "born_digital"] = Field(
        default="default",
        description=(
            "Conversion preset. 'born_digital' prefers embedded PDF text and enables "
            "a temporary ligature-gap workaround for publisher PDFs."
        ),
    )
    pdf_backend: Literal["docling_parse", "pypdfium2"] = Field(
        default="docling_parse",
        description="PDF backend used by Docling for standard pipeline conversion.",
    )
    do_ocr: bool = Field(
        default=True,
        description="Enable OCR in Docling's standard PDF pipeline.",
    )
    do_table_structure: bool = Field(
        default=True,
        description="Enable table structure extraction in Docling's standard pipeline.",
    )
    force_backend_text: bool = Field(
        default=False,
        description=(
            "Prefer deterministic backend text extraction when available instead of "
            "model-based page reconstruction."
        ),
    )
    table_cell_matching: bool = Field(
        default=True,
        description="Enable Docling table cell matching during table extraction.",
    )
    layout_model: Literal[
        "heron",
        "heron_101",
        "egret_medium",
        "egret_large",
        "egret_xlarge",
        "v2",
    ] = Field(
        default="heron",
        description="Docling layout model preset for the standard PDF pipeline.",
    )
    ocr_engine: Literal[
        "auto",
        "easyocr",
        "rapidocr",
        "tesseract_cli",
        "tesseract",
    ] = Field(
        default="auto",
        description="OCR engine used when OCR is enabled in the standard PDF pipeline.",
    )
    ocr_lang: list[str] = Field(
        default_factory=list,
        description=(
            "OCR language codes passed to the selected Docling OCR engine; leave empty "
            "to use engine defaults."
        ),
    )
    force_full_page_ocr: bool = Field(
        default=False,
        description="Force full-page OCR instead of region-limited OCR.",
    )
    ocr_bitmap_area_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum bitmap area ratio before Docling runs OCR on a region.",
    )
    repair_ligature_gaps: bool = Field(
        default=False,
        description=(
            "TEMP workaround: repair ASCII fi/fl/ff-style ligature gaps that some "
            "publisher PDFs emit after Docling extraction."
        ),
    )

    model_config = SettingsConfigDict(
        env_prefix="CONVERTER_",
        case_sensitive=False,
    )

    @model_validator(mode="after")
    def _apply_profile_defaults(self) -> ConverterConfig:
        if self.profile == "born_digital":
            self.pdf_backend = "pypdfium2"
            self.do_ocr = False
            self.force_backend_text = True
            self.repair_ligature_gaps = True
        return self


class ServerConfig(BaseSettings):
    """Server configuration settings."""

    host: str = Field(
        default="127.0.0.1",
        description=(
            "Interface the server binds to. Defaults to loopback: the server "
            "has no authentication and exposes a destructive /flush, so "
            "binding every interface must be a deliberate choice. Set to "
            "0.0.0.0 for containers."
        ),
    )
    port: int = Field(default=8999, ge=1, le=65535, description="Server port")
    base_recursion_limit: int = Field(
        default=1000, ge=1, description="Recursion limit for workflow"
    )
    estimated_chunks: int = Field(
        default=30, ge=1, description="Estimated number of chunks"
    )
    max_visits_per_node: int = Field(
        default=1,
        ge=1,
        description=(
            "Maximum render attempts per unit loop. At the default of 1 the "
            "critic never runs: the single render is also the final one, and a "
            "critique that cannot drive a retry is skipped. Raise to 2 or more "
            "to enable the LLM critic pass."
        ),
        validation_alias=AliasChoices("max_visits_per_node", "max_visits"),
    )
    render_mode: RenderMode = Field(
        default=RenderMode.ONTOLOGY_AND_FACTS,
        description="Rendering mode: ontology, facts, or ontology_and_facts.",
    )
    llm_graph_format: LLMGraphFormat = Field(
        default=LLMGraphFormat.TURTLE,
        description=(
            "Format used by the LLM when emitting RDF graph payloads: "
            "'turtle' (legacy, Turtle strings) or 'jsonld' (compact JSON-LD objects)."
        ),
    )
    ontology_context_mode: OntologyContextMode = Field(
        default=OntologyContextMode.SELECTED_SINGLE_ONTOLOGY,
        description=(
            "Per-unit ontology context: selected_single_ontology (LLM-picked catalog), "
            "selected_vector_search_ontology (Qdrant stitched ensemble), or "
            "fixed_single_ontology (catalog ontology_id; requires ontology_context_fixed_ontology_id)."
        ),
    )
    ontology_context_fixed_ontology_id: str = Field(
        default="",
        description=(
            "Catalog ontology id when ontology_context_mode is fixed_single_ontology "
            "(batch/server default from env)."
        ),
    )
    ontology_max_triples: int | None = Field(
        default=50000,
        ge=1,
        description="Maximum number of triples allowed in ontology graph. "
        "Updates that would exceed this limit are skipped with a warning. "
        "Set to None for unlimited.",
    )
    parallel_workers: int = Field(
        default=8,
        ge=1,
        description="Maximum number of concurrent unit workers in parallel pipeline "
        "(keep at or below LLM_MAX_INFLIGHT, which caps provider concurrency)",
    )
    enable_ontology_consolidation: bool = Field(
        default=False,
        description="Run optional ontology consolidation pass after normalization",
    )
    max_concurrent_processes: int | None = Field(
        default=None,
        ge=1,
        description=(
            "When set, limit concurrent /process and /process_unit handlers. "
            "Requests beyond the limit queue until a slot frees up; they are "
            "not rejected."
        ),
    )
    max_tenancy_scopes: int = Field(
        default=16,
        ge=1,
        description=(
            "How many tenant/project ToolBoxes to keep resident. Each holds a "
            "triple store connection and an ontology catalog; the expensive "
            "tools (LLM client, converter, embedding model) are shared across "
            "all of them. Least-recently-used scopes are evicted and closed. "
            "Bounded because scopes come from request parameters."
        ),
    )

    model_config = SettingsConfigDict(
        case_sensitive=False,
    )


class FusekiConfig(BaseSettings):
    """Fuseki triple store configuration."""

    uri: str | None = Field(
        default=None,
        description=(
            "Fuseki HTTP server root (e.g. http://localhost:3030), not a dataset "
            "path or #/dataset/... UI URL; use FUSEKI_DATASET for the dataset name."
        ),
    )
    auth: str | None = Field(default=None, description="Fuseki authentication")
    dataset: str | None = Field(
        default=None,
        description=(
            "Facts dataset name; if unset, derived from built-in default "
            f"tenant/project ({DEFAULT_TENANT!r}/{DEFAULT_PROJECT!r})."
        ),
    )
    ontologies_dataset: str | None = Field(
        default=None,
        description=(
            "Ontologies dataset; if unset, derived from the same default tenant/project "
            "as dataset (not read from the environment)."
        ),
    )

    model_config = SettingsConfigDict(
        env_prefix="FUSEKI_",
        case_sensitive=False,
    )

    @model_validator(mode="after")
    def _resolve_fuseki_datasets(self) -> FusekiConfig:
        if self.dataset is None:
            self.dataset = tenant_project_facts_name(DEFAULT_TENANT, DEFAULT_PROJECT)
        if self.ontologies_dataset is None:
            self.ontologies_dataset = tenant_project_ontologies_name(
                DEFAULT_TENANT, DEFAULT_PROJECT
            )
        return self


class DomainConfig(BaseSettings):
    """Domain and URI configuration.

    Reads the same ``CURRENT_DOMAIN`` variable that
    :class:`~ontocast.onto.state.AgentState` defaults from. Previously this
    class declared its own unrelated placeholder default and was never read by
    anything, so the documented knob and the value the pipeline actually used
    could not agree.
    """

    current_domain: str = Field(
        default=DEFAULT_DOMAIN,
        validation_alias=AliasChoices("current_domain", "CURRENT_DOMAIN"),
        description=(
            "IRI stem used to form document namespaces. Also read directly by "
            "AgentState when no explicit value is supplied."
        ),
    )

    model_config = SettingsConfigDict(
        case_sensitive=False,
    )


class PathConfig(BaseSettings):
    """Path and directory configuration."""

    working_directory: Path | None = Field(
        default=None,
        description="Working directory for OntoCast caches and artifacts",
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


class WebSearchConfig(BaseSettings):
    """Optional web-search settings for ontology grounding."""

    enabled: bool = Field(
        default=False,
        description=(
            "Enable optional web grounding. Node execution still starts without "
            "search and only searches when node output requests it."
        ),
    )
    provider: WebSearchProvider = Field(
        default=WebSearchProvider.DUCKDUCKGO, description="Web-search provider"
    )
    top_k: int = Field(default=3, ge=1, le=10, description="Number of results to fetch")
    timeout_seconds: float = Field(
        default=8.0, ge=1.0, le=60.0, description="Search request timeout"
    )
    max_snippet_chars: int = Field(
        default=400, ge=80, le=2000, description="Snippet truncation limit per hit"
    )
    max_total_chars: int = Field(
        default=1800, ge=200, le=10000, description="Total evidence text budget"
    )
    ontology_render_enabled: bool = Field(
        default=True,
        description=(
            "Allow search-eligible retries for ontology render prompts "
            "(first pass remains no-search)."
        ),
    )
    ontology_critic_enabled: bool = Field(
        default=True,
        description=(
            "Allow search-eligible retries for ontology critic prompts "
            "(first pass remains no-search)."
        ),
    )
    facts_render_enabled: bool = Field(
        default=False,
        description=(
            "Allow search-eligible retries for facts render prompts "
            "(first pass remains no-search)."
        ),
    )
    facts_critic_enabled: bool = Field(
        default=False,
        description=(
            "Allow search-eligible retries for facts critic prompts "
            "(first pass remains no-search)."
        ),
    )
    planner_enabled: bool = Field(
        default=True, description="Enable LLM planner for web-search decisions"
    )
    planner_max_queries: int = Field(
        default=3, ge=1, le=8, description="Maximum focused search queries per node"
    )
    planner_min_query_chars: int = Field(
        default=12,
        ge=3,
        le=100,
        description="Minimum query length accepted by guardrails",
    )
    planner_min_confidence: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum planner confidence to run search",
    )
    reuse_evidence_across_attempt: bool = Field(
        default=True,
        description=("Reuse node-scoped evidence between retries for the same unit."),
    )
    min_snippet_chars: int = Field(
        default=40,
        ge=0,
        le=1000,
        description="Minimum snippet length to keep a search hit",
    )
    allowed_domains: list[str] = Field(
        default_factory=list,
        description="Optional allowlist of source domains for evidence",
    )
    blocked_domains: list[str] = Field(
        default_factory=list,
        description="Optional blocklist of source domains for evidence",
    )
    region: str = Field(default="wt-wt", description="DuckDuckGo region code")
    safesearch: str = Field(
        default="moderate", description="DuckDuckGo safesearch mode"
    )

    @field_validator("allowed_domains", "blocked_domains", mode="before")
    @classmethod
    def parse_domains(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, list):
            return [entry.strip().lower() for entry in value if entry.strip()]
        if isinstance(value, str):
            raw_values = [entry.strip().lower() for entry in value.split(",")]
            return [entry for entry in raw_values if entry]
        return []

    model_config = SettingsConfigDict(
        env_prefix="WEB_SEARCH_",
        case_sensitive=False,
    )


class SiblingGuardScope(StrEnum):
    """Scope of the co-object sibling merge guard."""

    SUBJECT = "subject"
    PREDICATE = "predicate"


class AggregationConfig(BaseSettings):
    """Aggregation settings for entity clustering/disambiguation."""

    embedding_model: str = Field(
        default="paraphrase-multilingual-MiniLM-L12-v2",
        description="Sentence-transformers model name used for entity embeddings.",
    )
    similarity_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold used by DBSCAN clustering.",
    )
    candidate_similarity_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description=(
            "Lower cosine threshold used to generate permissive merge "
            "candidates before symbolic validation."
        ),
    )
    lexical_label_jaccard: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum label token-set Jaccard for the fuzzy lexical-alias merge tier."
        ),
    )
    lexical_sequence_ratio: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum SequenceMatcher ratio on URI normal forms for the fuzzy "
            "lexical-alias merge tier."
        ),
    )
    lexical_token_jaccard: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum normal-form token Jaccard for the fuzzy lexical-alias "
            "merge tier (both sides >= 2 tokens)."
        ),
    )
    functional_min_empirical_support: int = Field(
        default=2,
        ge=1,
        description=(
            "Minimum distinct subjects a predicate must be observed on "
            "before it counts as empirically single-valued for the "
            "functional-object merge guard."
        ),
    )
    sibling_guard_scope: SiblingGuardScope = Field(
        default=SiblingGuardScope.SUBJECT,
        description=(
            "Co-object sibling guard scope: 'subject' forbids merging any "
            "two objects of one subject; 'predicate' restricts the "
            "prohibition to objects sharing the same predicate."
        ),
    )

    model_config = SettingsConfigDict(
        env_prefix="AGG_",
        case_sensitive=False,
    )


class EmbeddingConfig(BaseSettings):
    """Embedding provider settings used by vector stores."""

    provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.HUGGINGFACE, description="Embedding model provider"
    )
    model_name: str = Field(
        default="paraphrase-multilingual-MiniLM-L12-v2",
        description="Embedding model identifier used by the selected provider.",
    )
    api_key: str | None = Field(
        default=None, description="Provider API key for hosted embedding services."
    )
    base_url: str | None = Field(
        default=None, description="Provider base URL (for Ollama-compatible endpoints)."
    )
    dimension: int = Field(
        default=384,
        ge=1,
        description="Expected dense embedding vector size for core and neighborhood vectors.",
    )
    bm25_model_name: str = Field(
        default="Qdrant/bm25",
        description="fastembed SparseTextEmbedding model id for the BM25 sparse lane.",
    )
    query_prefix: str = Field(
        default="",
        description=(
            "Prefix prepended to text embedded as a *query*. Asymmetric retrieval models "
            "underperform their spec without it — BGE wants "
            "'Represent this sentence for searching relevant passages: ', E5 wants "
            "'query: '. Empty (default) suits the symmetric paraphrase model. Part of "
            "the stored embedding contract: changing it requires a reindex."
        ),
    )
    document_prefix: str = Field(
        default="",
        description=(
            "Prefix prepended to text embedded as a *document* during indexing "
            "(E5 wants 'passage: '; BGE wants nothing). Part of the stored embedding "
            "contract: changing it requires a reindex."
        ),
    )

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        case_sensitive=False,
    )


class PatchRetrievalConfig(BaseSettings):
    """Scoring, filtering, and capping of ontology atoms after vector search (backend-agnostic).

    Default path is intentionally simple: per-window channel fusion → max-score IRI
    dedupe → per-ontology round-robin → window-scaled hard cap. Relative floors,
    hybrid tier merge, merged-score ratio, and MMR remain available as advanced
    opt-in (non-default) controls.
    """

    per_query_core_score_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Advanced: within each query, keep core hits whose score is at least this "
            "fraction of that query's best core score. 0 disables (default)."
        ),
    )
    per_query_neighborhood_score_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Advanced: within each query, keep neighborhood hits whose score is at least "
            "this fraction of that query's best neighborhood score. 0 disables (default)."
        ),
    )
    min_core_query_best_score: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "If > 0, queries whose top core score is below this contribute no core hits."
        ),
    )
    min_neighborhood_query_best_score: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "If > 0, queries whose top neighborhood score is below this contribute no "
            "neighborhood hits."
        ),
    )
    per_query_bm25_score_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Advanced: within each query, keep BM25 hits whose score is at least this "
            "fraction of that query's best BM25 score. 0 disables (default)."
        ),
    )
    min_bm25_query_best_score: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "If > 0, queries whose top BM25 score is below this contribute no BM25 hits."
        ),
    )
    min_merged_max_score: float = Field(
        default=0.18,
        ge=0.0,
        description=(
            "After merging hits across queries, if the highest retained score is below this, "
            "return an empty patch (no relevant ontology). Set to 0 to disable. Scores are "
            "per-window fused rank scores (RRF-style), not raw cosine."
        ),
    )
    merged_score_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Advanced: after merging hits across queries, keep atoms whose score is at "
            "least this fraction of the merged top score. 0 disables (default)."
        ),
    )
    cross_query_merge_mode: CrossQueryMergeMode = Field(
        default=CrossQueryMergeMode.MAX_SCORE,
        description=(
            "Cross-window merge: max_score (default; entity best score across windows), "
            "sum_score (sum of per-window scores, so a term several windows agree on "
            "outranks one window's top hit), or hybrid (tier-1 global seeds + "
            "per-ontology tier-2 coverage). All three are followed by the same "
            "round-robin / cap stage. Single-window retrieval makes max and sum identical."
        ),
    )
    max_atoms_tier1: int = Field(
        default=12,
        ge=0,
        description=(
            "Hybrid merge only: global cap on strong tier-1 seeds (max score per entity "
            "IRI). 0 means no tier-1 cap. Unused in the default max_score path."
        ),
    )
    per_ontology_seed_quota: int = Field(
        default=0,
        ge=0,
        description=(
            "Max seeds retained per ontology IRI when filling the final seed list "
            "(round-robin under max_score; tier-2 under hybrid). 0 means no per-ontology "
            "cap (global score order only), which is the default: a quota spreads the "
            "seed budget across ontologies that merely scored something, and measured "
            "worse on both recall and precision than plain global score order."
        ),
    )
    min_entity_score: float = Field(
        default=0.3,
        ge=0.0,
        description=(
            "Hybrid merge tier-2 only: minimum per-entity max fused score to qualify. "
            "Unused in the default max_score path."
        ),
    )
    per_ontology_atom_floor: int = Field(
        default=2,
        ge=0,
        description=(
            "Reserve pass before the global fill: each ontology contributing "
            "candidates is guaranteed min(floor, its candidate count) seed "
            "slots, allocated round-robin. Unlike per_ontology_seed_quota "
            "(a ceiling), the floor protects small modules from being starved "
            "by one dominant ontology at the atom cap. 0 disables."
        ),
    )
    small_module_closure_max_triples: int = Field(
        default=300,
        ge=0,
        description=(
            "Include a source ontology's whole (header-stripped) graph in the "
            "snapshot when it has at least one admitted atom and at most this "
            "many triples. Partial inclusion of a tiny vocabulary pushes the "
            "renderer to improvise near-miss property names. Inert unless the "
            "module wins at least one seed, so it pairs with "
            "per_ontology_atom_floor. The single largest lever measured on "
            "case6: with the floor at 2 and the triple budget at 1200 it took "
            "the needed-term recall from 3/11 to 11/11 and declared-property "
            "coverage from 37% to 74%, because a qualified-quantity or "
            "observation module is only useful whole. 0 disables."
        ),
    )
    per_role_atom_floor: int = Field(
        default=12,
        ge=0,
        description=(
            "Reserve pass guaranteeing predicate-role atoms a share of the seed "
            "budget before the global fill, in the same floor-not-ceiling shape "
            "as per_ontology_atom_floor. Dense similarity between prose and a "
            "noun phrase beats a verb phrase, so classes and individuals win a "
            "shared ranking and the properties carrying the graph structure are "
            "crowded out. 0 disables."
        ),
    )
    schema_closure_max_entities: int = Field(
        default=32,
        ge=0,
        description=(
            "Cap on terms admitted by rdfs:domain/rdfs:range closure over the "
            "retrieved seeds: properties whose domain or range names an admitted "
            "class (or its ancestors), and the domain/range classes of admitted "
            "properties. A class with no property that can link it is inert "
            "context. 0 disables."
        ),
    )
    schema_closure_ancestor_depth: int = Field(
        default=2,
        ge=0,
        description=(
            "How far to walk rdfs:subClassOf upward when matching a property's "
            "declared domain/range against an admitted class. Properties are "
            "usually declared on an ancestor of the class the text mentions."
        ),
    )
    mmr_lambda: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "MMR trade-off over dense core+neighborhood vectors: 1.0 keeps pure relevance "
            "(default; skips MMR), lower values increase diversity."
        ),
    )
    seeds_per_window: int = Field(
        default=4,
        ge=1,
        description=(
            "Target seeds per proposition window when scaling the effective atom cap: "
            "min(max_atoms, max(max_atoms_base, seeds_per_window * n_queries))."
        ),
    )
    max_atoms_base: int = Field(
        default=96,
        ge=0,
        description=(
            "Minimum effective atom cap before window scaling (0 defers entirely to "
            "seeds_per_window * n_queries). Raised to match max_atoms: the cap does not "
            "grow with catalog size, and a lower floor was discarding candidates that "
            "per-channel top_k had already paid to retrieve."
        ),
    )
    max_atoms: int = Field(
        default=96,
        ge=0,
        description=(
            "Hard cap for retained atoms after merge / optional MMR (0 means unlimited). "
            "Effective cap is min(max_atoms, max(max_atoms_base, seeds_per_window * n_queries)). "
            "Measured the single largest recall lever on multi-window input, where the "
            "candidate pool reaches ~170 atoms: raising 48 -> 96 moved seed term recall "
            "36% -> 64% on a linked 6-ontology catalog while improving precision. Beyond "
            "this the induced-subgraph triple budget, not the seed count, is the limit."
        ),
    )

    dump_ontology_ranks: bool = Field(
        default=False,
        description=(
            "Collect per-ontology rank diagnostics (best rank/score per channel, fused "
            "rank, whether the ontology survived the atom cut) into retrieval metrics "
            "under 'ontology_rank_diagnostics'. Diagnostic only: it walks every channel "
            "hit list per query and does not change retrieval behaviour."
        ),
    )

    model_config = SettingsConfigDict(
        env_prefix="ONTOLOGY_PATCH_",
        case_sensitive=False,
    )

    def effective_max_atoms(self, n_queries: int) -> int:
        """Window-scaled atom budget: min(hard_cap, max(base, seeds_per_window * n))."""
        if self.max_atoms <= 0:
            return 0
        windows = max(n_queries, 1)
        scaled = max(self.max_atoms_base, self.seeds_per_window * windows)
        return min(self.max_atoms, scaled)


class VectorStoreConfig(BaseSettings):
    """Backend-agnostic vector store retrieval and indexing settings."""

    backend: VectorStoreBackend = Field(
        default=VectorStoreBackend.AUTO,
        description=(
            "Which vector store implementation to use: 'auto' (infer from "
            "QDRANT_URI / LANCEDB_ENABLED, falling back to the in-memory "
            "store), 'memory', 'qdrant', 'lancedb', or 'none' to disable "
            "vector retrieval entirely."
        ),
    )
    top_k: int = Field(
        default=20,
        ge=1,
        description=(
            "Default number of fused vector hits per query for ontology-patch retrieval. "
            "Call sites may pass an explicit ``top_k`` to override this for a single "
            "retrieval; when omitted, patch search uses this value. Raising it past 20 "
            "measured no further seed-recall gain: the retained-atom cap "
            "(ONTOLOGY_PATCH_MAX_ATOMS_BASE) binds first, so the extra candidates are "
            "fetched and then discarded."
        ),
    )
    induced_subgraph_depth: int = Field(
        default=2,
        ge=0,
        description="Neighborhood expansion depth for induced subgraph retrieval.",
    )
    induced_subgraph_hub_seed_count: int = Field(
        default=16,
        ge=0,
        description=(
            "Induced subgraph: number of top-relevance seeds that receive full BFS hub "
            "expansion. 0 disables hub-only BFS (all seeds expand)."
        ),
    )
    induced_subgraph_ancestor_closure_depth: int = Field(
        default=3,
        ge=0,
        description=(
            "Induced subgraph schema shell: max rdfs:subClassOf hops upward per class seed."
        ),
    )
    induced_subgraph_max_total_triples: int = Field(
        default=1200,
        ge=1,
        description=(
            "Hard cap on triples returned for induced subgraph retrieval. This, "
            "not the atom cap, is what binds in practice: measured on the case6 "
            "8-module catalog, every seed-side knob (top_k, max_atoms, MMR, the "
            "atom floors) was flat while the snapshot sat pinned at the old 550, "
            "and raising it alone moved declared-property coverage 21% -> 36%. "
            "Gains flatten past ~1600."
        ),
    )
    induced_subgraph_estimated_triples_per_query: int = Field(
        default=24,
        ge=1,
        description=(
            "Estimated triple budget per proposition/query used to shape per-entity "
            "allocation in induced subgraph retrieval."
        ),
    )
    induced_subgraph_type_promotion_score_factor: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of a retrieved seed's score inherited by its promoted rdf:type "
            "IRIs during induced-subgraph budgeting. The seed always keeps its own "
            "score; this only scales the copy banked on the type. (Transferring the "
            "score to the type and zeroing the individual collapsed all typed "
            "individuals into a relevance-0 tie broken by raw IRI order, which "
            "starved high-ranked seeds under tight triple budgets.)"
        ),
    )
    induced_subgraph_seed_order: InducedSubgraphSeedOrder = Field(
        default=InducedSubgraphSeedOrder.SCORE,
        description=(
            "Seed expansion order under the induced-subgraph triple budget: 'score' "
            "expands in global relevance order; 'ontology_round_robin' interleaves "
            "seeds across source ontologies so no ontology is starved by another's "
            "high scorers. Ablation knob; 'score' is the measured default."
        ),
    )
    induced_subgraph_symbol_predicates: list[str] = Field(
        default_factory=lambda: [
            "http://www.w3.org/2004/02/skos/core#notation",
            "http://qudt.org/schema/qudt/symbol",
            "http://qudt.org/schema/qudt/ucumCode",
        ],
        description=(
            "Predicate IRIs admitted as seed descriptions in the induced subgraph, "
            "between names and glosses (default mirrors lexical_trigger_predicates). "
            "Without them a unit individual reaches the prompt label-only and the "
            "LLM cannot map surface tokens like 'meV' to its IRI. Empty disables."
        ),
    )
    induced_subgraph_candidate_pushdown: bool = Field(
        default=False,
        description=(
            "Build the induced-subgraph working graph from a SPARQL CONSTRUCT of the "
            "seeds' bounded neighborhood instead of the merged ontology graphs. Bounds "
            "memory and wire volume on large catalogs; on small ones the neighborhood is "
            "essentially the whole ontology and there is nothing to gain. Requires a "
            "backend with supports_sparql_construct(); falls back silently otherwise."
        ),
    )
    proposition_window_sentences: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Sentence window size used for proposition-level retrieval slicing.",
    )
    proposition_max_windows: int = Field(
        default=16,
        ge=1,
        description="Upper bound on proposition windows generated per document excerpt.",
    )
    proposition_retrieval_enabled: bool = Field(
        default=True,
        description="Enable proposition-level multi-query retrieval for induced graph mode.",
    )
    consistency_critic_min_fused_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum fused retrieval score for the consistency critic to report a "
            "potential cross-ontology conflict. This is a weighted reciprocal-rank score "
            "(sum of weight/rank over the core, neighborhood and BM25 channels), not a "
            "cosine similarity: with default fusion weights a rank-1 core hit alone "
            "scores 0.583 and rank-2 scores 0.292, so 0.5 means 'top-ranked in the "
            "dominant dense channel'. The former name and 0.7 default read as a "
            "similarity and silently required rank-1 in two channels at once."
        ),
    )
    embedding_batch_size: int = Field(
        default=64,
        ge=1,
        description="Batch size used for embedding requests during indexing.",
    )
    reindex_concurrency: int = Field(
        default=2,
        ge=1,
        description=(
            "Max ontologies to materialize/reindex concurrently during ToolBox "
            "initialize. Dense embeds are serialized via a process-wide lock; "
            "higher values mainly overlap triple-store I/O and BM25 with waits."
        ),
    )
    wipe_on_init: bool = Field(
        default=False,
        description=(
            "When true, ToolBox.initialize drops the current ontology/facts "
            "vector partition before recreating schema and reindexing. Use for "
            "clean-slate recovery (e.g. after embedding-model changes)."
        ),
    )
    prune_orphan_iris_on_init: bool = Field(
        default=True,
        description=(
            "When true, ToolBox.initialize deletes indexed ontology IRIs that "
            "are not in the synchronized catalog (covers IRI renames without a "
            "full wipe)."
        ),
    )
    fusion_core_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description=(
            "Core vector score weight for dual-vector ranking fusion. Weights are "
            "normalized across the three lanes before use, so only their ratio matters."
        ),
    )
    fusion_neighborhood_weight: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description=(
            "Neighborhood vector score weight for dual-vector ranking fusion. Lowered "
            "0.3 -> 0.15: the neighborhood text describes a term's edges rather than "
            "the term, so it corroborates the core lane more than it adds to it. "
            "Halving it "
            "raised seed recall at rank 30 (0.69 -> 0.81 on the matsci case4 excerpt) "
            "without changing which terms the core lane found."
        ),
    )
    fusion_bm25_weight: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description=(
            "BM25 sparse-lane weight for rank fusion (normalized with core and "
            "neighborhood weights when BM25 retrieval is enabled). Raised 0.2 -> 0.8: "
            "a term whose surface form is a symbol or notation (unit symbols, chemical "
            "formulae, gene symbols) is frequently invisible to the dense lanes, so the "
            "sparse lane is its only evidence. At 0.2 the normalized weights were "
            "0.583/0.250/0.167, meaning a rank-1 BM25 hit was outvoted 3.5:1 by a rank-1 "
            "dense hit. Measured on the matsci case4 excerpt, where "
            "'matsci-units#millielectronvolt' is a rank-1 BM25 hit and appears in no "
            "dense lane at all: merged seed rank 32 -> 7, ground-truth recall at rank 40 "
            "0.62 -> 0.81. This does not weaken the dense lanes -- it stops the sparse "
            "lane from being a tie-breaker."
        ),
    )
    minimal_label_limit: int = Field(
        default=5,
        ge=0,
        description=(
            "Maximum declared surface forms (rdfs:label, skos:prefLabel, dcterms:title, "
            "skos:altLabel) folded into each atom's sparse BM25 text. A vocabulary may "
            "declare more aliases than this; symbol aliases sort last and are dropped "
            "first, so raising this widens what the sparse lane can match. Changing it "
            "changes stored sparse vectors and requires a reindex."
        ),
    )
    index_undescribed_iris: bool = Field(
        default=False,
        description=(
            "If True, atomize every IRI in an ontology graph, including ones appearing "
            "only in object or predicate position. Default False: an ontology mints an "
            "atom only for terms it describes (a subject-position triple, or a label). "
            "A merely referenced IRI carries no local text, so its atom is its mangled "
            "local name -- 'a0e0l2i0m1h0t 3d0' for a QUDT dimension vector -- and such "
            "strings embed near the corpus centroid, making them hubs that rank "
            "against every query. Measured on the 8-module matsci catalog: 247 of 690 "
            "atoms "
            "(36%) were undescribed references, and dimension vectors alone took 51 of "
            "140 dense retrieval slots on one document, crowding four ontologies out "
            "entirely. Referenced IRIs stay reachable via induced-subgraph expansion; "
            "they just stop being seeds. Changing this requires a reindex."
        ),
    )
    embed_standard_vocab_iris: bool = Field(
        default=False,
        description=(
            "If True, atomize focal IRIs in standard RDF/OWL/SKOS/DC/SHACL/schema.org "
            "namespaces instead of skipping them. These are scaffolding an ontology "
            "reuses rather than terms it defines, so they carry no retrieval signal for "
            "the document being processed. Changing this requires a reindex."
        ),
    )
    extra_excluded_namespace_prefixes: list[str] = Field(
        default_factory=list,
        description=(
            "Additional IRI prefixes whose entities are never atomized from ontology "
            "sources, on top of the standard-vocabulary set. Use for an upper ontology "
            "or external vocabulary a catalog references but does not define (BFO, SOSA, "
            "OM-2). Mostly redundant once index_undescribed_iris is False -- an "
            "undefined reference is already skipped -- so reach for it when a vocabulary "
            "*is* vendored but should still stay out of the semantic lane. Changing this "
            "requires a reindex."
        ),
    )
    dedup_mode: VectorStoreDedupMode = Field(
        default=VectorStoreDedupMode.IRI,
        description=(
            "Row/point identity policy for ontology vectors: 'iri' stores one logical "
            "record per entity key, while 'atom_id' keeps every atom variant separate."
        ),
    )
    dedup_include_version: bool = Field(
        default=True,
        description=(
            "When dedup_mode='iri', include ontology_version in the identity key so "
            "different ontology versions remain isolated."
        ),
    )
    dedup_include_hash: bool = Field(
        default=True,
        description=(
            "When dedup_mode='iri', include ontology_hash in the identity key so "
            "different ontology snapshots remain isolated."
        ),
    )
    dedup_query_hits_by_iri: bool = Field(
        default=True,
        description=(
            "Drop duplicate retrieval hits sharing the same logical IRI key and keep "
            "the best-scoring one."
        ),
    )
    ontology_table: str | None = Field(
        default=None,
        description=(
            "Ontology atom table/collection name; derived from tenant/project when unset."
        ),
    )
    facts_table: str | None = Field(
        default=None,
        description=(
            "Facts table/collection reserved for future fact vectors; created on init."
        ),
    )
    label_predicates: list[str] = Field(
        default_factory=lambda: [
            "http://www.w3.org/2000/01/rdf-schema#label",
            "http://www.w3.org/2004/02/skos/core#prefLabel",
            "http://purl.org/dc/terms/title",
            "http://www.w3.org/2004/02/skos/core#altLabel",
            "http://purl.org/dc/terms/alternative",
        ],
        description=(
            "Predicate IRIs whose literal objects are indexed as declared labels, "
            "in descending priority (default: rdfs:label, skos:prefLabel, "
            "dcterms:title, skos:altLabel, dcterms:alternative). Changing this "
            "changes stored vectors and requires a reindex."
        ),
    )
    symbol_predicates: list[str] = Field(
        default_factory=lambda: [
            "http://www.w3.org/2004/02/skos/core#notation",
            "http://qudt.org/schema/qudt/symbol",
            "http://qudt.org/schema/qudt/ucumCode",
        ],
        description=(
            "Predicate IRIs whose literal objects are indexed as symbols/notations "
            "(default: skos:notation, qudt:symbol, qudt:ucumCode). This is the "
            "indexing half of the pair whose retrieval half is "
            "INDUCED_SUBGRAPH_SYMBOL_PREDICATES; previously only the retrieval "
            "half was configurable, so overriding it changed what surfaced "
            "without changing what was indexed. Changing this requires a reindex."
        ),
    )
    lexical_trigger_enabled: bool = Field(
        default=True,
        description=(
            "Enable the lexical-trigger lane: scan raw chunk text for notation/symbol "
            "tokens and inject matching atoms as additive retrieval seeds."
        ),
    )
    lexical_trigger_predicates: list[str] = Field(
        default_factory=lambda: [
            "http://www.w3.org/2004/02/skos/core#notation",
            "http://qudt.org/schema/qudt/symbol",
            "http://qudt.org/schema/qudt/ucumCode",
        ],
        description=(
            "Predicate IRIs whose literal objects become case-preserved lexical triggers "
            "(default: skos:notation, qudt:symbol, qudt:ucumCode)."
        ),
    )
    lexical_trigger_heuristic_enabled: bool = Field(
        default=True,
        description=(
            "Promote bare code-shaped rdfs:label/skos:altLabel values as triggers when "
            "no predicate-declared notation exists for the entity."
        ),
    )
    lexical_trigger_min_len: int = Field(
        default=2,
        ge=1,
        description="Minimum length for heuristic label/altLabel trigger promotion.",
    )
    lexical_trigger_max_len: int = Field(
        default=24,
        ge=1,
        description="Maximum length for heuristic label/altLabel trigger promotion.",
    )
    lexical_trigger_heuristic_max_per_entity: int = Field(
        default=2,
        ge=0,
        description="Cap on heuristic triggers per entity.",
    )
    lexical_trigger_max_atoms: int = Field(
        default=16,
        ge=0,
        description=(
            "Maximum lexical-trigger atoms injected per retrieval call, additive to the "
            "semantic atom budget."
        ),
    )
    lexical_trigger_score: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description=(
            "Score assigned to lexical-trigger hits. Calibrated against fused "
            "reciprocal-rank scores: a rank-1 core hit scores 0.583, the merged-atom "
            "floor is 0.18. The previous hardcoded 1.0 outranked every semantic seed "
            "and monopolized hub-BFS expansion slots."
        ),
    )
    lexical_trigger_fusion: LexicalTriggerFusion = Field(
        default=LexicalTriggerFusion.MAX_MERGE,
        description=(
            "How trigger hits combine with semantic hits: 'max_merge' promotes an "
            "already-retrieved atom to max(semantic score, trigger score) and appends "
            "unseen atoms; 'append' (legacy) only appends unseen atoms, silently "
            "discarding the trigger evidence for atoms retrieval already found."
        ),
    )
    query_unit_signals_enabled: bool = Field(
        default=False,
        description=(
            "Match number-adjacent tokens in the unit text ('4-15 days', "
            "'200 kV', '0.5 %') case-insensitively and plural-tolerantly "
            "against catalog surface forms (labels, symbols, UCUM codes) and "
            "inject the matched entities as additional snapshot seeds at "
            "lexical_trigger_score, outside the semantic atom budget. Off by "
            "default until the recall-corpus sweep validates it."
        ),
    )
    symbol_case_mismatch_policy: SymbolCaseMismatchPolicy = Field(
        default=SymbolCaseMismatchPolicy.DEMOTE,
        description=(
            "Treatment of retrieved atoms whose case-preserved symbol surfaces "
            "(skos:notation, qudt:symbol, qudt:ucumCode) match a query token "
            "only case-insensitively, with no exact-case match on any surface. "
            "The BM25 document text is case-folded before indexing, so prose "
            "'meV' also retrieves unit:MegaEV (symbol 'MeV') — one token away "
            "from a 10^9 unit error. 'demote' multiplies the atom score by "
            "symbol_case_mismatch_demote_factor, 'drop' removes the atom, "
            "'off' keeps legacy behavior. Exact-case and label-only matches "
            "are never affected."
        ),
    )
    symbol_case_mismatch_demote_factor: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Score multiplier applied by symbol_case_mismatch_policy='demote'."
        ),
    )

    model_config = SettingsConfigDict(
        env_prefix="VECTOR_STORE_",
        case_sensitive=False,
    )

    @model_validator(mode="after")
    def _resolve_table_names(self) -> VectorStoreConfig:
        if self.ontology_table is None:
            self.ontology_table = tenant_project_ontologies_name(
                DEFAULT_TENANT, DEFAULT_PROJECT
            )
        if self.facts_table is None:
            self.facts_table = tenant_project_facts_name(
                DEFAULT_TENANT, DEFAULT_PROJECT
            )
        return self


class QdrantConfig(BaseSettings):
    """Qdrant-specific vector store connection settings."""

    uri: str | None = Field(default=None, description="Qdrant HTTP endpoint URI.")
    api_key: str | None = Field(default=None, description="Qdrant API key.")
    ontology_collection: str | None = Field(
        default=None,
        description="Qdrant collection for ontology atom vectors; derived when unset.",
    )
    facts_collection: str | None = Field(
        default=None,
        description=(
            "Qdrant collection reserved for future fact vectors; created on init."
        ),
    )
    grpc_port: int = Field(default=6334, description="Qdrant gRPC port.")
    use_grpc: bool = Field(default=False, description="Use gRPC client transport.")
    vector_size: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Vector size override. When set, must equal EmbeddingConfig.dimension; "
            "when unset, the embedding dimension is used."
        ),
    )
    distance: VectorDistance = Field(
        default=VectorDistance.COSINE,
        description=(
            "Qdrant vector distance when creating collections "
            "(Cosine, Dot, Euclid, Manhattan; same as qdrant_client Distance)."
        ),
    )
    upsert_batch_size: int = Field(
        default=256,
        ge=1,
        description="Batch size used for Qdrant upsert operations.",
    )
    timeout_seconds: int = Field(
        default=30,
        ge=1,
        description=(
            "Per-request timeout for Qdrant calls, in whole seconds (the client "
            "accepts nothing finer). Without one, an unreachable or hung Qdrant "
            "blocks a pipeline worker indefinitely."
        ),
    )

    model_config = SettingsConfigDict(
        env_prefix="QDRANT_",
        case_sensitive=False,
    )

    @model_validator(mode="after")
    def _resolve_qdrant_collections(self) -> QdrantConfig:
        if self.ontology_collection is None:
            self.ontology_collection = tenant_project_ontologies_name(
                DEFAULT_TENANT, DEFAULT_PROJECT
            )
        if self.facts_collection is None:
            self.facts_collection = tenant_project_facts_name(
                DEFAULT_TENANT, DEFAULT_PROJECT
            )
        return self


class LanceDBConfig(BaseSettings):
    """Embedded LanceDB vector store settings."""

    enabled: bool = Field(
        default=False,
        description=(
            "Enable embedded LanceDB when QDRANT_URI is unset. "
            "Uses a local directory via lancedb.connect(data_dir)."
        ),
    )
    data_dir: Path | str = Field(
        default="~/.lancedb_data",
        description=(
            "Local filesystem directory passed to lancedb.connect(...) "
            "(supports ~ expansion)."
        ),
    )
    ontology_table: str | None = Field(
        default=None,
        description="Lance table for ontology atom vectors; derived when unset.",
    )
    facts_table: str | None = Field(
        default=None,
        description="Lance table reserved for future fact vectors; created on init.",
    )

    model_config = SettingsConfigDict(
        env_prefix="LANCEDB_",
        case_sensitive=False,
    )

    @model_validator(mode="after")
    def _resolve_lancedb_tables(self) -> LanceDBConfig:
        if self.ontology_table is None:
            self.ontology_table = tenant_project_ontologies_name(
                DEFAULT_TENANT, DEFAULT_PROJECT
            )
        if self.facts_table is None:
            self.facts_table = tenant_project_facts_name(
                DEFAULT_TENANT, DEFAULT_PROJECT
            )
        return self


class FactsValidationConfig(BaseSettings):
    """Deterministic post-checks applied to LLM-rendered facts graphs."""

    object_property_literal_check: bool = Field(
        default=True,
        description=(
            "Quarantine string literals sitting on predicates whose schema range "
            "is a class (e.g. qudt:unit with range qudt:Unit). Quarantined triples "
            "are surfaced to the facts critic so the renderer resolves the token "
            "to an IRI from the ontology context."
        ),
    )
    repair_visits: int = Field(
        default=1,
        ge=0,
        description=(
            "Deterministic repair budget per unit: extra render_facts_update "
            "calls fed with machine-found MANDATORY fixes (quarantined "
            "literals, unknown terms, alias leftovers) and numeric-coverage "
            "candidates. Applies even at MAX_VISITS=1, where the LLM critic "
            "never runs."
        ),
    )
    property_alias_min_ratio: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description=(
            "SequenceMatcher cutoff for deterministic near-miss property "
            "rewrites in catalog namespaces (token containment always "
            "qualifies)."
        ),
    )
    merge_repair_passes: int = Field(
        default=1,
        ge=0,
        description=(
            "Deterministic un-merge budget at the post-aggregation validation "
            "gate: error findings on merged subjects turn into full-cluster "
            "pair vetoes and the facts units are re-aggregated, up to this "
            "many passes. 0 records findings without repairing."
        ),
    )
    suspect_multi_value_severity: Literal["error", "warning"] = Field(
        default="error",
        description=(
            "Severity of SUSPECT_MULTI_VALUE gate findings (multiple distinct "
            "numeric values on one predicate, or multiple objects on a "
            "dominantly single-valued predicate). Only error findings drive "
            "the un-merge repair."
        ),
    )
    additional_standard_namespaces: list[str] = Field(
        default_factory=lambda: ["https://schema.org/", "http://schema.org/"],
        description=(
            "Namespaces exempt from UNKNOWN_TERM findings in addition to the "
            "RDF/OWL substrate and annotation/provenance terms. Only "
            "meta-vocabularies are built in; a domain vocabulary a deployment "
            "genuinely shares across catalogs (SOSA/SSN, CSVW, FOAF, "
            "schema.org, Dublin Core application profiles) is exempted here. "
            "schema.org is the default because the shipped citation "
            "vocabulary uses it."
        ),
    )
    quantity_fallback_vocabulary: dict[str, str] = Field(
        default_factory=lambda: {
            "value_class": "qudt:QuantityValue",
            "numeric_value": "qudt:numericValue",
            "unit": "qudt:unit",
        },
        description=(
            "Vocabulary the facts prompt names as the fallback for bounded or "
            "approximate quantities when the retrieved context supplies no "
            "suitable class. Roles: value_class, numeric_value, unit. Defaults "
            "to QUDT; override for catalogs modelling quantities otherwise. An "
            "empty mapping forbids the fallback and keeps the renderer inside "
            "the provided context. Terms named here are treated as a deliberate "
            "fallback by the NON_CATALOG_VOCABULARY finding rather than as an "
            "unexplained outside term."
        ),
    )
    functional_min_single_support: int = Field(
        default=3,
        ge=1,
        description=(
            "Minimum number of single-valued subjects a predicate needs before "
            "the gate treats it as empirically functional. Below this the "
            "evidence is too thin to call a second value a violation."
        ),
    )
    shapes_dir: str | None = Field(
        default=None,
        description=(
            "Directory of SHACL shape files (.ttl) for the validation gate. "
            "Shapes inlined in the ontology context (sh:NodeShape) are picked "
            "up automatically; SHACL runs only when pyshacl is installed "
            "(extra: 'shacl'). Setting this without the extra installed, or "
            "pointing it at a directory with no readable shapes, logs a "
            "warning rather than silently skipping validation."
        ),
    )

    model_config = SettingsConfigDict(
        env_prefix="FACTS_",
        case_sensitive=False,
    )


class ToolConfig(BaseSettings):
    """Configuration for tools (LLM, triple stores, paths, chunking)."""

    llm_config: LLMConfig = Field(default_factory=LLMConfig)
    chunk_config: ChunkConfig = Field(default_factory=ChunkConfig)
    converter_config: ConverterConfig = Field(default_factory=ConverterConfig)
    path_config: PathConfig = Field(default_factory=PathConfig)
    fuseki: FusekiConfig = Field(default_factory=FusekiConfig)
    domain: DomainConfig = Field(default_factory=DomainConfig)
    web_search: WebSearchConfig = Field(default_factory=WebSearchConfig)
    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    patch_retrieval: PatchRetrievalConfig = Field(
        default_factory=PatchRetrievalConfig,
        description="Ontology patch retrieval: post-vector scoring, MMR, and limits.",
    )
    facts_validation: FactsValidationConfig = Field(
        default_factory=FactsValidationConfig,
        description="Deterministic post-checks on LLM-rendered facts graphs.",
    )
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    lancedb: LanceDBConfig = Field(default_factory=LanceDBConfig)

    @model_validator(mode="after")
    def _reject_dual_vector_backends(self) -> ToolConfig:
        if self.qdrant.uri and self.lancedb.enabled:
            raise ValueError(
                "Configure only one vector store backend: set QDRANT_URI or "
                "LANCEDB_ENABLED=true, not both."
            )
        return self


class Config(BaseSettings):
    """Main OntoCast configuration.

    This class aggregates all configuration sections and provides
    a unified interface for accessing configuration values.
    """

    # Tool configuration (for ToolBox)
    tool_config: ToolConfig = Field(default_factory=ToolConfig)

    # Server configuration (for server.py)
    server: ServerConfig = Field(default_factory=ServerConfig)

    # Additional settings
    logging_level: str | None = Field(default=None, description="Logging level")
    clean: bool = Field(
        default=False,
        description=(
            "When true, ``ontocast process`` batch mode flushes the triple store "
            "(configured datasets) before loading ontologies."
        ),
    )

    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore",
    )

    @classmethod
    def in_memory(cls, **overrides: Any) -> "Config":
        """Build a configuration that needs no external services.

        Selects the in-memory triple store (a full pyoxigraph SPARQL engine)
        and the in-memory vector store, so the whole pipeline runs inside the
        calling process. This is the recommended starting point for embedding
        OntoCast in another application:

        ```python
        tools = await ToolBox.acreate(Config.in_memory())
        ```

        Environment variables still populate any section not named in
        ``overrides``; only the store selection is forced.

        Args:
            **overrides: Fields to set on the returned ``Config``.

        Returns:
            A configuration bound to the process-local backends.
        """
        config = cls(**overrides)
        config.tool_config.fuseki.uri = None
        config.tool_config.qdrant.uri = None
        config.tool_config.lancedb.enabled = False
        config.tool_config.vector_store.backend = VectorStoreBackend.MEMORY
        return config

    def for_tenancy(self, tenant: str, project: str) -> "Config":
        """Return a deep copy of this config bound to ``tenant`` / ``project``.

        The copy is what makes per-scope isolation real. Vector store managers
        receive ``tool_config.vector_store`` and ``tool_config.qdrant`` **by
        reference** (`tool/vector_store/factory.py`) and mutate them when
        tenancy is applied, so two scopes sharing a ``Config`` would alias each
        other's collection names.

        Args:
            tenant: Tenant identifier.
            project: Project identifier within the tenant.

        Returns:
            An independent ``Config`` with dataset, collection and table names
            resolved for the requested partition.

        Raises:
            ValueError: If either identifier is blank.
        """
        scope = TenancyScope.build(tenant, project)
        copy = self.model_copy(deep=True)
        tool_config = copy.tool_config

        tool_config.fuseki.dataset = scope.facts_name
        tool_config.fuseki.ontologies_dataset = scope.ontologies_name
        tool_config.qdrant.facts_collection = scope.facts_name
        tool_config.qdrant.ontology_collection = scope.ontologies_name
        tool_config.lancedb.facts_table = scope.facts_name
        tool_config.lancedb.ontology_table = scope.ontologies_name
        tool_config.vector_store.facts_table = scope.facts_name
        tool_config.vector_store.ontology_table = scope.ontologies_name
        return copy

    def get_tool_config(self) -> ToolConfig:
        """Get tool configuration.

        Returns:
            ToolConfig: Configuration for tools
        """
        return self.tool_config

    def validate_llm_config(self) -> None:
        """Validate LLM configuration and raise errors for missing required settings."""
        provider = self.tool_config.llm_config.provider
        if (
            provider
            in (
                LLMProvider.OPENAI,
                LLMProvider.ANTHROPIC,
                LLMProvider.GOOGLE,
            )
            and not self.tool_config.llm_config.api_key
        ):
            raise ValueError(
                f"LLM_API_KEY environment variable is required for {provider.value} provider"
            )
