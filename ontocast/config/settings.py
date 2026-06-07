"""Configuration management for OntoCast.

This module provides hierarchical configuration classes that map to the
environment variables and usage patterns in the OntoCast system.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client.http.models import Distance as QdrantDistance

from ontocast.onto.enum import LLMGraphFormat, OntologyContextMode, RenderMode
from ontocast.onto.tenancy import (
    DEFAULT_PROJECT,
    DEFAULT_TENANT,
    tenant_project_facts_name,
    tenant_project_ontologies_name,
)


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
    RRF = "rrf"


class QdrantDedupMode(StrEnum):
    """How Qdrant point identity is derived during upsert."""

    ATOM_ID = "atom_id"
    IRI = "iri"


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
        description=(
            "Maximum concurrent provider LLM requests shared across all documents."
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
    section_tag_min_chars: int = Field(
        default=80,
        description=(
            "Min stripped length for LLM section tagging; smaller segments merge "
            "into neighbors before tagging"
        ),
    )

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
    max_visits_per_node: int = Field(
        default=1,
        ge=1,
        description="Maximum number of visits allowed per node",
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
        description="Maximum number of triples allowed in ontology graph. "
        "Updates that would exceed this limit are skipped with a warning. "
        "Set to None for unlimited.",
    )
    parallel_workers: int = Field(
        default=4,
        description="Maximum number of concurrent unit workers in parallel pipeline",
    )
    parallel_facts_retries: int = Field(
        default=3,
        description="Retry budget for unit facts loop",
    )
    parallel_ontology_retries: int = Field(
        default=3,
        description="Retry budget for unit ontology loop",
    )
    enable_ontology_consolidation: bool = Field(
        default=False,
        description="Run optional ontology consolidation pass after normalization",
    )
    max_concurrent_processes: int | None = Field(
        default=None,
        ge=1,
        description=(
            "When set, limit concurrent /process and /process_unit handlers; "
            "additional requests receive HTTP 503 until a slot is free."
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

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        case_sensitive=False,
    )


class PatchRetrievalConfig(BaseSettings):
    """Scoring, filtering, and capping of ontology atoms after vector search (backend-agnostic)."""

    per_query_core_score_ratio: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description=(
            "Within each query, keep core hits whose score is at least this fraction "
            "of that query's best core score."
        ),
    )
    per_query_neighborhood_score_ratio: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description=(
            "Within each query, keep neighborhood hits whose score is at least this "
            "fraction of that query's best neighborhood score."
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
        default=0.85,
        ge=0.0,
        le=1.0,
        description=(
            "Within each query, keep BM25 hits whose score is at least this fraction "
            "of that query's best BM25 score."
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
            "return an empty patch (no relevant ontology). Set to 0 to disable (fused "
            "cosine-style scores; tune with your embedding model)."
        ),
    )
    merged_score_ratio: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description=(
            "After merging hits across queries, keep atoms whose score is at least this "
            "fraction of the merged top score. 0 disables."
        ),
    )
    cross_query_merge_mode: CrossQueryMergeMode = Field(
        default=CrossQueryMergeMode.HYBRID,
        description=(
            "Cross-window merge: hybrid (max-score tier-1 + per-ontology tier-2), "
            "max_score (entity best per window), or rrf (legacy reciprocal rank sum)."
        ),
    )
    max_atoms_tier1: int = Field(
        default=12,
        ge=0,
        description=(
            "Hybrid merge: global cap on strong tier-1 seeds (max score per entity IRI). "
            "0 means no tier-1 cap."
        ),
    )
    per_ontology_seed_quota: int = Field(
        default=3,
        ge=0,
        description=(
            "Hybrid merge: additional seeds per ontology IRI in tier-2 (multi-ontology coverage)."
        ),
    )
    min_entity_score: float = Field(
        default=0.3,
        ge=0.0,
        description=(
            "Hybrid merge tier-2: minimum per-entity max fused score to qualify as a seed."
        ),
    )
    mmr_lambda: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description=(
            "MMR trade-off over dense core+neighborhood vectors: 1.0 keeps pure relevance, "
            "0.0 maximizes diversity."
        ),
    )
    max_atoms: int = Field(
        default=25,
        ge=0,
        description="Hard cap for retained atoms after filtering / MMR (0 means unlimited).",
    )

    model_config = SettingsConfigDict(
        env_prefix="ONTOLOGY_PATCH_",
        case_sensitive=False,
    )


class QdrantConfig(BaseSettings):
    """Qdrant vector store settings."""

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
    distance: QdrantDistance = Field(
        default=QdrantDistance.COSINE,
        description=(
            "Qdrant vector distance when creating collections "
            "(Cosine, Dot, Euclid, Manhattan; same as qdrant_client Distance)."
        ),
    )
    top_k: int = Field(
        default=10,
        ge=1,
        description=(
            "Default number of fused vector hits per query for ontology-patch retrieval "
            "(QDRANT_TOP_K). Call sites may pass an explicit ``top_k`` to override this "
            "for a single retrieval; when omitted, patch search uses this value."
        ),
    )
    induced_subgraph_depth: int = Field(
        default=2,
        ge=0,
        description="Neighborhood expansion depth for induced subgraph retrieval.",
    )
    induced_subgraph_hub_seed_count: int = Field(
        default=8,
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
        default=550,
        ge=1,
        description="Hard cap on triples returned for induced subgraph retrieval.",
    )
    induced_subgraph_estimated_triples_per_query: int = Field(
        default=24,
        ge=1,
        description=(
            "Estimated triple budget per proposition/query used to shape per-entity "
            "allocation in induced subgraph retrieval."
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
    consistency_critic_similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum vector retrieval score to report potential cross-ontology conflicts.",
    )
    embedding_batch_size: int = Field(
        default=64,
        ge=1,
        description="Batch size used for embedding requests during indexing.",
    )
    upsert_batch_size: int = Field(
        default=256,
        ge=1,
        description="Batch size used for Qdrant upsert operations.",
    )
    fusion_core_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Core vector score weight for dual-vector ranking fusion.",
    )
    fusion_neighborhood_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Neighborhood vector score weight for dual-vector ranking fusion.",
    )
    fusion_bm25_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description=(
            "BM25 sparse-lane weight for rank fusion (normalized with core and "
            "neighborhood weights when BM25 retrieval is enabled)."
        ),
    )
    dedup_mode: QdrantDedupMode = Field(
        default=QdrantDedupMode.IRI,
        description=(
            "Point identity policy for ontology vectors: 'iri' stores one logical point "
            "per entity key, while 'atom_id' keeps every atom variant as a separate point."
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


class ToolConfig(BaseSettings):
    """Configuration for tools (LLM, triple stores, paths, chunking)."""

    llm_config: LLMConfig = Field(default_factory=LLMConfig)
    chunk_config: ChunkConfig = Field(default_factory=ChunkConfig)
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
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)


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
            "When true, ``--input-path`` batch mode flushes the triple store "
            "(configured datasets) before loading ontologies."
        ),
    )

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
