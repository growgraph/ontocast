"""Minimal tool contracts for atomic render/critic loops."""

from typing import Protocol

from pydantic import BaseModel

from ontocast.config import FactsValidationConfig, WebSearchConfig
from ontocast.onto.enum import WorkflowNode
from ontocast.tool.llm import LLMTool


class SearchHit(BaseModel):
    """Single web-search hit used as optional grounding context."""

    title: str
    url: str
    snippet: str


class AtomicLLMProvider(Protocol):
    """Provides budget-aware LLM instances for atomic loop calls."""

    async def get_llm_tool(self, budget_tracker) -> LLMTool:
        """Return an LLM tool tied to the given budget tracker."""
        ...


class AtomicSearchProvider(Protocol):
    """Provides optional web-search retrieval for ontology grounding."""

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        """Return web hits relevant to the query."""
        ...


class AtomicToolBox:
    """Small tool surface used by atomic render/critic paths."""

    def __init__(
        self,
        llm_provider: AtomicLLMProvider,
        search_provider: AtomicSearchProvider | None = None,
        web_search_config: WebSearchConfig | None = None,
        web_search_enabled: bool = False,
        web_search_top_k: int = 3,
        web_search_max_snippet_chars: int = 400,
        web_search_max_total_chars: int = 1800,
        web_search_for_ontology_render: bool = True,
        web_search_for_ontology_critic: bool = True,
        web_search_for_facts_render: bool = False,
        web_search_for_facts_critic: bool = False,
        web_search_planner_enabled: bool = True,
        web_search_planner_max_queries: int = 3,
        web_search_planner_min_query_chars: int = 12,
        web_search_planner_min_confidence: float = 0.35,
        web_search_reuse_evidence_across_attempt: bool = True,
        web_search_allowed_domains: tuple[str, ...] = (),
        web_search_blocked_domains: tuple[str, ...] = (),
        web_search_min_snippet_chars: int = 40,
        facts_validation_config: FactsValidationConfig | None = None,
        citation_vocabulary: dict[str, str] | None = None,
    ):
        self.llm_provider = llm_provider
        self.search_provider = search_provider
        self.web_search_config = web_search_config
        self.object_property_literal_check = (
            facts_validation_config.object_property_literal_check
            if facts_validation_config is not None
            else True
        )
        # Finding-driven repair renders: each one is a provider call.
        self.facts_llm_repair_visits = (
            facts_validation_config.llm_repair_visits
            if facts_validation_config is not None
            else 1
        )
        # Code predicates for the LLM-free code -> catalog IRI repair.
        self.code_predicates: tuple[str, ...] = (
            tuple(facts_validation_config.code_predicates)
            if facts_validation_config is not None
            else ()
        )
        self.property_alias_min_ratio = (
            facts_validation_config.property_alias_min_ratio
            if facts_validation_config is not None
            else 0.85
        )
        # Bibliographic terms for citation-metadata units. Configuration rather
        # than retrieval: a reference list is not domain content, so its
        # vocabulary never reaches the catalog.
        self.citation_vocabulary: dict[str, str] = dict(citation_vocabulary or {})
        # Fallback vocabulary the facts prompt names for bounded quantities when
        # retrieval supplied no suitable class. None means "use the default";
        # an explicitly empty mapping forbids the fallback.
        self.quantity_fallback_vocabulary: dict[str, str] | None = (
            dict(facts_validation_config.quantity_fallback_vocabulary)
            if facts_validation_config is not None
            else None
        )
        # Non-meta vocabularies a deployment shares across catalogs and does not
        # want reported as unknown terms.
        self.additional_standard_namespaces: tuple[str, ...] = (
            tuple(facts_validation_config.additional_standard_namespaces)
            if facts_validation_config is not None
            else ()
        )

        if web_search_config is not None:
            self.web_search_enabled = web_search_config.enabled
            self.web_search_top_k = web_search_config.top_k
            self.web_search_max_snippet_chars = web_search_config.max_snippet_chars
            self.web_search_max_total_chars = web_search_config.max_total_chars
            self.web_search_for_ontology_render = (
                web_search_config.ontology_render_enabled
            )
            self.web_search_for_ontology_critic = (
                web_search_config.ontology_critic_enabled
            )
            self.web_search_for_facts_render = web_search_config.facts_render_enabled
            self.web_search_for_facts_critic = web_search_config.facts_critic_enabled
            self.web_search_planner_enabled = web_search_config.planner_enabled
            self.web_search_planner_max_queries = web_search_config.planner_max_queries
            self.web_search_planner_min_query_chars = (
                web_search_config.planner_min_query_chars
            )
            self.web_search_planner_min_confidence = (
                web_search_config.planner_min_confidence
            )
            self.web_search_reuse_evidence_across_attempt = (
                web_search_config.reuse_evidence_across_attempt
            )
            self.web_search_allowed_domains = {
                value.strip().lower()
                for value in web_search_config.allowed_domains
                if value.strip()
            }
            self.web_search_blocked_domains = {
                value.strip().lower()
                for value in web_search_config.blocked_domains
                if value.strip()
            }
            self.web_search_min_snippet_chars = web_search_config.min_snippet_chars
        else:
            self.web_search_enabled = web_search_enabled
            self.web_search_top_k = web_search_top_k
            self.web_search_max_snippet_chars = web_search_max_snippet_chars
            self.web_search_max_total_chars = web_search_max_total_chars
            self.web_search_for_ontology_render = web_search_for_ontology_render
            self.web_search_for_ontology_critic = web_search_for_ontology_critic
            self.web_search_for_facts_render = web_search_for_facts_render
            self.web_search_for_facts_critic = web_search_for_facts_critic
            self.web_search_planner_enabled = web_search_planner_enabled
            self.web_search_planner_max_queries = web_search_planner_max_queries
            self.web_search_planner_min_query_chars = web_search_planner_min_query_chars
            self.web_search_planner_min_confidence = web_search_planner_min_confidence
            self.web_search_reuse_evidence_across_attempt = (
                web_search_reuse_evidence_across_attempt
            )
            self.web_search_allowed_domains = {
                value.strip().lower()
                for value in web_search_allowed_domains
                if value.strip()
            }
            self.web_search_blocked_domains = {
                value.strip().lower()
                for value in web_search_blocked_domains
                if value.strip()
            }
            self.web_search_min_snippet_chars = web_search_min_snippet_chars

    async def get_llm_tool(self, budget_tracker) -> LLMTool:
        """Return a budget-aware LLM tool instance."""
        return await self.llm_provider.get_llm_tool(budget_tracker)

    async def search(
        self, query: str, max_results: int | None = None
    ) -> list[SearchHit]:
        """Run optional web search and return normalized hits."""
        if not self.web_search_enabled or self.search_provider is None:
            return []

        result_limit = max_results if max_results is not None else self.web_search_top_k
        return await self.search_provider.search(query=query, max_results=result_limit)

    def web_grounding_enabled_for_node(self, node: WorkflowNode) -> bool:
        """Return whether web grounding is enabled for a workflow node."""
        if not self.web_search_enabled:
            return False
        mapping = {
            WorkflowNode.TEXT_TO_ONTOLOGY: self.web_search_for_ontology_render,
            WorkflowNode.CRITICISE_ONTOLOGY: self.web_search_for_ontology_critic,
            WorkflowNode.TEXT_TO_FACTS: self.web_search_for_facts_render,
            WorkflowNode.CRITICISE_FACTS: self.web_search_for_facts_critic,
        }
        return mapping.get(node, False)
