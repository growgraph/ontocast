"""Minimal tool contracts for atomic render/critic loops."""

from typing import Protocol

from pydantic import BaseModel

from ontocast.config import FactsValidationConfig, WebSearchConfig
from ontocast.onto.enum import WorkflowNode
from ontocast.tool.facts_validation import FactsAcceptancePolicy, ValidationPolicy
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


def _domain_set(values: list[str]) -> set[str]:
    """Normalize a configured domain list to a lowercase lookup set."""
    return {value.strip().lower() for value in values if value.strip()}


class AtomicToolBox:
    """Small tool surface used by atomic render/critic paths.

    Configuration arrives as config *sections*, never as unpacked scalars. An
    earlier signature accepted both a :class:`WebSearchConfig` and seventeen
    flat ``web_search_*`` parameters mirroring its fields, chosen between by an
    ``if/else``; production passed the section and only tests took the flat
    branch, so the tested configuration path was not the one that shipped. Each
    default also existed three times -- here, in ``settings.py``, and inline at
    the read sites. Now ``settings.py`` is the single source.
    """

    def __init__(
        self,
        llm_provider: AtomicLLMProvider,
        search_provider: AtomicSearchProvider | None = None,
        web_search_config: WebSearchConfig | None = None,
        facts_validation_config: FactsValidationConfig | None = None,
        citation_vocabulary: dict[str, str] | None = None,
    ):
        """Build the atomic tool surface.

        Args:
            llm_provider: Supplies budget-aware LLM tools.
            search_provider: Optional web-search backend. Without one, search
                returns no hits regardless of configuration.
            web_search_config: Web-grounding settings. Defaults to
                :class:`WebSearchConfig`, which is disabled unless configured.
            facts_validation_config: Facts-gate settings consumed by the render
                and repair paths. Defaults to :class:`FactsValidationConfig`.
            citation_vocabulary: Bibliographic terms for citation-metadata
                units. Configuration rather than retrieval: a reference list is
                not domain content, so its vocabulary never reaches the catalog.
        """
        web_search = web_search_config or WebSearchConfig()
        facts_validation = facts_validation_config or FactsValidationConfig()

        self.llm_provider = llm_provider
        self.search_provider = search_provider
        self.web_search_config = web_search

        self.object_property_literal_check = (
            facts_validation.object_property_literal_check
        )
        # Finding-driven repair renders: each one is a provider call.
        self.facts_llm_repair_visits = facts_validation.llm_repair_visits
        # Code predicates for the LLM-free code -> catalog IRI repair.
        self.code_predicates: tuple[str, ...] = tuple(facts_validation.code_predicates)
        self.property_alias_min_ratio = facts_validation.property_alias_min_ratio
        self.citation_vocabulary: dict[str, str] = dict(citation_vocabulary or {})
        # Fallback vocabulary the facts prompt names for bounded quantities when
        # retrieval supplied no suitable class. An explicitly empty mapping
        # forbids the fallback.
        self.quantity_fallback_vocabulary: dict[str, str] | None = dict(
            facts_validation.quantity_fallback_vocabulary
        )
        # Non-meta vocabularies a deployment shares across catalogs and does not
        # want reported as unknown terms.
        self.additional_standard_namespaces: tuple[str, ...] = tuple(
            facts_validation.additional_standard_namespaces
        )
        # Everything the deterministic term checks must treat as blessed, as
        # one object -- see ValidationPolicy.
        self.validation_policy = ValidationPolicy(
            additional_standard_namespaces=self.additional_standard_namespaces,
            quantity_fallback_vocabulary=self.quantity_fallback_vocabulary,
            code_predicates=self.code_predicates,
            numeric_identifier_guard=facts_validation.numeric_identifier_guard,
        )
        # A sibling of ValidationPolicy, deliberately not a field on it.
        # ValidationPolicy answers "what must never be flagged"; this answers
        # "what blocks a unit from leaving the loop". Both travel to the unit
        # loop, but the term checks and the catalog lint have no business
        # knowing about acceptance.
        self.acceptance_policy = FactsAcceptancePolicy(
            blocking_fix_severity=facts_validation.accept_blocking_severity,
        )

        self.web_search_enabled = web_search.enabled
        self.web_search_top_k = web_search.top_k
        self.web_search_max_snippet_chars = web_search.max_snippet_chars
        self.web_search_max_total_chars = web_search.max_total_chars
        self.web_search_for_ontology_render = web_search.ontology_render_enabled
        self.web_search_for_ontology_critic = web_search.ontology_critic_enabled
        self.web_search_for_facts_render = web_search.facts_render_enabled
        self.web_search_for_facts_critic = web_search.facts_critic_enabled
        self.web_search_planner_enabled = web_search.planner_enabled
        self.web_search_planner_max_queries = web_search.planner_max_queries
        self.web_search_planner_min_query_chars = web_search.planner_min_query_chars
        self.web_search_planner_min_confidence = web_search.planner_min_confidence
        self.web_search_reuse_evidence_across_attempt = (
            web_search.reuse_evidence_across_attempt
        )
        self.web_search_allowed_domains = _domain_set(web_search.allowed_domains)
        self.web_search_blocked_domains = _domain_set(web_search.blocked_domains)
        self.web_search_min_snippet_chars = web_search.min_snippet_chars

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
