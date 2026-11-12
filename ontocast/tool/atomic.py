"""Minimal tool contracts for atomic render/critic loops."""

import copy
from collections.abc import Callable
from typing import Protocol

from pydantic import BaseModel

from ontocast.config import (
    FactsValidationConfig,
    OntologyValidationConfig,
    WebSearchConfig,
)
from ontocast.onto.enum import WorkflowNode
from ontocast.onto.prefixes import PrefixTable, standard_namespace_prefixes
from ontocast.tool.facts_validation import (
    CriticPatchPolicy,
    FactsAcceptancePolicy,
    ValidationPolicy,
)
from ontocast.tool.facts_validation.terms import (
    CatalogRoles,
    SubjectContract,
    value_class_base,
)
from ontocast.tool.llm import LLMTool
from ontocast.tool.ontology_validation.shape import CatalogSurfaceIndex


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


class AtomicOntologyCatalog(Protocol):
    """The slice of an ontology catalog the per-unit repairs read.

    A unit sees a retrieved *snapshot* of the catalog; the repairs that decide
    whether a term is real need the *whole* catalog, since a term the snapshot
    did not retrieve is still a term.
    """

    def catalog_terms(self) -> set[str]:
        """Every IRI any served ontology declares or references."""
        ...

    def catalog_surface_index(self) -> CatalogSurfaceIndex:
        """Labels and local names of every catalog term, case-insensitive."""
        ...

    def catalog_roles(self) -> CatalogRoles:
        """What every catalog term is, and the catalog's class hierarchy."""
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
        ontology_validation_config: OntologyValidationConfig | None = None,
        citation_vocabulary: dict[str, str] | None = None,
        ontology_catalog: AtomicOntologyCatalog | None = None,
    ):
        """Build the atomic tool surface.

        Args:
            llm_provider: Supplies budget-aware LLM tools.
            search_provider: Optional web-search backend. Without one, search
                returns no hits regardless of configuration.
            web_search_config: Web-grounding settings. Defaults to
                :class:`WebSearchConfig`, which is disabled unless configured.
            facts_validation_config: Facts-gate settings consumed by the render
                and critic paths. Defaults to :class:`FactsValidationConfig`.
            ontology_validation_config: Ontology-delta settings, including the
                ontology critic's pass budget and patch limits. Defaults to
                :class:`OntologyValidationConfig`.
            citation_vocabulary: Bibliographic terms for citation-metadata
                units. Configuration rather than retrieval: a reference list is
                not domain content, so its vocabulary never reaches the catalog.
            ontology_catalog: The scope's full ontology catalog, read by the
                per-unit repairs for term membership. The shared surface is
                built without one; :meth:`scoped_to_catalog` binds it per
                scope.
        """
        web_search = web_search_config or WebSearchConfig()
        facts_validation = facts_validation_config or FactsValidationConfig()
        ontology_validation = ontology_validation_config or OntologyValidationConfig()

        self.llm_provider = llm_provider
        self.search_provider = search_provider
        self._prefix_tables: Callable[[], PrefixTable] | None = None
        self.web_search_config = web_search
        self.ontology_catalog: AtomicOntologyCatalog | None = ontology_catalog

        self.object_property_literal_check = (
            facts_validation.object_property_literal_check
        )
        # Review-and-patch passes: each one is a provider call.
        self.facts_critic_passes = facts_validation.critic_passes
        self.ontology_critic_passes = ontology_validation.critic_passes
        self.ontology_max_class_depth = ontology_validation.max_class_depth
        # Below this many rendered triples the facts critic is skipped: a
        # review of an empty graph is a call that changes nothing.
        self.facts_critic_min_triples = facts_validation.critic_min_triples
        # Insert-only completion passes after the critic loop, each a provider
        # call, taken only while measurements are still missing.
        self.facts_completion_passes = facts_validation.completion_passes
        self.facts_patch_policy = CriticPatchPolicy(
            max_delete_share=facts_validation.critic_max_delete_share,
            min_deletes=facts_validation.critic_min_deletes,
            allow_subject_rename=facts_validation.critic_allow_subject_rename,
        )
        # Insert-only: rendered deletes are dropped before apply, and the
        # critic gets no delete allowance, so no path removes a triple.
        self.ontology_insert_only = ontology_validation.update_mode == "insert_only"
        self.ontology_patch_policy = CriticPatchPolicy(
            max_delete_share=(
                0.0
                if self.ontology_insert_only
                else ontology_validation.critic_max_delete_share
            ),
            min_deletes=(
                0
                if self.ontology_insert_only
                else ontology_validation.critic_min_deletes
            ),
            allow_subject_rename=False,
        )
        # Numeric-coverage knobs for the deterministic per-unit validator.
        self.numeric_coverage_limit = facts_validation.numeric_coverage_limit
        self.numeric_coverage_mandatory = facts_validation.numeric_coverage_mandatory
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
        # strict: the facts are projected onto the catalog and the declared
        # vocabulary above; open: generic core vocabularies fill its gaps.
        # Read by the facts prompt and by the term checks alike.
        self.facts_vocabulary: str = facts_validation.vocabulary
        # Everything the deterministic term checks must treat as blessed, as
        # one object -- see ValidationPolicy.
        self.validation_policy = ValidationPolicy(
            additional_standard_namespaces=self.additional_standard_namespaces,
            vocabulary=facts_validation.vocabulary,
            quantity_fallback_vocabulary=self.quantity_fallback_vocabulary,
            code_predicates=self.code_predicates,
            numeric_identifier_guard=facts_validation.numeric_identifier_guard,
            domain_adherence_min_share=facts_validation.domain_adherence_min_share,
            domain_adherence_min_terms=facts_validation.domain_adherence_min_terms,
        )
        # A sibling of ValidationPolicy, deliberately not a field on it.
        # ValidationPolicy answers "what must never be flagged"; this answers
        # "what blocks a unit from leaving the loop". Both travel to the unit
        # loop, but the term checks and the catalog lint have no business
        # knowing about acceptance.
        self.acceptance_policy = FactsAcceptancePolicy(
            blocking_fix_severity=facts_validation.accept_blocking_severity,
        )
        self.ontology_acceptance_policy = FactsAcceptancePolicy(
            blocking_finding_kinds=frozenset(
                ontology_validation.accept_blocking_finding_kinds
            ),
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

    def scoped_to_catalog(
        self,
        catalog: AtomicOntologyCatalog,
        prefix_tables: Callable[[], PrefixTable] | None = None,
    ) -> "AtomicToolBox":
        """A shallow copy of this surface bound to ``catalog``.

        The surface is tenancy-independent and shared by every scope; a
        catalog is not. Binding on a copy rather than on ``self`` keeps one
        scope's catalog from being read by a unit of another scope that runs on
        the same shared instance.

        Args:
            catalog: The scope's ontology catalog.
            prefix_tables: Returns the scope's current prefix table; read on
                every call, since the catalog changes while a document runs.
        """
        scoped = copy.copy(self)
        scoped.ontology_catalog = catalog
        scoped._prefix_tables = prefix_tables
        return scoped

    def prefix_table(self) -> PrefixTable:
        """The bound scope's prefix table; an empty one when none is bound.

        An empty table still shows every well-known namespace under its
        standard prefix.
        """
        if self._prefix_tables is None:
            return PrefixTable({}, {})
        return self._prefix_tables()

    def catalog_terms(self) -> set[str]:
        """Every IRI the bound catalog declares or references, across all of it.

        Membership set for the per-unit repairs: a predicate present here is a
        real catalog term even when the unit's retrieved snapshot omits it, and
        must never be rewritten toward a look-alike the snapshot does carry.
        Empty when no catalog is bound. The catalog memoises the set on its
        served versions, so calling this per unit is cheap; treat the result
        as read-only.
        """
        if self.ontology_catalog is None:
            return set()
        return self.ontology_catalog.catalog_terms()

    def catalog_surface_index(self) -> CatalogSurfaceIndex:
        """Labels and local names of every term of the bound catalog.

        Read by the per-unit duplicate check: a term the unit's snapshot did
        not retrieve is still a term a render must not re-mint. Empty when no
        catalog is bound; memoised by the catalog, so treat it as read-only.
        """
        if self.ontology_catalog is None:
            return CatalogSurfaceIndex()
        return self.ontology_catalog.catalog_surface_index()

    def catalog_roles(self) -> CatalogRoles:
        """What the bound catalog says each term is; empty when none is bound."""
        if self.ontology_catalog is None:
            return CatalogRoles()
        return self.ontology_catalog.catalog_roles()

    def value_classes(self) -> frozenset[str]:
        """Classes whose instances are values, over the bound catalog.

        The well-known value classes and the configured
        ``FACTS_QUANTITY_FALLBACK_VOCABULARY`` ``value_class``, with every
        catalog class under one of them.
        """
        bindings = {
            prefix: namespace
            for namespace, prefix in standard_namespace_prefixes().items()
        }
        bindings.update(self.prefix_table().parse_map())
        return self.catalog_roles().classes_under(
            value_class_base(self.quantity_fallback_vocabulary, bindings)
        )

    def subject_contract(self) -> SubjectContract:
        """What makes a statement name its subject, over the bound catalog.

        The statement classes (an observation is about its feature of
        interest), the value classes (:meth:`value_classes`), and the
        properties linking a statement and its subject in each direction.
        """
        bindings = {
            prefix: namespace
            for namespace, prefix in standard_namespace_prefixes().items()
        }
        bindings.update(self.prefix_table().parse_map())
        return self.catalog_roles().subject_contract(
            value_class_base(self.quantity_fallback_vocabulary, bindings)
        )

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
