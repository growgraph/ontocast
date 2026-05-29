"""Helpers for optional web-grounded prompts with explicit plan/fetch steps."""

import logging
from typing import TypeVar
from urllib.parse import urlparse

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from ontocast.agent.common import call_llm_with_retry
from ontocast.onto.enum import WorkflowNode
from ontocast.onto.model import (
    ExternalEvidenceCacheEntry,
    ExternalEvidenceHit,
    ExternalEvidencePlan,
    ExternalEvidenceRequest,
)
from ontocast.onto.unit_states import UnitFactsState, UnitOntologyState
from ontocast.tool.atomic import AtomicToolBox, SearchHit

logger = logging.getLogger(__name__)
UnitStateT = TypeVar("UnitStateT", UnitFactsState, UnitOntologyState)

_planner_template = """
You are planning optional web-search grounding for a knowledge-graph workflow.
Decide conservatively whether external web evidence is necessary.

Target workflow node:
{target_node}

Source text:
{content_text}

User instruction:
{user_instruction}

Node request rationale:
{search_rationale}

Node query hints:
{query_hints}

Rules:
1. Prefer NOT searching unless there is genuine ambiguity, domain-standard uncertainty,
   or term disambiguation need.
2. If searching, propose short, focused queries, not broad summaries of the entire text.
3. Never propose more than {max_queries} queries.
4. If no search is needed, set should_search=false, intent=\"none\", and queries=[].

{format_instructions}
"""


def _get_int(tools: AtomicToolBox, key: str, default: int) -> int:
    value = getattr(tools, key, default)
    return int(value) if isinstance(value, int | float) else default


def _get_float(tools: AtomicToolBox, key: str, default: float) -> float:
    value = getattr(tools, key, default)
    return float(value) if isinstance(value, int | float) else default


def _get_bool(tools: AtomicToolBox, key: str, default: bool) -> bool:
    value = getattr(tools, key, default)
    return bool(value) if isinstance(value, bool) else default


def _get_set(tools: AtomicToolBox, key: str) -> set[str]:
    value = getattr(tools, key, set())
    if isinstance(value, set):
        return {str(entry).strip().lower() for entry in value if str(entry).strip()}
    return set()


def _web_grounding_enabled_for_node(
    tools: AtomicToolBox, target_node: WorkflowNode
) -> bool:
    checker = getattr(tools, "web_grounding_enabled_for_node", None)
    if checker is None or not callable(checker):
        return False
    return bool(checker(target_node))


def build_evidence_query(
    content_text: str, user_instruction: str, max_chars: int = 220
) -> str:
    """Backward-compatible fallback query from content and user guidance."""
    source = user_instruction.strip() if user_instruction.strip() else content_text
    query = " ".join(source.split())
    return query[:max_chars].strip()


def _resolve_user_instruction(state: UnitFactsState | UnitOntologyState) -> str:
    if isinstance(state, UnitOntologyState):
        return state.ontology_user_instruction
    return state.facts_user_instruction


def _resolve_content_text(state: UnitFactsState | UnitOntologyState) -> str:
    return state.content_unit.extraction_text


def _resolve_search_request(
    state: UnitFactsState | UnitOntologyState, target_node: WorkflowNode
) -> ExternalEvidenceRequest:
    return state.get_external_evidence_request(target_node)


def _normalize_query(query: str) -> str:
    return " ".join(query.split()).strip()


def _extract_domain(url: str) -> str:
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    if domain.startswith("www."):
        return domain[4:]
    return domain


def _domain_matches(domain: str, patterns: set[str]) -> bool:
    for pattern in patterns:
        if domain == pattern or domain.endswith(f".{pattern}"):
            return True
    return False


def sanitize_external_evidence_plan(
    plan: ExternalEvidencePlan, tools: AtomicToolBox
) -> ExternalEvidencePlan:
    """Apply deterministic guardrails to planner output."""
    deduped_queries: list[str] = []
    seen_queries: set[str] = set()
    min_chars = max(3, _get_int(tools, "web_search_planner_min_query_chars", 12))
    for raw_query in plan.queries:
        query = _normalize_query(raw_query)
        if len(query) < min_chars:
            continue
        alpha_chars = sum(1 for char in query if char.isalpha())
        if alpha_chars < max(4, min_chars // 2):
            continue
        lowered = query.lower()
        if lowered in seen_queries:
            continue
        deduped_queries.append(query)
        seen_queries.add(lowered)

    max_queries = max(1, _get_int(tools, "web_search_planner_max_queries", 3))
    min_confidence = _get_float(tools, "web_search_planner_min_confidence", 0.35)
    deduped_queries = deduped_queries[:max_queries]
    should_search = (
        plan.should_search
        and plan.intent != "none"
        and plan.confidence >= min_confidence
        and len(deduped_queries) > 0
    )
    return ExternalEvidencePlan(
        should_search=should_search,
        rationale=plan.rationale,
        intent=plan.intent if should_search else "none",
        confidence=plan.confidence,
        queries=deduped_queries if should_search else [],
    )


def normalize_search_hits(
    hits: list[SearchHit], tools: AtomicToolBox
) -> list[ExternalEvidenceHit]:
    """Filter and normalize search hits with deterministic quality checks."""
    normalized_hits: list[ExternalEvidenceHit] = []
    seen_urls: set[str] = set()
    allowed_domains = _get_set(tools, "web_search_allowed_domains")
    blocked_domains = _get_set(tools, "web_search_blocked_domains")
    min_snippet_chars = max(0, _get_int(tools, "web_search_min_snippet_chars", 40))

    for hit in hits:
        url = hit.url.strip()
        if not url or url in seen_urls:
            continue
        domain = _extract_domain(url)
        if not domain:
            continue
        if blocked_domains and _domain_matches(domain, blocked_domains):
            continue
        if allowed_domains and not _domain_matches(domain, allowed_domains):
            continue

        snippet = " ".join(hit.snippet.split()).strip()
        if len(snippet) < min_snippet_chars:
            continue

        seen_urls.add(url)
        normalized_hits.append(
            ExternalEvidenceHit(
                title=hit.title.strip() or url,
                url=url,
                snippet=snippet,
                domain=domain,
            )
        )

    return normalized_hits


async def plan_external_evidence_for_node(
    state: UnitStateT, tools: AtomicToolBox, target_node: WorkflowNode
) -> UnitStateT:
    """Plan evidence retrieval for a workflow node using LLM + guardrails."""
    state.node_visits[WorkflowNode.PLAN_EXTERNAL_EVIDENCE] += 1

    if not _web_grounding_enabled_for_node(tools, target_node):
        state.set_external_evidence_cache_entry(
            target_node, ExternalEvidenceCacheEntry()
        )
        state.external_evidence_hits = []
        state.external_evidence_text = ""
        state.external_evidence_source_count = 0
        state.external_evidence_domains = []
        return state

    request = _resolve_search_request(state, target_node)
    if not request.initiate_search:
        state.set_external_evidence_cache_entry(
            target_node, ExternalEvidenceCacheEntry()
        )
        state.external_evidence_hits = []
        state.external_evidence_text = ""
        state.external_evidence_source_count = 0
        state.external_evidence_domains = []
        state.external_evidence_planned_at_node = target_node
        return state

    cached_entry = state.get_external_evidence_cache_entry(target_node)
    if (
        _get_bool(tools, "web_search_reuse_evidence_across_attempt", True)
        and cached_entry.text
        and cached_entry.plan.should_search
    ):
        state.load_external_evidence_for_node(target_node)
        return state

    user_instruction = _resolve_user_instruction(state)
    content_text = _resolve_content_text(state)

    if not _get_bool(tools, "web_search_planner_enabled", True):
        fallback_query = build_evidence_query(
            content_text=content_text, user_instruction=user_instruction
        )
        fallback_plan = ExternalEvidencePlan(
            should_search=bool(fallback_query) or bool(request.query_hints),
            rationale="Planner disabled; fallback query from content/instruction.",
            intent="background",
            confidence=1.0,
            queries=[
                *([fallback_query] if fallback_query else []),
                *request.query_hints,
            ],
        )
        sanitized_plan = sanitize_external_evidence_plan(fallback_plan, tools)
        state.set_external_evidence_cache_entry(
            target_node,
            ExternalEvidenceCacheEntry(
                plan=sanitized_plan,
                hits=[],
                text="",
                source_count=0,
                domains=[],
            ),
        )
        state.load_external_evidence_for_node(target_node)
        return state

    parser = PydanticOutputParser(pydantic_object=ExternalEvidencePlan)
    prompt = PromptTemplate(
        template=_planner_template,
        input_variables=[
            "target_node",
            "content_text",
            "user_instruction",
            "max_queries",
            "search_rationale",
            "query_hints",
            "format_instructions",
        ],
    )
    llm_tool = await tools.get_llm_tool(state.budget_tracker)
    try:
        planned: ExternalEvidencePlan = await call_llm_with_retry(
            llm_tool=llm_tool,
            prompt=prompt,
            parser=parser,
            prompt_kwargs={
                "target_node": target_node.value,
                "content_text": content_text,
                "user_instruction": user_instruction,
                "max_queries": str(
                    max(1, _get_int(tools, "web_search_planner_max_queries", 3))
                ),
                "search_rationale": request.rationale or "none",
                "query_hints": (
                    "\n".join(f"- {hint}" for hint in request.query_hints)
                    if request.query_hints
                    else "none"
                ),
                "format_instructions": parser.get_format_instructions(),
            },
        )
    except Exception as error:
        logger.warning(
            "Evidence planner failed for %s; skipping external evidence (%s).",
            target_node.value,
            str(error),
        )
        planned = ExternalEvidencePlan(
            should_search=False,
            rationale="Planner failure fallback: skip search.",
            intent="none",
            confidence=0.0,
            queries=[],
        )

    merged_plan = ExternalEvidencePlan(
        should_search=planned.should_search or bool(request.query_hints),
        rationale=planned.rationale or request.rationale,
        intent=planned.intent,
        confidence=max(planned.confidence, request.confidence),
        queries=[*planned.queries, *request.query_hints],
    )
    sanitized_plan = sanitize_external_evidence_plan(merged_plan, tools)
    state.set_external_evidence_cache_entry(
        target_node,
        ExternalEvidenceCacheEntry(
            plan=sanitized_plan,
            hits=[],
            text="",
            source_count=0,
            domains=[],
        ),
    )
    state.load_external_evidence_for_node(target_node)
    return state


async def fetch_external_evidence_for_node(
    state: UnitStateT, tools: AtomicToolBox, target_node: WorkflowNode
) -> UnitStateT:
    """Fetch and render evidence for a previously planned workflow node."""
    state.node_visits[WorkflowNode.FETCH_EXTERNAL_EVIDENCE] += 1

    if not _web_grounding_enabled_for_node(tools, target_node):
        state.external_evidence_hits = []
        state.external_evidence_text = ""
        state.external_evidence_source_count = 0
        state.external_evidence_domains = []
        return state

    request = _resolve_search_request(state, target_node)
    if not request.initiate_search:
        state.set_external_evidence_cache_entry(
            target_node, ExternalEvidenceCacheEntry()
        )
        state.external_evidence_hits = []
        state.external_evidence_text = ""
        state.external_evidence_source_count = 0
        state.external_evidence_domains = []
        state.external_evidence_planned_at_node = target_node
        return state

    cache_entry = state.get_external_evidence_cache_entry(target_node)
    plan = cache_entry.plan
    if (
        _get_bool(tools, "web_search_reuse_evidence_across_attempt", True)
        and cache_entry.text
        and plan.should_search
    ):
        state.load_external_evidence_for_node(target_node)
        return state

    if not plan.should_search or not plan.queries:
        state.set_external_evidence_cache_entry(
            target_node, ExternalEvidenceCacheEntry()
        )
        state.load_external_evidence_for_node(target_node)
        return state

    combined_hits: list[SearchHit] = []
    for query in plan.queries:
        search_hits = await tools.search(query)
        combined_hits.extend(search_hits)

    normalized_hits = normalize_search_hits(combined_hits, tools)
    evidence_text = render_external_evidence(
        hits=normalized_hits,
        max_snippet_chars=max(40, _get_int(tools, "web_search_max_snippet_chars", 400)),
        max_total_chars=max(200, _get_int(tools, "web_search_max_total_chars", 1800)),
    )
    state.set_external_evidence_cache_entry(
        target_node,
        ExternalEvidenceCacheEntry(
            plan=plan,
            hits=normalized_hits,
            text=evidence_text,
            source_count=len(normalized_hits),
            domains=sorted({hit.domain for hit in normalized_hits}),
        ),
    )
    state.load_external_evidence_for_node(target_node)
    return state


def render_external_evidence(
    hits: list[ExternalEvidenceHit],
    max_snippet_chars: int,
    max_total_chars: int,
) -> str:
    """Render bounded external evidence as a prompt chapter."""
    if not hits:
        return ""

    rendered_lines: list[str] = []
    remaining_chars = max_total_chars
    for index, hit in enumerate(hits, start=1):
        clean_snippet = " ".join(hit.snippet.split())
        if len(clean_snippet) > max_snippet_chars:
            clean_snippet = f"{clean_snippet[: max_snippet_chars - 3]}..."

        line = f"{index}. {hit.title} | {hit.url}\n   {clean_snippet}"
        if len(line) > remaining_chars:
            if remaining_chars < 80:
                break
            truncated = line[: remaining_chars - 3].rstrip()
            line = f"{truncated}..."
            rendered_lines.append(line)
            break

        rendered_lines.append(line)
        remaining_chars -= len(line)

    if not rendered_lines:
        return ""

    return (
        "### EXTERNAL EVIDENCE (WEB SEARCH)\n"
        "Use these sources to clarify uncertain terms or standards only.\n"
        "When evidence conflicts, prioritize the source text and ontology context.\n\n"
        f"{chr(10).join(rendered_lines)}"
    )
