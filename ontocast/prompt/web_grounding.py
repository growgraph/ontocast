"""Optional web-grounding surface for LLM prompts and schema generation."""

from __future__ import annotations

from ontocast.onto.enum import WorkflowNode
from ontocast.onto.model import ExternalEvidenceRequest
from ontocast.onto.unit_states import UnitFactsState, UnitOntologyState

WEB_SEARCH_REQUEST_FIELD = "external_evidence_request"
WEB_SEARCH_REQUEST_DEF = "ExternalEvidenceRequest"

SEARCH_GUIDELINES_ONTOLOGY_RENDER = """\
11. **External evidence is optional and advisory**:
   - Use web evidence only to resolve ambiguity, terminology, standards, or domain conventions.
   - If external snippets conflict with source text or ontology context, prioritize source text and ontology context.
   - Avoid adding entities/relations that are only weakly supported by web snippets.

12. **Search decision output**:
   - Include `external_evidence_request` in your structured response.
   - Set `initiate_search=true` only when web evidence is necessary to resolve uncertainty
     before a better retry can be produced.
   - Otherwise keep `initiate_search=false`.
   - When true, provide short `rationale` and optional focused `query_hints`.
"""

SEARCH_GUIDELINES_ONTOLOGY_CRITIC = """\
- Treat external web evidence as optional support only. If evidence conflicts with source text or ontology context, prioritize source text and ontology context.
- Include `external_evidence_request` in your structured response:
  - Set `initiate_search=true` only when external web evidence is needed to resolve ambiguity.
  - Keep `initiate_search=false` when source text + ontology are sufficient.
  - Provide concise `rationale` and optional focused `query_hints` when search is requested.
"""

SEARCH_GUIDELINES_FACTS_RENDER = """\
11. Decide whether external evidence is needed for a retry and set `external_evidence_request`:
    - Set `initiate_search=true` only when ambiguity/term disambiguation/standards lookup materially blocks quality.
    - Otherwise keep `initiate_search=false`.
    - Provide concise `rationale` and optional focused `query_hints` when search is requested.
"""

SEARCH_GUIDELINES_FACTS_CRITIC = """\
# SEARCH DECISION OUTPUT

Include `external_evidence_request` in your structured response:
- Set `initiate_search=true` only if external web evidence is necessary to resolve uncertainty
  that blocks a confident critique.
- Keep `initiate_search=false` when the source text + ontology are sufficient.
- When true, provide concise `rationale` and optional focused `query_hints`.
"""

_GUIDELINES_BY_NODE: dict[WorkflowNode, str] = {
    WorkflowNode.TEXT_TO_ONTOLOGY: SEARCH_GUIDELINES_ONTOLOGY_RENDER,
    WorkflowNode.CRITICISE_ONTOLOGY: SEARCH_GUIDELINES_ONTOLOGY_CRITIC,
    WorkflowNode.TEXT_TO_FACTS: SEARCH_GUIDELINES_FACTS_RENDER,
    WorkflowNode.CRITICISE_FACTS: SEARCH_GUIDELINES_FACTS_CRITIC,
}


def search_guidelines_for(node: WorkflowNode, enabled: bool) -> str:
    """Return node-specific search guideline prose, or empty when disabled."""
    if not enabled:
        return ""
    return _GUIDELINES_BY_NODE.get(node, "")


def persist_search_request(
    state: UnitFactsState | UnitOntologyState,
    node: WorkflowNode,
    request: ExternalEvidenceRequest | None,
    enabled: bool,
) -> None:
    """Store LLM search request when enabled; reset to default when disabled."""
    if enabled and request is not None:
        state.set_external_evidence_request(node, request)
    else:
        state.set_external_evidence_request(node, ExternalEvidenceRequest())
