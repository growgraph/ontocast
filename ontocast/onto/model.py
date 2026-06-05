import pathlib
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ontocast.onto.llm_graph_payload import LLMGraphWire
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import GraphUpdate


class BasePydanticModel(BaseModel):
    """Base class for Pydantic models with serialization capabilities."""

    def __init__(self, **kwargs):
        """Initialize the model with given keyword arguments."""
        super().__init__(**kwargs)

    def save_json(self, file_path: str | pathlib.Path) -> None:
        """Write model state to a JSON file.

        Args:
            file_path: Path to save the JSON file.
        """
        state_json = self.model_dump_json(indent=4)
        if isinstance(file_path, str):
            file_path = pathlib.Path(file_path)
        file_path.write_text(state_json)

    @classmethod
    def load(cls, file_path: str | pathlib.Path):
        """Load state from a JSON file.

        Args:
            file_path: Path to the JSON file.

        Returns:
            The loaded model instance.
        """
        if isinstance(file_path, str):
            file_path = pathlib.Path(file_path)
        state_json = file_path.read_text()
        return cls.model_validate_json(state_json)


def create_ontology_selector_report_model(
    num_ontologies: int,
) -> type[BasePydanticModel]:
    """Create a dynamic OntologySelectorReport model with answer_index constraint.

    The answer_index field is constrained to be between 1 and num_ontologies + 1,
    where:
    - 1 to num_ontologies: corresponds to the ontology at that index (1-based)
    - num_ontologies + 1: represents "None" (no suitable ontology)

    Args:
        num_ontologies: The number of ontologies in the selection list.

    Returns:
        A dynamically created Pydantic model class with the appropriate constraint.
    """
    max_index = num_ontologies + 1

    class OntologySelectorReport(BasePydanticModel):
        """Report from ontology selection process.

        Attributes:
            answer_index: Index of the selected option (1-based).
                1 to num_ontologies: select the ontology at that position.
                num_ontologies + 1: select None (no suitable ontology).
        """

        answer_index: int = Field(
            ge=1,
            le=max_index,
            description=(
                f"Index of the selected ontology from the numbered list (1-{num_ontologies}) "
                f"or {max_index} for 'None' (no suitable ontology). "
                f"Use the number corresponding to your choice from the list."
            ),
        )

    # Set the class name for better error messages
    OntologySelectorReport.__name__ = f"OntologySelectorReport_{num_ontologies}"
    return OntologySelectorReport


# Keep a base class for backward compatibility and type hints
class OntologySelectorReport(BasePydanticModel):
    """Base class for ontology selection report.

    Note: Use create_ontology_selector_report_model() to create
    a model with the correct answer_index constraint.
    """

    answer_index: int = Field(
        description="Index of the selected ontology from the numbered list (1-based). "
        "The maximum value depends on the number of ontologies available."
    )


class SemanticTriplesFactsReport(BaseModel):
    """Report containing semantic triples and evaluation scores.

    Graph payloads follow ``LLM_GRAPH_FORMAT``; both wire encodings parse to ``RDFGraph``.

    Attributes:
        semantic_graph: Semantic triples (facts) representing the document.
        ontology_relevance_score: Score 0-100 for how relevant the ontology
            is to the document. 0 is the worst, 100 is the best.
        triples_generation_score: Score 0-100 for how well the facts extraction /
            triples generation was performed. 0 is the worst, 100 is the best.
    """

    semantic_graph: LLMGraphWire = Field(
        default_factory=RDFGraph,
        description=(
            "Semantic triples (facts) representing the document. "
            "Encoding is defined by deployment llm_graph_format and OUTPUT INSTRUCTION."
        ),
    )
    ontology_relevance_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description=(
            "Score between 0 and 100 of how well "
            "the ontology represents the domain of the document."
        ),
    )
    triples_generation_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description=(
            "Score 0-100 for how well the semantic triples "
            "represent the document. 0 is the worst, 100 is the best."
        ),
    )


class ExternalEvidenceRequest(BaseModel):
    """Node-level request for optional web search.

    Nodes use this to explicitly signal whether downstream evidence planning/fetching
    should run for another pass.
    """

    initiate_search: bool = Field(
        default=False,
        description="Whether this node requests external evidence before retrying.",
    )
    rationale: str = Field(
        default="",
        description="Short reason explaining why search is needed (or not needed).",
    )
    query_hints: list[str] = Field(
        default_factory=list,
        description="Optional focused query hints for planner targeting.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in this search decision.",
    )

    @field_validator("query_hints", mode="before")
    @classmethod
    def normalize_query_hints(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            return []
        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            hint = " ".join(item.split()).strip()
            if hint:
                normalized.append(hint)
        return normalized


class FactsRenderReport(BaseModel):
    """Facts rendering output with optional search decision."""

    semantic_graph: LLMGraphWire = Field(
        default_factory=RDFGraph,
        description=(
            "Semantic triples (facts) representing the document. "
            "Encoding is defined by deployment llm_graph_format and OUTPUT INSTRUCTION."
        ),
    )
    ontology_relevance_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description=(
            "Score between 0 and 100 of how well "
            "the ontology represents the domain of the document."
        ),
    )
    triples_generation_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description=(
            "Score 0-100 for how well the semantic triples "
            "represent the document. 0 is the worst, 100 is the best."
        ),
    )
    external_evidence_request: ExternalEvidenceRequest = Field(
        default_factory=ExternalEvidenceRequest,
        description="Optional request to run web search before retrying.",
    )

    @model_validator(mode="before")
    @classmethod
    def _flatten_legacy_facts_report(cls, data: object) -> object:
        if not isinstance(data, dict) or "facts_report" not in data:
            return data
        payload = dict(data)
        nested = payload.pop("facts_report")
        if isinstance(nested, dict):
            for key, value in nested.items():
                if key not in payload:
                    payload[key] = value
        return payload


class GraphUpdateRenderReport(BaseModel):
    """Graph update rendering output with optional search decision."""

    graph_update: GraphUpdate = Field(
        description="Structured graph patch payload: ordered insert/delete triple operations."
    )
    external_evidence_request: ExternalEvidenceRequest = Field(
        default_factory=ExternalEvidenceRequest,
        description="Optional request to run web search before retrying.",
    )


class TripleFix(BaseModel):
    """A single actionable correction to an RDF facts or ontology graph.

    ``incorrect_value`` / ``correct_value`` are plain strings; encoding follows
    deployment ``llm_graph_format`` and GRAPH FORMAT INSTRUCTION.
    """

    text_fragment: str = Field(
        description="Exact quote from source text justifying this change"
    )

    action: Literal["ADD", "REMOVE", "REPLACE"] = Field(
        description=(
            "Type of fix:\n"
            "- ADD: Add new triple, prefix declaration, or missing information\n"
            "- REMOVE: Delete incorrect or redundant triple\n"
            "- REPLACE: Substitute one entity, property, or literal for another"
        )
    )

    severity: Literal["critical", "important", "minor"] = Field(
        description=(
            "Severity level: "
            "'critical' (breaks semantic graph), "
            "'important' (significant gap), or "
            "'minor' (polish). "
            "Note: 'major' will be automatically converted to 'important'."
        )
    )

    @field_validator("severity", mode="before")
    @classmethod
    def normalize_severity(cls, v: str) -> str:
        """Normalize severity values to accepted literals.

        Maps 'major' to 'important' for backward compatibility with prompts
        that use 'major' terminology. This allows the LLM to use either term.
        """
        if isinstance(v, str):
            v_lower = v.lower().strip()
            if v_lower == "major":
                return "important"
            # Return as-is if already valid (will be validated by Literal)
            return v
        return v

    target: str | None = Field(
        default=None,
        description=(
            "What is being fixed. Examples:\n"
            "- 'triple' (for triple-level changes)\n"
            "- 'entity' (replacing cd: with ontology entity)\n"
            "- 'property' (using correct property)\n"
            "- 'datatype' (fixing literal type)\n"
            "- 'prefix' (adding namespace declaration)\n"
            "- 'language_tag' (adding/fixing @lang)"
        ),
    )

    incorrect_value: str | None = Field(
        default=None,
        description=(
            "Current incorrect triple/entity/value (for REMOVE and REPLACE). "
            "Encoding is defined by deployment llm_graph_format and GRAPH FORMAT INSTRUCTION."
        ),
    )

    correct_value: str | None = Field(
        default=None,
        description=(
            "Proposed correct triple/entity/value (for ADD and REPLACE). "
            "Encoding is defined by deployment llm_graph_format and GRAPH FORMAT INSTRUCTION."
        ),
    )

    explanation: str = Field(
        description=(
            "Why this fix is needed. Examples:\n"
            "- 'Missing xsd:date datatype for temporal literal'\n"
            "- 'Namespace prefix fca: not declared'\n"
            "- 'Property onto:decidedBy is canonical, not cd:judgedBy'"
        )
    )

    def to_markdown(self) -> str:
        """Convert this TripleFix to markdown format.

        Returns:
            Markdown formatted string representing this fix.
        """
        lines = []

        # Add the action and target
        action_text = f"**{self.action}**"
        if self.target:
            action_text += f" ({self.target})"
        lines.append(f"- {action_text}")

        # Add text fragment if available
        if self.text_fragment:
            lines.append(f'  - **Source text:** "{self.text_fragment}"')

        # Add incorrect value for REMOVE and REPLACE actions
        if self.action in ["REMOVE", "REPLACE"] and self.incorrect_value:
            lines.append(f"  - **Current (incorrect):** `{self.incorrect_value}`")

        # Add correct value for ADD and REPLACE actions
        if self.action in ["ADD", "REPLACE"] and self.correct_value:
            lines.append(f"  - **Proposed (correct):** `{self.correct_value}`")

        # Add explanation
        if self.explanation:
            lines.append(f"  - **Reason:** {self.explanation}")

        return "\n".join(lines)


def _coerce_critique_score(v: object) -> float:
    """Coerce LLM score output (may be a JSON string) to float."""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return 0.0
    try:
        return float(str(v))
    except (TypeError, ValueError):
        return 0.0


class OntologyCritiqueReport(BaseModel):
    """Report from ontology update critique process."""

    success: bool = Field(
        description="True if the presented ontology is appropriate, complete, consistent and represents well the domain of the provided text, False otherwise."
    )
    score: float = Field(
        ge=0,
        le=100,
        description="Score 0-100 for how well the presented ontology serves as the ontology for the document. 0 is the worst, 100 is the best.",
    )

    @field_validator("score", mode="before")
    @classmethod
    def coerce_score(cls, v: object) -> float:
        return _coerce_critique_score(v)

    actionable_ontology_fixes: list[TripleFix] = Field(
        default_factory=list,
        description=(
            "List of specific fixes to correct the ontology graph. "
            "For each fix, provide text evidence, action type, and relevant triples "
            "in deployment graph syntax (Turtle or JSON-LD per output instructions)."
        ),
    )

    systemic_critique_summary: str = Field(
        default="",
        description="A high-level summary of systemic deficiencies in the ontology (e.g., poor hierarchy structure, redundant concepts, lack of appropriate granularity, or general failures in Domain Coverage). This addresses strategic issues beyond individual term fixes.",
    )
    external_evidence_request: ExternalEvidenceRequest = Field(
        default_factory=ExternalEvidenceRequest,
        description="Optional request to run web search before retrying.",
    )


class FactsCritiqueReport(BaseModel):
    success: bool = Field(
        description="True if the facts triples fully represent the document, False otherwise."
    )

    score: float = Field(
        ge=0,
        le=100,
        description=(
            "Score 0-100 for how well the triples of facts represent the original document. "
            "0 is the worst, 100 is the best."
        ),
    )

    @field_validator("score", mode="before")
    @classmethod
    def coerce_score(cls, v: object) -> float:
        return _coerce_critique_score(v)

    actionable_triple_fixes: list[TripleFix] = Field(
        default_factory=list,
        description=(
            "List of specific fixes to correct the facts graph. "
            "For each fix, provide text evidence, action type, and relevant triples "
            "in deployment graph syntax (Turtle or JSON-LD per output instructions)."
        ),
    )

    systemic_critique_summary: str = Field(
        default="",
        description=(
            "A high-level, non-itemized summary of systemic or pattern-based issues identified across the facts graph.\n"
            "Focus on strategic problems rather than individual triple fixes, such as:\n"
            "- Consistent failure to extract certain data types (e.g., dates, currencies)\n"
            "- Structural patterns like creating entities instead of reusing existing ontology entities\n"
            "- Repeated misinterpretation of specific ontology properties or classes\n"
            "- Missing coverage of entire categories of information\n\n"
            "This guides strategic improvements to the fact-extraction process."
        ),
    )
    external_evidence_request: ExternalEvidenceRequest = Field(
        default_factory=ExternalEvidenceRequest,
        description="Optional request to run web search before retrying.",
    )


class ExternalEvidenceHit(BaseModel):
    """Normalized external evidence hit metadata."""

    title: str = Field(default="")
    url: str = Field(default="")
    snippet: str = Field(default="")
    domain: str = Field(default="")


class ExternalEvidencePlan(BaseModel):
    """Structured plan for optional external evidence retrieval."""

    should_search: bool = Field(
        default=False,
        description="Whether external evidence retrieval should run for this node.",
    )
    rationale: str = Field(
        default="",
        description="Short explanation of why search is or is not needed.",
    )
    intent: Literal[
        "none",
        "definition",
        "disambiguation",
        "standard",
        "verification",
        "background",
    ] = Field(
        default="none",
        description="Primary reason for searching external evidence.",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence in the decision."
    )
    queries: list[str] = Field(
        default_factory=list, description="Targeted search queries."
    )

    @field_validator("queries", mode="before")
    @classmethod
    def normalize_queries(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            return []
        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            query = " ".join(item.split()).strip()
            if query:
                normalized.append(query)
        return normalized


class ExternalEvidenceCacheEntry(BaseModel):
    """Node-scoped external evidence planning/fetch outputs."""

    plan: ExternalEvidencePlan = Field(default_factory=ExternalEvidencePlan)
    hits: list[ExternalEvidenceHit] = Field(default_factory=list)
    text: str = Field(default="")
    source_count: int = Field(default=0, ge=0)
    domains: list[str] = Field(default_factory=list)


class OntologyRenderReport(BaseModel):
    """Ontology rendering output with optional search decision."""

    ontology: Ontology = Field(description="Rendered ontology payload.")
    external_evidence_request: ExternalEvidenceRequest = Field(
        default_factory=ExternalEvidenceRequest,
        description="Optional request to run web search before retrying.",
    )


class Suggestions(BaseModel):
    """Report from knowledge graph critique process.

    Attributes:
        systemic_critique_summary: A compilation of general improvement suggestions.
        actionable_fixes: An itemized list of concrete suggestions for improvement.
    """

    actionable_fixes: list[TripleFix] = Field(
        default_factory=list,
        description="An itemized list of concrete suggestions for improvement.",
    )

    systemic_critique_summary: str = Field(
        default="", description="A general improvement suggestion."
    )

    @classmethod
    def from_critique_report(
        cls, critique: OntologyCritiqueReport | FactsCritiqueReport
    ) -> "Suggestions":
        """Create Suggestions from any critique report.

        Args:
            critique: Either an OntologyCritiqueReport or FactsCritiqueReport to convert.

        Returns:
            Suggestions object with actionable fixes and systemic critique summary.
        """
        fixes = getattr(critique, "actionable_triple_fixes", None)
        if fixes is None:
            fixes = getattr(critique, "actionable_ontology_fixes", None)
        if fixes is None:
            raise ValueError(f"Unsupported critique report type: {type(critique)}")
        actionable_fixes = fixes

        return cls(
            actionable_fixes=actionable_fixes,
            systemic_critique_summary=critique.systemic_critique_summary,
        )

    def to_markdown(self) -> str:
        """Convert actionable fixes and systemic critique summary to a unified markdown block.

        Returns:
            Markdown formatted string with both actionable fixes and systemic critique summary.
        """
        result = ""

        # Add systemic critique summary if available
        if self.systemic_critique_summary:
            result += "## Systemic Critique Summary\n\n"
            result += self.systemic_critique_summary + "\n\n"

        # Add actionable fixes if available
        if self.actionable_fixes:
            result += "## Actionable Fixes\n\n"

            for i, fix in enumerate(self.actionable_fixes, 1):
                result += f"{i}. {fix.to_markdown()}"
                if i < len(self.actionable_fixes):
                    result += "\n\n"

        return result
