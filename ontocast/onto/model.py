import re
from collections.abc import Sequence
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ontocast.onto.llm_graph_payload import LLMGraphWire
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import GraphUpdate, TripleOp


def _coerce_free_text(v: object) -> str:
    """Coerce LLM free-text output to a string.

    Several providers answer a single-string field with a bulleted list. The
    content is usable as-is, so join it rather than rejecting the whole report
    and burning a retry.
    """
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, list | tuple):
        return "\n".join(part for part in (str(item).strip() for item in v) if part)
    return str(v)


class BasePydanticModel(BaseModel):
    """Shared base for the pipeline's Pydantic models.

    Carries no behaviour of its own since the JSON save/load helpers were
    removed with their only consumer; it is kept as the common ancestor the
    state and report models already declare.
    """

    def __init__(self, **kwargs):
        """Initialize the model with given keyword arguments."""
        super().__init__(**kwargs)


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

    @field_validator("rationale", mode="before")
    @classmethod
    def coerce_rationale(cls, v: object) -> str:
        return _coerce_free_text(v)

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
    """Graph update rendering output with optional search decision.

    The wire shape is deliberately flat, and deliberately the same shape as
    :class:`FactsRenderReport`: two sibling graph fields, no wrapper object and
    no list. The previous shape nested the graph inside
    ``graph_update.triple_operations[]``, and a *singleton* list holding one
    long JSON-LD document is a shape models close wrongly -- measured at 20/24
    malformed above ~4k characters on gpt-5-mini, versus 0/22 whenever the list
    happened to hold two or more operations and the ``},{`` boundary reinforced
    the array frame. Nothing here may reintroduce a list-of-one around a large
    payload.

    The internal :class:`GraphUpdate` keeps its ordered ``TripleOp`` list; this
    is a wire encoding, not the patch model. See :meth:`to_graph_update`.
    """

    insert_graph: LLMGraphWire = Field(
        default_factory=RDFGraph,
        description=(
            "Triples to ADD. Encoding is defined by deployment llm_graph_format "
            "and OUTPUT INSTRUCTION. Omit or leave empty when adding nothing."
        ),
    )
    delete_graph: LLMGraphWire = Field(
        default_factory=RDFGraph,
        description=(
            "Triples to REMOVE, matching the stored triples exactly. Encoding is "
            "defined by deployment llm_graph_format and OUTPUT INSTRUCTION. Omit "
            "or leave empty when removing nothing."
        ),
    )
    external_evidence_request: ExternalEvidenceRequest = Field(
        default_factory=ExternalEvidenceRequest,
        description="Optional request to run web search before retrying.",
    )

    def to_graph_update(self) -> GraphUpdate:
        """Compile the wire payload into an ordered patch.

        Deletes are ordered before inserts, and an empty side contributes no
        operation. Interleaving the two within a single render is not
        expressible on this wire: it would only matter for a patch that removes
        and re-adds the same triple, which nets out to nothing.
        """
        operations: list[TripleOp] = []
        if len(self.delete_graph) > 0:
            operations.append(TripleOp(type="delete", graph=self.delete_graph))
        if len(self.insert_graph) > 0:
            operations.append(TripleOp(type="insert", graph=self.insert_graph))
        return GraphUpdate(triple_operations=operations)


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

    triple_ids: list[int] = Field(
        default_factory=list,
        description=(
            "Ids of the statements this fix removes, taken from the bracketed "
            "numbers in the graph chapter. REQUIRED for REMOVE and REPLACE. "
            "Cite the id; never retype the statement."
        ),
    )

    @field_validator("triple_ids", mode="before")
    @classmethod
    def coerce_triple_ids(cls, v: object) -> object:
        """Accept the shapes providers actually return for a list of ints.

        A bare ``12``, a string ``"12"``, a stringified list ``"[12, 13]"`` and a
        list of strings all mean the same thing. This field is the only way a
        REMOVE or REPLACE can address anything, so a formatting quirk here would
        discard the fix -- and, because the report validates as a whole, every
        other fix alongside it.
        """
        if v is None:
            return []
        if isinstance(v, int) and not isinstance(v, bool):
            return [v]
        if isinstance(v, str):
            v = [part for part in re.split(r"[^0-9]+", v) if part]
        if not isinstance(v, (list, tuple, set)):
            return v
        ids: list[int] = []
        for item in v:
            if isinstance(item, bool):
                continue
            if isinstance(item, int):
                ids.append(item)
                continue
            if isinstance(item, str):
                digits = "".join(ch for ch in item if ch.isdigit())
                if digits:
                    ids.append(int(digits))
        return ids

    incorrect_value: str | None = Field(
        default=None,
        description=(
            "Legacy. Prefer `triple_ids`. Only consulted when a REMOVE or "
            "REPLACE cites no ids, because a retyped statement matches the "
            "stored one a minority of the time."
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

    @field_validator("text_fragment", "explanation", mode="before")
    @classmethod
    def coerce_free_text(cls, v: object) -> str:
        """Coerce the two required free-text fields.

        Both are required with no default, so a provider answering either with a
        bulleted list raised and discarded the whole critique report. Deliberately
        not applied to ``incorrect_value``/``correct_value``: those carry graph
        syntax, where joining a list would corrupt the payload rather than
        recover it.
        """
        return _coerce_free_text(v)

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

        # Add the addressed statements for REMOVE and REPLACE actions
        if self.action in ["REMOVE", "REPLACE"]:
            if self.triple_ids:
                cited = ", ".join(str(triple_id) for triple_id in self.triple_ids)
                lines.append(f"  - **Statements:** `[{cited}]`")
            elif self.incorrect_value:
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

    @field_validator("systemic_critique_summary", mode="before")
    @classmethod
    def coerce_systemic_critique_summary(cls, v: object) -> str:
        return _coerce_free_text(v)

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

    @field_validator("systemic_critique_summary", mode="before")
    @classmethod
    def coerce_systemic_critique_summary(cls, v: object) -> str:
        return _coerce_free_text(v)

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

    @field_validator("rationale", mode="before")
    @classmethod
    def coerce_rationale(cls, v: object) -> str:
        """Mirror the coercion on ``ExternalEvidenceRequest.rationale``."""
        return _coerce_free_text(v)

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


class FactsUnitFindingKind(StrEnum):
    """Kinds of deterministic per-unit facts findings."""

    QUARANTINED_LITERAL = "quarantined_literal"
    UNKNOWN_TERM = "unknown_term"
    PROPERTY_ALIAS = "property_alias"
    CLOSED_RANGE_LITERAL = "closed_range_literal"
    LITERAL_TYPE_OBJECT = "literal_type_object"
    NUMERIC_COVERAGE = "numeric_coverage"
    LABEL_ONLY_NUMBER = "label_only_number"
    SCALAR_AS_BOUNDS = "scalar_as_bounds"
    DOMAIN_VIOLATION = "domain_violation"
    #: The render used no term from the ontology it was given. Not a
    #: property of any single triple, so no other kind can express it.
    DOMAIN_ADHERENCE = "domain_adherence"
    #: Not machine-found: a fix the LLM critic proposed, carried through the
    #: same repair pipeline so a rejection costs one rewrite-in-place render
    #: instead of a full re-extraction.
    CRITIC_FIX = "critic_fix"


class UnitFinding(BaseModel):
    """One deterministic, machine-found issue in a rendered unit graph.

    Base shape shared by the facts and ontology validators; the subclasses
    pin ``kind`` to their own enum. Mandatory findings are violations the
    renderer must fix; non-mandatory findings list candidates the renderer
    adjudicates item by item.
    """

    #: Declared here as a plain string so code serving both phases -- the
    #: acceptance policy, the findings prompt block -- can read it without
    #: knowing which enum it came from. Each subclass narrows it to its own
    #: ``StrEnum``, whose members *are* strings, so nothing is widened in
    #: practice and no kind from one phase can satisfy the other's annotation.
    kind: str
    mandatory: bool = True
    message: str
    subject: str = ""
    predicate: str = ""
    value: str = ""
    suggestions: list[str] = Field(default_factory=list)


class FactsUnitFinding(UnitFinding):
    """One deterministic, machine-found issue in a rendered facts graph."""

    kind: FactsUnitFindingKind


class OntologyUnitFindingKind(StrEnum):
    """Kinds of deterministic per-unit ontology findings.

    Computed against the unit's net insert/delete *delta*, never the whole
    working graph — validating snapshot+delta would attribute every
    pre-existing catalog defect to this unit, and the facts ``UNKNOWN_TERM``
    rule is semantically inverted here (minting new terms is the ontology
    renderer's job).
    """

    #: Term minted under a namespace no ontology in the unit's context
    #: declares terms under — the reduce-time partition will drop it as
    #: unattributed, so this predicts silent triple loss.
    FOREIGN_NAMESPACE = "foreign_namespace"
    #: ``owl:Restriction`` blank node with fewer than two meaningful
    #: predicates: it constrains nothing and its subClassOf edge points at
    #: noise.
    DEGENERATE_RESTRICTION = "degenerate_restriction"
    #: Newly declared class/property without ``rdfs:label``/``skos:prefLabel``.
    MISSING_LABEL = "missing_label"
    #: The unit's ``rdfs:subClassOf`` insert closes a cycle through the
    #: snapshot-plus-delta hierarchy.
    SUBCLASS_CYCLE = "subclass_cycle"
    #: A term the catalog knows as a class used as a property, or vice versa.
    ROLE_CONFUSION = "role_confusion"
    #: New term whose label duplicates an existing catalog term's surface
    #: form — likely a re-mint of a concept that should be reused.
    LABEL_COLLISION = "label_collision"
    #: Max-cardinality-1 / functional declaration contradicted by a
    #: min-cardinality >= 2 on the same property.
    CARDINALITY_CONTRADICTION = "cardinality_contradiction"
    #: The unit deletes catalog content whose subject it does not redeclare —
    #: ontology deletes propagate onto shared, versioned catalog terminals
    #: cross-document, so an unowned delete is destructive beyond this unit.
    FOREIGN_DELETE = "foreign_delete"


class OntologyUnitFinding(UnitFinding):
    """One deterministic, machine-found issue in a unit's ontology delta."""

    kind: OntologyUnitFindingKind


def format_findings_for_prompt(
    findings: Sequence[UnitFinding],
    *,
    advisory_heading: str = "## Verify numeric coverage",
) -> str:
    """Render findings as MANDATORY-fixes + advisory blocks for a prompt.

    The default advisory heading is the facts loop's (its only advisory kind
    is numeric coverage) and is part of prompts already in the LLM cache —
    callers with different advisory content pass their own heading rather
    than changing the default.
    """
    mandatory = [finding for finding in findings if finding.mandatory]
    advisory = [finding for finding in findings if not finding.mandatory]
    sections: list[str] = []
    if mandatory:
        lines = [
            "## MANDATORY fixes (deterministic validation — apply every item)",
            "Fix each item by REWRITING the offending term or value in place. "
            "Never resolve a finding by deleting the statement or dropping "
            "extracted data — a response that only removes triples is wrong; "
            "the corrected statement must survive with its subject and value "
            "intact.",
        ]
        for index, finding in enumerate(mandatory, 1):
            line = f"{index}. {finding.message}"
            if finding.suggestions:
                line += " Candidates: " + ", ".join(
                    f"<{suggestion}>" for suggestion in finding.suggestions
                )
            lines.append(line)
        sections.append("\n".join(lines))
    if advisory:
        lines = [advisory_heading]
        lines.extend(finding.message for finding in advisory)
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


class FactsGateRepairKind(StrEnum):
    """Kinds of machine-applied repair at the post-aggregation gate.

    Distinct from :class:`FactsUnitFindingKind`: these are shape-driven and
    apply to the merged graph, where the SHACL report is available.
    """

    SHACL_RETYPE = "shacl_retype"
    SHACL_CODE_RESOLVED = "shacl_code_resolved"
    SHACL_PRUNE = "shacl_prune"
    CODE_RESOLVED = "code_resolved"
    LITERAL_VARIANT_PRUNED = "literal_variant_pruned"


class GraphRepairRecord(BaseModel):
    """One machine-applied deterministic rewrite on a facts graph.

    LLM-free by construction: every repair either rewrites a term the catalog
    already declares or removes a node that asserts nothing. Records what the
    repair passes changed (near-miss predicate rewrites, literal ``rdf:type``
    coercions, shape-driven retyping/pruning) so downstream consumers can
    distinguish machine-altered triples from what the LLM asserted.
    """

    kind: FactsUnitFindingKind | FactsGateRepairKind
    source: str
    target: str
    triple_count: int = 1


class UnitFailure(BaseModel):
    """One content unit that produced no usable output.

    Carried to the document level so a caller can tell "nothing to extract"
    from "every unit failed" -- previously both produced an empty result with
    ``status: success``.
    """

    unit_index: int
    phase: Literal["ontology", "facts", "summarize"]
    stage: str | None = None
    reason: str | None = None


class LoopAttempt(BaseModel):
    """Telemetry record for one attempt inside a per-unit render/critic loop.

        Shared by the facts and the ontology loop — the fields are phase-neutral
        and a record's home (``UnitFactsState.attempt_log`` vs
        ``UnitOntologyState.attempt_log``) says which loop produced it.

        ``n_deterministic_findings`` / ``n_mandatory_findings`` count findings
        against the graph as of this record: for ``llm_repair`` records that is the
        residual *after* the repair render, so summing the last repair record per
        unit yields the true document-level residual.

    ``kind="critic_patch"`` is the LLM-free application of a critique: it costs
        nothing and is recorded separately from the ``critic`` call that produced
        the fixes, so "what the critique cost" and "what it changed" stay
        distinguishable. ``kind="llm_repair"`` is retained for artifacts written by
        earlier releases, when a separate finding-driven repair render existed.
    """

    render_attempt: int = 0
    critic_attempt: int = 0
    kind: Literal["render", "critic", "critic_patch", "llm_repair"] = "render"
    score: float | None = None
    success: bool | None = None
    n_actionable_fixes: int = 0
    #: Why this attempt was accepted or rejected: ``clean``,
    #: ``mandatory_findings``, or ``critic_critical``. The score gate this
    #: replaced produced no recordable reason at all.
    accept_reason: str = ""
    #: Proposed ``TripleFix`` severities for a critic attempt. Recorded because
    #: the materiality gate reads ``critical``, and a severity label produced
    #: from a field description with no rubric is exactly the kind of signal
    #: that has to be watched rather than assumed.
    severity_counts: dict[str, int] = Field(default_factory=dict)
    action_severity_counts: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Proposed fixes keyed 'ACTION:severity'. Severity alone cannot be "
            "read: a REMOVE never blocks acceptance whatever its severity, so "
            "'critical' counts mix fixes that gate a render with fixes that "
            "cannot. Splitting by action is what makes the two distinguishable "
            "from the artifacts."
        ),
    )
    #: What a ``critic_patch`` attempt actually did.
    n_fixes_applied: int = 0
    n_fixes_noop: int = Field(
        default=0,
        description=(
            "Fixes whose delete set and insert set were the same statements. "
            "Counted because a critique made mostly of these is a critic "
            "producing motion rather than corrections."
        ),
    )
    n_triples_deleted: int = 0
    n_triples_inserted: int = 0
    patch_rolled_back: bool = Field(
        default=False,
        description="The pass left the unit worse and was undone whole.",
    )
    patch_delete_capped: bool = Field(
        default=False,
        description=(
            "The delete-share cap fired: the pass kept its inserts and dropped "
            "its deletes."
        ),
    )
    incumbent_accepted: bool | None = Field(
        default=None,
        description=(
            "What the retired score gate would have decided for this critic "
            "attempt, recorded so the switch to a findings-based gate can be "
            "judged against a distribution rather than an argument."
        ),
    )
    n_deterministic_findings: int = 0
    n_mandatory_findings: int = 0
    repair_failed: bool = False
    #: Why the repair render failed. The unit stays SUCCESS -- the pre-repair
    #: graph is intact and usable -- but clearing the failure used to erase the
    #: diagnosis with it, leaving ``repair_failed=True`` and no way to tell a
    #: provider timeout from an unparseable response.
    failure_stage: str | None = None
    failure_reason: str | None = None
    #: The repair render answered the findings prompt by deleting triples
    #: without resolving a mandatory finding, and was rolled back. Counted
    #: document-level so a prompt or validator change that starts provoking
    #: delete-only responses is visible as a number rather than a log line.
    repair_delete_only: bool = False
    triple_count: int = 0
    #: Ontology critic only: triples in this unit's net insert delta at the
    #: time of the record. The unit's own product is the delta, not the
    #: snapshot+delta working graph that ``triple_count`` measures there.
    delta_triple_count: int | None = None
    #: Ontology critic only: proposed fixes whose ``incorrect_value`` names a
    #: term the snapshot declares and this unit's delta does not touch — the
    #: critic litigating pre-existing catalog content the renderer cannot own.
    #: A substring heuristic, so a lower bound.
    n_fixes_targeting_snapshot: int = 0


class FactsValidationFindingKind(StrEnum):
    """Kinds of deterministic post-aggregation facts findings."""

    FUNCTIONAL_VIOLATION = "functional_violation"
    SUSPECT_MULTI_VALUE = "suspect_multi_value"
    DEGENERATE_COREFERENCE = "degenerate_coreference"
    SHACL = "shacl"
    NON_CATALOG_VOCABULARY = "non_catalog_vocabulary"
    DANGLING_REFERENCE = "dangling_reference"
    MIXED_OBJECT_KINDS = "mixed_object_kinds"


class FactsValidationFinding(BaseModel):
    """One invariant violation detected in the aggregated facts graph.

    Error-severity findings of the merge-signature kinds
    (``FUNCTIONAL_VIOLATION``, ``SUSPECT_MULTI_VALUE``,
    ``DEGENERATE_COREFERENCE``) on subjects that resulted from an identity
    merge drive the deterministic un-merge repair (full-cluster pair vetoes
    plus re-aggregation). SHACL findings never drive it — a constraint
    violation says a node is under-specified, not that two entities were
    wrongly identified. Warning findings are telemetry only.
    """

    kind: FactsValidationFindingKind
    severity: Literal["error", "warning"] = "error"
    message: str
    subject: str = ""
    predicate: str = ""
    values: list[str] = Field(default_factory=list)
    component: str = Field(
        default="",
        description=(
            "SHACL constraint component IRI (sh:MinCountConstraintComponent, "
            "…) for SHACL findings; empty otherwise. Grouping by it is what "
            "turns a list of violations into a diagnosis."
        ),
    )
    source_shape: str = Field(
        default="",
        description="SHACL shape that reported the violation; empty otherwise.",
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

    @field_validator("systemic_critique_summary", mode="before")
    @classmethod
    def coerce_systemic_critique_summary(cls, v: object) -> str:
        return _coerce_free_text(v)

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
