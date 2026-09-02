"""Per-document record of what a batch run cost and how it was configured.

The pipeline already computes all of this -- ``BudgetTracker`` accumulates it and
the HTTP path returns it in ``ProcessResultMetadata`` -- but a ``ontocast
process`` run logged it at INFO and then dropped it, so a finished batch left
its TTL output with no record of the model, the settings, or the tokens that
produced it. Written beside the facts dump, one file per document.
"""

from __future__ import annotations

from statistics import median
from typing import Any

from pydantic import BaseModel, Field

from ontocast.onto.model import LoopAttempt
from ontocast.onto.state import BudgetTracker
from ontocast.util.graph_metrics import GraphShapeMetrics


class RunManifestLLM(BaseModel):
    """The provider settings that shaped the output.

    Mirrors the discriminators :func:`ontocast.tool.llm.llm_cache_config` puts
    in the cache key, so two dumps whose manifests agree here were produced by
    the same model under the same generation settings.
    """

    provider: str
    model_name: str
    temperature: float | None = None
    think: bool | None = None
    num_ctx: int | None = None
    num_predict: int | None = None
    requests_per_second: float | None = Field(
        default=None,
        description=(
            "Provider request pacing the run used; None = unpaced. A "
            "throttled run (llm/rate_limited or llm/timeouts in "
            "budget.counters) has directional cost figures only."
        ),
    )
    max_retries: int | None = None


class RunManifestLoops(BaseModel):
    """The effective per-unit loop budgets the run actually used.

    A run whose ``--max-visits`` never reached the loop is indistinguishable
    from one that used it, unless the effective budget is written down: call
    accounting can show the critic did not run, but not whether the flag was
    lost or never passed. A run must be auditable from its own dump.
    """

    max_visits: int = Field(
        description="Retries of a *failed* fresh extraction; not a critic switch."
    )
    max_critic_visits: int | None = Field(
        default=None,
        description="Deprecated alias for the facts pass count; None when unset.",
    )
    facts_critic_passes: int = Field(
        default=0, description="Review-and-patch passes allowed per facts unit."
    )
    ontology_critic_passes: int = Field(
        default=0, description="Review-and-patch passes allowed per ontology unit."
    )


class RunManifestSelection(BaseModel):
    """The content-selection settings the run actually used.

    An output directory's *name* used to be the only record of which sections
    a run was given, so a volume difference between two runs took a ``git
    blame`` over ``config/settings.py`` to attribute to a default flip, instead
    of a diff of two manifests.
    """

    target_sections: list[str] | None = None
    exclude_sections: list[str] | None = None
    summarize_sections: list[str] | None = None
    summary_max_sentences: int | None = None
    bibliography_mode: str | None = None
    labeled_units: int | None = Field(
        default=None,
        description=(
            "Content units that carried a section_label after classification. "
            "Together with unlabeled_units this records whether a section "
            "filter could act at all: an exclusion list against mostly "
            "unlabeled units is a no-op the arm name would never reveal."
        ),
    )
    unlabeled_units: int | None = None
    section_label_histogram: dict[str, int] | None = Field(
        default=None,
        description="section_label -> unit count, '(unlabeled)' included.",
    )


class RunManifestValidationConfig(BaseModel):
    """The validation-facing configuration the run actually used.

    Arms are launched by env vars nothing records; every row here is a knob
    whose setting changed a measured outcome in some past run and had to be
    reconstructed from logs. The manifest is the record.
    """

    context_from_units: bool | None = None
    json_mode: bool | None = None
    shapes_prompt_contract: str | None = None
    shapes_prompt_selection: bool | None = Field(
        default=None,
        description=(
            "Whether the conformance chapter was selected per unit by the "
            "ontology-context join. 'auto' resolves by catalog size, so two "
            "arms with identical settings can differ here; this records the "
            "behavior that actually ran."
        ),
    )
    shapes_triples: int | None = Field(
        default=None,
        description=(
            "Size of the merged shapes partition at dump time; 0 or absent "
            "means the gate and the prompt contract had no shapes."
        ),
    )
    shacl_inference: str | None = None
    numeric_coverage_mandatory: bool | None = None
    facts_user_instruction_chars: int | None = Field(
        default=None,
        description=(
            "Length of the per-request facts_user_instruction; 0 = none. The "
            "text itself stays out of the manifest -- deployment guidance can "
            "carry secrets and the dump is shareable."
        ),
    )


class RunManifestCritic(BaseModel):
    """What an LLM critic decided, and on what evidence.

    One record per loop: ``critic`` summarizes the facts loop,
    ``ontology_critic`` the ontology loop. The facts loop once accepted a
    render on ``critique.success or critique.score > 90`` -- a score the model
    was asked for with no rubric and no statement of the threshold. Whether
    such a gate is calibrated is a question about the score distribution, and
    until this existed no artifact recorded a single score -- the answer had to
    be mined out of the LLM disk cache, which only worked because caching
    happened to be on. Both loops now gate on the deterministic findings and
    record what that score gate *would* have said, so the change can be judged
    from a distribution rather than an argument.

    ``fixes_*`` describe what the critique actually did, which is a different
    question from what it proposed: a critique can name a dozen corrections and
    change nothing, and for a long time nothing here could tell the difference.

    A run must carry its own evidence for the decisions it made.
    """

    calls: int = Field(default=0, description="Critic calls billed for this document.")
    accepted: int = Field(
        default=0, description="Calls whose verdict let the unit exit the loop."
    )
    score_min: float | None = None
    score_median: float | None = None
    score_max: float | None = None
    score_histogram: dict[str, int] = Field(
        default_factory=dict,
        description="Decile buckets ('70-79') -> count. Empty when no call ran.",
    )
    fix_severity_histogram: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Proposed TripleFix severities summed over the document. The "
            "materiality gate reads 'critical'; a run where 'important' swamps "
            "it is a run whose severity labels carry no signal."
        ),
    )
    fix_action_severity_histogram: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "The same fixes keyed 'ACTION:severity'. Read this before "
            "concluding anything from the severity histogram: a REMOVE cannot "
            "block acceptance at any severity, so 'critical' alone conflates "
            "fixes that gate a render with fixes that never could."
        ),
    )
    accept_reason_histogram: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Why each verdict landed: 'clean', 'mandatory_findings', "
            "'critic_critical'. Separates a critic that found nothing from one "
            "overruled by the deterministic lane -- indistinguishable in the "
            "accepted count alone."
        ),
    )
    incumbent_accepted: int = Field(
        default=0,
        description=(
            "Calls the retired score gate would have accepted. Recorded so "
            "replacing that gate can be judged against a distribution."
        ),
    )
    patch_passes: int = Field(
        default=0, description="Critique applications attempted, LLM-free."
    )
    fixes_applied: int = Field(
        default=0, description="Proposed fixes that reached the graph."
    )
    fixes_noop: int = Field(
        default=0,
        description=(
            "Fixes that removed exactly what they re-added. High against "
            "`fixes_applied` means the critic is producing motion, not "
            "corrections."
        ),
    )
    patches_rolled_back: int = Field(
        default=0,
        description=(
            "Passes undone for leaving the unit worse. Non-zero means the "
            "critique is provoking data-destroying edits."
        ),
    )
    triples_deleted: int = Field(default=0)
    triples_inserted: int = Field(default=0)


def summarize_loop(
    telemetry: dict[int, list[LoopAttempt]],
) -> RunManifestCritic:
    """Reduce per-unit attempt logs to the document's critic record.

    Args:
        telemetry: ``AgentState.facts_loop_telemetry`` or
            ``AgentState.ontology_loop_telemetry`` -- unit index to its
            ordered attempt log.

    Returns:
        The document-level critic summary; all-zero when no critic call ran,
        which is what a zero pass budget buys.
    """
    attempts = [
        attempt
        for unit_attempts in telemetry.values()
        for attempt in unit_attempts
        if attempt.kind == "critic"
    ]
    patches = [
        attempt
        for unit_attempts in telemetry.values()
        for attempt in unit_attempts
        if attempt.kind == "critic_patch"
    ]
    scores = sorted(a.score for a in attempts if a.score is not None)
    histogram: dict[str, int] = {}
    for score in scores:
        bucket = int(score // 10) * 10
        key = f"{bucket}-{bucket + 9}"
        histogram[key] = histogram.get(key, 0) + 1
    severities: dict[str, int] = {}
    action_severities: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for attempt in attempts:
        for severity, count in attempt.severity_counts.items():
            severities[severity] = severities.get(severity, 0) + count
        for key, count in attempt.action_severity_counts.items():
            action_severities[key] = action_severities.get(key, 0) + count
        if attempt.accept_reason:
            reasons[attempt.accept_reason] = reasons.get(attempt.accept_reason, 0) + 1
    return RunManifestCritic(
        calls=len(attempts),
        accepted=sum(1 for a in attempts if a.success),
        score_min=scores[0] if scores else None,
        score_median=median(scores) if scores else None,
        score_max=scores[-1] if scores else None,
        score_histogram=histogram,
        fix_severity_histogram=severities,
        fix_action_severity_histogram=dict(sorted(action_severities.items())),
        accept_reason_histogram=dict(sorted(reasons.items())),
        incumbent_accepted=sum(1 for a in attempts if a.incumbent_accepted),
        patch_passes=len(patches),
        fixes_applied=sum(a.n_fixes_applied for a in patches),
        fixes_noop=sum(a.n_fixes_noop for a in patches),
        patches_rolled_back=sum(1 for a in patches if a.patch_rolled_back),
        triples_deleted=sum(a.n_triples_deleted for a in patches),
        triples_inserted=sum(a.n_triples_inserted for a in patches),
    )


class RunManifest(BaseModel):
    """What produced one document's dump, and what it cost."""

    source: str = Field(description="Input file name.")
    line_number: int | None = Field(
        default=None, description="1-based line, for JSONL inputs."
    )
    ontocast_version: str
    render_mode: str
    loops: RunManifestLoops | None = None
    critic: RunManifestCritic | None = None
    ontology_critic: RunManifestCritic | None = None
    validation_config: RunManifestValidationConfig | None = None
    ontology_reduce_metrics: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "``AgentState.ontology_reduce_metrics``: apply/partition counters "
            "plus the reduce policies' evidence -- minted_duplicates and "
            "their pairs, deletes_dropped_unredeclared, apply_deletes_no_match, "
            "fresh_ontologies_merged. The case10 sampling run computed all of "
            "these and recorded none, because no manifest field carried them."
        ),
    )
    selection: RunManifestSelection | None = None
    graph_metrics: GraphShapeMetrics | None = Field(
        default=None,
        description=(
            "Connectivity of the serialized facts graph — fragmentation "
            "regressions surface per document instead of needing offline "
            "analysis."
        ),
    )
    current_domain: str
    doc_iri: str | None = None
    tenant: str | None = None
    project: str | None = None
    llm: RunManifestLLM
    budget: BudgetTracker
    ontology_triples: int = 0
    facts_triples: int = 0
    facts_triples_serialized: int = Field(
        default=0,
        description=(
            "Triples in the provenance-stripped graph the .facts.ttl dump "
            "actually holds. facts_triples counts the raw aggregated graph, "
            "so the two routinely differ by a factor of several, with nothing "
            "in either number explaining it."
        ),
    )
    retrieval_metrics: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "``AgentState.retrieval_metrics`` for this document -- the same "
            "payload ``/process`` returns in ``ProcessResultMetadata``. Without "
            "it a batch run carried no retrieval telemetry at all, which also "
            "left ONTOLOGY_PATCH_DUMP_ONTOLOGY_RANKS with no reader outside the "
            "HTTP path. Keys are enumerated by "
            ":class:`~ontocast.onto.enum.RetrievalMetric`."
        ),
    )
