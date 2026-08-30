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


class RunManifestLoops(BaseModel):
    """The effective per-unit loop budgets the run actually used.

    A run whose ``--max-visits`` never reached the loop is indistinguishable
    from one that used it, unless the effective budget is written down: call
    accounting can show the critic did not run, but not whether the flag was
    lost or never passed. A run must be auditable from its own dump.
    """

    max_visits: int = Field(
        description="Render/critic budget per unit; at 1 the LLM critic never runs."
    )
    max_critic_visits: int | None = Field(
        default=None,
        description="Critic attempts per render attempt; None couples to max_visits.",
    )
    llm_repair_visits: int = Field(
        description="Finding-driven repair renders allowed per unit."
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


class RunManifestCritic(BaseModel):
    """What an LLM critic decided, and on what evidence.

    One record per loop: ``critic`` summarizes the facts loop,
    ``ontology_critic`` the ontology loop. The facts loop once accepted a
    render on ``critique.success or critique.score > 90`` -- a score the model
    was asked for with no rubric and no statement of the threshold. Whether
    such a gate is calibrated is a question about the score distribution, and
    until this existed no artifact recorded a single score -- the answer had to
    be mined out of the LLM disk cache, which only worked because caching
    happened to be on. The ontology loop still runs
    that gate (backed there by a scoring rubric whose top band it demands),
    and this record is how its accept rate gets measured before the gate is
    recalibrated.

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
        which is the default at ``MAX_VISITS=1``.
    """
    attempts = [
        attempt
        for unit_attempts in telemetry.values()
        for attempt in unit_attempts
        if attempt.kind == "critic"
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
