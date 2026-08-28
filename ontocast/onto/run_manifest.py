"""Per-document record of what a batch run cost and how it was configured.

The pipeline already computes all of this -- ``BudgetTracker`` accumulates it and
the HTTP path returns it in ``ProcessResultMetadata`` -- but a ``ontocast
process`` run logged it at INFO and then dropped it, so a finished batch left
its TTL output with no record of the model, the settings, or the tokens that
produced it. Written beside the facts dump, one file per document.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

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

    The 2026-08 matsci ablation compared ``--max-visits 1`` against
    ``--max-visits 2`` — but call accounting later proved the critic never ran
    in the second arm, and no artifact recorded what the run received, so
    flag-lost versus flag-not-passed was undecidable. An arm must be auditable
    from its own dump.
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

    A benchmark directory's *name* used to be the only record of its
    ``--exclude-sections`` — and the 2026-08 case7/case8 volume difference took
    a ``git blame`` over ``config/settings.py`` to attribute to the
    ``bibliography_mode`` default flip, instead of a diff of two manifests.
    """

    target_sections: list[str] | None = None
    exclude_sections: list[str] | None = None
    summarize_sections: list[str] | None = None
    summary_max_sentences: int | None = None
    bibliography_mode: str | None = None


class RunManifest(BaseModel):
    """What produced one document's dump, and what it cost."""

    source: str = Field(description="Input file name.")
    line_number: int | None = Field(
        default=None, description="1-based line, for JSONL inputs."
    )
    ontocast_version: str
    render_mode: str
    loops: RunManifestLoops | None = None
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
            "so the two differed ~3x on the 2026-08 matsci runs with nothing "
            "explaining it."
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
