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


class RunManifest(BaseModel):
    """What produced one document's dump, and what it cost."""

    source: str = Field(description="Input file name.")
    line_number: int | None = Field(
        default=None, description="1-based line, for JSONL inputs."
    )
    ontocast_version: str
    render_mode: str
    current_domain: str
    doc_iri: str | None = None
    tenant: str | None = None
    project: str | None = None
    llm: RunManifestLLM
    budget: BudgetTracker
    ontology_triples: int = 0
    facts_triples: int = 0
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
