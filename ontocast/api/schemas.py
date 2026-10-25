"""Pydantic models for the OntoCast HTTP API."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from ontocast._version import __version__


class HealthOkResponse(BaseModel):
    status: str = "healthy"
    version: str = __version__
    llm_provider: str | None = None


class HealthErrorResponse(BaseModel):
    status: str = "unhealthy"
    error: str


class ConfigResponse(BaseModel):
    version: str = __version__
    settings: dict[str, Any] = Field(
        description=(
            "Every setting the server was started with, keyed by environment "
            "variable, credentials redacted. Per-request parameters of "
            "/process and /process_unit override some of these for that "
            "request only."
        ),
    )
    chunking_mode: str = Field(
        description=(
            "Mode the chunker runs in: CHUNK_CHUNKING_MODE, or 'naive' when "
            "semantic chunking is unavailable in this environment."
        ),
    )


class InfoResponse(BaseModel):
    name: str = "ontocast"
    version: str = __version__
    description: str = (
        "Agentic ontology assisted framework for semantic triple extraction"
    )
    capabilities: list[str] = Field(
        default_factory=lambda: ["text-to-triples", "ontology-extraction"]
    )
    input_types: list[str] = Field(
        default_factory=lambda: ["txt", "json", "pdf", "pptx"]
    )
    output_types: list[str] = Field(default_factory=lambda: ["turtle", "json"])
    llm_cache: dict | None = Field(
        default=None,
        description="In-memory and on-disk LLM cache statistics when available.",
    )
    max_concurrent_processes: int | None = Field(
        default=None,
        description="Configured cap on concurrent /process handlers, if any.",
    )


class FlushOkResponse(BaseModel):
    status: str = "success"
    message: str


class StatusErrorBody(BaseModel):
    status: str = "error"
    error: str
    error_type: str | None = None
    error_code: str | None = None


class ProcessResultData(BaseModel):
    facts: str
    ontology: str | None = Field(
        default=None,
        deprecated=True,
        description=(
            "Deprecated singular ontology payload; always null. Use ontology_artifacts."
        ),
    )
    ontology_artifacts: list[dict] = Field(default_factory=list)


class ProcessResultMetadata(BaseModel):
    status: str | None = None
    chunks_processed: int
    chunks_remaining: int
    budget: dict
    retrieval_metrics: dict = Field(default_factory=dict)
    unit_context_sources: dict[str, dict[int, list[str]]] = Field(
        default_factory=dict,
        description=(
            "Ontology IRIs each unit's prompt context was drawn from, by lane "
            "('ontology', 'facts') and unit index."
        ),
    )
    facts_repairs: dict[int, list[dict]] = Field(
        default_factory=dict,
        description=(
            "Deterministic machine repairs applied per content unit, keyed by "
            "unit index. Lets a consumer tell machine-altered triples from what "
            "the model asserted."
        ),
    )
    failed_units: list[dict] = Field(
        default_factory=list,
        description=(
            "Content units that produced no output, with the stage and reason. "
            "Empty on a fully successful run."
        ),
    )
    improvement_suggestions: list[str] = Field(
        default_factory=list,
        description=(
            "Advisory notes from the structural check and consistency critic "
            "(orphan terms, potential cross-ontology conflicts). Advisory only: "
            "nothing in the pipeline acts on them."
        ),
    )
    facts_conformance: dict = Field(
        default_factory=dict,
        description=(
            "Validation summary for the returned facts graph: whether SHACL "
            "ran and the graph conforms, counts by finding kind and by SHACL "
            "constraint component, and the LLM-free repairs applied. Set by "
            "/process and /process_unit alike; empty when the gate did not run."
        ),
    )
    facts_validation_findings: list[dict] = Field(
        default_factory=list,
        description=(
            "Residual findings behind the summary, after every repair stage. "
            "A consumer that needs to know *which* nodes are non-conformant "
            "reads these."
        ),
    )
    facts_gate_repairs: list[dict] = Field(
        default_factory=list,
        description=(
            "LLM-free repairs applied to the merged graph by the gate "
            "(shape-driven retyping, code resolution, placeholder pruning). "
            "Per-unit repairs are in facts_repairs."
        ),
    )
    ontology_versions: list[dict] = Field(
        default_factory=list,
        description=(
            "Ontology versions this run wrote (role 'created' or 'updated', "
            "with iri, version_iri, ontology_id, version, prior_version_iris) "
            "and read without changing (role 'read', with iri and version_iri)."
        ),
    )


class ProcessOkResponse(BaseModel):
    status: Literal["success"] = "success"
    data: ProcessResultData
    metadata: ProcessResultMetadata


class ProcessErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    error: str
    error_type: str | None = None
    error_code: str | None = None
    error_details: dict | None = None


class ProcessItemResult(BaseModel):
    """One uploaded document's outcome."""

    filename: str
    status_code: int = Field(
        description="200 when the document was extracted, 422 when it could not "
        "be, 500 on an unexpected error."
    )
    outcome: ProcessOkResponse | ProcessErrorResponse = Field(discriminator="status")


class ProcessBatchResponse(BaseModel):
    """``/process`` and ``/process_unit``: one result per document, in upload order."""

    results: list[ProcessItemResult]


class OntologyMutationResponse(BaseModel):
    status: str = "success"
    iri: str
    ontology_id: str
    version: str
    version_iri: str


class OntologyVersionSummary(BaseModel):
    """The header of one stored version."""

    version_iri: str
    version: str
    prior_version_iris: list[str]
    created_at: datetime
    modified_at: datetime
    removed: bool = False


class OntologySummary(BaseModel):
    """One catalog ontology with its head(s)."""

    iri: str
    ontology_id: str
    title: str
    description: str | None = None
    versions: int
    heads: list[OntologyVersionSummary]
    diverged: bool = False


class OntologyListResponse(BaseModel):
    """Every ontology in the partition's catalog."""

    status: str = "success"
    ontologies: list[OntologySummary]


class OntologyVersionsResponse(BaseModel):
    """The version DAG of one ontology, newest first."""

    status: str = "success"
    iri: str
    ontology_id: str
    versions: list[OntologyVersionSummary]


class OntologyDeleteResponse(BaseModel):
    status: str = "success"
    iri: str


class OntologyVersionDeleteResponse(BaseModel):
    """One version tombstoned; the ontology's new heads."""

    status: str = "success"
    version_iri: str
    heads: list[OntologyVersionSummary]


class OntologyDiffResponse(BaseModel):
    """Content triples added and removed between two versions."""

    status: str = "success"
    from_iri: str
    to_iri: str
    added: list[str]
    removed: list[str]
    added_count: int
    removed_count: int


class ShapesMutationResponse(BaseModel):
    """One SHACL shapes document stored in the tenant's shapes partition."""

    status: str = "success"
    graph_uri: str
    triples: int


class ShapesDeleteResponse(BaseModel):
    status: str = "success"
    graph_uri: str


class ShapesListResponse(BaseModel):
    """Named graphs held in the shapes partition, and their merged size."""

    status: str = "success"
    graph_uris: list[str]
    triples: int
