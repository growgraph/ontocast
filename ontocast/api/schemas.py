"""Pydantic models for the OntoCast HTTP API."""

from pydantic import BaseModel, Field

from ontocast._version import __version__


class HealthOkResponse(BaseModel):
    status: str = "healthy"
    version: str = __version__
    llm_provider: str | None = None


class HealthErrorResponse(BaseModel):
    status: str = "unhealthy"
    error: str


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
        default_factory=lambda: ["text", "json", "pdf", "markdown"]
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


class ProcessOkResponse(BaseModel):
    status: str = "success"
    data: ProcessResultData
    metadata: ProcessResultMetadata


class ProcessErrorResponse(BaseModel):
    status: str = "error"
    error: str
    error_type: str | None = None
    error_code: str | None = None
    error_details: dict | None = None


class OntologyMutationResponse(BaseModel):
    status: str = "success"
    iri: str
    ontology_id: str | None = None
    version: str | None = None
    hash: str | None = None


class OntologyDeleteResponse(BaseModel):
    status: str = "success"
    iri: str
