"""Pydantic models for the OntoCast HTTP API."""

from pydantic import BaseModel, Field


class HealthOkResponse(BaseModel):
    status: str = "healthy"
    version: str = "0.1.1"
    llm_provider: str | None = None


class HealthErrorResponse(BaseModel):
    status: str = "unhealthy"
    error: str


class InfoResponse(BaseModel):
    name: str = "ontocast"
    version: str = "0.1.1"
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
        description="Deprecated singular ontology payload; use ontology_artifacts.",
    )
    ontology_artifacts: list[dict] = Field(default_factory=list)


class ProcessResultMetadata(BaseModel):
    status: str | None = None
    chunks_processed: int
    chunks_remaining: int
    budget: dict
    retrieval_metrics: dict = Field(default_factory=dict)


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
