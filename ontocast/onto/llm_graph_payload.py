"""Strict LLM wire coercion for RDF graph fields on canonical Pydantic models."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Annotated, Any

from pydantic import BeforeValidator, ValidationInfo

from ontocast.onto.enum import LLMGraphFormat
from ontocast.onto.rdfgraph import RDFGraph

llm_graph_format_ctx: ContextVar[LLMGraphFormat] = ContextVar(
    "llm_graph_format", default=LLMGraphFormat.TURTLE
)


def _coerce_turtle_graph_payload(value: Any) -> RDFGraph:
    if isinstance(value, RDFGraph):
        return value
    if isinstance(value, (dict, list)):
        raise ValueError(
            "llm_graph_format=turtle expects a Turtle string for graph fields, "
            "not a JSON object. Provide @prefix declarations and triples as one string."
        )
    if isinstance(value, str):
        return RDFGraph._from_str(value)
    raise TypeError(
        f"llm_graph_format=turtle: graph field must be a string, got {type(value).__name__}"
    )


def _coerce_jsonld_graph_payload(value: Any) -> RDFGraph:
    if isinstance(value, RDFGraph):
        return value
    if isinstance(value, dict):
        return RDFGraph._from_jsonld_obj(value)
    if isinstance(value, list):
        return RDFGraph._from_jsonld_obj(value)
    if isinstance(value, str):
        if RDFGraph._is_jsonld_str(value):
            return RDFGraph._from_str(value)
        raise ValueError(
            "llm_graph_format=jsonld expects a compact JSON-LD object with "
            '"@context" and "@graph", not a Turtle string.'
        )
    raise TypeError(
        f"llm_graph_format=jsonld: graph field must be a JSON-LD object, got {type(value).__name__}"
    )


def coerce_llm_graph_wire(value: Any, info: ValidationInfo) -> RDFGraph:
    """Coerce LLM wire payloads to RDFGraph using validation context or ContextVar."""
    ctx = info.context if info.context else {}
    fmt = ctx.get("llm_graph_format") or llm_graph_format_ctx.get()
    if fmt == LLMGraphFormat.TURTLE:
        return _coerce_turtle_graph_payload(value)
    return _coerce_jsonld_graph_payload(value)


LLMGraphWire = Annotated[RDFGraph, BeforeValidator(coerce_llm_graph_wire)]
