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


def turtle_graph_field_description() -> str:
    return (
        "RDF graph as a single Turtle string. Declare all @prefix bindings used. "
        "Include only @prefix lines and triples; no comments. "
        'Example: "@prefix ex: <http://example.org/> . ex:John a ex:Person ."'
    )


def jsonld_graph_field_description() -> str:
    return (
        "Compact JSON-LD object (not a string) with @context mapping every prefix "
        "to its namespace IRI and @graph as an array of subject nodes. "
        'Each node must have "@id". Typed literals use '
        '{"@value": "...", "@type": "xsd:date"}; language tags use '
        '{"@value": "...", "@language": "en"}. Never use Turtle ^^ syntax.'
    )


def triple_op_turtle_graph_description() -> str:
    return (
        "RDF graph for this operation as a single Turtle string with @prefix "
        "declarations and triples to insert or delete."
    )


def triple_op_jsonld_graph_description() -> str:
    return (
        "RDF graph for this operation as a compact JSON-LD object with "
        "@context and @graph (see OUTPUT INSTRUCTION)."
    )


def ontology_graph_turtle_description() -> str:
    return (
        "Domain ontology RDF graph as a single Turtle string. "
        "Declare all prefixes; include rdfs:label and rdfs:comment on new entities."
    )


def ontology_graph_jsonld_description() -> str:
    return (
        "Domain ontology RDF graph as a compact JSON-LD object with "
        "@context and @graph (see OUTPUT INSTRUCTION)."
    )


def semantic_graph_turtle_description() -> str:
    return (
        "Semantic facts graph as a single Turtle string. "
        "Use the cd: facts namespace for new instances per OPERATIONAL GUIDELINES."
    )


def semantic_graph_jsonld_description() -> str:
    return (
        "Semantic facts graph as a compact JSON-LD object with "
        "@context and @graph (see OUTPUT INSTRUCTION)."
    )


def triple_fix_turtle_value_description(field_name: str) -> str:
    return (
        f"{field_name} as a Turtle string: @prefix declarations when needed, "
        "then one or more triples."
    )


def triple_fix_jsonld_value_description(field_name: str) -> str:
    return (
        f"{field_name} as a string containing valid JSON for one subject node "
        "(inline @context or compact IRIs). Use @value/@type for typed literals."
    )
