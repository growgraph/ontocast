"""Format-bound JSON Schema generation for canonical LLM report models."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue

from ontocast.onto.enum import LLMGraphFormat
from ontocast.prompt.web_grounding import (
    WEB_SEARCH_REQUEST_DEF,
    WEB_SEARCH_REQUEST_FIELD,
)

_GRAPH_FIELD_NAMES = frozenset({"graph", "semantic_graph"})

_GRAPH_WIRE_DESCRIPTIONS: dict[LLMGraphFormat, str] = {
    LLMGraphFormat.TURTLE: (
        "Plain Turtle string: @prefix declarations and triples only. "
        "Never UPDATE query syntax (INSERT DATA, DELETE DATA, PREFIX)."
    ),
    LLMGraphFormat.JSONLD: (
        "Compact JSON-LD object with @context and @graph. "
        "Never UPDATE query syntax or Turtle ^^/@prefix inside JSON."
    ),
}


def _wire_graph_json_schema(fmt: LLMGraphFormat) -> JsonSchemaValue:
    if fmt == LLMGraphFormat.TURTLE:
        return {
            "type": "string",
            "description": _GRAPH_WIRE_DESCRIPTIONS[fmt],
        }
    return {
        "type": "object",
        "additionalProperties": True,
        "description": _GRAPH_WIRE_DESCRIPTIONS[fmt],
    }


def _patch_graph_field_schemas(node: Any, fmt: LLMGraphFormat) -> None:
    """Replace permissive RDFGraph unions with a single wire shape for known fields."""
    if not isinstance(node, dict):
        return

    properties = node.get("properties")
    if isinstance(properties, dict):
        for name, prop in properties.items():
            if name in _GRAPH_FIELD_NAMES and isinstance(prop, dict):
                patched = _wire_graph_json_schema(fmt)
                if "description" in prop:
                    patched["description"] = prop["description"]
                if "title" in prop:
                    patched["title"] = prop["title"]
                properties[name] = patched
            else:
                _patch_graph_field_schemas(prop, fmt)

    for key in ("$defs", "definitions", "items", "allOf", "anyOf", "oneOf"):
        child = node.get(key)
        if isinstance(child, dict):
            for sub in child.values():
                _patch_graph_field_schemas(sub, fmt)
        elif isinstance(child, list):
            for sub in child:
                _patch_graph_field_schemas(sub, fmt)
        elif child is not None:
            _patch_graph_field_schemas(child, fmt)


class FormatBoundJsonSchemaGenerator(GenerateJsonSchema):
    """Emit Turtle string or JSON-LD object schemas for RDF graph wire fields."""

    def __init__(self, fmt: LLMGraphFormat, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._fmt = fmt

    def generate(self, schema, mode="validation"):
        result = super().generate(schema, mode=mode)
        _patch_graph_field_schemas(result, self._fmt)
        return result


def _schema_generator_for(fmt: LLMGraphFormat) -> type[FormatBoundJsonSchemaGenerator]:
    class _BoundGenerator(FormatBoundJsonSchemaGenerator):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(fmt, *args, **kwargs)

    return _BoundGenerator


def _strip_web_search_from_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove optional web-search request field and its def from a report schema."""
    stripped = dict(schema)
    properties = stripped.get("properties")
    if isinstance(properties, dict):
        properties = dict(properties)
        properties.pop(WEB_SEARCH_REQUEST_FIELD, None)
        stripped["properties"] = properties

    required = stripped.get("required")
    if isinstance(required, list):
        stripped["required"] = [
            name for name in required if name != WEB_SEARCH_REQUEST_FIELD
        ]

    for defs_key in ("$defs", "definitions"):
        defs = stripped.get(defs_key)
        if isinstance(defs, dict) and WEB_SEARCH_REQUEST_DEF in defs:
            defs = dict(defs)
            defs.pop(WEB_SEARCH_REQUEST_DEF, None)
            stripped[defs_key] = defs

    return stripped


def schema_for_model(
    model: type[BaseModel],
    fmt: LLMGraphFormat,
    *,
    web_search_enabled: bool = True,
) -> dict[str, Any]:
    """JSON Schema for ``model`` with graph fields locked to ``fmt`` wire encoding."""
    schema = model.model_json_schema(schema_generator=_schema_generator_for(fmt))
    if not web_search_enabled:
        schema = _strip_web_search_from_schema(schema)
    return schema


def format_instructions_for_model(
    model: type[BaseModel],
    fmt: LLMGraphFormat,
    *,
    web_search_enabled: bool = True,
) -> str:
    """LangChain-compatible format instructions using format-bound schema."""
    schema = schema_for_model(model, fmt, web_search_enabled=web_search_enabled)
    schema_str = json.dumps(schema, indent=2)
    return (
        "The output should be formatted as a JSON instance that conforms to the "
        "JSON schema below.\n\n"
        "As an example, for the schema "
        '{"properties": {"foo": {"title": "Foo", "description": "a list of strings", '
        '"type": "array", "items": {"type": "string"}}}, "required": ["foo"]}\n'
        'the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. '
        'The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\n'
        "Here is the output schema:\n"
        f"```\n{schema_str}\n```"
    )
