"""Graph format profiles: unified prompt, context, and parser configuration."""

from __future__ import annotations

from contextlib import AbstractContextManager
from contextvars import Token
from dataclasses import dataclass
from typing import TypeVar

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

from ontocast.onto.enum import LLMGraphFormat
from ontocast.onto.llm_graph_payload import llm_graph_format_ctx
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.prompt.facts_guidelines import format_facts_operational_guidelines
from ontocast.prompt.llm_json_schema import format_instructions_for_model

T = TypeVar("T", bound=BaseModel)

# --- Output instructions (single-format per profile; graph-update = base + suffix) ---

# Fresh render: one block per deployment format (Turtle ontology/facts, or JSON-LD).
# Graph update: shared base (_OUTPUT_INSTRUCTION_GRAPH_UPDATE_BASE) plus a format suffix
# (_OUTPUT_INSTRUCTION_GRAPH_UPDATE_TURTLE_GRAPH or _JSONLD_GRAPH) appended by
# GraphFormatProfile.render_update_output_instruction().

_OUTPUT_INSTRUCTION_ONTOLOGY_TTL = """\n\n
# OUTPUT INSTRUCTION

1. The ontology `graph` field must be a single Turtle string.
2. Define all prefixes for every namespace used (rdf, rdfs, owl, schema, domain prefixes, etc.).
"""

_OUTPUT_INSTRUCTION_FACTS_TTL = """\n\n
# OUTPUT INSTRUCTION

1. The `semantic_graph` field must be a single Turtle string.
2. Define all prefixes for every namespace used (rdf, rdfs, owl, xsd, schema, cd, domain prefixes, etc.).
3. Use only @prefix declarations and triples; no comments.
"""

_OUTPUT_INSTRUCTION_JSONLD = """\n\n
# OUTPUT INSTRUCTION

Provide each RDF graph field as a compact JSON-LD **object** (not a string) with:

1. "@context": a map of every prefix alias used to its full namespace IRI. Always declare
   rdf, rdfs, owl, xsd, schema, the facts prefix (e.g. cd), and any domain ontology prefixes.
2. "@graph": an array of subject nodes. Each node MUST have "@id" (compact IRI) and SHOULD
   include "@type" plus all predicate-value pairs for that subject grouped in one object.
3. Use compact IRIs (`prefix:local`) throughout - never expand to full URIs in the body.
4. Typed literals MUST use the value/type form: {"@value": "2024-01-15", "@type": "xsd:date"}.
   Language-tagged literals use {"@value": "...", "@language": "en"}.
5. Multi-valued predicates use a JSON array of objects/values.
6. Object references use {"@id": "prefix:local"} (or a plain compact IRI string when unambiguous).
7. No comments, no trailing prose - output strictly valid JSON.
8. Never use Turtle syntax (no ^^, no @prefix) inside JSON values.
"""

_OUTPUT_INSTRUCTION_GRAPH_UPDATE_BASE = """\n\n
# OUTPUT INSTRUCTION

Generate structured graph patch operations that modify the existing graph incrementally.
Do not replace the entire graph. Do not emit raw UPDATE query syntax or query-language keywords.

Follow the Pydantic schema exactly. Use `triple_operations` only: each entry has `type`
(`insert` or `delete`) and `graph` (plain triples for that operation, encoded per the
graph-format instructions below).

IMPORTANT: The `type` field (`insert` or `delete`) signals which triples to add or remove.
The `graph` field ALWAYS contains plain triples — never wrap them in DELETE DATA { } or
INSERT DATA { } blocks. That is update-query syntax and will fail validation.
"""

_OUTPUT_INSTRUCTION_GRAPH_UPDATE_TURTLE_GRAPH = """

For each `TripleOp.graph` field, provide a **single Turtle string** with:
- `@prefix` declarations for every namespace used in that operation
- Only the triples to insert or delete (no comments)
- NEVER use UPDATE query syntax (`INSERT DATA`, `DELETE DATA`, bare `PREFIX` lines) in this field
- Only `@prefix` lines and triples — parseable as plain Turtle, not as an UPDATE query
"""

_OUTPUT_INSTRUCTION_GRAPH_UPDATE_JSONLD_GRAPH = """

For each `TripleOp.graph` field, provide a compact JSON-LD **object** (not a string) with:

1. "@context": a map of every prefix alias used to its full namespace IRI.
   Always declare rdf, rdfs, owl, xsd, schema, the facts prefix (e.g. cd), and any
   domain ontology prefixes referenced by the operation.
2. "@graph": an array of subject nodes. Each node MUST have "@id" (compact IRI) and SHOULD
   include "@type" plus all predicate-value pairs for that subject grouped in one object.
3. Use compact IRIs (`prefix:local`) throughout - never expand to full URIs in the body.
4. Typed literals MUST use the value/type form: {"@value": "...", "@type": "xsd:date"}.
   Language-tagged literals use {"@value": "...", "@language": "en"}.
5. No comments, no trailing prose - output strictly valid JSON.
6. NEVER use UPDATE query syntax or Turtle ^^/@prefix inside JSON values.
"""

_OUTPUT_INSTRUCTION_CRITIQUE_TURTLE = """\n\n
# GRAPH FORMAT INSTRUCTION (LLM_GRAPH_FORMAT=turtle)

The deployment emits RDF graph fixes in Turtle syntax.
For each `incorrect_value` and `correct_value` in actionable fixes, provide a **string**
containing valid Turtle: `@prefix` declarations when needed, then one or more triples.
Example: "@prefix ex: <http://example.org/> . ex:alice ex:worksFor ex:acme ."
"""

_OUTPUT_INSTRUCTION_CRITIQUE_JSONLD = """\n\n
# GRAPH FORMAT INSTRUCTION (LLM_GRAPH_FORMAT=jsonld)

Render output uses embedded JSON-LD objects for graph fields, but critique fixes use **strings**
containing JSON for one subject node each.
For each `incorrect_value` and `correct_value`, provide a **string** with valid JSON for one
subject node (inline `@context` or compact IRIs only):
Example: "{\\"@context\\": {\\"ex\\": \\"http://example.org/\\"}, \\"@id\\": \\"ex:alice\\", \\"ex:worksFor\\": {\\"@id\\": \\"ex:acme\\"}}"
Use `{"@value": "...", "@type": "xsd:date"}` for typed literals and `{"@value": "...", "@language": "en"}`
for language-tagged literals. Never use Turtle ^^ syntax inside these JSON strings.
"""


@dataclass(frozen=True)
class GraphFormatProfile:
    """Prompt, context serialization, and parser configuration for one LLM graph format."""

    format: LLMGraphFormat

    def context_fence_lang(self) -> str:
        return "ttl" if self.format == LLMGraphFormat.TURTLE else "json"

    def serialize_graph_for_prompt(self, graph: RDFGraph) -> str:
        if self.format == LLMGraphFormat.TURTLE:
            return graph.serialize_canonical_turtle()
        return graph.serialize_compact_jsonld_for_prompt()

    def format_ontology_chapter(self, graph: RDFGraph, *, suffix: str = "") -> str:
        body = self.serialize_graph_for_prompt(graph)
        chapter = f"\n\n# ONTOLOGY\n\n```{self.context_fence_lang()}\n{body}\n```\n"
        return chapter + suffix

    def format_facts_chapter(self, graph: RDFGraph) -> str:
        body = self.serialize_graph_for_prompt(graph)
        return (
            "\n\n# SEMANTIC GRAPH OF FACTS\n"
            "The following facts were extracted\n\n"
            f"```{self.context_fence_lang()}\n{body}\n```\n"
        )

    def render_fresh_output_instruction(self, *, target: str = "facts") -> str:
        if self.format == LLMGraphFormat.JSONLD:
            return _OUTPUT_INSTRUCTION_JSONLD
        if target == "ontology":
            return _OUTPUT_INSTRUCTION_ONTOLOGY_TTL
        return _OUTPUT_INSTRUCTION_FACTS_TTL

    def render_update_output_instruction(self) -> str:
        base = _OUTPUT_INSTRUCTION_GRAPH_UPDATE_BASE
        if self.format == LLMGraphFormat.JSONLD:
            return base + _OUTPUT_INSTRUCTION_GRAPH_UPDATE_JSONLD_GRAPH
        return base + _OUTPUT_INSTRUCTION_GRAPH_UPDATE_TURTLE_GRAPH

    def critique_graph_instruction(self) -> str:
        if self.format == LLMGraphFormat.JSONLD:
            return _OUTPUT_INSTRUCTION_CRITIQUE_JSONLD
        return _OUTPUT_INSTRUCTION_CRITIQUE_TURTLE

    def facts_operational_guidelines(
        self,
        *,
        facts_namespace: str,
        domain_ontologies_clause: str,
        search_guidelines: str = "",
    ) -> str:
        return format_facts_operational_guidelines(
            facts_namespace=facts_namespace,
            domain_ontologies_clause=domain_ontologies_clause,
            jsonld=self.format == LLMGraphFormat.JSONLD,
            search_guidelines=search_guidelines,
        )

    def format_instructions(
        self,
        report_cls: type[BaseModel],
        *,
        web_search_enabled: bool = True,
    ) -> str:
        return format_instructions_for_model(
            report_cls,
            self.format,
            web_search_enabled=web_search_enabled,
        )

    def parse_report(self, report_cls: type[T], text: str) -> T:
        token = llm_graph_format_ctx.set(self.format)
        try:
            parser = PydanticOutputParser(pydantic_object=report_cls)
            return parser.parse(text)
        finally:
            llm_graph_format_ctx.reset(token)

    def llm_graph_format_context(self) -> AbstractContextManager[LLMGraphFormat]:
        return _LLMGraphFormatContext(self.format)


class _LLMGraphFormatContext(AbstractContextManager[LLMGraphFormat]):
    def __init__(self, fmt: LLMGraphFormat) -> None:
        self._fmt = fmt
        self._token: Token[LLMGraphFormat] | None = None

    def __enter__(self) -> LLMGraphFormat:
        self._token = llm_graph_format_ctx.set(self._fmt)
        return self._fmt

    def __exit__(self, *exc: object) -> None:
        if self._token is not None:
            llm_graph_format_ctx.reset(self._token)


_PROFILES: dict[LLMGraphFormat, GraphFormatProfile] = {
    LLMGraphFormat.TURTLE: GraphFormatProfile(format=LLMGraphFormat.TURTLE),
    LLMGraphFormat.JSONLD: GraphFormatProfile(format=LLMGraphFormat.JSONLD),
}


def get_graph_format_profile(fmt: LLMGraphFormat) -> GraphFormatProfile:
    return _PROFILES[fmt]


def critique_graph_format_instruction(llm_graph_format: LLMGraphFormat) -> str:
    """Backward-compatible helper."""
    return get_graph_format_profile(llm_graph_format).critique_graph_instruction()


# Backward-compatible aliases (avoid importing graph_format from common.py at load time)
output_instruction_ttl = _PROFILES[
    LLMGraphFormat.TURTLE
].render_fresh_output_instruction(target="ontology")
output_instruction_empty = _PROFILES[
    LLMGraphFormat.TURTLE
].render_fresh_output_instruction(target="facts")
output_instruction_jsonld = _PROFILES[
    LLMGraphFormat.JSONLD
].render_fresh_output_instruction()
output_instruction_graph_update = _PROFILES[
    LLMGraphFormat.TURTLE
].render_update_output_instruction()
output_instruction_graph_update_jsonld = _PROFILES[
    LLMGraphFormat.JSONLD
].render_update_output_instruction()
output_instruction_critique_turtle = _PROFILES[
    LLMGraphFormat.TURTLE
].critique_graph_instruction()
output_instruction_critique_jsonld = _PROFILES[
    LLMGraphFormat.JSONLD
].critique_graph_instruction()
