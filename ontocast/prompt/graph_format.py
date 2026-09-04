"""Graph format profiles: unified prompt, context, and parser configuration."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from contextvars import Token
from dataclasses import dataclass
from typing import TypeVar

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

from ontocast.onto.enum import LLMGraphFormat, OntologyChapterFormat
from ontocast.onto.llm_graph_payload import llm_graph_format_ctx
from ontocast.onto.ontology_condense import (
    CondenseReport,
    TextCaps,
    condense_graph_for_prompt,
)
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.triple_index import TripleIndex, build_triple_index
from ontocast.prompt.facts_guidelines import format_facts_operational_guidelines
from ontocast.prompt.graph_index import render_index_table, render_indexed_turtle
from ontocast.prompt.llm_json_schema import format_instructions_for_model
from ontocast.prompt.term_sheet import build_ontology_term_sheet

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class IndexedChapter:
    """A prompt chapter and the triple ids it hands the model.

    The two travel together because they must not drift: the index is only
    meaningful for the exact graph state the text was rendered from, and the
    resolver checks that before it will delete anything.
    """

    text: str
    index: TripleIndex


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

Generate a graph patch that modifies the existing graph incrementally.
Do not replace the entire graph. Do not emit raw UPDATE query syntax or query-language keywords.

Emit exactly two graph fields at the TOP LEVEL of your JSON response:

- `insert_graph` — the triples to ADD
- `delete_graph` — the triples to REMOVE, written to match the stored triples exactly

Both are optional: omit a field, or leave it empty, when there is nothing to add or remove.
Most updates only add, so `insert_graph` alone is the normal answer. There is no wrapper
object and no list of operations — never nest these fields inside another field.

IMPORTANT: both fields ALWAYS contain plain triples — never wrap them in DELETE DATA { } or
INSERT DATA { } blocks. That is update-query syntax and will fail validation.
"""

_OUTPUT_INSTRUCTION_GRAPH_UPDATE_TURTLE_GRAPH = """

Provide `insert_graph` and `delete_graph` each as a **single Turtle string** with:
- `@prefix` declarations for every namespace used in that string
- Only the triples to insert or delete (no comments)
- NEVER use UPDATE query syntax (`INSERT DATA`, `DELETE DATA`, bare `PREFIX` lines) in this field
- Only `@prefix` lines and triples — parseable as plain Turtle, not as an UPDATE query

Shape of the whole response:

```json
{
  "insert_graph": "@prefix ex: <https://example.org/onto#> .\\nex:Foo a owl:Class .",
  "delete_graph": ""
}
```
"""

_OUTPUT_INSTRUCTION_GRAPH_UPDATE_JSONLD_GRAPH = """

Provide `insert_graph` and `delete_graph` each as a compact JSON-LD **object**
(not a string) with:

1. "@context": a map of every prefix alias used to its full namespace IRI.
   Always declare rdf, rdfs, owl, xsd, schema, the facts prefix (e.g. cd), and any
   domain ontology prefixes referenced.
2. "@graph": an array of subject nodes. Each node MUST have "@id" (compact IRI) and SHOULD
   include "@type" plus all predicate-value pairs for that subject grouped in one object.
3. Use compact IRIs (`prefix:local`) throughout - never expand to full URIs in the body.
4. Typed literals MUST use the value/type form: {"@value": "...", "@type": "xsd:date"}.
   Language-tagged literals use {"@value": "...", "@language": "en"}.
5. No comments, no trailing prose - output strictly valid JSON.
6. NEVER use UPDATE query syntax or Turtle ^^/@prefix inside JSON values.

Shape of the whole response — `insert_graph` sits at the top level, and its object is
closed with `}` before the response's final `}`:

```json
{
  "insert_graph": {
    "@context": {"owl": "http://www.w3.org/2002/07/owl#", "ex": "https://example.org/onto#"},
    "@graph": [
      {"@id": "ex:Foo", "@type": "owl:Class", "rdfs:label": {"@value": "foo", "@language": "en"}},
      {"@id": "ex:bar", "@type": "owl:ObjectProperty", "rdfs:domain": {"@id": "ex:Foo"}}
    ]
  }
}
```
"""

_CRITIQUE_ADDRESSING_INSTRUCTION = """\n\n
# HOW TO ADDRESS A STATEMENT

Every statement in the graph chapters above carries an id -- the bracketed number
before it, or its row in the TRIPLE INDEX table.

1. To REMOVE a statement, put its id in `triple_ids`. Leave `correct_value` empty.
2. To REPLACE a statement, put the id(s) you are replacing in `triple_ids` and put
   the replacement in `correct_value`.
3. To ADD, leave `triple_ids` empty and put the new statement in `correct_value`.

Cite the id. Do NOT retype an existing statement into `incorrect_value`: a retyped
statement has to match the stored one exactly, and a single differing predicate,
prefix, or literal form silently discards the whole fix. The id cannot be wrong.

Cite several ids in one fix when a change affects several statements about the same
subject. Cite only ids you can see; never guess a number.
"""

_OUTPUT_INSTRUCTION_CRITIQUE_TURTLE = (
    _CRITIQUE_ADDRESSING_INSTRUCTION
    + """\n\n
# GRAPH FORMAT INSTRUCTION (LLM_GRAPH_FORMAT=turtle)

The deployment emits RDF graph fixes in Turtle syntax.
Provide `correct_value` as a **string** containing valid Turtle: `@prefix` declarations
when needed, then one or more triples.
Example: "@prefix schema: <https://schema.org/> . cd:alice schema:worksFor cd:acme ."
"""
)

_OUTPUT_INSTRUCTION_CRITIQUE_JSONLD = (
    _CRITIQUE_ADDRESSING_INSTRUCTION
    + """\n\n
# GRAPH FORMAT INSTRUCTION (LLM_GRAPH_FORMAT=jsonld)

Render output uses embedded JSON-LD objects for graph fields, but critique fixes use a **string**
containing JSON for one subject node.
Provide `correct_value` as a **string** with valid JSON for one subject node
(inline `@context` or compact IRIs only):
Example: "{\\"@context\\": {\\"schema\\": \\"https://schema.org/\\"}, \\"@id\\": \\"cd:alice\\", \\"schema:worksFor\\": {\\"@id\\": \\"cd:acme\\"}}"
Use `{"@value": "...", "@type": "xsd:date"}` for typed literals and `{"@value": "...", "@language": "en"}`
for language-tagged literals. Never use Turtle ^^ syntax inside these JSON strings.
"""
)


def _fence_lang(fmt: LLMGraphFormat) -> str:
    """Code-fence language tag for a graph serialized in ``fmt``."""
    return "ttl" if fmt == LLMGraphFormat.TURTLE else "json"


@dataclass(frozen=True)
class GraphFormatProfile:
    """Prompt, context serialization, and parser configuration for one LLM graph format."""

    format: LLMGraphFormat
    #: Syntax of the ``# ONTOLOGY`` chapter built by :meth:`format_ontology_chapter`.
    #: ``INHERIT`` follows :attr:`format`; ``TURTLE`` decouples the read-only
    #: context from the wire the model writes on. Nothing else on the profile
    #: -- output instructions, the facts chapter, parsing -- reads it.
    ontology_chapter_format: OntologyChapterFormat = OntologyChapterFormat.INHERIT

    @property
    def renders_term_sheet(self) -> bool:
        """Whether the ontology chapter is a term sheet rather than a graph."""
        return self.ontology_chapter_format == OntologyChapterFormat.TERM_SHEET

    @property
    def ontology_chapter_discriminator(self) -> str:
        """What actually determines the chapter text, for a memo key.

        Two profiles that render the same chapter must share a memo entry -- a
        JSON-LD profile pinned to Turtle and a plain Turtle profile produce the
        same bytes -- so the wire, not the setting, is the discriminator for a
        graph chapter. A term sheet is not a serialization of the graph at all,
        and reports a Turtle wire only because that is the cheaper of the two it
        is not, so it needs its own name here or it would be served the Turtle
        chapter.
        """
        if self.renders_term_sheet:
            return str(OntologyChapterFormat.TERM_SHEET)
        return str(self.ontology_chapter_wire)

    @property
    def ontology_chapter_wire(self) -> LLMGraphFormat:
        """The syntax :meth:`format_ontology_chapter` serializes in.

        Meaningless when :attr:`renders_term_sheet` -- a term sheet is not a
        serialization of the graph -- and reported as Turtle there so a caller
        that only wants the cheaper-of-the-two answer is not misled.
        """
        if self.ontology_chapter_format in (
            OntologyChapterFormat.TURTLE,
            OntologyChapterFormat.TERM_SHEET,
        ):
            return LLMGraphFormat.TURTLE
        return self.format

    def context_fence_lang(self) -> str:
        return _fence_lang(self.format)

    def serialize_graph_for_prompt(
        self, graph: RDFGraph, *, wire: LLMGraphFormat | None = None
    ) -> str:
        """Serialize ``graph`` for a prompt chapter.

        Args:
            graph: The graph to serialize.
            wire: Syntax to use; defaults to the profile's own format.
        """
        fmt = self.format if wire is None else wire
        if fmt == LLMGraphFormat.TURTLE:
            return graph.serialize_canonical_turtle()
        return graph.serialize_compact_jsonld_for_prompt()

    def format_ontology_chapter(
        self,
        graph: RDFGraph,
        *,
        suffix: str = "",
        max_triples: int | None = None,
        text_caps: TextCaps | None = None,
        on_report: Callable[[CondenseReport], None] | None = None,
    ) -> str:
        """Serialize the ontology chapter, condensing it toward ``max_triples``.

        This is the one point every ontology chapter passes through -- both unit
        loops, render and critique, the shared snapshot and the ontology loop's
        mutable working graph -- so it is where the prompt budget is enforced.

        The chapter is written in :attr:`ontology_chapter_wire`, which the
        deployment may pin to Turtle independently of the wire format: here the
        ontology is context the model reads, not a graph it patches, so its
        syntax is free to be the denser one. The indexed variant the ontology
        critic uses stays on the wire format -- its output is a patch against
        the very statements it reads.

        ``suffix`` is built by the caller from the uncondensed graph, which stays
        correct: the index only names terms by ``rdfs:label`` and their
        domain/range, none of which condensing drops.

        ``text_caps`` bounds the individual literals before either rendering.
        The triple budget is a count and says nothing about how long one
        ``rdfs:comment`` may be, so a chapter well inside it can still be
        unbounded; capping here covers every chapter the pipeline builds.

        ``on_report`` is handed what condensing and capping actually did. A cap
        whose effect cannot be read back is a cap nobody can size: whether a
        catalog reaches one at all is a property of that catalog, not something
        a default can be chosen for in advance.
        """
        condensed, report = condense_graph_for_prompt(graph, max_triples, text_caps)
        if on_report is not None:
            on_report(report)
        if self.renders_term_sheet:
            body = build_ontology_term_sheet(condensed)
            return f"\n\n# ONTOLOGY\n\n{body}\n" + suffix
        wire = self.ontology_chapter_wire
        body = self.serialize_graph_for_prompt(condensed, wire=wire)
        chapter = f"\n\n# ONTOLOGY\n\n```{_fence_lang(wire)}\n{body}\n```\n"
        return chapter + suffix

    def _indexed_body(self, graph: RDFGraph, index: TripleIndex) -> str:
        """Body plus ids, in whichever way this format can carry them.

        Turtle takes the ids inline, where the critic is already reading. JSON-LD
        cannot: rule 7 of its output instruction demands strictly valid JSON, and
        an ``[12]`` marker inside a node object would contradict it -- so the ids
        ride in a table after the fenced block instead. One resolver serves both.
        """
        if self.format == LLMGraphFormat.TURTLE:
            return render_indexed_turtle(graph, index)
        return self.serialize_graph_for_prompt(graph)

    def _index_appendix(self, graph: RDFGraph, index: TripleIndex) -> str:
        if self.format == LLMGraphFormat.TURTLE or index.is_empty:
            return ""
        return "\n" + render_index_table(graph, index) + "\n"

    def format_facts_chapter_indexed(self, graph: RDFGraph) -> IndexedChapter:
        """The facts chapter with a citable id on every triple.

        Scope is the whole unit graph: for facts the graph *is* the unit's
        product, so every triple in it is the critic's to change.
        """
        graph.sanitize_prefixes_namespaces()
        index = build_triple_index(graph)
        body = self._indexed_body(graph, index)
        return IndexedChapter(
            text=(
                "\n\n# SEMANTIC GRAPH OF FACTS\n"
                "The following facts were extracted. "
                f"{self._addressing_note(index, len(graph))}\n\n"
                f"```{self.context_fence_lang()}\n{body}\n```\n"
                f"{self._index_appendix(graph, index)}"
            ),
            index=index,
        )

    def format_ontology_chapter_indexed(
        self,
        graph: RDFGraph,
        *,
        scope: RDFGraph | None = None,
        suffix: str = "",
        max_triples: int | None = None,
    ) -> IndexedChapter:
        """The ontology chapter with ids, assigned **after** condensing.

        Condensing drops triples to fit the prompt budget. Numbering before it
        would hand out ids for statements the critic never sees, and a delete
        cited against one of those is ordered blind.

        ``scope`` narrows which statements are addressable. The ontology critic
        is shown the retrieved snapshot *plus* this unit's delta but owns only
        the delta, so passing the delta here leaves catalog statements visible
        and unciteable -- a delete that would propagate to every document
        sharing the terminal simply cannot be expressed.

        Text caps deliberately do not reach here. The critic cites statements by
        index and may order a delete against one; a clipped literal would not
        match the statement it names, so the delete would silently miss. The
        indexed chapter shows the statements exactly as they are stored.
        """
        condensed, _ = condense_graph_for_prompt(graph, max_triples)
        condensed.sanitize_prefixes_namespaces()
        index = build_triple_index(condensed, scope=scope)
        body = self._indexed_body(condensed, index)
        return IndexedChapter(
            text=(
                "\n\n# ONTOLOGY\n"
                f"{self._addressing_note(index, len(condensed))}\n\n"
                f"```{self.context_fence_lang()}\n{body}\n```\n"
                f"{self._index_appendix(condensed, index)}"
                f"{suffix}"
            ),
            index=index,
        )

    def _addressing_note(self, index: TripleIndex, shown: int) -> str:
        """Explain the ids, and the absence of an id, in one sentence each."""
        if shown == 0:
            # Nothing to address, so nothing to explain. A note naming the index
            # for an empty chapter invites a citation there is no statement for.
            return ""
        if len(index) == 0:
            return (
                "Every statement below is existing context: read it, but none "
                "of it can be cited or removed."
            )
        if self.format == LLMGraphFormat.TURTLE:
            note = (
                "The bracketed number before each statement is its id; cite ids "
                "in `triple_ids` to change or remove a statement."
            )
            if len(index) < shown:
                note += (
                    " Statements marked `[-]` are existing context: read them, "
                    "but they cannot be cited or removed."
                )
            return note
        note = (
            "The TRIPLE INDEX below lists the id of each addressable statement; "
            "cite ids in `triple_ids` to change or remove one."
        )
        if len(index) < shown:
            note += (
                " Statements absent from the table are existing context: read "
                "them, but they cannot be cited or removed."
            )
        return note

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
        quantity_fallback_vocabulary: dict[str, str] | None = None,
        search_guidelines: str = "",
    ) -> str:
        return format_facts_operational_guidelines(
            facts_namespace=facts_namespace,
            domain_ontologies_clause=domain_ontologies_clause,
            jsonld=self.format == LLMGraphFormat.JSONLD,
            quantity_fallback_vocabulary=quantity_fallback_vocabulary,
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


_PROFILES: dict[tuple[LLMGraphFormat, OntologyChapterFormat], GraphFormatProfile] = {
    (fmt, chapter): GraphFormatProfile(format=fmt, ontology_chapter_format=chapter)
    for fmt in LLMGraphFormat
    for chapter in OntologyChapterFormat
}


def get_graph_format_profile(
    fmt: LLMGraphFormat,
    *,
    ontology_chapter_format: OntologyChapterFormat = OntologyChapterFormat.INHERIT,
) -> GraphFormatProfile:
    """The profile for a wire format.

    Args:
        fmt: Syntax the model emits graph payloads in.
        ontology_chapter_format: Syntax of the ``# ONTOLOGY`` chapter in the
            prompts built from this profile. The facts loop passes the
            deployment's ``ONTOLOGY_CHAPTER_FORMAT``; callers that leave it at
            ``INHERIT`` get a chapter in ``fmt``.
    """
    return _PROFILES[(fmt, ontology_chapter_format)]
