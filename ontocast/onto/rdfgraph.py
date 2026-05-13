import json
import logging
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping
from contextvars import ContextVar
from typing import Any, Union

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema
from pyld import jsonld
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import NamespaceManager

from ontocast.onto.constants import COMMON_PREFIXES
from ontocast.onto.iri_policy import normalize_namespace_iri, sanitize_prefix_map
from ontocast.util import render_text_hash

logger = logging.getLogger(__name__)

PREFIX_PATTERN = re.compile(r"@prefix\s+(\w+):\s+<[^>]+>\s+\.")
PREFIX_DECLARATION_PATTERN = re.compile(r"@prefix\s+(\w+):\s+<([^>]+)>\s+\.")
# Pattern to match prefix usage: prefix:something (not in @prefix declarations)
PREFIX_USAGE_PATTERN = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*):[^\s]")
INTEGER_TYPED_LITERAL_PATTERN = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"\^\^xsd:integer')
DECIMAL_TYPED_LITERAL_PATTERN = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"\^\^xsd:decimal')
DOUBLE_TYPED_LITERAL_PATTERN = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"\^\^xsd:double')
DATE_TYPED_LITERAL_PATTERN = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"\^\^xsd:date')
# Bare numeric value followed by ^^ datatype — missing surrounding quotes
UNQUOTED_TYPED_LITERAL_PATTERN = re.compile(
    r'(?<!")\b(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\^\^([\w:]+)'
)

# Context variable to store known prefixes during parsing
_known_prefixes_context: ContextVar[dict[str, str] | None] = ContextVar[
    dict[str, str] | None
]("known_prefixes", default=None)


def extract_known_prefixes(
    graph: "RDFGraph",
    extra_prefix: str | None = None,
    extra_namespace: str | None = None,
) -> dict[str, str]:
    """Collect all namespace prefixes from *graph* for use in LLM Turtle output repair.

    Reads both the graph's NamespaceManager bindings and the ``@prefix``
    declarations emitted by ``serialize_canonical_turtle``.  The serializer
    step is necessary because rdflib auto-generates ephemeral names
    (``ns1:``, ``ns2:``, …) for namespaces that have no explicit binding;
    these names appear in the Turtle sent to the LLM but are never stored
    back in the NamespaceManager, so they would be invisible to
    ``_ensure_prefixes`` when repairing the LLM's response.

    Args:
        graph: The RDFGraph to extract prefixes from (typically an ontology
            or facts graph).
        extra_prefix: Optional extra prefix name to register explicitly,
            e.g. from ``Ontology.prefix``.
        extra_namespace: Namespace URI paired with ``extra_prefix``.

    Returns:
        Mapping from prefix names to namespace URI strings.
    """
    known: dict[str, str] = {}

    for prefix, namespace_uri in graph.namespaces():
        if prefix:
            known[prefix] = str(namespace_uri)

    # Serialize to Turtle and capture any additional @prefix declarations,
    # including the ephemeral nsN: names rdflib emits for unbound namespaces.
    try:
        turtle_str = graph.serialize_canonical_turtle()
        for match in PREFIX_DECLARATION_PATTERN.finditer(turtle_str):
            p = match.group(1)
            if p not in known:
                known[p] = match.group(2)
    except Exception:
        pass

    if extra_prefix and extra_namespace:
        known[extra_prefix] = extra_namespace

    return known


class RDFGraph(Graph):
    """Subclass of rdflib.Graph with Pydantic schema support.

    This class extends rdflib.Graph to provide serialization and deserialization
    capabilities for Pydantic models, with special handling for Turtle format.
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type, handler: GetCoreSchemaHandler):
        """Get the Pydantic core schema for this class.

        Args:
            _source_type: The source type.
            handler: The core schema handler.

        Returns:
            A union schema accepting:
            - existing ``RDFGraph`` instances,
            - Turtle / JSON-LD strings (parsed via ``_from_str``),
            - JSON-LD ``dict`` / ``list`` objects (parsed via ``_from_jsonld_obj``).
        """
        return core_schema.union_schema(
            [
                core_schema.is_instance_schema(cls),
                core_schema.chain_schema(
                    [
                        core_schema.str_schema(),
                        core_schema.no_info_plain_validator_function(cls._from_str),
                    ]
                ),
                core_schema.no_info_plain_validator_function(cls._from_any),
            ],
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._to_turtle_str,
                info_arg=False,
                return_schema=core_schema.str_schema(),
            ),
        )

    def __add__(self, other: Union["RDFGraph", Graph, Iterable]) -> "RDFGraph":
        """Addition operator for RDFGraph instances.

        Merges the RDF graphs while maintaining the RDFGraph type.

        Args:
            other: The graph to add to this one.

        Returns:
            RDFGraph: A new RDFGraph containing the merged triples.
        """
        # Create a new RDFGraph instance
        result = RDFGraph()

        # Copy all triples from both graphs
        for triple in self:
            result.add(triple)
        for triple in other:
            result.add(triple)

        # Copy namespace bindings from self
        for prefix, uri in self.namespaces():
            result.bind(prefix, uri)

        # Copy namespace bindings from other if it's a Graph
        if isinstance(other, Graph):
            for prefix, uri in other.namespaces():
                result.bind(prefix, uri)

        return result

    def __iadd__(self, other: Union["RDFGraph", Graph, Iterable]) -> "RDFGraph":
        """In-place addition operator for RDFGraph instances.

        Merges the RDF graphs while maintaining the RDFGraph type and binding prefixes.

        Args:
            other: The graph to add to this one.

        Returns:
            RDFGraph: self after modification.
        """
        # Use __add__ to get the merged result with proper prefix binding
        result = self.__add__(other)

        # Clear current graph and copy the result
        self.remove((None, None, None))  # Remove all triples

        # Copy all triples from result
        for triple in result:
            self.add(triple)

        # Copy namespace bindings from result
        for prefix, uri in result.namespaces():
            self.bind(prefix, uri)

        return self

    def copy(self) -> "RDFGraph":
        """Create a copy of this RDFGraph.

        Returns:
            RDFGraph: A new RDFGraph instance with all triples and namespace bindings copied.
        """
        result = RDFGraph()

        # Copy all triples
        for triple in self:
            result.add(triple)

        # Copy namespace bindings
        for prefix, uri in self.namespaces():
            result.bind(prefix, uri)

        return result

    def __copy__(self) -> "RDFGraph":
        """Ensure shallow copies preserve RDFGraph type."""
        return self.copy()

    def __deepcopy__(self, memo: dict[int, Any]) -> "RDFGraph":
        """Ensure deep copies preserve RDFGraph type."""
        copied = self.copy()
        memo[id(self)] = copied
        return copied

    @staticmethod
    def _ensure_prefixes(turtle_str: str) -> str:
        """Ensure all common prefixes and used custom prefixes are declared in the Turtle string.

        This method:
        1. Adds missing common prefixes (rdf, rdfs, owl, etc.)
        2. Detects prefixes that are used but not declared
        3. Adds declarations for used prefixes if they're available in the context

        Args:
            turtle_str: The input Turtle string.

        Returns:
            str: The Turtle string with all necessary prefixes declared.
        """
        declared_prefixes = set(
            match.group(1) for match in PREFIX_PATTERN.finditer(turtle_str)
        )

        # Add missing common prefixes
        missing_common = {
            prefix: uri
            for prefix, uri in COMMON_PREFIXES.items()
            if prefix not in declared_prefixes
        }

        # Detect prefixes that are used but not declared
        used_prefixes = set()
        for match in PREFIX_USAGE_PATTERN.finditer(turtle_str):
            prefix = match.group(1)
            # Skip if already declared or is a common prefix we're about to add
            if prefix not in declared_prefixes and prefix not in missing_common:
                used_prefixes.add(prefix)

        # Get known prefixes from context (set by caller)
        known_prefixes = _known_prefixes_context.get()
        missing_custom = {}
        if known_prefixes and used_prefixes:
            for prefix in used_prefixes:
                if prefix in known_prefixes:
                    namespace_uri = known_prefixes[prefix]
                    # Format as Turtle prefix declaration
                    if not namespace_uri.startswith("<"):
                        namespace_uri = f"<{namespace_uri}>"
                    missing_custom[prefix] = namespace_uri

        all_missing = {**missing_common, **missing_custom}

        if not all_missing:
            return turtle_str

        prefix_block = (
            "\n".join(
                f"@prefix {prefix}: {uri} ." for prefix, uri in all_missing.items()
            )
            + "\n\n"
        )

        return prefix_block + turtle_str

    @staticmethod
    def _is_jsonld_str(s: str) -> bool:
        """Check if a string appears to be JSON-LD format.

        Args:
            s: The string to check.

        Returns:
            bool: True if the string appears to be JSON-LD.
        """
        s = s.strip()
        if not (s.startswith("{") or s.startswith("[")):
            return False
        try:
            # Try to parse as JSON
            data = json.loads(s)
            # Check if it's a dict/object with @context or @id, or an array containing such objects
            if isinstance(data, dict):
                return "@context" in data or "@id" in data
            elif isinstance(data, list):
                return any(
                    isinstance(item, dict) and ("@context" in item or "@id" in item)
                    for item in data
                )
            return False
        except (json.JSONDecodeError, ValueError):
            return False

    @classmethod
    def _from_str(cls, data_str: str) -> "RDFGraph":
        """Create an RDFGraph instance from a string (Turtle or JSON-LD).

        Automatically detects the format and parses accordingly.

        Args:
            data_str: The input string in Turtle or JSON-LD format.

        Returns:
            RDFGraph: A new RDFGraph instance.
        """
        if cls._is_jsonld_str(data_str):
            return cls._from_jsonld_str(data_str)
        else:
            return cls._from_turtle_str(data_str)

    @classmethod
    def _from_jsonld_obj(cls, jsonld_obj: dict | list) -> "RDFGraph":
        """Create an RDFGraph from a JSON-LD ``dict`` or ``list`` object.

        Used when LLMs emit JSON-LD as a native JSON object in a structured
        response field instead of as an embedded string.

        Args:
            jsonld_obj: Parsed JSON-LD value (object or array of objects).

        Returns:
            RDFGraph: A new RDFGraph instance.
        """
        return cls._from_jsonld_str(json.dumps(jsonld_obj))

    @classmethod
    def _from_any(cls, value: Any) -> "RDFGraph":
        """Dispatch RDFGraph creation from arbitrary structured input.

        Accepts dicts/lists (treated as JSON-LD), strings (Turtle or JSON-LD),
        and existing RDFGraph instances. Used as the catch-all branch in
        the Pydantic schema for graph fields that may receive JSON-LD objects
        from LLM structured output.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, (dict, list)):
            return cls._from_jsonld_obj(value)
        if isinstance(value, str):
            return cls._from_str(value)
        raise TypeError(
            f"Cannot construct RDFGraph from value of type {type(value).__name__}"
        )

    @classmethod
    def _from_turtle_str(cls, turtle_str: str) -> "RDFGraph":
        """Create an RDFGraph instance from a Turtle string.

        This method uses context variables to access known prefixes that may be
        needed to complete missing prefix declarations in the Turtle string.

        Args:
            turtle_str: The input Turtle string.

        Returns:
            RDFGraph: A new RDFGraph instance.
        """
        normalized_turtle = cls._normalize_turtle_input(turtle_str)
        patched_turtle = cls._coerce_invalid_numeric_typed_literals(
            cls._quote_unquoted_typed_literals(cls._ensure_prefixes(normalized_turtle))
        )
        g = cls()
        try:
            g.parse(data=patched_turtle, format="turtle")
            g._sanitize_prefix_boundaries_from_turtle(normalized_turtle)
            return g
        except Exception as parse_error:
            repaired_turtle = cls._repair_common_turtle_issues(
                patched_turtle, parse_error_message=str(parse_error)
            )
            if repaired_turtle == patched_turtle:
                raise
            logger.warning(
                "Recovering malformed Turtle after parse failure: %s",
                str(parse_error),
            )
            repaired_graph = cls()
            repaired_graph.parse(data=repaired_turtle, format="turtle")
            repaired_graph._sanitize_prefix_boundaries_from_turtle(normalized_turtle)
            return repaired_graph

    def _sanitize_prefix_boundaries_from_turtle(self, turtle_str: str) -> None:
        declared_prefixes = {
            match.group(1): match.group(2)
            for match in PREFIX_DECLARATION_PATTERN.finditer(turtle_str)
        }
        if not declared_prefixes:
            return
        sanitized = sanitize_prefix_map(declared_prefixes, context="auto")
        changed_namespaces = [
            (original, sanitized[prefix])
            for prefix, original in declared_prefixes.items()
            if sanitized[prefix] != original
        ]
        for original_namespace, normalized_namespace in changed_namespaces:
            self.remap_namespaces(
                old_namespace=original_namespace,
                new_namespace=normalized_namespace,
            )
        for prefix, namespace in sanitized.items():
            self.bind(prefix, Namespace(namespace), override=True)

    @staticmethod
    def _normalize_turtle_input(turtle_str: str) -> str:
        """Normalize Turtle text from LLM output before RDF parsing."""
        normalized = unicodedata.normalize("NFKC", turtle_str)
        normalized = normalized.replace("\ufeff", "")
        normalized = normalized.replace("\u200b", "")
        normalized = normalized.replace("\u200c", "")
        normalized = normalized.replace("\u200d", "")
        normalized = normalized.replace("\u2060", "")
        normalized = normalized.replace("\xa0", " ")

        cleaned_chars = []
        for ch in normalized:
            if ch in ("\n", "\r", "\t"):
                cleaned_chars.append(ch)
                continue
            if unicodedata.category(ch).startswith("C"):
                continue
            cleaned_chars.append(ch)
        return "".join(cleaned_chars)

    @staticmethod
    def _quote_unquoted_typed_literals(turtle_str: str) -> str:
        """Wrap bare numeric values that carry a ^^ datatype annotation in quotes.

        The LLM sometimes emits ``1975^^xsd:integer`` instead of
        ``"1975"^^xsd:integer``.  The unquoted form causes the Turtle parser to
        treat ``^^`` as property-path operators, raising
        "Bad syntax (EOF found in middle of path syntax)".
        """
        return UNQUOTED_TYPED_LITERAL_PATTERN.sub(r'"\1"^^\2', turtle_str)

    @classmethod
    def _coerce_invalid_numeric_typed_literals(cls, turtle_str: str) -> str:
        """Drop numeric datatype when lexical form is invalid for that datatype."""

        def replace_integer(match: re.Match[str]) -> str:
            lexical = match.group(1)
            try:
                int(lexical)
                return match.group(0)
            except ValueError:
                return f'"{lexical}"'

        def replace_decimal(match: re.Match[str]) -> str:
            lexical = match.group(1)
            try:
                float(lexical)
                return match.group(0)
            except ValueError:
                return f'"{lexical}"'

        coerced = INTEGER_TYPED_LITERAL_PATTERN.sub(replace_integer, turtle_str)
        coerced = DECIMAL_TYPED_LITERAL_PATTERN.sub(replace_decimal, coerced)
        coerced = DOUBLE_TYPED_LITERAL_PATTERN.sub(replace_decimal, coerced)
        coerced = cls._coerce_invalid_date_typed_literals(coerced)
        return coerced

    @staticmethod
    def _coerce_invalid_date_typed_literals(turtle_str: str) -> str:
        """Normalize date literals with invalid xsd:date lexical forms.

        - `YYYY` is coerced to `xsd:gYear`
        - invalid date values fall back to plain literals
        """

        def replace_date(match: re.Match[str]) -> str:
            lexical = match.group(1)
            if re.fullmatch(r"\d{4}", lexical):
                return f'"{lexical}"^^xsd:gYear'
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", lexical):
                return match.group(0)
            return f'"{lexical}"'

        return DATE_TYPED_LITERAL_PATTERN.sub(replace_date, turtle_str)

    @classmethod
    def _repair_common_turtle_issues(
        cls, turtle_str: str, parse_error_message: str
    ) -> str:
        """Apply minimal repairs for common malformed Turtle patterns."""
        repaired = turtle_str

        # Typical LLM truncation: dangling ';' or ',' at EOF in property list.
        if "EOF found when expected verb in property list" in parse_error_message:
            repaired = cls._repair_truncated_turtle(repaired)

        # Unquoted ``^^`` literals can surface as path-syntax EOF errors.
        if "EOF found in middle of path syntax" in parse_error_message:
            repaired = cls._repair_truncated_turtle(repaired)

        # LLM repeats the subject on lines that follow a ';' continuation.
        if "expected '.' or '}' or ']' at end of statement" in parse_error_message:
            repaired = cls._repair_repeated_subject_after_semicolon(repaired)

        repaired = cls._repair_missing_object_before_dot(repaired)
        return repaired

    @staticmethod
    def _repair_truncated_turtle(turtle_str: str) -> str:
        """Repair common LLM Turtle truncation patterns.

        This only applies a minimal fix when content ends with dangling property-list
        punctuation (';' or ',') and no terminating '.'.
        """
        stripped = turtle_str.rstrip()
        if not stripped:
            return turtle_str
        if stripped.endswith(";") or stripped.endswith(","):
            return f"{stripped[:-1].rstrip()} .\n"
        return turtle_str

    @staticmethod
    def _looks_like_subject_token(token: str) -> bool:
        """Return True if *token* could be a Turtle subject IRI or prefixed name.

        Keywords that are only valid as predicates (``a``) or literals
        (``true``, ``false``) return False so that legitimate short-form
        predicate-object continuations are never mis-identified as repeated
        subjects.
        """
        if token in ("a", "true", "false"):
            return False
        if token.startswith("@") or token.startswith("#") or token.startswith('"'):
            return False
        if token.startswith("<") and ">" in token:
            return True
        if token.startswith("_:"):
            return True
        if ":" in token:
            colon_idx = token.index(":")
            prefix = token[:colon_idx]
            local = token[colon_idx + 1 :]
            return bool(prefix) and bool(local)
        return False

    @classmethod
    def _repair_repeated_subject_after_semicolon(cls, turtle_str: str) -> str:
        """Repair Turtle where a new full triple appears on a line following a ``;``.

        The LLM sometimes emits::

            cd:foo ns1:P1 ns2:Q1 ;
            cd:foo ns1:P2 ns2:Q2 .

        After a ``;`` the Turtle parser expects only a predicate–object pair
        (the subject is implied), but the LLM restates the subject, which
        triggers "expected '.' or '}' or ']' at end of statement".

        Repair strategy: when a line that ends with ``;`` is immediately
        followed by a non-blank line whose first token looks like a subject
        (prefixed name / IRI, not the ``a`` keyword), replace the trailing
        ``;`` with ``.`` so that each subject block is properly terminated and
        the next line starts a fresh triple.
        """
        lines = turtle_str.splitlines(keepends=True)
        result: list[str] = []

        for line in lines:
            content = line.rstrip("\r\n")
            stripped = content.strip()

            if (
                result
                and stripped
                and not stripped.startswith("@prefix")
                and not stripped.startswith("#")
            ):
                prev_raw = result[-1]
                prev_content = prev_raw.rstrip("\r\n").strip()
                if prev_content.endswith(";"):
                    tokens = stripped.split()
                    if len(tokens) >= 3 and cls._looks_like_subject_token(tokens[0]):
                        # Current line is a complete triple (s p o …) right after a ';'.
                        # Terminate the previous statement with '.' instead of ';'.
                        prev_body = prev_raw.rstrip("\r\n").rstrip()
                        eol = prev_raw[len(prev_raw.rstrip("\r\n")) :]
                        result[-1] = prev_body[:-1] + "." + eol
                        logger.debug(
                            "Repaired repeated subject after ';': %s", stripped
                        )

            result.append(line)

        return "".join(result)

    @staticmethod
    def _repair_missing_object_before_dot(turtle_str: str) -> str:
        """Remove malformed lines that end after predicate with no object."""
        repaired_lines: list[str] = []
        previous_meaningful_line: str | None = None
        for line in turtle_str.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("@prefix"):
                repaired_lines.append(line)
                continue
            if stripped.endswith(" ."):
                body = stripped[:-2].strip()
                token_count = len(body.split())
                previous_continues_property_list = (
                    previous_meaningful_line is not None
                    and (
                        previous_meaningful_line.endswith(";")
                        or previous_meaningful_line.endswith(",")
                    )
                )
                # Keep valid predicate-object continuation lines in property lists
                # (e.g. "ex:appealsTo ex:Cassation ."), but drop malformed
                # predicate-only lines and standalone subject-predicate lines.
                if token_count < 2 or (
                    token_count == 2 and not previous_continues_property_list
                ):
                    logger.debug(
                        "Dropping malformed Turtle line with missing object: %s",
                        stripped,
                    )
                    continue
            repaired_lines.append(line)
            previous_meaningful_line = stripped
        repaired = "\n".join(repaired_lines)
        if turtle_str.endswith("\n"):
            repaired += "\n"
        return repaired

    @classmethod
    def set_known_prefixes(cls, prefixes: dict[str, str] | None) -> None:
        """Set known prefixes in the context for use during parsing.

        This should be called before parsing TTL strings that may use prefixes
        from an ontology or other source. The prefixes will be automatically
        added if they're used but not declared in the TTL string.

        Args:
            prefixes: Dictionary mapping prefix names to namespace URIs.
                Example: {"fcaont": "https://growgraph.dev/fcaont#"}
        """
        _known_prefixes_context.set(prefixes)

    @classmethod
    def get_known_prefixes(cls) -> dict[str, str] | None:
        """Get currently known prefixes from context.

        Returns:
            Dictionary mapping prefix names to namespace URIs, or None.
        """
        return _known_prefixes_context.get()

    @classmethod
    def _from_jsonld_str(cls, jsonld_str: str) -> "RDFGraph":
        """Create an RDFGraph instance from a JSON-LD string.

        Args:
            jsonld_str: The input JSON-LD string.

        Returns:
            RDFGraph: A new RDFGraph instance with namespace prefixes extracted from @context.
        """
        # Use pyld to convert JSON-LD to n-quads, then parse to avoid rdflib's deprecated ConjunctiveGraph
        # This adapts to the new convention by using pyld directly instead of rdflib's JSON-LD parser
        jsonld_data = json.loads(jsonld_str)
        normalized = jsonld.normalize(
            jsonld_data,
            {"algorithm": "URDNA2015", "format": "application/n-quads"},
        )

        # jsonld.normalize returns a string when format is "application/n-quads"
        normalized_str = normalized if isinstance(normalized, str) else str(normalized)

        # Parse the normalized n-quads into RDFGraph
        g = cls()
        g.parse(data=normalized_str, format="nquads")

        # Extract prefixes from @context in JSON-LD and bind them
        try:
            context = None

            # Handle single object or array
            if isinstance(jsonld_data, dict):
                context = jsonld_data.get("@context")
            elif isinstance(jsonld_data, list) and jsonld_data:
                # For arrays, check first item for @context
                first_item = jsonld_data[0]
                if isinstance(first_item, dict):
                    context = first_item.get("@context")

            # Bind prefixes from @context
            if context and isinstance(context, dict):
                for prefix, uri in context.items():
                    if isinstance(uri, str) and not prefix.startswith("@"):
                        # Skip JSON-LD keywords (starting with @)
                        try:
                            g.bind(prefix, uri)
                        except Exception as e:
                            logger.debug(f"Failed to bind prefix '{prefix}': {e}")

        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            logger.debug(f"Could not extract prefixes from JSON-LD @context: {e}")

        return g

    @staticmethod
    def _to_turtle_str(g: Any) -> str:
        """Convert an RDFGraph to a Turtle string.

        For graphs backed by the *oxigraph* store the serialisation is
        delegated to ``pyoxigraph`` so that RDF 1.2 triple-term syntax
        (``<<( s p o )>>``) is emitted correctly.

        Args:
            g: The RDFGraph instance.

        Returns:
            str: The Turtle (or Turtle-star) string representation.
        """
        if hasattr(g, "store") and type(g.store).__name__ == "OxigraphStore":
            return g.serialize_turtle_star()
        return g.serialize(format="turtle")

    def serialize_turtle_star(self) -> str:
        """Serialize an oxigraph-backed graph to Turtle-star via *pyoxigraph*.

        This method extracts all quads belonging to this graph's context
        from the underlying ``pyoxigraph.Store`` and serialises them into
        the default graph using ``pyoxigraph.serialize`` with the Turtle
        format, which natively supports RDF 1.2 ``<<( … )>>`` syntax.

        Returns:
            Turtle-star string.

        Raises:
            RuntimeError: If the graph is not backed by an oxigraph store.
        """
        try:
            import pyoxigraph as ox
            from oxrdflib._converter import to_ox
        except ImportError as exc:
            raise RuntimeError(
                "pyoxigraph / oxrdflib must be installed for Turtle-star serialisation"
            ) from exc

        inner_store: ox.Store = self.store._inner  # type: ignore[attr-defined]
        graph_ctx_raw = to_ox(self.identifier)
        assert isinstance(
            graph_ctx_raw,
            (ox.NamedNode, ox.BlankNode, ox.DefaultGraph),
        )
        graph_ctx: ox.NamedNode | ox.BlankNode | ox.DefaultGraph = graph_ctx_raw

        # Copy quads into a temporary store under the default graph so
        # that ``ox.serialize`` can emit plain Turtle (Turtle-star).
        tmp = ox.Store()
        used_iri_terms: set[str] = set()

        def _collect_used_iris(term: Any) -> None:
            if isinstance(term, ox.NamedNode):
                used_iri_terms.add(term.value)
                return
            if isinstance(term, ox.Triple):
                _collect_used_iris(term.subject)
                _collect_used_iris(term.predicate)
                _collect_used_iris(term.object)

        for quad in inner_store.quads_for_pattern(
            None,
            None,
            None,
            graph_ctx,
        ):
            _collect_used_iris(quad.subject)
            _collect_used_iris(quad.predicate)
            _collect_used_iris(quad.object)
            tmp.add(
                ox.Quad(quad.subject, quad.predicate, quad.object, ox.DefaultGraph())
            )

        namespace_to_prefix: dict[str, str] = {}
        for prefix, namespace in self.namespaces():
            if not prefix:
                continue
            prefix_str = str(prefix)
            namespace_str = str(namespace)
            current = namespace_to_prefix.get(namespace_str)
            if current is None or (len(prefix_str), prefix_str) < (
                len(current),
                current,
            ):
                namespace_to_prefix[namespace_str] = prefix_str

        prefixes = {
            prefix: namespace
            for namespace, prefix in namespace_to_prefix.items()
            if any(iri.startswith(namespace) for iri in used_iri_terms)
        }
        raw: bytes = tmp.dump(
            format=ox.RdfFormat.TURTLE,
            from_graph=ox.DefaultGraph(),
            prefixes=prefixes or None,
        )  # type: ignore[assignment]
        return raw.decode()

    def serialize_canonical_turtle(self) -> str:
        """Serialize to Turtle after canonical namespace/prefix sanitization."""
        self.sanitize_prefixes_namespaces()
        serialized = self.serialize(format="turtle")
        if isinstance(serialized, bytes):
            return serialized.decode("utf-8")
        return str(serialized)

    def __new__(cls, *args, **kwargs):
        """Create a new RDFGraph instance."""
        instance = super().__new__(cls)
        return instance

    def serialize(
        self,
        destination: Any = None,
        format: str = "turtle",
        base: str | None = None,
        encoding: str | None = None,
        **args: Any,
    ) -> Any:
        """Serialize the graph, delegating to pyoxigraph for oxigraph stores.

        When the graph is backed by an *oxigraph* store and the requested
        format is ``"turtle"`` (or ``"ttl"``), serialisation is handled by
        ``pyoxigraph`` which natively supports RDF 1.2 triple terms.
        For all other stores or formats the default rdflib serialiser is
        used.
        """
        is_ox = type(self.store).__name__ == "OxigraphStore"
        if is_ox and format in ("turtle", "ttl"):
            ttl = self.serialize_turtle_star()
            if destination is not None:
                enc = encoding or "utf-8"
                with open(destination, "w", encoding=enc) as fh:
                    fh.write(ttl)
                return None
            return ttl
        return super().serialize(
            destination=destination,
            format=format,
            base=base,
            encoding=encoding,
            **args,
        )

    def update(
        self,
        update_object: Any,
        processor: Any = "sparql",
        initNs: Mapping[str, Any] | None = None,
        initBindings: Mapping[str, Any] | None = None,
        use_store_provided: bool = True,
        **kwargs: Any,
    ) -> None:
        """Execute SPARQL update using a base Graph view.

        rdflib's SPARQL update engine has internal checks that branch on exact
        ``Graph`` type, which can break for subclasses on ``INSERT/DELETE ... WHERE``.
        Running updates through a base ``Graph`` view avoids that edge case while
        still operating on the same underlying store/identifier.
        """
        graph_view = Graph(store=self.store, identifier=self.identifier)
        graph_view.namespace_manager = self.namespace_manager
        graph_view.update(
            update_object=update_object,
            processor=processor,
            initNs=initNs,
            initBindings=initBindings,
            use_store_provided=use_store_provided,
            **kwargs,
        )
        return None

    def sanitize_prefixes_namespaces(self):
        """
        Rematches prefixes in an RDFLib graph to correct namespaces when a namespace
        with the same URI exists. Handles cases where prefixes might not be bound
        as namespaces.

        Args:
            self (RDFGraph): The RDFLib graph to process

        Returns:
           RDFGraph: The graph with corrected prefix-namespace mappings
        """
        ns_manager = self.namespace_manager
        current_prefixes = {
            prefix: str(uri) for prefix, uri in dict(ns_manager.namespaces()).items()
        }
        if not current_prefixes:
            return self

        sanitized = sanitize_prefix_map(current_prefixes, context="auto")
        for prefix, original_namespace in current_prefixes.items():
            normalized_namespace = sanitized[prefix]
            if normalized_namespace != original_namespace:
                self.remap_namespaces(
                    old_namespace=original_namespace,
                    new_namespace=normalized_namespace,
                )

        new_ns_manager = NamespaceManager(self)
        uri_to_prefixes = defaultdict(list)
        for prefix, namespace in sanitized.items():
            uri_to_prefixes[namespace].append(prefix)

        for namespace, prefixes in uri_to_prefixes.items():
            best_prefix = sorted(prefixes, key=lambda p: (len(p), p))[0]
            new_ns_manager.bind(
                best_prefix,
                Namespace(normalize_namespace_iri(namespace, context="auto")),
                override=True,
            )
        self.namespace_manager = new_ns_manager
        return self

    def bind_implicit_namespaces(self, prefix_base: str | None = None) -> None:
        """Bind namespace prefixes for IRI stems used in the graph but not declared.

        Scans all URIRef terms in the graph, extracts their namespace stems
        (the IRI up to and including the last ``#`` or ``/``), and auto-binds
        any stem that appears in at least two IRIs but has no declared prefix.

        This is particularly useful for ontologies that use sub-namespaces
        (e.g. ``/concepts#`` and ``/relations#``) without explicit ``@prefix``
        declarations.  Without declared prefixes the LLM invents shortcuts like
        ``ont_10_culture:relations#P571`` which are syntactically invalid Turtle.

        Args:
            prefix_base: Optional string prepended to generated prefix names,
                typically the ``ontology_id``.  Produces prefixes of the form
                ``{prefix_base}_{slug}``; without it just ``{slug}`` is used.
        """
        declared_namespaces = {str(ns) for _, ns in self.namespaces()}
        standard_namespaces = {uri.strip("<>") for uri in COMMON_PREFIXES.values()}

        stem_counts: dict[str, int] = {}
        for s, p, o in self:
            for term in (s, p, o):
                if not isinstance(term, URIRef):
                    continue
                iri = str(term)
                if any(iri.startswith(std) for std in standard_namespaces):
                    continue
                for sep in ("#", "/"):
                    idx = iri.rfind(sep)
                    if idx > 0:
                        stem = iri[: idx + 1]
                        if stem not in declared_namespaces:
                            stem_counts[stem] = stem_counts.get(stem, 0) + 1
                        break

        for stem, count in stem_counts.items():
            if count < 2:
                continue
            slug = stem.rstrip("#/").rsplit("/", 1)[-1].replace("-", "_")
            prefix = f"{prefix_base}_{slug}" if prefix_base else slug
            self.bind(prefix, Namespace(stem), override=False)

    def unbind_chunk_namespaces(self, chunk_pattern="/chunk/") -> "RDFGraph":
        """
        Unbinds namespace prefixes that point to URIs containing a chunk pattern.
        Returns a new graph with chunk namespaces dereferenced (expanded to full URIs).
        Prefix target IRIs are normalized to end with ``#`` or ``/`` (``/`` appended
        when missing) before chunk detection and rebinding.

        Args:
            chunk_pattern (str): The pattern to look for in URIs (default: "/chunk/")

        Returns:
            RDFGraph: New graph with chunk-related namespaces unbound
        """
        current_prefixes = dict(self.namespace_manager.namespaces())

        # Normalize prefix target IRIs so chunk detection and rebinding agree on boundaries
        prefix_to_normalized: dict[str, str] = {
            prefix: normalize_namespace_iri(str(uri), context="auto")
            for prefix, uri in current_prefixes.items()
        }

        # Find prefixes that point to URIs containing the chunk pattern
        chunk_prefixes = []
        for prefix, uri_str in prefix_to_normalized.items():
            if chunk_pattern in uri_str:
                chunk_prefixes.append((prefix, uri_str))

        # Create new graph
        new_graph = RDFGraph()

        # Copy all triples (URIs are already expanded internally)
        for triple in self:
            new_graph.add(triple)

        # Bind only non-chunk namespace prefixes to the new graph
        for prefix, uri_str in prefix_to_normalized.items():
            if chunk_pattern not in uri_str:
                new_graph.bind(prefix, Namespace(uri_str))

        # Log what was removed
        if chunk_prefixes:
            logger.debug(f"Unbound {len(chunk_prefixes)} chunk-related namespace(s):")
            for prefix, uri in chunk_prefixes:
                logger.debug(f"  - '{prefix}': {uri}")

        return new_graph

    def remap_namespaces(self, old_namespace, new_namespace) -> None:
        updates = {}
        for s, p, o in self:
            new_s, new_p, new_o = s, p, o
            if isinstance(s, URIRef) and str(s).startswith(str(old_namespace)):
                new_s = URIRef(
                    str(s).replace(str(old_namespace), str(new_namespace), 1)
                )
            if isinstance(p, URIRef) and str(p).startswith(str(old_namespace)):
                new_p = URIRef(
                    str(p).replace(str(old_namespace), str(new_namespace), 1)
                )
            if isinstance(o, URIRef) and str(o).startswith(str(old_namespace)):
                new_o = URIRef(
                    str(o).replace(str(old_namespace), str(new_namespace), 1)
                )

            if (new_s, new_p, new_o) != (s, p, o):
                updates[(s, p, o)] = (new_s, new_p, new_o)

        for (s, p, o), (new_s, new_p, new_o) in updates.items():
            self.remove((s, p, o))
            self.add((new_s, new_p, new_o))

    def add_triple(self, subject: str, predicate: str, object_: str) -> None:
        """Add a triple to the graph.

        Args:
            subject: Subject URI as string
            predicate: Predicate URI as string
            object_: Object URI as string or literal value
        """
        # Convert strings to appropriate RDFLib objects
        subj = URIRef(subject)
        pred = URIRef(predicate)

        # Handle object - could be URI or literal
        if object_.startswith("http://") or object_.startswith("https://"):
            obj = URIRef(object_)
        else:
            # Treat as literal
            obj = Literal(object_)

        self.add((subj, pred, obj))
        logger.debug(f"Added triple: {subj} {pred} {obj}")

    def remove_triple(self, subject: str, predicate: str, object_: str) -> None:
        """Remove a triple from the graph.

        Args:
            subject: Subject URI as string
            predicate: Predicate URI as string
            object_: Object URI as string or literal value
        """
        # Convert strings to appropriate RDFLib objects
        subj = URIRef(subject)
        pred = URIRef(predicate)

        # Handle object - could be URI or literal
        if object_.startswith("http://") or object_.startswith("https://"):
            obj = URIRef(object_)
        else:
            # Treat as literal
            obj = Literal(object_)

        self.remove((subj, pred, obj))
        logger.debug(f"Removed triple: {subj} {pred} {obj}")

    def hash(self: Graph) -> str:
        # Serialize to JSON-LD
        data = self.serialize(format="json-ld")

        # Parse the JSON string
        doc = json.loads(data)

        # Canonicalize using URDNA2015 normalization
        normalized = jsonld.normalize(
            doc,
            {"algorithm": "URDNA2015", "format": "application/n-quads"},
        )
        # jsonld.normalize returns a string when format is "application/n-quads"
        normalized_str = normalized if isinstance(normalized, str) else str(normalized)
        return render_text_hash(normalized_str, digits=None)
