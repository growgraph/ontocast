import logging
import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Union

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import NamespaceManager

from ontocast.onto.constants import COMMON_PREFIXES

logger = logging.getLogger(__name__)

PREFIX_PATTERN = re.compile(r"@prefix\s+(\w+):\s+<[^>]+>\s+\.")


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
            A union schema that handles both Graph instances and string conversion.
        """
        return core_schema.union_schema(
            [
                core_schema.is_instance_schema(cls),
                core_schema.chain_schema(
                    [
                        core_schema.str_schema(),
                        core_schema.no_info_plain_validator_function(
                            cls._from_turtle_str
                        ),
                    ]
                ),
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

    @staticmethod
    def _ensure_prefixes(turtle_str: str) -> str:
        """Ensure all common prefixes are declared in the Turtle string.

        Args:
            turtle_str: The input Turtle string.

        Returns:
            str: The Turtle string with all common prefixes declared.
        """
        declared_prefixes = set(
            match.group(1) for match in PREFIX_PATTERN.finditer(turtle_str)
        )

        missing = {
            prefix: uri
            for prefix, uri in COMMON_PREFIXES.items()
            if prefix not in declared_prefixes
        }

        if not missing:
            return turtle_str

        prefix_block = (
            "\n".join(f"@prefix {prefix}: {uri} ." for prefix, uri in missing.items())
            + "\n\n"
        )

        return prefix_block + turtle_str

    @classmethod
    def _from_turtle_str(cls, turtle_str: str) -> "RDFGraph":
        """Create an RDFGraph instance from a Turtle string.

        Args:
            turtle_str: The input Turtle string.

        Returns:
            RDFGraph: A new RDFGraph instance.
        """
        turtle_str = bytes(turtle_str, "utf-8").decode("unicode_escape")
        patched_turtle = cls._ensure_prefixes(turtle_str)
        g = cls()
        g.parse(data=patched_turtle, format="turtle")
        return g

    @staticmethod
    def _to_turtle_str(g: Any) -> str:
        """Convert an RDFGraph to a Turtle string.

        Args:
            g: The RDFGraph instance.

        Returns:
            str: The Turtle string representation.
        """
        return g.serialize(format="turtle")

    def __new__(cls, *args, **kwargs):
        """Create a new RDFGraph instance."""
        instance = super().__new__(cls)
        return instance

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
        # Get the namespace manager
        ns_manager = self.namespace_manager

        # Collect all current prefix-URI mappings
        current_prefixes = dict(ns_manager.namespaces())

        # Group URIs by their string representation to find duplicates
        uri_to_prefixes = defaultdict(list)
        for prefix, uri in current_prefixes.items():
            uri_to_prefixes[str(uri)].append((prefix, uri))

        # Find the "canonical" namespace objects for each URI
        # (the actual Namespace objects that might be registered)
        canonical_namespaces = {}

        # Check if any of the URIs correspond to well-known namespaces
        # by trying to create Namespace objects and seeing if they're already registered
        for uri_str, prefix_uri_pairs in uri_to_prefixes.items():
            # Try to find if there's already a proper Namespace object for this URI
            namespace_candidates = []

            for prefix, uri_obj in prefix_uri_pairs:
                # Check if this is already a proper Namespace object
                if isinstance(uri_obj, Namespace):
                    namespace_candidates.append(uri_obj)
                else:
                    # Try to create a Namespace and see if it matches existing ones
                    try:
                        ns = Namespace(uri_str)
                        namespace_candidates.append(ns)
                    except:
                        continue

            # Use the first valid namespace candidate as canonical
            if namespace_candidates:
                canonical_namespaces[uri_str] = namespace_candidates[0]

        # Now rebuild the namespace manager with corrected mappings
        # Clear existing bindings first
        new_ns_manager = NamespaceManager(self)

        # Track which prefixes we want to keep/reassign
        final_mappings = {}

        for uri_str, prefix_uri_pairs in uri_to_prefixes.items():
            if len(prefix_uri_pairs) == 1:
                # No duplicates, keep as-is but ensure we use canonical namespace
                prefix, _ = prefix_uri_pairs[0]
                canonical_ns = canonical_namespaces.get(uri_str)
                if canonical_ns:
                    final_mappings[prefix] = canonical_ns
                else:
                    # Fallback to creating a new Namespace
                    final_mappings[prefix] = Namespace(uri_str)
            else:
                # Multiple prefixes for same URI - need to decide which to keep
                # Priority: 1) Proper Namespace objects,
                #           2) Shorter prefixes,
                #           3) Alphabetical
                prefix_uri_pairs.sort(
                    key=lambda x: (
                        not isinstance(x[1], Namespace),  # Namespace objects first
                        len(x[0]),  # Shorter prefixes next
                        x[0],  # Alphabetical order
                    )
                )

                # Keep the best prefix, map others to it if needed
                best_prefix, _ = prefix_uri_pairs[0]
                canonical_ns = canonical_namespaces.get(uri_str, Namespace(uri_str))
                final_mappings[best_prefix] = canonical_ns

                other_prefixes = [p for p, _ in prefix_uri_pairs[1:]]
                if other_prefixes:
                    logger.debug(
                        f"Consolidating prefixes {other_prefixes} "
                        f"-> '{best_prefix}' for URI: {uri_str}"
                    )

        # Apply the final mappings
        for prefix, namespace in final_mappings.items():
            new_ns_manager.bind(prefix, namespace, override=True)

        # Replace the graph's namespace manager
        self.namespace_manager = new_ns_manager

    def unbind_chunk_namespaces(self, chunk_pattern="/chunk/") -> "RDFGraph":
        """
        Unbinds namespace prefixes that point to URIs containing a chunk pattern.
        Returns a new graph with chunk namespaces dereferenced (expanded to full URIs).

        Args:
            chunk_pattern (str): The pattern to look for in URIs (default: "/chunk/")

        Returns:
            RDFGraph: New graph with chunk-related namespaces unbound
        """
        current_prefixes = dict(self.namespace_manager.namespaces())

        # Find prefixes that point to URIs containing the chunk pattern
        chunk_prefixes = []
        for prefix, uri in current_prefixes.items():
            uri_str = str(uri)
            if chunk_pattern in uri_str:
                chunk_prefixes.append((prefix, uri_str))

        # Create new graph
        new_graph = RDFGraph()

        # Copy all triples (URIs are already expanded internally)
        for triple in self:
            new_graph.add(triple)

        # Bind only non-chunk namespace prefixes to the new graph
        for prefix, uri in current_prefixes.items():
            uri_str = str(uri)
            if chunk_pattern not in uri_str:
                new_graph.bind(prefix, uri)

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
