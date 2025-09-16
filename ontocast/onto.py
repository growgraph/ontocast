import logging
import os
import pathlib
import re
from collections import defaultdict
from enum import StrEnum
from typing import Any, Optional, Union
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, GetCoreSchemaHandler
from pydantic_core import core_schema
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, NamespaceManager

from ontocast.util import CONVENTIONAL_MAPPINGS, iri2namespace, render_text_hash

logger = logging.getLogger(__name__)


DEFAULT_DOMAIN = "https://example.com"

ONTOLOGY_NULL_ID = "__null_id"
ONTOLOGY_NULL_IRI = f"{DEFAULT_DOMAIN}/{ONTOLOGY_NULL_ID}"


def derive_ontology_id(iri: str) -> str:
    if not isinstance(iri, str) or not iri.strip():
        return ONTOLOGY_NULL_ID

    normalized_iri = iri.strip().rstrip("/#")

    if normalized_iri in CONVENTIONAL_MAPPINGS:
        return CONVENTIONAL_MAPPINGS[normalized_iri]

    parsed = urlparse(normalized_iri)

    candidate = (
        parsed.path.rsplit("/", 1)[-1]
        if parsed.path and "/" in parsed.path
        else parsed.netloc.split(".")[0]
        if parsed.netloc
        else normalized_iri
    )

    return _clean_derived_id(candidate)


def _clean_derived_id(value: str) -> str:
    value = re.sub(r"\.(owl|ttl|rdf|xml)$", "", value, flags=re.IGNORECASE)
    match = re.match(r"^(.*?)\.(org|com|net|io|edu|gov|int|mil)$", value, re.IGNORECASE)
    if match:
        value = match.group(1)
    return re.sub(r"[^a-zA-Z0-9_-]", "", value).lower() or ONTOLOGY_NULL_ID


class Status(StrEnum):
    """Enumeration of possible workflow status values."""

    SUCCESS = "success"
    FAILED = "failed"
    COUNTS_EXCEEDED = "counts exceeded"


class ToolType(StrEnum):
    """Enumeration of tool types used in the workflow."""

    LLM = "llm"
    TRIPLE_STORE = "triple store manager"
    ONTOLOGY_MANAGER = "ontology manager"
    CONVERTER = "document converter"
    CHUNKER = "document chunker"


class OntologyDecision(StrEnum):
    """Enumeration of Ontology Decisions used in the workflow."""

    SKIP_TO_FACTS = "ontology found; skip to facts"
    FAILURE_NO_ONTOLOGY = "ontology not found; fast forward to END"
    IMPROVE_CREATE_ONTOLOGY = "improve/create ontology"


class FailureStages(StrEnum):
    """Enumeration of possible failure stages in the workflow."""

    NO_CHUNKS_TO_PROCESS = "No chunks to process"
    ONTOLOGY_CRITIQUE = "The produced ontology did not pass the critique stage."
    FACTS_CRITIQUE = "The produced graph of facts did not pass the critique stage."
    PARSE_TEXT_TO_ONTOLOGY_TRIPLES = "Failed to parse the text into ontology triples."
    PARSE_TEXT_TO_FACTS_TRIPLES = "Failed to parse the text into facts triples."
    SUBLIMATE_ONTOLOGY = (
        "The produced semantic could not be validated "
        "or separated into ontology and facts (technical issue)."
    )


COMMON_PREFIXES = {
    "xsd": "<http://www.w3.org/2001/XMLSchema#>",
    "rdf": "<http://www.w3.org/1999/02/22-rdf-syntax-ns#>",
    "rdfs": "<http://www.w3.org/2000/01/rdf-schema#>",
    "owl": "<http://www.w3.org/2002/07/owl#>",
    "skos": "<http://www.w3.org/2004/02/skos/core#>",
    "foaf": "<http://xmlns.com/foaf/0.1/>",
    "schema": "<http://schema.org/>",
    "ex": "<http://example.org/>",
}

PROV = Namespace("http://www.w3.org/ns/prov#")
SCHEMA = Namespace("https://schema.org/")

PREFIX_PATTERN = re.compile(r"@prefix\s+(\w+):\s+<[^>]+>\s+\.")


class BasePydanticModel(BaseModel):
    """Base class for Pydantic models with serialization capabilities."""

    def __init__(self, **kwargs):
        """Initialize the model with given keyword arguments."""
        super().__init__(**kwargs)

    def serialize(self, file_path: str | pathlib.Path) -> None:
        """Serialize the state to a JSON file.

        Args:
            file_path: Path to save the JSON file.
        """
        state_json = self.model_dump_json(indent=4)
        if isinstance(file_path, str):
            file_path = pathlib.Path(file_path)
        file_path.write_text(state_json)

    @classmethod
    def load(cls, file_path: str | pathlib.Path):
        """Load state from a JSON file.

        Args:
            file_path: Path to the JSON file.

        Returns:
            The loaded model instance.
        """
        if isinstance(file_path, str):
            file_path = pathlib.Path(file_path)
        state_json = file_path.read_text()
        return cls.model_validate_json(state_json)


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

    def __add__(self, other: Union["RDFGraph", Graph]) -> "RDFGraph":
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

        # Copy namespace bindings
        for prefix, uri in self.namespaces():
            result.bind(prefix, uri)
        for prefix, uri in other.namespaces():
            result.bind(prefix, uri)

        return result

    def __iadd__(self, other: Union["RDFGraph", Graph]) -> "RDFGraph":
        """In-place addition operator for RDFGraph instances.

        Merges the RDF graphs while maintaining the RDFGraph type.

        Args:
            other: The graph to add to this one.

        Returns:
            RDFGraph: self after modification.
        """
        # Call parent's __iadd__ to merge the graphs
        super().__iadd__(other)
        # Return self to maintain RDFGraph type
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


class OntologySelectorReport(BasePydanticModel):
    """Report from ontology selection process.

    Attributes:
        ontology_id: Ontology id that could be used
            to represent the domain of the document, None if no ontology is suitable.
        present: Whether an ontology that could represent the domain of the document
            is present in the list of ontologies.
    """

    ontology_id: str | None = Field(
        description="id of the ontology"
        "to represent the domain of the document, None if no ontology is suitable"
    )
    ontology_iri: str | None = Field(
        description="URI / IRI of the ontology"
        "to represent the domain of the document, None if no ontology is suitable"
    )
    present: bool = Field(
        description="Whether an ontology that could represent "
        "the domain of the document is present in the list of ontologies"
    )


class SemanticTriplesFactsReport(BaseModel):
    """Report containing semantic triples and evaluation scores.

    Attributes:
        semantic_graph: Semantic triples (facts) representing the document
            in turtle (ttl) format.
        ontology_relevance_score: Score 0-100 for how relevant the ontology
            is to the document. 0 is the worst, 100 is the best.
        triples_generation_score: Score 0-100 for how well the facts extraction /
            triples generation was performed. 0 is the worst, 100 is the best.
    """

    semantic_graph: RDFGraph = Field(
        default_factory=RDFGraph,
        description="Semantic triples (facts) representing "
        "the document in turtle (ttl) format.",
    )
    ontology_relevance_score: Optional[float] = Field(
        description=(
            "Score between 0 and 100 of how well "
            "the ontology represents the domain  of the document."
        )
    )
    triples_generation_score: Optional[float] = Field(
        description=(
            "Score 0-100 for how well the semantic triples "
            "represent the document. 0 is the worst, 100 is the best."
        )
    )


class OntologyUpdateCritiqueReport(BaseModel):
    """Report from ontology update critique process.

    Attributes:
        ontology_update_success: True if the ontology update was performed
            successfully, False otherwise.
        ontology_update_score: Score 0-100 for how well the update improves
            the original domain ontology of the document.
        ontology_update_critique: A concrete explanation of why the
            ontology update is not satisfactory.
    """

    ontology_update_success: bool = Field(
        description="True if the ontology update "
        "was performed successfully, False otherwise."
    )
    ontology_update_score: float = Field(
        description="Score 0-100 for how well the update improves "
        "the original domain ontology of the document. "
        "0 is the worst, 100 is the best."
    )
    ontology_update_critique: str | None = Field(
        None,
        description="A concrete explanation of why "
        "the ontology update is not satisfactory. "
        "The explanation should be very specific and detailed.",
    )

    ontology_update_failed: str | None = Field(
        None, description="A null ontology update was returned."
    )


class KGCritiqueReport(BaseModel):
    """Report from knowledge graph critique process.

    Attributes:
        facts_graph_derivation_success: True if the facts graph derivation
            was performed successfully, False otherwise.
        facts_graph_derivation_score: Score 0-100 for how well the triples
            of facts represent the original document.
        facts_graph_derivation_critique_comment: A concrete explanation of
            why the semantic graph of facts derivation is not satisfactory.
    """

    facts_graph_derivation_success: bool = Field(
        description="True if the facts triples fully represent the document, False otherwise. "
    )
    facts_graph_derivation_score: float = Field(
        description="Score 0-100 for how well the triples of facts "
        "represent the original document. 0 is the worst, 100 is the best."
    )
    facts_graph_derivation_critique_comment: str | None = Field(
        None,
        description="A concrete explanation of why the semantic graph "
        "of facts derivation is not satisfactory. "
        "The explanation should be very specific and detailed.",
    )


class OntologyProperties(BaseModel):
    """Properties of an ontology.

    Attributes:
        ontology_id: Ontology identifier.
        title: Ontology title.
        description: A concise description of the ontology.
        version: Version of the ontology.
        iri: Ontology IRI (Internationalized Resource Identifier).
    """

    ontology_id: str | None = Field(
        default=None,
        description="Ontology identifier, an human readable lower case abbreviation. Must be provided.",
    )
    title: str | None = Field(
        default=None, description="Ontology title. Must be provided."
    )
    description: str | None = Field(
        default=None,
        description="A concise description (3-4 sentences) of the ontology "
        "(domain, purpose, applicability, etc.)",
    )
    version: str | None = Field(
        default=None,
        description="Version of the ontology (use semantic versioning)",
    )
    iri: str | None = Field(
        default=None,
        description="Ontology IRI (Internationalized Resource Identifier)",
    )

    @property
    def namespace(self):
        """Get the namespace for this ontology.

        Returns:
            str: The namespace string.
        """
        return iri2namespace(self.iri, ontology=True)


class Ontology(OntologyProperties):
    """A Pydantic model representing an ontology with its RDF graph and description.

    Attributes:
        graph: The RDF graph containing the ontology data.
        current_domain: The domain used to construct the ontology IRI if ontology_id is set.
    """

    graph: RDFGraph = Field(
        default_factory=RDFGraph,
        description="Semantic triples (abstract entities/relations) "
        "that define the ontology in turtle (ttl) format as a string.",
    )
    current_domain: str = Field(
        default=DEFAULT_DOMAIN, description="Domain for ontology IRI construction."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **kwargs):
        # Pop current_domain if provided, else use DEFAULT_DOMAIN
        current_domain = kwargs.pop("current_domain", DEFAULT_DOMAIN)
        super().__init__(current_domain=current_domain, **kwargs)
        # --- Only apply fallback logic if graph does not contain a proper owl:Ontology subject ---
        # Try to sync from graph first
        graph_had_ontology = False
        if self.graph:
            # Try to extract from graph
            self.sync_properties_from_graph()
            # If after sync, both iri and ontology_id are set, do nothing further
            if (
                self.iri
                and self.iri != ONTOLOGY_NULL_IRI
                and self.ontology_id
                and self.ontology_id != ONTOLOGY_NULL_ID
            ):
                graph_had_ontology = True
        # Only apply fallback if graph did not provide a valid pair
        if not graph_had_ontology:
            if self.ontology_id and (not self.iri or self.iri == ONTOLOGY_NULL_IRI):
                self.iri = f"{self.current_domain}/{self.ontology_id}"
            elif self.ontology_id and self.iri:
                expected_iri = f"{self.current_domain}/{self.ontology_id}"
                if not self.iri.endswith(f"/{self.ontology_id}"):
                    logger.warning(
                        f"Ontology IRI '{self.iri}' does not match expected '{expected_iri}', we correct ontology IRI"
                    )
                    self.iri = expected_iri
            elif not self.ontology_id and self.iri and self.iri != ONTOLOGY_NULL_IRI:
                self.ontology_id = derive_ontology_id(self.iri)
        # Always ensure graph is up to date with properties
        self.sync_properties_to_graph()

    def set_properties(self, **kwargs):
        """Set ontology properties from keyword arguments and sync to graph.
        Only update properties if they are missing (None or empty).
        Also enforces ontology_id/iri consistency as in __init__, but only if graph does not provide a valid pair.
        """
        for k, v in kwargs.items():
            if hasattr(self, k):
                current = getattr(self, k)
                if not current and v:
                    setattr(self, k, v)
        # Try to sync from graph first
        graph_had_ontology = False
        if self.graph:
            self.sync_properties_from_graph()
            if (
                self.iri
                and self.iri != ONTOLOGY_NULL_IRI
                and self.ontology_id
                and self.ontology_id != ONTOLOGY_NULL_ID
            ):
                graph_had_ontology = True
        if not graph_had_ontology:
            if self.ontology_id and (not self.iri or self.iri == ONTOLOGY_NULL_IRI):
                self.iri = f"{self.current_domain}/{self.ontology_id}"
            elif self.ontology_id and self.iri:
                expected_iri = f"{self.current_domain}/{self.ontology_id}"
                if not self.iri.endswith(f"/{self.ontology_id}"):
                    logger.warning(
                        f"Ontology IRI '{self.iri}' does not match expected '{expected_iri}'"
                    )
            elif not self.ontology_id and self.iri and self.iri != ONTOLOGY_NULL_IRI:
                self.ontology_id = derive_ontology_id(self.iri)
        self.sync_properties_to_graph()

    def sync_properties_to_graph(self):
        """
        Update the RDF graph with the Ontology's properties.
        Only sync properties for the entity that is explicitly typed as owl:Ontology.
        Only add property triples if they do not already exist in the graph.
        Optimized to avoid multiple loops over triples.
        """

        if self.ontology_id is not None and self.ontology_id is not ONTOLOGY_NULL_ID:
            if self.iri and (not self.iri or self.iri == ONTOLOGY_NULL_IRI):
                self.iri = f"{self.current_domain}/{self.ontology_id}"
            elif self.iri:
                expected_iri = f"{self.current_domain}/{self.ontology_id}"
                if not self.iri.endswith(f"/{self.ontology_id}"):
                    logger.warning(
                        f"Ontology IRI '{self.iri}' does not match expected '{expected_iri}', fixing"
                    )
                    self.iri = expected_iri
        elif self.iri:
            self.ontology_id = derive_ontology_id(self.iri)

        if self.iri is ONTOLOGY_NULL_IRI or self.iri is None:
            return
        else:
            onto_iri = URIRef(self.iri)
        g = self.graph

        onto_triple = [
            subj
            for subj, _, o in g.triples((None, RDF.type, None))
            if o == OWL.Ontology
        ]
        if not onto_triple:
            if onto_iri is not None:
                # iri set as a property, but not in ontology
                g.add((onto_iri, RDF.type, OWL.Ontology))
        else:
            onto_iri_graph = onto_triple[0]
            onto_iri = onto_iri_graph

        # Collect all predicates for this subject in one pass
        existing_preds = set(p for _, p, _ in g.triples((onto_iri, None, None)))

        def add_if_missing(p, v):
            if p not in existing_preds:
                g.add((onto_iri, p, Literal(v)))

        # Add label/title
        if self.title:
            add_if_missing(RDFS.label, self.title)
        if self.ontology_id:
            add_if_missing(DCTERMS.title, self.ontology_id)
        # Add description
        if self.description:
            add_if_missing(DCTERMS.description, self.description)
            add_if_missing(RDFS.comment, self.description)
        # Add version
        if self.version:
            add_if_missing(OWL.versionInfo, self.version)

    def sync_properties_from_graph(self):
        """
        Update Ontology properties from the RDF graph if present,
        but only if missing, and only for entities explicitly typed as owl:Ontology.
        Optimized to avoid multiple loops over triples.
        """
        g = self.graph
        # Only proceed if this subject is explicitly typed as owl:Ontology
        onto_triple = [
            subj
            for subj, _, o in g.triples((None, RDF.type, None))
            if o == OWL.Ontology
        ]
        if not onto_triple:
            return
        onto_iri = onto_triple[0]
        self.iri = str(onto_iri)

        self.ontology_id = derive_ontology_id(self.iri)

        # Collect all predicates and objects for this subject in one pass
        pred_map = defaultdict(list)
        for _, p, o in g.triples((onto_iri, None, None)):
            pred_map[p].append(o)

        # Title: try rdfs:label, dcterms:title
        if not getattr(self, "title", None):
            title = None
            if RDFS.label in pred_map:
                title = str(pred_map[RDFS.label][0])
            elif DCTERMS.title in pred_map:
                title = str(pred_map[DCTERMS.title][0])
            if title:
                self.title = title

        # Description: try dcterms:description, rdfs:comment
        if not getattr(self, "description", None):
            description = None
            if DCTERMS.description in pred_map:
                description = str(pred_map[DCTERMS.description][0])
            elif RDFS.comment in pred_map:
                description = str(pred_map[RDFS.comment][0])
            if description:
                self.description = description
        # Version
        if not getattr(self, "version", None):
            if OWL.versionInfo in pred_map:
                self.version = str(pred_map[OWL.versionInfo][0])
        # Short name: try dcterms:title if not already used for title
        if not getattr(self, "ontology_id", None):
            if DCTERMS.title in pred_map:
                self.ontology_id = str(pred_map[DCTERMS.title][0])

    def __iadd__(self, other: Union["Ontology", RDFGraph]) -> "Ontology":
        """In-place addition operator for Ontology instances.

        Merges the RDF graphs and takes properties from the right-hand operand.

        Args:
            other: The ontology or graph to add to this one.

        Returns:
            Ontology: self after modification.
        """
        if isinstance(other, Ontology):
            self.graph += other.graph
            self.title = other.title
            self.ontology_id = other.ontology_id
            self.description = other.description
            self.iri = other.iri
            self.version = other.version
        else:
            self.graph += other
        return self

    @classmethod
    def from_file(cls, file_path: pathlib.Path, format: str = "turtle", **kwargs):
        """Create an Ontology instance by loading a graph from a file.

        Args:
            file_path: Path to the ontology file.
            format: Format of the input file (default: "turtle").
            **kwargs: Additional arguments to pass to the constructor.

        Returns:
            Ontology: A new Ontology instance.
        """
        graph: RDFGraph = RDFGraph()
        graph.parse(file_path, format=format)
        return cls(graph=graph, **kwargs)

    def describe(self) -> str:
        """Get a human-readable description of the ontology.

        Returns:
            str: A formatted description string.
        """
        return (
            f"Ontology id: {self.ontology_id}\n"
            f"Description: {self.description}\n"
            f"Ontology IRI: {self.iri}\n"
        )


NULL_ONTOLOGY = Ontology(
    ontology_id=ONTOLOGY_NULL_ID,
    title="null title",
    description="null description",
    graph=RDFGraph(),
    iri=ONTOLOGY_NULL_IRI,
)


class WorkflowNode(StrEnum):
    """Enumeration of workflow nodes in the processing pipeline."""

    CONVERT_TO_MD = "Convert to Markdown"
    CHUNK = "Chunk Text"
    SELECT_ONTOLOGY = "Select Ontology"
    TEXT_TO_ONTOLOGY = "Text to Ontology"
    TEXT_TO_FACTS = "Text to Facts"
    SUBLIMATE_ONTOLOGY = "Sublimate Ontology"
    CRITICISE_ONTOLOGY = "Criticise Ontology"
    CRITICISE_FACTS = "Criticise Facts"
    CHUNKS_EMPTY = "Chunks Empty?"
    AGGREGATE_FACTS = "Aggregate Facts"


class Chunk(BaseModel):
    """A chunk of text with associated metadata and RDF graph.

    Attributes:
        text: Text content of the chunk.
        hid: An almost unique (hash) id for the chunk.
        doc_iri: IRI of parent document.
        graph: RDF triples representing the facts from the current document.
        processed: Whether chunk has been processed.
    """

    text: str = Field(description="Text of the chunk")
    hid: str = Field(description="An almost unique (hash) id for the chunk")
    doc_iri: str = Field(description="IRI of parent doc")
    graph: Optional[RDFGraph] = Field(
        description="RDF triples representing the facts from a document chunk",
        default_factory=RDFGraph,
    )
    processed: bool = Field(default=False, description="Was the chunk processed?")

    @property
    def iri(self):
        """Get the IRI for this chunk.

        Returns:
            str: The chunk IRI.
        """
        return f"{self.doc_iri}/chunk/{self.hid}"

    @property
    def namespace(self):
        """Get the namespace for this chunk.

        Returns:
            str: The chunk namespace.
        """
        return iri2namespace(self.iri, ontology=False)

    def sanitize(self):
        self.graph = self.graph.unbind_chunk_namespaces()
        self.graph.sanitize_prefixes_namespaces()


class AgentState(BasePydanticModel):
    """State for the ontology-based knowledge graph agent.

    This class maintains the state of the agent during document processing,
    including input text, chunks, ontologies, and workflow status.

    Attributes:
        input_text: Input text to process.
        current_domain: IRI used for forming document namespace.
        doc_hid: An almost unique hash/id for the parent document.
        files: Files to process.
        current_chunk: Current document chunk for processing.
        chunks: List of chunks of the input text.
        chunks_processed: List of processed chunks.
        current_ontology: Current ontology object.
        ontology_addendum: Additional ontology content.
        failure_stage: Stage where failure occurred.
        failure_reason: Reason for failure.
        success_score: Score indicating success level.
        status: Current workflow status.
        node_visits: Number of visits per node.
        max_visits: Maximum number of visits allowed per node.
        max_chunks: Maximum number of chunks to process.
    """

    input_text: str = Field(description="Input text", default="")
    current_domain: str = Field(
        description="IRI used for forming document namespace", default=DEFAULT_DOMAIN
    )
    doc_hid: str | None = Field(
        description="An almost unique hash / id for the parent document of the chunk",
        default=None,
    )
    files: dict[str, bytes] = Field(
        default_factory=lambda: dict(), description="Files to process"
    )
    current_chunk: Optional[Chunk] = Field(
        description="Current document chunk for processing", default=None
    )
    chunks: list[Chunk] = Field(
        default_factory=lambda: list(), description="Chunks of the input text"
    )
    chunks_processed: list[Chunk] = Field(
        default_factory=lambda: list(), description="Chunks of the input text"
    )
    current_ontology: Ontology = Field(
        default_factory=lambda: Ontology(
            ontology_id=ONTOLOGY_NULL_ID,
            title="null title",
            description="null description",
            graph=RDFGraph(),
            iri=ONTOLOGY_NULL_IRI,
        ),
        description="Ontology object that contain the semantic graph "
        "as well as the description, name, short name, version, "
        "and IRI of the ontology",
    )
    aggregated_facts: Optional[RDFGraph] = Field(
        description="RDF triples representing aggregated facts "
        "from the current document",
        default_factory=RDFGraph,
    )
    ontology_addendum: Ontology = Field(
        default_factory=lambda: Ontology(
            ontology_id=ONTOLOGY_NULL_ID,
            title="null title",
            description="null description",
            graph=RDFGraph(),
            iri=ONTOLOGY_NULL_IRI,
        ),
        description="Ontology object that contain the semantic graph "
        "as well as the description, name, short name, version, "
        "and IRI of the ontology",
    )
    failure_stage: str | None = None
    failure_reason: str | None = None
    success_score: float = 0.0
    status: Status = Status.SUCCESS
    node_visits: defaultdict[WorkflowNode, int] = Field(
        default_factory=lambda: defaultdict(int),
        description="Number of visits per node",
    )
    max_visits: int = Field(
        default=3, description="Maximum number of visits allowed per node"
    )
    max_chunks: int | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
    skip_ontology_development: bool = Field(
        default=False, description="Skip ontology create/improve steps if True"
    )

    def model_post_init(self, __context):
        """Post-initialization hook for the model."""
        pass

    def __init__(self, **kwargs):
        """Initialize the agent state with given keyword arguments."""
        super().__init__(**kwargs)
        self.current_domain = os.getenv("CURRENT_DOMAIN", DEFAULT_DOMAIN)

    def set_text(self, text):
        """Set the input text and generate document hash.

        Args:
            text: The input text to set.
        """
        self.input_text = text
        self.doc_hid = render_text_hash(self.input_text)

    def set_failure(self, stage: str, reason: str, success_score: float = 0.0):
        """Set failure state with stage and reason.

        Args:
            stage: The stage where the failure occurred.
            reason: The reason for the failure.
            success_score: The success score at failure (default: 0.0).
        """
        self.failure_stage = stage
        self.failure_reason = reason
        self.success_score = success_score
        self.status = Status.FAILED

    def clear_failure(self):
        """Clear failure state and set status to success."""
        self.failure_stage = None
        self.failure_reason = None
        self.success_score = 0.0
        self.status = Status.SUCCESS

    @property
    def doc_iri(self):
        """Get the document IRI.

        Returns:
            str: The document IRI.
        """
        return f"{self.current_domain}/doc/{self.doc_hid}"

    @property
    def doc_namespace(self):
        """Get the document namespace.

        Returns:
            str: The document namespace.
        """
        return iri2namespace(self.doc_iri, ontology=False)
