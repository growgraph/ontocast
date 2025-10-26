import logging
import pathlib
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Annotated, Union

from pydantic import BaseModel, ConfigDict, Field
from rdflib import DCTERMS, OWL, RDF, RDFS, XSD, Literal, URIRef

from ontocast.onto.constants import DEFAULT_DOMAIN, ONTOLOGY_NULL_IRI
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import GraphUpdate, TripleOp
from ontocast.onto.util import derive_ontology_id
from ontocast.util import iri2namespace

logger = logging.getLogger(__name__)

# Semantic version pattern: MAJOR.MINOR.PATCH (e.g., 1.2.3)
SemanticVersion = Annotated[
    str,
    Field(
        pattern=r"^\d+\.\d+\.\d+$",
        description="Semantic version in MAJOR.MINOR.PATCH format (e.g., 1.2.3)",
    ),
]


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
        description="Ontology identifier, an human readable lower case abbreviation.",
    )
    title: str | None = Field(default=None, description="Ontology title.")
    description: str | None = Field(
        default=None,
        description="A concise description (3-4 sentences) of the ontology "
        "(domain, purpose, applicability, etc.)",
    )
    version: SemanticVersion | None = Field(
        default=None,
        description="Version of the ontology (use semantic versioning)",
    )
    iri: str = Field(
        default=ONTOLOGY_NULL_IRI,
        description="Ontology IRI (Internationalized Resource Identifier)",
    )
    updated_at: datetime | None = Field(
        default=None,
        description="Timestamp when the ontology was last updated (UTC)",
    )
    initial_version: SemanticVersion | None = Field(
        default=None,
        description=(
            "The initial version of the ontology when it was first loaded "
            "in this session"
        ),
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
        current_domain: The domain used to construct the ontology IRI
            if ontology_id is set.
    """

    graph: RDFGraph = Field(
        default_factory=RDFGraph,
        description="RDF triples that define an ontology "
        "in turtle format: use prefixes for namespaces, do NOT add comments.",
    )

    current_domain: str = Field(
        default=DEFAULT_DOMAIN, description="Domain for ontology IRI construction."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **kwargs):
        # Pop current_domain if provided, else use DEFAULT_DOMAIN
        current_domain = kwargs.pop("current_domain", DEFAULT_DOMAIN)
        super().__init__(**kwargs)
        self.current_domain = current_domain
        # Only apply fallback if graph doesn't contain an owl:Ontology
        # Try to sync from graph first
        graph_had_ontology = False
        if self.graph:
            # Try to extract from graph
            self.sync_properties_from_graph()
            # If after sync, both iri and ontology_id are set, do nothing further
            if self.iri and self.iri != ONTOLOGY_NULL_IRI and self.ontology_id:
                graph_had_ontology = True
        # Only apply fallback if graph did not provide a valid pair
        if not graph_had_ontology:
            if self.ontology_id and (not self.iri or self.iri == ONTOLOGY_NULL_IRI):
                self.iri = f"{self.current_domain}/{self.ontology_id}"
            elif self.ontology_id and self.iri:
                expected_iri = f"{self.current_domain}/{self.ontology_id}"
                if not self.iri.endswith(f"/{self.ontology_id}"):
                    logger.warning(
                        f"Ontology IRI '{self.iri}' does not match expected "
                        f"'{expected_iri}', correcting IRI"
                    )
                    self.iri = expected_iri
            elif not self.ontology_id and self.iri and self.iri != ONTOLOGY_NULL_IRI:
                self.ontology_id = derive_ontology_id(self.iri)
        # Set default values for fields that are still None
        if self.version is None:
            self.version = "1.0.0"

        # Always ensure graph is up to date with properties
        self.sync_properties_to_graph()

        # Set initial_version if not already set
        if self.initial_version is None and self.version:
            # Normalize version to ensure semantic versioning
            self.initial_version = self._normalize_version(self.version)

    @property
    def prefix(self) -> str | None:
        """Get the namespace prefix for this ontology.

        Returns:
            str | None: The namespace prefix if found, None otherwise.
        """
        prefixes = [
            prefix
            for prefix, iri in self.graph.namespaces()
            if iri == URIRef(self.namespace)
        ]
        if len(prefixes) == 0:
            return None
        else:
            return prefixes[0]

    def set_properties(self, **kwargs):
        """Set ontology properties from keyword arguments and sync to graph.
        Only update properties if they are missing (None or empty).
        Also enforces ontology_id/iri consistency as in __init__, but only
        if graph does not provide a valid pair.
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
            if self.iri and self.iri != ONTOLOGY_NULL_IRI and self.ontology_id:
                graph_had_ontology = True
        if not graph_had_ontology:
            if self.ontology_id and (not self.iri or self.iri == ONTOLOGY_NULL_IRI):
                self.iri = f"{self.current_domain}/{self.ontology_id}"
            elif self.ontology_id and self.iri:
                expected_iri = f"{self.current_domain}/{self.ontology_id}"
                if not self.iri.endswith(f"/{self.ontology_id}"):
                    logger.warning(
                        f"Ontology IRI '{self.iri}' does not match expected "
                        f"'{expected_iri}'"
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

        if self.ontology_id is not None:
            if not self.iri or self.iri == ONTOLOGY_NULL_IRI:
                self.iri = f"{self.current_domain}/{self.ontology_id}"
            elif self.iri:
                expected_iri = f"{self.current_domain}/{self.ontology_id}"
                if not self.iri.endswith(f"/{self.ontology_id}"):
                    logger.warning(
                        f"Ontology IRI '{self.iri}' does not match expected "
                        f"'{expected_iri}', fixing"
                    )
                    self.iri = expected_iri
        elif self.iri:
            self.ontology_id = derive_ontology_id(self.iri)

        if self.iri == ONTOLOGY_NULL_IRI:
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
        # Add version (update if exists)
        if self.version:
            # Remove existing version triples to update them
            for _, _, obj in g.triples((onto_iri, OWL.versionInfo, None)):
                g.remove((onto_iri, OWL.versionInfo, obj))
            # Add new version
            g.add((onto_iri, OWL.versionInfo, Literal(self.version)))
        # Add updated_at if set
        if self.updated_at:
            # Remove existing dcterms:modified triples to update them
            for _, _, obj in g.triples((onto_iri, DCTERMS.modified, None)):
                g.remove((onto_iri, DCTERMS.modified, obj))
            # Add new updated_at with datetime type
            g.add(
                (
                    onto_iri,
                    DCTERMS.modified,
                    Literal(self.updated_at.isoformat(), datatype=XSD.dateTime),
                )
            )

    def _normalize_version(self, version: str) -> str:
        """Normalize version string to semantic versioning format.

        Handles various version formats and converts them to MAJOR.MINOR.PATCH:
        - "3.5.1" -> "3.5.1" (already valid)
        - "3.5" -> "3.5.0" (adds missing PATCH)
        - "3" -> "3.0.0" (adds missing MINOR and PATCH)
        - Invalid formats -> "1.0.0"

        Args:
            version: The version string to normalize

        Returns:
            A valid semantic version string (MAJOR.MINOR.PATCH)
        """
        # Already valid semantic version
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version)
        if match:
            return version

        # Try to parse as MAJOR.MINOR (missing PATCH)
        match = re.match(r"^(\d+)\.(\d+)$", version)
        if match:
            major, minor = match.groups()
            normalized = f"{major}.{minor}.0"
            logger.info(
                f"Version '{version}' missing PATCH component, normalized to '{normalized}'"
            )
            return normalized

        # Try to parse as just MAJOR (missing MINOR and PATCH)
        match = re.match(r"^(\d+)$", version)
        if match:
            major = match.group(1)
            normalized = f"{major}.0.0"
            logger.info(
                f"Version '{version}' missing MINOR and PATCH components, normalized to '{normalized}'"
            )
            return normalized

        # Invalid format, use default
        logger.warning(
            f"Version '{version}' does not match any recognized format, "
            f"normalizing to '1.0.0'"
        )
        return "1.0.0"

    def _analyze_version_increment_type(
        self, updates: list[GraphUpdate]
    ) -> tuple[str, str]:
        """Analyze the updates to determine the appropriate version increment type.

        Args:
            updates: List of GraphUpdate objects that were applied to the ontology

        Returns:
            Tuple of (increment_type, reason) where increment_type is
            'major', 'minor', or 'patch' and reason explains the decision
        """
        if not updates:
            return ("patch", "No updates to analyze")

        # Count operations by type
        total_deletes = 0
        total_inserts = 0

        # Track specific types of changes
        class_changes = 0
        property_changes = 0
        instance_changes = 0

        for update in updates:
            for op in update.operations:
                if isinstance(op, TripleOp):
                    if op.type == "delete":
                        total_deletes += len(op.triples)
                        # Check if deleting core ontology constructs
                        for triple in op.triples:
                            if "rdf:type" in triple.predicate:
                                if any(
                                    cls in triple.object.lower()
                                    for cls in ["class", "property", "ontology"]
                                ):
                                    if (
                                        "owl:class" in triple.object
                                        or "rdfs:class" in triple.object
                                    ):
                                        class_changes += 1
                                    elif "owl:ontology" in triple.object:
                                        class_changes += 1
                    else:  # insert
                        total_inserts += len(op.triples)
                        # Check if adding core ontology constructs
                        for triple in op.triples:
                            if "rdf:type" in triple.predicate:
                                if (
                                    "owl:class" in triple.object
                                    or "rdfs:class" in triple.object
                                ):
                                    class_changes += 1
                                elif "owl:ontology" in triple.object:
                                    class_changes += 1
                                elif (
                                    "owl:objectproperty" in triple.object
                                    or "owl:datatypeproperty" in triple.object
                                    or "rdf:property" in triple.object
                                ):
                                    property_changes += 1
                                else:
                                    instance_changes += 1

        # Decision logic - conservative approach, favor PATCH

        # Check for substantial breaking changes first (MAJOR)
        if total_deletes > 5 and (class_changes > 2 or property_changes > 3):
            reason = (
                f"MAJOR: Deleted {total_deletes} triples including "
                f"{class_changes} classes and {property_changes} properties "
                "(significant breaking change)"
            )
            return ("major", reason)

        # Any deletions trigger MINOR (even small ones indicate changes)
        if total_deletes > 0:
            reason = (
                f"MINOR: Deleted {total_deletes} triples "
                f"({class_changes} classes, {property_changes} properties removed)"
            )
            return ("minor", reason)

        # Only increment MINOR for substantial new features (>=5 classes or properties)
        if class_changes >= 5 or property_changes >= 5:
            reason = (
                f"MINOR: Added {total_inserts} triples including "
                f"{class_changes} classes and {property_changes} properties "
                "(substantial new features)"
            )
            return ("minor", reason)

        # Default to PATCH for most additions
        # This includes: instances, descriptions, small numbers of classes/properties
        reason = f"PATCH: Added {total_inserts} triples"
        if class_changes > 0 or property_changes > 0:
            reason += f" ({class_changes} classes, {property_changes} properties)"
        reason += " (updates to existing structures)"
        return ("patch", reason)

    def _increment_version(self, increment_type: str = "patch") -> None:
        """Increment the ontology version using semantic versioning.

        Args:
            increment_type: Type of increment - 'major', 'minor', or 'patch'
        """
        # If version is None, set to default
        if self.version is None:
            self.version = "1.0.0"
            return

        # Normalize to ensure semantic versioning
        normalized_version = self._normalize_version(self.version)
        if normalized_version != self.version:
            logger.warning(
                f"Version '{self.version}' normalized to '{normalized_version}' "
                "before incrementing"
            )
            self.version = normalized_version

        # Parse and increment version string based on increment_type
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", self.version)
        if match:
            major, minor, patch = map(int, match.groups())

            if increment_type == "major":
                major += 1
                minor = 0
                patch = 0
                logger.info(
                    f"Incrementing MAJOR version from {self.version} to {major}.{minor}.{patch}"
                )
            elif increment_type == "minor":
                minor += 1
                patch = 0
                logger.info(
                    f"Incrementing MINOR version from {self.version} to {major}.{minor}.{patch}"
                )
            else:  # patch
                patch += 1
                logger.info(
                    f"Incrementing PATCH version from {self.version} to {major}.{minor}.{patch}"
                )

            self.version = f"{major}.{minor}.{patch}"
        else:
            # Should never reach here after normalization, but handle gracefully
            logger.error(f"Version '{self.version}' still invalid after normalization")
            self.version = "1.0.1"

        logger.info(f"Incremented ontology version to {self.version}")

    def mark_as_updated(self, updates: list[GraphUpdate] | None = None) -> None:
        """Mark the ontology as updated and update version.

        Sets the updated_at timestamp to now (UTC) and increments the version.
        Analyzes the updates to determine appropriate version increment type.

        Args:
            updates: Optional list of GraphUpdate objects that were applied.
                If provided, analyzes them to determine MAJOR/MINOR/PATCH increment.
        """
        self.updated_at = datetime.now(timezone.utc)

        # Analyze updates to determine increment type
        if updates:
            increment_type, reason = self._analyze_version_increment_type(updates)
            logger.info(f"Version increment analysis: {reason}")
            self._increment_version(increment_type)
        else:
            # Default to patch increment if no updates provided
            self._increment_version("patch")

        logger.info(
            f"Marked ontology {self.ontology_id} as updated at {self.updated_at}"
        )

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
        if self.title is None:
            title = None
            if RDFS.label in pred_map:
                title = str(pred_map[RDFS.label][0])
            elif DCTERMS.title in pred_map:
                title = str(pred_map[DCTERMS.title][0])
            if title:
                self.title = title

        # Description: try dcterms:description, rdfs:comment
        if self.description is None:
            description = None
            if DCTERMS.description in pred_map:
                description = str(pred_map[DCTERMS.description][0])
            elif RDFS.comment in pred_map:
                description = str(pred_map[RDFS.comment][0])
            if description:
                self.description = description
        # Version
        if self.version is None:
            if OWL.versionInfo in pred_map:
                version_str = str(pred_map[OWL.versionInfo][0])
                self.version = self._normalize_version(version_str)
        # Updated at
        if not getattr(self, "updated_at", None):
            if DCTERMS.modified in pred_map:
                # Get the first modified date
                modified_str = str(pred_map[DCTERMS.modified][0])
                # Try to parse as datetime
                try:
                    self.updated_at = datetime.fromisoformat(
                        modified_str.replace("Z", "+00:00")
                    )
                except (ValueError, AttributeError):
                    # If parsing fails, keep it as None
                    pass
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
            self.updated_at = other.updated_at
            self.initial_version = other.initial_version
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
