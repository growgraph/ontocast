import logging
import pathlib
import re
from collections import defaultdict
from datetime import datetime
from typing import Annotated, Union

from pydantic import BaseModel, ConfigDict, Field
from rdflib import DCTERMS, OWL, RDF, RDFS, XSD, Literal, URIRef

from ontocast.onto.constants import DEFAULT_DOMAIN, ONTOLOGY_NULL_IRI, PROV
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


class OntologyPropertiesWithLineage(OntologyProperties):
    """Properties of an ontology with versioning lineage information.

    This class extends OntologyProperties with hash-based versioning support,
    similar to git-style versioning. Each ontology has a hash of its graph
    and optionally multiple parent hashes to support parallel branches and merges.

    Attributes:
        hash: Hash of the ontology graph (computed from canonicalized graph).
        parent_hashes: List of hashes of parent ontologies. Supports multiple
            parents for parallel branches and merges. Can be empty, indicating
            this is a root ontology with no parents.
        created_at: Timestamp when the ontology version was created (UTC).
            This is set deterministically when a version is created, not by LLM.
    """

    hash: str | None = Field(
        default=None,
        description="Hash of the ontology graph (SHA256 of canonicalized graph)",
    )
    parent_hashes: list[str] = Field(
        default_factory=list,
        description=(
            "List of hashes of parent ontologies. Supports multiple parents "
            "for parallel branches and merges. Can be empty, indicating "
            "this is a root ontology with no parents."
        ),
    )
    created_at: datetime | None = Field(
        default=None,
        description="Timestamp when the ontology version was created (UTC). "
        "Set deterministically when a version is created, not by LLM.",
    )

    @property
    def versioned_iri(self) -> str:
        """Get the versioned URI for this ontology (for storage purposes).

        This creates a versioned URI using hash-based fragments for git-style
        versioning. Format: <base_iri>#<hash>. Falls back to semantic version
        fragment (#v1.2.3) or base IRI if hash is not available.

        This allows multiple versions of the same ontology to coexist in storage
        (e.g., Fuseki named graphs). The semantic ontology IRI in the graph
        remains unchanged; this is only used for storage organization.

        Returns:
            str: The versioned URI with hash fragment, or semantic version fragment,
            or base IRI if neither is available.

        Examples:
            >>> ont = Ontology(iri="https://growgraph.dev/fcaont", hash="abc123...")
            >>> ont.versioned_iri
            'https://growgraph.dev/fcaont#abc123...'
            >>> ont2 = Ontology(iri="http://example.org/ontology", version="1.0.0")
            >>> ont2.versioned_iri
            'http://example.org/ontology#v1.0.0'
        """
        if self.hash:
            # Use hash-based fragment for git-style versioning
            return f"{self.iri}#{self.hash}"
        elif self.version:
            # Fall back to semantic version fragment for backward compatibility
            return f"{self.iri}#v{self.version}"
        return self.iri


class Ontology(OntologyPropertiesWithLineage):
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

        # Check if this is explicitly a null ontology (only if both IRI is null AND no graph provided)
        # Don't return early if graph is provided - graph might contain ontology information
        is_explicitly_null = (
            self.iri == ONTOLOGY_NULL_IRI
            and self.ontology_id is None
            and (not self.graph or len(self.graph) == 0)
        )
        if is_explicitly_null:
            # This is explicitly a null ontology - don't derive ontology_id, don't compute hash, etc.
            return

        # Parse IRI fragment for hash-based or version-based identifiers
        if self.iri and "#" in self.iri:
            base_iri, fragment = self.iri.rsplit("#", 1)
            # Check if fragment is a hash (long hex string) or version (v1.2.3)
            if len(fragment) > 20 and all(c in "0123456789abcdef" for c in fragment):
                # Looks like a hash - extract it
                if self.hash is None:
                    self.hash = fragment
                    self.iri = base_iri  # Remove fragment from IRI
                    logger.debug(f"Extracted hash from IRI fragment: {fragment}")
            elif fragment.startswith("v") and re.match(r"^v\d+\.\d+\.\d+$", fragment):
                # Semantic version fragment - extract version
                version_str = fragment[1:]  # Remove 'v' prefix
                if self.version is None:
                    self.version = version_str
                self.iri = base_iri  # Remove fragment from IRI
                logger.debug(f"Extracted version from IRI fragment: {version_str}")

        # Try to sync from graph first (this is the primary source of truth)
        graph_had_ontology = False
        iri_from_graph = None
        if self.graph:
            # Try to extract from graph
            self.sync_properties_from_graph()
            # Check if graph provided valid ontology information
            # IRI should be set and not null, ontology_id should be set
            if self.iri and self.iri != ONTOLOGY_NULL_IRI:
                iri_from_graph = self.iri  # Remember that IRI came from graph
                if self.ontology_id:
                    graph_had_ontology = True
                else:
                    # IRI is set but ontology_id is missing - try to derive it
                    self.ontology_id = (
                        self._extract_ontology_id_from_prefixes()
                        or derive_ontology_id(self.iri)
                    )
                    if self.ontology_id:
                        graph_had_ontology = True

        # Only apply fallback if graph did not provide a valid pair
        if not graph_had_ontology:
            # Try to extract ontology_id from prefixes if IRI is available
            if self.iri and self.iri != ONTOLOGY_NULL_IRI and not self.ontology_id:
                # Prefer derivation from IRI over prefix
                derived_id = derive_ontology_id(self.iri)
                prefix_id = self._extract_ontology_id_from_prefixes()

                if derived_id:
                    self.ontology_id = derived_id
                    # If prefix exists but doesn't match ontology_id, rebind it
                    if prefix_id and prefix_id != derived_id:
                        self._rebind_prefix_to_ontology_id(prefix_id, derived_id)
                elif prefix_id:
                    # Fallback to prefix if IRI derivation fails
                    self.ontology_id = prefix_id

            # Fallback logic: construct IRI from ontology_id or vice versa
            # BUT: Never override IRI that came from graph
            if self.ontology_id and (not self.iri or self.iri == ONTOLOGY_NULL_IRI):
                self.iri = f"{self.current_domain}/{self.ontology_id}"
            elif self.ontology_id and self.iri and self.iri != ONTOLOGY_NULL_IRI:
                # IRI is set - check if it came from graph
                if iri_from_graph and self.iri == iri_from_graph:
                    # IRI came from graph - don't override, just log if pattern doesn't match
                    expected_iri = f"{self.current_domain}/{self.ontology_id}"
                    if (
                        not self.iri.endswith(f"/{self.ontology_id}")
                        and self.iri != expected_iri
                    ):
                        logger.debug(
                            f"Ontology IRI '{self.iri}' from graph does not match expected pattern "
                            f"'{expected_iri}', but keeping IRI from graph (authoritative)"
                        )
                else:
                    # IRI didn't come from graph - check if it matches expected pattern
                    expected_iri = f"{self.current_domain}/{self.ontology_id}"
                    if (
                        not self.iri.endswith(f"/{self.ontology_id}")
                        and self.iri != expected_iri
                    ):
                        logger.warning(
                            f"Ontology IRI '{self.iri}' does not match expected "
                            f"'{expected_iri}', correcting IRI"
                        )
                        self.iri = expected_iri
            elif not self.ontology_id and self.iri and self.iri != ONTOLOGY_NULL_IRI:
                # Extract ontology_id: prefer IRI derivation, rebind prefix if needed
                derived_id = derive_ontology_id(self.iri)
                prefix_id = self._extract_ontology_id_from_prefixes()

                if derived_id:
                    self.ontology_id = derived_id
                    # If prefix exists but doesn't match ontology_id, rebind it
                    if prefix_id and prefix_id != derived_id:
                        self._rebind_prefix_to_ontology_id(prefix_id, derived_id)
                elif prefix_id:
                    # Fallback to prefix if IRI derivation fails
                    self.ontology_id = prefix_id
        # Set default values for fields that are still None
        if self.version is None:
            self.version = "1.0.0"

        self._compute_and_set_hash()

        # Always ensure graph is up to date with properties (including hash/parent_hashes)
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

    def is_null(self) -> bool:
        """Check if this ontology is the null ontology.

        Returns:
            bool: True if this is NULL_ONTOLOGY or has null characteristics.
        """
        from ontocast.onto.null import NULL_ONTOLOGY

        # Check identity first (fastest)
        if self is NULL_ONTOLOGY:
            return True
        # Check characteristics
        return self.iri == ONTOLOGY_NULL_IRI and self.ontology_id is None

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
            if self.iri and not self.is_null() and self.ontology_id:
                graph_had_ontology = True
        if not graph_had_ontology:
            if self.ontology_id and (not self.iri or self.is_null()):
                self.iri = f"{self.current_domain}/{self.ontology_id}"
            elif self.ontology_id and self.iri:
                expected_iri = f"{self.current_domain}/{self.ontology_id}"
                if not self.iri.endswith(f"/{self.ontology_id}"):
                    logger.warning(
                        f"Ontology IRI '{self.iri}' does not match expected "
                        f"'{expected_iri}'"
                    )
            elif not self.ontology_id and self.iri and not self.is_null():
                self.ontology_id = derive_ontology_id(self.iri)
        self.sync_properties_to_graph()

    def sync_properties_to_graph(self):
        """
        Update the RDF graph with the Ontology's properties.
        Only sync properties for the entity that is explicitly typed as owl:Ontology.
        Only add property triples if they do not already exist in the graph.
        Optimized to avoid multiple loops over triples.
        """

        # Early return for NULL_ONTOLOGY - don't sync anything
        if self.is_null():
            return

        if self.ontology_id is not None:
            if not self.iri or self.is_null():
                self.iri = f"{self.current_domain}/{self.ontology_id}"
            elif self.iri:
                expected_iri = f"{self.current_domain}/{self.ontology_id}"
                # Only fix IRI if it doesn't match expected pattern AND it's not from an external source
                # Don't override IRIs that came from the graph or were explicitly provided
                if (
                    not self.iri.endswith(f"/{self.ontology_id}")
                    and self.iri != expected_iri
                ):
                    # Check if IRI looks like it came from an external source (not our default domain)
                    if self.current_domain not in self.iri:
                        # IRI is from external source (e.g., graph) - don't override
                        logger.debug(
                            f"Ontology IRI '{self.iri}' does not match expected pattern "
                            f"'{expected_iri}', but keeping IRI (likely from graph or external source)"
                        )
                    else:
                        # IRI is from our domain but doesn't match - fix it
                        logger.warning(
                            f"Ontology IRI '{self.iri}' does not match expected "
                            f"'{expected_iri}', fixing"
                        )
                        self.iri = expected_iri
        elif self.iri and not self.is_null():
            # Only derive ontology_id if this is not a null ontology
            self.ontology_id = derive_ontology_id(self.iri)

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
        # Add created_at if set (only if not already present in graph)
        if self.created_at:
            # Check if created_at already exists in graph - don't overwrite if present
            existing_created = [
                str(obj) for _, _, obj in g.triples((onto_iri, DCTERMS.created, None))
            ]
            if not existing_created:
                # Add new created_at with datetime type
                g.add(
                    (
                        onto_iri,
                        DCTERMS.created,
                        Literal(self.created_at.isoformat(), datatype=XSD.dateTime),
                    )
                )
        # Add hash (only if not already present in graph)
        # Use dcterms:identifier for hash (with "hash:" prefix to distinguish from other identifiers)
        if self.hash:
            # Check if hash already exists in graph
            existing_hash = [
                str(obj)
                for _, _, obj in g.triples((onto_iri, DCTERMS.identifier, None))
                if str(obj).startswith("hash:")
            ]
            if not existing_hash:
                g.add((onto_iri, DCTERMS.identifier, Literal(f"hash:{self.hash}")))

        # Add parent_hashes (multiple parents supported)
        # Use prov:wasDerivedFrom for each parent hash (standard PROV predicate)
        if self.parent_hashes:
            # Get existing parent hashes to avoid duplicates
            existing_parent_uris = {
                str(obj)
                for _, _, obj in g.triples((onto_iri, PROV.wasDerivedFrom, None))
            }

            # Add each parent hash as a URIRef if not already present
            for parent_hash in self.parent_hashes:
                parent_hash_uri = URIRef(f"urn:hash:{parent_hash}")
                if str(parent_hash_uri) not in existing_parent_uris:
                    g.add((onto_iri, PROV.wasDerivedFrom, parent_hash_uri))

    def _compute_and_set_hash(self) -> None:
        """Compute the hash of the ontology graph and set it.

        The hash is computed from the canonicalized graph using SHA256.
        The hash is computed from the graph WITHOUT hash/parent_hash triples,
        as these are metadata about the graph, not part of the graph content.
        This method should only be called if hash is not already set.
        """
        if self.graph and len(self.graph) > 0:
            try:
                # Find the ontology IRI from the graph if not set
                onto_iri = None
                if self.iri and not self.is_null():
                    onto_iri = URIRef(self.iri)
                else:
                    # Try to find ontology IRI from graph
                    onto_triples = [
                        subj
                        for subj, _, o in self.graph.triples((None, RDF.type, None))
                        if o == OWL.Ontology
                    ]
                    if onto_triples:
                        onto_iri = onto_triples[0]

                # Create a temporary graph without hash/parent_hash triples for hashing
                temp_graph = RDFGraph()

                # Copy all triples except metadata triples - these are metadata, not content
                # Metadata to exclude: hash, parent_hash, created_at, version, title, description
                for s, p, o in self.graph:
                    # Skip metadata triples for the ontology IRI
                    if onto_iri and s == onto_iri:
                        if (
                            p == DCTERMS.identifier
                            and isinstance(o, Literal)
                            and str(o).startswith("hash:")
                        ):
                            continue  # Skip hash identifier
                        if p == PROV.wasDerivedFrom:
                            continue  # Skip parent hash
                        if p == DCTERMS.created:
                            continue  # Skip created_at
                        if p == OWL.versionInfo:
                            continue  # Skip version
                        if p == RDFS.label:
                            continue  # Skip title/label
                        if p == DCTERMS.title:
                            continue  # Skip title
                        if p == DCTERMS.description:
                            continue  # Skip description
                        if p == RDFS.comment:
                            continue  # Skip description (comment)
                    temp_graph.add((s, p, o))

                # Copy namespace bindings
                for prefix, uri in self.graph.namespaces():
                    temp_graph.bind(prefix, uri)

                # Use RDFGraph.hash() directly
                self.hash = temp_graph.hash()
                logger.debug(
                    f"Computed hash for ontology {self.ontology_id}: {self.hash}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to compute hash for ontology {self.ontology_id}: {e}"
                )
                # Set a placeholder hash if computation fails
                self.hash = None

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
            for op in update.triple_operations:
                if isinstance(op, TripleOp):
                    if op.type == "delete":
                        total_deletes += len(op.graph)
                        # Check if deleting core ontology constructs
                        for subject, predicate, object_ in op.graph:
                            predicate_str = str(predicate)
                            object_str = str(object_)
                            if "rdf:type" in predicate_str:
                                if any(
                                    cls in object_str.lower()
                                    for cls in ["class", "property", "ontology"]
                                ):
                                    if (
                                        "owl:class" in object_str
                                        or "rdfs:class" in object_str
                                    ):
                                        class_changes += 1
                                    elif "owl:ontology" in object_str:
                                        class_changes += 1
                    else:  # insert
                        total_inserts += len(op.graph)
                        # Check if adding core ontology constructs
                        for subject, predicate, object_ in op.graph:
                            predicate_str = str(predicate)
                            object_str = str(object_)
                            if "rdf:type" in predicate_str:
                                if (
                                    "owl:class" in object_str
                                    or "rdfs:class" in object_str
                                ):
                                    class_changes += 1
                                elif "owl:ontology" in object_str:
                                    class_changes += 1
                                elif (
                                    "owl:objectproperty" in object_str
                                    or "owl:datatypeproperty" in object_str
                                    or "rdf:property" in object_str
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
        """Mark the ontology version and update semantic version.

        Note: Ontologies are immutable - modifications create new versions.
        This method only updates the semantic version number, not the creation timestamp.
        The creation timestamp is set when a new version is created.

        Analyzes the updates to determine appropriate version increment type.

        Args:
            updates: Optional list of GraphUpdate objects that were applied.
                If provided, analyzes them to determine MAJOR/MINOR/PATCH increment.
        """
        # Analyze updates to determine increment type
        if updates:
            increment_type, reason = self._analyze_version_increment_type(updates)
            logger.info(f"Version increment analysis: {reason}")
            self._increment_version(increment_type)
        else:
            # Default to patch increment if no updates provided
            self._increment_version("patch")

        logger.info(
            f"Updated semantic version for ontology {self.ontology_id} to {self.version}"
        )

    def _extract_ontology_id_from_prefixes(self) -> str | None:
        """Extract ontology_id from namespace prefixes that match the ontology IRI.

        Looks for prefixes where the namespace URI matches the ontology IRI or namespace.
        For example, if IRI is 'https://growgraph.dev/fcaont' and there's a prefix
        'fca' with namespace 'https://growgraph.dev/fcaont#', returns 'fca'.

        Returns:
            str | None: The prefix name if found, None otherwise.
        """
        if not self.graph or not self.iri or self.iri == ONTOLOGY_NULL_IRI:
            return None

        # Try exact IRI match first
        ontology_namespace = iri2namespace(self.iri, ontology=True)

        for prefix, namespace_uri in self.graph.namespaces():
            namespace_str = str(namespace_uri)
            # Check if namespace matches ontology IRI or namespace
            if namespace_str == self.iri or namespace_str == ontology_namespace:
                if prefix and prefix not in [
                    "rdf",
                    "rdfs",
                    "owl",
                    "xsd",
                    "dc",
                    "dcterms",
                    "skos",
                    "foaf",
                    "schema",
                    "prov",
                ]:
                    logger.debug(f"Found prefix '{prefix}' matching IRI '{self.iri}'")
                    return prefix

        return None

    def _rebind_prefix_to_ontology_id(self, old_prefix: str, ontology_id: str) -> None:
        """Rebind a prefix to match the ontology_id.

        If a prefix exists that matches the ontology IRI but has a different name
        than the ontology_id, rebind it to use the ontology_id as the prefix name.
        This ensures consistency between the prefix name and ontology_id.

        Args:
            old_prefix: The existing prefix name that needs to be rebound.
            ontology_id: The ontology_id that should be used as the new prefix name.
        """
        if not self.graph or not self.iri or self.iri == ONTOLOGY_NULL_IRI:
            return

        ontology_namespace = iri2namespace(self.iri, ontology=True)

        # Find the namespace URI for the old prefix
        old_namespace_uri = None
        for prefix, namespace_uri in self.graph.namespaces():
            if prefix == old_prefix:
                old_namespace_uri = str(namespace_uri)
                break

        if old_namespace_uri and old_namespace_uri == ontology_namespace:
            # Only rebind if the namespace matches
            # Bind the new prefix with ontology_id (this will override if it exists)
            from rdflib import Namespace

            ns = Namespace(ontology_namespace)
            self.graph.namespace_manager.bind(ontology_id, ns, override=True)

            # If old prefix is different, we can optionally remove it
            # But keep it for now to avoid breaking existing references in the graph
            # The new prefix will be used going forward
            logger.debug(
                f"Rebound prefix: '{old_prefix}' -> '{ontology_id}' "
                f"for namespace '{ontology_namespace}'"
            )

    def sync_properties_from_graph(self):
        """
        Update Ontology properties from the RDF graph if present,
        but only if missing, and only for entities explicitly typed as owl:Ontology.
        Optimized to avoid multiple loops over triples.
        """
        g = self.graph
        if not g or len(g) == 0:
            return

        # Only proceed if this subject is explicitly typed as owl:Ontology
        onto_triple = [
            subj
            for subj, _, o in g.triples((None, RDF.type, None))
            if o == OWL.Ontology
        ]
        if not onto_triple:
            # No owl:Ontology found - try to extract IRI from prefixes as fallback
            if not self.iri or self.iri == ONTOLOGY_NULL_IRI:
                # Look for prefixes that might indicate the ontology IRI
                for prefix, namespace_uri in g.namespaces():
                    namespace_str = str(namespace_uri).rstrip("#/")
                    # Skip standard prefixes
                    if prefix and prefix not in [
                        "rdf",
                        "rdfs",
                        "owl",
                        "xsd",
                        "dc",
                        "dcterms",
                        "skos",
                        "foaf",
                        "schema",
                        "prov",
                    ]:
                        # Use this namespace as potential IRI
                        self.iri = namespace_str
                        self.ontology_id = prefix
                        logger.debug(
                            f"No owl:Ontology found, extracted IRI '{self.iri}' and "
                            f"ontology_id '{self.ontology_id}' from prefix '{prefix}'"
                        )
                        return
            return

        onto_iri = onto_triple[0]
        iri_str = str(onto_iri)

        # Strip hash fragment from IRI to ensure simplified representation
        # Hash fragments are long hex strings (64+ chars) used for versioning
        if "#" in iri_str:
            base_iri, fragment = iri_str.rsplit("#", 1)
            # Check if fragment is a hash (long hex string) or version (v1.2.3)
            if len(fragment) > 20 and all(
                c in "0123456789abcdef" for c in fragment.lower()
            ):
                # Looks like a hash - use base IRI only
                iri_str = base_iri
                logger.debug(
                    f"Stripped hash fragment from IRI in graph: {fragment[:20]}..."
                )
            elif fragment.startswith("v") and re.match(r"^v\d+\.\d+\.\d+$", fragment):
                # Semantic version fragment - use base IRI only
                iri_str = base_iri
                logger.debug(f"Stripped version fragment from IRI in graph: {fragment}")

        # Set IRI from graph (this is authoritative)
        if not self.iri or self.iri == ONTOLOGY_NULL_IRI:
            self.iri = iri_str
        elif self.iri != iri_str:
            # Graph has different IRI - prefer graph IRI but log the difference
            logger.debug(
                f"Graph IRI '{iri_str}' differs from provided IRI '{self.iri}', "
                f"using graph IRI"
            )
            self.iri = iri_str

        # Extract ontology_id: prefer derivation from IRI over prefix
        # If both exist, use IRI-derived ontology_id and rebind prefix to match
        if not self.ontology_id:
            # First try to derive from IRI (preferred)
            derived_id = derive_ontology_id(self.iri)
            prefix_id = self._extract_ontology_id_from_prefixes()

            if derived_id:
                self.ontology_id = derived_id
                # If prefix exists but doesn't match ontology_id, rebind it
                if prefix_id and prefix_id != derived_id:
                    self._rebind_prefix_to_ontology_id(prefix_id, derived_id)
            elif prefix_id:
                # Fallback to prefix if IRI derivation fails
                self.ontology_id = prefix_id

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
        # Created at - only read if not already set (preserve existing value)
        if not getattr(self, "created_at", None):
            if DCTERMS.created in pred_map:
                # Get the first created date
                created_str = str(pred_map[DCTERMS.created][0])
                # Try to parse as datetime
                try:
                    self.created_at = datetime.fromisoformat(
                        created_str.replace("Z", "+00:00")
                    )
                except (ValueError, AttributeError):
                    # If parsing fails, keep it as None
                    pass
        # Short name: try dcterms:title if not already used for title
        if not getattr(self, "ontology_id", None):
            if DCTERMS.title in pred_map:
                self.ontology_id = str(pred_map[DCTERMS.title][0])
        # Hash: read from dcterms:identifier with "hash:" prefix if present
        if self.hash is None:
            if DCTERMS.identifier in pred_map:
                for obj in pred_map[DCTERMS.identifier]:
                    obj_str = str(obj)
                    if obj_str.startswith("hash:"):
                        self.hash = obj_str[5:]  # Remove "hash:" prefix
                        break

        # Parent_hashes: read all from prov:wasDerivedFrom if present
        if len(self.parent_hashes) == 0:
            if PROV.wasDerivedFrom in pred_map:
                for parent_uri_obj in pred_map[PROV.wasDerivedFrom]:
                    parent_uri = str(parent_uri_obj)
                    # Extract hash from URN format: urn:hash:<hash>
                    if parent_uri.startswith("urn:hash:"):
                        parent_hash = parent_uri[9:]  # Remove "urn:hash:" prefix
                        self.parent_hashes.append(parent_hash)

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
            self.created_at = other.created_at
            self.initial_version = other.initial_version
            self.hash = other.hash
            self.parent_hashes = other.parent_hashes
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

    def to_lineage_node(self) -> dict:
        """Convert ontology to a lineage node representation.

        Returns a dictionary suitable for constructing a meta-graph representing
        the ontology lineage. This representation can be used to build the full
        ontology lineage graph.

        Returns:
            dict: Lineage node with hash, parents, and metadata.

        Example:
            >>> ont = Ontology(iri="https://example.org/ont", hash="abc123", parent_hashes=["def456"])
            >>> node = ont.to_lineage_node()
            >>> node["hash"]
            'abc123'
            >>> node["parents"]
            ['def456']
        """
        return {
            "hash": self.hash,
            "parents": self.parent_hashes,
            "iri": self.iri,
            "title": self.title,
            "version": self.version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @staticmethod
    def build_lineage_graph(ontologies: list["Ontology"]):
        """Build a NetworkX directed graph representing the lineage of all given ontologies.

        Constructs a directed graph where nodes represent ontologies (by their hash)
        and edges represent parent-child relationships. Each node includes metadata
        as node attributes (iri, title, version, created_at, etc.).

        Args:
            ontologies: List of Ontology instances to include in the lineage graph.

        Returns:
            networkx.DiGraph: A directed graph representing the full ontology lineage.
                Nodes are identified by hash strings, with edges from children to parents.
                Each node has attributes: iri, title, ontology_id, version, created_at.

        Example:
            >>> import networkx as nx
            >>> ont1 = Ontology(iri="https://example.org/ont1", hash="abc123")
            >>> ont2 = Ontology(iri="https://example.org/ont2", hash="def456", parent_hashes=["abc123"])
            >>> lineage = Ontology.build_lineage_graph([ont1, ont2])
            >>> isinstance(lineage, nx.DiGraph)
            True
            >>> "def456" in lineage.nodes()
            True
            >>> "abc123" in lineage["def456"]  # Check if edge exists
            True
        """
        import networkx as nx

        lineage_graph = nx.DiGraph()

        for ontology in ontologies:
            if not ontology.hash:
                logger.warning(
                    f"Skipping ontology {ontology.iri} in lineage graph: no hash"
                )
                continue

            # Add node with metadata attributes
            lineage_graph.add_node(
                ontology.hash,
                iri=ontology.iri,
                title=ontology.title,
                ontology_id=ontology.ontology_id,
                version=ontology.version,
                created_at=ontology.created_at.isoformat()
                if ontology.created_at
                else None,
            )

            # Add edges from this ontology to its parents
            if ontology.parent_hashes:
                for parent_hash in ontology.parent_hashes:
                    # Ensure parent node exists (even if not in the ontologies list)
                    if parent_hash not in lineage_graph:
                        lineage_graph.add_node(parent_hash)
                    lineage_graph.add_edge(ontology.hash, parent_hash)

        return lineage_graph

    def add_parent_hash(self, parent_hash: str) -> None:
        """Add a parent hash to the ontology's parent list.

        Appends the given hash to parent_hashes if not already present,
        and updates the RDF graph accordingly by adding a new prov:wasDerivedFrom triple.

        Args:
            parent_hash: The hash of the parent ontology to add.

        Example:
            >>> ont = Ontology(iri="https://example.org/ont", hash="abc123")
            >>> ont.add_parent_hash("def456")
            >>> "def456" in ont.parent_hashes
            True
        """
        if parent_hash not in self.parent_hashes:
            self.parent_hashes.append(parent_hash)
            # Update graph
            if self.iri and not self.is_null():
                onto_iri = URIRef(self.iri)
                parent_hash_uri = URIRef(f"urn:hash:{parent_hash}")
                self.graph.add((onto_iri, PROV.wasDerivedFrom, parent_hash_uri))
                logger.debug(
                    f"Added parent hash {parent_hash} to ontology {self.ontology_id}"
                )

    def validate_lineage(self) -> list[str]:
        """Validate the ontology lineage for integrity issues.

        Checks for cycles and ensures that self.hash is not in its own parent_hashes.
        Returns a list of warning messages if any issues are found.

        Returns:
            list[str]: List of warning messages describing any lineage issues found.
                Empty list if lineage is valid.

        Example:
            >>> ont = Ontology(iri="https://example.org/ont", hash="abc123", parent_hashes=["abc123"])
            >>> warnings = ont.validate_lineage()
            >>> len(warnings) > 0
            True
        """
        warnings = []

        if not self.hash:
            return warnings

        # Check if hash is in its own parent_hashes
        if self.parent_hashes and self.hash in self.parent_hashes:
            warnings.append(
                f"Ontology {self.ontology_id} (hash: {self.hash[:8]}...) "
                "has itself as a parent, which may indicate a cycle"
            )

        # Check for cycles using a simple depth-first search
        visited = set()
        to_visit = [(self.hash, [self.hash])]

        while to_visit:
            current_hash, path = to_visit.pop()
            if current_hash in visited:
                continue
            visited.add(current_hash)

            # Find ontology with this hash in the graph
            # This is a simplified check - in practice, you'd need access to all ontologies
            # For now, we just check immediate parents
            if self.parent_hashes:
                for parent_hash in self.parent_hashes:
                    if parent_hash == current_hash and len(path) > 1:
                        warnings.append(
                            f"Potential cycle detected in lineage: "
                            f"{' -> '.join(path)} -> {parent_hash}"
                        )
                    elif parent_hash not in visited:
                        to_visit.append((parent_hash, path + [parent_hash]))

        if warnings:
            for warning in warnings:
                logger.warning(warning)

        return warnings
