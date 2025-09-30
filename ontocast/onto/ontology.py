import logging
import pathlib
from collections import defaultdict
from typing import Union

from pydantic import BaseModel, ConfigDict, Field
from rdflib import DCTERMS, OWL, RDF, RDFS, Literal, URIRef

from ontocast.onto.constants import DEFAULT_DOMAIN, ONTOLOGY_NULL_ID, ONTOLOGY_NULL_IRI
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.util import derive_ontology_id
from ontocast.util import iri2namespace

logger = logging.getLogger(__name__)


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

    @property
    def prefix(self) -> str | None:
        """Get the namespace for this ontology.

        Returns:
            str: The namespace string.
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


class Ontology(OntologyProperties):
    """A Pydantic model representing an ontology with its RDF graph and description.

    Attributes:
        graph: The RDF graph containing the ontology data.
        current_domain: The domain used to construct the ontology IRI if ontology_id is set.
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
