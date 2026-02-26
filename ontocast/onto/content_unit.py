from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator
from rdflib import URIRef

from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.util import iri2namespace


class OutputType(StrEnum):
    FACTS = "facts"
    ONTOLOGIES = "ontologies"


class ContentUnit(BaseModel):
    """A chunk of text with associated metadata and RDF graph.

    Attributes:
        text: Text content of the chunk.
        index: Index of the chunk of the document.
        hid: An almost unique (hash) id for the chunk.
        doc_iri: IRI of parent document.
        graph: RDF triples representing the facts from the current document.
        processed: Whether chunk has been processed.
        type: Type of content unit (facts or ontology).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str = Field(description="Text of the chunk")
    index: int = Field(description="Index of the chunk of the document")
    hid: str = Field(description="An almost unique (hash) id for the chunk")
    doc_iri: URIRef = Field(description="IRI of parent doc")
    graph: RDFGraph = Field(
        description="RDF triples representing the facts from a document chunk in turtle format "
        "as a string in compact form: use prefixes for namespaces, do NOT add comments",
        default_factory=RDFGraph,
    )

    _graph_absolute: RDFGraph | None = PrivateAttr(default=None)

    processed: bool = Field(default=False, description="Was the chunk processed?")
    generated_at: datetime | None = Field(
        default=None, description="generated timestamp"
    )

    type: OutputType = Field(
        default=OutputType.FACTS, description="Type of content unit"
    )

    @field_validator("doc_iri", mode="before")
    @classmethod
    def _coerce_doc_iri(cls, value: URIRef | str) -> URIRef:
        if isinstance(value, URIRef):
            return value
        return URIRef(value)

    @property
    def graph_absolute(self):
        if self._graph_absolute is None:
            self._graph_absolute = self.graph.copy()
            self._graph_absolute.remap_namespaces(self.iri, self.iri_absolute)
        return self._graph_absolute

    @property
    def iri(self):
        """Get the IRI for this chunk.

        Returns:
            str: The chunk IRI.
        """
        return DEFAULT_IRI

    @property
    def iri_absolute(self):
        """Get the absolute IRI for this chunk.

        Returns:
            str: The chunk IRI.
        """
        return f"{self.doc_iri}/{self.hid}"

    @property
    def generated_at_iso(self):
        """Get the IRI for this chunk.

        Returns:
            str: The chunk IRI.
        """
        if self.generated_at is None:
            self.generated_at = datetime.now(timezone.utc)
        return self.generated_at.isoformat()

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

    def __len__(self):
        return len(self.text)
