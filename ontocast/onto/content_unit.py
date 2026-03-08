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


class SourceUnit(BaseModel):
    """Immutable source unit identity and input text.

    Attributes:
        text: Source text content for this unit.
        index: Position of this unit in the source document.
        hid: An almost unique (hash) id for this unit.
        doc_iri: IRI of parent document.
        type: Type of content unit (facts or ontology).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str = Field(description="Source text content for this unit")
    index: int = Field(description="Position of this unit in the source document")
    hid: str = Field(description="An almost unique (hash) id for this unit")
    doc_iri: URIRef = Field(description="IRI of parent doc")
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
    def iri(self):
        """Get the base IRI for this unit.

        Returns:
            str: The base unit IRI.
        """
        return DEFAULT_IRI

    @property
    def iri_absolute(self):
        """Get the absolute IRI for this unit.

        Returns:
            str: The unit IRI.
        """
        return f"{self.doc_iri}/{self.hid}"

    @property
    def namespace(self):
        """Get the namespace for this unit.

        Returns:
            str: The unit namespace.
        """
        return iri2namespace(self.iri, ontology=False)

    def __len__(self):
        return len(self.text)


class ContentUnit(SourceUnit):
    """A processing unit that extends source data with mutable output fields."""

    graph: RDFGraph = Field(
        description="RDF triples representing facts rendered from this source unit in turtle format "
        "as a string in compact form: use prefixes for namespaces, do NOT add comments",
        default_factory=RDFGraph,
    )

    _graph_absolute: RDFGraph | None = PrivateAttr(default=None)

    processed: bool = Field(default=False, description="Was this unit processed?")
    generated_at: datetime | None = Field(
        default=None, description="generated timestamp"
    )

    @property
    def graph_absolute(self):
        if self._graph_absolute is None:
            self._graph_absolute = self.graph.copy()
            self._graph_absolute.remap_namespaces(self.iri, self.iri_absolute)
        return self._graph_absolute

    @property
    def generated_at_iso(self):
        """Get generated timestamp in ISO format.

        Returns:
            str: Timestamp in ISO format.
        """
        if self.generated_at is None:
            self.generated_at = datetime.now(timezone.utc)
        return self.generated_at.isoformat()

    def sanitize(self):
        self.graph = self.graph.unbind_chunk_namespaces()
        self.graph.sanitize_prefixes_namespaces()
