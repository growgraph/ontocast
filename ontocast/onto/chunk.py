from pydantic import BaseModel, Field

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.util import iri2namespace


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
    graph: RDFGraph = Field(
        description="RDF triples representing the facts from a document chunk in turtle format "
        "as a string in compact form: use prefixes for namespaces, do NOT add comments",
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
