import pathlib
from typing import Optional

from pydantic import BaseModel, Field

from ontocast.onto.rdfgraph import RDFGraph


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
        description="Semantic triples (facts) representing the document "
        "in turtle format: use prefixes for namespaces, do NOT add comments",
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
