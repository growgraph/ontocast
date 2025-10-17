import pathlib

from pydantic import BaseModel, Field

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.prompt.common import (
    suggestion_concrete_template,
    suggestion_general_template,
)


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
    ontology_relevance_score: float | None = Field(
        description=(
            "Score between 0 and 100 of how well "
            "the ontology represents the domain of the document."
        )
    )
    triples_generation_score: float | None = Field(
        description=(
            "Score 0-100 for how well the semantic triples "
            "represent the document. 0 is the worst, 100 is the best."
        )
    )


class Suggestions(BaseModel):
    """Report from knowledge graph critique process.

    Attributes:
        systemic_critique_summary: A compilation of general improvement suggestions.
        actionable_fixes: An itemized list of concrete suggestions for improvement.
    """

    actionable_fixes: list[str] = Field(
        default_factory=list,
        description="An itemized list of concrete suggestions for improvement.",
    )

    systemic_critique_summary: str = Field(
        default="", description="A general improvement suggestion."
    )

    @classmethod
    def from_critique_report(
        cls, critique: "OntologyCritiqueReport | FactsCritiqueReport"
    ) -> "Suggestions":
        """Create Suggestions from any critique report.

        Args:
            critique: Either an OntologyCritiqueReport or FactsCritiqueReport to convert.

        Returns:
            Suggestions object with actionable fixes and systemic critique summary.
        """
        # Extract actionable fixes based on the type of critique report
        if isinstance(critique, OntologyCritiqueReport):
            actionable_fixes = critique.actionable_ontology_fixes
        elif isinstance(critique, FactsCritiqueReport):
            actionable_fixes = critique.actionable_triple_fixes
        else:
            raise ValueError(f"Unsupported critique report type: {type(critique)}")

        return cls(
            actionable_fixes=actionable_fixes,
            systemic_critique_summary=critique.systemic_critique_summary,
        )

    def to_prompt_templates(self) -> str:
        """Generate prompt templates from the suggestions.

        Returns:
            Combined string with general and concrete templates.
            Returns empty string if both fields are empty.
        """

        # Generate general template if systemic_critique_summary is not empty
        general_template = ""
        if self.systemic_critique_summary.strip():
            general_template = suggestion_general_template.format(
                general_suggestion=self.systemic_critique_summary
            )

        # Generate concrete template if actionable_fixes is not empty
        concrete_template = ""
        if self.actionable_fixes:
            suggestion_str = "\n- ".join(self.actionable_fixes)
            concrete_template = suggestion_concrete_template.format(
                suggestion_str=suggestion_str
            )

        # Combine both templates
        return general_template + concrete_template


class OntologyCritiqueReport(BaseModel):
    """Report from ontology update critique process."""

    success: bool = Field(
        description="True if the presented ontology is appropriate, complete, consistent and represents well the domain of the provided text, False otherwise."
    )
    score: float = Field(
        description="Score 0-100 for how well the presented ontology serves as the ontology for the document. 0 is the worst, 100 is the best."
    )

    actionable_ontology_fixes: list[str] = Field(
        default_factory=list,
        description="An itemized list of concrete, actionable suggestions for specific improvements to the ontology. Each suggestion must cite the specific text context that necessitates the change, addressing issues like Completeness or Abstraction.",
    )

    systemic_critique_summary: str = Field(
        default="",
        description="A high-level summary of systemic deficiencies in the ontology (e.g., poor hierarchy structure, redundant concepts, lack of appropriate granularity, or general failures in Domain Coverage). This addresses strategic issues beyond individual term fixes.",
    )


class FactsCritiqueReport(BaseModel):
    success: bool = Field(
        description="True if the facts triples fully represent the document, False otherwise. "
    )
    score: float = Field(
        description="Score 0-100 for how well the triples of facts "
        "represent the original document. 0 is the worst, 100 is the best."
    )
    actionable_triple_fixes: list[str] = Field(
        default_factory=list,
        description="An itemized list of specific, actionable suggestions detailing how to correct or improve the facts graph. Each entry must follow the output instructions: providing the text justification, and the INCORRECT/CORRECT triples (ADD/REMOVE/MODIFY) where applicable.",
    )

    systemic_critique_summary: str = Field(
        default="",
        description="A high-level summary of systemic or pattern-based issues (e.g., consistent failure to extract date literals, structural over-reliance on cd: individuals, or a common misinterpretation of a specific ontology property). This is for identifying strategic improvements to the fact-extraction process, not individual triple fixes.",
    )
