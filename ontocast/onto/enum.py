from enum import StrEnum


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
    FAILURE_NO_ONTOLOGY = "ontology not found; ffwd to END"
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
