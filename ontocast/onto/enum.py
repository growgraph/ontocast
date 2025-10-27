from enum import StrEnum


class Status(StrEnum):
    """Enumeration of possible workflow status values."""

    NOT_VISITED = "not visited"
    SUCCESS = "success"
    FAILED = "failed"
    COUNTS_EXCEEDED = "counts exceeded"


class OntologyDecision(StrEnum):
    """Enumeration of Ontology Decisions used in the workflow."""

    SKIP_TO_FACTS = "ontology found; skip to facts"
    FAILURE_NO_ONTOLOGY = "ontology not found; ffwd to END"
    IMPROVE_CREATE_ONTOLOGY = "improve/create ontology"


class FactsDecision(StrEnum):
    """Enumeration of Ontology Decisions used in the workflow."""

    TEXT_TO_FACTS = "adequate ontology; render facts"
    TEXT_TO_ONTOLOGY = "inadequate ontology; retry render onto"
    SERIALIZE = "skip to serialize"


class FailureStage(StrEnum):
    """Enumeration of possible failure stages in the workflow."""

    NO_CHUNKS_TO_PROCESS = "No chunks to process"
    ONTOLOGY_CRITIQUE = "The produced ontology did not pass the critique stage."
    FACTS_CRITIQUE = "The produced graph of facts did not pass the critique stage."
    GENERATE_TTL_FOR_ONTOLOGY = (
        "Failed to generate semantic triples (turtle) for ontology"
    )
    GENERATE_SPARQL_UPDATE_FOR_ONTOLOGY = (
        "Failed to generate SPARQL update for ontology"
    )
    GENERATE_TTL_FOR_FACTS = "Failed to generate semantic triples (turtle) for facts"
    GENERATE_SPARQL_UPDATE_FOR_FACTS = "Failed to generate SPARQL update for ontology"
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
    SERIALIZE = "Serialize"


class SPARQLOperationType(StrEnum):
    """Enumeration of SPARQL operation types.

    This enum is used across the system for type-safe SPARQL operations.
    """

    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
