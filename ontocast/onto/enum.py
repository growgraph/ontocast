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


class RenderMode(StrEnum):
    """Enumeration of supported rendering modes."""

    ONTOLOGY = "ontology"
    FACTS = "facts"
    ONTOLOGY_AND_FACTS = "ontology_and_facts"


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
    TEXT_TO_ONTOLOGY = "Text to Ontology"
    TEXT_TO_FACTS = "Text to Facts"
    CRITICISE_ONTOLOGY = "Criticise Ontology"
    CRITICISE_FACTS = "Criticise Facts"
    AGGREGATE_FACTS = "Aggregate Facts"
    SERIALIZE = "Serialize"
    PARALLEL_MAP_UNITS = "Parallel Map Units"
    SELECT_ONTOLOGY = "Select Ontology"
    BOOTSTRAP_ONTOLOGY = "Bootstrap Ontology"
    RENDER_ONTOLOGY_UPDATE = "Update Ontology"
    RENDER_FACTS = "Render Facts"
    NORMALIZE_ONTOLOGY_UPDATES = "Normalize Ontology Updates"
    CONSOLIDATE_ONTOLOGY = "Consolidate Ontology"
    MERGE_FACTS = "Merge Facts"
    PLAN_EXTERNAL_EVIDENCE = "Plan External Evidence"
    FETCH_EXTERNAL_EVIDENCE = "Fetch External Evidence"


class SPARQLOperationType(StrEnum):
    """Enumeration of SPARQL operation types.

    This enum is used across the system for type-safe SPARQL operations.
    """

    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
