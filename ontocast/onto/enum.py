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
    """Enumeration of routing decisions after ontology quality checks."""

    TEXT_TO_FACTS = "adequate ontology; render facts"
    TEXT_TO_ONTOLOGY = "inadequate ontology; retry render onto"
    SERIALIZE = "skip to serialize"


class SectionLabelSource(StrEnum):
    """How a chunk's ``section_label`` was decided.

    Ordered from strongest to weakest evidence. The source is not bookkeeping:
    the chunk-prepare cascade uses it to decide whether a label may still be
    overwritten by a later tier, and forward-fill refuses to cross a span whose
    source is :attr:`OUTLINE_UNRESOLVED`.
    """

    HEADING_PATTERN = "heading_pattern"
    HEADING_KEYWORD = "heading_keyword"
    HEADING_INHERITED = "heading_inherited"
    FRONT_MATTER = "front_matter"
    SPAN_OVERLAP = "span_overlap"
    CONTENT_DENSITY = "content_density"
    LLM = "llm"
    FORWARD_FILL = "forward_fill"
    OUTLINE_UNRESOLVED = "outline_unresolved"


class RenderMode(StrEnum):
    """Enumeration of supported rendering modes."""

    ONTOLOGY = "ontology"
    FACTS = "facts"
    ONTOLOGY_AND_FACTS = "ontology_and_facts"


class LLMGraphFormat(StrEnum):
    """Format used by the LLM when emitting RDF graph payloads.

    - ``turtle``: graph fields are Turtle strings (legacy behavior).
    - ``jsonld``: graph fields are compact JSON-LD objects embedded directly
      in the structured LLM response. Internally parsed back into ``RDFGraph``.
    """

    TURTLE = "turtle"
    JSONLD = "jsonld"


class OntologyContextMode(StrEnum):
    """How per-unit ontology context is sourced before ontology/facts rendering."""

    SELECTED_SINGLE_ONTOLOGY = "selected_single_ontology"
    SELECTED_VECTOR_SEARCH_ONTOLOGY = "selected_vector_search_ontology"
    FIXED_SINGLE_ONTOLOGY = "fixed_single_ontology"


class OntologyAssemblyMode(StrEnum):
    """How per-unit ontology context was assembled for prompts."""

    SELECTED_SINGLE_ONTOLOGY_LLM = "selected_single_ontology_llm"
    SELECTED_VECTOR_SEARCH_ENSEMBLE = "selected_vector_search_ensemble"
    FIXED_SINGLE_ONTOLOGY = "fixed_single_ontology"
    DOCUMENT_MERGED_REDUCED = "document_merged_reduced"


class FailureStage(StrEnum):
    """Enumeration of possible failure stages in the workflow."""

    NO_CHUNKS_TO_PROCESS = "No chunks to process"
    ONTOLOGY_CRITIQUE = "The produced ontology did not pass the critique stage."
    FACTS_CRITIQUE = "The produced graph of facts did not pass the critique stage."
    GENERATE_TTL_FOR_ONTOLOGY = (
        "Failed to generate semantic triples (turtle) for ontology"
    )
    GENERATE_GRAPH_UPDATE_FOR_ONTOLOGY = "Failed to generate graph update for ontology"
    GENERATE_TTL_FOR_FACTS = "Failed to generate semantic triples (turtle) for facts"
    GENERATE_GRAPH_UPDATE_FOR_FACTS = "Failed to generate graph update for facts"


class WorkflowNode(StrEnum):
    """Enumeration of workflow nodes in the processing pipeline."""

    CONVERT_TO_TEXT = "Convert to Text"
    CHUNK = "Chunk Text"
    SUMMARIZE_CHUNKS = "Summarize Chunks"
    TEXT_TO_ONTOLOGY = "Text to Ontology"
    TEXT_TO_FACTS = "Text to Facts"
    CRITICISE_ONTOLOGY = "Criticise Ontology"
    CRITICISE_FACTS = "Criticise Facts"
    AGGREGATE_FACTS = "Aggregate Facts"
    SERIALIZE = "Serialize"
    PARALLEL_MAP_UNITS = "Parallel Map Units"
    RENDER_ONTOLOGY_UPDATE = "Update Ontology"
    RENDER_FACTS = "Render Facts"
    NORMALIZE_ONTOLOGY_UPDATES = "Normalize Ontology Updates"
    CONSOLIDATE_ONTOLOGY = "Consolidate Ontology"
    MERGE_FACTS = "Merge Facts"
    VALIDATE_FACTS = "Validate Facts"
    PLAN_EXTERNAL_EVIDENCE = "Plan External Evidence"
    FETCH_EXTERNAL_EVIDENCE = "Fetch External Evidence"
    STRUCTURAL_CHECK = "Structural Check"
    CONSISTENCY_CRITIC = "Consistency Critic"


class SPARQLOperationType(StrEnum):
    """Enumeration of SPARQL operation types.

    This enum is used across the system for type-safe SPARQL operations.
    """

    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"


class VectorStoreBackend(StrEnum):
    """Which vector store implementation backs ontology patch retrieval.

    ``AUTO`` infers the backend from whichever connection setting is populated
    -- Qdrant if ``QDRANT_URI`` is set, LanceDB if it is enabled, otherwise the
    dependency-free in-memory store. Naming a backend explicitly makes the
    choice fail loudly when its configuration is missing.
    """

    AUTO = "auto"
    MEMORY = "memory"
    QDRANT = "qdrant"
    LANCEDB = "lancedb"
    NONE = "none"


class VectorDistance(StrEnum):
    """Vector distance metric used when creating a vector collection.

    Values match ``qdrant_client.http.models.Distance`` exactly, so existing
    ``QDRANT_DISTANCE`` environment values keep working. Declaring it here
    rather than importing Qdrant's enum keeps the Qdrant SDK off the import
    path of :mod:`ontocast.config`, which every entry point loads.
    """

    COSINE = "Cosine"
    DOT = "Dot"
    EUCLID = "Euclid"
    MANHATTAN = "Manhattan"
