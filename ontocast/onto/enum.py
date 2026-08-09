from enum import StrEnum


class Status(StrEnum):
    """Enumeration of possible workflow status values."""

    NOT_VISITED = "not visited"
    SUCCESS = "success"
    FAILED = "failed"
    COUNTS_EXCEEDED = "counts exceeded"


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


class RetrievalMetric(StrEnum):
    """Top-level keys of ``AgentState.retrieval_metrics``.

    These are wire names. The dict is serialized verbatim into
    ``ProcessResultMetadata.retrieval_metrics`` on ``/process`` and
    ``/process_unit`` and into the batch run manifest, so a member's *value*
    may never change without a breaking release; the member name is free to.
    Collecting them here replaces bare string literals scattered over three
    modules, where a typo produced a silently missing metric and nothing
    enumerated what a run should emit.

    Only the flat top level is enumerated. The per-retrieval telemetry that
    lands nested under :attr:`PATCH_RETRIEVAL` is the patch retriever's own
    namespace with its own lifecycle, and flattening it here would assert a
    structure that does not exist.
    """

    # Ontology context assembly (written per unit, merged onto the document).
    ONTOLOGY_CONTEXT_MODE = "ontology_context_mode"
    PATCH_RETRIEVAL = "patch_retrieval"
    #: Why a unit's ontology snapshot came back empty. Written per unit and
    #: merged last-writer-wins, so on a multi-unit document only the final
    #: unit's reason survives.
    EMPTY_SNAPSHOT_REASON = "empty_snapshot_reason"
    ONTOLOGY_WRITABLE_COUNT = "ontology_writable_count"
    ONTOLOGY_PRIMARY_UNITS = "ontology_primary_units"

    # Facts fan-out.
    FACTS_ANCHOR_COUNT = "facts_anchor_count"
    FACTS_ANCHOR_UNITS = "facts_anchor_units"
    FACTS_LLM_REPAIR_RENDERS_TOTAL = "facts_llm_repair_renders_total"
    FACTS_LLM_REPAIR_RENDERS_FAILED = "facts_llm_repair_renders_failed"
    FACTS_FINDINGS_RESIDUAL = "facts_findings_residual"

    # Aggregation and the un-merge repair.
    FACTS_REJECTED_MERGES = "facts_rejected_merges"
    FACTS_MERGE_REPAIR_PASSES = "facts_merge_repair_passes"
    FACTS_MERGE_VETOES = "facts_merge_vetoes"
    FACTS_MERGE_REPAIRS_REJECTED = "facts_merge_repairs_rejected"

    # Validation gate. Written identically by both entry paths.
    VALIDATED_WITHOUT_ONTOLOGY_CONTEXT = "validated_without_ontology_context"
    FACTS_VALIDATION_FINDINGS = "facts_validation_findings"
    FACTS_VALIDATION_ERRORS = "facts_validation_errors"
    FACTS_SHACL_VIOLATIONS_BEFORE = "facts_shacl_violations_before"
    FACTS_SHACL_VIOLATIONS_AFTER = "facts_shacl_violations_after"
    FACTS_SHACL_REPAIRS = "facts_shacl_repairs"
    FACTS_SHACL_AUTOFIX_PASSES = "facts_shacl_autofix_passes"
    FACTS_SHACL_AUTOFIX_REVERTED = "facts_shacl_autofix_reverted"

    # Post-aggregation checks.
    STRUCTURAL_ONTOLOGY_COMPONENTS_MAX = "structural_ontology_components_max"
    CONSISTENCY_CONFLICTS = "consistency_conflicts"


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
    TEXT_TO_ONTOLOGY = "Text to Ontology"
    TEXT_TO_FACTS = "Text to Facts"
    CRITICISE_ONTOLOGY = "Criticise Ontology"
    CRITICISE_FACTS = "Criticise Facts"
    SERIALIZE = "Serialize"
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

    Two backends are supported: ``QDRANT`` (server) and ``LANCEDB`` (embedded),
    each shipped as its own optional extra.

    ``AUTO`` infers the backend from whichever connection setting is populated
    -- Qdrant if ``QDRANT_URI`` is set, LanceDB if it is enabled, otherwise
    ``NONE``. ``NONE`` is the default for an unconfigured install: ontology
    context then comes from a single working ontology, which is the default
    :class:`OntologyContextMode`. Naming a backend explicitly makes the choice
    fail loudly when its configuration is missing.
    """

    AUTO = "auto"
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
