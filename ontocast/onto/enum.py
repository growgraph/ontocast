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
    #: Triples in the resolved ontology snapshot, written by every context mode.
    #: Before this existed only the vector resolver recorded a size, nested under
    #: :attr:`PATCH_RETRIEVAL`, so the two modes that bound nothing were also the
    #: two that reported nothing.
    ONTOLOGY_SNAPSHOT_TRIPLES = "ontology_snapshot_triples"

    # Ontology fan-out: the per-unit critic ledger and the deterministic
    # findings residual, mirrors of the facts block below. Added before any
    # gate change so the sampling corpus records the incumbent gate's own
    # accept rate -- the facts gate was replaced on numbers (28/34 rejected,
    # median score 79) and the ontology gate gets the same treatment.
    ONTOLOGY_FINDINGS_RESIDUAL = "ontology_findings_residual"
    ONTOLOGY_MANDATORY_RESIDUAL = "ontology_mandatory_residual"
    ONTOLOGY_CRITIC_CALLS = "ontology_critic_calls"
    ONTOLOGY_CRITIC_ACCEPTED = "ontology_critic_accepted"

    # Facts fan-out.
    FACTS_ANCHOR_COUNT = "facts_anchor_count"
    FACTS_ANCHOR_UNITS = "facts_anchor_units"
    FACTS_LLM_REPAIR_RENDERS_TOTAL = "facts_llm_repair_renders_total"
    FACTS_LLM_REPAIR_RENDERS_FAILED = "facts_llm_repair_renders_failed"
    FACTS_REPAIR_DELETE_ONLY = "facts_repair_delete_only"
    FACTS_FINDINGS_RESIDUAL = "facts_findings_residual"
    FACTS_MANDATORY_RESIDUAL = "facts_mandatory_residual"
    FACTS_CRITIC_CALLS = "facts_critic_calls"
    FACTS_CRITIC_FIXES_APPLIED = "facts_critic_fixes_applied"
    FACTS_CRITIC_FIXES_RESIDUAL = "facts_critic_fixes_residual"
    FACTS_CRITIC_FIXES_NOOP = "facts_critic_fixes_noop"
    FACTS_CRITIC_PATCHES_ROLLED_BACK = "facts_critic_patches_rolled_back"
    ONTOLOGY_CRITIC_FIXES_APPLIED = "ontology_critic_fixes_applied"
    ONTOLOGY_CRITIC_FIXES_RESIDUAL = "ontology_critic_fixes_residual"
    ONTOLOGY_CRITIC_FIXES_NOOP = "ontology_critic_fixes_noop"
    ONTOLOGY_CRITIC_PATCHES_ROLLED_BACK = "ontology_critic_patches_rolled_back"
    FACTS_CRITIC_ACCEPTED = "facts_critic_accepted"
    #: Units whose critic call failed (timeout, unparseable response) and
    #: left the loop unreviewed, and units the loop did not send to the
    #: critic at all (empty render, citation metadata).
    FACTS_CRITIC_UNITS_UNREVIEWED = "facts_critic_units_unreviewed"
    FACTS_CRITIC_UNITS_SKIPPED = "facts_critic_units_skipped"
    #: Per-fix outcomes of the compiled critique: fixes undone for leaving
    #: the unit worse, inserts refused for minting a placeholder or an
    #: annotation-only node, and payloads naming a prefix nothing declared.
    FACTS_CRITIC_FIXES_ROLLED_BACK = "facts_critic_fixes_rolled_back"
    FACTS_CRITIC_FIXES_JUNK_REFUSED = "facts_critic_fixes_junk_refused"
    FACTS_CRITIC_FIXES_UNRESOLVED_PREFIX = "facts_critic_fixes_unresolved_prefix"
    #: The insert-only completion pass: calls billed, triples that stayed
    #: in, and missed measurements the inventory no longer lists afterwards.
    FACTS_COMPLETION_CALLS = "facts_completion_calls"
    FACTS_COMPLETION_TRIPLES_INSERTED = "facts_completion_triples_inserted"
    FACTS_COMPLETION_MEASUREMENTS_RECOVERED = "facts_completion_measurements_recovered"

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

    - ``jsonld`` (default): graph fields are compact JSON-LD objects embedded
      directly in the structured LLM response. Internally parsed back into
      ``RDFGraph``.
    - ``turtle``: graph fields are Turtle strings (legacy encoding, kept for
      providers whose structured output handles strings better than objects).
    """

    TURTLE = "turtle"
    JSONLD = "jsonld"


class OntologyChapterFormat(StrEnum):
    """Syntax of the ``# ONTOLOGY`` chapter in the facts prompts.

    - ``inherit`` (default): the chapter follows :class:`LLMGraphFormat`, so
      the model reads the ontology in the syntax it is asked to write.
    - ``turtle``: the chapter is Turtle whatever the wire format is. In a
      facts prompt the ontology is read-only context -- nothing the model
      emits has to match its syntax -- and Turtle spends fewer characters per
      triple than pretty-printed JSON-LD, so this trades the read/write
      symmetry for a shorter prompt. The graph payloads the model emits stay
      in the wire format.
    - ``term_sheet``: the chapter is a line-per-term listing rather than a
      serialized graph -- name, surface forms, type, hierarchy, domain/range
      and usage contract, without the per-statement RDF scaffolding or the
      prose written for a human reader. Legal on the facts path only: the
      ontology loop emits a patch against the statements it reads, so its
      chapter has to remain a graph.
    """

    INHERIT = "inherit"
    TURTLE = "turtle"
    TERM_SHEET = "term_sheet"


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
