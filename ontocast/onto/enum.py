from enum import StrEnum


class Status(StrEnum):
    """Enumeration of possible workflow status values."""

    NOT_VISITED = "not visited"
    SUCCESS = "success"
    FAILED = "failed"
    COUNTS_EXCEEDED = "counts exceeded"


class SectionLabelSource(StrEnum):
    """Which tier gave a piece of section-label evidence. Values are wire names.

    :attr:`OUTLINE_UNRESOLVED` marks a section the outline saw a heading for
    but could not name: no tier spreads a neighbour's label into it.
    """

    HEADING_PATTERN = "heading_pattern"
    HEADING_KEYWORD = "heading_keyword"
    HEADING_INHERITED = "heading_inherited"
    #: A heading no pattern or keyword named, nearest a label by embedding.
    HEADING_EMBEDDING = "heading_embedding"
    FRONT_MATTER = "front_matter"
    CONTENT_DENSITY = "content_density"
    #: A citation run cut out of a unit by the block classifier.
    CONTENT_BLOCK = "content_block"
    LLM = "llm"
    OUTLINE_UNRESOLVED = "outline_unresolved"


class LabelStrength(StrEnum):
    """Whether a section label may take part in a drop decision. Wire names."""

    STRONG = "strong"
    WEAK = "weak"


class UnitRoute(StrEnum):
    """What happens to a unit before extraction. Values are wire names."""

    EXTRACT = "extract"
    #: Kept; the facts renderer extracts bibliographic metadata only.
    CITATIONS = "citations"
    #: Kept and flagged ``is_non_content``; checks that presume domain prose stand down.
    NON_CONTENT = "non_content"
    SKIP = "skip"


class RouteReason(StrEnum):
    """Why a unit took its route. Values are wire names (run manifest, logs)."""

    CONTENT = "content"
    #: A label that could have dropped the unit was weak.
    WEAK_LABEL = "weak_label"
    #: A strong citations or boilerplate label the content disagrees with.
    LABEL_CONTRADICTED = "label_contradicted"
    #: A boilerplate label under a schema nobody chose: no default exclusions.
    FALLBACK_SCHEMA = "fallback_schema"
    #: A boilerplate label the request's exclude_sections did not name.
    EXCLUSION_OVERRIDDEN = "exclusion_overridden"
    #: The content alone is a reference list.
    CITATION_LIST = "citation_list"
    #: A strong citations-class label and content that agrees.
    CITATION_LABEL = "citation_label"
    #: A strong boilerplate-class label on text that states no measurement.
    BOILERPLATE = "boilerplate"
    #: Front/back matter recognised from the content.
    NON_CONTENT = "non_content"
    #: The content alone is a table of contents or a list of figures or tables.
    CONTENTS_LIST = "contents_list"
    EXCLUDED_BY_REQUEST = "excluded_by_request"
    UNDERSIZED = "undersized"


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
    # findings residual, mirrors of the facts block below, so the ontology
    # gate's accept rate can be read from a run's artifacts.
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
    #: Units the ontology loop did not send to the critic: at one pass, a unit
    #: with no mandatory deterministic finding.
    ONTOLOGY_CRITIC_UNITS_SKIPPED = "ontology_critic_units_skipped"
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
    #: The insert-only completion pass: calls made, triples that stayed
    #: in, and missed measurements the inventory no longer lists afterwards.
    FACTS_COMPLETION_CALLS = "facts_completion_calls"
    FACTS_COMPLETION_TRIPLES_INSERTED = "facts_completion_triples_inserted"
    FACTS_COMPLETION_MEASUREMENTS_RECOVERED = "facts_completion_measurements_recovered"

    # Aggregation and the un-merge repair.
    FACTS_REJECTED_MERGES = "facts_rejected_merges"
    #: The refused merges by reason (same_unit, sibling, literal_conflict,
    #: functional_iri_conflict, initials_conflict, role, type, lexical,
    #: cluster_veto), a pair counting under each reason that refused it.
    FACTS_MERGE_REJECTIONS_BY_REASON = "facts_merge_rejections_by_reason"
    FACTS_MERGE_REPAIR_PASSES = "facts_merge_repair_passes"
    FACTS_MERGE_VETOES = "facts_merge_vetoes"
    FACTS_MERGE_REPAIRS_REJECTED = "facts_merge_repairs_rejected"
    #: Coreference resolution: pairs put to the model, and its verdicts.
    FACTS_COREFERENCE_PAIRS = "facts_coreference_pairs"
    FACTS_COREFERENCE_SAME = "facts_coreference_same"
    FACTS_COREFERENCE_DIFFERENT = "facts_coreference_different"
    #: Link pass: instances isolated after the gate, the links the model
    #: proposed for them, those added, and the rejected ones by reason.
    FACTS_LINK_ISOLATED = "facts_link_isolated"
    FACTS_LINK_PROPOSED = "facts_link_proposed"
    FACTS_LINK_ADDED = "facts_link_added"
    FACTS_LINK_REJECTIONS_BY_REASON = "facts_link_rejections_by_reason"

    # Validation gate. Written identically by both entry paths.
    VALIDATED_WITHOUT_ONTOLOGY_CONTEXT = "validated_without_ontology_context"
    FACTS_VALIDATION_FINDINGS = "facts_validation_findings"
    FACTS_VALIDATION_ERRORS = "facts_validation_errors"
    FACTS_SHACL_VIOLATIONS_BEFORE = "facts_shacl_violations_before"
    FACTS_SHACL_VIOLATIONS_AFTER = "facts_shacl_violations_after"
    FACTS_SHACL_REPAIRS = "facts_shacl_repairs"
    FACTS_SHACL_AUTOFIX_PASSES = "facts_shacl_autofix_passes"
    FACTS_SHACL_AUTOFIX_REVERTED = "facts_shacl_autofix_reverted"
    #: Fact-namespace instances dropped for carrying no concrete type after
    #: aggregation; the IRIs are on ``AgentState.facts_untyped_dropped``.
    FACTS_UNTYPED_DROPPED = "facts_untyped_dropped"
    #: Instances untyped after aggregation that the gate typed from the
    #: domains and ranges of their links (against the whole ontology context,
    #: which a unit's context may have lacked), instead of dropping them.
    FACTS_UNTYPED_INFERRED = "facts_untyped_inferred"
    #: Instances dropped for stating nothing: linked to no other instance nor
    #: the document, and holding no value, only a type and names.
    FACTS_UNLINKED_DROPPED = "facts_unlinked_dropped"
    #: Links between instances the gate's repairs removed (SHACL prune and
    #: the untyped drop), counted after any un-merge pass.
    FACTS_LINKS_REMOVED = "facts_links_removed"
    #: Connected groups of instances in the served facts, the document node
    #: joining what it links, and instances linked to nothing (those left hold
    #: a value of their own). Both describe the graph after the link pass.
    FACTS_COMPONENTS = "facts_components"
    FACTS_ISOLATED = "facts_isolated"
    #: Labels two or more served instances share: the duplicates aggregation
    #: left, or distinct things with one name.
    FACTS_TWIN_LABELS = "facts_twin_labels"

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

    - ``turtle`` (default): graph fields are Turtle strings. Fewer tokens per
      triple than JSON-LD, and an IRI object cannot be written as a string by
      accident: ``unit:NanoM`` is an IRI, ``"unit:NanoM"`` visibly a literal.
    - ``jsonld``: graph fields are compact JSON-LD objects embedded directly in
      the structured LLM response, for providers whose structured output
      handles long strings worse than nested objects. Internally parsed back
      into ``RDFGraph``.
    """

    TURTLE = "turtle"
    JSONLD = "jsonld"


class LLMOutputLayout(StrEnum):
    """Whitespace the LLM is asked to use in its structured responses.

    - ``compact`` (default): the JSON response is minified, and a Turtle graph
      string keeps each subject on one line without indentation. Indentation
      counts as output tokens and carries nothing the parser reads.
    - ``free``: no layout instruction; the model chooses, and typically
      pretty-prints JSON with indentation.
    """

    FREE = "free"
    COMPACT = "compact"


class OntologyChapterFormat(StrEnum):
    """Syntax of the ``# ONTOLOGY`` chapter in the facts prompts.

    - ``auto`` (default): ``term_sheet`` where the render mode is facts-only,
      ``inherit`` otherwise. The cheapest chapter each mode can legally read,
      chosen without asking an operator to know which those are. Resolved once
      when the configuration is built, so nothing downstream -- profile lookup,
      cache key, run manifest -- ever sees this member.
    - ``inherit``: the chapter follows :class:`LLMGraphFormat`, so
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

    AUTO = "auto"
    INHERIT = "inherit"
    TURTLE = "turtle"
    TERM_SHEET = "term_sheet"


class OntologyContextScope(StrEnum):
    """Whether the ontology chapter is resolved per unit or once per document.

    - ``unit`` (default): each content unit retrieves its own context. The
      smallest chapter per unit, and a different chapter for every one of them,
      so a provider's prefix cache can serve none of them.
    - ``document``: every unit's context is resolved once and unioned, and the
      union is shown to all of them. Larger per call and recall-safe by
      construction -- the union contains everything each unit's own retrieval
      selected -- but identical across the fan-out, which is what makes the
      chapter cacheable after the first call.
    """

    UNIT = "unit"
    DOCUMENT = "document"


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
    UNSUPPORTED_DOCUMENT = "The document type is not supported"
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
    RESOLVE_HOME_ONTOLOGY = "Resolve Home Ontology"
    ONTOLOGY_PLAN = "Plan Ontology"
    TEXT_TO_ONTOLOGY = "Text to Ontology"
    TEXT_TO_FACTS = "Text to Facts"
    CRITICISE_ONTOLOGY = "Criticise Ontology"
    CRITICISE_FACTS = "Criticise Facts"
    SERIALIZE = "Serialize"
    RENDER_ONTOLOGY_UPDATE = "Update Ontology"
    RENDER_FACTS = "Render Facts"
    NORMALIZE_ONTOLOGY_UPDATES = "Normalize Ontology Updates"
    MERGE_FACTS = "Merge Facts"
    RESOLVE_COREFERENCE = "Resolve Coreference"
    VALIDATE_FACTS = "Validate Facts"
    LINK_FACTS = "Link Facts"
    PLAN_EXTERNAL_EVIDENCE = "Plan External Evidence"
    FETCH_EXTERNAL_EVIDENCE = "Fetch External Evidence"
    STRUCTURAL_CHECK = "Structural Check"
    CONSISTENCY_CRITIC = "Consistency Critic"


class LLMPhase(StrEnum):
    """What an LLM call was for: the key of ``BudgetTracker.tokens_by_phase``.

    A phase covers every provider call its step makes, parse retries and
    re-renders included. ``UNLABELED`` collects calls made outside any labelled
    step; in a pipeline run it should stay empty.
    """

    FACTS_RENDER = "facts_render"
    FACTS_CRITIC = "facts_critic"
    FACTS_COMPLETION = "facts_completion"
    FACTS_COREFERENCE = "facts_coreference"
    FACTS_LINK = "facts_link"
    ONTOLOGY_IDENTITY = "ontology_identity"
    ONTOLOGY_PLAN = "ontology_plan"
    ONTOLOGY_RENDER = "ontology_render"
    ONTOLOGY_CRITIC = "ontology_critic"
    ONTOLOGY_CONSOLIDATION = "ontology_consolidation"
    ONTOLOGY_CURATION = "ontology_curation"
    ONTOLOGY_SELECTION = "ontology_selection"
    EVIDENCE_PLANNING = "evidence_planning"
    UNIT_SUMMARY = "unit_summary"
    SECTION_CLASSIFICATION = "section_classification"
    UNLABELED = "unlabeled"


class FactsLinkRejection(StrEnum):
    """Why the link pass refused a proposed link; the first failed check names it."""

    UNKNOWN_ENTITY = "unknown_entity"
    NO_ISOLATED_ENDPOINT = "no_isolated_endpoint"
    SELF_LINK = "self_link"
    UNKNOWN_PROPERTY = "unknown_property"
    NOT_OBJECT_PROPERTY = "not_object_property"
    DOMAIN_MISMATCH = "domain_mismatch"
    RANGE_MISMATCH = "range_mismatch"
    FUNCTIONAL_TAKEN = "functional_taken"
    EVIDENCE_MISSING = "evidence_missing"
    DUPLICATE = "duplicate"


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
