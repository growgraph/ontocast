from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict, Field, field_validator
from rdflib import URIRef

from ontocast.onto.constants import DEFAULT_DOMAIN
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import (
    FailureStage,
    LLMGraphFormat,
    OntologyContextMode,
    RenderMode,
    Status,
)
from ontocast.onto.iri_policy import normalize_namespace_iri
from ontocast.onto.model import (
    BasePydanticModel,
    FactsLoopAttempt,
    FactsValidationFinding,
    GraphRepairRecord,
    UnitFailure,
)
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import GraphUpdate, TripleOp
from ontocast.onto.token_usage import TokenUsage
from ontocast.util.hash import render_text_hash
from ontocast.util.optional import require

if TYPE_CHECKING:
    from docling_core.types.doc import DoclingDocument

# Top-level SPARQL update keywords at line start (used to split compound LLM output).
_TOP_LEVEL_UPDATE_START_RE = re.compile(r"(?m)^(?=(?:INSERT|DELETE|WITH)\b)")


def _docling_document_cls() -> Any:
    """Resolve ``DoclingDocument`` on demand.

    ``docling-core`` ships in the ``documents`` extra: it drags pandas, pyarrow
    and transformers, none of which the light core needs. Resolving the class
    lazily keeps ``AgentState`` importable without it, at the cost of typing
    :attr:`AgentState.docling_doc` as ``Any`` -- pydantic resolves field
    annotations when the class is created, so a ``TYPE_CHECKING``-only import
    would make the model itself unbuildable.
    """
    return require(
        "docling_core.types.doc", feature="Parsed Docling documents"
    ).DoclingDocument


#: Suffix marking a duration key as *summed across parallel unit workers* rather
#: than wall clock. See :class:`BudgetTracker` for the full key convention.
UNIT_SUM_SUFFIX = "/unit_sum"

#: Suffix marking a duration key as a peak rather than an accumulation. Summing
#: two peaks is meaningless, so :meth:`BudgetTracker.merge_from` takes the max
#: for these keys instead.
PEAK_SUFFIX = "_max"


class BudgetTracker(BasePydanticModel):
    """Lightweight tracker for LLM usage statistics and generated triples.

    ``node_durations`` follows a key convention that distinguishes wall clock
    from time summed across concurrent workers -- without it, a fan-out node's
    entry is ambiguous and the two are silently added together:

    ``"<node>"``
        True wall clock for a pipeline node. Written only by the ``_timed``
        wrapper in :mod:`ontocast.stategraph.create`.
    ``"<node>/unit_sum"``
        Per-unit loop time summed over every parallel worker. Divided by the
        wall-clock entry this yields *effective workers* -- see
        :meth:`parallel_efficiency`.
    ``"<node>/<stage>"``
        Any other sub-stage measurement (``worker_wait``, ``loop_lag_total``,
        ``llm/provider``, ...).
    """

    chars_sent: int = Field(default=0, description="Total characters sent to LLM")
    chars_received: int = Field(
        default=0, description="Total characters received from LLM"
    )
    calls_count: int = Field(default=0, description="Total number of LLM API calls")
    cache_hits: int = Field(
        default=0,
        description="LLM calls satisfied from disk cache (no provider tokens)",
    )
    input_tokens: int = Field(
        default=0, description="Billed input tokens (when reported by provider)"
    )
    output_tokens: int = Field(
        default=0, description="Billed output tokens (when reported by provider)"
    )

    # Kept apart from the billed totals rather than folded in: a cache-replayed
    # run costs nothing, so adding these to input_tokens/output_tokens would
    # report spend that never happened. Reported together they answer the other
    # question -- what the workload costs cold -- which is what the replay
    # protocol in docs/user_guide/performance.md is measuring.
    cached_input_tokens: int = Field(
        default=0, description="Input tokens replayed from the OntoCast disk cache"
    )
    cached_output_tokens: int = Field(
        default=0, description="Output tokens replayed from the OntoCast disk cache"
    )

    # Detail keys from LangChain's UsageMetadata, summed over billed and
    # replayed calls alike: they describe the shape of the workload, and a
    # reasoning model's thinking tokens matter whether or not this particular
    # run paid for them.
    reasoning_tokens: int = Field(
        default=0, description="Thinking tokens, counted inside the output totals"
    )
    cache_read_input_tokens: int = Field(
        default=0,
        description=(
            "Input tokens served from the provider's own prompt cache, counted "
            "inside the input totals and billed at a reduced rate"
        ),
    )
    cache_creation_input_tokens: int = Field(
        default=0, description="Input tokens written to the provider's prompt cache"
    )

    # Triple generation tracking
    ontology_triples_generated: int = Field(
        default=0, description="Total number of triples generated for ontology updates"
    )
    facts_triples_generated: int = Field(
        default=0, description="Total number of triples generated for facts"
    )
    ontology_operations_count: int = Field(
        default=0, description="Total number of ontology update operations"
    )
    facts_operations_count: int = Field(
        default=0, description="Total number of facts update operations"
    )

    node_durations: dict[str, float] = Field(
        default_factory=dict,
        description="Accumulated seconds per pipeline node/stage (see class docstring)",
    )
    counters: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Named event counts (e.g. how often a per-document computation ran). "
            "Summed on merge, like node_durations."
        ),
    )

    def add_duration(self, name: str, seconds: float) -> None:
        """Accumulate seconds for a named node or stage.

        Keys ending in :data:`PEAK_SUFFIX` take the maximum instead, since
        adding two peaks would report a stall that never happened.
        """
        if name.endswith(PEAK_SUFFIX):
            self.node_durations[name] = max(self.node_durations.get(name, 0.0), seconds)
            return
        self.node_durations[name] = self.node_durations.get(name, 0.0) + seconds

    def incr(self, name: str, n: int = 1) -> None:
        """Increment a named counter.

        Args:
            name: Counter key, e.g. ``"ctx/merge_document_ontology.calls"``.
            n: Amount to add.
        """
        self.counters[name] = self.counters.get(name, 0) + n

    def parallel_efficiency(self, node: str) -> float | None:
        """Effective worker count for a fan-out node, or ``None`` if unmeasured.

        The ratio of time summed across unit workers to the node's wall clock.
        A value near ``parallel_workers`` means the fan-out is running at full
        width; a value near ``1.0`` means the units are effectively serialised
        -- typically by synchronous CPU work blocking the event loop, which
        ``"<node>/loop_lag_total"`` quantifies.

        Args:
            node: The node key, e.g. ``str(WorkflowNode.RENDER_FACTS)``.

        Returns:
            float | None: Effective workers, or ``None`` when either the wall
            clock or the ``/unit_sum`` entry is missing or zero.
        """
        wall = self.node_durations.get(node)
        unit_sum = self.node_durations.get(f"{node}{UNIT_SUM_SUFFIX}")
        if not wall or unit_sum is None:
            return None
        return unit_sum / wall

    def _add_usage_detail(self, usage: TokenUsage) -> None:
        """Accumulate the provider-detail keys shared by billed and cached calls."""
        if usage.reasoning_tokens is not None:
            self.reasoning_tokens += usage.reasoning_tokens
        if usage.cache_read_input_tokens is not None:
            self.cache_read_input_tokens += usage.cache_read_input_tokens
        if usage.cache_creation_input_tokens is not None:
            self.cache_creation_input_tokens += usage.cache_creation_input_tokens

    def add_usage(
        self,
        chars_sent: int,
        chars_received: int,
        *,
        usage: TokenUsage | None = None,
    ) -> None:
        """Record a billed provider call.

        Args:
            chars_sent: Prompt length in characters.
            chars_received: Response length in characters.
            usage: Token counts, when the provider reported any.
        """
        self.chars_sent += chars_sent
        self.chars_received += chars_received
        self.calls_count += 1
        if usage is None:
            return
        if usage.input_tokens is not None:
            self.input_tokens += usage.input_tokens
        if usage.output_tokens is not None:
            self.output_tokens += usage.output_tokens
        self._add_usage_detail(usage)

    def add_cache_hit(
        self,
        chars_sent: int,
        chars_received: int,
        *,
        usage: TokenUsage | None = None,
    ) -> None:
        """Record a disk-cache hit (does not increment calls_count).

        Args:
            chars_sent: Prompt length in characters.
            chars_received: Cached response length in characters.
            usage: Token counts stored with the cache entry. ``None`` for
                entries written before usage was persisted, which report as
                unknown rather than as zero.
        """
        self.cache_hits += 1
        self.chars_sent += chars_sent
        self.chars_received += chars_received
        if usage is None:
            return
        if usage.input_tokens is not None:
            self.cached_input_tokens += usage.input_tokens
        if usage.output_tokens is not None:
            self.cached_output_tokens += usage.output_tokens
        self._add_usage_detail(usage)

    def add_ontology_update(self, num_operations: int, num_triples: int) -> None:
        """Add ontology update statistics.

        Args:
            num_operations: Number of update operations generated
            num_triples: Number of triples in these operations
        """
        self.ontology_operations_count += num_operations
        self.ontology_triples_generated += num_triples

    def add_facts_update(self, num_operations: int, num_triples: int) -> None:
        """Add facts update statistics.

        Args:
            num_operations: Number of update operations generated
            num_triples: Number of triples in these operations
        """
        self.facts_operations_count += num_operations
        self.facts_triples_generated += num_triples

    def merge_from(self, other: BudgetTracker) -> None:
        """Accumulate counters from another tracker (e.g. parallel unit workers)."""
        self.chars_sent += other.chars_sent
        self.chars_received += other.chars_received
        self.calls_count += other.calls_count
        self.cache_hits += other.cache_hits
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.cached_output_tokens += other.cached_output_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        self.ontology_triples_generated += other.ontology_triples_generated
        self.facts_triples_generated += other.facts_triples_generated
        self.ontology_operations_count += other.ontology_operations_count
        self.facts_operations_count += other.facts_operations_count
        for name, seconds in other.node_durations.items():
            self.add_duration(name, seconds)
        for name, count in other.counters.items():
            self.incr(name, count)

    def get_summary(self) -> str:
        """Get a summary of LLM usage and generated triples."""
        parts = [
            f"LLM: {self.calls_count} calls, "
            f"{self.chars_sent:,} sent, "
            f"{self.chars_received:,} received",
        ]
        if self.cache_hits > 0:
            parts.append(f"{self.cache_hits:,} cache hits")

        if self.input_tokens > 0 or self.output_tokens > 0:
            parts.append(
                f"{self.input_tokens:,} in / {self.output_tokens:,} out tokens"
            )

        # Reported separately so a replayed run reads as "free this time, but
        # this is what it costs", rather than as no token usage at all.
        if self.cached_input_tokens > 0 or self.cached_output_tokens > 0:
            parts.append(
                f"{self.cached_input_tokens:,} in / "
                f"{self.cached_output_tokens:,} out tokens replayed"
            )

        detail = [
            f"{value:,} {label}"
            for label, value in (
                ("reasoning", self.reasoning_tokens),
                ("provider-cache read", self.cache_read_input_tokens),
                ("provider-cache write", self.cache_creation_input_tokens),
            )
            if value > 0
        ]
        if detail:
            parts.append("of which " + ", ".join(detail))

        if self.ontology_triples_generated > 0 or self.facts_triples_generated > 0:
            parts.append(
                f"Triples: {self.ontology_triples_generated} ontology, "
                f"{self.facts_triples_generated} facts"
            )

        return " | ".join(parts)

    def get_duration_summary(self) -> str:
        """Get per-node wall-clock durations, slowest first."""
        if not self.node_durations:
            return ""
        ranked = sorted(
            self.node_durations.items(), key=lambda item: item[1], reverse=True
        )
        return "Durations: " + ", ".join(
            f"{name} {seconds:.1f}s" for name, seconds in ranked
        )

    def get_parallelism_summary(self) -> str:
        """Effective worker count and event-loop stall per fan-out node.

        Reports only nodes that recorded a ``/unit_sum`` entry, so it is empty
        for pipelines without a fan-out. ``lag`` is the time the event loop was
        blocked by synchronous work while units were meant to be running
        concurrently -- it is the difference between the width configured and
        the width achieved.
        """
        parts: list[str] = []
        for key in sorted(self.node_durations):
            if not key.endswith(UNIT_SUM_SUFFIX):
                continue
            node = key[: -len(UNIT_SUM_SUFFIX)]
            effective = self.parallel_efficiency(node)
            if effective is None:
                continue
            fragment = f"{node} {effective:.1f}x"
            lag = self.node_durations.get(f"{node}/loop_lag_total")
            if lag:
                fragment += f" (loop lag {lag:.1f}s)"
            parts.append(fragment)
        if not parts:
            return ""
        return "Effective workers: " + ", ".join(parts)


class AgentState(BasePydanticModel):
    """State for the ontology-based knowledge graph agent.

    This class maintains the state of the agent during document processing,
    including input text, content units, ontologies, and workflow status.

    Attributes:
        docling_doc: Parsed document in native Docling format.
        current_domain: IRI used for forming document namespace.
        doc_hid: An almost unique hash/id for the parent document.
        raw_input: Single raw input payload as {filename: bytes}.
        failure_stage: Stage where failure occurred.
        failure_reason: Reason for failure.
        status: Current workflow status.
        max_visits: Maximum render attempts per unit loop.
        max_chunks: Maximum number of source content units to split and process.
    """

    # Typed `Any` rather than `DoclingDocument | None` so that importing
    # AgentState does not require the `documents` extra; the validator below
    # still enforces the real type whenever docling-core is installed.
    docling_doc: Any = Field(
        default=None,
        description="Parsed document in native Docling format.",
    )
    current_domain: str = Field(
        description=(
            "IRI used for forming the document namespace. Defaults to the "
            "CURRENT_DOMAIN environment variable, then to DEFAULT_DOMAIN; an "
            "explicit constructor argument always wins."
        ),
        default_factory=lambda: os.getenv("CURRENT_DOMAIN") or DEFAULT_DOMAIN,
    )
    doc_hid: str = Field(
        description="An almost unique hash / id for the parent document of the current unit",
        default="default_doc",
    )
    raw_input: dict[str, bytes] = Field(
        default_factory=dict,
        description="Single raw input payload: {filename: bytes}.",
    )
    content_units: list[ContentUnit] = Field(
        default_factory=list,
        description="Pending content units to process.",
    )
    ontology_artifacts: list[Ontology] = Field(
        default_factory=list,
        description="Final per-anchor ontology artifacts produced for this document.",
    )
    reduced_ontology_artifacts: list[Ontology] = Field(
        default_factory=list,
        description="Reduced ontology artifacts after explicit ontology reduce step.",
    )
    reduced_ontology_by_anchor: dict[str, Ontology] = Field(
        default_factory=dict,
        description="Reduced ontology artifacts indexed by anchor IRI.",
    )
    ontology_reduce_metrics: dict[str, int | float | str] = Field(
        default_factory=dict,
        description="Metrics emitted by ontology reduce stage.",
    )
    unit_patch_sources: dict[int, list[str]] = Field(
        default_factory=dict,
        description="Retrieved ontology source IRIs per content unit index.",
    )
    retrieval_metrics: dict[str, int | float | str | dict[str, Any]] = Field(
        default_factory=dict,
        description="Runtime retrieval/evaluation metrics for observability.",
    )
    aggregated_facts: RDFGraph = Field(
        description="RDF triples representing aggregated facts "
        "from the current document",
        default_factory=RDFGraph,
    )
    facts_ontology_context: RDFGraph = Field(
        default_factory=RDFGraph,
        description=(
            "Merged reduced-ontology graph used as read-only schema for facts. "
            "Derived from reduced_ontology_artifacts, which are frozen once the "
            "ontology stage completes, so the facts fan-out builds it once and "
            "merge/validate reuse it rather than each repeating the merge."
        ),
    )
    ontology_user_instruction: str = Field(
        description="Specific user instructions for ontology extraction, e.g. `Focus on extracting places`",
        default="",
    )

    ontology_selection_user_instruction: str = Field(
        description=(
            "Specific user instructions for ontology selection, "
            "e.g. `Prefer ontologies focused on finance`"
        ),
        default="",
    )

    facts_user_instruction: str = Field(
        description="Specific user instructions for facts extraction, e.g. `Focus on extracting places`",
        default="",
    )

    ontology_context_fixed_ontology_id: str = Field(
        description=(
            "Catalog ontology id when ontology_context_mode is fixed_single_ontology "
            "(resolved via OntologyManager)."
        ),
        default="",
    )

    tenant: str | None = Field(
        default=None,
        description="Tenant id when request selected tenancy via query/CLI.",
    )
    project: str | None = Field(
        default=None,
        description="Project id when request selected tenancy via query/CLI.",
    )

    source_url: str | None = Field(
        description="Source URL from JSON input file (for provenance tracking)",
        default=None,
    )

    document_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Caller-asserted document identity metadata attached to the parent "
            "doc_iri as provenance (DOI, ISBN, scheme+value business ids, title, …)."
        ),
    )

    ontology_updates_applied: list[GraphUpdate] = Field(
        default_factory=list,
        description="A list of graph update that improve the current ontology",
    )

    facts_units: list[ContentUnit] = Field(
        default_factory=list,
        description="Successful per-unit facts outputs collected during parallel map phase",
    )

    facts_loop_telemetry: dict[int, list[FactsLoopAttempt]] = Field(
        default_factory=dict,
        description=(
            "Per-unit facts loop attempt records (render/critic/repair) keyed "
            "by content unit index; makes visit efficacy measurable."
        ),
    )

    facts_repairs_applied: dict[int, list[GraphRepairRecord]] = Field(
        default_factory=dict,
        description=(
            "Deterministic rewrites applied to rendered facts graphs, keyed by "
            "content unit index — records which triples the machine altered "
            "from what the LLM asserted."
        ),
    )

    unit_failures: list[UnitFailure] = Field(
        default_factory=list,
        description=(
            "Content units that produced no usable output, with the stage and "
            "reason. Without this, a run in which every unit failed was "
            "indistinguishable from one that found nothing to extract."
        ),
    )

    aggregation_clusters: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Final URI -> source entities merged into it during facts "
            "aggregation (clusters with >= 2 members only); consumed by the "
            "validation gate to turn findings into un-merge pair vetoes."
        ),
    )

    facts_validation_findings: list[FactsValidationFinding] = Field(
        default_factory=list,
        description=(
            "Post-aggregation invariant findings (functional violations, "
            "suspect multi-values, degenerate coreference, SHACL) remaining "
            "after the un-merge and SHACL-autofix repair stages."
        ),
    )

    facts_gate_repairs: list[GraphRepairRecord] = Field(
        default_factory=list,
        description=(
            "LLM-free repairs the validation gate applied to the merged graph "
            "(shape-driven retyping, code resolution, placeholder pruning). "
            "Document-level counterpart of facts_repairs_applied, which is "
            "per unit."
        ),
    )

    facts_conformance: dict = Field(
        default_factory=dict,
        description=(
            "Rolled-up validation result: whether SHACL ran and the graph "
            "conforms, plus counts by finding kind, SHACL constraint component "
            "and shape. Grouping is what makes the residue diagnosable — a "
            "flat violation list does not distinguish one systematic modelling "
            "gap from many independent defects."
        ),
    )

    ontology_units: list[ContentUnit] = Field(
        default_factory=list,
        description="Successful per-unit ontology outputs collected during parallel map phase",
    )
    ontology_provenance_artifact: RDFGraph = Field(
        default_factory=RDFGraph,
        description="Provenance/reification triples stripped from normalized ontology.",
    )

    failure_stage: FailureStage | None = None
    failure_reason: str | None = None

    improvements_suggestions: list[str] = Field(
        description="Itemized concrete and actionable instructions for improvements of extraction of facts/ontology",
        default_factory=list,
    )

    status: Status = Status.SUCCESS
    max_visits: int = Field(
        default=1,
        description=(
            "Maximum render attempts per unit loop. Mirrors "
            "``ServerConfig.max_visits_per_node``, which every entry path "
            "supplies; at 1 the critic never runs."
        ),
    )
    max_chunks: int | None = None
    target_sections: list[str] | None = Field(
        default=None,
        description="Sections to include when chunking. None = no filter.",
    )
    exclude_sections: list[str] | None = Field(
        default=None,
        description=(
            "Sections to drop when chunking. None = use the resolved section "
            "schema's default_exclude; [] = no exclusion; list = explicit "
            "denylist."
        ),
    )
    summarize_sections: list[str] | None = Field(
        default=None,
        description="Sections to summarize. None = skip summarization node.",
    )
    summary_max_sentences: int = Field(
        default=5,
        description="Max sentences per chunk summary when summarization is enabled.",
    )
    document_type_hint: str | None = Field(
        default=None,
        description=(
            "Optional free-text hint about the source material (e.g. '10-K filing', "
            "'journal article') used to resolve section label schema and LLM tagging."
        ),
    )
    section_schema_id: str | None = Field(
        default=None,
        description=(
            "Section label schema id from ontocast.config.section_labels (e.g. academic, "
            "financial). Overrides document_type_hint when set."
        ),
    )
    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)
    render_mode: RenderMode = Field(
        default=RenderMode.ONTOLOGY_AND_FACTS,
        description=("Rendering mode: ontology, facts, or ontology_and_facts."),
    )
    llm_graph_format: LLMGraphFormat = Field(
        default=LLMGraphFormat.JSONLD,
        description=(
            "Format used by the LLM for emitting RDF graph payloads: "
            "'jsonld' (default; compact JSON-LD objects embedded directly in the "
            "structured response) or 'turtle' (legacy Turtle strings)."
        ),
    )
    ontology_context_mode: OntologyContextMode = Field(
        default=OntologyContextMode.SELECTED_SINGLE_ONTOLOGY,
        description=(
            "Per-unit ontology context: selected_single_ontology (LLM-picked catalog), "
            "selected_vector_search_ontology (vector-store ensemble; Qdrant or LanceDB), "
            "or fixed_single_ontology (catalog ontology_id via ontology_context_fixed_ontology_id)."
        ),
    )
    # Budget Tracking
    budget_tracker: BudgetTracker = Field(
        default_factory=BudgetTracker,
        description="Budget statistics tracker (LLM usage and generated triples)",
    )

    @property
    def needs_section_prepare(self) -> bool:
        """Whether the request carries explicit section-dependent options.

        Section tagging itself is default-on in chunk prepare (driven by
        ``CHUNK_SECTION_CLASSIFIER``); schema-default exclusions apply even
        when this is False.
        """
        return (
            self.target_sections is not None
            or self.summarize_sections is not None
            or self.exclude_sections is not None
        )

    @property
    def use_summarization(self) -> bool:
        """Whether per-unit summaries should be produced in the fan-out."""
        return self.summarize_sections is not None

    @property
    def render_ontology(self) -> bool:
        """Whether ontology rendering should run."""
        return self.render_mode in (
            RenderMode.ONTOLOGY,
            RenderMode.ONTOLOGY_AND_FACTS,
        )

    @property
    def render_facts(self) -> bool:
        """Whether facts rendering should run."""
        return self.render_mode in (
            RenderMode.FACTS,
            RenderMode.ONTOLOGY_AND_FACTS,
        )

    def get_content_unit_progress_info(self) -> tuple[int, int]:
        """Get current content unit number and total content units."""
        total_content_units = len(self.content_units)
        current_content_unit_number = 1 if total_content_units > 0 else 0
        return current_content_unit_number, total_content_units

    def get_content_unit_progress_string(self) -> str:
        """Get a formatted string showing content unit progress."""
        current, total = self.get_content_unit_progress_info()
        if total == 0:
            return "no content units"
        return f"content unit {current}/{total}"

    @classmethod
    def render_updated_graph(
        cls, graph: RDFGraph, updates: list[GraphUpdate], max_triples: int | None = None
    ) -> tuple[RDFGraph, bool]:
        """Create a copy of the given graph with all GraphUpdate objects applied.

        This method:
        1. Creates a copy of the input graph
        2. Generates SPARQL queries from all GraphUpdate objects
        3. Executes the queries on the copied graph
        4. Checks if the updated graph exceeds max_triples limit
        5. Returns the updated graph copy, or original if limit exceeded

        Args:
            graph: The RDFGraph to update
            updates: List of GraphUpdate objects to apply
            max_triples: Maximum number of triples allowed. If None, no limit enforced.

        Returns:
            Tuple of (RDFGraph, bool): The updated graph (or original if limit exceeded),
            and a boolean indicating if the update was applied (True) or skipped (False)
        """
        if not updates:
            return graph, True

        # Create a copy of the input graph
        # Use RDFGraph's copy method to preserve type
        updated_graph = RDFGraph()
        for triple in graph:
            updated_graph.add(triple)
        # Copy namespace bindings
        for prefix, namespace in graph.namespaces():
            updated_graph.bind(prefix, namespace)

        all_prefixes = {}
        for graph_update in updates:
            for op in graph_update.triple_operations:
                # Extract prefixes from TripleOp operations
                if isinstance(op, TripleOp) and op.prefixes:
                    all_prefixes.update(op.prefixes)

        # Bind prefixes to the copied graph
        for prefix, uri in all_prefixes.items():
            updated_graph.bind(prefix, uri)

        # Apply each GraphUpdate to the copied graph
        for graph_update in updates:
            # Generate SPARQL queries from the GraphUpdate
            queries = graph_update.generate_sparql_queries()

            # Execute each query on the copied graph
            for query in queries:
                cls._apply_update_query(updated_graph, query)

        # Check if updated graph exceeds max_triples limit
        if max_triples is not None and len(updated_graph) > max_triples:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"Ontology update skipped: would exceed limit "
                f"({len(updated_graph)} > {max_triples} triples). "
                f"Original size: {len(graph)} triples."
            )
            return graph, False  # Return original, unchanged

        return updated_graph, True

    @classmethod
    def _apply_update_query(cls, graph: RDFGraph, query: str) -> None:
        """Apply one SPARQL update query, splitting compound LLM output proactively."""
        parts = cls._split_compound_sparql_query(query)
        for part in parts:
            graph.update(part)

    @staticmethod
    def _split_compound_sparql_query(query: str) -> list[str]:
        """Split a query string containing concatenated top-level UPDATE statements.

        LLMs frequently emit several ``INSERT DATA`` / ``DELETE DATA`` blocks joined
        after a shared ``PREFIX`` block.  Splitting on top-level keyword boundaries
        before calling ``graph.update`` avoids parse errors entirely.

        A single-statement query is returned as a one-element list.
        """
        stripped = query.strip()
        if not stripped:
            return [stripped]

        starts = [m.start() for m in _TOP_LEVEL_UPDATE_START_RE.finditer(stripped)]
        if len(starts) <= 1:
            return [stripped]

        prefix_block = stripped[: starts[0]].strip()
        parts: list[str] = []
        for i, start in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else len(stripped)
            body = stripped[start:end].strip()
            if body:
                parts.append(f"{prefix_block}\n{body}" if prefix_block else body)
        return parts or [stripped]

    def set_docling_doc(self, doc: "DoclingDocument") -> None:
        """Set the parsed document and generate document hash.

        Args:
            doc: The DoclingDocument to set.
        """
        self.docling_doc = doc
        self.doc_hid = render_text_hash(doc.model_dump_json())

    @field_validator("docling_doc", mode="before")
    @classmethod
    def _coerce_docling_doc(cls, value: object) -> Any:
        if value is None:
            return None
        docling_document = _docling_document_cls()
        if isinstance(value, docling_document):
            return value
        if isinstance(value, dict):
            return docling_document.model_validate(value)
        raise TypeError(f"Expected DoclingDocument or dict, got {type(value).__name__}")

    def set_failure(self, stage: FailureStage, reason: str) -> None:
        """Set failure state with stage and reason.

        Args:
            stage: The stage where the failure occurred.
            reason: The reason for the failure.
        """
        self.failure_stage = stage
        self.failure_reason = reason
        self.status = Status.FAILED

    def clear_failure(self):
        """Clear failure state and set status to success."""
        self.failure_stage = None
        self.failure_reason = None
        self.status = Status.SUCCESS

    @property
    def doc_iri(self) -> URIRef:
        """Get the document IRI.

        Returns:
            str: The document IRI.
        """
        return URIRef(f"{self.current_domain}/doc/{self.doc_hid}")

    @property
    def doc_namespace(self):
        """Get the document namespace.

        Returns:
            str: The document namespace.
        """
        return normalize_namespace_iri(self.doc_iri, context="facts")

    @property
    def graph_uri(self):
        return self.doc_namespace

    @property
    def ontology_ids(self) -> list[str]:
        """Ontology ids for all current ontology artifacts."""
        artifacts = (
            self.reduced_ontology_artifacts
            if self.reduced_ontology_artifacts
            else self.ontology_artifacts
        )
        return [ontology.ontology_id for ontology in artifacts if ontology.ontology_id]
