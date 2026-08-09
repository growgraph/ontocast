import asyncio
import logging
import pathlib
from io import BytesIO
from typing import TYPE_CHECKING

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from ontocast.config import Config
from ontocast.onto.constants import ONTOLOGY_NULL_IRI
from ontocast.onto.enum import OntologyContextMode
from ontocast.onto.ontology import Ontology, OntologyProperties
from ontocast.onto.ontology_access import document_ontology_access
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.onto.tenancy import TenancyScope
from ontocast.runtime import ToolBoxRuntime
from ontocast.tool import (
    AtomicToolBox,
    ChunkerTool,
    ConverterTool,
    EmbeddingBasedAggregator,
    FusekiTripleStoreManager,
    InMemoryTripleStoreManager,
)
from ontocast.tool.agg.entity_aligner import EntityAligner
from ontocast.tool.cache import Cacher
from ontocast.tool.llm import LLMTool
from ontocast.tool.ontology_manager import OntologyManager
from ontocast.tool.sparql import SPARQLTool
from ontocast.tool.triple_manager.core import TripleStoreManager
from ontocast.tool.vector_store import (
    EmbeddingTool,
    OntologyPatchRetriever,
    VectorStoreManager,
    create_vector_store_manager,
)
from ontocast.util.loop import require_no_running_loop

if TYPE_CHECKING:
    from ontocast.registry import ToolBoxRegistry

logger = logging.getLogger(__name__)


async def update_ontology_properties(o: Ontology, llm_tool: LLMTool):
    """Update ontology properties using LLM analysis, only if missing.

    This function uses the LLM tool to analyze and update the properties
    of a given ontology based on its graph content, but only if any key
    property is missing or empty.
    """
    # Only update if any key property is missing or empty
    if (o.title is None) or (o.ontology_id is None) or (o.description is None):
        props = await render_ontology_summary(o, llm_tool)
        o.set_properties(**props.model_dump())


async def update_ontology_manager(
    om: OntologyManager,
    llm_tool: LLMTool,
    *,
    max_concurrency: int | None = None,
):
    """Update properties for all ontologies in the manager.

    Ontologies that already have title, ontology_id, and description are skipped.
    Remaining LLM calls run concurrently up to ``max_concurrency`` (defaults to
    the LLM tool's ``llm_max_inflight``).

    Args:
        om: The ontology manager containing ontologies to update.
        llm_tool: The LLM tool instance for analysis.
        max_concurrency: Optional override for parallel LLM enrich calls.
    """
    import asyncio
    import time

    pending = [
        o
        for o in om.ontologies
        if (o.title is None) or (o.ontology_id is None) or (o.description is None)
    ]
    if not pending:
        return

    limit = max_concurrency
    if limit is None:
        limit = max(1, getattr(llm_tool.config, "llm_max_inflight", 1))
    semaphore = asyncio.Semaphore(max(1, limit))

    async def _one(ontology: Ontology) -> None:
        async with semaphore:
            await update_ontology_properties(ontology, llm_tool)

    started = time.perf_counter()
    await asyncio.gather(*[_one(o) for o in pending])
    logger.info(
        "Ontology property enrich finished for %d ontolog(ies) in %.2fs",
        len(pending),
        time.perf_counter() - started,
    )


class ToolBox:
    """A container class for all tools used in the ontology processing workflow.

    This class initializes and manages various tools needed for document processing,
    ontology management, and LLM interactions.

    Args:
        config: Configuration object containing all necessary settings.
    """

    @classmethod
    async def acreate(cls, config: Config) -> "ToolBox":
        """Construct a ToolBox from inside a running event loop.

        Equivalent to ``ToolBox(config)``, except that LLM provider setup is
        awaited rather than driven through :func:`asyncio.run` -- which is
        illegal in a loop and is what makes the plain constructor unusable from
        async code. Embedders should prefer this, and pair it with
        ``async with`` so backend connections are released:

        ```python
        async with await ToolBox.acreate(config) as tools:
            await tools.initialize()
        ```

        Args:
            config: Fully resolved configuration.

        Returns:
            A ready ToolBox. Call :meth:`initialize` to sync ontologies and
            prepare backend schema.
        """
        runtime = await ToolBoxRuntime.acreate(config)
        return cls(config, runtime=runtime)

    def __init__(
        self,
        config: Config,
        *,
        llm: LLMTool | None = None,
        runtime: "ToolBoxRuntime | None" = None,
    ):
        """Build a ToolBox bound to whatever partition ``config`` names.

        Args:
            config: Fully resolved configuration.
            llm: Pre-built LLM tool, used when no ``runtime`` is supplied.
                ``LLMTool.create`` otherwise runs, which cannot be called from
                inside a running event loop -- prefer :meth:`acreate` there.
            runtime: Shared tenancy-independent tools. Supplied by
                :class:`~ontocast.registry.ToolBoxRegistry` so scoped ToolBoxes
                do not each load an embedding model; built fresh when omitted.
        """
        # Store the config for later use
        self.config = config

        # Get tool configuration
        tool_config = config.get_tool_config()

        # Tools that do not vary by tenant live on the runtime, so a registry of
        # scoped ToolBoxes shares one LLM client, converter and embedding model.
        self.runtime = runtime or ToolBoxRuntime(config, llm=llm)

        # Create triple store manager: Fuseki when configured, otherwise in-memory.
        use_fuseki = tool_config.fuseki.uri and tool_config.fuseki.auth
        if use_fuseki and tool_config.fuseki.uri and tool_config.fuseki.auth:
            self.triple_store_manager: TripleStoreManager = FusekiTripleStoreManager(
                uri=tool_config.fuseki.uri,
                auth=tool_config.fuseki.auth,
                dataset=tool_config.fuseki.dataset,
                ontologies_dataset=tool_config.fuseki.ontologies_dataset,
            )
        else:
            self.triple_store_manager = InMemoryTripleStoreManager()

        self.ontology_manager: OntologyManager = OntologyManager()
        self.ontology_manager.register_triple_store(self.triple_store_manager)
        # Tenancy the in-memory catalog currently reflects; None until first set.
        self._active_tenancy: tuple[str, str] | None = None
        # Guards the tenancy retarget, which mutates ToolBox-wide state (dataset
        # names, the ontology catalog, vector-store table names) across awaits.
        # It is driven by a per-request query parameter with no concurrency cap,
        # so without this two requests for different tenants can interleave and
        # read or write each other's partition. Created lazily: __init__ may run
        # outside an event loop.
        self._tenancy_lock: asyncio.Lock | None = None
        self._tenancy_lock_loop: asyncio.AbstractEventLoop | None = None
        # Set by attach_registry() when this ToolBox fronts a multi-tenant host.
        self._registry: "ToolBoxRegistry | None" = None

        # Graph algorithms over graphs it is handed; it does not fetch.
        self.sparql_tool: SPARQLTool = SPARQLTool(
            triple_store_manager=self.triple_store_manager
        )

        self.vector_store: VectorStoreManager | None = None
        self.patch_retriever: OntologyPatchRetriever | None = None
        self.vector_store_ready: bool = False
        self.vector_store_last_error: Exception | None = None

        # The factory owns backend selection, including resolving AUTO and
        # returning None when the backend is explicitly disabled. Both
        # supported backends need the BM25 tool, so the only case that skips it
        # is having no vector store at all.
        needs_sparse = bool(tool_config.qdrant.uri or tool_config.lancedb.enabled)
        vector_store = create_vector_store_manager(
            tool_config,
            embedding=self.embedding_tool,
            sparse_embedding=(
                self.runtime.sparse_embedding_tool(tool_config.embedding)
                if needs_sparse
                else None
            ),
        )
        if vector_store is not None:
            self.vector_store = vector_store
            self.patch_retriever = OntologyPatchRetriever(
                vector_store=vector_store,
                sparql_tool=self.sparql_tool,
                patch=tool_config.patch_retrieval,
                ontology_manager=self.ontology_manager,
            )
            self.ontology_manager.register_vector_store(self.patch_retriever)

    # -- shared runtime delegates -----------------------------------------
    #
    # These tools do not vary by tenant and live on the runtime, but they were
    # ToolBox attributes for the whole life of the project and are read -- and
    # substituted -- that way across the pipeline, the CLI, the worker and the
    # tests. Each delegates in both directions, so a caller replacing
    # `tools.converter` replaces the shared one, which is what it always meant.

    @property
    def runtime(self) -> ToolBoxRuntime:
        """Shared tenancy-independent tools.

        Materialized empty on first access when it was never assigned. Several
        unit tests build a ToolBox with ``ToolBox.__new__(ToolBox)`` to exercise
        one tool without standing up the whole container, and used to set that
        tool as a plain attribute; this keeps that working, and reading a tool
        that was never set still raises ``AttributeError`` for its own name.
        """
        runtime = self.__dict__.get("_runtime")
        if runtime is None:
            runtime = ToolBoxRuntime.__new__(ToolBoxRuntime)
            self.__dict__["_runtime"] = runtime
        return runtime

    @runtime.setter
    def runtime(self, value: ToolBoxRuntime) -> None:
        self.__dict__["_runtime"] = value

    @property
    def shared_cache(self) -> Cacher:
        """Shared on-disk cache backing the LLM and converter tools."""
        return self.runtime.shared_cache

    @shared_cache.setter
    def shared_cache(self, value: Cacher) -> None:
        self.runtime.shared_cache = value

    @property
    def llm(self) -> LLMTool:
        """Shared LLM tool."""
        return self.runtime.llm

    @llm.setter
    def llm(self, value: LLMTool) -> None:
        self.runtime.llm = value

    @property
    def llm_provider(self):
        """Configured LLM provider."""
        return self.runtime.llm_provider

    @llm_provider.setter
    def llm_provider(self, value) -> None:
        self.runtime.llm_provider = value

    @property
    def search_provider(self):
        """Configured web-search provider, or None when disabled."""
        return self.runtime.search_provider

    @search_provider.setter
    def search_provider(self, value) -> None:
        self.runtime.search_provider = value

    @property
    def atomic_tools(self) -> AtomicToolBox:
        """Per-unit tool surface used by the render/critic loops."""
        return self.runtime.atomic_tools

    @atomic_tools.setter
    def atomic_tools(self, value: AtomicToolBox) -> None:
        self.runtime.atomic_tools = value

    @property
    def converter(self) -> ConverterTool:
        """Document converter."""
        return self.runtime.converter

    @converter.setter
    def converter(self, value: ConverterTool) -> None:
        self.runtime.converter = value

    @property
    def chunker(self) -> ChunkerTool:
        """Text chunker."""
        return self.runtime.chunker

    @chunker.setter
    def chunker(self, value: ChunkerTool) -> None:
        self.runtime.chunker = value

    @property
    def aggregator(self) -> EmbeddingBasedAggregator:
        """Facts aggregator."""
        return self.runtime.aggregator

    @aggregator.setter
    def aggregator(self, value: EmbeddingBasedAggregator) -> None:
        self.runtime.aggregator = value

    @property
    def embedding_tool(self) -> EmbeddingTool:
        """Dense embedding provider."""
        return self.runtime.embedding_tool

    @embedding_tool.setter
    def embedding_tool(self, value: EmbeddingTool) -> None:
        self.runtime.embedding_tool = value

    def get_entity_aligner(
        self,
        embedding_model: str | None = None,
        similarity_threshold: float | None = None,
    ) -> EntityAligner:
        """Return a cached entity aligner for the given embedding settings."""
        tool_config = self.config.get_tool_config()
        return self.runtime.get_entity_aligner(
            embedding_model or tool_config.aggregation.embedding_model,
            similarity_threshold
            if similarity_threshold is not None
            else tool_config.aggregation.similarity_threshold,
        )

    async def get_llm_tool(self, budget_tracker):
        """Return the shared LLM tool, charging usage to ``budget_tracker``.

        Args:
            budget_tracker: The budget tracker to charge for this task's calls.

        Returns:
            LLMTool: The shared LLM tool.
        """
        return await self.runtime.get_llm_tool(budget_tracker)

    def require_triple_store_manager(self) -> TripleStoreManager:
        """Return the configured triple store manager or raise a clear error."""
        manager = self.triple_store_manager
        if manager is None:
            raise RuntimeError("Triple store backend is not configured")
        return manager

    def require_vector_store(self) -> VectorStoreManager:
        """Return the configured vector store or raise a directive error."""
        if self.vector_store is None:
            raise RuntimeError(
                "No vector store is configured. Set QDRANT_URI (Qdrant server) "
                "or LANCEDB_ENABLED=true (embedded LanceDB); each needs its "
                "matching extra, ontocast[qdrant] or ontocast[lancedb]."
            )
        return self.vector_store

    def require_patch_retriever(self) -> OntologyPatchRetriever:
        """Return the ontology patch retriever or raise a directive error."""
        if self.patch_retriever is None:
            raise RuntimeError(
                "Ontology patch retrieval needs a vector store. Set QDRANT_URI "
                "or LANCEDB_ENABLED=true."
            )
        return self.patch_retriever

    async def aclose(self) -> None:
        """Release every backend connection this ToolBox opened.

        The ToolBox owns an httpx client (Fuseki) and a Qdrant client, neither
        of which was previously closed anywhere -- ``FusekiTripleStoreManager``
        even defined ``close()`` that nothing called. Long-lived hosts that
        build a ToolBox per tenant, and tests that build many, leaked sockets.

        Also closes any scoped ToolBoxes this one spawned through
        :meth:`for_scope`, so shutting down the ToolBox an application holds
        releases every tenant's connections too.

        Safe to call more than once, and never raises: teardown failures are
        logged, since a caller shutting down cannot act on them.
        """
        registry = self._registry
        if registry is not None:
            self._registry = None
            try:
                await registry.aclose()
            except Exception as exc:
                logger.warning("Error closing tenancy registry: %s", exc)

        if self.triple_store_manager is not None:
            try:
                await self.triple_store_manager.close()
            except Exception as exc:
                logger.warning("Error closing triple store manager: %s", exc)

        if self.vector_store is not None:
            try:
                await asyncio.to_thread(self.vector_store.close)
            except Exception as exc:
                logger.warning("Error closing vector store: %s", exc)

    async def __aenter__(self) -> "ToolBox":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    @property
    def scope(self) -> TenancyScope | None:
        """The partition this ToolBox is bound to, once tenancy is assigned."""
        if self._active_tenancy is None:
            return None
        return TenancyScope.build(*self._active_tenancy)

    def attach_registry(self, registry: "ToolBoxRegistry") -> None:
        """Resolve other partitions through ``registry`` rather than a private one.

        Only needed to share one registry across several ToolBoxes;
        :meth:`for_scope` builds its own on first use otherwise.
        """
        self._registry = registry

    def ensure_tenancy_registry(self) -> "ToolBoxRegistry":
        """Return this ToolBox's registry, creating it on first use.

        Built lazily so a single-tenant embedder never allocates one, and so
        nothing has to be wired at construction time.
        """
        if self._registry is None:
            from ontocast.registry import ToolBoxRegistry

            self._registry = ToolBoxRegistry(
                self.config,
                self.runtime,
                max_scopes=self.config.server.max_tenancy_scopes,
            )
        return self._registry

    async def for_scope(
        self,
        tenant: str,
        project: str,
        *,
        ontology_context_mode: OntologyContextMode | None = None,
        fail_on_vector_store_error: bool = False,
    ) -> "ToolBox":
        """Return a ToolBox bound to ``tenant`` / ``project``.

        Returns ``self`` when the scope already matches. Otherwise resolves
        through the attached registry, which shares this ToolBox's runtime, so
        the new scope costs a triple store and an ontology catalog rather than
        another embedding model.

        Isolation is by construction: each scope owns a deep copy of ``Config``.
        That copy matters -- vector store managers hold their config sections by
        reference and rewrite collection names when tenancy is applied, so
        scopes sharing a ``Config`` would alias each other.

        Args:
            tenant: Tenant identifier.
            project: Project identifier within the tenant.
            ontology_context_mode: Mode to initialize a newly built scope for.
            fail_on_vector_store_error: Raise rather than log when vector store
                preparation fails for a newly built scope.

        Returns:
            A ToolBox bound to the requested partition.
        """
        requested = TenancyScope.build(tenant, project)
        if self._active_tenancy == requested.key:
            return self
        return await self.ensure_tenancy_registry().get(
            requested,
            ontology_context_mode=ontology_context_mode,
            fail_on_vector_store_error=fail_on_vector_store_error,
        )

    async def update_tenancy(self, tenant: str, project: str) -> None:
        """Retarget Fuseki datasets and Qdrant collections for ``tenant`` / ``project``."""
        await self.update_tenancy_with_vector_mode(
            tenant,
            project,
            initialize_vector_store=True,
            fail_on_vector_store_error=True,
        )

    def _get_tenancy_lock(self) -> asyncio.Lock:
        """Return the tenancy lock bound to the running loop.

        Rebuilt when the loop changes: the CLI bootstrap runs several
        ``asyncio.run`` calls, and an ``asyncio.Lock`` created on a closed loop
        cannot be awaited on a later one.
        """
        loop = asyncio.get_running_loop()
        if self._tenancy_lock is None or self._tenancy_lock_loop is not loop:
            self._tenancy_lock = asyncio.Lock()
            self._tenancy_lock_loop = loop
        return self._tenancy_lock

    async def update_tenancy_with_vector_mode(
        self,
        tenant: str,
        project: str,
        *,
        initialize_vector_store: bool,
        fail_on_vector_store_error: bool,
    ) -> None:
        """Retarget tenancy and optionally initialize vector store collections.

        Serialized: the body mutates ToolBox-wide state across awaits, and the
        HTTP layer calls this per request from a ``?tenant=`` query parameter
        with no concurrency cap. Interleaving two switches leaves the catalog
        and the store handles describing different tenants.
        """
        async with self._get_tenancy_lock():
            await self._update_tenancy_with_vector_mode_locked(
                tenant,
                project,
                initialize_vector_store=initialize_vector_store,
                fail_on_vector_store_error=fail_on_vector_store_error,
            )

    async def _update_tenancy_with_vector_mode_locked(
        self,
        tenant: str,
        project: str,
        *,
        initialize_vector_store: bool,
        fail_on_vector_store_error: bool,
    ) -> None:
        t, p = tenant.strip(), project.strip()
        if not t or not p:
            raise ValueError("tenant and project must be non-empty")

        tenancy_changed = (t, p) != self._active_tenancy

        triple = self.triple_store_manager
        if triple is not None and triple.supports_tenancy_partition():
            await triple.update_tenancy(t, p)
            if isinstance(triple, FusekiTripleStoreManager):
                fuseki_cfg = self.config.tool_config.fuseki
                fuseki_cfg.dataset = triple.dataset
                fuseki_cfg.ontologies_dataset = triple.ontologies_dataset

        if tenancy_changed:
            # The catalog, its alias-collision ledger, and the graph caches are all
            # partition-scoped. Carrying them across a switch leaks one tenant's
            # ontologies into another's requests -- and its alias ledger can reject
            # a legitimately distinct ontology that reuses an ontology_id.
            # ``None`` means this is the first assignment, which happens at startup
            # before ``initialize()``; leave the population to it rather than
            # fetching twice. Any later switch must repopulate -- including when the
            # partition we are leaving was empty. Seed TTLs are deliberately not
            # replayed here: writing them into a different tenant as a side effect
            # of a query parameter would be a surprise.
            is_first_assignment = self._active_tenancy is None
            self.ontology_manager.reset_catalog()
            self._active_tenancy = (t, p)
            if not is_first_assignment and triple is not None:
                for ontology in await triple.afetch_ontologies():
                    self.ontology_manager.add_ontology(ontology, skip_vector_index=True)

        if self.vector_store is not None:
            # No config copy-back: `create_vector_store_manager` passes
            # `tool_config.vector_store` by reference, so `apply_tenancy` has
            # already rewritten the very object Config holds.
            self.vector_store.apply_tenancy(t, p)
            if initialize_vector_store:
                try:
                    await self.vector_store.initialize()
                    self.vector_store_ready = True
                    self.vector_store_last_error = None
                except Exception as exc:
                    self.vector_store_ready = False
                    self.vector_store_last_error = exc
                    if fail_on_vector_store_error:
                        raise
                    logger.warning(
                        "Vector store tenancy initialization failed; continuing without vector retrieval: %s",
                        exc,
                    )

    async def clean_tenancy_data(self, tenant: str, project: str) -> None:
        """Flush triple-store and vector-store partitions for ``tenant`` / ``project``.

        Takes the tenancy lock: this is destructive, and a concurrent retarget
        would let it resolve partition names against a scope that changed
        mid-flight.
        """
        t, p = tenant.strip(), project.strip()
        if not t or not p:
            raise ValueError("tenant and project must be non-empty")

        async with self._get_tenancy_lock():
            triple = self.triple_store_manager
            if triple is not None:
                if not triple.supports_tenancy_partition():
                    raise NotImplementedError(
                        f"Triple store {type(triple).__name__} has no tenant/project partitions"
                    )
                await triple.clean_tenancy(t, p)

            vector = self.vector_store
            if vector is not None and vector.supports_tenancy_partition():
                await vector.clean_tenancy(t, p)

    def get_atomic_tools(self) -> AtomicToolBox:
        """Return the minimal toolbox used by atomic render/critic paths."""
        return self.atomic_tools

    def serialize(self, state: AgentState) -> None:
        """Persist the document's ontologies and facts.

        Drives :meth:`aserialize` under a single :func:`asyncio.run`, so a
        document with N ontologies costs one event loop and one backend
        connection rather than N of each -- the per-call sync entry points open
        (and tear down) a fresh HTTP client every time.

        Raises:
            RuntimeError: If called from inside a running event loop; await
                :meth:`aserialize` there instead.
        """
        require_no_running_loop("ToolBox.serialize", "ToolBox.aserialize")
        asyncio.run(self.aserialize(state))

    async def aserialize(self, state: AgentState) -> None:
        """Persist the document's ontologies and facts (async form)."""
        ontologies_to_serialize = document_ontology_access(
            state
        ).serialization_targets()
        for ontology in ontologies_to_serialize:
            if ontology and ontology.hash:
                self.ontology_manager.add_ontology(ontology)

        if self.triple_store_manager is not None:
            for ontology in ontologies_to_serialize:
                await self.triple_store_manager.aserialize(ontology)
            if state.render_facts:
                await self.triple_store_manager.aserialize(
                    state.aggregated_facts,
                    graph_uri=state.graph_uri,
                )

    def should_initialize_vector_store(
        self, ontology_context_mode: OntologyContextMode | None
    ) -> bool:
        return (
            self.vector_store is not None
            and ontology_context_mode
            == OntologyContextMode.SELECTED_VECTOR_SEARCH_ONTOLOGY
        )

    def is_vector_store_ready(self) -> bool:
        return self.vector_store is not None and self.vector_store_ready

    async def initialize(
        self,
        *,
        ontology_context_mode: OntologyContextMode | None = None,
        fail_on_vector_store_error: bool = True,
        wipe_vector_store: bool | None = None,
        prune_orphan_iris: bool | None = None,
    ) -> None:
        """Initialize the toolbox with ontologies and their properties.

        This method synchronizes ontologies between filesystem and triple store,
        then fetches ontologies from the triple store and updates their properties
        using the LLM tool.

        Args:
            ontology_context_mode: When vector search mode, ensure the vector store
                is ready before materializing atoms.
            fail_on_vector_store_error: Raise on vector init failure when True.
            wipe_vector_store: Drop the current vector partition before init.
                ``None`` uses ``VECTOR_STORE_WIPE_ON_INIT`` (default False).
            prune_orphan_iris: Delete indexed IRIs absent from the sync catalog.
                ``None`` uses ``VECTOR_STORE_PRUNE_ORPHAN_IRIS_ON_INIT`` (default True).
        """
        import asyncio
        import time

        init_started = time.perf_counter()
        vsc = self.config.tool_config.vector_store
        do_wipe = vsc.wipe_on_init if wipe_vector_store is None else wipe_vector_store
        do_prune = (
            vsc.prune_orphan_iris_on_init
            if prune_orphan_iris is None
            else prune_orphan_iris
        )

        if self.triple_store_manager is not None:
            await self.triple_store_manager.async_init()

        if self.should_initialize_vector_store(ontology_context_mode):
            vector_store = self.vector_store
            if vector_store is None:
                self.vector_store_ready = False
                self.vector_store_last_error = RuntimeError(
                    "Vector store is not configured"
                )
                if fail_on_vector_store_error:
                    raise self.vector_store_last_error
                logger.warning(
                    "Vector store was requested for initialization but is not configured"
                )
            else:
                try:
                    if do_wipe:
                        logger.warning(
                            "Wiping vector store partition before initialize "
                            "(wipe_vector_store=True)"
                        )
                        await vector_store.wipe_store()
                    await vector_store.initialize()
                    self.vector_store_ready = True
                    self.vector_store_last_error = None
                except Exception as exc:
                    self.vector_store_ready = False
                    self.vector_store_last_error = exc
                    if fail_on_vector_store_error:
                        raise
                    logger.warning(
                        "Vector store initialization failed; continuing without vector retrieval: %s",
                        exc,
                    )

        sync_started = time.perf_counter()
        synchronized_ontologies = await self._synchronize_ontologies()
        logger.info(
            "Ontology sync finished: %d ontolog(ies) in %.2fs",
            len(synchronized_ontologies),
            time.perf_counter() - sync_started,
        )

        if do_prune and self.is_vector_store_ready() and self.vector_store is not None:
            triple = self.triple_store_manager
            catalog_is_authoritative = (
                triple is None or triple.last_catalog_was_complete()
            )
            if not catalog_is_authoritative:
                # Pruning deletes indexed ontologies that the catalog no longer
                # mentions. A catalog that only partly loaded mentions fewer
                # ontologies than exist, so pruning against it deletes live
                # data on the strength of a network error.
                logger.warning(
                    "Skipping vector-store orphan prune: the ontology catalog "
                    "loaded incompletely, so absent IRIs are not evidence of "
                    "deletion."
                )
            else:
                keep_iris = {o.iri for o in synchronized_ontologies if o.iri}
                orphans = await asyncio.to_thread(
                    self.vector_store.prune_orphan_ontology_iris, keep_iris
                )
                if orphans:
                    logger.info(
                        "Pruned %d orphan ontology IRI(s) from vector store: %s",
                        len(orphans),
                        orphans,
                    )

        for ontology in synchronized_ontologies:
            self.ontology_manager.add_ontology(ontology, skip_vector_index=True)

        concurrency = max(1, vsc.reindex_concurrency)
        semaphore = asyncio.Semaphore(concurrency)

        async def _materialize_one(ontology: Ontology) -> None:
            async with semaphore:
                onto_started = time.perf_counter()
                await self._materialize_ontology(ontology)
                logger.info(
                    "Materialized ontology %s in %.2fs",
                    ontology.iri,
                    time.perf_counter() - onto_started,
                )

        materialize_started = time.perf_counter()
        await asyncio.gather(
            asyncio.gather(*[_materialize_one(o) for o in synchronized_ontologies]),
            update_ontology_manager(om=self.ontology_manager, llm_tool=self.llm),
        )
        logger.info(
            "Ontology materialize + enrich finished for %d ontolog(ies) in %.2fs "
            "(reindex_concurrency=%d); initialize total %.2fs",
            len(synchronized_ontologies),
            time.perf_counter() - materialize_started,
            concurrency,
            time.perf_counter() - init_started,
        )

    def _load_seed_ontologies_from_directory(self) -> list[Ontology]:
        """Load seed ontologies from ``ontology_directory`` (*.ttl)."""
        ontology_dir = self.config.tool_config.path_config.ontology_directory
        if ontology_dir is None:
            return []
        directory = pathlib.Path(ontology_dir).expanduser()
        if not directory.is_dir():
            return []
        ontologies: list[Ontology] = []
        for path in sorted(directory.glob("*.ttl")):
            try:
                ontologies.append(Ontology.from_file(path))
                logger.debug("Loaded seed ontology from %s", path)
            except Exception as exc:
                logger.error("Failed to load seed ontology %s: %s", path, exc)
        return ontologies

    async def _synchronize_ontologies(self) -> list[Ontology]:
        """Synchronize seed ontologies from disk into the triple store."""
        import asyncio

        seed_ontologies = await asyncio.to_thread(
            self._load_seed_ontologies_from_directory
        )
        if seed_ontologies:
            logger.info(
                "Found %d seed ontologies in ontology_directory", len(seed_ontologies)
            )

        triple_store_ontologies: list[Ontology] = []
        if self.triple_store_manager is not None:
            triple_store_ontologies = (
                await self.triple_store_manager.afetch_ontologies()
            )
            logger.info(
                "Found %d ontologies in triple store", len(triple_store_ontologies)
            )

        triple_store_iris = {o.iri for o in triple_store_ontologies}
        for seed_onto in seed_ontologies:
            if seed_onto.iri not in triple_store_iris:
                logger.info(
                    "Syncing seed ontology to triple store: %s (version: %s)",
                    seed_onto.iri,
                    seed_onto.version,
                )
                triple_store_ontologies.append(seed_onto)

        return triple_store_ontologies

    async def _materialize_ontology(self, ontology: Ontology) -> None:
        """Write ontology to the triple store and rebuild vector atoms."""
        import asyncio

        if self.triple_store_manager is not None:
            await self.triple_store_manager.aserialize(ontology)

        if self.is_vector_store_ready() and self.vector_store is not None:
            await asyncio.to_thread(self.vector_store.reindex_ontology, ontology)

    async def ingest_ontology_ttl(
        self, ttl: bytes, *, filename: str | None = None
    ) -> Ontology:
        """Register Turtle in the triple store and the vector index.

        ``ontology_directory`` is a read-only seed fixture consulted once at
        init, so nothing is written there: an ingested ontology lives in the
        triple store and vector index only and does not survive a rebuild from
        seeds.
        """
        import asyncio

        graph = RDFGraph()

        def _parse() -> None:
            graph.parse(BytesIO(ttl), format="turtle")

        await asyncio.to_thread(_parse)
        o = Ontology(graph=graph)
        if not o.iri or o.iri == ONTOLOGY_NULL_IRI:
            raise ValueError("Loaded turtle does not define a valid ontology IRI")
        if not o.hash:
            raise ValueError("Ontology hash could not be computed")
        self.ontology_manager.validate_identity_uniqueness(o)

        await self._materialize_ontology(o)
        self.ontology_manager.add_ontology(o, skip_vector_index=True)
        return o

    async def delete_ontology_by_iri(self, ontology_iri: str) -> None:
        """Remove ontology from manager, vector store, and triple store.

        ``ontology_directory`` is deliberately untouched. Deletion used to
        unlink any seed TTL declaring this IRI, which destroyed curated input
        the next init reloads from — an irreversible edit to the user's files
        in response to a store-level delete.
        """
        import asyncio

        self.ontology_manager.remove_ontology_by_iri(ontology_iri)
        if self.vector_store is not None:
            await asyncio.to_thread(self.vector_store.delete_ontology, ontology_iri)

        if self.triple_store_manager is not None:
            await self.triple_store_manager.drop_all_ontology_graphs_for_iri(
                ontology_iri
            )


async def render_ontology_summary(ontology: Ontology, llm_tool) -> OntologyProperties:
    """Generate a summary of ontology properties using LLM analysis.

    This function uses the LLM tool to analyze an RDF graph and generate
    a structured summary of its properties. Only unset fields are requested.

    Args:
        ontology: The ontology to analyze (for checking which fields are set).
        llm_tool: The LLM tool instance for analysis.

    Returns:
        OntologyProperties: A structured summary containing only the missing properties.
    """
    from typing import Any, cast

    from pydantic import create_model

    # Sample the graph intelligently (first 100 sections)
    # This provides context without overwhelming the LLM
    sampled_graph = sample_ontology_graph(ontology.graph, max_triples=100)
    # Serialize with consistent ordering to ensure determinism
    ontology_str = sampled_graph.serialize()

    # Determine which fields are unset and need LLM inference
    unset_fields = {}
    fields_to_fetch = []

    # Fields we want to potentially fetch from LLM (excluding internal fields like created_at)
    fields_to_check = ["title", "description", "ontology_id", "version", "iri"]

    # For Ontology objects, only fetch fields that are unset
    for field in fields_to_check:
        value = getattr(ontology, field, None)
        if value is None or (field == "iri" and value == ONTOLOGY_NULL_IRI):
            fields_to_fetch.append(field)
            # Get the field definition from the base model
            base_field = OntologyProperties.model_fields[field]
            unset_fields[field] = (base_field.annotation, base_field)

    if not unset_fields:
        # All fields are already set, return empty props
        return OntologyProperties()

    # Create a dynamic model with only unset fields
    DynamicProps = create_model("DynamicOntologyProps", **cast(Any, unset_fields))

    # Define the output parser
    parser = PydanticOutputParser(pydantic_object=DynamicProps)

    # Create the prompt template with format instructions
    field_list_str = "\n- ".join(fields_to_fetch)
    format_instructions = parser.get_format_instructions()

    # Build the template - use format_instructions as a separate variable to avoid brace conflicts
    template = (
        "Below is a sample of an ontology in Turtle format:\n\n"
        "```ttl\n{ontology_str}\n```\n\n"
        "Extract ONLY the following properties that are missing:\n"
        f"- {field_list_str}\n\n"
        "{format_instructions}"
    )

    prompt = PromptTemplate(
        template=template,
        input_variables=["ontology_str"],
        partial_variables={"format_instructions": format_instructions},
    )

    response = await llm_tool(prompt.format_prompt(ontology_str=ontology_str))
    dynamic_props = parser.parse(response.content)

    # Convert dynamic props to OntologyProperties
    result = OntologyProperties()
    for field in unset_fields.keys():
        value = getattr(dynamic_props, field, None)
        if value is not None:
            setattr(result, field, value)

    return result


def sample_ontology_graph(graph: RDFGraph, max_triples: int = 100) -> RDFGraph:
    """Sample an ontology graph to provide a representative subset.

    This function serializes the graph to Turtle format and takes the first
    N blank-line separated sections. This is deterministic and simpler than
    complex triple selection logic.

    Args:
        graph: The full ontology graph
        max_triples: Maximum number of sections to include in the sample

    Returns:
        RDFGraph: A sampled version of the ontology with representative triples
    """
    # Serialize to turtle
    turtle_str = graph.serialize_canonical_turtle()

    # Split on blank lines (typical turtle format uses \n\n to separate blocks)
    sections = turtle_str.split("\n\n")

    # Take first max_triples sections (or fewer if graph is smaller)
    num_sections = min(len(sections), max_triples)
    sampled_turtle = "\n\n".join(sections[:num_sections])

    # Parse back into a graph
    sampled = RDFGraph()
    sampled.parse(data=sampled_turtle, format="turtle")

    # Copy namespace bindings from original graph
    for prefix, namespace in graph.namespaces():
        if prefix:
            sampled.bind(prefix, namespace)

    return sampled
