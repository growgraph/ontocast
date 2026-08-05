import asyncio
import logging
import pathlib
from io import BytesIO

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from ontocast.config import Config, WebSearchProvider
from ontocast.onto.constants import ONTOLOGY_NULL_IRI
from ontocast.onto.enum import OntologyContextMode
from ontocast.onto.ontology import Ontology, OntologyProperties
from ontocast.onto.ontology_access import document_ontology_access
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
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
from ontocast.tool.graph_diff import DiffTool
from ontocast.tool.graph_version_manager import GraphVersionManager
from ontocast.tool.llm import LLMTool, _active_budget_tracker
from ontocast.tool.ontology_manager import OntologyManager
from ontocast.tool.sparql import SPARQLTool
from ontocast.tool.triple_manager.core import TripleStoreManager
from ontocast.tool.vector_store import (
    EmbeddingTool,
    FastembedBm25SparseTool,
    OntologyPatchRetriever,
    VectorStoreManager,
    create_vector_store_manager,
)
from ontocast.tool.web_search import DuckDuckGoSearchProvider

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
        llm = await LLMTool.acreate(
            config=config.get_tool_config().llm_config, cache=Cacher(config=config)
        )
        return cls(config, llm=llm)

    def __init__(self, config: Config, *, llm: LLMTool | None = None):
        # Store the config for later use
        self.config = config

        # Get tool configuration
        tool_config = config.get_tool_config()

        # Create shared cache instance with config
        self.shared_cache = Cacher(config=config)

        # LLM configuration - pass the entire LLM config to the tool.
        # `acreate` passes a pre-built tool because LLMTool.create() drives
        # provider setup through asyncio.run(), which raises inside a loop.
        self.llm_provider = tool_config.llm_config.provider
        self.llm: LLMTool = llm or LLMTool.create(
            config=tool_config.llm_config, cache=self.shared_cache
        )
        self.search_provider = None
        if tool_config.web_search.enabled:
            if tool_config.web_search.provider == WebSearchProvider.DUCKDUCKGO:
                self.search_provider = DuckDuckGoSearchProvider(
                    timeout_seconds=tool_config.web_search.timeout_seconds,
                    region=tool_config.web_search.region,
                    safesearch=tool_config.web_search.safesearch,
                )
            else:
                raise ValueError(
                    f"Unsupported web-search provider: {tool_config.web_search.provider}"
                )
        self.atomic_tools = AtomicToolBox(
            llm_provider=self,
            search_provider=self.search_provider,
            web_search_config=tool_config.web_search,
            facts_validation_config=tool_config.facts_validation,
            citation_vocabulary=tool_config.chunk_config.citation_vocabulary,
        )

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
        self.converter: ConverterTool = ConverterTool(
            cache=self.shared_cache,
            converter_config=tool_config.converter_config,
        )
        self.chunker: ChunkerTool = ChunkerTool(
            chunk_config=tool_config.chunk_config, cache=self.shared_cache
        )
        self.aggregator: EmbeddingBasedAggregator = EmbeddingBasedAggregator(
            embedding_model=tool_config.aggregation.embedding_model,
            similarity_threshold=tool_config.aggregation.similarity_threshold,
            candidate_similarity_threshold=(
                tool_config.aggregation.candidate_similarity_threshold
            ),
            lexical_label_jaccard=tool_config.aggregation.lexical_label_jaccard,
            lexical_sequence_ratio=tool_config.aggregation.lexical_sequence_ratio,
            lexical_token_jaccard=tool_config.aggregation.lexical_token_jaccard,
            functional_min_empirical_support=(
                tool_config.aggregation.functional_min_empirical_support
            ),
            sibling_guard_scope=str(tool_config.aggregation.sibling_guard_scope),
        )
        self._entity_aligners: dict[tuple[str, float], EntityAligner] = {}

        # SPARQL, version management, and diff tools
        self.sparql_tool: SPARQLTool = SPARQLTool(
            triple_store_manager=self.triple_store_manager
        )
        self.version_manager: GraphVersionManager = GraphVersionManager()
        self.diff_tool: DiffTool = DiffTool()

        self.embedding_tool: EmbeddingTool = EmbeddingTool.create(tool_config.embedding)
        self.vector_store: VectorStoreManager | None = None
        self.patch_retriever: OntologyPatchRetriever | None = None
        self.vector_store_ready: bool = False
        self.vector_store_last_error: Exception | None = None

        # The factory owns backend selection, including resolving AUTO and
        # returning None when the backend is explicitly disabled. Only the
        # external backends need a BM25 tool; the in-memory store scores BM25
        # itself rather than pulling fastembed.
        needs_sparse = tool_config.qdrant.uri or tool_config.lancedb.enabled
        vector_store = create_vector_store_manager(
            tool_config,
            embedding=self.embedding_tool,
            sparse_embedding=(
                FastembedBm25SparseTool(config=tool_config.embedding)
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

    def get_entity_aligner(
        self,
        embedding_model: str | None = None,
        similarity_threshold: float | None = None,
    ) -> EntityAligner:
        """Return a cached entity aligner for the given embedding settings."""
        tool_config = self.config.get_tool_config()
        model = embedding_model or tool_config.aggregation.embedding_model
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else tool_config.aggregation.similarity_threshold
        )
        cache_key = (model, threshold)
        aligner = self._entity_aligners.get(cache_key)
        if aligner is None:
            aligner = EntityAligner(
                embedding_model=model,
                similarity_threshold=threshold,
            )
            self._entity_aligners[cache_key] = aligner
        return aligner

    async def get_llm_tool(self, budget_tracker):
        """Return the shared LLM tool, charging usage to ``budget_tracker``.

        The tracker is bound to the *calling task* rather than to the shared
        tool instance. Assigning it to the instance -- as this did previously --
        meant that with ``PARALLEL_WORKERS`` unit workers in flight, whichever
        one bound last collected every concurrent call's usage; document totals
        still summed correctly, but per-unit attribution was arbitrary.

        Args:
            budget_tracker: The budget tracker to charge for this task's calls.

        Returns:
            LLMTool: The shared LLM tool.
        """
        _active_budget_tracker.set(budget_tracker)
        return self.llm

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
                "No vector store is configured. Set VECTOR_STORE_BACKEND=memory "
                "for the dependency-free in-memory store, or configure "
                "QDRANT_URI / LANCEDB_ENABLED."
            )
        return self.vector_store

    def require_patch_retriever(self) -> OntologyPatchRetriever:
        """Return the ontology patch retriever or raise a directive error."""
        if self.patch_retriever is None:
            raise RuntimeError(
                "Ontology patch retrieval needs a vector store. Set "
                "VECTOR_STORE_BACKEND=memory for the dependency-free backend."
            )
        return self.patch_retriever

    async def aclose(self) -> None:
        """Release every backend connection this ToolBox opened.

        The ToolBox owns an httpx client (Fuseki) and a Qdrant client, neither
        of which was previously closed anywhere -- ``FusekiTripleStoreManager``
        even defined ``close()`` that nothing called. Long-lived hosts that
        build a ToolBox per tenant, and tests that build many, leaked sockets.

        Safe to call more than once, and never raises: teardown failures are
        logged, since a caller shutting down cannot act on them.
        """
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
            self.vector_store.apply_tenancy(t, p)
            vsc = self.config.tool_config.vector_store
            vsc.ontology_table = self.vector_store.store_config.ontology_table
            vsc.facts_table = self.vector_store.store_config.facts_table
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
        ontologies_to_serialize = document_ontology_access(
            state
        ).serialization_targets()
        for ontology in ontologies_to_serialize:
            if ontology and ontology.hash:
                self.ontology_manager.add_ontology(ontology)

        if self.triple_store_manager is not None:
            for ontology in ontologies_to_serialize:
                self.triple_store_manager.serialize(ontology)
            if state.render_facts:
                self.triple_store_manager.serialize(
                    state.aggregated_facts,
                    graph_uri=state.graph_uri,
                )

    async def aserialize(self, state: AgentState) -> None:
        """Async-safe form of :meth:`serialize`.

        :meth:`serialize` reaches a Fuseki write path that wraps a coroutine in
        :func:`asyncio.run`. That is fine on the graph path, where LangGraph
        offloads sync nodes to a worker thread, and raises for an embedder
        calling it directly from a coroutine.
        """
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
        """Persist Turtle to ``ontology_directory``, triple store, and vector index."""
        import asyncio

        ontology_dir = self.config.tool_config.path_config.ontology_directory
        if ontology_dir is None:
            raise ValueError("ontology_directory is not configured")
        ontology_dir = pathlib.Path(ontology_dir).expanduser()
        ontology_dir.mkdir(parents=True, exist_ok=True)

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
        """Remove ontology from manager, vector store, seed files, and triple store."""
        import asyncio

        self.ontology_manager.remove_ontology_by_iri(ontology_iri)
        if self.vector_store is not None:
            await asyncio.to_thread(self.vector_store.delete_ontology, ontology_iri)

        cfg_od = self.config.tool_config.path_config.ontology_directory
        if cfg_od is not None:
            self._unlink_ttl_files_if_ontology_iri(
                ontology_iri, pathlib.Path(cfg_od).expanduser(), "*.ttl"
            )

        if self.triple_store_manager is not None:
            await self.triple_store_manager.drop_all_ontology_graphs_for_iri(
                ontology_iri
            )

    @staticmethod
    def _unlink_ttl_files_if_ontology_iri(
        ontology_iri: str, directory: pathlib.Path, glob_pat: str
    ) -> None:
        if not directory.is_dir():
            return
        for path in sorted(directory.glob(glob_pat)):
            try:
                loaded = Ontology.from_file(path)
            except Exception:
                continue
            if loaded.iri == ontology_iri:
                path.unlink(missing_ok=True)
                logger.info("Removed ontology TTL %s", path)


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
