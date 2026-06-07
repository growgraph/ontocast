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
from ontocast.tool.llm import LLMTool
from ontocast.tool.ontology_manager import OntologyManager
from ontocast.tool.sparql import SPARQLTool
from ontocast.tool.triple_manager.core import TripleStoreManager
from ontocast.tool.vector_store import (
    EmbeddingTool,
    FastembedBm25SparseTool,
    OntologyPatchRetriever,
    QdrantVectorStore,
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


async def update_ontology_manager(om: OntologyManager, llm_tool: LLMTool):
    """Update properties for all ontologies in the manager.

    This function iterates through all ontologies in the manager and updates
    their properties using the LLM tool.

    Args:
        om: The ontology manager containing ontologies to update.
        llm_tool: The LLM tool instance for analysis.
    """
    for o in om.ontologies:
        await update_ontology_properties(o, llm_tool)


class ToolBox:
    """A container class for all tools used in the ontology processing workflow.

    This class initializes and manages various tools needed for document processing,
    ontology management, and LLM interactions.

    Args:
        config: Configuration object containing all necessary settings.
    """

    def __init__(self, config: Config):
        # Store the config for later use
        self.config = config

        # Get tool configuration
        tool_config = config.get_tool_config()

        # Create shared cache instance with config
        self.shared_cache = Cacher(config=config)

        # LLM configuration - pass the entire LLM config to the tool
        self.llm_provider = tool_config.llm_config.provider
        self.llm: LLMTool = LLMTool.create(
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
        self.converter: ConverterTool = ConverterTool(cache=self.shared_cache)
        self.chunker: ChunkerTool = ChunkerTool(
            chunk_config=tool_config.chunk_config, cache=self.shared_cache
        )
        self.aggregator: EmbeddingBasedAggregator = EmbeddingBasedAggregator(
            embedding_model=tool_config.aggregation.embedding_model,
            similarity_threshold=tool_config.aggregation.similarity_threshold,
        )
        self._entity_aligners: dict[tuple[str, float], EntityAligner] = {}

        # SPARQL, version management, and diff tools
        self.sparql_tool: SPARQLTool = SPARQLTool(
            triple_store_manager=self.triple_store_manager
        )
        self.version_manager: GraphVersionManager = GraphVersionManager()
        self.diff_tool: DiffTool = DiffTool()

        self.embedding_tool: EmbeddingTool = EmbeddingTool.create(tool_config.embedding)
        self.vector_store: QdrantVectorStore | None = None
        self.patch_retriever: OntologyPatchRetriever | None = None
        self.vector_store_ready: bool = False
        self.vector_store_last_error: Exception | None = None

        if tool_config.qdrant.uri:
            q_vs = tool_config.qdrant.vector_size
            emb_dim = tool_config.embedding.dimension
            if q_vs is not None and q_vs != emb_dim:
                raise ValueError(
                    "QdrantConfig.vector_size must match "
                    "EmbeddingConfig.dimension when set "
                    f"(got vector_size={q_vs}, embedding.dimension={emb_dim})"
                )
            # BM25 is always enabled whenever vector search is enabled.
            sparse_embedding = FastembedBm25SparseTool(config=tool_config.embedding)
            self.vector_store = QdrantVectorStore(
                config=tool_config.qdrant,
                embedding=self.embedding_tool,
                sparse_embedding=sparse_embedding,
            )
            self.patch_retriever = OntologyPatchRetriever(
                vector_store=self.vector_store,
                sparql_tool=self.sparql_tool,
                patch=tool_config.patch_retrieval,
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
        """Return the shared LLM tool with the given budget tracker attached.

        Args:
            budget_tracker: The budget tracker instance to use.

        Returns:
            LLMTool: Shared LLM tool with the specified budget tracker.
        """
        self.llm.budget_tracker = budget_tracker
        return self.llm

    def require_triple_store_manager(self) -> TripleStoreManager:
        """Return the configured triple store manager or raise a clear error."""
        manager = self.triple_store_manager
        if manager is None:
            raise RuntimeError("Triple store backend is not configured")
        return manager

    async def update_tenancy(self, tenant: str, project: str) -> None:
        """Retarget Fuseki datasets and Qdrant collections for ``tenant`` / ``project``."""
        await self.update_tenancy_with_vector_mode(
            tenant,
            project,
            initialize_vector_store=True,
            fail_on_vector_store_error=True,
        )

    async def update_tenancy_with_vector_mode(
        self,
        tenant: str,
        project: str,
        *,
        initialize_vector_store: bool,
        fail_on_vector_store_error: bool,
    ) -> None:
        """Retarget tenancy and optionally initialize vector store collections."""
        t, p = tenant.strip(), project.strip()
        if not t or not p:
            raise ValueError("tenant and project must be non-empty")

        triple = self.triple_store_manager
        if triple is not None and triple.supports_tenancy_partition():
            await triple.update_tenancy(t, p)
            if isinstance(triple, FusekiTripleStoreManager):
                fuseki_cfg = self.config.tool_config.fuseki
                fuseki_cfg.dataset = triple.dataset
                fuseki_cfg.ontologies_dataset = triple.ontologies_dataset

        if self.vector_store is not None:
            self.vector_store.apply_tenancy(t, p)
            qcfg = self.config.tool_config.qdrant
            qcfg.ontology_collection = self.vector_store.config.ontology_collection
            qcfg.facts_collection = self.vector_store.config.facts_collection
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
        """Flush triple-store and vector-store partitions for ``tenant`` / ``project``."""
        t, p = tenant.strip(), project.strip()
        if not t or not p:
            raise ValueError("tenant and project must be non-empty")

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
    ) -> None:
        """Initialize the toolbox with ontologies and their properties.

        This method synchronizes ontologies between filesystem and triple store,
        then fetches ontologies from the triple store and updates their properties
        using the LLM tool.
        """
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

        # Synchronize ontologies, push to remote triple store + vector index, then register
        synchronized_ontologies = await self._synchronize_ontologies()
        for ontology in synchronized_ontologies:
            await self._materialize_ontology(ontology)
        for ontology in synchronized_ontologies:
            self.ontology_manager.add_ontology(ontology, skip_vector_index=True)
        await update_ontology_manager(om=self.ontology_manager, llm_tool=self.llm)

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
