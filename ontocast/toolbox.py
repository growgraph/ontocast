import logging

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from ontocast.config import Config
from ontocast.onto.ontology import Ontology, OntologyProperties
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.tool import (
    ChunkerTool,
    ConverterTool,
    FilesystemTripleStoreManager,
    FusekiTripleStoreManager,
    Neo4jTripleStoreManager,
)
from ontocast.tool.aggregate import ChunkRDFGraphAggregator
from ontocast.tool.cache import Cacher
from ontocast.tool.graph_diff import DiffTool
from ontocast.tool.graph_version_manager import GraphVersionManager
from ontocast.tool.llm import LLMTool
from ontocast.tool.ontology_manager import OntologyManager
from ontocast.tool.sparql import SPARQLTool
from ontocast.tool.triple_manager.core import (
    TripleStoreManager,
)

logger = logging.getLogger(__name__)


def update_ontology_properties(o: Ontology, llm_tool: LLMTool):
    """Update ontology properties using LLM analysis, only if missing.

    This function uses the LLM tool to analyze and update the properties
    of a given ontology based on its graph content, but only if any key
    property is missing or empty.
    """
    # Only update if any key property is missing or empty
    if not (o.title and o.ontology_id and o.description and o.version):
        props = render_ontology_summary(o.graph, llm_tool)
        o.set_properties(**props.model_dump())


def update_ontology_manager(om: OntologyManager, llm_tool: LLMTool):
    """Update properties for all ontologies in the manager.

    This function iterates through all ontologies in the manager and updates
    their properties using the LLM tool.

    Args:
        om: The ontology manager containing ontologies to update.
        llm_tool: The LLM tool instance for analysis.
    """
    for o in om.ontologies:
        update_ontology_properties(o, llm_tool)


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

        # Extract configuration values
        working_directory = tool_config.path_config.working_directory
        ontology_directory = tool_config.path_config.ontology_directory

        # Create shared cache instance with config
        self.shared_cache = Cacher(config=config)

        # LLM configuration - pass the entire LLM config to the tool
        self.llm_provider = tool_config.llm_config.provider
        self.llm: LLMTool = LLMTool.create(
            config=tool_config.llm_config, cache=self.shared_cache
        )

        # Initialize managers based on backend configuration
        self.filesystem_manager: FilesystemTripleStoreManager | None = None
        self.triple_store_manager: TripleStoreManager | None = None

        # Automatically determine which backends to use based on available configuration
        use_fuseki = tool_config.fuseki.uri and tool_config.fuseki.auth
        use_neo4j = tool_config.neo4j.uri and tool_config.neo4j.auth
        use_filesystem_triple_store = (
            working_directory is not None and ontology_directory is not None
        )
        use_filesystem_manager = working_directory is not None

        # Validate that we have at least one backend configured
        if not any([use_fuseki, use_neo4j, use_filesystem_triple_store]):
            raise ValueError(
                "No backend configured. Please provide Fuseki/Neo4j credentials or working directory and ontology directory."
            )

        # Create main triple store manager (only one can be active)
        if use_fuseki and tool_config.fuseki.uri and tool_config.fuseki.auth:
            clean = config.server.clean
            self.triple_store_manager = FusekiTripleStoreManager(
                uri=tool_config.fuseki.uri,
                auth=tool_config.fuseki.auth,
                dataset=tool_config.fuseki.dataset,
                ontologies_dataset=tool_config.fuseki.ontologies_dataset,
                clean=clean,
            )
        elif use_neo4j and tool_config.neo4j.uri and tool_config.neo4j.auth:
            clean = config.server.clean
            self.triple_store_manager = Neo4jTripleStoreManager(
                uri=tool_config.neo4j.uri, auth=tool_config.neo4j.auth, clean=clean
            )
        elif use_filesystem_triple_store:
            if working_directory is None or ontology_directory is None:
                raise ValueError(
                    "Working directory and ontology directory must be provided for filesystem triple store"
                )
            self.triple_store_manager = FilesystemTripleStoreManager(
                working_directory=working_directory,
                ontology_path=ontology_directory,
            )

        # Create filesystem manager (can be combined with other backends)
        if use_filesystem_manager:
            if working_directory is None or ontology_directory is None:
                raise ValueError(
                    "Working directory and ontology directory must be provided for filesystem manager"
                )
            self.filesystem_manager = FilesystemTripleStoreManager(
                working_directory=working_directory,
                ontology_path=ontology_directory,
            )

        self.ontology_manager: OntologyManager = OntologyManager()
        self.converter: ConverterTool = ConverterTool(cache=self.shared_cache)
        self.chunker: ChunkerTool = ChunkerTool(
            chunk_config=tool_config.chunk_config, cache=self.shared_cache
        )
        self.aggregator: ChunkRDFGraphAggregator = ChunkRDFGraphAggregator()

        # SPARQL, version management, and diff tools
        self.sparql_tool: SPARQLTool = SPARQLTool(
            triple_store_manager=self.triple_store_manager
        )
        self.version_manager: GraphVersionManager = GraphVersionManager()
        self.diff_tool: DiffTool = DiffTool()

    def get_llm_tool_with_budget_tracker(self, budget_tracker):
        """Get an LLM tool instance with a specific budget tracker.

        Args:
            budget_tracker: The budget tracker instance to use.

        Returns:
            LLMTool: LLM tool with the specified budget tracker.
        """
        # Create a new LLM tool with the budget tracker
        return LLMTool.create(
            config=self.config.tool_config.llm_config,
            cache=self.shared_cache,
            budget_tracker=budget_tracker,
        )

    def update_dataset(self, dataset: str) -> None:
        """Update the dataset for the Fuseki triple store manager.

        This method allows changing the dataset without recreating the entire
        ToolBox, which is efficient for API requests that specify different datasets.

        Args:
            dataset: The new dataset name to use.
        """
        if self.triple_store_manager is not None:
            from ontocast.tool.triple_manager.fuseki import FusekiTripleStoreManager

            if isinstance(self.triple_store_manager, FusekiTripleStoreManager):
                self.triple_store_manager.update_dataset(dataset)
            else:
                logger.warning(
                    "Cannot update dataset: triple store manager is not Fuseki"
                )

    def serialize(self, state: AgentState) -> None:
        if self.filesystem_manager is not None:
            self.filesystem_manager.serialize_graph(state.current_ontology.graph)
        if self.triple_store_manager is not None:
            # Store ontology in main dataset for reasoning
            self.triple_store_manager.serialize_graph(
                state.current_ontology.graph, graph_uri=state.current_ontology.iri
            )
            # Store ontology in ontologies dataset for management (if available)
            if hasattr(self.triple_store_manager, "serialize_ontology_graph"):
                # Type ignore because we're checking for the method dynamically
                self.triple_store_manager.serialize_ontology_graph(  # type: ignore
                    state.current_ontology.graph, graph_uri=state.current_ontology.iri
                )

        if state.aggregated_facts and len(state.aggregated_facts) > 0:
            if self.filesystem_manager is not None:
                self.filesystem_manager.serialize_graph(
                    state.aggregated_facts,
                    graph_uri=state.doc_namespace,
                )
            if self.triple_store_manager is not None:
                self.triple_store_manager.serialize_graph(
                    state.aggregated_facts,
                    graph_uri=state.doc_namespace,
                )

    def initialize(self) -> None:
        """Initialize the toolbox with ontologies and their properties.

        This method synchronizes ontologies between filesystem and triple store,
        then fetches ontologies from the triple store and updates their properties
        using the LLM tool.
        """

        # Synchronize ontologies and get the final set
        ontologies = self._synchronize_ontologies()

        # Use the synchronized ontologies
        if ontologies is not None:
            self.ontology_manager.ontologies = ontologies
            update_ontology_manager(om=self.ontology_manager, llm_tool=self.llm)
        elif self.filesystem_manager is not None:
            # Fallback to filesystem if no triple store manager
            self.ontology_manager.ontologies = (
                self.filesystem_manager.fetch_ontologies()
            )
            update_ontology_manager(om=self.ontology_manager, llm_tool=self.llm)

    def _synchronize_ontologies(self) -> list | None:
        """Synchronize ontologies between filesystem and triple store.

        This method checks both filesystem_manager and triple_store_manager for
        ontologies and populates triple_store_manager with any ontologies from
        filesystem_manager that are not present in triple_store_manager.

        Returns:
            list | None: The final set of ontologies after synchronization, or None if no triple store manager.
        """
        if self.triple_store_manager is None:
            logger.debug("No triple store manager available for synchronization")
            return None

        # Get ontologies from filesystem if available
        filesystem_ontologies = []
        if self.filesystem_manager is not None:
            filesystem_ontologies = self.filesystem_manager.fetch_ontologies()
            logger.debug(f"Found {len(filesystem_ontologies)} ontologies in filesystem")

        # Get ontologies from triple store
        triple_store_ontologies = self.triple_store_manager.fetch_ontologies()
        logger.debug(f"Found {len(triple_store_ontologies)} ontologies in triple store")

        # Create a set of existing ontology IRIs in triple store for quick lookup
        existing_iris = {onto.iri for onto in triple_store_ontologies}

        # Find ontologies in filesystem that are not in triple store
        new_ontologies = []
        for fs_ontology in filesystem_ontologies:
            if fs_ontology.iri not in existing_iris:
                new_ontologies.append(fs_ontology)
                logger.debug(f"Found new ontology in filesystem: {fs_ontology.iri}")

        # Store new ontologies in triple store
        if new_ontologies:
            logger.info(f"Syncing {len(new_ontologies)} new ontologies to triple store")
            for ontology in new_ontologies:
                # Store ontology in main dataset for reasoning
                self.triple_store_manager.serialize_graph(
                    graph=ontology.graph, graph_uri=ontology.iri
                )
                # Store ontology in ontologies dataset for management (if available)
                if hasattr(self.triple_store_manager, "serialize_ontology_graph"):
                    # Type ignore because we're checking for the method dynamically
                    self.triple_store_manager.serialize_ontology_graph(  # type: ignore
                        graph=ontology.graph, graph_uri=ontology.iri
                    )
                logger.debug(f"Synced ontology to triple store: {ontology.iri}")
        else:
            logger.debug("No new ontologies to sync from filesystem to triple store")

        # Return the final set of ontologies (triple store + newly synced)
        return triple_store_ontologies


def render_ontology_summary(graph: RDFGraph, llm_tool) -> OntologyProperties:
    """Generate a summary of ontology properties using LLM analysis.

    This function uses the LLM tool to analyze an RDF graph and generate
    a structured summary of its properties.

    Args:
        graph: The RDF graph to analyze.
        llm_tool: The LLM tool instance for analysis.

    Returns:
        OntologyProperties: A structured summary of the ontology properties.
    """
    ontology_str = graph.serialize(format="turtle")

    # Define the output parser
    parser = PydanticOutputParser(pydantic_object=OntologyProperties)

    # Create the prompt template with format instructions
    prompt = PromptTemplate(
        template=(
            "Below is an ontology in Turtle format:\n\n"
            "```ttl\n{ontology_str}\n```\n\n"
            "{format_instructions}"
        ),
        input_variables=["ontology_str"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    response = llm_tool(prompt.format_prompt(ontology_str=ontology_str))

    return parser.parse(response.content)
