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
from ontocast.tool.llm import LLMTool
from ontocast.tool.ontology_manager import OntologyManager
from ontocast.tool.triple_manager.core import TripleStoreManagerWithAuth


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
        # Get tool configuration
        tool_config = config.get_tool_config()

        # Extract configuration values
        working_directory = tool_config.paths.working_directory
        ontology_directory = tool_config.paths.ontology_directory

        # LLM configuration - pass the entire LLM config to the tool
        self.llm_provider = tool_config.llm.provider
        self.llm: LLMTool = LLMTool.create(config=tool_config.llm)

        # Filesystem manager for initial ontology loading (if ontology_directory provided)
        self.filesystem_manager: FilesystemTripleStoreManager | None = None
        self.triple_store_manager: TripleStoreManagerWithAuth | None = None

        if ontology_directory is not None and working_directory is not None:
            self.filesystem_manager = FilesystemTripleStoreManager(
                working_directory=working_directory,
                ontology_path=ontology_directory,
            )

        # Main triple store manager - prefer Fuseki over Neo4j, fallback to filesystem
        # Get clean flag from server config
        clean = config.server.clean

        if tool_config.fuseki.uri and tool_config.fuseki.auth:
            self.triple_store_manager = FusekiTripleStoreManager(
                uri=tool_config.fuseki.uri,
                auth=tool_config.fuseki.auth,
                dataset="dataset0",  # Default dataset name
                clean=clean,
            )
        elif tool_config.neo4j.uri and tool_config.neo4j.auth:
            self.triple_store_manager = Neo4jTripleStoreManager(
                uri=tool_config.neo4j.uri, auth=tool_config.neo4j.auth, clean=clean
            )

        self.ontology_manager: OntologyManager = OntologyManager()
        self.converter: ConverterTool = ConverterTool()
        self.chunker: ChunkerTool = ChunkerTool()
        self.aggregator: ChunkRDFGraphAggregator = ChunkRDFGraphAggregator()

    def serialize(self, state: AgentState) -> None:
        if self.filesystem_manager is not None:
            self.filesystem_manager.serialize_ontology(state.current_ontology)
        if self.triple_store_manager is not None:
            self.triple_store_manager.serialize_ontology(state.current_ontology)

        if state.aggregated_facts and len(state.aggregated_facts) > 0:
            if self.filesystem_manager is not None:
                self.filesystem_manager.serialize_facts(
                    state.aggregated_facts,
                    spec=state.doc_namespace,
                    chunk_uri=getattr(state, "chunk_uri", None),
                )
            if self.triple_store_manager is not None:
                self.triple_store_manager.serialize_facts(
                    state.aggregated_facts,
                    spec=state.doc_namespace,
                    chunk_uri=getattr(state, "chunk_uri", None),
                )


def init_toolbox(toolbox: ToolBox):
    """Initialize the toolbox with ontologies and their properties.

    This function fetches ontologies from the triple store and updates
    their properties using the LLM tool. If a filesystem manager is available
    for initial loading, it will be used to load ontologies from files first.

    Args:
        toolbox: The ToolBox instance to initialize.
    """

    # If we have a filesystem manager, use it to load initial ontologies
    if toolbox.filesystem_manager is not None:
        initial_ontologies = toolbox.filesystem_manager.fetch_ontologies()

        if toolbox.triple_store_manager is not None:
            # Store these ontologies in the main triple store manager
            for ontology in initial_ontologies:
                toolbox.triple_store_manager.serialize_ontology(ontology)

    # Now fetch ontologies from the main triple store manager
    tm = (
        toolbox.triple_store_manager
        if toolbox.triple_store_manager is not None
        else toolbox.filesystem_manager
    )
    if tm is not None:
        toolbox.ontology_manager.ontologies = tm.fetch_ontologies()
        update_ontology_manager(om=toolbox.ontology_manager, llm_tool=toolbox.llm)


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
