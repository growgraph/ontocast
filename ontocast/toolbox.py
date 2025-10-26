import logging

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from ontocast.config import Config
from ontocast.onto.constants import ONTOLOGY_NULL_IRI
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
    if (o.title is None) or (o.ontology_id is None) or (o.description is None):
        props = render_ontology_summary(o, llm_tool)
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
            self.filesystem_manager.serialize(state.current_ontology)
            self.filesystem_manager.serialize(
                state.aggregated_facts,
                graph_uri=state.doc_namespace,
            )
        if (
            self.triple_store_manager is not None
            and self.triple_store_manager != self.filesystem_manager
        ):
            # Store ontology in main dataset for reasoning
            self.triple_store_manager.serialize(state.current_ontology)
            self.triple_store_manager.serialize(
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
        self.ontology_manager.ontologies = self._synchronize_ontologies()
        update_ontology_manager(om=self.ontology_manager, llm_tool=self.llm)

    def _synchronize_ontologies(self) -> list[Ontology]:
        """Synchronize ontologies between filesystem and triple store.

        This method checks both filesystem_manager and triple_store_manager for
        ontologies and populates triple_store_manager with any ontologies from
        filesystem_manager that are not present in triple_store_manager.

        Returns:
            list: The final set of ontologies after synchronization
        """

        ontologies = []

        if self.filesystem_manager is not None:
            ontologies += self.filesystem_manager.fetch_ontologies()
            logger.debug(f"Found {len(ontologies)} ontologies in filesystem")

        triple_store_ontologies = []
        if (
            self.triple_store_manager is not None
            and self.triple_store_manager != self.filesystem_manager
        ):
            triple_store_ontologies += self.triple_store_manager.fetch_ontologies()
            logger.debug(
                f"Found {len(triple_store_ontologies)} ontologies in triple store"
            )

        # Find ontologies in filesystem that are not in triple store
        for o in triple_store_ontologies:
            if o.iri not in [item.iri for item in ontologies]:
                ontologies.append(o)
                logger.debug(f"Found new ontology in filesystem: {o.iri}")

        return ontologies


def render_ontology_summary(ontology: Ontology, llm_tool) -> OntologyProperties:
    """Generate a summary of ontology properties using LLM analysis.

    This function uses the LLM tool to analyze an RDF graph and generate
    a structured summary of its properties. Only unset fields are requested.

    Args:
        ontology: The ontology to analyze (for checking which fields are set).
        llm_tool: The LLM tool instance for analysis.

    Returns:
        OntologyProperties: A structured summary containing only the missing properties.
    """
    from pydantic import create_model

    # Sample the graph intelligently (first 100 triples + ontology metadata)
    # This provides context without overwhelming the LLM
    sampled_graph = sample_ontology_graph(ontology.graph)
    ontology_str = sampled_graph.serialize(format="turtle")

    # Determine which fields are unset and need LLM inference
    unset_fields = {}
    fields_to_fetch = []

    # Fields we want to potentially fetch from LLM (excluding internal fields like updated_at)
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
    DynamicProps = create_model("DynamicOntologyProps", **unset_fields)

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

    response = llm_tool(prompt.format_prompt(ontology_str=ontology_str))
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

    This function extracts key triples from the ontology, prioritizing:
    1. Ontology metadata (ontology-level properties)
    2. Class definitions (owl:Class, rdfs:Class)
    3. Property definitions (owl:ObjectProperty, owl:DatatypeProperty)
    4. Fills the rest with triples from the graph

    Args:
        graph: The full ontology graph
        max_triples: Maximum number of triples to include in the sample

    Returns:
        RDFGraph: A sampled version of the ontology with representative triples
    """
    from rdflib import OWL, RDF, RDFS

    sampled = RDFGraph()

    # Priority 1: Get ontology entity and its properties
    for s, p, o in graph.triples((None, RDF.type, OWL.Ontology)):
        sampled.add((s, p, o))
        # Get all properties of the ontology
        for prop, obj in graph.predicate_objects(s):
            sampled.add((s, prop, obj))

    # Priority 2: Get class definitions
    for s, p, o in graph.triples((None, RDF.type, OWL.Class)):
        if len(sampled) >= max_triples // 2:
            break
        sampled.add((s, p, o))

    for s, p, o in graph.triples((None, RDF.type, RDFS.Class)):
        if len(sampled) >= max_triples // 2:
            break
        sampled.add((s, p, o))

    # Priority 3: Get property definitions
    for prop_type in [OWL.ObjectProperty, OWL.DatatypeProperty]:
        for s, p, o in graph.triples((None, RDF.type, prop_type)):
            if len(sampled) >= max_triples * 3 // 4:
                break
            sampled.add((s, p, o))

    # Priority 4: Fill the rest with any remaining triples
    for triple in graph:
        if len(sampled) >= max_triples:
            break
        if triple not in sampled:
            sampled.add(triple)

    # Copy namespaces
    for prefix, namespace in graph.namespaces():
        sampled.bind(prefix, namespace)

    return sampled
