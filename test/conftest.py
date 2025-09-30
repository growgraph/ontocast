import os
from pathlib import Path

import pytest
from suthing import FileHandle

from ontocast.onto.constants import DEFAULT_DOMAIN
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.tool import (
    FilesystemTripleStoreManager,
    LLMTool,
    OntologyManager,
)
from ontocast.tool.triple_manager import Neo4jTripleStoreManager
from ontocast.tool.triple_manager.fuseki import FusekiTripleStoreManager
from ontocast.toolbox import ToolBox, init_toolbox


@pytest.fixture
def current_domain():
    return os.getenv("CURRENT_DOMAIN", DEFAULT_DOMAIN)


@pytest.fixture
def llm_base_url():
    return os.getenv("LLM_BASE_URL", None)


@pytest.fixture
def provider():
    return os.getenv("LLM_PROVIDER", "openai")


@pytest.fixture
def model_name():
    return os.getenv("LLM_MODEL_NAME", None)


@pytest.fixture
def temperature():
    return 0.1


@pytest.fixture
def test_ontology():
    return RDFGraph._from_turtle_str(
        """
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix ex: <http://example.org/to/> .
    @prefix schema: <https://schema.org/> .
    @prefix dcterms: <http://purl.org/dc/terms/> .
    
    ex: rdf:type owl:Ontology ;
        rdfs:label "Test Domain Ontology" ;
        dcterms:title "test_onto"^^rdf:XMLLiteral ;
        rdfs:comment "An ontology for testing that covers basic concepts and relationships in a test domain. Used for validating ontology processing functionality." .
    
    ex:SpaceTimeEvent a rdfs:Class ;
        rdfs:label "Event" ;
        rdfs:comment "Some kind of event with spacetime coordinates" ;
        rdfs:subClassOf schema:Event .    """
    )


@pytest.fixture
def ontology_path():
    return Path("data/ontologies")


@pytest.fixture
def working_directory():
    return None
    # return Path("test/tmp")


@pytest.fixture
def llm_tool(provider, model_name, temperature, llm_base_url):
    llm_tool = LLMTool.create(
        provider=provider,
        model=model_name,
        temperature=temperature,
        base_url=llm_base_url,
    )
    return llm_tool


@pytest.fixture
def tsm_tool(ontology_path, working_directory):
    return FilesystemTripleStoreManager(
        working_directory=working_directory, ontology_path=ontology_path
    )


@pytest.fixture
def om_tool_fname():
    return "test/data/om_tool.json"


@pytest.fixture
def tools(
    ontology_path, working_directory, model_name, temperature, provider, llm_base_url
) -> ToolBox:
    tools: ToolBox = ToolBox(
        llm_base_url=llm_base_url,
        llm_provider=provider,
        working_directory=working_directory,
        ontology_directory=ontology_path,
        model_name=model_name,
        temperature=temperature,
    )
    init_toolbox(tools)
    return tools


@pytest.fixture
def state_chunked_filename():
    return "test/data/state_chunked.json"


@pytest.fixture
def state_chunked(state_chunked_filename):
    return AgentState.load(state_chunked_filename)


@pytest.fixture
def state_onto_selected_filename():
    return "test/data/state_ontology_selected.json"


@pytest.fixture
def state_ontology_selected(state_onto_selected_filename):
    return AgentState.load(state_onto_selected_filename)


@pytest.fixture
def state_ontology_rendered_filename():
    return "test/data/state_onto_rendered.json"


@pytest.fixture
def state_ontology_rendered(state_ontology_rendered_filename):
    return AgentState.load(state_ontology_rendered_filename)


@pytest.fixture
def state_ontology_criticized_filename():
    return "test/data/state_onto_criticized.json"


@pytest.fixture
def state_ontology_criticized(state_ontology_criticized_filename):
    return AgentState.load(state_ontology_criticized_filename)


@pytest.fixture
def state_rendered_facts_filename():
    return "test/data/state_rendered_facts.json"


@pytest.fixture
def state_rendered_facts(state_rendered_facts_filename):
    return AgentState.load(state_rendered_facts_filename)


@pytest.fixture
def state_sublimated_filename():
    return "test/data/state_sublimated.json"


@pytest.fixture
def state_sublimated(state_sublimated_filename):
    return AgentState.load(state_sublimated_filename)


@pytest.fixture
def state_facts_failed_filename():
    return "test/data/state_facts_failed.json"


@pytest.fixture
def state_facts_failed(state_facts_failed_filename):
    return AgentState.load(state_facts_failed_filename)


@pytest.fixture
def state_facts_success_filename():
    return "test/data/state_facts_success.json"


@pytest.fixture
def state_facts_success(state_facts_success_filename):
    return AgentState.load(state_facts_success_filename)


@pytest.fixture
def state_onto_null_filename():
    return "test/data/state_null_ontology_selected.json"


@pytest.fixture
def agent_state_select_ontology_null(state_onto_null_filename):
    return AgentState.load(state_onto_null_filename)


@pytest.fixture
def om_tool(om_tool_fname):
    try:
        return OntologyManager.load(om_tool_fname)
    except (FileNotFoundError, Exception):
        return OntologyManager()


@pytest.fixture
def max_iter():
    return 2


@pytest.fixture
def apple_report():
    r = FileHandle.load(Path("data/json/fin.10Q.apple.json"))
    return {"text": r["text"]}


@pytest.fixture
def random_report():
    return FileHandle.load(Path("data/json/random.json"))


@pytest.fixture
def agent_state_onto_fresh():
    return AgentState.load("test/data/state_onto_addendum.json")


@pytest.fixture
def agent_state_onto_critique_success():
    return AgentState.load("test/data/agent_state.onto.null.critique.success.json")


@pytest.fixture(scope="session")
def neo4j_uri():
    return os.environ.get("NEO4J_URI", "bolt://localhost:7687")


@pytest.fixture(scope="session")
def neo4j_auth():
    return os.environ.get("NEO4J_AUTH", "neo4j/test")


@pytest.fixture(scope="session")
def neo4j_triple_store_manager(neo4j_uri, neo4j_auth):
    if not (neo4j_uri and neo4j_auth):
        pytest.skip("Neo4j not configured in environment.")
    return Neo4jTripleStoreManager(uri=neo4j_uri, auth=neo4j_auth, clean=True)


@pytest.fixture(scope="session")
def fuseki_triple_store_manager():
    uri = os.environ.get("FUSEKI_URI", "http://localhost:3030/test")
    auth = os.environ.get("FUSEKI_AUTH", None)
    if not uri:
        pytest.skip("Fuseki not configured in environment.")
    return FusekiTripleStoreManager(uri=uri, auth=auth, dataset="test", clean=True)


def triple_store_roundtrip(manager, test_ontology):
    ontology = Ontology(graph=test_ontology)
    # Store ontology
    manager.serialize_ontology(ontology)
    # Fetch ontologies
    ontologies = manager.fetch_ontologies()
    # There should be at least one ontology with the correct ontology_id
    assert any(o.ontology_id == "to" for o in ontologies)
    # The ontology graph should have the same number of triples as the input
    assert len(ontologies[0].graph) == len(ontology.graph)


def triple_store_serialize_facts(manager):
    """Test serializing facts (RDF triples) to triple store and retrieving them."""
    # Create test facts
    facts = RDFGraph._from_turtle_str(
        """
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix ex: <http://example.org/test/> .
    @prefix schema: <https://schema.org/> .
    
    ex:Person a rdfs:Class ;
        rdfs:label "Person" ;
        rdfs:comment "A human being" .
    
    ex:John a ex:Person ;
        rdfs:label "John Doe" ;
        schema:name "John Doe" ;
        schema:email "john@example.com" .
    
    ex:Jane a ex:Person ;
        rdfs:label "Jane Smith" ;
        schema:name "Jane Smith" ;
        schema:email "jane@example.com" .
    
    ex:knows a rdf:Property ;
        rdfs:label "knows" ;
        rdfs:comment "Relationship between people who know each other" .
    
    ex:John ex:knows ex:Jane .
    """
    )
    # Verify we have the expected number of triples
    expected_triple_count = len(facts)
    assert expected_triple_count == 15, "Test facts should contain triples"
    # Serialize facts to triple store
    result = manager.serialize_facts(facts)
    assert result is not None, "serialize_facts should return a result"


def triple_store_serialize_empty_facts(manager):
    """Test serializing empty facts graph."""
    # Create empty facts
    empty_facts = RDFGraph()
    # Serialize empty facts - should not raise an error
    result = manager.serialize_facts(empty_facts)
    assert result is not None, (
        "serialize_facts should return a result even for empty graph"
    )
