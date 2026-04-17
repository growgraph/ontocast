"""Pytest configuration for test suite."""

import importlib
import json
import logging
import os
import uuid
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator, Optional

import pytest
from suthing import FileHandle

if TYPE_CHECKING:
    from langchain_huggingface import HuggingFaceEmbeddings

from ontocast.config import (
    Config,
    LLMConfig,
    LLMProvider,
    OpenAIModel,
    PathConfig,
    QdrantConfig,
    ToolConfig,
)
from ontocast.onto.constants import DEFAULT_DOMAIN
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.state import AgentState
from ontocast.tool import (
    FilesystemTripleStoreManager,
    LLMTool,
    OntologyManager,
)
from ontocast.tool.triple_manager.mock import (
    MockFusekiTripleStoreManager,
    MockNeo4jTripleStoreManager,
)
from ontocast.toolbox import ToolBox
from test.qdrant_util import QdrantSessionTestContext, qdrant_reachable

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def qdrant_session_test_context(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[QdrantSessionTestContext, Any, None]:
    """Unique Qdrant collections for the pytest session; deleted in finalizer."""
    from qdrant_client import QdrantClient

    base = QdrantConfig()
    if base.uri is None:
        pytest.skip("QDRANT_URI not configured")
    if not qdrant_reachable(uri=base.uri, api_key=base.api_key):
        pytest.skip(f"Qdrant not reachable at {base.uri}")

    run_id = uuid.uuid4().hex[:8]
    qcfg = base.model_copy(
        update={
            "ontology_collection": f"ontocast_pytest_{run_id}_ontologies",
            "facts_collection": f"ontocast_pytest_{run_id}_facts",
        }
    )
    workspace = tmp_path_factory.mktemp("qdrant_smoke_workspace")
    ontology_dir = workspace / "ontologies"
    ontology_dir.mkdir()

    ctx = QdrantSessionTestContext(
        qdrant_config=qcfg,
        working_directory=workspace,
        ontology_directory=ontology_dir,
    )

    yield ctx

    client = QdrantClient(
        url=qcfg.uri,
        api_key=qcfg.api_key,
        grpc_port=qcfg.grpc_port,
        prefer_grpc=qcfg.use_grpc,
    )
    for name in (qcfg.ontology_collection, qcfg.facts_collection):
        if name and client.collection_exists(collection_name=name):
            client.delete_collection(collection_name=name)


# Suppress deprecation warnings from third-party libraries that we cannot control
# Note: We adapt to new conventions where possible (e.g., using pyld directly for JSON-LD
# instead of rdflib's deprecated ConjunctiveGraph). These suppressions are only for
# warnings from external libraries that we cannot modify.

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=".*@model_validator.*mode='after'.*",
    module="docling_core",
)


def pytest_configure(config):
    """Configure pytest to suppress known deprecation warnings from third-party libraries."""
    # Suppress Pydantic deprecation warnings from docling_core (third-party library we cannot modify)
    config.addinivalue_line(
        "filterwarnings",
        "ignore::DeprecationWarning:docling_core",
    )


@pytest.fixture
def current_domain():
    return os.getenv("CURRENT_DOMAIN", DEFAULT_DOMAIN)


@pytest.fixture
def llm_base_url():
    return os.getenv("LLM_BASE_URL", None)


@pytest.fixture
def provider():
    return os.getenv("LLM_PROVIDER", LLMProvider.OPENAI)


@pytest.fixture
def model_name():
    return OpenAIModel(os.getenv("LLM_MODEL_NAME", OpenAIModel.GPT4_O_MINI))


@pytest.fixture
def temperature():
    return 0.1


@pytest.fixture
def test_ontology():
    from ontocast.onto.ontology import Ontology

    graph = RDFGraph._from_turtle_str(
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
    return Ontology(graph=graph)


@pytest.fixture
def ontology_path():
    return Path("data/ontologies")


@pytest.fixture
def working_directory():
    return None
    # return Path("test/tmp")


@pytest.fixture
def llm_tool(provider, model_name, temperature, llm_base_url):
    config = LLMConfig(
        provider=LLMProvider(provider),
        model_name=model_name,
        temperature=temperature,
        base_url=llm_base_url,
    )
    llm_tool = LLMTool.create(config=config)
    return llm_tool


@pytest.fixture
def tsm_tool(ontology_path, working_directory):
    return FilesystemTripleStoreManager(
        working_directory=working_directory, ontology_path=ontology_path
    )


@pytest.fixture
def tools(
    ontology_path,
    working_directory,
    model_name,
    temperature,
    provider,
    llm_base_url,
    om_tool_fname,
) -> ToolBox:
    # Create LLM config
    llm_config = LLMConfig(
        provider=LLMProvider(provider),
        model_name=model_name,
        temperature=temperature,
        base_url=llm_base_url,
    )

    # Create path config
    path_config = PathConfig(
        working_directory=working_directory,
        ontology_directory=ontology_path,
    )

    # Create tool config
    tool_config = ToolConfig(
        llm_config=llm_config,
        path_config=path_config,
    )

    # Create main config
    config = Config(tool_config=tool_config)

    tools: ToolBox = ToolBox(config=config)
    import asyncio

    asyncio.run(tools.initialize())

    # Load ontologies from JSON file if it exists (using Pydantic's load method)
    json_path = Path(om_tool_fname)
    if json_path.exists():
        try:
            loaded_om = OntologyManager.load(json_path)
            # Merge loaded ontologies into the toolbox's ontology manager
            for iri, versions in loaded_om.ontology_versions.items():
                for ontology in versions:
                    tools.ontology_manager.add_ontology(ontology)
        except Exception:
            # Silently fail if JSON loading fails
            pass

    return tools


@pytest.fixture
def state_chunked(state_chunked_filename):
    return AgentState.load(state_chunked_filename)


@pytest.fixture
def state_ontology_selected(state_onto_selected_filename):
    return AgentState.load(state_onto_selected_filename)


@pytest.fixture
def state_ontology_rendered(state_ontology_rendered_filename):
    return AgentState.load(state_ontology_rendered_filename)


@pytest.fixture
def state_ontology_criticized(state_ontology_criticized_filename):
    return AgentState.load(state_ontology_criticized_filename)


@pytest.fixture
def state_rendered_facts(state_rendered_facts_filename):
    return AgentState.load(state_rendered_facts_filename)


@pytest.fixture
def state_facts_failed(state_facts_failed_filename):
    return AgentState.load(state_facts_failed_filename)


@pytest.fixture
def state_facts_success(state_facts_success_filename):
    return AgentState.load(state_facts_success_filename)


@pytest.fixture
def agent_state_ontology_null(state_onto_null_filename):
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


@pytest.fixture(scope="session")
def neo4j_uri():
    return os.environ.get("NEO4J_URI", "bolt://localhost:7687")


@pytest.fixture(scope="session")
def neo4j_auth():
    return os.environ.get("NEO4J_AUTH", "neo4j/test")


@pytest.fixture(scope="session")
def neo4j_triple_store_manager(neo4j_uri, neo4j_auth):
    """Mock Neo4j triple store manager for testing."""
    return MockNeo4jTripleStoreManager(uri=neo4j_uri, auth=neo4j_auth, clean=True)


@pytest.fixture(scope="session")
def fuseki_triple_store_manager():
    """Mock Fuseki triple store manager for testing."""
    uri = os.environ.get("FUSEKI_URI", "http://localhost:3030")
    auth = os.environ.get("FUSEKI_AUTH", None)
    if auth and "/" in auth:
        auth = tuple(auth.split("/", 1))
    return MockFusekiTripleStoreManager(uri=uri, auth=auth, dataset="test", clean=True)


@pytest.fixture(scope="session")
def real_embeddings() -> Optional["HuggingFaceEmbeddings"]:
    """Fixture providing real HuggingFace embeddings if available, otherwise None.

    Uses the same model as in split_chunks.py for consistency.
    Session-scoped so the model is loaded only once per test session and reused.
    """

    try:
        torch = importlib.import_module("torch")
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_kwargs={
                "device": "cuda"
                if torch is not None and torch.cuda.is_available()
                else "cpu"
            },
            encode_kwargs={"normalize_embeddings": False},
        )
        return embeddings
    except ImportError as e:
        logger.error(f"Could not import HuggingFaceEmbeddings: {e}")
        return None
    except Exception:
        return None


@pytest.fixture(scope="session")
def mock_embeddings():
    try:
        from langchain_core.embeddings import Embeddings
    except ImportError as e:
        logger.error(f"Could not import Embeddings: {e}")

    class MockEmbeddings(Embeddings):
        """Mock embeddings for testing.

        Returns deterministic embeddings based on text content.
        """

        def __init__(self, embedding_dim: int = 384):
            """Initialize mock embeddings.

            Args:
                embedding_dim: Dimension of the embedding vectors. Defaults to 384.
            """
            self.embedding_dim = embedding_dim
            # Simple hash-based embedding for deterministic results
            self._cache: dict[str, list[float]] = {}

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            """Generate embeddings for a list of texts."""
            return [self.embed_query(text) for text in texts]

        def embed_query(self, text: str) -> list[float]:
            """Generate an embedding for a single text."""
            if text in self._cache:
                return self._cache[text]

            from ontocast.util import render_text_hash

            hash_int = int(render_text_hash(text, digits=None), 16)

            embedding = []
            for i in range(self.embedding_dim):
                val = (hash_int + i * 17) % 1000
                embedding.append((val / 1000.0) - 0.5)

            self._cache[text] = embedding
            return embedding

    return MockEmbeddings()


@pytest.fixture(scope="session")
def embeddings(real_embeddings, mock_embeddings):
    """Fixture providing embeddings - prefers real embeddings, falls back to mock.

    Session-scoped so the model is loaded only once per test session.
    """
    if real_embeddings is not None:
        return real_embeddings
    return mock_embeddings


@pytest.fixture
def sample_text():
    """Fixture providing realistic sample text (~10k characters) from clinical trial JSON."""
    json_file = (
        Path(__file__).parent.parent
        / "data"
        / "json"
        / "clinical.trials.NCT01239745.json"
    )
    if json_file.exists():
        data = json.load(open(json_file))

        def json_to_md(data, depth=1):
            md = []
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (str, int, float, bool, type(None))):
                        md.append(f"{key}: {value}\n")
                    elif isinstance(value, dict):
                        md.append(f"{key}:\n")
                        md.extend(json_to_md(value, depth + 1))
                    elif isinstance(value, list):
                        md.append(f"{key}:\n")
                        for item in value:
                            if isinstance(item, (str, int, float, bool, type(None))):
                                md.append(f"  - {item}\n")
                            else:
                                md.extend(json_to_md(item, depth + 1))
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, (str, int, float, bool, type(None))):
                        md.append(f"- {item}\n")
                    else:
                        md.extend(json_to_md(item, depth))
            return md

        text_lines = json_to_md(data)
        text = "".join(text_lines)
        return text[:10000]

    # Fallback
    return (
        "This is the first sentence. "
        "This is the second sentence. "
        "This is the third sentence. "
        "This is the fourth sentence. "
        "This is the fifth sentence. "
        "This is the sixth sentence. "
        "This is the seventh sentence. "
        "This is the eighth sentence. "
        "This is the ninth sentence. "
        "This is the tenth sentence."
    ) * 100


@pytest.fixture
def long_text():
    """Fixture providing longer text for testing min/max size constraints."""
    paragraphs = []
    for i in range(5):
        sentences = []
        for j in range(10):
            sentences.append(
                f"This is paragraph {i + 1}, sentence {j + 1}. "
                f"It contains some content to make it longer. "
                f"Here is more text to ensure we have enough characters."
            )
        paragraphs.append(" ".join(sentences))
    return "\n\n".join(paragraphs)


# --- Aggregator test fixtures (used by test_aggregator.py) ---


@pytest.fixture
def normalizer():
    """EntityNormalizer instance for aggregator tests."""
    from ontocast.tool.agg.normalizer import EntityNormalizer

    return EntityNormalizer()


@pytest.fixture
def cluster_representative_selector():
    """ClusterRepresentativeSelector instance for aggregator tests."""
    from ontocast.tool.agg.clustering import ClusterRepresentativeSelector

    return ClusterRepresentativeSelector()


@pytest.fixture
def uri_builder():
    """URIBuilder instance for aggregator tests."""
    from ontocast.tool.agg.uri_builder import URIBuilder

    return URIBuilder()


@pytest.fixture
def graph_rewriter():
    """GraphRewriter instance for aggregator tests (add_sameas_links=True)."""
    from ontocast.tool.agg.rewriter import GraphRewriter

    return GraphRewriter(add_sameas_links=False)


def triple_store_roundtrip(manager, test_ontology):
    # test_ontology is already an Ontology object, use it directly
    ontology = test_ontology
    # Store ontology
    manager.serialize(ontology)
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
    result = manager.serialize(facts)
    assert result is not None, "serialize should return a result"


def triple_store_serialize_empty_facts(manager):
    """Test serializing empty facts graph."""
    # Create empty facts
    empty_facts = RDFGraph()
    # Serialize empty facts - should not raise an error
    result = manager.serialize(empty_facts)
    assert result is not None, "serialize should return a result even for empty graph"
