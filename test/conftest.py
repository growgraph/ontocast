"""Pytest configuration for test suite."""

import logging
import os
import uuid
import warnings
from typing import TYPE_CHECKING, Any, Generator, Optional

import pytest

if TYPE_CHECKING:
    from ontocast.tool.sentence_transformer import SharedSentenceTransformerEmbeddings

from ontocast.config import (
    LLMProvider,
    OpenAIModel,
    QdrantConfig,
)
from ontocast.onto.rdfgraph import RDFGraph
from test.qdrant_util import QdrantSessionTestContext, qdrant_reachable

logger = logging.getLogger(__name__)

# Tests that construct a ToolBox with default LLMConfig (provider=OPENAI) build a
# real ChatOpenAI client, which validates credentials at construction time, not at
# call time. These tests never call the LLM, but need *a* key-shaped value to avoid
# openai.OpenAIError in a clean checkout (no .env). setdefault preserves a real key
# from a local .env or CI secret if one is already set.
os.environ.setdefault("LLM_API_KEY", "sk-test-placeholder-not-a-real-key")


#: Settings that change *what the pipeline does* rather than where it stores
#: things. A leak in any of these invalidates the run with no other symptom --
#: a stray RENDER_MODE=facts skips the entire ontology block while the suite
#: still reports green -- so they are checked, and the ~200 storage/tuning
#: settings are not.
_PIPELINE_MODE_SELECTORS = ("render_mode", "ontology_context_mode", "llm_graph_format")


def _assert_pipeline_config_not_leaked() -> None:
    """Fail fast when the environment overrides a pipeline mode selector.

    The suite is only meaningful against declared defaults. Two things have
    broken that: `pytest-dotenv` loading a developer's live `.env` (it does so
    even with no `env_files` set), and a variable exported in the shell. The
    `-pno:dotenv` entry in `addopts` prevents the first, but a preventive
    measure with no detector fails silently -- and that entry has already been
    mangled once by `toml-sort` reordering the array. This is the detector.
    """
    from ontocast.config.settings import ServerConfig

    effective = ServerConfig()
    leaked = {
        name: (getattr(effective, name), ServerConfig.model_fields[name].default)
        for name in _PIPELINE_MODE_SELECTORS
        if getattr(effective, name) is not ServerConfig.model_fields[name].default
    }
    if leaked:
        detail = ", ".join(
            f"{name.upper()}={found.value!r} (declared default {default.value!r})"
            for name, (found, default) in sorted(leaked.items())
        )
        raise pytest.UsageError(
            f"Pipeline configuration leaked into the test environment: {detail}. "
            "The suite must run against declared defaults. Unset the variable in "
            "your shell, or -- if a dotenv file is being loaded -- check that "
            "'-pno:dotenv' survived in the addopts array of pyproject.toml."
        )


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
        workspace=workspace,
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
    _assert_pipeline_config_not_leaked()
    # Suppress Pydantic deprecation warnings from docling_core (third-party library we cannot modify)
    config.addinivalue_line(
        "filterwarnings",
        "ignore::DeprecationWarning:docling_core",
    )


def pytest_collection_modifyitems(config, items) -> None:
    """Deselect slow and integration tests by default unless selected with -m."""
    if not config.getoption("-m"):
        selected = []
        deselected = []
        for item in items:
            if "slow" in item.keywords or "integration" in item.keywords:
                deselected.append(item)
            else:
                selected.append(item)
        if deselected:
            config.hook.pytest_deselected(items=deselected)
            items[:] = selected


@pytest.fixture
def provider():
    return os.getenv("LLM_PROVIDER", LLMProvider.OPENAI)


@pytest.fixture
def model_name():
    # Returned as-is rather than coerced into OpenAIModel: LLM_MODEL_NAME may
    # legitimately name a model this package has no preset for (a new release,
    # or another vendor behind an OpenAI-compatible base_url).
    return os.getenv("LLM_MODEL_NAME", OpenAIModel.GPT4_O_MINI)


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


@pytest.fixture(scope="session")
def real_embeddings() -> Optional["SharedSentenceTransformerEmbeddings"]:
    """Real local embeddings if available, otherwise None.

    Uses the *configured* chunker model and the process-shared encoder, so this
    fixture holds no second copy of a checkpoint the code under test already
    loaded. Session-scoped so it is built once per test session.
    """
    try:
        from ontocast.config import ChunkConfig
        from ontocast.tool.sentence_transformer import (
            SharedSentenceTransformerEmbeddings,
            get_shared_encoder,
        )

        return SharedSentenceTransformerEmbeddings(
            get_shared_encoder(ChunkConfig().embedding_model), normalize=False
        )
    except ImportError as e:
        logger.error(f"Could not load a local sentence-transformer: {e}")
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

            from ontocast.util.hash import render_text_hash

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
    """Deterministic prose (~10k characters) for the semantic chunker.

    Synthesised rather than read from a corpus file: the chunker assertions are
    round-trip length and word coverage, which are properties of *text*, not of
    any particular document -- and a test that reaches outside `test/` cannot
    run from the published sdist, which ships `/test` and nothing else.

    The vocabulary rotates per sentence and the topic shifts per paragraph, so
    the embedding boundaries the chunker looks for actually exist. A block of
    one repeated sentence would make the semantic split degenerate and the
    coverage assertion vacuous.
    """
    topics = [
        ("catalyst", "perovskite lattice", "annealing", "crystallite"),
        ("plaintiff", "appellate chamber", "admissibility", "remedy"),
        ("liquidity", "deferred revenue", "amortisation", "covenant"),
        ("cohort", "randomisation", "endpoint", "titration"),
        ("segmenter", "embedding", "retrieval", "namespace"),
    ]
    verbs = ("describes", "constrains", "reports", "qualifies", "supersedes")
    paragraphs = []
    for index, terms in enumerate(topics):
        sentences = []
        for step in range(14):
            subject = terms[step % len(terms)]
            obj = terms[(step + 2) % len(terms)]
            verb = verbs[(index + step) % len(verbs)]
            sentences.append(
                f"Section {index + 1} paragraph note {step + 1}: the {subject} "
                f"{verb} the {obj} under measurement conditions {index + 1}.{step + 1}, "
                f"and the recorded value of {100 + index * 10 + step} remains "
                f"within the tolerance stated for this {subject}."
            )
        paragraphs.append(" ".join(sentences))
    text = "\n\n".join(paragraphs)
    assert len(text) >= 10000, f"sample_text shrank to {len(text)} characters"
    return text[:10000]


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
