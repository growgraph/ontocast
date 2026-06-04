import importlib
import logging
import re
import threading
from typing import Any, Literal

from pydantic import Field

from ontocast.config import ChunkConfig
from ontocast.tool.cache import Cacher, ToolCacher
from ontocast.tool.chunk.sizing import size_bounded_text
from ontocast.tool.chunk.util import SENTENCE_SPLIT_REGEX, SemanticChunker
from ontocast.tool.onto import Tool

logger = logging.getLogger(__name__)

# Optional imports for semantic chunking
torch_module: Any | None = None
embedding_model_cls: Any | None = None
try:
    torch_module = importlib.import_module("torch")
    langchain_huggingface_module = importlib.import_module("langchain_huggingface")
    embedding_model_cls = getattr(
        langchain_huggingface_module, "HuggingFaceEmbeddings", None
    )
    SEMANTIC_CHUNKING_AVAILABLE = embedding_model_cls is not None
except ImportError:
    SEMANTIC_CHUNKING_AVAILABLE = False


class ChunkerTool(Tool):
    """Tool for semantic chunking of documents.

    Falls back to naive chunking if sentence-transformers is not available.
    Includes caching to avoid re-chunking the same text with the same parameters.
    """

    model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        description="HuggingFace model name for embeddings",
    )
    config: ChunkConfig = Field(
        default_factory=ChunkConfig, description="Chunking configuration parameters"
    )
    chunking_mode: Literal["semantic", "naive"] = Field(
        default="semantic" if SEMANTIC_CHUNKING_AVAILABLE else "naive",
        description="Chunking mode: semantic (requires sentence-transformers) or naive (fallback)",
    )
    cache: Any = Field(default=None, exclude=True)

    def __init__(
        self,
        chunk_config: ChunkConfig | None = None,
        cache: Cacher | None = None,
        **kwargs,
    ):
        """Initialize the ChunkerTool.

        Args:
            chunk_config: Chunking configuration. If None, uses default ChunkConfig.
            cache: Optional shared Cacher instance. If None, creates a new one.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._model: Any | None = None
        self._model_lock = threading.Lock()  # Lock for thread-safe model initialization

        # Initialize cache - use shared cacher or create new one
        if cache is not None:
            self.cache = ToolCacher(cache, "chunker")
        else:
            # Fallback for backward compatibility
            shared_cache = Cacher()
            self.cache = ToolCacher(shared_cache, "chunker")

        # Override config if provided
        if chunk_config is not None:
            self.config = chunk_config

        # Override chunking mode if semantic chunking is not available
        if not SEMANTIC_CHUNKING_AVAILABLE and self.chunking_mode == "semantic":
            self.chunking_mode = "naive"
            logger.warning(
                "Semantic chunking not available (sentence-transformers not installed). "
                "Falling back to naive chunking."
            )

    def _init_model(self):
        """Initialize the embedding model in a thread-safe manner.

        Uses double-checked locking pattern to ensure the model is only
        initialized once, even when called concurrently from multiple threads.
        """
        # Fast path: if model already initialized, return immediately
        if self._model is not None:
            return

        # Acquire lock for thread-safe initialization
        with self._model_lock:
            # Double-check: another thread might have initialized it while we waited
            if self._model is None and SEMANTIC_CHUNKING_AVAILABLE:
                if embedding_model_cls is not None:
                    try:
                        self._model = embedding_model_cls(
                            model_name=self.model,
                            model_kwargs={
                                "device": "cuda"
                                if torch_module is not None
                                and torch_module.cuda.is_available()
                                else "cpu"
                            },
                            encode_kwargs={"normalize_embeddings": False},
                        )
                        logger.debug(f"Initialized embedding model: {self.model}")
                    except Exception as e:
                        logger.error(f"Failed to initialize embedding model: {e}")
                        # Set to a sentinel value to prevent repeated failed attempts
                        self._model = None

    def naive_split(self, doc: str) -> list[str]:
        """Split text by paragraph/sentence boundaries up to ``max_size``.

        Unlike :meth:`_naive_chunk`, does not enforce ``min_size`` filtering.
        """
        paragraphs = re.split(r"\n\s*\n", doc.strip())

        chunks: list[str] = []
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if (
                current_chunk
                and len(current_chunk) + len(paragraph) + 2 > self.config.max_size
            ):
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

            if len(current_chunk) > self.config.max_size:
                if len(current_chunk) - len(paragraph) - 2 > 0:
                    prev_chunk = current_chunk[
                        : len(current_chunk) - len(paragraph) - 2
                    ].strip()
                    if prev_chunk:
                        chunks.append(prev_chunk)

                sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                temp_chunk = ""

                for sentence in sentences:
                    if len(temp_chunk) + len(sentence) + 1 > self.config.max_size:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = sentence
                    else:
                        if temp_chunk:
                            temp_chunk += " " + sentence
                        else:
                            temp_chunk = sentence

                current_chunk = temp_chunk

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def size_text(self, doc: str) -> list[str]:
        """Split ``doc`` to respect ``min_size`` / ``max_size`` using naive boundaries."""
        return size_bounded_text(doc, self.config, self.naive_split)

    def _naive_chunk(self, doc: str) -> list[str]:
        """Naive chunking fallback when semantic chunking is not available.

        Args:
            doc: The document text to chunk.

        Returns:
            List of text chunks.
        """
        chunks = self.size_text(doc)

        logger.info(f"Naive chunking produced {len(chunks)} chunks")
        return chunks

    def __call__(self, doc: str) -> list[str]:
        """Chunk the document using either semantic or naive chunking.

        Args:
            doc: The document text to chunk.

        Returns:
            List of text chunks.
        """
        # Prepare configuration for caching
        config_dict = {
            "model": self.model,
            "chunking_mode": self.chunking_mode,
            "max_size": self.config.max_size,
            "min_size": self.config.min_size,
            "breakpoint_threshold_type": self.config.breakpoint_threshold_type,
            "breakpoint_threshold_amount": self.config.breakpoint_threshold_amount,
        }

        # Check cache first
        cached_result = self.cache.get(doc, config=config_dict)
        if cached_result is not None:
            logger.debug("Cache hit for document chunking")
            return cached_result

        # Perform chunking
        if self.chunking_mode == "naive":
            result = self._naive_chunk(doc)
        else:
            # Semantic chunking (requires sentence-transformers)
            if not SEMANTIC_CHUNKING_AVAILABLE:
                logger.warning(
                    "Semantic chunking requested but not available. Falling back to naive chunking."
                )
                result = self._naive_chunk(doc)
            else:
                self._init_model()
                documents = [doc]

                if self._model is None:
                    logger.warning(
                        "Model not initialized. Falling back to naive chunking."
                    )
                    result = self._naive_chunk(doc)
                elif SemanticChunker is None:
                    logger.warning(
                        "SemanticChunker not available. Falling back to naive chunking."
                    )
                    result = self._naive_chunk(doc)
                else:
                    text_splitter = SemanticChunker(
                        embeddings=self._model,
                        chunk_config=self.config,
                        sentence_split_regex=SENTENCE_SPLIT_REGEX,
                    )

                    # SemanticChunker now handles max_size internally
                    result_docs = text_splitter.create_documents(documents)
                    result = [doc.page_content for doc in result_docs]

                    # Log chunk lengths for debugging
                    lens = [len(chunk) for chunk in result]
                    logger.info(
                        f"Semantic chunking produced {len(result)} chunks with lengths: {lens}"
                    )

        # Cache the result
        self.cache.set(doc, result, config=config_dict)
        logger.debug("Cached document chunking result")

        return result
