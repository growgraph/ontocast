import importlib.util
import logging
import re
from functools import lru_cache
from typing import Any, Literal

from pydantic import Field

from ontocast.config import ChunkConfig
from ontocast.tool.cache import CHUNKER_CACHE_SUBDIR, Cacher, ToolCacher
from ontocast.tool.chunk.proposition import SENTENCE_SPLIT_REGEX
from ontocast.tool.chunk.sizing import size_bounded_text
from ontocast.tool.onto import Tool
from ontocast.tool.sentence_transformer import (
    SharedSentenceTransformerEmbeddings,
    get_shared_encoder,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _embedding_model_available() -> bool:
    """Whether a local sentence-transformer can be loaded at all.

    This is what :meth:`ChunkerTool.embed_texts` — and therefore
    embedding-based schema detection — needs. It does **not** imply semantic
    chunking is available; see :func:`_semantic_chunking_available`.
    """
    return importlib.util.find_spec("sentence_transformers") is not None


@lru_cache(maxsize=1)
def _semantic_chunking_available() -> bool:
    """Whether the full semantic-chunking stack is importable.

    ``tool.chunk.util`` imports hdbscan, umap and sklearn at module scope, so a
    model alone is not enough. Probed with ``find_spec`` rather than a real
    import so constructing a ChunkerTool does not pay umap's numba warm-up on
    every process start.
    """
    if not _embedding_model_available():
        return False
    return all(
        importlib.util.find_spec(name) is not None
        for name in ("hdbscan", "umap", "sklearn")
    )


class ChunkerTool(Tool):
    """Tool for semantic chunking of documents.

    Falls back to naive chunking if sentence-transformers is not available.
    Includes caching to avoid re-chunking the same text with the same parameters.
    """

    config: ChunkConfig = Field(
        default_factory=ChunkConfig, description="Chunking configuration parameters"
    )
    chunking_mode: Literal["semantic", "naive"] = Field(
        default="semantic",
        description="Chunking mode: semantic (requires sentence-transformers) or naive (fallback)",
    )
    cache: Any = Field(default=None, exclude=True)

    def __init__(
        self,
        chunk_config: ChunkConfig | None = None,
        cache: Cacher | None = None,
        **kwargs: Any,
    ):
        """Initialize the ChunkerTool.

        Args:
            chunk_config: Chunking configuration. If None, uses default ChunkConfig.
            cache: Optional shared Cacher instance. If None, creates a new one.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        # The model itself is process-shared and its construction is locked by
        # get_shared_encoder; all this holds is the per-tool adapter around it.
        self._embeddings: SharedSentenceTransformerEmbeddings | None = None
        self._embeddings_unavailable = False

        # Initialize cache - use shared cacher or create new one
        if cache is not None:
            self.cache = ToolCacher(cache, CHUNKER_CACHE_SUBDIR)
        else:
            # Standalone use (CLI helpers, direct library use): fall back to a
            # private Cacher on the configured/default directory.
            shared_cache = Cacher()
            self.cache = ToolCacher(shared_cache, CHUNKER_CACHE_SUBDIR)

        # Override config if provided
        if chunk_config is not None:
            self.config = chunk_config

        # Probe heavy deps only when semantic mode is requested
        if self.chunking_mode == "semantic" and not _semantic_chunking_available():
            self.chunking_mode = "naive"
            logger.warning(
                "Semantic chunking not available (needs the 'semantic-chunking' "
                "extra: sentence-transformers, hdbscan, umap-learn). "
                "Falling back to naive chunking."
            )

    def embeddings(self) -> SharedSentenceTransformerEmbeddings | None:
        """Embeddings over the process-shared encoder, or ``None`` if unavailable.

        The encoder is shared with retrieval and entity clustering when their
        model names match, so this loads no weights of its own in that case, and
        its inference is serialised against theirs.
        """
        if self._embeddings is not None or self._embeddings_unavailable:
            return self._embeddings
        if not _embedding_model_available():
            self._embeddings_unavailable = True
            return None
        try:
            self._embeddings = SharedSentenceTransformerEmbeddings(
                get_shared_encoder(
                    self.config.embedding_model,
                    feature=(
                        "Semantic chunking and schema detection. Install the "
                        "'semantic-chunking' extra"
                    ),
                ),
                normalize=False,
            )
        except Exception as exc:
            # Record the failure rather than retrying the load on every call:
            # a missing or broken checkpoint does not become available later in
            # the same process.
            logger.error("Failed to initialize chunker embedding model: %s", exc)
            self._embeddings_unavailable = True
            return None
        return self._embeddings

    def embed_texts(self, texts: list[str]) -> list[list[float]] | None:
        """Embed short texts with the chunker's model, or ``None`` if unavailable.

        Exposed so document-type detection can reuse the model already loaded
        for semantic chunking instead of constructing a second one. Returns
        ``None`` -- rather than raising -- when the semantic extras are absent,
        so callers degrade to their deterministic tiers exactly as chunking
        itself degrades to ``naive``.

        Args:
            texts: Short strings to embed (headings or sampled paragraphs).

        Returns:
            One embedding per input, or ``None`` when no model is available.
        """
        if not texts:
            return []
        embeddings = self.embeddings()
        if embeddings is None:
            return None
        try:
            return embeddings.embed_documents(texts)
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("Embedding failed, skipping semantic tier: %s", exc)
            return None

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
        """Chunk a document into semantic segments.

        Args:
            doc: The document text to chunk.

        Returns:
            List of text chunks.
        """
        # Prepare configuration for caching. The "model" key name is kept even
        # though its source moved to ChunkConfig -- the dict is hashed, so
        # renaming it would invalidate every cached chunking for no reason.
        config_dict = {
            "model": self.config.embedding_model,
            "chunking_mode": self.chunking_mode,
            "max_size": self.config.max_size,
            "min_size": self.config.min_size,
        }

        # Check cache first
        cached_result = self.cache.get(doc, config=config_dict)
        if cached_result is not None:
            logger.debug("Cache hit for document chunking")
            return cached_result

        # Perform chunking
        embeddings = None if self.chunking_mode == "naive" else self.embeddings()
        if embeddings is None or not _semantic_chunking_available():
            if self.chunking_mode != "naive":
                logger.warning(
                    "Semantic chunking requested but not available. "
                    "Falling back to naive chunking."
                )
            result = self._naive_chunk(doc)
        else:
            from ontocast.tool.chunk.util import SemanticChunker

            text_splitter = SemanticChunker(
                embeddings=embeddings,
                chunk_config=self.config,
                sentence_split_regex=SENTENCE_SPLIT_REGEX,
            )

            try:
                # SemanticChunker now handles max_size internally
                result_docs = text_splitter.create_documents([doc])
                result = [chunk.page_content for chunk in result_docs]
            except ValueError as exc:
                # Degenerate inputs (too few distinct sentences for the
                # HDBSCAN neighborhood) must not fail chunking outright.
                logger.warning(
                    "Semantic chunking failed (%s); falling back to "
                    "naive chunking for this text.",
                    exc,
                )
                result = self._naive_chunk(doc)

            # Log chunk lengths for debugging
            lens = [len(chunk) for chunk in result]
            logger.info(
                f"Semantic chunking produced {len(result)} chunks with lengths: {lens}"
            )

        # Cache the result
        self.cache.set(doc, result, config=config_dict)
        logger.debug("Cached document chunking result")

        return result
