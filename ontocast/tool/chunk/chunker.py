import logging
import re
from typing import Literal

from pydantic import Field

from ontocast.config import ChunkConfig
from ontocast.tool.onto import Tool

try:
    import torch  # type: ignore
    from langchain_huggingface import HuggingFaceEmbeddings

    from ontocast.tool.chunk.util import SemanticChunker

    SEMANTIC_CHUNKING_AVAILABLE = True
except ImportError:
    HuggingFaceEmbeddings = None  # type: ignore
    torch = None  # type: ignore
    SemanticChunker = None  # type: ignore
    SEMANTIC_CHUNKING_AVAILABLE = False

logger = logging.getLogger(__name__)


class ChunkerTool(Tool):
    """Tool for semantic chunking of documents.

    Falls back to naive chunking if sentence-transformers is not available.
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

    def __init__(
        self,
        chunk_config: ChunkConfig | None = None,
        **kwargs,
    ):
        """Initialize the ChunkerTool.

        Args:
            chunk_config: Chunking configuration. If None, uses default ChunkConfig.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._model = None

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
        if self._model is None and SEMANTIC_CHUNKING_AVAILABLE:
            if HuggingFaceEmbeddings is not None:  # type: ignore
                self._model = HuggingFaceEmbeddings(
                    model_name=self.model,
                    model_kwargs={
                        "device": "cuda"
                        if torch is not None and torch.cuda.is_available()
                        else "cpu"
                    },
                    encode_kwargs={"normalize_embeddings": False},
                )

    def _naive_chunk(self, doc: str) -> list[str]:
        """Naive chunking fallback when semantic chunking is not available.

                Args:
                    doc: The document text to chunk.
        git
                Returns:
                    List of text chunks.
        """
        # Split by paragraphs first (double newlines)
        paragraphs = re.split(r"\n\s*\n", doc.strip())

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # If adding this paragraph would exceed max_size, start a new chunk
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

            # If a single paragraph is too large, split it by sentences
            if len(current_chunk) > self.config.max_size:
                # Save the previous chunk if it exists
                if len(current_chunk) - len(paragraph) - 2 > 0:
                    prev_chunk = current_chunk[
                        : len(current_chunk) - len(paragraph) - 2
                    ].strip()
                    if prev_chunk:
                        chunks.append(prev_chunk)

                # Split the large paragraph by sentences
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

        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        # Filter out chunks that are too small
        chunks = [chunk for chunk in chunks if len(chunk) >= self.config.min_size]

        logger.info(f"Naive chunking produced {len(chunks)} chunks")
        return chunks

    def __call__(self, doc: str) -> list[str]:
        """Chunk the document using either semantic or naive chunking.

        Args:
            doc: The document text to chunk.

        Returns:
            List of text chunks.
        """
        if self.chunking_mode == "naive":
            return self._naive_chunk(doc)

        # Semantic chunking (requires sentence-transformers)
        if not SEMANTIC_CHUNKING_AVAILABLE:
            logger.warning(
                "Semantic chunking requested but not available. Falling back to naive chunking."
            )
            return self._naive_chunk(doc)

        self._init_model()
        documents = [doc]

        if self._model is None:
            logger.warning("Model not initialized. Falling back to naive chunking.")
            return self._naive_chunk(doc)

        if SemanticChunker is None:  # type: ignore
            logger.warning(
                "SemanticChunker not available. Falling back to naive chunking."
            )
            return self._naive_chunk(doc)

        text_splitter = SemanticChunker(
            buffer_size=self.config.buffer_size,
            breakpoint_threshold_type=self.config.breakpoint_threshold_type,
            breakpoint_threshold_amount=self.config.breakpoint_threshold_amount,
            embeddings=self._model,
            min_chunk_size=self.config.min_size,
            sentence_split_regex=r"(?:(?:\n{2,}(?=#+))|(?:\n{2,}(?=- ))"
            r"|(?<=[a-z][.?!])\s+(?=\b[A-Z]\w{8,}\b)|(?<!#)(?=#+))",
        )

        def recursive_chunking(docs_list: list[str], stop_flag=False):
            lens = [len(d) for d in docs_list]
            logger.info(f"chunk lengths: {lens}")
            if all(len(d) < self.config.max_size for d in docs_list) or stop_flag:
                return docs_list
            else:
                new_docs = []
                for d in docs_list:
                    if len(d) > self.config.max_size:
                        cdocs_ = text_splitter.create_documents([d])
                        cdocs = [d.page_content for d in cdocs_]
                        if len(cdocs[-1]) < self.config.min_size:
                            cdocs = cdocs[:-2] + [cdocs[-2] + cdocs[-1]]
                        new_docs.extend(cdocs)
                    else:
                        new_docs.append(d)
                stop_flag = len(docs_list) == len(new_docs)
                return recursive_chunking(new_docs, stop_flag=stop_flag)

        docs = recursive_chunking(documents)
        return docs
