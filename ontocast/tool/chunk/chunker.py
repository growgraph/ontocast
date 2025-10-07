import logging

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import Field

from ontocast.config import ChunkConfig
from ontocast.tool.chunk.util import SemanticChunker
from ontocast.tool.onto import Tool

logger = logging.getLogger(__name__)


class ChunkerTool(Tool):
    """Tool for semantic chunking of documents."""

    model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        description="HuggingFace model name for embeddings",
    )
    config: ChunkConfig = Field(
        default_factory=ChunkConfig, description="Chunking configuration parameters"
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

    def _init_model(self):
        if self._model is None:
            self._model = HuggingFaceEmbeddings(
                model_name=self.model,
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                encode_kwargs={"normalize_embeddings": False},
            )

    def __call__(self, doc: str) -> list[str]:
        self._init_model()
        documents = [doc]

        text_splitter = SemanticChunker(
            buffer_size=self.config.buffer_size,
            breakpoint_threshold_type=self.config.breakpoint_threshold_type,
            breakpoint_threshold_amount=self.config.breakpoint_threshold_amount,
            embeddings=self._model,
            min_chunk_size=self.config.min_size,
            sentence_split_regex=r"(?:(?:\n{2,}(?=#+))|(?:\n{2,}(?=- ))"
            r"|(?<=[a-z][.?!])\s+(?=\b[A-Z]\w{8,}\b)|(?<!#)(?=#+))",
        )

        def recursive_chunking(docs, stop_flag=False):
            lens = [len(d) for d in docs]
            logger.info(f"chunk lengths: {lens}")
            if all(len(doc) < self.config.max_size for doc in docs) or stop_flag:
                return docs
            else:
                new_docs = []
                for d in docs:
                    if len(d) > self.config.max_size:
                        cdocs_ = text_splitter.create_documents([d])
                        cdocs = [d.page_content for d in cdocs_]
                        if len(cdocs[-1]) < self.config.min_size:
                            cdocs = cdocs[:-2] + [cdocs[-2] + cdocs[-1]]
                        new_docs.extend(cdocs)
                    else:
                        new_docs.append(d)
                stop_flag = len(docs) == len(new_docs)
                return recursive_chunking(new_docs, stop_flag=stop_flag)

        docs = recursive_chunking(documents)
        return docs
