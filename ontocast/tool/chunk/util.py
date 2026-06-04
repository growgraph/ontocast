import copy
import re
from typing import Any, Iterable, List, Sequence

import numpy as np
from hdbscan import HDBSCAN
from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.embeddings import Embeddings
from sklearn.decomposition import PCA
from umap import UMAP

from ontocast.config import ChunkConfig
from ontocast.tool.chunk.sizing import merge_small_parts

# Regex pattern for splitting text into sentences
# Matches: paragraph breaks (double newlines) OR sentence endings followed by capital letters
SENTENCE_SPLIT_REGEX = r"(?:\n\s*\n+)|(?<=[.!?])\s+(?=[A-Z][a-z])"


def split_proposition_windows(
    text: str,
    max_sentences: int = 2,
    max_windows: int = 16,
) -> list[str]:
    """Split text into short proposition-like windows for retrieval."""
    cleaned = text.strip()
    if not cleaned:
        return []
    if max_sentences <= 0:
        raise ValueError("max_sentences must be >= 1")
    if max_windows <= 0:
        raise ValueError("max_windows must be >= 1")

    # Keep this splitter lightweight and deterministic.
    sentence_parts = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|\n\s*\n+", cleaned)
        if part.strip()
    ]
    if not sentence_parts:
        return [cleaned[:1000]] if cleaned else []

    windows: list[str] = []
    step = max_sentences
    for index in range(0, len(sentence_parts), step):
        window = " ".join(sentence_parts[index : index + max_sentences]).strip()
        if window:
            windows.append(window)
        if len(windows) >= max_windows:
            break
    return windows or [cleaned[:1000]]


class SemanticChunker(BaseDocumentTransformer):
    def __init__(
        self,
        embeddings: Embeddings,
        chunk_config: ChunkConfig,
        sentence_split_regex: str,
    ):
        """Initialize SemanticChunker.

        Args:
            embeddings: Embeddings model for generating sentence embeddings.
            chunk_config: Chunking configuration containing min_size, max_size, etc.
            sentence_split_regex: Regular expression pattern for splitting text into sentences.
        """
        self.embeddings = embeddings
        self.chunk_config = chunk_config
        self.min_size = chunk_config.min_size
        self.max_size = chunk_config.max_size
        self.sentence_split_regex = sentence_split_regex

    def _build_sentence_windows(
        self,
        sentences: List[str],
        window_size: int = 5,
    ) -> List[str]:
        if len(sentences) <= window_size:
            return [" ".join(sentences)]

        windows = []
        for i in range(len(sentences)):
            start = max(0, i - window_size // 2)
            end = min(len(sentences), start + window_size)
            window = " ".join(sentences[start:end])
            windows.append(window)

        return windows

    def _get_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Embeds sentences directly without buffering.

        Since we cluster all sentences together, the clustering algorithm
        naturally captures semantic relationships without needing context windows.
        """
        return np.array(self.embeddings.embed_documents(sentences))

    def _cluster_sentences(
        self, vectors: np.ndarray, sentences: List[str]
    ) -> np.ndarray:
        """Pipeline: PCA -> UMAP -> HDBSCAN with parameters favoring more clusters.

        Uses HDBSCAN hyperparameters tuned to create more clusters, which helps
        ensure chunks respect max_size constraints. Large clusters will be split
        post-processing.

        Args:
            vectors: Embedding vectors for sentences.
            sentences: Original sentence texts for length validation.

        Returns:
            Cluster labels for each sentence.
        """
        # 1. PCA to reduce noise
        pca_dims = min(vectors.shape[0] - 1, 50)
        if pca_dims > 1:
            vectors = PCA(n_components=pca_dims).fit_transform(vectors)

        # 2. UMAP to 5 dimensions
        # n_neighbors=2 captures very local structure for chunking
        reducer = UMAP(n_components=5, n_neighbors=2, min_dist=0.0, metric="cosine")
        reduced_vectors = reducer.fit_transform(vectors)

        # 3. HDBSCAN with parameters favoring more clusters
        # Calculate optimal min_cluster_size based on max_size constraint
        # We want clusters small enough that they can be combined without exceeding max_size
        if len(sentences) > 0:
            avg_sentence_len = sum(len(s) for s in sentences) / len(sentences)
            # Target: clusters should be small enough that 2-3 clusters can fit in max_size
            # This encourages more, smaller clusters
            target_cluster_size = max(2, int(self.max_size / (avg_sentence_len * 2.5)))
            min_cluster_size = min(target_cluster_size, len(sentences) // 3, 10)
            min_cluster_size = max(2, min_cluster_size)  # At least 2, at most 10
        else:
            min_cluster_size = 2

        # Use cluster_selection_epsilon to encourage more splits
        # Higher epsilon = more aggressive splitting = more clusters
        # We use a small epsilon (0.1-0.3) to encourage splits while maintaining semantics
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=1,  # Lower min_samples = more clusters
            metric="euclidean",
            cluster_selection_epsilon=0.1,  # Encourage more splits
            cluster_selection_method="eom",  # Excess of Mass method
        )
        labels = clusterer.fit_predict(reduced_vectors)

        return labels

    def split_text(self, text: str) -> List[str]:
        # Atomic split into sentences - chunks must contain whole sentences
        # Use capturing groups to preserve delimiters
        # Wrap the regex in a capturing group so delimiters are included in split result
        pattern_with_capture = f"({self.sentence_split_regex})"
        parts = re.split(pattern_with_capture, text)

        # Reconstruct sentences with their following delimiters
        # parts alternates: [text1, delimiter1, text2, delimiter2, ..., textN]
        # Handle case where text starts with delimiter (parts[0] empty)
        sentences = []
        delimiters = []  # Track delimiter after each sentence

        # Skip leading empty part if text starts with delimiter
        start_idx = 1 if parts and not parts[0].strip() else 0

        i = start_idx
        while i < len(parts):
            if i % 2 == start_idx % 2:  # Text parts (same parity as start)
                text_part = parts[i].strip()
                if text_part:  # Non-empty text
                    sentences.append(parts[i])  # Keep original (with whitespace)
                    # Get the delimiter that follows (if any)
                    if i + 1 < len(parts):
                        delimiters.append(parts[i + 1])
                    else:
                        delimiters.append("")  # No delimiter after last sentence
            i += 1

        # Filter out empty sentences
        if not sentences:
            return [text] if text.strip() else []

        if len(sentences) <= 1:
            # If single sentence, return it even if it exceeds max_size
            # (we can't split sentences, so we must keep it whole)
            return sentences

        windows = self._build_sentence_windows(sentences, window_size=5)
        vectors = self._get_embeddings(windows)
        labels = self._cluster_sentences(vectors, sentences)

        # Process sentences in original order, grouping consecutive sentences
        # from the same cluster into chunks
        chunks = []
        i = 0
        while i < len(sentences):
            label = labels[i]

            # Collect consecutive sentences with the same label
            cluster_sentences = [sentences[i]]
            cluster_delimiters = [delimiters[i] if i < len(delimiters) else ""]
            i += 1

            while i < len(sentences) and labels[i] == label:
                cluster_sentences.append(sentences[i])
                cluster_delimiters.append(delimiters[i] if i < len(delimiters) else "")
                i += 1

            # Process this cluster
            if label == -1:
                # Noise cluster: each sentence becomes its own chunk
                for idx, sentence in enumerate(cluster_sentences):
                    chunk = sentence
                    if idx < len(cluster_delimiters) and cluster_delimiters[idx]:
                        chunk += cluster_delimiters[idx]
                    chunks.append(chunk)
            else:
                # Regular cluster: group sentences respecting max_size
                cluster_len = sum(len(s) for s in cluster_sentences)
                delimiter_len = sum(len(d) for d in cluster_delimiters)
                total_cluster_len = cluster_len + delimiter_len

                if total_cluster_len <= self.max_size:
                    # Cluster fits in one chunk
                    chunk_parts = []
                    for j, sentence in enumerate(cluster_sentences):
                        chunk_parts.append(sentence)
                        if j < len(cluster_delimiters) and cluster_delimiters[j]:
                            chunk_parts.append(cluster_delimiters[j])
                    chunks.append("".join(chunk_parts))
                else:
                    # Split cluster into multiple chunks
                    current_chunk = []
                    current_delims = []
                    current_len = 0

                    for j, sentence in enumerate(cluster_sentences):
                        sentence_len = len(sentence)
                        delim = (
                            cluster_delimiters[j] if j < len(cluster_delimiters) else ""
                        )
                        delim_len = len(delim)

                        if current_len + sentence_len + delim_len > self.max_size:
                            # Current chunk is full
                            if current_chunk:
                                chunk_parts = []
                                for k, s in enumerate(current_chunk):
                                    chunk_parts.append(s)
                                    if k < len(current_delims) and current_delims[k]:
                                        chunk_parts.append(current_delims[k])
                                chunks.append("".join(chunk_parts))
                            current_chunk = [sentence]
                            current_delims = [delim]
                            current_len = sentence_len + delim_len
                        else:
                            current_chunk.append(sentence)
                            current_delims.append(delim)
                            current_len += sentence_len + delim_len

                    # Add remaining chunk
                    if current_chunk:
                        chunk_parts = []
                        for k, s in enumerate(current_chunk):
                            chunk_parts.append(s)
                            if k < len(current_delims) and current_delims[k]:
                                chunk_parts.append(current_delims[k])
                        chunks.append("".join(chunk_parts))

        return merge_small_parts(
            chunks,
            self.min_size,
            self.max_size,
            separator="",
        )

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        return self.split_documents(list(documents))

    def create_documents(
        self, texts: List[str], metadatas: List[dict] | None = None
    ) -> List[Document]:
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(_metadatas[i])
                documents.append(Document(page_content=chunk, metadata=metadata))
        return documents

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        texts = []
        metadatas = []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas)
