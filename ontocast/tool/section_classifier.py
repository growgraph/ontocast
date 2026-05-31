"""Embedding-based section heading and content classification."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable

from pydantic import Field

from ontocast.onto.section import CANONICAL_SECTION_LABELS
from ontocast.tool.onto import Tool

logger = logging.getLogger(__name__)

EmbedFn = Callable[[list[str]], list[list[float]]]

_HEADING_PROTOTYPES: dict[str, list[str]] = {
    "abstract": ["Abstract", "ABSTRACT", "Executive Summary", "Synopsis"],
    "introduction": [
        "Introduction",
        "1. Introduction",
        "I. Introduction",
        "Overview",
        "Preamble",
        "Foreword",
        "Preface",
        "Motivation",
    ],
    "related_work": [
        "Related Work",
        "Related Literature",
        "Prior Work",
        "Prior Art",
        "Literature Review",
        "Literature Survey",
        "Related Studies",
        "State of the Art",
        "Survey",
    ],
    "background": ["Background", "2. Background"],
    "methods": [
        "Methods",
        "Methodology",
        "Materials and Methods",
        "Experimental Setup",
        "Proposed Method",
        "Approach",
        "Implementation",
        "Design",
        "Procedure",
        "Protocol",
        "Framework",
        "Architecture",
    ],
    "results": [
        "Results",
        "Experimental Results",
        "Findings",
        "Evaluation",
        "Experiments",
        "Ablation Study",
        "Outcomes",
        "Performance",
        "Benchmarks",
    ],
    "discussion": [
        "Discussion",
        "Results and Discussion",
        "Analysis",
        "Interpretation",
        "Observations",
    ],
    "conclusion": [
        "Conclusion",
        "Conclusions",
        "Concluding Remarks",
        "Summary",
        "Final Remarks",
        "Final Thoughts",
        "Takeaways",
        "Wrap-up",
    ],
    "future_work": [
        "Future Work",
        "Future Directions",
        "Future Research",
        "Next Steps",
        "Roadmap",
        "Outlook",
    ],
    "limitations": ["Limitations", "Limitation"],
    "acknowledgements": ["Acknowledgements", "Acknowledgments"],
    "data": ["Data", "Dataset", "Datasets", "Corpus", "Data Collection"],
    "appendix": ["Appendix", "Appendices", "Supplementary Material"],
    "references": ["References", "Bibliography", "Works Cited"],
}

_CONTENT_PROTOTYPES: dict[str, list[str]] = {
    "introduction": [
        "This paper introduces the problem and outlines our contributions.",
        "We motivate the study and summarize the structure of the paper.",
    ],
    "related_work": [
        "Prior studies have explored similar approaches in the literature.",
        "We review existing methods and compare them to our setting.",
    ],
    "background": [
        "We provide background definitions and notation used throughout.",
    ],
    "methods": [
        "We describe the experimental setup and evaluation protocol.",
        "Our model architecture and training procedure are as follows.",
    ],
    "results": [
        "Table 1 shows accuracy improved compared to the baseline.",
        "We report quantitative metrics on the benchmark dataset.",
    ],
    "discussion": [
        "These findings suggest several implications for future research.",
        "We interpret the results and discuss possible explanations.",
    ],
    "conclusion": [
        "In conclusion, we demonstrated that the proposed approach is effective.",
        "We summarize our contributions and outline limitations.",
    ],
    "future_work": [
        "Future work will extend the model to additional domains.",
    ],
    "limitations": [
        "Our study is limited by dataset size and annotation quality.",
    ],
    "abstract": [
        "We present a novel method and evaluate it on standard benchmarks.",
    ],
    "acknowledgements": [
        "We thank the reviewers and funding agencies for their support.",
    ],
    "data": [
        "The dataset contains labeled examples for training and evaluation.",
        "We describe the corpus and data collection procedure.",
    ],
    "appendix": [
        "Additional tables and figures are provided in the appendix.",
    ],
    "references": [
        "References are listed in alphabetical order below.",
    ],
}

_DEFAULT_HEADING_THRESHOLD = 0.65
_DEFAULT_CONTENT_THRESHOLD = 0.45
_LOW_CONFIDENCE_HEADING_THRESHOLD = 0.45
_CONTENT_SNIPPET_CHARS = 300


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise ValueError("Cannot average empty vector list")
    dim = len(vectors[0])
    sums = [0.0] * dim
    for vector in vectors:
        for index, value in enumerate(vector):
            sums[index] += value
    count = float(len(vectors))
    return [value / count for value in sums]


class SectionClassifierTool(Tool):
    """Classify section headings and chunk snippets via embedding prototypes."""

    heading_threshold: float = Field(default=_DEFAULT_HEADING_THRESHOLD)
    content_threshold: float = Field(default=_DEFAULT_CONTENT_THRESHOLD)
    low_confidence_heading_threshold: float = Field(
        default=_LOW_CONFIDENCE_HEADING_THRESHOLD
    )

    _heading_centroids: dict[str, list[float]] | None = None
    _content_centroids: dict[str, list[float]] | None = None

    def _ensure_heading_centroids(self, embed_fn: EmbedFn) -> dict[str, list[float]]:
        if self._heading_centroids is not None:
            return self._heading_centroids
        centroids: dict[str, list[float]] = {}
        for label, phrases in _HEADING_PROTOTYPES.items():
            vectors = embed_fn(phrases)
            if len(vectors) != len(phrases):
                raise ValueError(
                    f"Embedding provider returned {len(vectors)} vectors "
                    f"for {len(phrases)} heading prototypes"
                )
            centroids[label] = _mean_vector(vectors)
        self._heading_centroids = centroids
        return centroids

    def _ensure_content_centroids(self, embed_fn: EmbedFn) -> dict[str, list[float]]:
        if self._content_centroids is not None:
            return self._content_centroids
        centroids: dict[str, list[float]] = {}
        for label, phrases in _CONTENT_PROTOTYPES.items():
            vectors = embed_fn(phrases)
            if len(vectors) != len(phrases):
                raise ValueError(
                    f"Embedding provider returned {len(vectors)} vectors "
                    f"for {len(phrases)} content prototypes"
                )
            centroids[label] = _mean_vector(vectors)
        self._content_centroids = centroids
        return centroids

    def _best_label(
        self,
        vector: list[float],
        centroids: dict[str, list[float]],
        allowed_labels: set[str] | None,
        threshold: float,
    ) -> tuple[str | None, float]:
        best_label: str | None = None
        best_score = -1.0
        for label, centroid in centroids.items():
            if allowed_labels is not None and label not in allowed_labels:
                continue
            score = _cosine_similarity(vector, centroid)
            if score > best_score:
                best_score = score
                best_label = label
        if best_label is None or best_score < threshold:
            return None, best_score
        return best_label, best_score

    def classify_heading(
        self,
        heading: str,
        embed_fn: EmbedFn,
        *,
        threshold: float | None = None,
    ) -> tuple[str | None, float]:
        """Classify a heading string; returns (label, best_cosine_score)."""
        text = heading.strip()
        if not text:
            return None, 0.0
        centroids = self._ensure_heading_centroids(embed_fn)
        vectors = embed_fn([text])
        if not vectors:
            return None, 0.0
        use_threshold = self.heading_threshold if threshold is None else threshold
        return self._best_label(vectors[0], centroids, None, use_threshold)

    def classify_heading_with_confidence(
        self,
        heading: str,
        embed_fn: EmbedFn,
    ) -> tuple[str | None, float, bool]:
        """Return (label, score, needs_llm_fallback)."""
        label, score = self.classify_heading(
            heading,
            embed_fn,
            threshold=self.low_confidence_heading_threshold,
        )
        if label is not None and score >= self.heading_threshold:
            return label, score, False
        if label is not None and score >= self.low_confidence_heading_threshold:
            return label, score, True
        return None, score, True

    def classify_chunk(
        self,
        text: str,
        embed_fn: EmbedFn,
        allowed_labels: list[str] | None = None,
        *,
        threshold: float | None = None,
    ) -> str | None:
        """Classify a content chunk snippet when no section headings exist."""
        snippet = text.strip()[:_CONTENT_SNIPPET_CHARS]
        if not snippet:
            return None
        allowed: set[str] | None = None
        if allowed_labels:
            allowed = {
                label.strip().lower() for label in allowed_labels if label.strip()
            }
        centroids = self._ensure_content_centroids(embed_fn)
        vectors = embed_fn([snippet])
        if not vectors:
            return None
        use_threshold = self.content_threshold if threshold is None else threshold
        label, _score = self._best_label(vectors[0], centroids, allowed, use_threshold)
        return label

    @staticmethod
    def normalise_llm_label(raw: str | None) -> str | None:
        if raw is None:
            return None
        cleaned = raw.strip().lower().replace(" ", "_").replace("-", "_")
        if cleaned in CANONICAL_SECTION_LABELS:
            return cleaned
        return None
