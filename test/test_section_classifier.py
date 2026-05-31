"""Tests for embedding-based section classification."""

from ontocast.tool.section_classifier import SectionClassifierTool


def _mock_embed(texts: list[str]) -> list[list[float]]:
    """Map texts to axis-aligned vectors so prototype centroids align by section family."""

    def vector_for(text: str) -> list[float]:
        lowered = text.lower()
        if any(
            token in lowered
            for token in (
                "result",
                "finding",
                "evaluation",
                "experiment",
                "ablation",
            )
        ):
            return [1.0, 0.0, 0.0]
        if any(
            token in lowered
            for token in ("method", "approach", "setup", "implementation")
        ):
            return [0.0, 1.0, 0.0]
        if any(token in lowered for token in ("conclusion", "summary", "remark")):
            return [0.0, 0.0, 1.0]
        return [0.1, 0.1, 0.1]

    return [vector_for(text) for text in texts]


def test_classify_heading_returns_label_for_similar_prototype() -> None:
    tool = SectionClassifierTool(heading_threshold=0.01)
    label, score = tool.classify_heading("Experimental Results", _mock_embed)
    assert label == "results"
    assert score > 0.0


def test_classify_chunk_respects_allowed_labels() -> None:
    tool = SectionClassifierTool(content_threshold=0.01)
    text = "Table 1 shows accuracy improved compared to the baseline on the test set."
    label = tool.classify_chunk(text, _mock_embed, allowed_labels=["results"])
    assert label in ("results", None)


def test_normalise_llm_label() -> None:
    assert SectionClassifierTool.normalise_llm_label("Results") == "results"
    assert SectionClassifierTool.normalise_llm_label("not_a_section") is None


def test_classify_heading_new_labels() -> None:
    tool = SectionClassifierTool(heading_threshold=0.01)

    def embed_with_axes(texts: list[str]) -> list[list[float]]:
        def vector_for(text: str) -> list[float]:
            lowered = text.lower()
            if any(token in lowered for token in ("data", "dataset", "corpus")):
                return [0.0, 0.0, 0.0, 1.0]
            if any(token in lowered for token in ("appendix", "supplement")):
                return [0.0, 0.0, 1.0, 0.0]
            if any(
                token in lowered for token in ("reference", "bibliography", "cited")
            ):
                return [0.0, 1.0, 0.0, 0.0]
            return [0.1, 0.1, 0.1, 0.1]

        return [vector_for(text) for text in texts]

    data_label, _ = tool.classify_heading("Dataset", embed_with_axes)
    appendix_label, _ = tool.classify_heading("Appendix", embed_with_axes)
    refs_label, _ = tool.classify_heading("Bibliography", embed_with_axes)
    assert data_label == "data"
    assert appendix_label == "appendix"
    assert refs_label == "references"
