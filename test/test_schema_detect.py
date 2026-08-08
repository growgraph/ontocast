"""Tests for document-type schema detection.

Driven by ``test/data/schema_corpus.json`` -- one real document per cell of the
partition, reduced to its heading sequence and a paragraph sample. Vocabulary
authored without a document to check it against is vocabulary nobody can check,
so the corpus is what makes these assertions mean anything.

The lexical tier is deterministic and model-free and carries the bulk of the
suite. The semantic and content tiers load an embedding model and are marked
``slow``; they skip when the model is unavailable.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ontocast.config import ChunkConfig
from ontocast.config.section_labels import (
    clear_section_label_caches,
    load_section_label_schema,
    schema_id_from_hint,
)
from ontocast.tool.chunk.chunker import ChunkerTool
from ontocast.tool.chunk.outline import markdown_headings
from ontocast.tool.chunk.prepare import (
    PrepareOptions,
    prepare_content_units,
    resolve_prepare_schema,
)
from ontocast.tool.chunk.schema_detect import (
    MAX_HEADINGS_FOR_CONTENT_TIER,
    SchemaEvidence,
    candidate_schema_ids,
    detect_document_schema,
    score_content,
    score_headings_lexical,
    score_headings_semantic,
)
from ontocast.toolbox import ToolBox
from test.docling_test_helpers import doc_from_markdown_lines

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS_PATH = REPO_ROOT / "test" / "data" / "schema_corpus.json"
DATA_JSON = REPO_ROOT / "data" / "json"

# Defaults of ChunkConfig.section_schema_detect_*; restated here so a threshold
# change has to be made deliberately in both places rather than silently
# invalidating every margin below.
MIN_SCORE = 2.0
MIN_MARGIN = 1.8
CONTENT_MIN_MARGIN = 4.0


def _corpus() -> list[dict[str, Any]]:
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))["entries"]


CORPUS = _corpus()
CORPUS_BY_ID = {entry["schema_id"]: entry for entry in CORPUS}


@pytest.fixture(autouse=True)
def _clear_caches():
    clear_section_label_caches()
    yield
    clear_section_label_caches()


def _document_headings(path: pathlib.Path) -> list[str]:
    text = json.loads(path.read_text(encoding="utf-8")).get("text", "")
    return [node.text for node in markdown_headings(text)]


def _scores(evidence: list[SchemaEvidence]) -> dict[str, float]:
    return {item.schema_id: item.score for item in evidence}


def _embedder(chunker: ChunkerTool):
    """The chunker's embedding callable, or skip when no model is available."""
    if chunker.embed_texts(["probe"]) is None:
        pytest.skip("semantic extras / embedding model unavailable")
    return chunker.embed_texts


# --------------------------------------------------------------------------
# The corpus itself
# --------------------------------------------------------------------------


def test_corpus_covers_every_candidate_cell() -> None:
    """A cell with no sample is a cell whose vocabulary nobody verified."""
    assert set(CORPUS_BY_ID) == set(candidate_schema_ids())


def test_corpus_entries_carry_provenance() -> None:
    for entry in CORPUS:
        assert entry["source"], entry["schema_id"]
        assert entry["license"], entry["schema_id"]
        assert entry["headings"] or entry["paragraphs"], entry["schema_id"]


# --------------------------------------------------------------------------
# Tier 1 -- lexical
# --------------------------------------------------------------------------


@pytest.mark.parametrize("schema_id", sorted(CORPUS_BY_ID))
def test_corpus_entry_detects_its_own_cell(schema_id: str) -> None:
    """Every cell is recovered from headings alone, with no model loaded."""
    detection = detect_document_schema(
        CORPUS_BY_ID[schema_id]["headings"],
        min_score=MIN_SCORE,
        min_margin=MIN_MARGIN,
    )
    assert detection is not None, f"{schema_id} abstained"
    assert detection.schema_id == schema_id
    assert detection.tier == "lexical"


def test_lexical_tier_touches_no_embedding_model() -> None:
    """Detection on a well-headed document must not reach for a model.

    The embedding tier is a fallback, not a cost the common case pays.
    """

    def exploding_embed(texts: list[str]) -> list[list[float]] | None:
        raise AssertionError("lexical tier must not embed")

    detection = detect_document_schema(
        CORPUS_BY_ID["financial"]["headings"],
        embed=exploding_embed,
        min_score=MIN_SCORE,
        min_margin=MIN_MARGIN,
    )
    assert detection is not None
    assert detection.schema_id == "financial"


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("fin.10Q.apple.json", "financial"),
        ("fin.10Q.nvidia.json", "financial"),
        ("fin.10Q.sfix.json", "financial"),
        ("chem.204703_1_5.0167542.json", "academic"),
        (
            "chem.bassani-et-al-2024-nanocrystal-assemblies-current-advances-"
            "and-open-problems.json",
            "academic",
        ),
    ],
)
def test_in_repo_documents_detect(filename: str, expected: str) -> None:
    """The documents the pipeline is actually exercised on resolve correctly."""
    path = DATA_JSON / filename
    if not path.exists():
        pytest.skip(f"{filename} not vendored")
    detection = detect_document_schema(
        _document_headings(path), min_score=MIN_SCORE, min_margin=MIN_MARGIN
    )
    assert detection is not None
    assert detection.schema_id == expected


def test_only_exclusive_headings_score() -> None:
    """A heading several schemas recognise says nothing about which cell it is.

    Scoring it fractionally only adds noise; scoring it zero is what produces
    the corpus margins.
    """
    # Recognised by several schemas, so exclusive to none.
    shared_only = score_headings_lexical(["References", "Appendix", "Background"])
    assert all(item.score == 0.0 for item in shared_only)

    # "Risk Factors" is financial and nothing else.
    exclusive = _scores(score_headings_lexical(["References", "Risk Factors"]))
    assert exclusive["financial"] == 1.0


# --------------------------------------------------------------------------
# The partition
# --------------------------------------------------------------------------


def test_general_is_never_a_positive_detection() -> None:
    """The residual cell has no profile, so it cannot be detected into."""
    assert "general" not in candidate_schema_ids()
    assert load_section_label_schema("general").document_profile.strip() == ""
    for entry in CORPUS:
        assert "general" not in _scores(score_headings_lexical(entry["headings"]))


@pytest.mark.parametrize(
    ("entry_id", "sibling"),
    [
        # Normative requirements for implementers vs instructions for a user.
        ("standard", "manual"),
        ("manual", "standard"),
        # An invention disclosure with claims vs an agreement between parties.
        ("patent", "legal"),
        ("legal", "patent"),
    ],
)
def test_at_risk_partition_pairs_stay_separated(entry_id: str, sibling: str) -> None:
    """The two boundaries most likely to collapse do not.

    Each sibling must stay below the acceptance floor entirely -- not merely
    lose -- so a slightly different document cannot flip the ranking.
    """
    scores = _scores(score_headings_lexical(CORPUS_BY_ID[entry_id]["headings"]))
    assert scores[entry_id] >= MIN_SCORE
    assert scores[sibling] < MIN_SCORE


def test_clinical_protocol_beats_academic_on_margin() -> None:
    """The sharpest boundary in the partition, pinned.

    A published trial protocol shares IMRaD headings with an academic paper, so
    ``academic`` legitimately scores here. What separates the cells is the
    margin, not the absence of the runner-up -- this is the tightest accepted
    detection in the corpus and the first to break if vocabulary drifts.
    """
    evidence = score_headings_lexical(CORPUS_BY_ID["clinical"]["headings"])
    scores = _scores(evidence)
    assert scores["academic"] >= MIN_SCORE
    assert scores["clinical"] / scores["academic"] >= MIN_MARGIN
    assert evidence[0].schema_id == "clinical"


def test_each_corpus_entry_yields_exactly_one_accepted_cell() -> None:
    """No document falls into two cells: the winner clears the margin over all."""
    for entry in CORPUS:
        evidence = score_headings_lexical(entry["headings"])
        best, runner_up = evidence[0], evidence[1]
        assert best.schema_id == entry["schema_id"]
        margin = best.score / runner_up.score if runner_up.score else float("inf")
        assert margin >= MIN_MARGIN, entry["schema_id"]


# --------------------------------------------------------------------------
# Abstention -- thin evidence must never become a confident answer
# --------------------------------------------------------------------------


def test_shared_vocabulary_abstains() -> None:
    """Headings common to every document type carry no schema information."""
    detection = detect_document_schema(
        ["Background", "References", "Appendix", "Summary", "Overview"],
        min_score=MIN_SCORE,
        min_margin=MIN_MARGIN,
    )
    assert detection is None


def test_heading_free_document_abstains_without_content_tier() -> None:
    detection = detect_document_schema(
        [],
        CORPUS_BY_ID["financial"]["paragraphs"],
        min_score=MIN_SCORE,
        min_margin=MIN_MARGIN,
    )
    assert detection is None


def test_empty_document_abstains() -> None:
    assert detect_document_schema([]) is None


def test_content_tier_stays_shut_on_a_well_headed_document() -> None:
    """The weak tier is reachable only where the strong ones had nothing.

    Passing an embedder that raises proves the content tier is not entered when
    headings exist, without needing a model.
    """

    def exploding_embed(texts: list[str]) -> list[list[float]] | None:
        raise AssertionError("content tier must not run on a headed document")

    headings = ["Background", "References"] * (MAX_HEADINGS_FOR_CONTENT_TIER + 1)
    detection = detect_document_schema(
        headings,
        CORPUS_BY_ID["financial"]["paragraphs"],
        embed=None,
        allow_content_tier=True,
        min_score=MIN_SCORE,
        min_margin=MIN_MARGIN,
    )
    assert detection is None
    # And with an embedder present, the guard is the heading count, not the
    # absence of a model.
    assert len(headings) > MAX_HEADINGS_FOR_CONTENT_TIER
    with pytest.raises(AssertionError):
        detect_document_schema(
            headings,
            CORPUS_BY_ID["financial"]["paragraphs"],
            embed=exploding_embed,
            allow_content_tier=True,
        )


# --------------------------------------------------------------------------
# Precedence and threading
# --------------------------------------------------------------------------

# Headings that detect as `financial` on the lexical tier.
_FINANCIAL_DOC = """# Item 2. Management's Discussion and Analysis
Revenue increased year over year across all reportable segments.

# Item 1A. Risk Factors
Our business is subject to macroeconomic conditions.

# Notes to Condensed Consolidated Financial Statements
Basis of presentation and summary of significant accounting policies.

# Gross Margin
Gross margin percentage decreased in the quarter.
"""


def _decision(text: str, options: PrepareOptions, **config_kwargs):
    return resolve_prepare_schema(text, ChunkConfig(**config_kwargs), options)


def test_detection_fills_the_gap_when_nothing_is_specified() -> None:
    decision = _decision(_FINANCIAL_DOC, PrepareOptions())
    assert decision.schema_id == "financial"
    assert decision.source == "detected"
    assert decision.detection is not None
    assert decision.detection.tier == "lexical"


def test_explicit_schema_id_beats_the_document_body() -> None:
    decision = _decision(_FINANCIAL_DOC, PrepareOptions(section_schema_id="academic"))
    assert decision.schema_id == "academic"
    assert decision.source == "explicit"
    assert decision.detection is None


def test_matching_hint_beats_the_document_body() -> None:
    decision = _decision(
        _FINANCIAL_DOC, PrepareOptions(document_type_hint="clinical trial protocol")
    )
    assert decision.schema_id == "clinical"
    assert decision.source == "hint"


def test_unmatched_hint_carries_no_schema_information() -> None:
    """A hint matching nothing must not suppress detection.

    ``resolve_section_schema_id`` answers ``academic`` both for "asked for
    academic" and "said nothing", so gating on its output would silently
    disable detection for every unrecognised hint.
    """
    decision = _decision(
        _FINANCIAL_DOC, PrepareOptions(document_type_hint="quarterly widget report")
    )
    assert decision.schema_id == "financial"
    assert decision.source == "detected"


@pytest.mark.parametrize(
    ("hint", "expected"),
    [
        # Whole-word needles still match.
        ("SEC 10-Q filing", "financial"),
        ("quarterly report for FY24", "financial"),
        ("ISO 9001 conformance", "standard"),
        ("EPO patent application", "patent"),
        # Needles embedded inside longer words must not: 'epo' in 'report',
        # 'paper' in 'newspaper', 'iso' in 'isotope'.
        ("quarterly widget report", None),
        ("newspaper clipping", None),
        ("isotope measurements", None),
        # No word boundary can separate the noun "novel" from the adjective, so
        # the needle is dropped rather than left to mislabel academic prose.
        ("study of novel materials", None),
        ("a novel", None),
    ],
)
def test_hints_match_whole_words_only(hint: str, expected: str | None) -> None:
    """A hint must not be matched by a needle buried inside one of its words.

    ``schema_id_from_hint`` gates automatic detection, so a false positive here
    does not merely mislabel -- it silently suppresses detection entirely and
    imposes an unrelated schema on the whole document.
    """
    assert schema_id_from_hint(hint) == expected


def test_most_specific_hint_wins() -> None:
    """Overlapping needles resolve by length, not by YAML order."""
    assert schema_id_from_hint("annual report") == "financial"
    assert schema_id_from_hint("case report") == "clinical"


def test_detection_can_be_switched_off() -> None:
    decision = _decision(_FINANCIAL_DOC, PrepareOptions(), section_schema_detect="off")
    assert decision.schema_id == "academic"
    assert decision.source == "default"


def test_no_evidence_falls_back_to_the_manifest_default() -> None:
    decision = _decision("Some prose with no headings at all.\n", PrepareOptions())
    assert decision.schema_id == "academic"
    assert decision.source == "default"


def test_detected_schema_is_the_one_handed_to_the_llm_backfill(monkeypatch) -> None:
    """The schema that tagged the document is the schema the LLM validates against.

    Re-deriving it inside the backfill would ignore the detection and validate
    against ``academic``; ``normalise_llm_label`` drops labels absent from its
    schema, so the failure is missing labels rather than an error.
    """
    captured: dict[str, Any] = {}

    async def fake_backfill(segments, tools, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "ontocast.tool.chunk.prepare.llm_backfill_section_labels", fake_backfill
    )

    config = ChunkConfig(min_size=40, max_size=500, section_classifier="llm")

    async def unused_llm(_prompt):
        raise AssertionError("LLM should not be called")

    tools = cast(
        ToolBox,
        SimpleNamespace(
            chunker=ChunkerTool(chunk_config=config),
            config=SimpleNamespace(
                chunk_config=config,
                server=SimpleNamespace(parallel_workers=2),
            ),
            llm=unused_llm,
        ),
    )
    doc = doc_from_markdown_lines(_FINANCIAL_DOC)
    asyncio.run(
        prepare_content_units(
            doc, tools.chunker, config, PrepareOptions(summarize_sections=["*"]), tools
        )
    )

    expected = resolve_prepare_schema(_FINANCIAL_DOC, config, PrepareOptions())
    assert expected.schema_id == "financial"
    assert captured["schema"] is not None
    assert captured["schema"].id == "financial"
    # Same cached object, not merely the same id.
    assert captured["schema"] is load_section_label_schema("financial")


# --------------------------------------------------------------------------
# Tier 2 -- semantic (model)
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_semantic_tier_is_never_confidently_wrong() -> None:
    """Where the semantic tier ranks wrongly, the guards catch it.

    Measured over the corpus it ranks 7/9 correctly; it puts ``manual`` above
    ``standard`` on RFC 7231 and drifts ``news`` towards ``financial``. Both
    fail acceptance -- one on margin, one on score -- so the tier abstains
    rather than relabelling a document. That property, not the ranking, is what
    makes it safe as a fallback.
    """
    chunker = ChunkerTool(chunk_config=ChunkConfig())
    embed = _embedder(chunker)
    for entry in CORPUS:
        evidence = score_headings_semantic(entry["headings"], embed)
        best, runner_up = evidence[0], evidence[1]
        if best.schema_id == entry["schema_id"]:
            continue
        margin = best.score / runner_up.score if runner_up.score else float("inf")
        assert best.score < MIN_SCORE or margin < MIN_MARGIN, (
            f"{entry['schema_id']} misdetected as {best.schema_id} "
            f"at score {best.score:.2f}, margin {margin:.2f}"
        )


@pytest.mark.slow
def test_semantic_tier_recovers_a_document_the_lexical_tier_misses() -> None:
    """Paraphrased headings no keyword matches still land in the right cell.

    This is the whole reason the tier exists: a filing whose headings are worded
    off-catalog gives the lexical tier nothing to accept, and would otherwise be
    scored against the academic default.
    """
    chunker = ChunkerTool(chunk_config=ChunkConfig())
    embed = _embedder(chunker)
    headings = [
        "Discussion of Operating Performance",
        "Principal Business Uncertainties",
        "Statements of Cash Position",
        "Segment Profitability Review",
        "Auditor Attestation",
    ]
    lexical = detect_document_schema(
        headings, min_score=MIN_SCORE, min_margin=MIN_MARGIN
    )
    assert lexical is None, "precondition: the lexical tier must abstain here"

    detection = detect_document_schema(
        headings, embed=embed, min_score=MIN_SCORE, min_margin=MIN_MARGIN
    )
    assert detection is not None
    assert detection.tier == "semantic"
    assert detection.schema_id == "financial"


# --------------------------------------------------------------------------
# Tier 3 -- content (model, deliberately hobbled)
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_content_tier_recovers_a_heading_free_financial_document() -> None:
    """Where the content tier does work, it works well.

    Financial prose is unmistakable, which is why the tier exists at all.
    """
    chunker = ChunkerTool(chunk_config=ChunkConfig())
    embed = _embedder(chunker)
    detection = detect_document_schema(
        [],
        CORPUS_BY_ID["financial"]["paragraphs"],
        embed=embed,
        allow_content_tier=True,
        min_score=MIN_SCORE,
        min_margin=MIN_MARGIN,
        content_min_margin=CONTENT_MIN_MARGIN,
    )
    assert detection is not None
    assert detection.tier == "content"
    assert detection.schema_id == "financial"


@pytest.mark.slow
def test_content_tier_misreads_academic_prose() -> None:
    """The measured failure that keeps the content tier off by default.

    Chemistry body prose scores ``standard`` well above ``academic`` and clears
    the content margin, so on a heading-free paper the tier would confidently
    pick the wrong cell. This test pins that -- if it ever starts failing the
    tier got better and ``CHUNK_SECTION_SCHEMA_DETECT=auto`` deserves another
    look; until then ``headings`` stays the default.
    """
    chunker = ChunkerTool(chunk_config=ChunkConfig())
    embed = _embedder(chunker)
    evidence = score_content(CORPUS_BY_ID["academic"]["paragraphs"], embed)
    assert evidence[0].schema_id != "academic"
    assert evidence[0].score / evidence[1].score >= CONTENT_MIN_MARGIN


@pytest.mark.slow
def test_content_tier_excludes_news() -> None:
    """``news`` is a measured semantic attractor: any front matter drifts to it."""
    chunker = ChunkerTool(chunk_config=ChunkConfig())
    embed = _embedder(chunker)
    evidence = score_content(CORPUS_BY_ID["news"]["paragraphs"], embed)
    assert "news" not in {item.schema_id for item in evidence}
