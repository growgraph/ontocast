"""``AGG_SIMILARITY_THRESHOLD`` is not the in-pipeline aggregator's threshold.

The aggregator clusters and gates at ``AGG_CANDIDATE_SIMILARITY_THRESHOLD``;
``AGG_SIMILARITY_THRESHOLD`` drives only the cross-graph ``EntityAligner``.
Tuning the former while the latter sits at its default changes nothing in a
pipeline run, so the aggregator says so at construction time -- and stays
quiet when the caller pins the candidate threshold itself, which is how the
aligner builds its compatibility aggregator.
"""

import logging

import pytest

from ontocast.config import AggregationConfig
from ontocast.tool import EmbeddingBasedAggregator

pytestmark = pytest.mark.unit

LOGGER = "ontocast.tool.agg.aggregate"


@pytest.fixture(autouse=True)
def _no_threshold_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGG_SIMILARITY_THRESHOLD", raising=False)
    monkeypatch.delenv("AGG_CANDIDATE_SIMILARITY_THRESHOLD", raising=False)


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING and record.name == LOGGER
    ]


def test_warns_when_only_the_aligner_threshold_is_tuned(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER)
    EmbeddingBasedAggregator(AggregationConfig(similarity_threshold=0.9))

    messages = _warnings(caplog)
    assert len(messages) == 1
    assert "AGG_SIMILARITY_THRESHOLD=0.9" in messages[0]
    assert "AGG_CANDIDATE_SIMILARITY_THRESHOLD" in messages[0]


def test_quiet_when_the_candidate_threshold_is_tuned_too(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER)
    EmbeddingBasedAggregator(
        AggregationConfig(similarity_threshold=0.9, candidate_similarity_threshold=0.8)
    )
    assert _warnings(caplog) == []


def test_quiet_at_defaults(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER)
    EmbeddingBasedAggregator(AggregationConfig())
    assert _warnings(caplog) == []


def test_quiet_when_the_caller_pins_the_candidate_threshold(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The aligner path: the override *is* the threshold, nothing is inert."""
    caplog.set_level(logging.WARNING, logger=LOGGER)
    EmbeddingBasedAggregator(
        AggregationConfig(similarity_threshold=0.9),
        candidate_similarity_threshold=0.9,
    )
    assert _warnings(caplog) == []
