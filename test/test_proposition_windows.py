"""The retrieval query window: what bounds it, and what an unset bound must not do.

Every arm measured before a bound existed was windowed by sentence count. If
adding a bound moved those windows even slightly, each of those arms would
silently stop being comparable -- so the first group here asserts equality
across the knob space rather than at the defaults alone.

The rest pin the *mechanism* of each opt-in bound: what it guarantees that the
sentence count cannot. None of them asserts a recall figure; recall is measured
outside the repository, against a corpus, and a number pinned here would be
true of one corpus on one day.
"""

from __future__ import annotations

import pytest

from ontocast.tool.chunk.proposition import (
    FALLBACK_CHARS_PER_TOKEN,
    split_proposition_windows,
)

pytestmark = pytest.mark.unit

#: Technical prose with the shapes the bounds exist for: a citation run that the
#: period-splitter shatters, measurements with units, and a range.
TEXT = (
    "Films were annealed at 100 C for 30 min. The PL peak shifted by 20 meV. "
    "Samples aged for 4-15 days under ambient conditions. "
    "Emission narrowed by 5 nm. Cite This: J. Phys. Chem. Lett. 2019, 10, 655. "
    "See Fig. 3 and Ref. 12 for details. The shift saturated after two weeks."
)


def _counter(texts: list[str]) -> list[int]:
    """A deterministic stand-in for an encoder tokenizer: words plus wrappers.

    The splitter takes the counter as an argument precisely so it can be
    measured without loading a checkpoint; what is under test is the packing,
    not any particular tokenization. Word-piece counts are additive across a
    space -- concatenating two texts costs what they cost apart -- and this
    reproduces that, which is the property the packer's arithmetic relies on.
    """
    return [2 + len(text.split()) for text in texts]


# --------------------------------------------------------------------------
# An unset bound changes nothing.
# --------------------------------------------------------------------------


def test_unset_bounds_reproduce_sentence_windows_exactly() -> None:
    """Each new bound is opt-in, and unset it must be byte-identical."""
    for sentences in (1, 2, 3, 5):
        for windows in (1, 2, 16, 10**6):
            for stride in (None, 1, 2):
                reference = split_proposition_windows(
                    TEXT,
                    max_sentences=sentences,
                    max_windows=windows,
                    stride=stride,
                )
                assert reference == split_proposition_windows(
                    TEXT,
                    max_sentences=sentences,
                    max_windows=windows,
                    stride=stride,
                    max_tokens=None,
                    token_counter=_counter,
                    abbreviation_aware=False,
                    measurement_aware=False,
                    overlap=0.0,
                )


def test_unset_bounds_leave_the_character_budget_untouched() -> None:
    """The character budget predates these knobs and keeps its exact windows."""
    for budget in (60, 200, 800):
        for stride in (None, 1, 2):
            assert split_proposition_windows(
                TEXT, max_windows=10**6, max_chars=budget, stride=stride
            ) == split_proposition_windows(
                TEXT,
                max_windows=10**6,
                max_chars=budget,
                stride=stride,
                max_tokens=None,
                token_counter=_counter,
                abbreviation_aware=False,
                measurement_aware=False,
                overlap=0.0,
            )


def test_every_new_bound_is_off_in_the_shipped_configuration() -> None:
    from ontocast.config import VectorStoreConfig

    config = VectorStoreConfig()
    assert config.proposition_window_max_tokens is None
    assert config.proposition_window_overlap == 0.0
    assert config.proposition_abbreviation_aware is False
    assert config.proposition_measurement_aware is False


# --------------------------------------------------------------------------
# Abbreviation-aware boundaries.
# --------------------------------------------------------------------------


def test_the_period_splitter_shatters_a_citation() -> None:
    """The defect the abbreviation guard exists for, stated as a baseline."""
    windows = split_proposition_windows(TEXT, max_sentences=2, max_windows=10**6)

    assert any(len(window) < 20 for window in windows)


def test_abbreviation_awareness_keeps_a_citation_in_one_piece() -> None:
    """An initial, a truncated title word and a volume number are one atom."""
    windows = split_proposition_windows(
        TEXT, max_sentences=1, max_windows=10**6, abbreviation_aware=True
    )

    assert all(len(window) >= 20 for window in windows)
    assert any("J. Phys. Chem. Lett." in window for window in windows)


def test_abbreviation_awareness_still_ends_an_ordinary_sentence() -> None:
    """A sentence closing on a short unit token is not an abbreviation.

    The guard is shape-based, so "20 meV." looks exactly like a truncation. What
    separates them is the fragment around it: prose with ordinary lowercase
    words is a whole sentence, a run of capitalised stubs is not.
    """
    text = "The peak shifted by 20 meV. The film then stabilised over two weeks."
    windows = split_proposition_windows(
        text, max_sentences=1, max_windows=10**6, abbreviation_aware=True
    )

    assert len(windows) == 2


def test_abbreviation_awareness_merges_a_lowercase_continuation() -> None:
    """A period followed by a lowercase word never ended a sentence."""
    text = "Values were averaged over 5 runs and, cf. the appendix, rounded up."
    windows = split_proposition_windows(
        text, max_sentences=1, max_windows=10**6, abbreviation_aware=True
    )

    assert windows == [text]


# --------------------------------------------------------------------------
# The token budget.
# --------------------------------------------------------------------------


def test_the_token_budget_bounds_every_window_by_the_encoder_s_own_unit() -> None:
    """No window exceeds the budget -- including one built from a long sentence.

    This is what the character budget cannot promise: it emits an over-long
    sentence whole, so a passage of long sentences is bounded in name only.
    """
    for budget in (10, 16, 24):
        windows = split_proposition_windows(
            TEXT, max_windows=10**6, max_tokens=budget, token_counter=_counter
        )
        assert windows
        overhead = _counter([""])[0]
        assert all(count - overhead <= budget for count in _counter(windows))


def test_the_token_budget_cuts_a_sentence_the_encoder_would_truncate() -> None:
    """A sentence over budget is cut at whitespace, not emitted whole.

    Emitted whole it would be truncated by the encoder and its tail would reach
    no lane at all; cut here, the tail is a query of its own.
    """
    sentence = "The film " + " ".join(f"stage{index}" for index in range(60)) + " ends."
    windows = split_proposition_windows(
        sentence, max_windows=10**6, max_tokens=16, token_counter=_counter
    )

    assert len(windows) > 1
    assert " ".join(windows) == sentence


def test_the_token_budget_replaces_the_character_budget() -> None:
    """Two bounds honoured at once means the tighter one silently wins."""
    by_tokens = split_proposition_windows(
        TEXT, max_windows=10**6, max_tokens=16, token_counter=_counter
    )
    with_chars = split_proposition_windows(
        TEXT,
        max_windows=10**6,
        max_tokens=16,
        max_chars=40,
        token_counter=_counter,
    )

    assert by_tokens == with_chars


def test_a_token_budget_without_a_tokenizer_approximates_and_says_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Degrading to characters is a documented fallback, not a silent revert.

    Silently reverting to the sentence bound would leave a deployment believing
    truncation was ruled out when nothing bounds the window at all.
    """
    import ontocast.tool.chunk.proposition as module

    module._warned_no_tokenizer = False
    with caplog.at_level("WARNING"):
        windows = split_proposition_windows(TEXT, max_windows=10**6, max_tokens=64)

    assert "tokenizer" in caplog.text
    assert windows == split_proposition_windows(
        TEXT, max_windows=10**6, max_chars=round(64 * FALLBACK_CHARS_PER_TOKEN)
    )


# --------------------------------------------------------------------------
# Measurement-aware boundaries.
# --------------------------------------------------------------------------


#: A single sentence long enough to be cut, whose natural cut lands between a
#: number and the unit it is written with.
MEASURED = (
    "The absorption edge of the films exhibits a red shift of 10 meV after "
    "ageing for 15 days at 85 C in ambient air with 50 % relative humidity."
)


def test_a_break_is_never_left_between_a_number_and_its_unit() -> None:
    """A window ending on "at 85" retrieves nothing that "85 C" would."""
    guarded = split_proposition_windows(
        MEASURED,
        max_windows=10**6,
        max_tokens=10,
        token_counter=_counter,
        measurement_aware=True,
    )

    assert not any(window.rstrip().split()[-1].isdigit() for window in guarded)


def test_the_measurement_guard_only_moves_a_break_backwards() -> None:
    """Moving a break later would push the window past the budget it was set to."""
    plain = split_proposition_windows(
        MEASURED, max_windows=10**6, max_tokens=10, token_counter=_counter
    )
    guarded = split_proposition_windows(
        MEASURED,
        max_windows=10**6,
        max_tokens=10,
        token_counter=_counter,
        measurement_aware=True,
    )

    assert plain != guarded
    overhead = _counter([""])[0]
    assert all(count - overhead <= 10 for count in _counter(guarded))
    assert " ".join(guarded) == MEASURED


def test_the_measurement_guard_is_inert_where_no_sentence_is_cut() -> None:
    """It guards a cut; a bound that never cuts inside a sentence has none."""
    assert split_proposition_windows(
        TEXT, max_windows=10**6, max_chars=300, measurement_aware=True
    ) == split_proposition_windows(TEXT, max_windows=10**6, max_chars=300)


# --------------------------------------------------------------------------
# Fractional overlap.
# --------------------------------------------------------------------------


def test_overlap_repeats_text_and_costs_queries() -> None:
    """Overlap buys boundary-straddling statements at the price of more queries."""
    disjoint = split_proposition_windows(
        TEXT, max_windows=10**6, max_tokens=16, token_counter=_counter
    )
    overlapped = split_proposition_windows(
        TEXT, max_windows=10**6, max_tokens=16, token_counter=_counter, overlap=0.5
    )

    assert len(overlapped) > len(disjoint)
    # Consecutive windows share text: that sharing is the whole mechanism, and
    # it is what a statement straddling a boundary is bought with.
    assert any(
        set(first.split()) & set(second.split())
        for first, second in zip(overlapped, overlapped[1:], strict=False)
    )


def test_overlap_always_advances() -> None:
    """A high overlap costs queries; it must not fail to terminate."""
    windows = split_proposition_windows(
        TEXT, max_windows=10**6, max_tokens=8, token_counter=_counter, overlap=0.95
    )

    assert windows


def test_overlap_applies_to_the_character_budget_too() -> None:
    """Fractional overlap is a property of a budget, not of one particular unit."""
    disjoint = split_proposition_windows(TEXT, max_windows=10**6, max_chars=120)
    overlapped = split_proposition_windows(
        TEXT, max_windows=10**6, max_chars=120, overlap=0.5
    )

    assert len(overlapped) > len(disjoint)


def test_a_nonsense_bound_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        split_proposition_windows(TEXT, max_tokens=0)
    with pytest.raises(ValueError, match="overlap"):
        split_proposition_windows(TEXT, max_chars=100, overlap=1.0)


# --------------------------------------------------------------------------
# Domain independence.
# --------------------------------------------------------------------------


def test_no_bound_carries_domain_vocabulary() -> None:
    """Every rule here is typography, English or measurement *shape*.

    A domain word in the splitter would make retrieval quality a property of the
    corpus it was written against, which is exactly the failure the equal-
    information bounds are meant to remove.
    """
    from ontocast.tool.chunk.proposition import ABBREVIATIONS

    assert all(word.replace(".", "").isalpha() for word in ABBREVIATIONS)
    assert all(len(word) <= 6 for word in ABBREVIATIONS)
