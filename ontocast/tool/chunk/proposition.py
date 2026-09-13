"""Lightweight text splitting helpers (no ML dependencies).

The default path here is pure regex and stays that way. The opt-in bounds --
an encoder-token budget and the measurement-aware break guard -- take their
extra knowledge as *arguments* (a token counter, the shared measurement
lexicon), so this module never grows a dependency on an encoder or on a
domain vocabulary.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

from ontocast.util.measurement_lexicon import unit_adjacent_numbers

logger = logging.getLogger(__name__)

# Regex pattern for splitting text into sentences
# Matches: paragraph breaks (double newlines) OR sentence endings followed by capital letters
SENTENCE_SPLIT_REGEX = r"(?:\n\s*\n+)|(?<=[.!?])\s+(?=[A-Z][a-z])"

#: One text's worth of tokens per text, or ``None`` where the provider exposes
#: no tokenizer. :meth:`EmbeddingTool.token_lengths` has exactly this shape; the
#: splitter takes it as a callable so it stays free of the encoder.
TokenCounter = Callable[[list[str]], list[int] | None]

#: Characters per token assumed when a token budget is requested but the
#: provider exposes no tokenizer. Deliberately on the low side of what dense
#: technical prose measures, so the fallback under-fills rather than overruns
#: the encoder's limit -- the failure the budget exists to prevent.
FALLBACK_CHARS_PER_TOKEN = 3.5

#: Words whose trailing period is part of the word. General English and
#: bibliographic shapes only: a domain vocabulary here would make the splitter
#: corpus-specific, which is exactly what the sentence bound is being replaced
#: for. Written without the period, lowercase; matched case-insensitively.
ABBREVIATIONS: frozenset[str] = frozenset(
    {
        # bibliographic
        "al",
        "cf",
        "ed",
        "eds",
        "edn",
        "eq",
        "eqs",
        "fig",
        "figs",
        "no",
        "nos",
        "pp",
        "ref",
        "refs",
        "sec",
        "sect",
        "suppl",
        "tab",
        "tabs",
        "vol",
        "vols",
        # general English
        "approx",
        "ca",
        "e.g",
        "i.e",
        "vs",
        "viz",
        # titles and name particles
        "dr",
        "prof",
        "mr",
        "mrs",
        "ms",
        "st",
        "jr",
        "sr",
        "inc",
        "ltd",
    }
)

#: A short alphabetic token whose period may be an abbreviation mark rather
#: than a full stop -- an initial ("J."), a truncation ("Phys."), a symbol
#: ("nm."). Shape only; whether it *is* one is decided by its neighbours.
_ABBREVIATION_SHAPE = re.compile(r"^[A-Za-z]{1,6}\.$")

#: A fragment that does not continue a sentence: it opens with a capital, a
#: quote or a bracket. Anything else -- a lowercase word, a digit, a symbol --
#: continues what came before, so the period before it was not a full stop.
_SENTENCE_OPENER = re.compile(r"^[\"'“‘(\[]*[A-Z]")

#: Word tokens a fragment needs before it can be read as a whole sentence.
_MIN_SENTENCE_WORDS = 4

_LOWERCASE_WORD = re.compile(r"^[a-z]+[.,;:]?$")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n\s*\n+")


def split_proposition_windows(
    text: str,
    max_sentences: int = 2,
    max_windows: int = 16,
    stride: int | None = None,
    max_chars: int | None = None,
    max_tokens: int | None = None,
    token_counter: TokenCounter | None = None,
    abbreviation_aware: bool = False,
    measurement_aware: bool = False,
    overlap: float = 0.0,
) -> list[str]:
    """Split text into short proposition-like windows for retrieval.

    Args:
        text: Passage to split.
        max_sentences: Sentences per window. Not applied when ``max_chars`` or
            ``max_tokens`` is set: the three are alternative ways of saying how
            much text one query carries, and honouring more than one means the
            tighter one silently wins.
        max_windows: Ceiling on the windows returned; over it, windows are sampled
            evenly across the passage rather than truncated, so the tail still
            contributes a query.
        stride: Sentences advanced between windows. ``None`` (default) strides by
            the window's own length, giving contiguous, disjoint windows. A smaller
            stride overlaps them, which is what lets a statement spanning a window
            boundary appear whole in some window -- with disjoint windows it appears
            in none, and neither half retrieves what the pair together names.
            Sentence-granular, so it says nothing about how much text is repeated;
            ``overlap`` is the fractional form the budget modes take.
        max_chars: Characters per window. ``None`` (default) bounds windows by
            sentence count instead, which is the historical behaviour and is
            reproduced exactly.

            A sentence count is a poor bound on how much text a query carries. Two
            sentences of technical prose range over an order of magnitude in
            length, and the encoder truncates on *tokens*, so a sentence-bounded
            window can silently lose its tail. The splitter also has no
            abbreviation handling and breaks on every ``.``, so "J. Phys. Chem.
            Lett." is four "sentences" and a two-sentence window over a citation
            carries nothing retrievable at all.

            A character budget addresses both ends: it caps the long windows that
            truncate, and it *coalesces* short fragments, because it keeps taking
            sentences until the budget is reached. A single sentence longer than
            the budget is emitted whole rather than cut -- the encoder truncates it
            either way, and cutting first only loses more.
        max_tokens: Encoder word pieces per window, measured with ``token_counter``.
            ``None`` (default) leaves the other bounds in charge; when set it
            replaces both of them, and characters stop being consulted at all.

            This is the only bound that equalises what a query carries, because it
            is the unit the encoder itself counts in: characters per token drift
            with notation, so a character budget that fits one passage truncates
            the next. Set below the encoder's sequence limit, it makes truncation
            impossible by construction -- which a character budget can only
            approximate. Unlike the character budget it *does* cut a sentence that
            alone exceeds the budget, at a whitespace boundary, because a sentence
            emitted whole would be truncated by the encoder anyway and the tail
            would then reach no lane at all.
        token_counter: Word pieces per text, typically
            ``EmbeddingTool.token_lengths``. Required by ``max_tokens``; when it is
            absent or reports ``None``, the budget degrades to a character
            approximation (``FALLBACK_CHARS_PER_TOKEN``) and says so once, rather
            than silently reverting to the sentence bound.
        abbreviation_aware: Merge fragments the period-splitter created inside an
            abbreviation, an initial or a citation run. A prerequisite for any
            equal-size packing rather than a candidate of its own: without it the
            packer's atoms include "Chem. Lett.", and a window bound counted in
            those atoms is not counting sentences. General English and
            bibliographic shapes only -- see :data:`ABBREVIATIONS`.
        measurement_aware: Forbid a break between a number and the unit it is
            written with, or inside a range, using the number/unit *shapes* in
            :mod:`ontocast.util.measurement_lexicon`. Only binds where a cut
            inside a sentence is possible, which today means ``max_tokens`` with a
            sentence over budget: a window ending on "a red shift of ~10" retrieves
            nothing that "~10 meV" would.
        overlap: Fraction of a window repeated at the start of the next one, for
            the budget modes. ``0.0`` (default) leaves windows disjoint. The
            fractional form exists because ``stride`` counts sentences, which under
            a budget says nothing about how much text is shared. Overlap multiplies
            queries, so it is paid for per window and, once ``max_windows`` binds,
            in coverage elsewhere in the passage.

    Returns:
        list[str]: Windows in document order.
    """
    cleaned = text.strip()
    if not cleaned:
        return []
    if max_sentences <= 0:
        raise ValueError("max_sentences must be >= 1")
    if max_windows <= 0:
        raise ValueError("max_windows must be >= 1")
    if max_chars is not None and max_chars <= 0:
        raise ValueError("max_chars must be >= 1")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be >= 1")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    step = max_sentences if stride is None else stride
    if step <= 0:
        raise ValueError("stride must be >= 1")

    # Keep this splitter lightweight and deterministic.
    sentence_parts = [
        part.strip() for part in _SENTENCE_SPLIT.split(cleaned) if part.strip()
    ]
    if abbreviation_aware:
        sentence_parts = _merge_abbreviations(sentence_parts)
    if not sentence_parts:
        return [cleaned[:1000]] if cleaned else []

    pieces = _windows(
        sentence_parts,
        max_sentences=max_sentences,
        step=step,
        stride=stride,
        max_chars=max_chars,
        max_tokens=max_tokens,
        token_counter=token_counter,
        measurement_aware=measurement_aware,
        overlap=overlap,
    )

    windows: list[str] = []
    seen: set[str] = set()
    for window in pieces:
        # A stride below the window size makes the final windows suffixes of one
        # another once the tail is shorter than a full window; emitting those twice
        # would pay for an identical query and skew the fusion ranks it feeds.
        if window and window not in seen:
            seen.add(window)
            windows.append(window)

    if len(windows) > max_windows:
        # Sample evenly across the text rather than keeping the first ``max_windows``.
        # Truncating dropped the tail of a long chunk entirely, so its closing sections
        # never contributed a retrieval query at all. Positions span both endpoints, so
        # the final window is always represented.
        if max_windows == 1:
            windows = windows[:1]
        else:
            last = len(windows) - 1
            picked = {
                round(position * last / (max_windows - 1))
                for position in range(max_windows)
            }
            windows = [windows[index] for index in sorted(picked)]

    return windows or [cleaned[:1000]]


def _windows(
    sentence_parts: list[str],
    *,
    max_sentences: int,
    step: int,
    stride: int | None,
    max_chars: int | None,
    max_tokens: int | None,
    token_counter: TokenCounter | None,
    measurement_aware: bool,
    overlap: float,
) -> list[str]:
    """Dispatch to the bound in force, tightest declaration first."""
    if max_tokens is not None:
        sizer = _token_sizer(token_counter)
        if sizer is not None:
            atoms = _token_atoms(
                sentence_parts, max_tokens, sizer, measurement_aware=measurement_aware
            )
            return _pack(atoms, max_tokens, overlap, join_cost=0)
        max_chars = max(1, round(max_tokens * FALLBACK_CHARS_PER_TOKEN))
    if max_chars is None:
        return _windows_by_sentence_count(sentence_parts, max_sentences, step)
    if stride is not None:
        return _windows_by_char_budget(sentence_parts, max_chars, stride)
    atoms = [(part, len(part)) for part in sentence_parts]
    return _pack(atoms, max_chars, overlap)


def _windows_by_sentence_count(
    sentence_parts: list[str], max_sentences: int, step: int
) -> list[str]:
    """Windows of a fixed sentence count, strided by ``step`` sentences."""
    return [
        " ".join(sentence_parts[index : index + max_sentences]).strip()
        for index in range(0, len(sentence_parts), step)
    ]


def _windows_by_char_budget(
    sentence_parts: list[str], max_chars: int, stride: int
) -> list[str]:
    """Windows packed to a character budget, strided by whole sentences.

    The sentence-granular stride is kept for the character budget because it is
    the historical spelling of overlap; ``overlap`` is the fractional form and
    the two are mutually exclusive.
    """
    windows: list[str] = []
    start = 0
    while start < len(sentence_parts):
        length = 0
        stop = start
        while stop < len(sentence_parts):
            addition = len(sentence_parts[stop]) + (1 if stop > start else 0)
            if stop > start and length + addition > max_chars:
                break
            length += addition
            stop += 1
        windows.append(" ".join(sentence_parts[start:stop]).strip())
        start = start + stride
    return windows


def _pack(
    atoms: list[tuple[str, int]], budget: int, overlap: float, join_cost: int = 1
) -> list[str]:
    """Greedily fill windows to ``budget``, repeating ``overlap`` of each.

    Each window starts at an atom and takes as many following atoms as fit, so a
    window's atom count varies with how long its atoms are -- which is the point
    of a budget. An atom that alone exceeds the budget becomes its own window.

    The next window starts far enough back that roughly ``overlap`` of the
    window just emitted is repeated, measured in the same unit as the budget.
    Progress is guaranteed: the start always advances by at least one atom, so a
    high overlap costs queries rather than looping.

    ``join_cost`` is what the separator itself costs in the budget's unit: one
    character, but no word piece, since a tokenizer does not emit one for a
    space.
    """
    windows: list[str] = []
    start = 0
    while start < len(atoms):
        size = 0
        stop = start
        while stop < len(atoms):
            addition = atoms[stop][1] + (join_cost if stop > start else 0)
            if stop > start and size + addition > budget:
                break
            size += addition
            stop += 1
        windows.append(" ".join(text for text, _ in atoms[start:stop]).strip())
        if overlap <= 0.0:
            start = stop
            continue
        target = overlap * size
        repeated = 0
        index = stop
        while index - 1 > start and repeated + atoms[index - 1][1] <= target:
            index -= 1
            repeated += atoms[index][1]
        start = max(index, start + 1)
    return windows


def _token_sizer(
    token_counter: TokenCounter | None,
) -> Callable[[list[str]], list[int]] | None:
    """Net word pieces per text: the provider's count minus its fixed overhead.

    A tokenizer counts the special tokens it wraps every text in, so summing raw
    counts over the atoms of a window over-charges the window by one wrapper per
    atom. Subtracting the wrapper -- measured once, from the empty string --
    makes the atom sizes additive, which is what the packer needs.

    Returns:
        A sizer, or ``None`` when the provider exposes no tokenizer.
    """
    if token_counter is None:
        _warn_no_tokenizer()
        return None
    try:
        baseline = token_counter([""])
    except Exception:  # noqa: BLE001 - a tokenizer failure must not break retrieval
        baseline = None
    if baseline is None:
        _warn_no_tokenizer()
        return None
    overhead = baseline[0] if baseline else 0

    def sizer(texts: list[str]) -> list[int]:
        if not texts:
            return []
        counts = token_counter(texts)
        if counts is None:
            # Only reachable if the provider degrades mid-pass; charge the
            # fallback rather than treating unknown as free.
            return [
                max(1, round(len(text) / FALLBACK_CHARS_PER_TOKEN)) for text in texts
            ]
        return [max(1, count - overhead) for count in counts]

    return sizer


_warned_no_tokenizer = False


def _warn_no_tokenizer() -> None:
    global _warned_no_tokenizer
    if _warned_no_tokenizer:
        return
    _warned_no_tokenizer = True
    logger.warning(
        "A token budget for retrieval windows was requested but the embedding "
        "provider exposes no tokenizer; falling back to ~%.1f characters per "
        "token. The budget is then an approximation and no longer rules out "
        "encoder truncation.",
        FALLBACK_CHARS_PER_TOKEN,
    )


def _token_atoms(
    sentence_parts: list[str],
    budget: int,
    sizer: Callable[[list[str]], list[int]],
    *,
    measurement_aware: bool,
) -> list[tuple[str, int]]:
    """Sentences, with any sentence over budget cut at a whitespace boundary.

    Cutting is what separates the token budget from the character one. A
    sentence longer than the encoder accepts is truncated by the encoder
    whatever we do, and its tail then reaches no lane at all; cut here, the tail
    is a query of its own.
    """
    sizes = sizer(sentence_parts)
    texts: list[str] = []
    for part, size in zip(sentence_parts, sizes, strict=True):
        if size <= budget:
            texts.append(part)
            continue
        texts.extend(
            _cut_sentence(part, budget, sizer, measurement_aware=measurement_aware)
        )
    # Re-measure: a cut piece's size is not the sum of its words' sizes, because
    # word pieces are found across the whole string.
    return list(zip(texts, sizer(texts), strict=True))


def _cut_sentence(
    text: str,
    budget: int,
    sizer: Callable[[list[str]], list[int]],
    *,
    measurement_aware: bool,
) -> list[str]:
    """One over-budget sentence as whitespace-bounded pieces.

    Each piece is measured as the substring it will actually be, not as the sum
    of its words: word pieces are found across a whole string, so summing per-
    word counts overshoots and would let a piece land over the budget the cut
    exists to respect. Prefix counts are monotone, so one batched measurement
    per piece finds the longest prefix that fits.
    """
    words = list(re.finditer(r"\S+", text))
    if len(words) <= 1:
        return [text]
    spans = _measurement_spans(text) if measurement_aware else []

    pieces: list[str] = []
    start = 0
    while start < len(words):
        begin = words[start].start()
        prefixes = [text[begin : word.end()] for word in words[start:]]
        stop = start + 1
        for offset, size in enumerate(sizer(prefixes)):
            if size > budget:
                break
            stop = start + offset + 1
        if stop < len(words):
            stop = _shift_break(words, start, stop, spans)
        pieces.append(text[begin : words[stop - 1].end()])
        start = stop
    return pieces


def _measurement_spans(text: str) -> list[tuple[int, int]]:
    """Character spans running from a number to the unit it is written with.

    A range yields one mention per number, all ending past the shared unit, so
    the first mention's span already covers the whole range and the guard needs
    no separate notion of one.
    """
    return [(mention.start, mention.end) for mention in unit_adjacent_numbers(text)]


def _shift_break(
    words: list[re.Match[str]],
    start: int,
    stop: int,
    spans: list[tuple[int, int]],
) -> int:
    """Move a break out of a number/unit pair, backwards, or leave it alone.

    Backwards rather than forwards: moving the break earlier keeps the window
    within its budget, while moving it later would push the window over the
    encoder's limit -- and a truncated window loses the pair anyway. A break
    that cannot move without emptying the window stays where it is.
    """
    while stop - 1 > start:
        offset = words[stop - 1].end()
        if not any(begin < offset < end for begin, end in spans):
            return stop
        stop -= 1
    return stop


def _merge_abbreviations(parts: list[str]) -> list[str]:
    """Rejoin fragments split on a period that was not a full stop.

    The splitter breaks on every ``.``, so an initial, a truncated word or a
    citation run becomes several "sentences" -- and a two-sentence window over
    one of them carries nothing retrievable. Three shape rules, no vocabulary
    beyond the general abbreviation list:

    * the next fragment does not open a sentence (it starts lowercase, with a
      digit, or with a symbol), so the period before it was not a full stop;
    * the previous fragment ends in a known abbreviation;
    * the previous fragment ends in an abbreviation *shape* and does not read as
      a whole sentence -- too few words, or no lowercase word in it, which is
      what a run of initials and truncations looks like.
    """
    merged: list[str] = []
    for part in parts:
        if merged and not _is_boundary(merged[-1], part):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged


def _is_boundary(before: str, after: str) -> bool:
    """Whether the period between two fragments ended a sentence."""
    if not _SENTENCE_OPENER.match(after):
        return False
    tokens = before.split()
    if not tokens:
        return False
    last = tokens[-1]
    if last.rstrip(".").lower() in ABBREVIATIONS:
        return False
    if _ABBREVIATION_SHAPE.match(last) and not _reads_as_sentence(tokens):
        return False
    return True


def _reads_as_sentence(tokens: list[str]) -> bool:
    """Whether a fragment has the shape of a whole sentence.

    Enough words to be one, and at least one ordinary lowercase word -- the test
    that separates prose from a run of initials and truncated titles, which
    carry capitals throughout.
    """
    if len(tokens) < _MIN_SENTENCE_WORDS:
        return False
    return any(_LOWERCASE_WORD.match(token) for token in tokens)
