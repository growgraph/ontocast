"""How long is a retrieval query window, really — in characters and in tokens?

`VECTOR_STORE_PROPOSITION_WINDOW_SENTENCES` bounds a query by **sentence count**,
and nothing bounds its length. But length is what governs the two things that
decide whether a query works:

* how many distinct terms one embedding vector has to carry, and therefore how
  deep a ranked list must be to return them all;
* whether the encoder silently truncates the query — sentence-transformers cuts to
  the checkpoint's ``max_seq_length`` before the model sees the text, and returns a
  correctly shaped vector for the prefix, so nothing downstream can tell.

Two sentences can be forty characters or four hundred. This measures that spread
before anything is tuned against it, because a knob whose unit does not match the
constraint it is meant to respect cannot be tuned, only guessed at.

It also measures the **window budget**: `PROPOSITION_MAX_WINDOWS` caps how many
windows a unit yields, and over the cap windows are *subsampled* — that text
reaches no lane at all. Whether the cap binds is a property of unit length, so it
is measured here per unit scale rather than assumed from the harness's own.

No LLM call, no vector store, no service. It loads the encoder to use its real
tokenizer, and nothing else. It lives here rather than in the measurement repo for
the same reason :mod:`test.retrieval_sweep` does: it needs the library's own
windower and the configured encoder, and that repo deliberately depends on
neither, so that it can score artifacts produced by any version.

Usage::

    uv run python -m test.window_stats --corpus <dir with cases.jsonl>
    uv run python -m test.window_stats --source-dir <dir of .txt> --passage-chars 3172
"""

from __future__ import annotations

import json
import pathlib
import statistics
from typing import Any

import click

_CLICK_DIR = click.Path(path_type=pathlib.Path, file_okay=False, exists=True)

#: The sentence-count settings worth characterising: the shipped default is 2.
_SENTENCE_SETTINGS = (1, 2, 3, 5)


def _percentile(values: list[int], fraction: float) -> int:
    """Nearest-rank percentile; plain and dependency-free."""
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _load_corpus_texts(corpus: pathlib.Path) -> list[str]:
    """Case texts from a recall corpus."""
    lines = (corpus / "cases.jsonl").read_text().splitlines()
    return [json.loads(line)["text"] for line in lines if line.strip()]


def _split_passages(text: str, size: int) -> list[str]:
    """Pack paragraphs up to ``size`` characters, never splitting one.

    Mirrors the packing the recall-corpus builder uses, so a corpus and a re-cut
    of its own sources are measured on comparable terms. Deliberately a *length*
    probe: it does not reproduce the production chunker's semantics -- sections,
    bibliography routing, the density split -- so it answers "what does a unit of
    this length do to the window budget", not "what does a production unit
    contain".
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    passages: list[str] = []
    current: list[str] = []
    length = 0
    for paragraph in paragraphs:
        if current and length + len(paragraph) > size:
            passages.append("\n\n".join(current))
            current, length = [], 0
        current.append(paragraph)
        length += len(paragraph)
    if current:
        passages.append("\n\n".join(current))
    return passages


def _load_source_passages(source_dir: pathlib.Path, passage_chars: int) -> list[str]:
    """Documents re-cut at a chosen passage length."""
    passages: list[str] = []
    for path in sorted(source_dir.glob("*.txt")):
        passages.extend(_split_passages(path.read_text(), size=passage_chars))
    return passages


def _encoder() -> tuple[Any, int | None]:
    """The configured embedding tool and its truncation ceiling.

    Built from the production config rather than by loading a checkpoint by name,
    so the numbers describe the encoder a deployment actually uses -- including
    its query prefix, which counts against the same budget as the text.
    """
    from ontocast.config import EmbeddingConfig
    from ontocast.tool.vector_store.embedding import EmbeddingTool

    tool = EmbeddingTool.create(EmbeddingConfig())
    return tool, tool.sequence_limit


def _measure(
    texts: list[str],
    max_sentences: int,
    max_windows: int,
    encoder: Any,
    limit: int | None,
) -> dict[str, Any]:
    """Window every text and describe the windows produced."""
    from ontocast.tool.chunk.proposition import split_proposition_windows

    chars: list[int] = []
    per_unit: list[int] = []
    capped_units = 0
    dropped_windows = 0

    for text in texts:
        windows = split_proposition_windows(
            text, max_sentences=max_sentences, max_windows=max_windows
        )
        # What the unit *would* have produced without the cap, to see the loss.
        uncapped = split_proposition_windows(
            text, max_sentences=max_sentences, max_windows=10**6
        )
        if len(uncapped) > max_windows:
            capped_units += 1
            dropped_windows += len(uncapped) - len(windows)
        per_unit.append(len(uncapped))
        chars.extend(len(w) for w in windows)

    row: dict[str, Any] = {
        "max_sentences": max_sentences,
        "windows": len(chars),
        "chars_min": min(chars) if chars else 0,
        "chars_median": int(statistics.median(chars)) if chars else 0,
        "chars_p90": _percentile(chars, 0.90),
        "chars_max": max(chars) if chars else 0,
        "windows_per_unit_median": (
            int(statistics.median(per_unit)) if per_unit else 0
        ),
        "windows_per_unit_max": max(per_unit) if per_unit else 0,
        "units_hitting_window_cap": capped_units,
        "windows_dropped_by_cap": dropped_windows,
    }

    if limit is not None:
        # Re-window without the cap for the token view: truncation is a property of
        # a window, not of whether the budget happened to keep it.
        all_windows: list[str] = []
        for text in texts:
            all_windows.extend(
                split_proposition_windows(
                    text, max_sentences=max_sentences, max_windows=10**6
                )
            )
        counts = encoder.token_lengths(all_windows)
        if counts is None:
            return row
        over = [n for n in counts if n > limit]
        row.update(
            {
                "tokens_median": int(statistics.median(counts)) if counts else 0,
                "tokens_p90": _percentile(counts, 0.90),
                "tokens_max": max(counts) if counts else 0,
                "over_limit": len(over),
                "over_limit_pct": (100.0 * len(over) / len(counts)) if counts else 0.0,
                "chars_per_token": (
                    round(sum(len(w) for w in all_windows) / sum(counts), 2)
                    if sum(counts)
                    else 0.0
                ),
            }
        )
    return row


@click.command()
@click.option("--corpus", "corpora", multiple=True, type=_CLICK_DIR)
@click.option(
    "--source-dir",
    type=_CLICK_DIR,
    default=None,
    help="Documents to re-cut, for measuring at a different unit scale.",
)
@click.option(
    "--passage-chars",
    multiple=True,
    type=int,
    help="Passage lengths to cut --source-dir at; repeat.",
)
@click.option(
    "--max-windows",
    default=16,
    show_default=True,
    help="The PROPOSITION_MAX_WINDOWS cap under test.",
)
@click.option("--json-out", type=click.Path(path_type=pathlib.Path), default=None)
def main(
    corpora: tuple[pathlib.Path, ...],
    source_dir: pathlib.Path | None,
    passage_chars: tuple[int, ...],
    max_windows: int,
    json_out: pathlib.Path | None,
) -> None:
    """Report window length and window-budget pressure per sentence setting."""
    if not corpora and not source_dir:
        raise click.UsageError("pass --corpus and/or --source-dir")

    encoder, limit = _encoder()
    click.echo(
        f"encoder sequence limit: {limit if limit is not None else 'unreported'} "
        f"word pieces"
        + (
            ""
            if encoder.token_lengths(["probe"]) is not None
            else "   (provider exposes no tokenizer; token columns omitted)"
        )
    )

    samples: list[tuple[str, list[str]]] = [
        (corpus.name, _load_corpus_texts(corpus)) for corpus in corpora
    ]
    if source_dir is not None:
        for size in passage_chars or (1200,):
            samples.append(
                (
                    f"{source_dir.name}@{size}c",
                    _load_source_passages(source_dir, size),
                )
            )

    results: dict[str, list[dict[str, Any]]] = {}
    for name, texts in samples:
        unit_chars = [len(t) for t in texts]
        click.echo(
            f"\n=== {name} ===  {len(texts)} units, "
            f"median {int(statistics.median(unit_chars))} chars, "
            f"max {max(unit_chars)}"
        )
        click.echo(
            f"  {'sent':>4}{'wins':>7}{'ch_med':>8}{'ch_p90':>8}{'ch_max':>8}"
            f"{'tok_med':>9}{'tok_p90':>9}{'>limit':>8}{'ch/tok':>8}"
            f"{'w/unit':>8}{'capped':>8}{'lost_w':>8}"
        )
        rows = []
        for setting in _SENTENCE_SETTINGS:
            row = _measure(texts, setting, max_windows, encoder, limit)
            rows.append(row)
            click.echo(
                f"  {row['max_sentences']:>4}{row['windows']:>7}"
                f"{row['chars_median']:>8}{row['chars_p90']:>8}{row['chars_max']:>8}"
                f"{row.get('tokens_median', 0):>9}{row.get('tokens_p90', 0):>9}"
                f"{row.get('over_limit_pct', 0.0):>7.0f}%"
                f"{row.get('chars_per_token', 0.0):>8.2f}"
                f"{row['windows_per_unit_median']:>8}"
                f"{row['units_hitting_window_cap']:>8}"
                f"{row['windows_dropped_by_cap']:>8}"
            )
        results[name] = rows

    if json_out is not None:
        json_out.write_text(json.dumps(results, indent=2, sort_keys=True))
        click.echo(f"\n-> {json_out}")


if __name__ == "__main__":
    main()
