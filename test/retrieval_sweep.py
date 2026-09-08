"""In-process sweep over retrieval parameters against a recall corpus.

The pytest tiers answer "is retrieval working". This answers "which settings work
best", and it exists because that question cannot be answered one knob at a time: the
seed budget, the expansion ceiling and the closure budgets interact, so the useful
object is a *frontier* over combinations rather than a ranking of single deltas.

Everything here is embeddings and graph work -- **no LLM call is made**, so a sweep of
several hundred points costs nothing but wall-clock. That is the whole point: whether a
term the answer needs is present in the snapshot is decidable without generating
anything, and a term absent from the snapshot cannot be used downstream. Snapshot
recall is therefore an upper bound on what any model can link, which makes a cheap
sweep able to *reject* a configuration outright and leaves generation to rank only the
survivors.

The catalog is embedded once and every point scores against that same index: each
swept knob is applied at merge or expansion time, so none of them joins the embedding
fingerprint. Re-indexing per point would dominate the runtime and measure nothing.

Usage::

    ONTOCAST_RECALL_CORPUS=<corpus dir> LANCEDB_ENABLED=true \\
        uv run python -m test.retrieval_sweep --grid grid.json --out results.json

The grid is a JSON list of points, each naming the fields it overrides::

    [{"id": "baseline"},
     {"id": "tight", "vector_store": {"top_k": 10},
      "patch_retrieval": {"small_module_closure_max_total_triples": 450}}]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from ontocast.toolbox import ToolBox
from test.retrieval_gt import RecallCase, corpus_root, load_corpus
from test.retrieval_runner import (
    build_toolbox,
    index_catalog,
    lancedb_store_config,
    score_cases,
)

_SECTIONS = ("vector_store", "patch_retrieval")


def _apply(tools: ToolBox, point: dict[str, Any]) -> dict[str, Any]:
    """Apply one point's overrides to the live toolbox.

    Returns the settings actually in force, read back from the config rather than
    echoed from the request: a knob that silently failed to apply would otherwise be
    reported as though it had, and the whole sweep would attribute its neighbour's
    effect to it.

    Args:
        tools: Toolbox whose retriever and vector-store config are mutated in place.
        point: Grid entry; keys ``vector_store`` and ``patch_retrieval`` hold field
            names mapped to values.

    Returns:
        dict[str, Any]: The applied values, per section.
    """
    tool_config = tools.config.tool_config
    retriever = tools.patch_retriever
    assert retriever is not None
    targets = {
        "vector_store": tool_config.vector_store,
        # The retriever holds its own reference; mutate the object it actually reads.
        "patch_retrieval": retriever.patch,
    }
    applied: dict[str, Any] = {}
    for section in _SECTIONS:
        overrides = point.get(section) or {}
        target = targets[section]
        for field, value in overrides.items():
            info = type(target).model_fields.get(field)
            if info is None:
                raise ValueError(f"{section}: no such setting {field!r}")
            # Validate through the field's own annotation rather than assigning the
            # raw JSON value. A grid is plain JSON, so an enum arrives as a string
            # and reaches code that expects the enum -- which fails at the far end
            # of a long run, on whichever point happened to draw it.
            setattr(target, field, TypeAdapter(info.annotation).validate_python(value))
        applied[section] = {field: getattr(target, field) for field in overrides}
    return applied


def _defaults(tools: ToolBox, grid: list[dict[str, Any]]) -> dict[str, Any]:
    """Starting value of every field the grid touches, for restoring between points.

    A point overriding fewer fields than its predecessor must not inherit the
    predecessor's values, or the sweep measures a running accumulation instead of the
    combinations it lists.
    """
    tool_config = tools.config.tool_config
    retriever = tools.patch_retriever
    assert retriever is not None
    targets = {
        "vector_store": tool_config.vector_store,
        "patch_retrieval": retriever.patch,
    }
    seen: dict[str, dict[str, Any]] = {section: {} for section in _SECTIONS}
    for point in grid:
        for section in _SECTIONS:
            for field in point.get(section) or {}:
                if field not in seen[section]:
                    seen[section][field] = getattr(targets[section], field)
    return seen


def _restore(tools: ToolBox, defaults: dict[str, Any]) -> None:
    retriever = tools.patch_retriever
    assert retriever is not None
    targets = {
        "vector_store": tools.config.tool_config.vector_store,
        "patch_retrieval": retriever.patch,
    }
    for section, fields in defaults.items():
        for field, value in fields.items():
            setattr(targets[section], field, value)


async def _sweep(
    tools: ToolBox,
    ontologies: list[Any],
    cases: list[RecallCase],
    grid: list[dict[str, Any]],
    on_row: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    await index_catalog(tools, ontologies)
    defaults = _defaults(tools, grid)
    rows: list[dict[str, Any]] = []
    for index, point in enumerate(grid, start=1):
        point_id = str(point.get("id") or f"p{index:04d}")
        _restore(tools, defaults)
        applied = _apply(tools, point)
        started = time.perf_counter()
        counts = await score_cases(tools, ontologies, cases)
        row = {
            "id": point_id,
            "applied": applied,
            "seconds": round(time.perf_counter() - started, 2),
            **counts.as_dict(),
        }
        rows.append(row)
        # Persist after every point. A sweep is long enough that losing it to an
        # interrupted session costs more than the write does, and a partial
        # frontier is still readable -- the points are independent.
        if on_row is not None:
            on_row(rows)
        print(
            f"[{index:>4}/{len(grid)}] {point_id:<24} "
            f"seed {row['seed_term_recall']:6.1%}  "
            f"snapshot {row['snapshot_term_recall']:6.1%}  "
            f"triples {row['mean_snapshot_triples']:8.1f}  "
            f"on-topic {row['on_topic_precision']:6.1%}",
            flush=True,
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", required=True, type=Path, help="JSON list of points")
    parser.add_argument("--out", required=True, type=Path, help="JSON results file")
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="LanceDB directory; reused across points (default: a temp dir)",
    )
    args = parser.parse_args(argv)

    root = corpus_root()
    if root is None:
        parser.error("set ONTOCAST_RECALL_CORPUS to a corpus directory")
    cases, ontologies = load_corpus(root)
    grid = json.loads(args.grid.read_text())
    if not isinstance(grid, list) or not grid:
        parser.error(f"{args.grid}: expected a non-empty JSON list of points")

    index_dir = args.index_dir or (args.out.parent / ".sweep-index")
    index_dir.mkdir(parents=True, exist_ok=True)
    ontology_dir = index_dir / "ontologies"
    ontology_dir.mkdir(exist_ok=True)

    tools = build_toolbox(lancedb_store_config(index_dir, "sweep"), ontology_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def write(rows: list[dict[str, Any]]) -> None:
        args.out.write_text(
            json.dumps(
                {
                    "corpus": root.name,
                    "cases": len(cases),
                    "ontologies": len(ontologies),
                    "requested_points": len(grid),
                    "points": rows,
                },
                indent=2,
                sort_keys=True,
            )
        )

    rows = asyncio.run(_sweep(tools, ontologies, cases, grid, on_row=write))
    write(rows)
    print(f"\n{len(rows)} points -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
