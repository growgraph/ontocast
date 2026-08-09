"""Inspect section classification for a document without running extraction.

Section labels drive which parts of a document are extracted, so a wrong label
silently changes the output: a chunk filtered out by ``--target-sections`` never
reaches the pipeline and never appears in a log. This command shows the
classifier's decisions -- the detected outline and the resulting chunk labels,
each with the tier that decided it -- before any extraction cost is incurred.

Example:
    ontocast sections --input-path ./paper.pdf
    ontocast sections --input-path ./paper.pdf --target-sections results --json
"""

from __future__ import annotations

import asyncio
import json
import pathlib
from typing import Literal, cast

import click

from ontocast.api.parse import (
    parse_document_type_hint_param,
    parse_section_schema_id_param,
    parse_sections_list_param,
)
from ontocast.config import Config
from ontocast.onto.docling_helpers import json_payload_text, plain_text_to_docling_doc
from ontocast.tool.cache import Cacher
from ontocast.tool.chunk.chunker import ChunkerTool
from ontocast.tool.chunk.outline import build_document_outline
from ontocast.tool.chunk.prepare import (
    PrepareOptions,
    prepare_content_units,
    resolve_prepare_schema,
)
from ontocast.tool.chunk.sections import document_text_for_section_tagging
from ontocast.tool.converter import ConverterTool
from ontocast.toolbox import ToolBox


def _load_document(path: pathlib.Path, converter: ConverterTool):
    """Read a document the same way the Convert node does.

    JSON and plain-text inputs are routed *around* the Docling converter -- it
    rejects them outright -- so inspecting the very files the pipeline is
    normally driven with (``data/json/*.json``) requires mirroring that routing
    rather than calling the converter for everything.
    """
    suffix = path.suffix.lower()
    if suffix in converter.supported_extensions:
        return converter(path)
    if suffix == ".json":
        text = json_payload_text(json.loads(path.read_text(encoding="utf-8")))
        if text is None:
            raise click.ClickException(
                f"{path} holds no document text (expected a 'text' field)"
            )
        return plain_text_to_docling_doc(text, path.name)
    if suffix in (".txt", ".md", ".markdown"):
        return plain_text_to_docling_doc(path.read_text(encoding="utf-8"), path.name)
    raise click.ClickException(f"Unsupported input type {suffix!r} for {path}")


def _outline_rows(document_text: str, schema, text_headings: bool) -> list[dict]:
    outline = build_document_outline(
        document_text, schema, include_text_headings=text_headings
    )
    return [
        {
            "offset": node.start,
            "level": node.level,
            "kind": "section" if node.sectionlike else "sub/desc",
            "heading": node.text,
            "normalised": node.normalised,
            "label": node.label,
            "source": node.source.value,
            "confidence": round(node.confidence, 2),
        }
        for node in outline.nodes
    ]


def _chunk_rows(chunks) -> list[dict]:
    return [
        {
            "index": index,
            "chars": len(chunk.text),
            "label": chunk.section_label,
            "source": (
                chunk.section_label_source.value
                if chunk.section_label_source is not None
                else None
            ),
            "confidence": round(chunk.section_label_confidence, 2),
            "preview": " ".join(chunk.text.split())[:70],
        }
        for index, chunk in enumerate(chunks)
    ]


def _print_table(title: str, rows: list[dict], columns: list[tuple[str, int]]) -> None:
    click.echo(click.style(f"\n{title}", bold=True))
    if not rows:
        click.echo("  (none)")
        return
    header = "  ".join(name[:width].ljust(width) for name, width in columns)
    click.echo(f"  {header}")
    click.echo(f"  {'-' * len(header)}")
    for row in rows:
        cells = []
        for name, width in columns:
            value = row.get(name)
            text = "-" if value is None else str(value)
            cells.append(text[:width].ljust(width))
        click.echo(f"  {'  '.join(cells)}")


@click.command("sections")
@click.option(
    "--input-path",
    type=click.Path(exists=True, path_type=pathlib.Path),
    required=True,
    help="Document to inspect (PDF, markdown, text or JSON).",
)
@click.option(
    "--section-schema-id",
    default=None,
    help="Section label schema: academic, financial, legal, clinical, manual, "
    "fiction, patent, standard, news, general. Omit to detect it.",
)
@click.option(
    "--document-type-hint",
    default=None,
    help="Free-text document type; overrides detection when it matches.",
)
@click.option(
    "--target-sections",
    default=None,
    help="Comma-separated labels to keep, to preview what a filtered run selects.",
)
@click.option(
    "--exclude-sections",
    default=None,
    help="Comma-separated labels to drop; empty string disables exclusion.",
)
@click.option(
    "--section-classifier",
    type=click.Choice(["off", "heading", "heuristic", "llm"]),
    default=None,
    help="Override CHUNK_SECTION_CLASSIFIER. Only 'llm' makes LLM calls.",
)
@click.option("--as-json", "as_json", is_flag=True, help="Emit JSON instead of tables.")
def main(
    input_path: pathlib.Path,
    section_schema_id: str | None,
    document_type_hint: str | None,
    target_sections: str | None,
    exclude_sections: str | None,
    section_classifier: str | None,
    as_json: bool,
) -> None:
    """Show the detected section outline and per-chunk labels for a document."""
    config = Config()
    tool_config = config.get_tool_config()
    chunk_config = tool_config.chunk_config
    if section_classifier is not None:
        chunk_config.section_classifier = cast(
            Literal["llm", "heuristic", "heading", "off"], section_classifier
        )
    # A diagnostic must survive the condition it diagnoses: the empty-selection
    # gate is a pipeline policy, and "0 chunks kept" is exactly the answer this
    # command exists to show.
    chunk_config.section_filter_on_empty = "warn"

    # Only the 'llm' tier needs an LLM, and building a full ToolBox requires
    # provider credentials. Inspecting a document must stay free, so the
    # deterministic tiers get just the converter and chunker.
    cache = Cacher(config=config)
    converter = ConverterTool(
        cache=cache, converter_config=tool_config.converter_config
    )
    chunker = ChunkerTool(chunk_config=chunk_config, cache=cache)
    tools = ToolBox(config) if chunk_config.section_classifier == "llm" else None

    docling_doc = _load_document(input_path, converter)
    document_text = document_text_for_section_tagging(docling_doc)

    hint = parse_document_type_hint_param(document_type_hint)
    schema_id = parse_section_schema_id_param(section_schema_id)

    options = PrepareOptions(
        section_schema_id=schema_id,
        document_type_hint=hint,
        # None and [] mean different things downstream (schema defaults versus
        # exclusion disabled), so an omitted flag must stay None.
        target_sections=(
            parse_sections_list_param(target_sections, param="target-sections")
            if target_sections is not None
            else None
        ),
        exclude_sections=(
            parse_sections_list_param(exclude_sections, param="exclude-sections")
            if exclude_sections is not None
            else None
        ),
    )
    # Resolved through the same helper the pipeline uses, so the schema reported
    # here is necessarily the schema the chunks were actually labeled against.
    decision = resolve_prepare_schema(document_text, chunk_config, options, chunker)
    schema = decision.schema
    resolved_id = decision.schema_id

    chunks = asyncio.run(
        prepare_content_units(docling_doc, chunker, chunk_config, options, tools)
    )

    outline = _outline_rows(document_text, schema, chunk_config.section_text_headings)
    chunk_rows = _chunk_rows(chunks)
    labeled = sum(1 for row in chunk_rows if row["label"] is not None)
    schema_rows = [
        {
            "candidate": ev.schema_id,
            "score": round(ev.score, 1),
            "share": f"{ev.share:.0%}",
            "evidence": ", ".join(ev.examples[:3]),
        }
        for ev in (decision.detection.evidence[:5] if decision.detection else [])
        if ev.score > 0
    ]
    summary = {
        "input": str(input_path),
        "schema": resolved_id,
        "schema_source": decision.source,
        "schema_tier": decision.detection.tier if decision.detection else None,
        "schema_margin": (
            round(decision.detection.margin, 2) if decision.detection else None
        ),
        "classifier": chunk_config.section_classifier,
        "density": chunk_config.section_density,
        "document_chars": len(document_text),
        "headings": len(outline),
        "chunks": len(chunk_rows),
        "labeled_chunks": labeled,
    }

    if as_json:
        click.echo(
            json.dumps(
                {
                    "summary": summary,
                    "schema_candidates": schema_rows,
                    "outline": outline,
                    "chunks": chunk_rows,
                },
                indent=2,
            )
        )
        return

    provenance = decision.source
    if decision.detection is not None:
        margin = decision.detection.margin
        # An infinite margin means no other schema scored at all -- "infx" reads
        # as a formatting bug rather than as the strongest possible evidence.
        margin_text = (
            "uncontested" if margin == float("inf") else f"{margin:.1f}x margin"
        )
        provenance = (
            f"{decision.source} via {decision.detection.tier} tier, {margin_text}"
        )
    click.echo(click.style(f"{input_path}", bold=True))
    click.echo(
        f"  schema={resolved_id} ({provenance})  "
        f"classifier={chunk_config.section_classifier}  "
        f"density={chunk_config.section_density}  chars={len(document_text)}"
    )
    if schema_rows:
        _print_table(
            "Schema candidates",
            schema_rows,
            [("candidate", 12), ("score", 6), ("share", 6), ("evidence", 60)],
        )
    _print_table(
        "Outline",
        outline,
        [
            ("offset", 7),
            ("kind", 8),
            ("label", 16),
            ("source", 18),
            ("confidence", 5),
            ("heading", 60),
        ],
    )
    _print_table(
        "Chunks",
        chunk_rows,
        [
            ("index", 5),
            ("chars", 6),
            ("label", 16),
            ("source", 18),
            ("confidence", 5),
            ("preview", 60),
        ],
    )
    click.echo(
        f"\n  {labeled}/{len(chunk_rows)} chunks labeled "
        f"({len(chunk_rows) - labeled} unresolved)"
    )


if __name__ == "__main__":
    main()
