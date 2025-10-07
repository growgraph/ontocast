import json
import logging
import pathlib
import sys

import click
from suthing import FileHandle

from ontocast.cli.util import crawl_directories
from ontocast.tool.chunk.chunker import ChunkerTool

logger = logging.getLogger(__name__)


def json_to_md(data: dict | list, title: str = "JSON Data", depth: int = 1) -> str:
    """
    Convert nested JSON data to Markdown format.

    Args:
        data: The JSON data (dict or list)
        title: Title for the top-level markdown document
        depth: Current heading depth (internal recursion parameter)

    Returns:
        Markdown formatted string
    """
    if not data:
        return ""

    md = []

    # Add title only at the top level
    if depth == 1 and title:
        md.append(f"# {title}\n\n")

    if isinstance(data, dict):
        # First, handle simple key-value pairs
        simple_pairs = []
        complex_items = []

        for key, value in data.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                simple_pairs.append((key, value))
            else:
                complex_items.append((key, value))

        # Add simple pairs first
        for key, value in simple_pairs:
            md.append(f"**{key}**: {_format_value(value)}\n")

        if simple_pairs:
            md.append("\n")

        # Then handle complex items with headers
        for key, value in complex_items:
            header_level = depth + 1

            # Clean spacing - add extra newline before major sections
            if header_level < 3 and md and not md[-1].endswith("\n\n"):
                md.append("\n")

            md.append(f"{'#' * header_level} {key}\n")

            if isinstance(value, dict):
                if value:  # Skip empty dicts
                    md.append(json_to_md(value, title=f"doc: {key}", depth=depth + 1))
            elif isinstance(value, list):
                md.extend(_handle_list(value, header_level, depth))

    elif isinstance(data, list):
        md.extend(_handle_list(data, depth, depth))

    return "".join(md)


def _handle_list(items: list, header_level: int, depth: int) -> list[str]:
    """Handle list items and return markdown lines."""
    md = []

    if not items:
        md.append("*Empty list*\n\n")
        return md

    if all(isinstance(item, (str, int, float, bool, type(None))) for item in items):
        # Simple list items
        for item in items:
            md.append(f"- {_format_value(item)}\n")
        md.append("\n")
    else:
        # Complex list items
        for i, item in enumerate(items):
            if isinstance(item, dict) and len(items) > 1:
                md.append(f"{'#' * (header_level + 1)} Item {i + 1}\n\n")
            md.append(json_to_md(item, title=f"# doc {i}", depth=depth + 1))

    return md


def _format_value(value) -> str:
    """Format a simple value for markdown output."""
    if value is None:
        return "*null*"
    elif isinstance(value, bool):
        return str(value).lower()
    elif isinstance(value, str):
        # Escape markdown special characters in content
        return value.replace("*", r"-")
    else:
        return str(value)


def process(fn_json: pathlib.Path, output_path: pathlib.Path, chunker: ChunkerTool):
    logger.debug(f"Processing fn_json: {fn_json}")

    jdata = FileHandle.load(fn_json)

    text = jdata.get("text", None)
    if text is None:
        if len(jdata.keys()) > 2:
            md_lines = json_to_md(jdata)
            text = "".join(md_lines)
        else:
            raise ValueError(f"Not sure about the json format {fn_json}")

    docs_txt = chunker(text)

    sizes = [len(x) for x in docs_txt]
    logger.debug(f"Chunk size: {sizes}")

    chunked = {"chunks": docs_txt}

    logger.debug(f"Saving to {output_path / fn_json.name}")

    with open(output_path / fn_json.name, "w", encoding="utf-8") as f:
        json.dump(chunked, f, ensure_ascii=False, indent=4)

    return docs_txt


@click.command()
@click.option("--input-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--output-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--prefix", type=click.STRING, default=None)
def main(input_path, output_path, prefix):
    input_path = input_path.expanduser()
    output_path = output_path.expanduser()

    chunker = ChunkerTool(
        breakpoint_threshold_amount=95,
        breakpoint_threshold_type="percentile",
        max_chunk_size=20000,
        model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )

    files = sorted(
        crawl_directories(input_path.expanduser(), suffixes=(".json",), prefix=prefix)
    )

    for f in files:
        process(f, output_path, chunker)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    main()
