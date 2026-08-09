import logging
import pathlib

from ontocast.config import ConverterConfig
from ontocast.tool.converter import ConverterTool

logger = logging.getLogger(__name__)


def crawl_directories(
    input_path: pathlib.Path,
    suffixes: tuple[str, ...] = (".pdf", ".json"),
    prefix: str | None = None,
) -> list[pathlib.Path]:
    """Collect input files from a directory tree, or accept a single file.

    Args:
        input_path: A file to use directly, or a directory to search recursively.
        suffixes: Accepted file extensions.
        prefix: When given, keep only files whose stem starts with it.

    Returns:
        Matching paths; empty when a directory holds none. Suffixes match
        case-insensitively (``report.PDF`` is a PDF).

    Raises:
        ValueError: If ``input_path`` does not exist, names a file whose
            suffix is not in ``suffixes``, or names a file excluded by
            ``prefix``. Callers surface this to the user -- returning an empty
            list here would read as "nothing to do" and exit cleanly, which is
            what made a mistyped path silently do nothing.
    """
    file_paths: list[pathlib.Path] = []
    accepted = {suffix.lower() for suffix in suffixes}

    if input_path.is_file():
        if input_path.suffix.lower() not in accepted:
            raise ValueError(
                f"Unsupported input file {input_path}: expected one of "
                f"{', '.join(sorted(suffixes))}"
            )
        if prefix is not None and not input_path.stem.startswith(prefix):
            # An explicitly named file that the prefix filter excludes is a
            # contradiction in the invocation, not an empty result.
            raise ValueError(
                f"Input file {input_path} does not match prefix {prefix!r}."
            )
        return [input_path]

    if not input_path.is_dir():
        raise ValueError(f"The path {input_path} is neither a file nor a directory.")

    for file in input_path.rglob("*"):
        if (
            file.is_file()
            and file.suffix.lower() in accepted
            and (file.stem.startswith(prefix) if prefix is not None else True)
        ):
            file_paths.append(file)
    return file_paths


def pdf2markdown(
    file_path: pathlib.Path,
    converter: ConverterTool | None = None,
    converter_config: ConverterConfig | None = None,
):
    if file_path.suffix == ".pdf":
        if converter is None:
            converter = ConverterTool(converter_config=converter_config)
        doc = converter(file_path)
        return doc.export_to_markdown()
    else:
        raise ValueError(f"Unsupported extension {str(file_path.suffix)}")
