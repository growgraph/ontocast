import json
import logging
import pathlib
import sys

import click

from ontocast.util.files import crawl_directories, pdf2markdown

logger = logging.getLogger(__name__)


def process(output_path, f: pathlib.Path):
    fn_json = (output_path / f.name).with_suffix(".json")
    jdata = pdf2markdown(f)
    with open(fn_json, "w", encoding="utf-8") as fpnt:
        json.dump(jdata, fpnt, ensure_ascii=False, indent=4)


@click.command()
@click.option("--input-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--output-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--prefix", type=click.STRING, default=None)
def main(input_path, output_path, prefix):
    input_path = input_path.expanduser()
    output_path = output_path.expanduser()

    try:
        files = sorted(crawl_directories(input_path, suffixes=(".pdf",), prefix=prefix))
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--input-path") from exc
    if not files:
        # Exiting 0 on an empty crawl is indistinguishable from success (#53).
        raise click.ClickException(f"No matching .pdf files under {input_path}.")

    for f in files:
        logger.debug(f"processing {f}")
        process(output_path, f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    main()
