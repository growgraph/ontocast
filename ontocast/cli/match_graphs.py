"""CLI for matching two RDF graphs and computing PR/F1."""

from __future__ import annotations

import json
import pathlib

import click

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.agg.matcher import GroundTruthSide, MatchRegime, TripleSetMatcher


def _load_graph(path: pathlib.Path) -> RDFGraph:
    if not path.is_file():
        raise click.ClickException(f"File not found: {path}")
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    graph = RDFGraph()
    if suffix in {".json", ".jsonld"}:
        graph.parse(data=text, format="json-ld")
    else:
        graph.parse(data=text, format="turtle")
    return graph


@click.command()
@click.option(
    "--left",
    "left_path",
    required=True,
    type=click.Path(path_type=pathlib.Path, dir_okay=False),
    help="Left RDF graph path (.ttl/.jsonld).",
)
@click.option(
    "--right",
    "right_path",
    required=True,
    type=click.Path(path_type=pathlib.Path, dir_okay=False),
    help="Right RDF graph path (.ttl/.jsonld).",
)
@click.option(
    "--regime",
    type=click.Choice([mode.value for mode in MatchRegime]),
    default=MatchRegime.ONTOLOGY_LOOSE.value,
    show_default=True,
    help="Matching regime.",
)
@click.option(
    "--ground-truth-side",
    type=click.Choice([side.value for side in GroundTruthSide]),
    default=GroundTruthSide.RIGHT.value,
    show_default=True,
    help="Which input graph is considered ground truth.",
)
@click.option(
    "--similarity-threshold",
    type=float,
    default=0.80,
    show_default=True,
    help="Minimum cosine similarity to consider an entity pair.",
)
@click.option(
    "--embedding-model",
    type=str,
    default="paraphrase-multilingual-MiniLM-L12-v2",
    show_default=True,
    help="Sentence-transformers embedding model name.",
)
@click.option(
    "--json-output",
    "json_output",
    is_flag=True,
    default=False,
    help="Print full JSON result payload.",
)
def main(
    left_path: pathlib.Path,
    right_path: pathlib.Path,
    regime: str,
    ground_truth_side: str,
    similarity_threshold: float,
    embedding_model: str,
    json_output: bool,
) -> None:
    """Match two RDF graphs and print equivalence/metric results."""
    if not 0.0 <= similarity_threshold <= 1.0:
        raise click.BadParameter(
            "similarity_threshold must be between 0 and 1",
            param_hint="--similarity-threshold",
        )

    left_graph = _load_graph(left_path.expanduser())
    right_graph = _load_graph(right_path.expanduser())

    matcher = TripleSetMatcher(
        embedding_model=embedding_model,
        similarity_threshold=similarity_threshold,
    )
    result = matcher.match(
        left_graph=left_graph,
        right_graph=right_graph,
        regime=MatchRegime(regime),
        ground_truth_side=GroundTruthSide(ground_truth_side),
    )
    payload = result.model_dump(mode="json")

    click.echo(f"Regime: {payload['regime']}")
    click.echo(f"Ground truth side: {payload['ground_truth_side']}")
    click.echo(f"Entity matches: {len(payload['entity_matches'])}")
    click.echo(
        "Metrics: "
        f"P={payload['metrics']['precision']:.4f} "
        f"R={payload['metrics']['recall']:.4f} "
        f"F1={payload['metrics']['f1']:.4f}"
    )
    if json_output:
        click.echo(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
