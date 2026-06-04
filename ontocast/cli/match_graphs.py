"""Match two TTL graphs locally (same pipeline as /match/* HTTP APIs)."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import click
from rdflib.term import Node

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.agg.entity_aligner import EntityAligner
from ontocast.tool.agg.match_common import (
    collect_ontology_entities,
    prepare_fact_triples,
    prepare_metric_triples,
    project_triples,
)
from ontocast.tool.agg.match_derivation import derive_pair_matches
from ontocast.tool.agg.match_models import (
    EntityMatch,
    MatchRegime,
    TaggedGraph,
    as_uri_ref,
)
from ontocast.tool.agg.triple_evaluator import TripleSetEvaluator

GT_GRAPH_ID = "gt"
PREDICTED_GRAPH_ID = "predicted"


def _load_ttl(path: pathlib.Path) -> RDFGraph:
    graph = RDFGraph()
    graph.parse(data=path.read_text(encoding="utf-8"), format="turtle")
    return graph


def _format_triple(triple: tuple[Node, Node, Node]) -> str:
    return f"({triple[0]!s}, {triple[1]!s}, {triple[2]!s})"


def _run_match(
    gt_graph: RDFGraph,
    predicted_graph: RDFGraph,
    *,
    regime: MatchRegime,
    similarity_threshold: float,
    embedding_model: str,
) -> tuple[dict[str, Any], list[EntityMatch]]:
    aligner = EntityAligner(
        embedding_model=embedding_model,
        similarity_threshold=similarity_threshold,
    )
    alignment = aligner.align_graphs(
        [
            TaggedGraph(id=GT_GRAPH_ID, graph=gt_graph),
            TaggedGraph(id=PREDICTED_GRAPH_ID, graph=predicted_graph),
        ],
        regime=regime,
    )
    entity_matches = derive_pair_matches(
        alignment.clusters,
        predicted_graph_id=PREDICTED_GRAPH_ID,
        gt_graph_id=GT_GRAPH_ID,
        similarity_threshold=similarity_threshold,
    )
    metrics = TripleSetEvaluator().evaluate(
        predicted_graph=predicted_graph,
        gt_graph=gt_graph,
        entity_matches=entity_matches,
    )
    payload: dict[str, Any] = {
        "regime": regime.value,
        "similarity_threshold": similarity_threshold,
        "embedding_model": embedding_model,
        "entity_count": alignment.entity_count,
        "cluster_count": alignment.cluster_count,
        "entity_match_count": len(entity_matches),
        "metrics": metrics.model_dump(mode="json"),
    }
    return payload, entity_matches


def _print_verbose(
    gt_graph: RDFGraph,
    predicted_graph: RDFGraph,
    entity_matches: list[EntityMatch],
    metrics_payload: dict[str, Any],
) -> None:
    click.echo("\n--- entity matches (predicted -> gt) ---")
    for match in sorted(
        entity_matches,
        key=lambda item: (-item.similarity, str(item.predicted_entity)),
    ):
        click.echo(
            f"  {match.similarity:.4f}  {match.predicted_entity}  ->  {match.gt_entity}"
        )

    predicted_to_gt = {
        as_uri_ref(matched.predicted_entity): as_uri_ref(matched.gt_entity)
        for matched in entity_matches
    }
    raw_predicted = project_triples(predicted_graph, predicted_to_gt)
    predicted = prepare_metric_triples(raw_predicted)
    ground_truth = prepare_metric_triples(set(gt_graph))
    true_positives = predicted & ground_truth
    false_positives = predicted - ground_truth
    false_negatives = ground_truth - predicted

    click.echo("\n--- informative triples (after URI projection, labels excluded) ---")
    click.echo(f"TP ({len(true_positives)}):")
    for triple in sorted(true_positives, key=_format_triple):
        click.echo(f"  + {_format_triple(triple)}")
    click.echo(f"FP ({len(false_positives)}):")
    for triple in sorted(false_positives, key=_format_triple):
        click.echo(f"  ! {_format_triple(triple)}")
    click.echo(f"FN ({len(false_negatives)}):")
    for triple in sorted(false_negatives, key=_format_triple):
        click.echo(f"  - {_format_triple(triple)}")

    ontology_entities = collect_ontology_entities(predicted | ground_truth)
    predicted_facts = prepare_fact_triples(predicted, ontology_entities)
    ground_truth_facts = prepare_fact_triples(ground_truth, ontology_entities)
    fact_tp = predicted_facts & ground_truth_facts
    fact_fp = predicted_facts - ground_truth_facts
    fact_fn = ground_truth_facts - predicted_facts

    click.echo("\n--- fact triples (domain s/o, non-schema predicates) ---")
    click.echo(f"TP ({len(fact_tp)}):")
    for triple in sorted(fact_tp, key=_format_triple):
        click.echo(f"  + {_format_triple(triple)}")
    click.echo(f"FP ({len(fact_fp)}):")
    for triple in sorted(fact_fp, key=_format_triple):
        click.echo(f"  ! {_format_triple(triple)}")
    click.echo(f"FN ({len(fact_fn)}):")
    for triple in sorted(fact_fn, key=_format_triple):
        click.echo(f"  - {_format_triple(triple)}")

    m = metrics_payload["metrics"]
    click.echo("\n--- summary ---")
    click.echo(
        f"triple  P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} "
        f"(tp={m['true_positives']} fp={m['false_positives']} fn={m['false_negatives']})"
    )
    click.echo(
        f"entity  P={m['entity_precision']:.4f} R={m['entity_recall']:.4f} "
        f"F1={m['entity_f1']:.4f}"
    )
    click.echo(
        f"fact    P={m['fact_precision']:.4f} R={m['fact_recall']:.4f} "
        f"F1={m['fact_f1']:.4f}"
    )


@click.command()
@click.option(
    "--gt",
    "gt_path",
    required=True,
    type=click.Path(path_type=pathlib.Path),
    help="Ground-truth TTL file.",
)
@click.option(
    "--predicted",
    "predicted_path",
    required=True,
    type=click.Path(path_type=pathlib.Path),
    help="Predicted / generated TTL file.",
)
@click.option(
    "--regime",
    type=click.Choice(["ontology_loose", "ontology_strict"], case_sensitive=False),
    default="ontology_loose",
    show_default=True,
)
@click.option(
    "--similarity-threshold",
    type=float,
    default=0.80,
    show_default=True,
)
@click.option(
    "--embedding-model",
    type=str,
    default="paraphrase-multilingual-MiniLM-L12-v2",
    show_default=True,
)
@click.option(
    "--json-out",
    "json_out",
    default=None,
    type=click.Path(dir_okay=False, path_type=pathlib.Path),
    help="Write full result JSON (metrics, alignment counts, entity_matches).",
)
@click.option(
    "--verbose/--no-verbose",
    default=True,
    help="Print entity matches and triple-level TP/FP/FN.",
)
def main(
    gt_path: pathlib.Path,
    predicted_path: pathlib.Path,
    regime: str,
    similarity_threshold: float,
    embedding_model: str,
    json_out: pathlib.Path | None,
    verbose: bool,
) -> None:
    """Align and score two TTL graphs (same logic as validation match_triples.py)."""
    if not 0.0 <= similarity_threshold <= 1.0:
        raise click.BadParameter(
            "similarity_threshold must be between 0 and 1",
            param_hint="--similarity-threshold",
        )

    gt_path = gt_path.expanduser().resolve()
    predicted_path = predicted_path.expanduser().resolve()
    click.echo(f"GT:         {gt_path}")
    click.echo(f"Predicted:  {predicted_path}")

    gt_graph = _load_ttl(gt_path)
    predicted_graph = _load_ttl(predicted_path)
    click.echo(
        f"GT triples: {len(gt_graph)}  Predicted triples: {len(predicted_graph)}"
    )

    payload, entity_matches = _run_match(
        gt_graph,
        predicted_graph,
        regime=MatchRegime(regime),
        similarity_threshold=similarity_threshold,
        embedding_model=embedding_model,
    )
    payload["entity_matches"] = [
        match.model_dump(mode="json") for match in entity_matches
    ]

    if verbose:
        _print_verbose(gt_graph, predicted_graph, entity_matches, payload)
    else:
        m = payload["metrics"]
        click.echo(
            f"\nP={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} | "
            f"entity F1={m['entity_f1']:.4f} | fact F1={m['fact_f1']:.4f} | "
            f"matches={payload['entity_match_count']}"
        )

    if json_out is not None:
        json_out = json_out.expanduser().resolve()
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        click.echo(f"Wrote {json_out}")


if __name__ == "__main__":
    main()
