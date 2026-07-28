"""Per-ontology retrieval rank diagnostics.

Answers "why did ontology X contribute nothing to the snapshot?" with numbers rather
than inference. For each ontology reached by any channel, records where its best atom
ranked in the dense core lane, the dense neighborhood lane, the sparse BM25 lane, and
in the fused order -- then whether it survived the ``max_atoms`` cut.

Reading the table:

* ontology absent entirely -> it was never indexed, or its atoms never entered any
  channel's ``top_k``; the problem is upstream of ranking.
* present with ``bm25_rank`` unset but a usable ``core_rank`` -> the sparse lane has no
  lexical surface to match; look at what text is indexed for it.
* present in every channel but ``fused_rank`` beyond ``cutoff_rank`` -> it is losing the
  global race on score, not missing from the index.
* ``fused_rank`` just past ``cutoff_rank`` -> a budget/allocation question.

Collection is opt-in (``ONTOLOGY_PATCH_DUMP_ONTOLOGY_RANKS``) because it walks every
channel hit list for every query.
"""

from __future__ import annotations

from typing import Any

from ontocast.tool.vector_store.core import GraphAtom, OntologySearchHitsByChannel

_CHANNELS = ("core", "neighborhood", "bm25")


def _channel_hit_lists(
    query_hits: OntologySearchHitsByChannel,
) -> dict[str, list[Any]]:
    return {
        "core": query_hits.core_hits,
        "neighborhood": query_hits.neighborhood_hits,
        "bm25": query_hits.bm25_hits,
    }


def build_ontology_rank_diagnostics(
    hits_by_query: list[OntologySearchHitsByChannel],
    ranked_atoms: list[GraphAtom],
    selected_atoms: list[GraphAtom],
) -> dict[str, Any]:
    """Summarize per-ontology ranking outcomes for one ensemble retrieval.

    Args:
        hits_by_query: Raw per-query, per-channel hits, each channel already ordered
            best-first by that channel's own score.
        ranked_atoms: Fused atoms in final order, *before* the ``max_atoms`` cut.
        selected_atoms: Atoms that survived the cut and became subgraph seeds.

    Returns:
        dict[str, Any]: ``{"cutoff_rank": int, "ranked_total": int,
        "ontologies": {iri: {...}}}``. Ranks are 1-based; a channel the ontology never
        appeared in is omitted from its entry rather than reported as zero.
    """
    per_ontology: dict[str, dict[str, Any]] = {}

    def entry(iri: str) -> dict[str, Any]:
        return per_ontology.setdefault(
            iri, {"atoms_ranked": 0, "atoms_selected": 0, "made_cut": False}
        )

    for query_hits in hits_by_query:
        for channel, hits in _channel_hit_lists(query_hits).items():
            for rank, hit in enumerate(hits, start=1):
                iri = hit.atom.ontology_iri
                if not iri:
                    continue
                record = entry(iri)
                rank_key = f"{channel}_rank"
                previous = record.get(rank_key)
                if previous is None or rank < previous:
                    record[rank_key] = rank
                    record[f"{channel}_score"] = round(float(hit.score), 6)

    for rank, atom in enumerate(ranked_atoms, start=1):
        iri = atom.ontology_iri
        if not iri:
            continue
        record = entry(iri)
        record["atoms_ranked"] += 1
        if "fused_rank" not in record:
            record["fused_rank"] = rank
            record["fused_score"] = round(float(atom.score or 0.0), 6)

    for atom in selected_atoms:
        iri = atom.ontology_iri
        if not iri:
            continue
        record = entry(iri)
        record["atoms_selected"] += 1
        record["made_cut"] = True

    return {
        "cutoff_rank": len(selected_atoms),
        "ranked_total": len(ranked_atoms),
        "ontologies": per_ontology,
    }


def render_ontology_rank_diagnostics(diagnostics: dict[str, Any]) -> str:
    """Render :func:`build_ontology_rank_diagnostics` output as a fixed-width table.

    Args:
        diagnostics: Payload produced by :func:`build_ontology_rank_diagnostics`.

    Returns:
        str: Human-readable block, empty when no ontology was reached.
    """
    ontologies: dict[str, dict[str, Any]] = diagnostics.get("ontologies", {})
    if not ontologies:
        return ""

    def sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, str]:
        return (item[1].get("fused_rank", 10**9), item[0])

    header = (
        f"{'ontology':<52}{'core':>10}{'neigh':>10}{'bm25':>10}"
        f"{'fused':>10}{'seeds':>8}  cut"
    )
    lines = [
        f"cutoff_rank={diagnostics.get('cutoff_rank')} "
        f"ranked_total={diagnostics.get('ranked_total')}",
        header,
        "-" * len(header),
    ]
    for iri, record in sorted(ontologies.items(), key=sort_key):
        cells = [str(record.get(f"{channel}_rank", "-")) for channel in _CHANNELS] + [
            str(record.get("fused_rank", "-"))
        ]
        marker = "yes" if record.get("made_cut") else "NO SEEDS"
        short = iri if len(iri) <= 51 else "…" + iri[-50:]
        lines.append(
            f"{short:<52}{cells[0]:>10}{cells[1]:>10}{cells[2]:>10}"
            f"{cells[3]:>10}{record.get('atoms_selected', 0):>8}  {marker}"
        )
    return "\n".join(lines)
