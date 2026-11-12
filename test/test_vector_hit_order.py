"""A channel's hits are cut in one total order, however the store returns ties."""

from __future__ import annotations

import random

import pytest

from ontocast.tool.vector_store.core import GraphAtom, OntologySearchHit, top_hits

pytestmark = pytest.mark.unit


def _hit(name: str, score: float) -> OntologySearchHit:
    atom = GraphAtom(
        atom_id=f"id-{name}",
        ontology_iri="https://ex.org/o",
        iri=f"https://ex.org/o#{name}",
        entity_role="resource",
        core_representation=name,
        neighborhood_representation=name,
    )
    return OntologySearchHit(atom=atom, score=score)


# Five hits tie at 0.5 across the cut at three.
POOL = [
    _hit("a", 0.9),
    _hit("b", 0.7),
    *(_hit(n, 0.5) for n in "vwxyz"),
    _hit("q", 0.1),
]


def _store(seed: int):
    """A store that orders ties at random and returns only the first ``limit``."""
    calls: list[int] = []

    def fetch(limit: int) -> list[OntologySearchHit]:
        calls.append(limit)
        rng = random.Random(seed * 1000 + limit)
        shuffled = sorted(POOL, key=lambda hit: (-hit.score, rng.random()))
        return shuffled[:limit]

    return fetch, calls


def test_ties_at_the_cut_give_the_same_hits_in_the_same_order() -> None:
    results = {
        tuple(hit.atom.iri for hit in top_hits(_store(seed)[0], 3))
        for seed in range(20)
    }

    assert results == {
        ("https://ex.org/o#a", "https://ex.org/o#b", "https://ex.org/o#v")
    }


def test_the_store_is_asked_for_more_until_the_tie_is_seen_whole() -> None:
    fetch, calls = _store(0)

    top_hits(fetch, 3)

    # 6 ends inside the tie (0.5 at both the cut and the end); 12 sees past it.
    assert calls == [6, 12]


def test_fewer_hits_than_the_limit_are_all_returned_in_order() -> None:
    fetch, _ = _store(1)

    hits = top_hits(fetch, 20)

    assert [hit.atom.iri.rsplit("#", 1)[-1] for hit in hits] == [
        "a",
        "b",
        "v",
        "w",
        "x",
        "y",
        "z",
        "q",
    ]
