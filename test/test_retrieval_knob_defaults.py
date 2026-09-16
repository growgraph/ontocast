"""Every retrieval knob added for tuning must be inert at its default.

A sweep is read against its baseline point. If adding a knob moved that baseline,
every comparison to previously measured behaviour would silently shift, and the
knob would be credited or blamed for a change it did not cause. So each of these
asserts the same thing from a different angle: the default reproduces what the
code did before the knob existed.
"""

from __future__ import annotations

import logging

import pytest

from ontocast.config import EmbeddingConfig, VectorStoreConfig
from ontocast.tool.chunk.proposition import split_proposition_windows
from ontocast.tool.vector_store.core import GraphAtom, OntologySearchHit
from ontocast.tool.vector_store.embedding import EmbeddingTool
from ontocast.tool.vector_store.util import (
    effective_bm25_top_k,
    effective_top_k,
    rank_fuse_channel_hits,
)


def _hit(atom_id: str, score: float) -> OntologySearchHit:
    return OntologySearchHit(
        atom=GraphAtom(
            atom_id=atom_id,
            ontology_iri="https://example.org/onto",
            iri=f"https://example.org/onto#{atom_id}",
            core_representation=atom_id,
            neighborhood_representation=atom_id,
        ),
        score=score,
    )


def _fuse(hits: list[OntologySearchHit], **kwargs) -> list[str]:
    fused = rank_fuse_channel_hits(
        hits,
        [],
        list(reversed(hits)),
        core_weight=0.5,
        neighborhood_weight=0.1,
        bm25_weight=0.4,
        limit=10,
        **kwargs,
    )
    return [hit.atom.atom_id for hit in fused]


def test_bm25_top_k_unset_follows_the_dense_lanes() -> None:
    """The knob is opt-in: unset, the sparse lane is as deep as the dense ones."""
    config = VectorStoreConfig(top_k=17)
    assert config.bm25_top_k is None
    assert effective_bm25_top_k(config, None) == effective_top_k(config, None) == 17
    # An explicit per-call top_k still reaches the sparse lane when unset.
    assert effective_bm25_top_k(config, 5) == 5


def test_bm25_top_k_when_set_overrides_even_an_explicit_top_k() -> None:
    """A configured sparse depth is a deliberate statement about that lane."""
    config = VectorStoreConfig(top_k=20, bm25_top_k=3)
    assert effective_bm25_top_k(config, None) == 3
    assert effective_bm25_top_k(config, 40) == 3
    assert effective_top_k(config, 40) == 40


def test_fusion_rank_constant_defaults_to_the_unsmoothed_behaviour() -> None:
    """At 0 the fused scores are exactly weight/rank, as before the knob existed."""
    assert VectorStoreConfig().fusion_rank_constant == 0.0
    hits = [_hit(f"a{i}", 1.0 - i / 10) for i in range(5)]
    fused = rank_fuse_channel_hits(
        hits,
        [],
        [],
        core_weight=1.0,
        neighborhood_weight=0.0,
        bm25_weight=0.0,
        limit=5,
    )
    assert [round(hit.score, 6) for hit in fused] == [
        round(1.0 / rank, 6) for rank in range(1, 6)
    ]
    assert _fuse(hits) == _fuse(hits, rank_constant=0.0)


def test_fusion_rank_constant_flattens_the_decay() -> None:
    """Raising it makes cross-lane agreement outweigh position within one lane."""
    hits = [_hit(f"a{i}", 1.0 - i / 10) for i in range(4)]
    steep = rank_fuse_channel_hits(
        hits,
        [],
        [],
        core_weight=1.0,
        neighborhood_weight=0.0,
        bm25_weight=0.0,
        limit=4,
    )
    flat = rank_fuse_channel_hits(
        hits,
        [],
        [],
        core_weight=1.0,
        neighborhood_weight=0.0,
        bm25_weight=0.0,
        limit=4,
        rank_constant=60.0,
    )
    steep_gap = steep[0].score - steep[1].score
    flat_gap = flat[0].score - flat[1].score
    assert flat_gap < steep_gap
    # Order within a single lane is unchanged; only the spacing is.
    assert [h.atom.atom_id for h in steep] == [h.atom.atom_id for h in flat]


TEXT = "One a. Two b. Three c. Four d. Five e."


def test_window_stride_defaults_to_contiguous_windows() -> None:
    """Unset stride reproduces the disjoint windows the splitter always made."""
    assert VectorStoreConfig().proposition_window_stride is None
    assert split_proposition_windows(
        TEXT, max_sentences=2
    ) == split_proposition_windows(TEXT, max_sentences=2, stride=2)


def test_window_stride_below_the_window_size_overlaps() -> None:
    """A statement straddling a boundary needs some window to contain both halves."""
    disjoint = split_proposition_windows(TEXT, max_sentences=2)
    overlapped = split_proposition_windows(TEXT, max_sentences=2, stride=1)
    assert "Two b. Three c." not in disjoint
    assert "Two b. Three c." in overlapped
    # Tail suffixes must not be emitted twice; an identical query would be paid for
    # again and would skew the ranks it feeds.
    assert len(overlapped) == len(set(overlapped))


def test_window_stride_must_be_positive() -> None:
    with pytest.raises(ValueError):
        split_proposition_windows(TEXT, stride=0)


def test_window_sentences_is_no_longer_capped_at_four() -> None:
    """The cap sat just short of where widening starts discarding query text.

    Which limit actually binds is the embedding model's sequence length, and that
    is reported rather than guessed -- see the truncation telemetry.
    """
    assert (
        VectorStoreConfig(proposition_window_sentences=8).proposition_window_sentences
        == 8
    )


def test_sequence_limit_is_unknown_for_providers_that_do_not_state_one() -> None:
    """The base contract returns None rather than inventing a limit."""

    class _Tool(EmbeddingTool):
        def _embed_raw(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] for _ in texts]

    tool = _Tool(config=EmbeddingConfig())
    assert tool.sequence_limit is None
    assert tool.count_over_limit(["anything"]) is None


# ------------------------------- configurations that cannot mean what they say


def test_mmr_with_atom_floors_is_rejected() -> None:
    """Two selection policies for one budget: the config must not pick silently.

    MMR replaces the selection outright, so a non-zero floor stops being a
    guarantee under it -- with no error and no metric that would show it.
    """
    from pydantic import ValidationError

    from ontocast.config import PatchRetrievalConfig, ToolConfig

    with pytest.raises(ValidationError, match="MMR_LAMBDA"):
        ToolConfig(
            patch_retrieval=PatchRetrievalConfig(
                mmr_lambda=0.5, per_ontology_atom_floor=2, per_role_atom_floor=12
            )
        )


def test_mmr_is_configurable_once_the_floors_are_off() -> None:
    from ontocast.config import PatchRetrievalConfig, ToolConfig

    config = ToolConfig(
        patch_retrieval=PatchRetrievalConfig(
            mmr_lambda=0.5, per_ontology_atom_floor=0, per_role_atom_floor=0
        )
    )
    assert config.patch_retrieval.mmr_lambda == 0.5


def test_the_default_config_is_not_rejected() -> None:
    """The floors ship non-zero and MMR ships off, so the default must construct."""
    from ontocast.config import ToolConfig

    assert ToolConfig().patch_retrieval.mmr_lambda == 1.0


def test_unreachable_max_atoms_warns_naming_the_cap(caplog) -> None:
    """`MAX_ATOMS` above the window-scaled ceiling is inert, and says so.

    A third of a 121-point sweep once measured nothing because every point
    above the ceiling was the same run.
    """
    from ontocast.config import PatchRetrievalConfig, ToolConfig, VectorStoreConfig

    with caplog.at_level(logging.WARNING):
        ToolConfig(
            patch_retrieval=PatchRetrievalConfig(max_atoms=192, max_atoms_base=96),
            vector_store=VectorStoreConfig(proposition_max_windows=16),
        )
    assert "ONTOLOGY_PATCH_MAX_ATOMS=192 cannot bind" in caplog.text
    assert "PROPOSITION_MAX_WINDOWS" in caplog.text


def test_reachable_max_atoms_does_not_warn(caplog) -> None:
    from ontocast.config import PatchRetrievalConfig, ToolConfig

    with caplog.at_level(logging.WARNING):
        ToolConfig(
            patch_retrieval=PatchRetrievalConfig(max_atoms=96, max_atoms_base=96)
        )
    assert "cannot bind" not in caplog.text


@pytest.mark.anyio
@pytest.mark.parametrize("rank_constant", [0.0, 5.0, 60.0])
async def test_the_relevance_gate_means_the_same_at_every_rank_constant(
    rank_constant: float,
) -> None:
    """The gate is a fraction of the attainable best, not a number on the scale.

    The rank constant divides every fused score by about ``1 + constant``. Read
    as an absolute threshold, a gate calibrated without smoothing therefore
    rejected *every* candidate once smoothing was on -- and the symptom, an
    empty ontology context, reads as a retrieval failure rather than as a
    miscalibration. Scaling the gate by the same factor is what lets smoothing
    be tested on its own merits.
    """
    hits = [_hit("keep", 0.9), _hit("also", 0.6)]
    kept = await _retrieve_with(
        hits, min_merged_max_score=0.18, rank_constant=rank_constant
    )
    assert kept, f"gate emptied the snapshot at rank_constant={rank_constant}"


@pytest.mark.anyio
async def test_a_gate_above_the_attainable_best_still_empties_the_snapshot() -> None:
    """The gate must still be able to say "nothing here is relevant"."""
    hits = [_hit("weak", 0.9)]
    assert not await _retrieve_with(hits, min_merged_max_score=1.5, rank_constant=0.0)


async def _retrieve_with(
    hits: list[OntologySearchHit], *, min_merged_max_score: float, rank_constant: float
) -> list[str]:
    """Entity IRIs surviving one retrieval at the given gate and smoothing."""
    from ontocast.config import (
        EmbeddingConfig,
        PatchRetrievalConfig,
        QdrantConfig,
        VectorStoreConfig,
    )
    from ontocast.tool.vector_store.patch_retriever import OntologyPatchRetriever
    from test.test_vector_store_pipeline import (
        CountingEmbeddingTool,
        StubSPARQLTool,
        StubVectorStore,
        _channel_hits,
    )

    vector_store = StubVectorStore(
        store_config=VectorStoreConfig(fusion_rank_constant=rank_constant),
        qdrant_config=QdrantConfig(),
        embedding=CountingEmbeddingTool(config=EmbeddingConfig(dimension=8)),
    )
    vector_store.set_hits_by_query([_channel_hits(core_hits=hits)])
    sparql_tool = StubSPARQLTool(triple_store_manager=None)
    retriever = OntologyPatchRetriever(
        vector_store=vector_store,
        sparql_tool=sparql_tool,
        patch=PatchRetrievalConfig(min_merged_max_score=min_merged_max_score),
    )
    await retriever.aretrieve_ensemble(queries=["q1"], top_k=4, expand_sparql=True)
    return list(sparql_tool.last_entity_uris)


def test_window_char_budget_is_unset_by_default() -> None:
    """The budget is opt-in: unset, windows are bounded by sentence count."""
    from ontocast.config import VectorStoreConfig

    assert VectorStoreConfig().proposition_window_max_chars is None


def test_an_unset_char_budget_reproduces_sentence_windows_exactly() -> None:
    """The budget must not perturb the path it did not replace.

    Every arm measured before this knob existed was windowed by sentence count.
    If adding the budget moved those windows even slightly, each of those arms
    would silently stop being comparable -- so this asserts equality across the
    knob space rather than at the default alone.
    """
    text = (
        "Films were annealed at 100 C for 30 min. The PL peak shifted by 20 meV. "
        "Samples aged for 4-15 days under ambient conditions. "
        "Emission narrowed by 5 nm. J. Phys. Chem. Lett. 2019, 10, 655. "
        "The shift saturated after two weeks."
    )
    for sentences in (1, 2, 3, 5):
        for windows in (1, 2, 16, 10**6):
            for stride in (None, 1, 2):
                assert split_proposition_windows(
                    text,
                    max_sentences=sentences,
                    max_windows=windows,
                    stride=stride,
                ) == split_proposition_windows(
                    text,
                    max_sentences=sentences,
                    max_windows=windows,
                    stride=stride,
                    max_chars=None,
                )


def test_the_char_budget_bounds_every_window_it_can() -> None:
    """Only a single sentence longer than the budget may exceed it."""
    text = " ".join(f"Sentence number {index} of the passage." for index in range(40))
    for budget in (60, 120, 400):
        windows = split_proposition_windows(text, max_windows=10**6, max_chars=budget)
        assert windows
        for window in windows:
            # Over budget is permitted only where one sentence alone exceeds it,
            # because cutting a sentence loses more than the encoder's own
            # truncation would.
            assert len(window) <= budget or " " not in window.rstrip(".")


def test_the_char_budget_coalesces_fragments_a_sentence_count_cannot() -> None:
    """The splitter has no abbreviation handling, so citations shatter.

    A sentence bound takes those shards as whole windows; a character budget
    keeps taking sentences until it has something worth embedding.
    """
    text = "Cite This: J. Phys. Chem. Lett. 2019, 10, 655. See Fig. 3 and Ref. 12."
    by_sentences = split_proposition_windows(text, max_sentences=2, max_windows=10**6)
    by_chars = split_proposition_windows(text, max_windows=10**6, max_chars=200)
    assert min(len(w) for w in by_sentences) < 20
    assert len(by_chars) == 1
    assert min(len(w) for w in by_chars) > 20


def test_the_char_budget_never_splits_a_sentence() -> None:
    """A sentence over the budget is emitted whole, not cut."""
    long_sentence = "The " + "very " * 60 + "long sentence."
    windows = split_proposition_windows(long_sentence, max_windows=10**6, max_chars=50)
    assert windows == [long_sentence]


def test_the_char_budget_rejects_a_nonsense_value() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        split_proposition_windows("A sentence.", max_chars=0)
