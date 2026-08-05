"""Regression tests for _point_to_atom across Qdrant's Record vs ScoredPoint shapes.

``client.search``/``query_points`` return ``ScoredPoint`` (has ``score``); ``client.scroll``/
``retrieve`` return ``Record`` (no ``score`` field at all, not even ``None``). Any code path
that feeds a bare ``Record`` into a helper expecting ``point.score`` raises ``AttributeError``
(pydantic models don't return ``None`` for genuinely absent fields). These tests pin the fix:
``_point_to_atom`` must take an explicit ``score`` parameter instead of probing ``point``.
"""

from __future__ import annotations

from qdrant_client.http import models as qdrant_models

from ontocast.config import EmbeddingConfig, QdrantConfig
from ontocast.tool.vector_store.qdrant import QdrantVectorStoreManager
from test.qdrant_util import DeterministicEmbeddingTool


class _FakeScrollClient:
    """Stand-in for QdrantClient exposing only what fetch_atoms_by_ids touches."""

    def __init__(self, records: list[qdrant_models.Record]) -> None:
        self._records = records

    def collection_exists(self, *, collection_name: str) -> bool:
        return True

    def scroll(self, **kwargs: object) -> tuple[list[qdrant_models.Record], None]:
        return self._records, None


def _build_store() -> QdrantVectorStoreManager:
    embedding = DeterministicEmbeddingTool(
        config=EmbeddingConfig(dimension=8, model_name="pytest-point-to-atom")
    )
    return QdrantVectorStoreManager(qdrant_config=QdrantConfig(), embedding=embedding)


def _payload(atom_id: str) -> dict[str, object]:
    return {
        "atom_id": atom_id,
        "ontology_iri": "https://example.org/o",
        "iri": f"https://example.org/o#{atom_id}",
        "core_representation": "core text",
        "neighborhood_representation": "",
    }


def test_point_to_atom_accepts_record_without_score_attribute() -> None:
    store = _build_store()
    record = qdrant_models.Record(id="p1", payload=_payload("a1"))
    assert not hasattr(record, "score")

    atom = store._point_to_atom(record)

    assert atom.atom_id == "a1"
    assert atom.score is None


def test_point_to_atom_uses_explicit_score_for_scored_point() -> None:
    store = _build_store()
    scored_point = qdrant_models.ScoredPoint(
        id="p1", version=0, score=0.42, payload=_payload("a1")
    )

    atom = store._point_to_atom(scored_point, score=float(scored_point.score))

    assert atom.score == 0.42


def test_fetch_atoms_by_ids_survives_scroll_records_without_score() -> None:
    store = _build_store()
    record = qdrant_models.Record(id="p1", payload=_payload("a1"))
    store._client = _FakeScrollClient([record])  # ty: ignore[invalid-assignment]

    atoms = store.fetch_atoms_by_ids(["a1"])

    assert [atom.atom_id for atom in atoms] == ["a1"]
    assert atoms[0].score is None


def test_atom_payload_round_trips_symbol_surfaces() -> None:
    """symbol_surfaces (sf6) must survive the payload round trip case-intact."""
    from ontocast.tool.vector_store.core import GraphAtom
    from ontocast.tool.vector_store.util import atom_from_payload, atom_payload

    atom = GraphAtom(
        atom_id="a1",
        ontology_iri="https://example.org/units",
        iri="http://qudt.org/vocab/unit/MegaEV",
        core_representation="megaelectronvolt",
        neighborhood_representation="",
        lexical_triggers=["MeV"],
        symbol_surfaces=["MeV"],
    )
    restored = atom_from_payload(atom_payload(atom))
    assert restored.symbol_surfaces == ["MeV"]
    assert restored.lexical_triggers == ["MeV"]

    # Payloads written before sf6 have no symbol_surfaces key: default empty.
    legacy = atom_payload(atom)
    legacy.pop("symbol_surfaces")
    assert atom_from_payload(legacy).symbol_surfaces == []
