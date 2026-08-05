"""Backend-agnostic helpers for ontology vector storage."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from ontocast.config import EmbeddingConfig, VectorStoreConfig, VectorStoreDedupMode
from ontocast.tool.vector_store.atomizer import GraphAtomizer
from ontocast.tool.vector_store.core import (
    GraphAtom,
    OntologySearchHit,
    canonicalize_entity_role,
)
from ontocast.util.hash import render_text_hash

META_EMBEDDING_DIMENSION = "embedding_dimension"
META_EMBEDDING_MODEL = "embedding_model"

# Mirrors VectorStoreConfig.minimal_label_limit; only a divergence is fingerprinted.
_DEFAULT_MINIMAL_LABEL_LIMIT = 5

# Bumped whenever the atomizer changes *which* literals become surface forms, or which
# entities become atoms at all. Unlike the cap, such a change alters the stored index at
# the default setting, so it is fingerprinted unconditionally. ``sf2``: added
# qudt:symbol / qudt:ucumCode, and ranked untagged and English literals ahead of other
# languages. ``sf3``: added lexical_triggers for the exact-match retrieval lane.
# ``sf4``: ontology sources atomize only IRIs they describe, so object-position
# references (dimension vectors, systems of units, version IRIs) no longer become atoms.
# A collection built under sf3 still holds those atoms; serving it would silently mix
# pruned and unpruned sources, so the fingerprint forces a reindex rather than allowing
# a half-pruned index. ``sf5``: entity_role is derived from property *declaration* as
# well as predicate-position usage, so a TBox-only module's properties become
# predicate-role atoms and gain a non-empty neighborhood representation. Both the role
# payload and the embedded neighborhood text change, so the old vectors cannot be served
# alongside the new ones. ``sf6``: dcterms:alternative joins the label predicates (its
# synonyms previously vanished from both retrieval lanes and the prompt), and atoms
# carry case-preserved symbol_surfaces so merge-time can demote counterfeit
# case-folded symbol matches (prose "meV" retrieving symbol "MeV"); both change the
# stored surface forms and the payload schema.
_SURFACE_FORM_CONTRACT = "sf6"


class EmbeddingContractMismatchError(ValueError):
    """Embedding vectors or store metadata disagree with the active embedding config."""


def embedding_contract_help(*, backend: str = "vector store") -> str:
    return (
        f"Align EmbeddingConfig (EMBEDDING_*) with the {backend}: use the same model "
        "and dimension as when the store was created, or drop the ontology table/"
        "collection and let initialize() recreate it."
    )


def atom_scope_fingerprint(store_config: VectorStoreConfig) -> str | None:
    """Fingerprint fragment for settings that change what gets stored per atom.

    Covers both *which entities* become atoms and *which literals* become their
    surface forms and lexical triggers. All of these change the stored payload,
    so serving an index built under different values silently degrades
    retrieval instead of raising.

    Returns ``None`` at the defaults, so collections built under them keep the
    fingerprint they already have and need no reindex on upgrade.

    Args:
        store_config: Active vector-store settings.

    Returns:
        str | None: Compact divergence marker, or ``None`` when nothing diverges.
    """
    defaults = VectorStoreConfig.model_fields
    parts: list[str] = []
    if store_config.index_undescribed_iris:
        parts.append("undescribed")
    if store_config.embed_standard_vocab_iris:
        parts.append("stdvocab")
    for prefix in sorted(store_config.extra_excluded_namespace_prefixes):
        parts.append(f"x:{prefix}")

    def _diverges(name: str) -> bool:
        factory = defaults[name].default_factory
        default = factory() if factory is not None else defaults[name].default
        return getattr(store_config, name) != default

    # The surface-form and trigger settings are pushed into the atomizer and
    # decide what lands in the stored payload, so they belong in the identity of
    # the vectors. They were previously omitted, which meant changing one served
    # a stale index rather than raising EmbeddingContractMismatchError.
    for name in ("label_predicates", "symbol_predicates", "lexical_trigger_predicates"):
        if _diverges(name):
            joined = ",".join(sorted(getattr(store_config, name)))
            parts.append(f"{name}={render_text_hash(joined)[:12]}")
    for name in (
        "lexical_trigger_enabled",
        "lexical_trigger_heuristic_enabled",
        "lexical_trigger_min_len",
        "lexical_trigger_max_len",
        "lexical_trigger_heuristic_max_per_entity",
    ):
        if _diverges(name):
            parts.append(f"{name}={getattr(store_config, name)}")
    return ",".join(parts) if parts else None


def embedding_model_fingerprint(
    embedding_config: EmbeddingConfig,
    *,
    minimal_label_limit: int | None = None,
    atom_scope: str | None = None,
) -> str:
    """Identity of the vectors a config produces, stored alongside the collection.

    Query/document prefixes belong here: they change the embedded text, so an index
    built without them is not comparable to queries issued with them, and the mismatch
    would otherwise show up only as quietly degraded retrieval. The sparse surface-form
    cap is included for the same reason -- it decides how many of a term's aliases enter
    the BM25 text. It contributes only when set to a non-default value, so collections
    built under the default keep their existing fingerprint. ``atom_scope`` follows the
    same rule for settings that decide which entities are atomized at all.

    The surface-form contract (``sf=``) is separate and always contributes: it records
    *which* literals become surface forms and which entities become atoms, both of which
    change the stored index even at default settings.

    Args:
        embedding_config: Dense/sparse model configuration.
        minimal_label_limit: Sparse surface-form cap, when it differs from the default.
        atom_scope: Atom-scope divergence from :func:`atom_scope_fingerprint`, if any.

    Returns:
        str: Stable fingerprint stored alongside the collection.
    """
    ec = embedding_config
    dense_part = f"dense:{ec.provider.value}:{ec.model_name}"
    affixes = f"|q={ec.query_prefix}|d={ec.document_prefix}"
    fingerprint = (
        f"{dense_part}|bm25={ec.bm25_model_name}{affixes}|sf={_SURFACE_FORM_CONTRACT}"
    )
    if (
        minimal_label_limit is not None
        and minimal_label_limit != _DEFAULT_MINIMAL_LABEL_LIMIT
    ):
        fingerprint += f"|minlabels={minimal_label_limit}"
    if atom_scope:
        fingerprint += f"|atoms={atom_scope}"
    return fingerprint


def embedding_fingerprint_matches(
    stored: str,
    embedding_config: EmbeddingConfig,
    *,
    minimal_label_limit: int | None = None,
    atom_scope: str | None = None,
) -> bool:
    """Whether ``stored`` is the fingerprint the given config would produce.

    Takes the same optional components as :func:`embedding_model_fingerprint`.
    Omitting them previously made this disagree with
    ``validate_embedding_contract_metadata`` for any non-default collection --
    it would report a match the validator rejects.
    """
    return stored == embedding_model_fingerprint(
        embedding_config,
        minimal_label_limit=minimal_label_limit,
        atom_scope=atom_scope,
    )


def collection_embedding_metadata(
    embedding_config: EmbeddingConfig,
    *,
    metadata_dim: int,
    minimal_label_limit: int | None = None,
    atom_scope: str | None = None,
) -> dict[str, Any]:
    return {
        META_EMBEDDING_DIMENSION: metadata_dim,
        META_EMBEDDING_MODEL: embedding_model_fingerprint(
            embedding_config,
            minimal_label_limit=minimal_label_limit,
            atom_scope=atom_scope,
        ),
    }


def coerce_metadata_int(value: Any, *, field: str, collection: str) -> int:
    if type(value) is bool:
        raise ValueError(
            f"Vector store '{collection}' metadata {field!r} has invalid type"
        )
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip(), 10)
        except ValueError as exc:
            raise ValueError(
                f"Vector store '{collection}' metadata {field!r} is not an integer"
            ) from exc
    raise ValueError(f"Vector store '{collection}' metadata {field!r} has invalid type")


def validate_embedding_contract_metadata(
    collection: str,
    raw_metadata: Mapping[str, Any] | None,
    *,
    embedding_config: EmbeddingConfig,
    expected_meta_dim: int,
    minimal_label_limit: int | None = None,
    atom_scope: str | None = None,
) -> None:
    if raw_metadata is None:
        meta: dict[str, Any] = {}
    else:
        meta = dict(raw_metadata)
    dim_key = META_EMBEDDING_DIMENSION
    model_key = META_EMBEDDING_MODEL
    if dim_key not in meta or model_key not in meta:
        raise EmbeddingContractMismatchError(
            f"Vector store '{collection}' is missing OntoCast embedding metadata "
            f"({dim_key!r}, {model_key!r}). Drop and recreate the store. "
            + embedding_contract_help()
        )
    stored_dim = coerce_metadata_int(
        meta[dim_key], field=dim_key, collection=collection
    )
    stored_model = meta[model_key]
    if not isinstance(stored_model, str):
        raise ValueError(
            f"Vector store '{collection}' metadata {model_key!r} must be a string"
        )
    expected_model = embedding_model_fingerprint(
        embedding_config,
        minimal_label_limit=minimal_label_limit,
        atom_scope=atom_scope,
    )
    if stored_dim != expected_meta_dim or stored_model != expected_model:
        raise EmbeddingContractMismatchError(
            f"Vector store '{collection}' embedding contract mismatch: "
            f"store has dimension={stored_dim}, model={stored_model!r}; "
            f"current config expects dimension={expected_meta_dim}, "
            f"model={expected_model!r}. " + embedding_contract_help()
        )


def normalized_fusion_weights(
    store_config: VectorStoreConfig,
) -> tuple[float, float, float]:
    cw = store_config.fusion_core_weight
    nw = store_config.fusion_neighborhood_weight
    bw = store_config.fusion_bm25_weight
    total = cw + nw + bw
    if total <= 0.0:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return (cw / total, nw / total, bw / total)


def normalized_core_neighborhood_weights(
    store_config: VectorStoreConfig,
) -> tuple[float, float]:
    cw, nw, _ = normalized_fusion_weights(store_config)
    total = cw + nw
    if total <= 0.0:
        return (0.5, 0.5)
    return (cw / total, nw / total)


def rank_fuse_channel_hits(
    core_hits: list[OntologySearchHit],
    neighborhood_hits: list[OntologySearchHit],
    bm25_hits: list[OntologySearchHit],
    *,
    core_weight: float,
    neighborhood_weight: float,
    bm25_weight: float,
    limit: int,
) -> list[OntologySearchHit]:
    rank_scores: dict[str, float] = {}
    best_hit_by_id: dict[str, OntologySearchHit] = {}

    for rank, hit in enumerate(core_hits, start=1):
        atom_id = hit.atom.atom_id
        rank_scores[atom_id] = rank_scores.get(atom_id, 0.0) + (core_weight / rank)
        prev = best_hit_by_id.get(atom_id)
        if prev is None or hit.score > prev.score:
            best_hit_by_id[atom_id] = hit
    for rank, hit in enumerate(neighborhood_hits, start=1):
        atom_id = hit.atom.atom_id
        rank_scores[atom_id] = rank_scores.get(atom_id, 0.0) + (
            neighborhood_weight / rank
        )
        prev = best_hit_by_id.get(atom_id)
        if prev is None or hit.score > prev.score:
            best_hit_by_id[atom_id] = hit
    for rank, hit in enumerate(bm25_hits, start=1):
        atom_id = hit.atom.atom_id
        rank_scores[atom_id] = rank_scores.get(atom_id, 0.0) + (bm25_weight / rank)
        prev = best_hit_by_id.get(atom_id)
        if prev is None or hit.score > prev.score:
            best_hit_by_id[atom_id] = hit

    ranked_atom_ids = sorted(
        rank_scores.keys(),
        key=lambda atom_id: (
            rank_scores[atom_id],
            float(best_hit_by_id[atom_id].score),
            atom_id,
        ),
        reverse=True,
    )[:limit]
    out: list[OntologySearchHit] = []
    for atom_id in ranked_atom_ids:
        source_hit = best_hit_by_id[atom_id]
        atom = source_hit.atom.model_copy(update={"score": rank_scores[atom_id]})
        out.append(OntologySearchHit(atom=atom, score=rank_scores[atom_id]))
    return out


def parse_created_at(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def atom_payload(atom: GraphAtom) -> dict[str, Any]:
    return {
        "atom_id": atom.atom_id,
        "ontology_iri": atom.ontology_iri,
        "ontology_id": atom.ontology_id,
        "ontology_hash": atom.ontology_hash,
        "ontology_version": atom.ontology_version,
        "iri": atom.iri,
        "entity_role": canonicalize_entity_role(atom.entity_role),
        "core_representation": atom.core_representation,
        "minimal_representation": atom.minimal_representation,
        "neighborhood_representation": atom.neighborhood_representation,
        "lexical_triggers": list(atom.lexical_triggers),
        "symbol_surfaces": list(atom.symbol_surfaces),
        "created_at": atom.created_at.isoformat(),
    }


def atom_from_payload(
    payload: Mapping[str, Any],
    *,
    score: float | None = None,
    default_id: str = "",
) -> GraphAtom:
    created_at_raw = payload.get("created_at")
    return GraphAtom(
        atom_id=str(payload.get("atom_id", default_id)),
        ontology_iri=str(payload.get("ontology_iri", "")),
        ontology_id=payload.get("ontology_id"),
        ontology_hash=payload.get("ontology_hash"),
        ontology_version=payload.get("ontology_version"),
        iri=str(payload.get("iri", "")),
        entity_role=canonicalize_entity_role(payload.get("entity_role")),
        core_representation=str(payload.get("core_representation", "")),
        minimal_representation=str(payload.get("minimal_representation", "")),
        neighborhood_representation=str(payload.get("neighborhood_representation", "")),
        lexical_triggers=_payload_str_list(payload, "lexical_triggers"),
        symbol_surfaces=_payload_str_list(payload, "symbol_surfaces"),
        created_at=parse_created_at(created_at_raw),
        score=score,
    )


def _payload_str_list(payload: Mapping[str, Any], key: str) -> list[str]:
    raw = payload.get(key)
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item).strip()]
    return []


def identity_key_for_atom(
    atom: GraphAtom,
    *,
    store_config: VectorStoreConfig,
) -> str:
    if store_config.dedup_mode == VectorStoreDedupMode.ATOM_ID:
        return atom.atom_id
    parts: list[str] = [
        atom.ontology_iri or "",
        atom.iri or "",
    ]
    if store_config.dedup_include_version:
        parts.append(atom.ontology_version or "")
    if store_config.dedup_include_hash:
        parts.append(atom.ontology_hash or "")
    return "|".join(parts)


def point_id_for_atom(
    atom: GraphAtom,
    *,
    store_config: VectorStoreConfig,
) -> str:
    if store_config.dedup_mode == VectorStoreDedupMode.ATOM_ID:
        return point_id(atom.atom_id)
    return point_id(identity_key_for_atom(atom, store_config=store_config))


def point_id(atom_id: str) -> str:
    try:
        return str(uuid.UUID(atom_id))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, atom_id))


def dedupe_hits_by_identity(
    hits: list[OntologySearchHit],
    *,
    store_config: VectorStoreConfig,
) -> list[OntologySearchHit]:
    if not hits:
        return []
    best_by_key: dict[str, OntologySearchHit] = {}
    order_index: dict[str, int] = {}
    for index, hit in enumerate(hits):
        key = identity_key_for_atom(hit.atom, store_config=store_config)
        previous = best_by_key.get(key)
        if previous is None:
            best_by_key[key] = hit
            order_index[key] = index
            continue
        if float(hit.score) > float(previous.score):
            best_by_key[key] = hit
    deduped = list(best_by_key.values())
    deduped.sort(
        key=lambda h: (
            -float(h.score),
            order_index[identity_key_for_atom(h.atom, store_config=store_config)],
        )
    )
    return deduped


def sync_atomizer_from_store_config(
    atomizer: GraphAtomizer, store_config: VectorStoreConfig
) -> None:
    """Mirror vector-store representation settings onto the atomizer."""
    atomizer.minimal_representation_label_limit = store_config.minimal_label_limit
    atomizer.label_predicates = list(store_config.label_predicates)
    atomizer.symbol_predicates = list(store_config.symbol_predicates)
    atomizer.index_undescribed_iris = store_config.index_undescribed_iris
    atomizer.embed_standard_vocab_iris = store_config.embed_standard_vocab_iris
    atomizer.extra_excluded_namespace_prefixes = list(
        store_config.extra_excluded_namespace_prefixes
    )
    atomizer.lexical_trigger_enabled = store_config.lexical_trigger_enabled
    atomizer.lexical_trigger_predicates = list(store_config.lexical_trigger_predicates)
    atomizer.lexical_trigger_heuristic_enabled = (
        store_config.lexical_trigger_heuristic_enabled
    )
    atomizer.lexical_trigger_min_len = store_config.lexical_trigger_min_len
    atomizer.lexical_trigger_max_len = store_config.lexical_trigger_max_len
    atomizer.lexical_trigger_heuristic_max_per_entity = (
        store_config.lexical_trigger_heuristic_max_per_entity
    )


def effective_top_k(store_config: VectorStoreConfig, top_k: int | None) -> int:
    if top_k is not None:
        return top_k
    return store_config.top_k


def iter_batches(items: list[Any], batch_size: int) -> list[list[Any]]:
    batches: list[list[Any]] = []
    for index in range(0, len(items), batch_size):
        batches.append(items[index : index + batch_size])
    return batches


def require_embedding_vector_length(
    vector: list[float],
    *,
    role: str,
    expected: int,
) -> None:
    if len(vector) != expected:
        raise EmbeddingContractMismatchError(
            f"{role} vector length {len(vector)} does not match the configured "
            f"embedding dimension {expected}. " + embedding_contract_help()
        )
