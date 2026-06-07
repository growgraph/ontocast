"""Embedded LanceDB vector store for ontology atoms."""

from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pydantic import Field

from ontocast.config import EmbeddingConfig, LanceDBConfig, VectorStoreConfig
from ontocast.onto.ontology import Ontology
from ontocast.onto.tenancy import (
    TENANCY_SEP,
    tenant_project_facts_name,
    tenant_project_ontologies_name,
)
from ontocast.tool.vector_store.atomizer import GraphAtomizer
from ontocast.tool.vector_store.core import (
    GraphAtom,
    OntologySearchHit,
    OntologySearchHitsByChannel,
    VectorStoreManager,
)
from ontocast.tool.vector_store.embedding import (
    EmbeddingTool,
    FastembedBm25SparseTool,
)
from ontocast.tool.vector_store.util import (
    atom_from_payload,
    atom_payload,
    collection_embedding_metadata,
    dedupe_hits_by_identity,
    effective_top_k,
    iter_batches,
    normalized_fusion_weights,
    point_id_for_atom,
    rank_fuse_channel_hits,
    require_embedding_vector_length,
    validate_embedding_contract_metadata,
)

logger = logging.getLogger(__name__)

_META_DIRNAME = ".ontocast_embedding_meta"


def _require_lancedb():
    try:
        import lancedb  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "LanceDB vector store requires the 'lancedb' extra. "
            "Install with: uv sync --extra lancedb"
        ) from exc


class LanceDBVectorStoreManager(VectorStoreManager):
    """Stores ontology atoms in a single embedded LanceDB database directory."""

    store_config: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    lancedb_config: LanceDBConfig = Field(default_factory=LanceDBConfig)
    embedding: EmbeddingTool = Field(..., exclude=True)
    sparse_embedding: FastembedBm25SparseTool | None = Field(default=None, exclude=True)
    atomizer: GraphAtomizer = Field(default_factory=GraphAtomizer, exclude=True)

    @property
    def embedding_config(self) -> EmbeddingConfig:
        return self.embedding.config

    def _require_sparse_embedding_tool(self) -> FastembedBm25SparseTool:
        if self.sparse_embedding is None:
            raise ValueError(
                "BM25 sparse embedding is required for vector search but "
                "sparse_embedding was not wired"
            )
        return self.sparse_embedding

    def _data_dir(self) -> Path:
        return Path(self.lancedb_config.data_dir).expanduser().resolve()

    def _connect(self):
        _require_lancedb()
        import lancedb

        data_dir = self._data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)
        return lancedb.connect(str(data_dir))

    def _ontology_table_name(self) -> str:
        name = self.lancedb_config.ontology_table
        if name is None:
            raise ValueError(
                "LanceDB ontology_table is unset; ensure LanceDBConfig validation"
                " ran or call apply_tenancy before vector operations"
            )
        return name

    def _facts_table_name(self) -> str:
        name = self.lancedb_config.facts_table
        if name is None:
            raise ValueError(
                "LanceDB facts_table is unset; ensure LanceDBConfig validation ran"
            )
        return name

    def _meta_path(self, table_name: str | None = None) -> Path:
        table = table_name or self._ontology_table_name()
        meta_dir = self._data_dir() / _META_DIRNAME
        meta_dir.mkdir(parents=True, exist_ok=True)
        return meta_dir / f"{table}.json"

    def supports_tenancy_partition(self) -> bool:
        return True

    def apply_tenancy(
        self,
        tenant: str,
        project: str,
        *,
        sep: str = TENANCY_SEP,
    ) -> None:
        """Switch active Lance tables for ``tenant`` / ``project``.

        Same naming as Qdrant collections; all tables live in one embedded DB.
        Call :meth:`initialize` after.
        """
        t, p = tenant.strip(), project.strip()
        ontology_name = tenant_project_ontologies_name(t, p, sep=sep)
        facts_name = tenant_project_facts_name(t, p, sep=sep)
        self.lancedb_config.ontology_table = ontology_name
        self.lancedb_config.facts_table = facts_name
        self.store_config.ontology_table = ontology_name
        self.store_config.facts_table = facts_name

    async def clean_tenancy(
        self,
        tenant: str,
        project: str,
        *,
        sep: str = TENANCY_SEP,
    ) -> None:
        """Drop Lance tables (and embedding metadata) for ``tenant`` / ``project``."""
        t, p = tenant.strip(), project.strip()
        table_names = (
            tenant_project_ontologies_name(t, p, sep=sep),
            tenant_project_facts_name(t, p, sep=sep),
        )
        db = self._connect()
        existing = self._list_tables(db)
        for name in table_names:
            if name in existing:
                db.drop_table(name)
                logger.info("Dropped LanceDB table %s", name)
            meta = self._meta_path(name)
            if meta.exists():
                meta.unlink()

    def _dense_dimension(self) -> int:
        return self.embedding_config.dimension

    def _write_embedding_meta(self) -> None:
        meta = collection_embedding_metadata(
            self.embedding_config,
            metadata_dim=self._dense_dimension(),
        )
        self._meta_path().write_text(json.dumps(meta), encoding="utf-8")

    def _read_embedding_meta(self) -> dict[str, Any]:
        path = self._meta_path()
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _validate_embedding_meta(self) -> None:
        table = self._ontology_table_name()
        validate_embedding_contract_metadata(
            table,
            self._read_embedding_meta() or None,
            embedding_config=self.embedding_config,
            expected_meta_dim=self._dense_dimension(),
        )

    async def initialize(self) -> None:
        """Create ontology/facts tables and indexes if missing."""
        await asyncio.to_thread(self._initialize_sync)

    def _list_tables(self, db: Any) -> set[str]:
        list_tables = getattr(db, "list_tables", None)
        if callable(list_tables):
            response = list_tables()
            if hasattr(response, "tables"):
                raw = response.tables
                if raw and isinstance(raw[0], str):
                    return set(raw)
                return {str(item) for item in raw}
            if isinstance(response, list):
                return {str(item) for item in response}
            return set()
        return set(db.table_names())

    def _initialize_sync(self) -> None:
        db = self._connect()
        ontology_name = self._ontology_table_name()
        tables = self._list_tables(db)

        if ontology_name in tables:
            self._validate_embedding_meta()
            table = db.open_table(ontology_name)
            self._ensure_indexes(table)
        elif not self._meta_path().exists():
            self._write_embedding_meta()

    def _ensure_indexes(self, table: Any) -> None:
        try:
            table.create_index(metric="cosine", vector_column_name="core_vector")
        except Exception:
            logger.debug("LanceDB core_vector index already exists or skipped")
        try:
            table.create_index(
                metric="cosine", vector_column_name="neighborhood_vector"
            )
        except Exception:
            logger.debug("LanceDB neighborhood_vector index already exists or skipped")
        try:
            table.create_fts_index("minimal_representation")
        except Exception:
            logger.debug("LanceDB FTS index already exists or skipped")

    def _record_from_atom(
        self,
        atom: GraphAtom,
        core_vector: list[float],
        neighborhood_vector: list[float],
    ) -> dict[str, Any]:
        payload = atom_payload(atom)
        payload["point_id"] = point_id_for_atom(atom, store_config=self.store_config)
        payload["core_vector"] = core_vector
        payload["neighborhood_vector"] = neighborhood_vector
        return payload

    def index_ontology(self, ontology: Ontology) -> int:
        atoms = self.atomizer.atomize(source=ontology, depth=1)
        if not atoms:
            return 0

        core_vectors = self._embed_texts_batched(
            [atom.core_representation for atom in atoms]
        )
        neighborhood_vectors = self._embed_texts_batched(
            [atom.neighborhood_representation for atom in atoms]
        )

        records = [
            self._record_from_atom(atom, core_vectors[i], neighborhood_vectors[i])
            for i, atom in enumerate(atoms)
        ]

        db = self._connect()
        table_name = self._ontology_table_name()
        tables = self._list_tables(db)
        if table_name not in tables:
            db.create_table(table_name, data=records)
            self._write_embedding_meta()
            table = db.open_table(table_name)
            self._ensure_indexes(table)
            return len(records)

        table = db.open_table(table_name)
        table.merge_insert(
            "point_id"
        ).when_matched_update_all().when_not_matched_insert_all().execute(  # type: ignore[attr-defined]
            records
        )
        return len(records)

    def _encode_single_query_vectors(
        self, query: str
    ) -> tuple[list[float], list[float], str]:
        triples = self._encode_query_vectors_batch([query])
        return triples[0]

    def _encode_query_vectors_batch(
        self, queries: list[str]
    ) -> list[tuple[list[float], list[float], str]]:
        n = len(queries)
        if n == 0:
            return []
        dense_vecs = self.embedding.embed(queries)
        if len(dense_vecs) != n:
            raise ValueError(
                "Embedding provider returned mismatched vectors for queries"
            )
        for i, vec in enumerate(dense_vecs):
            require_embedding_vector_length(
                vec,
                role=f"Query embedding[{i}]",
                expected=self._dense_dimension(),
            )
        return [(dense_vecs[i], dense_vecs[i], queries[i]) for i in range(n)]

    def _filter_clause(
        self,
        *,
        filter_iri: str | None,
        filter_version: str | None,
        filter_hash: str | None,
    ) -> str | None:
        parts: list[str] = []
        if filter_iri is not None:
            parts.append(
                f"ontology_iri = '{filter_iri.replace(chr(39), chr(39) + chr(39))}'"
            )
        if filter_version is not None:
            parts.append(
                f"ontology_version = '{filter_version.replace(chr(39), chr(39) + chr(39))}'"
            )
        if filter_hash is not None:
            parts.append(
                f"ontology_hash = '{filter_hash.replace(chr(39), chr(39) + chr(39))}'"
            )
        if not parts:
            return None
        return " AND ".join(parts)

    def _row_to_hit(
        self,
        row: dict[str, Any],
        *,
        score: float,
        apply_neighborhood_empty_penalty: bool = False,
    ) -> OntologySearchHit:
        if apply_neighborhood_empty_penalty:
            neighborhood_text = str(row.get("neighborhood_representation", ""))
            if neighborhood_text.strip().lower() == "no neighborhood facts available":
                score = 0.0
        atom = atom_from_payload(row, score=score)
        return OntologySearchHit(atom=atom, score=score)

    def _search_dense_channel(
        self,
        vector: list[float],
        *,
        vector_column: str,
        limit: int,
        where: str | None,
    ) -> list[OntologySearchHit]:
        db = self._connect()
        table_name = self._ontology_table_name()
        tables = self._list_tables(db)
        if table_name not in tables:
            return []
        table = db.open_table(table_name)
        builder = table.search(vector, vector_column_name=vector_column).limit(limit)
        if where:
            builder = builder.where(where)
        rows = builder.to_list()
        hits: list[OntologySearchHit] = []
        for row in rows:
            distance = float(row.get("_distance", 0.0))
            score = 1.0 - distance if distance <= 1.0 else 1.0 / (1.0 + distance)
            hits.append(
                self._row_to_hit(
                    row,
                    score=score,
                    apply_neighborhood_empty_penalty=(
                        vector_column == "neighborhood_vector"
                    ),
                )
            )
        return hits

    def _search_bm25_channel(
        self,
        query: str,
        *,
        limit: int,
        where: str | None,
    ) -> list[OntologySearchHit]:
        db = self._connect()
        table_name = self._ontology_table_name()
        tables = self._list_tables(db)
        if table_name not in tables:
            return []
        table = db.open_table(table_name)
        builder = table.search(query, query_type="fts").limit(limit)
        if where:
            builder = builder.where(where)
        rows = builder.to_list()
        hits: list[OntologySearchHit] = []
        for row in rows:
            score = float(row.get("_score", row.get("score", 0.0)))
            hits.append(self._row_to_hit(row, score=score))
        return hits

    def search_hits_by_vector(
        self,
        core_vector: list[float],
        neighborhood_vector: list[float],
        bm25_query: str | None = None,
        top_k: int | None = None,
        filter_iri: str | None = None,
        filter_version: str | None = None,
        filter_hash: str | None = None,
    ) -> OntologySearchHitsByChannel:
        eff_top_k = effective_top_k(self.store_config, top_k)
        where = self._filter_clause(
            filter_iri=filter_iri,
            filter_version=filter_version,
            filter_hash=filter_hash,
        )
        core_hits = self._search_dense_channel(
            core_vector,
            vector_column="core_vector",
            limit=eff_top_k,
            where=where,
        )
        neighborhood_hits = self._search_dense_channel(
            neighborhood_vector,
            vector_column="neighborhood_vector",
            limit=eff_top_k,
            where=where,
        )
        bm25_hits: list[OntologySearchHit] = []
        if bm25_query is not None:
            bm25_hits = self._search_bm25_channel(
                bm25_query, limit=eff_top_k, where=where
            )
        if self.store_config.dedup_query_hits_by_iri:
            core_hits = dedupe_hits_by_identity(
                core_hits, store_config=self.store_config
            )
            neighborhood_hits = dedupe_hits_by_identity(
                neighborhood_hits, store_config=self.store_config
            )
            bm25_hits = dedupe_hits_by_identity(
                bm25_hits, store_config=self.store_config
            )
        return OntologySearchHitsByChannel(
            core_hits=core_hits,
            neighborhood_hits=neighborhood_hits,
            bm25_hits=bm25_hits,
        )

    def search_patches(
        self,
        query: str,
        top_k: int | None = None,
        filter_iri: str | None = None,
        filter_version: str | None = None,
        filter_hash: str | None = None,
    ) -> list[GraphAtom]:
        hits = self.search_patch_hits(
            query=query,
            top_k=top_k,
            filter_iri=filter_iri,
            filter_version=filter_version,
            filter_hash=filter_hash,
        )
        return [hit.atom for hit in hits]

    def search_patch_hits(
        self,
        query: str,
        top_k: int | None = None,
        filter_iri: str | None = None,
        filter_version: str | None = None,
        filter_hash: str | None = None,
    ) -> list[OntologySearchHit]:
        core_q, neigh_q, bm25_q = self._encode_single_query_vectors(query)
        channel_hits = self.search_hits_by_vector(
            core_vector=core_q,
            neighborhood_vector=neigh_q,
            bm25_query=bm25_q,
            top_k=top_k,
            filter_iri=filter_iri,
            filter_version=filter_version,
            filter_hash=filter_hash,
        )
        eff_top_k = effective_top_k(self.store_config, top_k)
        cw, nw, bw = normalized_fusion_weights(self.store_config)
        return rank_fuse_channel_hits(
            channel_hits.core_hits,
            channel_hits.neighborhood_hits,
            channel_hits.bm25_hits,
            core_weight=cw,
            neighborhood_weight=nw,
            bm25_weight=bw,
            limit=eff_top_k,
        )

    def _search_patch_hits_for_query_triples(
        self,
        triples: list[tuple[list[float], list[float], str]],
        top_k: int,
        filter_iri: str | None,
        filter_version: str | None,
        filter_hash: str | None,
    ) -> list[OntologySearchHitsByChannel]:
        if not triples:
            return []

        def search_one(
            t: tuple[list[float], list[float], str],
        ) -> OntologySearchHitsByChannel:
            core_v, neigh_v, bm25_q = t
            return self.search_hits_by_vector(
                core_vector=core_v,
                neighborhood_vector=neigh_v,
                bm25_query=bm25_q,
                top_k=top_k,
                filter_iri=filter_iri,
                filter_version=filter_version,
                filter_hash=filter_hash,
            )

        workers = min(32, len(triples))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(search_one, triples))

    def search_patch_hits_many(
        self,
        queries: list[str],
        top_k: int | None = None,
        filter_iri: str | None = None,
        filter_version: str | None = None,
        filter_hash: str | None = None,
    ) -> list[OntologySearchHitsByChannel]:
        if not queries:
            return []
        eff_top_k = effective_top_k(self.store_config, top_k)
        triples = self._encode_query_vectors_batch(queries)
        return self._search_patch_hits_for_query_triples(
            triples,
            eff_top_k,
            filter_iri,
            filter_version,
            filter_hash,
        )

    async def asearch_patch_hits_many(
        self,
        queries: list[str],
        top_k: int | None = None,
        filter_iri: str | None = None,
        filter_version: str | None = None,
        filter_hash: str | None = None,
    ) -> list[OntologySearchHitsByChannel]:
        if not queries:
            return []
        eff_top_k = effective_top_k(self.store_config, top_k)
        triples = await asyncio.to_thread(self._encode_query_vectors_batch, queries)
        tasks = [
            asyncio.to_thread(
                self.search_hits_by_vector,
                core_v,
                neigh_v,
                bm25_q,
                eff_top_k,
                filter_iri,
                filter_version,
                filter_hash,
            )
            for core_v, neigh_v, bm25_q in triples
        ]
        return await asyncio.gather(*tasks)

    def fetch_vectors(
        self,
        atom_ids: list[str],
    ) -> dict[str, tuple[list[float], list[float]]]:
        if not atom_ids:
            return {}
        db = self._connect()
        table_name = self._ontology_table_name()
        tables = self._list_tables(db)
        if table_name not in tables:
            return {}
        table = db.open_table(table_name)
        out: dict[str, tuple[list[float], list[float]]] = {}
        for atom_id in atom_ids:
            escaped = atom_id.replace("'", "''")
            rows = table.search().where(f"atom_id = '{escaped}'").limit(1).to_list()
            if not rows:
                continue
            row = rows[0]
            core = row.get("core_vector")
            neighborhood = row.get("neighborhood_vector")
            if isinstance(core, list) and isinstance(neighborhood, list):
                out[atom_id] = (
                    [float(v) for v in core],
                    [float(v) for v in neighborhood],
                )
        return out

    def delete_ontology(
        self,
        iri: str,
        version: str | None = None,
        ontology_hash: str | None = None,
    ) -> None:
        where = self._filter_clause(
            filter_iri=iri,
            filter_version=version,
            filter_hash=ontology_hash,
        )
        if where is None:
            return
        db = self._connect()
        table_name = self._ontology_table_name()
        tables = self._list_tables(db)
        if table_name not in tables:
            return
        table = db.open_table(table_name)
        table.delete(where)

    def _embed_texts_batched(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for batch in iter_batches(texts, self.store_config.embedding_batch_size):
            batch_vectors = self.embedding.embed(batch)
            if len(batch_vectors) != len(batch):
                raise ValueError(
                    "Embedding provider returned mismatched vectors for batch"
                )
            for j, vec in enumerate(batch_vectors):
                require_embedding_vector_length(
                    vec,
                    role=f"Index embedding batch offset {len(vectors) + j}",
                    expected=self._dense_dimension(),
                )
            vectors.extend(batch_vectors)
        return vectors
