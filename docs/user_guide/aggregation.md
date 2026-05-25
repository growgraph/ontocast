# Entity Disambiguation and Aggregation

After per-unit facts extraction, OntoCast **merges** chunk-level graphs into a document-level facts graph with cross-chunk entity disambiguation.

## Overview

The merge stage (`tool/agg/aggregate.py`):

1. Collects facts graphs from all processed content units
2. Clusters entity mentions using embeddings and symbolic compatibility
3. Rewrites URIs to canonical identities
4. Annotates merged triples with provenance where applicable

Ontology aggregation uses a similar embedding-based pipeline for anchor selection and URI rewriting during document-level ontology reduce.

## Configuration

```bash
AGG_EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
AGG_SIMILARITY_THRESHOLD=0.80
```

| Variable | Description | Default |
|----------|-------------|---------|
| `AGG_EMBEDDING_MODEL` | Sentence-transformers model for entity embeddings | `paraphrase-multilingual-MiniLM-L12-v2` |
| `AGG_SIMILARITY_THRESHOLD` | Cosine similarity threshold for DBSCAN clustering | `0.80` |

Lower thresholds merge more aggressively (fewer duplicate entities, higher false-merge risk). Raise the threshold when precision matters more than recall.

## How Disambiguation Works

1. **Candidate extraction** — entities from each unit's facts graph
2. **Embedding** — dense vectors from `AGG_EMBEDDING_MODEL`
3. **Symbolic checks** — labels, `skos:altName`, IRI compatibility
4. **Clustering** — connected components over similarity + compatibility edges
5. **URI rewrite** — merge graphs under canonical entity URIs
6. **Provenance** — track which unit contributed each merged triple

The standalone **EntityAligner** (`tool/agg/entity_aligner.py`) powers global alignment for the `/match/entities` API (benchmark use), using the same embedding and symbolic regime concepts.

## Graph Matching API

For evaluation against ground truth, use the match endpoints (see [API Endpoints](api.md#graph-matching)):

- Align entities across multiple graphs globally
- Derive pairwise predicted↔GT mappings
- Compute triple and entity precision/recall/F1

The `match-dirs` CLI automates this for directory pairs of TTL files.

## Tuning Tips

1. **Inspect merge output** before lowering `AGG_SIMILARITY_THRESHOLD`.
2. **Domain-specific embeddings** — if you change `EMBEDDING_MODEL_NAME` for Qdrant, consider aligning `AGG_EMBEDDING_MODEL` for consistent geometry.
3. **Large documents** — more units increase merge complexity; use `--head-chunks` while tuning.

## Related

- [Workflow](workflow.md) — where merge fits in the pipeline
- [Core Concepts](concepts.md) — disambiguation overview
- [Configuration](configuration.md) — aggregation env vars
