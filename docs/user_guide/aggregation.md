# Entity Disambiguation and Aggregation

After per-unit facts extraction, OntoCast **merges** chunk-level graphs into a document-level facts graph with cross-chunk entity disambiguation.

## Overview

The merge stage (`tool/agg/aggregate.py`):

1. Collects facts graphs from all processed content units
2. Clusters entity mentions using embeddings and symbolic compatibility
3. Rewrites URIs to canonical identities
4. Annotates merged triples with provenance: an RDF 1.2 reifier per asserted
   triple linked to its source unit, and one `prov:Entity, schema:Text` node per
   unit carrying its index, content hash, timestamp and section label — see
   [Concepts](concepts.md#rdf-12-provenance) for the predicate table

Ontology aggregation uses a similar embedding-based pipeline for anchor selection and URI rewriting during document-level ontology reduce.

## Configuration

```bash
AGG_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
AGG_SIMILARITY_THRESHOLD=0.80
```

| Variable | Description | Default |
|----------|-------------|---------|
| `AGG_EMBEDDING_MODEL` | Sentence-transformers model for entity embeddings. Shares one process-wide model with `EMBEDDING_MODEL_NAME` and `CHUNK_EMBEDDING_MODEL` when the names match — see [Performance](performance.md#local-embedding-models) | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| `AGG_SIMILARITY_THRESHOLD` | Cosine threshold of the cross-graph `EntityAligner` (`/align_entities`, `match-graphs`); the in-pipeline aggregator does **not** read it | `0.80` |
| `AGG_CANDIDATE_SIMILARITY_THRESHOLD` | Cosine threshold of the in-pipeline aggregator (DBSCAN candidate clustering and the pairwise gate); deliberately permissive because candidates are validated symbolically | `0.70` |
| `AGG_LEXICAL_LABEL_JACCARD` | Minimum label token-set Jaccard for the fuzzy lexical-alias tier | `0.5` |
| `AGG_LEXICAL_SEQUENCE_RATIO` | Minimum SequenceMatcher ratio on URI normal forms | `0.90` |
| `AGG_LEXICAL_TOKEN_JACCARD` | Minimum normal-form token Jaccard (both sides ≥ 2 tokens) | `0.75` |
| `AGG_FUNCTIONAL_MIN_EMPIRICAL_SUPPORT` | Min distinct subjects before a predicate counts as empirically single-valued | `2` |
| `AGG_SIBLING_GUARD_SCOPE` | Co-object sibling guard scope: `subject` or `predicate` | `subject` |
| `AGG_LITERAL_CONFLICT_GUARD` | Veto merges between entities asserting disjoint literal values on a shared predicate (ablation flag) | `true` |
| `AGG_INITIALS_DISTINCT_GUARD` | Veto merges between entities whose labels are identical except for conflicting initials ("company S." vs "company T.") | `true` |
| `AGG_NATURAL_KEY_MERGE` | Positive identity evidence: instances sharing an identical short string value on a single-valued identifier-like predicate become merge candidates | `true` |
| `AGG_TYPE_GUARD_UNTYPED` | `permissive` lets a typed entity merge with an untyped one; `strict` fails typed-vs-untyped pairs closed | `permissive` |

Lower thresholds merge more aggressively (fewer duplicate entities, higher false-merge risk). Raise the threshold when precision matters more than recall.

## How Disambiguation Works

1. **Candidate extraction** — entities from each unit's facts graph
2. **Embedding** — dense vectors from `AGG_EMBEDDING_MODEL`
3. **Symbolic checks** — labels, `skos:altName`, IRI compatibility; **identical `URIRef` always compatible** (e.g. the same ontology class appearing in predicted and ground-truth graphs clusters with score 1.0 even when labels are missing or embeddings disagree)
4. **Merge guards** — safety checks that block identity merges between entities that provably describe distinct things: *literal-conflict* (disjoint numeric/temporal values on one predicate, e.g. 30 vs 230 μJ/cm², or string identifier sets with no compatible cross-pair; `AGG_LITERAL_CONFLICT_GUARD`), *functional-object conflict* (disjoint IRI objects on a schema-declared or empirically single-valued predicate, e.g. two different `qudt:unit`s), *sibling* (two objects of one subject never merge — range bounds, sample series, grant lists), *initials-distinct* (labels identical except for conflicting initials mark distinct entities; `AGG_INITIALS_DISTINCT_GUARD`), and a *strict lexical bar* for literal-bearing entities (only exact label/normal-form matches; fuzzy tiers are reserved for entities without data payloads). Guard vetoes hold **cluster-wide**: a vetoed pair cannot be united through a chain of accepted edges either.
5. **Natural-key evidence** (`AGG_NATURAL_KEY_MERGE`) — the one *positive* symbolic signal: two instances asserting the identical short string value on a single-valued identifier-like predicate (schema max-1, or observed single-valued on every subject) become merge candidates even when labels and embeddings disagree — "Application no. 36760/06" is the same case wherever its number appears. All guards still apply.
6. **Clustering** — connected components over similarity + compatibility edges, subject to the cluster-wide vetoes
7. **URI rewrite** — merge graphs under canonical entity URIs; a triple whose subject and object became identical only through the merge (a self-loop no source asserted) is dropped with a warning
8. **Provenance** — track which unit contributed each merged triple
9. **Validation gate** (stategraph only) — after merging, the `VALIDATE_FACTS`
   node checks post-merge invariants and applies LLM-free repairs; see
   [Facts Validation and SHACL](validation.md). *Merge-signature* error
   findings on merged subjects turn the offending cluster into pair vetoes and
   the facts units are re-aggregated (`FACTS_MERGE_REPAIR_PASSES`, default 1);
   because vetoes hold cluster-wide, a veto pass dissolves the flagged cluster
   instead of being re-defeated by transitive closure. SHACL findings are
   deliberately not part of that loop — a constraint violation is not evidence
   of a bad identity merge.

The standalone **EntityAligner** (`tool/agg/entity_aligner.py`) powers global alignment for the `/match/entities` API (benchmark use), using the same embedding and symbolic regime concepts (`ontology_loose` / `ontology_strict`).

## Graph Matching API

For evaluation against ground truth, use the match endpoints (see [API Endpoints](api.md#graph-matching)):

- Align entities across multiple graphs globally
- Derive pairwise predicted↔GT mappings
- Compute triple, facts, and entity precision/recall/F1

**Facts vs triple metrics:** triple-level scores count typing and taxonomy (`rdf:type`, `rdfs:subClassOf`, …). **Facts** scores measure only instance-to-instance relations (e.g. book → character via an ontology property), excluding schema predicates and triples that touch class/concept nodes in subject or object position. Relation property IRIs in predicate position still count toward facts.

Entity match payloads accept IRI strings or `URIRef` values; evaluation normalizes to `URIRef` for projection. **Entity false positives/negatives** count unmatched entities in each graph (set difference), so a shared ontology vocabulary IRI matched once is not also counted as an extra false positive on the other side.

The `match-graphs` CLI automates this for directory pairs of TTL files.

## Tuning Tips

1. **Inspect merge output** before lowering `AGG_SIMILARITY_THRESHOLD`.
2. **Domain-specific embeddings** — if you change `EMBEDDING_MODEL_NAME` for Qdrant, consider aligning `AGG_EMBEDDING_MODEL` for consistent geometry.
3. **Large documents** — more units increase merge complexity; use `--head-chunks` while tuning.

## Related

- [Workflow](workflow.md) — where merge fits in the pipeline
- [Core Concepts](concepts.md) — disambiguation overview
- [Configuration](configuration.md) — aggregation env vars
