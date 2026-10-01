# How ontology retrieval works

This page explains how `selected_vector_search_ontology` builds a content
unit's ontology context: what is indexed, how the unit's text becomes queries,
how hits are scored and cut, and how the kept terms grow into the subgraph the
model is shown. Read it before tuning retrieval settings or changing
`ontocast/tool/vector_store/`. To choose a mode and set retrieval up, see
[Choosing ontology context](../guides/ontology_context.md).

## The path at a glance

```text
index time   catalog ontology -> atoms -> core vector, neighborhood vector, sparse text

query time   unit text -> query windows -> three lanes per window -> rank fusion
             -> relevance floor -> merge across windows -> atom cap
             -> lexical triggers -> symbol case check -> seeds
             -> query unit signals -> reference expansion -> schema closure
             -> induced subgraph -> small-module closure -> ontology context
```

| Step | Module |
|---|---|
| Mode dispatch, empty-context diagnosis | `ontocast/stategraph/context_resolver.py` |
| Atomization | `ontocast/tool/vector_store/atomizer.py` |
| Backends and the index fingerprint | `ontocast/tool/vector_store/qdrant.py`, `lancedb.py`, `util.py` |
| Query windows | `ontocast/tool/chunk/proposition.py` |
| Fusion, cut, triggers, closures | `ontocast/tool/vector_store/patch_retriever.py` |
| Exact-match lanes | `ontocast/tool/vector_store/lexical_trigger.py`, `query_signals.py` |
| Induced subgraph | `ontocast/tool/sparql.py` |

The context then passes through the same size cap as every other mode,
[`ONTOLOGY_CONTEXT_MAX_TRIPLES`](../reference/configuration/pipeline.md#ontology_context_max_triples).

## Atoms: what is indexed

An ontology contributes one atom per term it **describes**: a term that is the
subject of at least one of its triples (a label counts). Terms it only
references, as the object of a triple, are not atomized. Such a term has no
local text, so its atom would be its local name alone; opaque names embed near
the centre of the index and surface against every query, crowding out real
terms. Referenced terms stay reachable, because subgraph expansion walks into
them from seeds; they are just never seeds themselves.
[`VECTOR_STORE_INDEX_UNDESCRIBED_IRIS`](../reference/configuration/ontology-retrieval.md#vector_store_index_undescribed_iris)
atomizes them too.

It follows that a term reaches retrieval only from a module that describes it.
For an external vocabulary to contribute its terms, include their declarations
in the catalog; referencing their IRIs is not enough.

Terms in the standard vocabularies (RDF, RDFS, OWL, XSD, SKOS, Dublin Core,
PROV, FOAF, SHACL, schema.org) are skipped unless
[`VECTOR_STORE_EMBED_STANDARD_VOCAB_IRIS`](../reference/configuration/ontology-retrieval.md#vector_store_embed_standard_vocab_iris)
is set;
[`VECTOR_STORE_EXTRA_EXCLUDED_NAMESPACE_PREFIXES`](../reference/configuration/ontology-retrieval.md#vector_store_extra_excluded_namespace_prefixes)
adds more. Provenance triples are stripped before atomization. A term declared
as a property, or used as a predicate, gets the predicate role; every other
term the resource role.

Each atom carries three texts, one per retrieval lane:

| Lane | Text |
|---|---|
| Core (dense) | The primary label with the term's informative types, its other labels, up to two descriptions, and the labels of its domain and range |
| Neighborhood (dense) | Structural clues: parents, domain and range, and the edges around the term |
| Sparse | The local name split into words plus the term's surface forms, symbols first. Descriptions are left out, so they cannot dominate term frequency without naming the term |

Surface forms are the literals of
[`VECTOR_STORE_LABEL_PREDICATES`](../reference/configuration/ontology-retrieval.md#vector_store_label_predicates),
in priority order, and of
[`VECTOR_STORE_SYMBOL_PREDICATES`](../reference/configuration/ontology-retrieval.md#vector_store_symbol_predicates).
Symbols are collected against their own budget, so a term with one label per
language cannot crowd them out. Untagged and English literals rank ahead of
other languages, which are demoted, not dropped. The total is capped by
[`VECTOR_STORE_MINIMAL_LABEL_LIMIT`](../reference/configuration/ontology-retrieval.md#vector_store_minimal_label_limit).
Each atom also stores its symbols with their case preserved, for the symbol
case check, and its lexical triggers (below).

Qdrant scores the sparse lane with BM25 and IDF weighting; LanceDB uses its
full-text index.

## Identity and deduplication

An atom's id hashes the ontology IRI, version and content hash, the term, and
its texts, so the same term from two versions of an ontology is two atoms.
[`VECTOR_STORE_DEDUP_MODE`](../reference/configuration/ontology-retrieval.md#vector_store_dedup_mode)
decides what the store keeps: `iri` keeps one record per term, keyed with the
ontology version and hash unless
[`VECTOR_STORE_DEDUP_INCLUDE_VERSION`](../reference/configuration/ontology-retrieval.md#vector_store_dedup_include_version)
or
[`VECTOR_STORE_DEDUP_INCLUDE_HASH`](../reference/configuration/ontology-retrieval.md#vector_store_dedup_include_hash)
is off; `atom_id` keeps every variant. At query time,
[`VECTOR_STORE_DEDUP_QUERY_HITS_BY_IRI`](../reference/configuration/ontology-retrieval.md#vector_store_dedup_query_hits_by_iri)
collapses hits on one term to the best one. Atoms for the `owl:Ontology`
header itself never become seeds.

## What needs a reindex

Each partition stores a fingerprint of the settings that shaped its contents,
and startup fails with `EmbeddingContractMismatchError` when the configuration
produces a different one. The fingerprint always includes:

- [`EMBEDDING_PROVIDER`](../reference/configuration/embeddings.md#embedding_provider),
  [`EMBEDDING_MODEL_NAME`](../reference/configuration/embeddings.md#embedding_model_name)
  and
  [`EMBEDDING_DIMENSION`](../reference/configuration/embeddings.md#embedding_dimension);
- [`EMBEDDING_QUERY_PREFIX`](../reference/configuration/embeddings.md#embedding_query_prefix)
  and
  [`EMBEDDING_DOCUMENT_PREFIX`](../reference/configuration/embeddings.md#embedding_document_prefix),
  which change the embedded text;
- [`EMBEDDING_BM25_MODEL_NAME`](../reference/configuration/embeddings.md#embedding_bm25_model_name);
- the surface-form contract id, `sf6`, which changes when a release changes
  which literals become surface forms or which terms become atoms.

These join it only when set away from their defaults, so an index built at
the defaults keeps its fingerprint: `VECTOR_STORE_MINIMAL_LABEL_LIMIT`,
`VECTOR_STORE_INDEX_UNDESCRIBED_IRIS`, `VECTOR_STORE_EMBED_STANDARD_VOCAB_IRIS`,
`VECTOR_STORE_EXTRA_EXCLUDED_NAMESPACE_PREFIXES`,
`VECTOR_STORE_LABEL_PREDICATES`, `VECTOR_STORE_SYMBOL_PREDICATES`, and the
lexical-trigger settings that shape the stored triggers
(`VECTOR_STORE_LEXICAL_TRIGGER_ENABLED`, `_PREDICATES`, `_HEURISTIC_ENABLED`,
`_MIN_LEN`, `_MAX_LEN`, `_HEURISTIC_MAX_PER_ENTITY`). A Qdrant collection is
also checked for its vector size,
[`QDRANT_DISTANCE`](../reference/configuration/embeddings.md#qdrant_distance)
and its IDF sparse vector.

Every other retrieval setting acts at query time, merge time or expansion
time, and needs no reindex. That includes
[`VECTOR_STORE_INDUCED_SUBGRAPH_SYMBOL_PREDICATES`](../reference/configuration/ontology-retrieval.md#vector_store_induced_subgraph_symbol_predicates),
the retrieval half of `VECTOR_STORE_SYMBOL_PREDICATES`; keep the two in
agreement.

Independently of the fingerprint, `ToolBox.initialize` re-indexes every
catalog ontology on each start, replacing its atoms, with
[`VECTOR_STORE_REINDEX_CONCURRENCY`](../reference/configuration/ontology-retrieval.md#vector_store_reindex_concurrency)
ontologies at a time and
[`VECTOR_STORE_EMBEDDING_BATCH_SIZE`](../reference/configuration/ontology-retrieval.md#vector_store_embedding_batch_size)
texts per embedding request. Before that, orphan pruning deletes indexed ontologies
missing from the catalog; it refuses to run against an empty catalog, and
skips when the catalog loaded only partly, since an absent IRI is then no
evidence of deletion.

## Query windows

A unit is queried as a series of short windows, not as one text
([`VECTOR_STORE_PROPOSITION_RETRIEVAL_ENABLED`](../reference/configuration/ontology-retrieval.md#vector_store_proposition_retrieval_enabled)
off queries the whole unit at once). A window's size is bounded by exactly one
of:

| Bound | Setting | Behavior |
|---|---|---|
| Sentences | [`VECTOR_STORE_PROPOSITION_WINDOW_SENTENCES`](../reference/configuration/ontology-retrieval.md#vector_store_proposition_window_sentences) | Applies unless one of the two below is set |
| Characters | [`VECTOR_STORE_PROPOSITION_WINDOW_MAX_CHARS`](../reference/configuration/ontology-retrieval.md#vector_store_proposition_window_max_chars) | Takes sentences until the budget is met, so it both caps long windows and joins short fragments. A sentence longer than the budget is kept whole |
| Encoder tokens | [`VECTOR_STORE_PROPOSITION_WINDOW_MAX_TOKENS`](../reference/configuration/ontology-retrieval.md#vector_store_proposition_window_max_tokens) | Counts in the unit the encoder truncates on, so below its sequence limit no window is truncated. Cuts an over-long sentence at whitespace. Needs a provider that exposes a tokenizer; without one it approximates by characters and logs that it did |

The sentence splitter breaks on every period, so an abbreviated journal name
becomes several "sentences".
[`VECTOR_STORE_PROPOSITION_ABBREVIATION_AWARE`](../reference/configuration/ontology-retrieval.md#vector_store_proposition_abbreviation_aware)
rejoins fragments split inside an abbreviation, an initial or a citation run.
[`VECTOR_STORE_PROPOSITION_MEASUREMENT_AWARE`](../reference/configuration/ontology-retrieval.md#vector_store_proposition_measurement_aware)
forbids a break between a number and its unit or inside a range; it only
binds where a window may cut inside a sentence, that is under a token budget.

Windows are disjoint by default, so a statement whose halves straddle a
boundary appears whole in no window.
[`VECTOR_STORE_PROPOSITION_WINDOW_STRIDE`](../reference/configuration/ontology-retrieval.md#vector_store_proposition_window_stride)
overlaps sentence windows;
[`VECTOR_STORE_PROPOSITION_WINDOW_OVERLAP`](../reference/configuration/ontology-retrieval.md#vector_store_proposition_window_overlap)
is the fractional form for character and token budgets. Overlap means more
windows.

[`VECTOR_STORE_PROPOSITION_MAX_WINDOWS`](../reference/configuration/ontology-retrieval.md#vector_store_proposition_max_windows)
caps the windows per unit. Over the cap, windows are sampled evenly from start
to end, so the whole unit is still covered but text in a skipped window
reaches no lane. Whether the cap binds depends on how long your units are; once
it does, overlap buys its extra windows by skipping others.

The encoder truncates a window longer than its sequence limit without any
error. `patch_retrieval.queries_truncated` and `query_sequence_limit` in the
retrieval metrics show whether that happens; check them before widening
windows.

## Lanes and rank fusion

Each window is searched in three lanes, each returning
[`VECTOR_STORE_TOP_K`](../reference/configuration/ontology-retrieval.md#vector_store_top_k)
hits;
[`VECTOR_STORE_BM25_TOP_K`](../reference/configuration/ontology-retrieval.md#vector_store_bm25_top_k)
sets a different depth for the sparse lane. The lanes fail differently: dense
lanes drift to topical near-misses, the sparse lane to unrelated text sharing a
token, so the depth at which each stops helping differs.

The lanes are combined by weighted reciprocal rank. A hit at rank *r* in a
lane adds `weight / (constant + r)` to the atom's fused score. The weights,
[`VECTOR_STORE_FUSION_CORE_WEIGHT`](../reference/configuration/ontology-retrieval.md#vector_store_fusion_core_weight),
[`VECTOR_STORE_FUSION_NEIGHBORHOOD_WEIGHT`](../reference/configuration/ontology-retrieval.md#vector_store_fusion_neighborhood_weight)
and
[`VECTOR_STORE_FUSION_BM25_WEIGHT`](../reference/configuration/ontology-retrieval.md#vector_store_fusion_bm25_weight),
are normalized to sum to 1, so only their ratios matter. The constant is
[`VECTOR_STORE_FUSION_RANK_CONSTANT`](../reference/configuration/ontology-retrieval.md#vector_store_fusion_rank_constant).
Raw lane scores only break ties, so BM25 and cosine scales never compete.

Consequences at the default weights and constant:

- An atom ranked first in every lane scores 1, the best attainable score
  (`1 / (1 + constant)` in general).
- A rank-1 hit in the core lane alone scores about 0.42, in the sparse lane
  alone about 0.48, in the neighborhood lane alone about 0.09.
- The sparse lane weighs as much as the core lane because a term known by a
  symbol, such as a unit or a formula, is often absent from the dense lanes,
  and the sparse lane is then its only evidence.
- At constant 0, rank 2 is worth half of rank 1, so the fused order is decided
  mostly by which lane ranked an atom first. Raising the constant makes
  agreement across lanes outweigh position within one.
- A lane of depth *N* hands out ranks 1 to *N* at full weight however weak its
  tail is, so depth acts as a weight.

## From hits to seeds

1. **Relevance floor.** If the best fused score of any window is below
   [`ONTOLOGY_PATCH_MIN_MERGED_MAX_SCORE`](../reference/configuration/ontology-retrieval.md#ontology_patch_min_merged_max_score)
   times the best attainable score, the unit retrieves nothing
   (`threshold_rejected` counts the rejected candidates). Stated as a fraction,
   the floor means the same at every weight and constant.
2. **Merge across windows.**
   [`ONTOLOGY_PATCH_CROSS_QUERY_MERGE_MODE`](../reference/configuration/ontology-retrieval.md#ontology_patch_cross_query_merge_mode)
   `max_score` keeps each term's best window score; `sum_score` adds them up,
   so a term several windows agree on beats one window's top hit.
   [`ONTOLOGY_PATCH_MERGED_SCORE_RATIO`](../reference/configuration/ontology-retrieval.md#ontology_patch_merged_score_ratio),
   off by default, then drops terms below that fraction of the top score.
3. **Atom cap.** The number of terms kept is
   `min(MAX_ATOMS, max(MAX_ATOMS_BASE, SEEDS_PER_WINDOW × windows))`, from
   [`ONTOLOGY_PATCH_MAX_ATOMS`](../reference/configuration/ontology-retrieval.md#ontology_patch_max_atoms)
   (0 means no cap),
   [`ONTOLOGY_PATCH_MAX_ATOMS_BASE`](../reference/configuration/ontology-retrieval.md#ontology_patch_max_atoms_base)
   and
   [`ONTOLOGY_PATCH_SEEDS_PER_WINDOW`](../reference/configuration/ontology-retrieval.md#ontology_patch_seeds_per_window).
   The window count is itself capped, so `MAX_ATOMS` above
   `max(MAX_ATOMS_BASE, SEEDS_PER_WINDOW × VECTOR_STORE_PROPOSITION_MAX_WINDOWS)`
   can never bind; startup warns when it is set there.
4. **Fill order.** Reserves are taken first.
   [`ONTOLOGY_PATCH_PER_ROLE_ATOM_FLOOR`](../reference/configuration/ontology-retrieval.md#ontology_patch_per_role_atom_floor)
   reserves slots for predicate terms: prose reads as noun phrases, so classes
   outscore the properties that connect them.
   [`ONTOLOGY_PATCH_PER_ONTOLOGY_ATOM_FLOOR`](../reference/configuration/ontology-retrieval.md#ontology_patch_per_ontology_atom_floor)
   guarantees each contributing ontology a few slots, so one large ontology
   cannot starve a small module. The rest fills in global score order, or
   round-robin over ontologies (best-scoring first) when
   [`ONTOLOGY_PATCH_PER_ONTOLOGY_SEED_QUOTA`](../reference/configuration/ontology-retrieval.md#ontology_patch_per_ontology_seed_quota)
   caps each ontology's share.
   [`ONTOLOGY_PATCH_MMR_LAMBDA`](../reference/configuration/ontology-retrieval.md#ontology_patch_mmr_lambda)
   below 1 replaces this fill with a diversity rerank over the dense vectors;
   it chooses the whole budget itself, so startup rejects it while either
   floor is non-zero.
5. **Lexical triggers.** Some terms are named by a literal token, such as a
   unit symbol, a chemical formula or a gene symbol, rather than a phrase. At
   index time each atom stores case-preserved triggers from
   [`VECTOR_STORE_LEXICAL_TRIGGER_PREDICATES`](../reference/configuration/ontology-retrieval.md#vector_store_lexical_trigger_predicates),
   plus code-shaped labels when
   [`VECTOR_STORE_LEXICAL_TRIGGER_HEURISTIC_ENABLED`](../reference/configuration/ontology-retrieval.md#vector_store_lexical_trigger_heuristic_enabled)
   is on. At query time the unit's raw text is matched against them
   case-sensitively. Under the default
   [`VECTOR_STORE_LEXICAL_TRIGGER_FUSION`](../reference/configuration/ontology-retrieval.md#vector_store_lexical_trigger_fusion),
   `max_merge`, a term already kept is raised to
   [`VECTOR_STORE_LEXICAL_TRIGGER_SCORE`](../reference/configuration/ontology-retrieval.md#vector_store_lexical_trigger_score)
   if that is higher, and new terms are added, up to
   [`VECTOR_STORE_LEXICAL_TRIGGER_MAX_ATOMS`](../reference/configuration/ontology-retrieval.md#vector_store_lexical_trigger_max_atoms)
   outside the atom cap; `append` only adds new terms. The default trigger
   score sits below a rank-1 hit in the core or sparse lane, so triggers join
   the seeds rather than outrank them.
6. **Symbol case check.** Dense and sparse texts are case-folded, so a symbol
   in prose also retrieves a term whose symbol differs only in case, which for
   unit prefixes is a different unit. A term whose symbols match a query token
   only after case folding, with no exact-case match, is demoted by
   [`VECTOR_STORE_SYMBOL_CASE_MISMATCH_DEMOTE_FACTOR`](../reference/configuration/ontology-retrieval.md#vector_store_symbol_case_mismatch_demote_factor),
   dropped, or kept, per
   [`VECTOR_STORE_SYMBOL_CASE_MISMATCH_POLICY`](../reference/configuration/ontology-retrieval.md#vector_store_symbol_case_mismatch_policy).
   Exact-case and label-only matches are never touched.
7. **Query unit signals.** A token right after a number, as in "200 kV" or
   "4-15 days", is matched case-insensitively and with or without a plural
   "s" against catalog labels and symbols. Matches join the seeds at the
   trigger score, outside the atom cap. The token rules are Latin-script and
   English-centric, so on another script the lane finds less rather than
   anything wrong. It acts at query time only;
   [`VECTOR_STORE_QUERY_UNIT_SIGNALS_ENABLED`](../reference/configuration/ontology-retrieval.md#vector_store_query_unit_signals_enabled)
   turns it off.

## From seeds to subgraph

1. **Reference expansion.** Ontologies declaring classes that the seeds point
   to through `rdfs:subClassOf`, `rdfs:domain` or `rdfs:range` join the set
   (`expanded_ontology_iris` in the metrics).
2. **Version filter.** Each hit names the ontology version and content hash
   it was indexed from, and the catalog is narrowed to those. When no catalog
   entry matches, the filter relaxes for that ontology to the same version,
   then to any version, with a warning, rather than drop the ontology from
   the context.
3. **Working graph.** The selected ontologies are merged into one working
   graph. With
   [`VECTOR_STORE_INDUCED_SUBGRAPH_CANDIDATE_PUSHDOWN`](../reference/configuration/ontology-retrieval.md#vector_store_induced_subgraph_candidate_pushdown),
   a triple store that supports SPARQL `CONSTRUCT` returns just the seeds'
   bounded neighborhood instead; other stores use the merge. Pushdown only
   pays when that neighborhood is a small part of the catalog; compare
   `catalog_context_triples` with it on and off.
4. **Schema closure.** Properties whose domain or range names a kept class, or
   one of its ancestors up to
   [`ONTOLOGY_PATCH_SCHEMA_CLOSURE_ANCESTOR_DEPTH`](../reference/configuration/ontology-retrieval.md#ontology_patch_schema_closure_ancestor_depth)
   levels up, join the seeds, as do the domain and range classes of kept
   properties, up to
   [`ONTOLOGY_PATCH_SCHEMA_CLOSURE_MAX_ENTITIES`](../reference/configuration/ontology-retrieval.md#ontology_patch_schema_closure_max_entities)
   terms. They score below every retrieved seed, so when the triple budget
   binds they never displace a term the text matched.
5. **Induced subgraph.** Expansion spends a triple budget,
   [`VECTOR_STORE_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES`](../reference/configuration/ontology-retrieval.md#vector_store_induced_subgraph_max_total_triples):
    - Property seeds get their definitions first, within a share of the budget.
    - An individual seed keeps its own score and promotes its `rdf:type`
      classes to seeds, at its score times
      [`VECTOR_STORE_INDUCED_SUBGRAPH_TYPE_PROMOTION_SCORE_FACTOR`](../reference/configuration/ontology-retrieval.md#vector_store_induced_subgraph_type_promotion_score_factor).
    - Class seeds are ordered by score, ties broken by retrieval rank, or
      interleaved across ontologies with
      [`VECTOR_STORE_INDUCED_SUBGRAPH_SEED_ORDER`](../reference/configuration/ontology-retrieval.md#vector_store_induced_subgraph_seed_order)`=ontology_round_robin`.
    - Each class seed gets a schema shell: its types and up to
      [`VECTOR_STORE_INDUCED_SUBGRAPH_ANCESTOR_CLOSURE_DEPTH`](../reference/configuration/ontology-retrieval.md#vector_store_induced_subgraph_ancestor_closure_depth)
      `rdfs:subClassOf` steps upward.
    - The top
      [`VECTOR_STORE_INDUCED_SUBGRAPH_HUB_SEED_COUNT`](../reference/configuration/ontology-retrieval.md#vector_store_induced_subgraph_hub_seed_count)
      seeds (0 means all) expand breadth-first to
      [`VECTOR_STORE_INDUCED_SUBGRAPH_DEPTH`](../reference/configuration/ontology-retrieval.md#vector_store_induced_subgraph_depth)
      with most of the remaining budget; the others share the rest by score,
      one level shallower. No seed takes more than
      [`VECTOR_STORE_INDUCED_SUBGRAPH_ESTIMATED_TRIPLES_PER_QUERY`](../reference/configuration/ontology-retrieval.md#vector_store_induced_subgraph_estimated_triples_per_query)
      triples.
    - When a seed's quota cannot hold a whole level, triples are admitted by
      schema role: label, type, hierarchy, domain and range, then
      descriptions. The symbol predicates of
      `VECTOR_STORE_INDUCED_SUBGRAPH_SYMBOL_PREDICATES` come between names and
      descriptions, so a tight budget drops comments before the short codes
      that let the model map a symbol in the text to its term.
    - A final pass restores property and hierarchy links between kept terms,
      repairs connectivity, and prunes degenerate restrictions, redundant
      generic types, disconnected terms and orphan blank nodes. Components
      without a seed are dropped, along with references to what they held,
      so the context never names a term it does not define. Only prefixes
      the remaining triples use are bound.
6. **Small-module closure.** A source ontology that won at least one seed and
   has at most
   [`ONTOLOGY_PATCH_SMALL_MODULE_CLOSURE_MAX_TRIPLES`](../reference/configuration/ontology-retrieval.md#ontology_patch_small_module_closure_max_triples)
   triples is added whole, without its header: a partial view of a tiny
   vocabulary pushes the model to improvise near-miss names. Modules are
   admitted in order of their best seed score;
   [`ONTOLOGY_PATCH_SMALL_MODULE_CLOSURE_MAX_TOTAL_TRIPLES`](../reference/configuration/ontology-retrieval.md#ontology_patch_small_module_closure_max_total_triples)
   caps their total, skipping a module too large for what is left and moving
   on to the next. This runs after the triple budget, so it can take the
   context past it; `ONTOLOGY_CONTEXT_MAX_TRIPLES` still applies. The per-ontology
   atom floor is what lets a small module win its first seed.

## Catalog I/O

Retrieval runs once per content unit, so how it reads the catalog decides its
cost. On a triple store with SPARQL it never loads the whole catalog: the
seeds' cross-ontology references are resolved with targeted `SELECT` queries,
and only the ontologies that survive are fetched as graphs. Stores without
SPARQL fall back to a full scan with the same result. Fetched graphs and their
merged union are cached per version by the ontology manager, so a document
pays for each ontology once; see [Ontology catalog](ontology_catalog.md).

| Metric under `patch_retrieval` | Meaning |
|---|---|
| `catalog_access_mode` | `sparql`, or `full_fetch_fallback` when the store has no SPARQL or a query failed |
| `catalog_context_mode` | `merged_catalog`, or `sparql_candidate` with pushdown |
| `catalog_context_triples` | Size of the working graph the context was cut from |
| `catalog_graph_cache_hits`, `catalog_graph_cache_misses` | Per-version graph cache |
| `catalog_merge_cache_hits`, `catalog_merge_cache_misses` | Merged working-graph cache |

On Fuseki, `full_fetch_fallback` means queries are failing and retrieval has
slowed down rather than stopped; check the server log.

## The consistency critic

After the ontology stage, and only in this mode, the consistency critic takes
up to eight labels and local names from the document's ontology updates and
searches the index for each, keeping the top three hits. A hit from a catalog
ontology the document did not draw on, with a fused score of at least
[`VECTOR_STORE_CONSISTENCY_CRITIC_MIN_FUSED_SCORE`](../reference/configuration/ontology-retrieval.md#vector_store_consistency_critic_min_fused_score),
is reported as a possible conflict in `improvement_suggestions`, up to five;
`retrieval_metrics.consistency_conflicts` counts them. The threshold is on the
fused scale, not a cosine similarity: at the default weights a rank-1 hit in a
single lane stays below 0.5, so the default reports only terms that at least
two lanes rank near the top. The critic changes nothing in the graph.

## Assemble, propose, apply

The context a unit is shown is a snapshot: a view of catalog ontologies that
records the ontologies it was cut from and the ones the unit may write to,
and is never registered in the catalog. An ontology render proposes an update
against that snapshot, and the update is applied to the full catalog
ontologies it names.

```text
assemble   catalog ontologies -> snapshot
propose    (snapshot, unit text) -> update (what to add or remove)
apply      (update, catalog ontologies) -> new catalog versions
```

In vector mode the snapshot is a partial projection of the catalog while the
update lands on whole ontologies. The ontology render and critic prompts say
the context is a retrieved subset, and the reduce stage checks the update
against the full ontologies before writing it; see [Validation and
SHACL](../guides/validation.md).

## Measuring retrieval on your catalog

`test/test_retrieval_recall.py` reports the retrieval funnel for a corpus you
point it at; [Contributing](../contributing.md) describes its setup. Measured
results for reference corpora live in `ontocast-validation`. In production,
read `metadata.retrieval_metrics.patch_retrieval` and, with
[`ONTOLOGY_PATCH_DUMP_ONTOLOGY_RANKS`](../reference/configuration/ontology-retrieval.md#ontology_patch_dump_ontology_ranks),
the per-ontology rank table; [Choosing ontology
context](../guides/ontology_context.md#diagnostics) lists the keys.
