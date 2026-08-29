---
search:
  boost: 3
---

# Ontology Context

Before the LLM renders ontology updates for each content unit, OntoCast assembles **ontology context** — the background TTL/JSON-LD the model sees when extracting concepts.

Context is assembled **per unit** inside the ontology loop, not at document level.

!!! note "Facts units do not re-resolve context"

    When an ontology stage has run, the facts fan-out reuses one merged view of
    the document's ontologies (assembly mode `document_merged_reduced`) rather
    than resolving per unit again. The mode below therefore governs the
    **ontology** loop, plus facts-only entry points (`RENDER_MODE=facts` and
    `/process_unit`), where there is no prior ontology stage to merge.

    "Reduced" there names the map/**reduce** stage, not size reduction: the
    merge is a plain union of every ontology artifact with `owl:Ontology`
    headers stripped. Its size is bounded by
    [`ONTOLOGY_CONTEXT_MAX_TRIPLES`](#how-large-is-the-context), like every
    other prompt.

## How large is the context?

Only vector mode bounds retrieval itself; every mode is bounded at
serialization by `ONTOLOGY_CONTEXT_MAX_TRIPLES` (default `4000`).

| Mode | What reaches the LLM |
|---|---|
| `selected_single_ontology` | The whole selected catalog ontology, condensed to the budget |
| `fixed_single_ontology` | The whole pinned ontology, condensed to the budget |
| `selected_vector_search_ontology` | A retrieved subgraph capped at `VECTOR_STORE_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES` (`1200`), which binds first. That is a growth budget during expansion, not a final ceiling — the small-module closure runs after it and can add whole modules past it, which the budget above then catches |
| Facts prompts | Union of all ontology artifacts, condensed to the budget |

Cost per triple, measured through the prompt serializers: **50.7 chars in
Turtle, 102.6 in JSON-LD** — so the default budget is ~51k tokens as Turtle and
~103k as JSON-LD. See [Performance](performance.md#how-much-a-triple-costs).

Condensing drops header/list noise first, then redundant structure (generic
types, stub restrictions, orphan blank nodes), then glosses (`rdfs:comment`,
`skos:definition`, `skos:scopeNote`, `skos:altLabel`). Labels, types, hierarchy
and domain/range are never dropped, so a catalog that cannot fit is passed
through oversized with a warning rather than silently gutted — see
[Configuration](configuration.md#ontology-context-size-ontology_context_max_triples).

## Context Modes (`ONTOLOGY_CONTEXT_MODE`)

Set via `ONTOLOGY_CONTEXT_MODE` (server default) or per-request `ontology_context_mode`.

### `selected_single_ontology` (default)

The LLM selects one catalog ontology per content unit from seed ontologies in the triple store / `ONTOCAST_ONTOLOGY_DIRECTORY`.

- Does **not** require a vector store
- Vector store initialization is skipped unless vector mode is requested
- Optional `ontology_selection_user_instruction` guides selection
- **Costs one extra LLM call per content unit** — the selection is itself a
  provider call, charged to the unit's budget tracker, on top of the render and
  any repair calls. The other two modes make no selection call.

### `selected_vector_search_ontology`

Retrieves a stitched ontology ensemble from the configured vector store using hybrid dense + BM25 retrieval, then expands an induced subgraph subject to triple budgets.

**Requires one vector backend:**

| Backend | Configuration |
|---------|---------------|
| **Qdrant** (server) | `QDRANT_URI` (and optionally `QDRANT_API_KEY`) |
| **LanceDB** (embedded) | `LANCEDB_ENABLED=true`, `LANCEDB_DATA_DIR=~/.lancedb_data` — install with `uv sync --extra lancedb` |

Also required: compatible `EMBEDDING_*` settings and indexed ontology atoms in the active tenant/project partition (Qdrant collection or LanceDB table).

!!! note "This is the only mode that runs the consistency critic"

    The `CONSISTENCY_CRITIC` stage returns immediately unless the effective mode
    is `selected_vector_search_ontology` (it also needs a live vector store and
    at least one ontology artifact). Under the other two modes the stage is
    present in the graph but does no work — so switching away from vector mode
    silently drops that check.

On `ToolBox.initialize` in this mode:

- Orphan ontology IRIs (indexed but absent from the synchronized catalog) are pruned by default (`VECTOR_STORE_PRUNE_ORPHAN_IRIS_ON_INIT=true`) so renamed ontologies do not linger in retrieval.
- Optional clean slate: `VECTOR_STORE_WIPE_ON_INIT=true` or CLI `--wipe-vector-store` drops the current partition before recreate+reindex.

`QDRANT_URI` and `LANCEDB_ENABLED=true` are mutually exclusive.

If vector infrastructure is unavailable, the API returns **409** with `error_code: VECTOR_STORE_UNAVAILABLE`.

**Key budget settings** (full reference: [Configuration — Ontology Patch Retrieval](configuration.md#ontology-patch-retrieval)):

Default path: per-window channel fusion → max-score IRI dedupe → global score order → window-scaled hard cap → expand. Setting a non-zero `ONTOLOGY_PATCH_PER_ONTOLOGY_SEED_QUOTA` inserts a per-ontology round-robin (visiting ontologies best-scoring first) in place of plain score order.

| Variable | Default | Role |
|----------|---------|------|
| `VECTOR_STORE_TOP_K` | `20` | Hits per channel per proposition window |
| `VECTOR_STORE_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES` | `1200` | Global triple cap for context |
| `VECTOR_STORE_INDUCED_SUBGRAPH_DEPTH` | `2` | BFS depth for hub seed expansion |
| `VECTOR_STORE_INDUCED_SUBGRAPH_HUB_SEED_COUNT` | `16` | Top seeds receiving full BFS budget |
| `VECTOR_STORE_INDUCED_SUBGRAPH_ANCESTOR_CLOSURE_DEPTH` | `3` | `rdfs:subClassOf` hops in schema shell |
| `VECTOR_STORE_INDUCED_SUBGRAPH_ESTIMATED_TRIPLES_PER_QUERY` | `24` | Per-entity BFS quota hint |
| `VECTOR_STORE_INDUCED_SUBGRAPH_CANDIDATE_PUSHDOWN` | `false` | Opt-in SPARQL neighborhood CONSTRUCT (see below) |
| `VECTOR_STORE_PROPOSITION_MAX_WINDOWS` | `16` | Window cap; long chunks sample evenly across the text |
| `ONTOLOGY_PATCH_CROSS_QUERY_MERGE_MODE` | `max_score` | Default merge; `sum_score` (rewards multi-window agreement) and `hybrid` are opt-in |
| `ONTOLOGY_PATCH_PER_ONTOLOGY_SEED_QUOTA` | `0` | Max seeds per ontology; `0` (default) uses global score order |
| `ONTOLOGY_PATCH_SEEDS_PER_WINDOW` | `4` | Scales effective atom cap with proposition windows |
| `ONTOLOGY_PATCH_MAX_ATOMS_BASE` | `96` | Floor for the effective atom cap |
| `ONTOLOGY_PATCH_MAX_ATOMS` | `96` | Hard cap: `min(max_atoms, max(base, seeds_per_window × n_queries))` |
| `ONTOLOGY_PATCH_MIN_MERGED_MAX_SCORE` | `0.18` | Empty patch when the best per-window fused score is below this |
| `ONTOLOGY_PATCH_MMR_LAMBDA` | `1.0` | `1.0` skips MMR (default); lower enables diversity rerank |

Advanced (off by default): `ONTOLOGY_PATCH_PER_QUERY_*_SCORE_RATIO`, `ONTOLOGY_PATCH_MERGED_SCORE_RATIO`, hybrid tier-1/tier-2 (`MAX_ATOMS_TIER1`, `MIN_ENTITY_SCORE`).

### Recommended preset for dense scientific text

Use vector search mode with the defaults above. For noisy catalogs, optionally tighten with advanced knobs:

```bash
ONTOLOGY_PATCH_MAX_ATOMS=24
ONTOLOGY_PATCH_MERGED_SCORE_RATIO=0.5
ONTOLOGY_PATCH_MMR_LAMBDA=0.85
VECTOR_STORE_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES=600
```

Tighten only against measured output — see [Diagnostics](#diagnostics). Lowering the seed
budget trades recall away directly, and on a large catalog the defaults are already the
recall-favouring choice.

Effective seed budget grows with proposition windows (`seeds_per_window × n_queries`, floored by `max_atoms_base`, capped by `max_atoms`). Set `ONTOLOGY_PATCH_MERGED_SCORE_RATIO` / per-query score ratios only when you need stricter precision.

Retrieval expands ontology scope beyond hit sources when seeds reference classes
in other catalog ontologies via `rdfs:subClassOf`, `rdfs:domain`, or `rdfs:range`.

### Diagnostics

There is no in-repo recall harness any more (it was removed in the 2026-08
test trim; retrieval quality is evaluated with the out-of-repo
`ontocast-validation` benchmark scripts, e.g.
`ontocast-validation/run/build_recall_corpus.py` for corpus construction).
When comparing configurations, note two things about recall measurement: the
**case**-level figures saturate as soon as cases carry several expected terms,
so per-**term** figures are the numbers to compare on; and approximate
nearest-neighbour search is not bit-reproducible across index builds, so
treat sub-percentage-point differences as run-to-run noise.
`test/test_retrieval_predicate_recall.py` covers predicate-surface indexing
in-repo.

Per-run metrics are available in production on
`state.retrieval_metrics["patch_retrieval"]`: `atoms_after_dedupe`, `atoms_final`,
`seed_iris`, `seeds_by_ontology`, `snapshot_triple_count`, `snapshot_pruned_uri_count`,
`snapshot_uri_components`.

#### Catalog I/O

Ensemble retrieval runs once per content unit, so how it reads the ontology catalog
dominates its cost. It never materializes the whole catalog: the seeds' cross-ontology
`rdfs:subClassOf` / `domain` / `range` references are resolved with targeted SPARQL
SELECTs, and only the ontologies that survive that filter are then fetched as graphs.
Backends without SPARQL (see [Triple Stores](triple_stores.md#custom-backends)) fall back
to the full-catalog scan and return identical results, just more slowly.

Graphs and their merged union are then served from `OntologyManager`, which is the single
read path for ontology graphs — see [Ontology Catalog](../architecture/ontology_catalog.md)
for why that is sound and where the responsibility boundary sits. In practice a document
pays for each ontology once, not once per content unit.

| Key | Meaning |
|---|---|
| `catalog_access_mode` | `sparql`, or `full_fetch_fallback` when the backend has no SPARQL or a query failed |
| `catalog_select_queries` | SELECTs issued (header catalog + reference hops) |
| `catalog_graphs_fetched` | Ontology graphs materialized for the expansion step |
| `catalog_context_mode` | `merged_catalog` (default) or `sparql_candidate` (pushdown) |
| `catalog_context_triples` | Size of the working graph the snapshot was built from |
| `catalog_graph_cache_hits` / `_misses` | Per-version ontology graph cache |
| `catalog_merge_cache_hits` / `_misses` | Merged working-graph cache, keyed by version set |

A `full_fetch_fallback` on a Fuseki deployment means queries are failing — retrieval is
degrading to slow rather than failing outright, which is worth investigating.

#### Candidate pushdown (opt-in)

`VECTOR_STORE_INDUCED_SUBGRAPH_CANDIDATE_PUSHDOWN=true` builds the working graph from a
single SPARQL `CONSTRUCT` of the seeds' bounded neighborhood instead of merging whole
ontology graphs. It requires a backend with `supports_sparql_construct()`, and silently
uses the merge path otherwise.

Only *candidate generation* moves into the engine. The budgeted admission that follows —
per-seed quotas, the global triple cap, connectivity repair, component pruning — stays in
Python, because every triple it admits depends on how many have been admitted already.

**It is off by default, and you should measure before turning it on.** Compare
`catalog_context_triples` between the two modes on your own corpus. On the Text2KGBench
benchmark set — 6 ontologies, 976 triples merged, 36 seeds — the candidate graph is **83%
of the merged graph**, so there is nothing to gain there. The value is bounding memory and
wire volume on a *large* catalog, where the seeds' neighborhood is a small fraction of what
is stored.

One known asymmetry: the cross-component schema-path repair can search past the fetched
neighborhood, so a rare connectivity bridge may be missing. That makes a snapshot
*smaller*, never wrong.

### Induced-subgraph behavior (defaults)

After seeds are chosen, expansion builds a budgeted snapshot:

- **All seed-bearing components are kept.** Seedless components are dropped; references to
  dropped IRIs are removed so the snapshot never names a term it does not define.
- **Individual seeds stay alongside their classes.** An individual's `rdf:type` classes are
  promoted into the seed set without discarding the individual (needed for the facts
  two-namespace contract's reference individuals). The individual **keeps its own
  retrieval score**; promoted classes inherit
  `score × VECTOR_STORE_INDUCED_SUBGRAPH_TYPE_PROMOTION_SCORE_FACTOR` (default `1.0`).
  Score ties in seed ordering break by retrieval rank, never by raw IRI bytes — byte
  order systematically starved `https://…` seeds behind `http://…` ones under tight
  budgets. `VECTOR_STORE_INDUCED_SUBGRAPH_SEED_ORDER=ontology_round_robin` optionally
  interleaves seeds across source ontologies instead of global score order.
- **BFS admission prefers schema role** (label, `rdf:type`, hierarchy, domain/range, then
  descriptions) when a seed's quota cannot hold a full level — not alphabetical order.
- **Seed symbols reach the prompt.** Notation predicates
  (`VECTOR_STORE_INDUCED_SUBGRAPH_SYMBOL_PREDICATES`, defaulting to the lexical-trigger
  predicates `skos:notation` / `qudt:symbol` / `qudt:ucumCode`) are admitted as seed
  descriptions between names and glosses, so tight budgets drop comments, not the short
  codes that let the LLM map a surface token like `meV` to its IRI.
- **Version/hash filters select, never exclude.** Retrieval hits pin the ontology
  versions/hashes their atoms were indexed from; the catalog is filtered to those. When
  no catalog entry matches, the filter relaxes per IRI to same-version entries, then to
  any entry, with a warning — instead of silently dropping the whole ontology from the
  prompt context. Content hashes are computed over the RDF *value* space, so they are
  stable across a triple-store round trip and identical between backends; the relaxation
  ladder is defense in depth, and a `catalog identity drift` warning in the log now
  signals a real problem rather than routine hash noise.
- **Only used prefixes are advertised.** The snapshot binds one prefix per namespace that
  actually appears in its triples (canonical vocabulary names preferred, e.g. `qudt:` over
  a stem-derived alias). Previously every merged ontology's prefixes were bound
  regardless of content, so prompts claimed namespaces the LLM could not see a single
  term from.
- **Author `@prefix` names survive the triple store.** Prefix bindings are serialization
  metadata that a SPARQL store never holds, so exports used to re-derive synthetic stem
  names (`matsci_units:` for a namespace the author binds as `matsciunits:`). On catalog
  registration and store write, used non-well-known bindings are persisted as SHACL
  prefix declarations (`sh:declare [ sh:prefix … ; sh:namespace … ]`) on the ontology
  subject — excluded from the content hash, so identity is unaffected — and rebound on
  fetch before implicit-stem recovery fills any remaining gap. Snapshot binding priority
  per namespace: canonical vocabulary name → author-declared name → plainest candidate.
  Both context paths (merged catalog and SPARQL candidate pushdown) recover the same
  names.
- **Long chunks** split into up to `VECTOR_STORE_PROPOSITION_MAX_WINDOWS` windows sampled
  at an even stride from start to end, so the tail of a long passage still issues queries.

### What becomes an atom

An ontology graph *mentions* far more IRIs than it *defines*. Every object of a
`qudt:hasDimensionVector`, `qudt:applicableSystem` or `owl:versionIRI` triple is an IRI
the ontology references without describing. Those are not indexed: an ontology mints an
atom only for a term it describes — one with a subject-position triple, or a label.

This matters more than it sounds. A referenced IRI has no local text, so its atom is its
mangled local name (`a0e0l2i0m1h0t 3d0` for a QUDT dimension vector), and meaningless
token strings embed near the corpus centroid — they are near-equidistant from every
query, so they surface against all of them. On the 8-module matsci catalog, 247 of 690
atoms (36%) were such references, and dimension vectors alone took 51 of 140 dense
retrieval slots on one document, pushing four ontologies out of the results entirely.

Referenced IRIs stay reachable — induced-subgraph expansion walks into them from seeds.
They just stop being seeds. Set `VECTOR_STORE_INDEX_UNDESCRIBED_IRIS=true` to restore
the old scope; both it and `VECTOR_STORE_EMBED_STANDARD_VOCAB_IRIS` change which atoms
exist and require a reindex.

A corollary worth knowing when authoring modules: a term only reaches the semantic lane
from the module that *describes* it. If a vocabulary is meant to contribute its terms,
vendor the declarations — referencing the IRIs is not enough.

### BM25 / index recreate

Lexical retrieval uses Qdrant's BM25 with IDF and indexes split local names plus
`rdfs:label` / `skos:prefLabel` / `dcterms:title` / `skos:altLabel`, and — for
quantity vocabularies — `qudt:symbol` / `qudt:ucumCode`. Collections built
before that contract fail loudly with `EmbeddingContractMismatchError` — wipe and reindex
(`VECTOR_STORE_WIPE_ON_INIT` or `--wipe-vector-store`).

The sparse lane carries substantial fusion weight (`VECTOR_STORE_FUSION_BM25_WEIGHT`,
default `0.8`, against `0.7` core and `0.15` neighborhood). Terms identified by a symbol
rather than a phrase are frequently absent from the dense lanes altogether, so the sparse
lane is not a tie-breaker for them — it is the only evidence there is. At the former
`0.2` the normalized weights were `0.583 / 0.250 / 0.167`, meaning a rank-1 BM25 hit was
outvoted 3.5:1 by a rank-1 dense hit; `matsci-units#millielectronvolt` was a rank-1 BM25
hit for a passage reporting `∼10−50 meV`, appeared in no dense lane, and still lost.

Surface forms are capped per atom (`VECTOR_STORE_MINIMAL_LABEL_LIMIT`, default 5) and
selected deterministically, ranked by predicate and then by language. Two consequences
are worth knowing when indexing an external vocabulary:

- **Symbols get their own budget.** They are not queued behind labels, so a term
  declaring many labels cannot crowd them out — QUDT's `unit:DEG_C` carries 23 labels
  against a cap of 5. Symbols lead the sparse lane and follow the primary label in the
  core representation, so the entity keeps a readable name.
- **Untagged and `en*` literals rank first.** Other languages are demoted, not dropped.
  Without this a term is named by whichever language sorts first alphabetically, and the
  cap fills with translations no English-language query matches.

This is what makes a **small** vocabulary findable by symbol in the sparse lane. It does
**not** make indexing a large symbol vocabulary into the shared semantic pool practical:
thousands of near-identical unit embeddings cluster together and displace domain terms under
the global atom cap (measured on the matsci recall corpus when QUDT was indexed
wholesale).

### Lexical-trigger lane (exact-match codes)

Some catalog terms are identified by a **literal token** in source text — unit symbols
(`meV`), chemical formulae (`CsPbBr3`), gene symbols, CAS numbers — rather than by
paraphrase similarity. Those are handled by a separate **lexical-trigger** lane:

- At **index** time each atom stores `lexical_triggers`: case-preserved tokens from
  `skos:notation`, `qudt:symbol`, `qudt:ucumCode`, and (optionally) code-shaped
  `rdfs:label` / `skos:altLabel` values.
- At **query** time the raw content-unit text is scanned case-sensitively; matching atoms
  fuse with the semantic hits at `VECTOR_STORE_LEXICAL_TRIGGER_SCORE` (default `0.35`,
  calibrated against fused reciprocal-rank scores). Under the default
  `VECTOR_STORE_LEXICAL_TRIGGER_FUSION=max_merge`, an atom retrieval already found is
  **promoted** to `max(semantic, trigger)` score — a case-exact notation match is
  evidence, not a duplicate — and unseen atoms are appended (cap
  `VECTOR_STORE_LEXICAL_TRIGGER_MAX_ATOMS=16`, outside the semantic atom budget).
  `append` restores the legacy add-only behavior for ablation.
- Toggle with `VECTOR_STORE_LEXICAL_TRIGGER_ENABLED` (default on). Predicate list and
  heuristic promotion are configurable via `VECTOR_STORE_LEXICAL_TRIGGER_*` — see
  [Configuration](configuration.md).

Requires a **reindex** after upgrading: the embedding contract fingerprint bumps to `sf3`.

### `fixed_single_ontology`

Always uses one catalog ontology identified by `ontology_context_fixed_ontology_id` (env: `ONTOLOGY_CONTEXT_FIXED_ONTOLOGY_ID` or per-request parameter).

The value may be:

- the catalog **ontology IRI** (canonical),
- the short **`ontology_id`** (e.g. `observation`), or
- the author **prefix** (e.g. `obs` when the Turtle binds `obs:` to that ontology namespace).

Returns **400** if the mode is fixed but no ontology id is provided. `ontocast serve` and `ontocast process` are stricter still: a fixed mode with no configured id fails at **startup**, not per request.

!!! warning "An id that matches nothing degrades silently"

    A *missing* id is a 400. An id that is present but matches no catalog entry
    is **not** an error: it logs a warning and the unit renders against an
    **empty ontology snapshot**, which usually looks like a bad extraction
    rather than a misconfiguration. Check the catalog
    (`GET /ontologies`) if a fixed-mode run suddenly produces sparse output.

## Ontology identity

- **Canonical catalog key** is the ontology IRI (`owl:Ontology` subject).
- **`ontology_id`** and author **`prefix`** are aliases registered on ingest; they may differ (e.g. `lifecycle` vs `life`).
- **Prompt snapshots** (`OntologySnapshot`) are views of catalog graphs (or stitched ensembles) with `source_iris` / `writable_iris` provenance — they have **no** catalog id. Single/fixed modes wrap one ontology; vector/merged modes assemble a multi-source graph.
- **Writeback** (`U → O*`): insert complements (`U \\ S`) are partitioned by namespace ownership onto writable catalog IRIs and merged onto each ontology's freshest terminal. Snapshots are never registered in the catalog.

## Assemble → propose → apply

```text
assemble:  catalog ontologies O*  →  OntologySnapshot S
propose:   (S, text) → GraphUpdate U   (complement only; do not restate S)
apply:     (U, O*) → versioned catalog ontologies O*'
```

Ontology-update prompts use **complement** framing (single writable IRI or multi-source). Facts prompts treat S as read-only schema (`cd:` for instances).

**In vector-retrieval mode, S is a lossy projection of O\*** — a stitched
induced subgraph, not the full ontologies — while U is applied to the *full*
catalog terminals. The prompts say so explicitly (a PARTIAL CONTEXT notice in
the render intro and critic criteria), and three reduce-time policies close
the gap that partiality opens: minted-duplicate reconciliation against the
full terminals, a redeclare-only delete policy, and fresh-path union merging.
See [Facts Validation and SHACL → Reduce-time policies](validation.md#reduce-time-policies-the-terminal-is-the-authority)
and the workspace design note `planning/ontology-update-semantics.md`.

## Per-Request Overrides

All modes can be overridden on `/process` and `/process_unit`, from the query
string, a JSON body, or a multipart form field alike. Precedence is
`query parameter > JSON/form body > ONTOLOGY_CONTEXT_MODE > selected_single_ontology`,
and an unrecognised value is rejected with **400**.

!!! warning "`ontology_context_fixed_ontology_id` overrides the mode, silently"

    A non-empty `ontology_context_fixed_ontology_id` **forces**
    `fixed_single_ontology` — it beats an explicit `ontology_context_mode` on the
    same request and the server default, with no error and no warning in the
    response. This is deliberate (it lets a client pin an ontology without also
    restating the mode), but it means

    ```text
    ?ontology_context_mode=selected_vector_search_ontology
    &ontology_context_fixed_ontology_id=legal_core
    ```

    runs in **fixed** mode, not vector mode. Send an empty
    `ontology_context_fixed_ontology_id` to use any other mode.

```bash
curl -X POST "http://localhost:8999/process?ontology_context_mode=fixed_single_ontology&ontology_context_fixed_ontology_id=legal_core" \
  -F "file=@contract.pdf"
```

JSON body equivalent:

```json
{
  "text": "...",
  "ontology_context_mode": "selected_vector_search_ontology"
}
```

## Seeding the Catalog

1. Place TTL files in `ONTOCAST_ONTOLOGY_DIRECTORY`, or
2. Upload via `POST /ontologies` (see [API Endpoints](api.md))

Ontologies are synced to the triple store on startup when configured.

## Vector Indexing

When a vector backend is configured (Qdrant or LanceDB) and vector mode is used, ontology atoms are embedded (core + neighborhood representations) and upserted into the tenant/project ontologies partition. BM25 sparse vectors provide a lexical retrieval lane fused with dense scores.

- **Qdrant** — collections `{tenant}--{project}--ontologies` / `--facts`
- **LanceDB** — tables with the same naming pattern under `LANCEDB_DATA_DIR`

Dedup policy (`VECTOR_STORE_DEDUP_MODE`): `iri` (one point per entity key) or `atom_id` (every atom variant).

## Related

- [Configuration](configuration.md) — full env var reference
- [Ontology Catalog](../architecture/ontology_catalog.md) — `OntologyManager` read path and cache
- [Tenancy](tenancy.md) — collection naming and catalog reset on tenant switch
- [User Instructions](user_instructions.md) — selection and extraction guidance
