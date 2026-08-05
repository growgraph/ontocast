# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Terms used in these entries

Retrieval and aggregation changes are justified against measurements, and the
entries name the evaluation sets and metrics involved. For readers outside the
project:

- **Evaluation corpora.** *Text2KGBench* is a public benchmark used here as a
  regression guard. The *materials-science corpus* is an internal evaluation
  set: eight mutually referencing ontology modules (a domain vocabulary, a
  units vocabulary, a qualified-value vocabulary, and others) with passages of
  real scientific prose. Individual passages used for tuning are referred to by
  a case number. Results measured only on the internal corpus are single-corpus
  fits and are flagged as such.
- **Seed recall vs. snapshot recall.** *Seed recall* is the share of expected
  ontology terms that survive retrieval ranking and budget truncation.
  *Snapshot recall* is the share that are actually defined in the ontology
  graph handed to the model — a term can be absent from the seeds yet still
  reach the model by being pulled in as a neighbour of one that was retrieved.
- **On-topic precision.** The share of terms in the assembled ontology context
  that belong to the ontology a given passage is about. It is reported for
  context, not optimised: a missing term cannot be used at all, whereas a
  surplus one only consumes prompt space. The measure also penalises correct
  multi-ontology contexts, since a units or provenance term legitimately
  drawn in from a sibling module counts against it.
- **Surface-form contract (`sf3`, `sf4`, …).** A version stamp on how ontology
  terms are converted into indexed text. Changing it changes the stored
  vectors, so a bump requires re-indexing existing collections; entries that
  bump it say so explicitly.


## [0.5.0]

### Breaking

- **`ontocast serve` binds loopback by default.** The bind interface moved to
  `HOST` (default `127.0.0.1`); it was hardcoded to `0.0.0.0`. The server has
  no authentication and exposes a destructive `POST /flush` that can target any
  tenancy partition, so exposing every interface is now an explicit choice.
  Containers and any non-loopback bind must set `HOST=0.0.0.0`, which logs a
  warning recommending an authenticating proxy.
- **`PARALLEL_FACTS_RETRIES` / `PARALLEL_ONTOLOGY_RETRIES` removed.** Both were
  read by nothing; the per-unit retry budget is and was `MAX_VISITS_PER_NODE`.
  Setting them never had any effect, so removing them changes no behaviour --
  but a deployment passing them will now fail fast on an unknown setting rather
  than silently ignoring them.
- **`/process` answers 422 when no content unit produced output.** Previously a
  document whose units all failed -- or whose conversion failed -- returned
  HTTP 200 with `status: "success"` and empty facts, indistinguishable from a
  document with nothing to extract. Malformed request parameters now answer
  400 rather than 500, and `/process` maps a conversion failure to 422 for
  parity with `/process_unit`.
- **Surface-form contract `sf5` -> `sf6` — reindex required.**
  `dcterms:alternative` joins the label predicates (atomizer labels,
  annotation set, and seed gloss predicates): a vocabulary using the standard
  DCMI synonym predicate previously lost every synonym in both retrieval
  lanes and the prompt. Atoms additionally carry case-preserved
  `symbol_surfaces` (`skos:notation` / `qudt:symbol` / `qudt:ucumCode`
  literals before case folding), persisted through both Qdrant and LanceDB
  payloads. Existing collections raise `EmbeddingContractMismatchError` and
  must be dropped and re-indexed.
- **Internal ontology-apply API reshaped for delete propagation.**
  `partition_inserts_by_namespace` is now `partition_triples_by_namespace`
  (it partitions delete deltas too, and accepts `base_overrides`);
  `apply_partitioned_inserts` is now `apply_partitioned_updates`, takes
  `partitioned_deletes` / `base_overrides`, and returns
  `(artifacts, metrics, applied_updates)`. `normalize_ontology_units` gained
  `delete_graph`. `build_ontology_delta_graph` returns an `OntologyDelta`
  (insert + delete graphs) instead of a bare insert graph.
  `repair_property_aliases` returns `(rewritten, findings, applied_records)`.
- **A base `pip install ontocast` was unimportable and is now fixed.**
  `ontocast/onto/state.py` imported `docling_core` at module scope while
  `docling-core` was declared only in the `all` extra, so
  `from ontocast.toolbox import ToolBox` -- the README's own example -- raised
  `ModuleNotFoundError` on a default install. `docling-core[chunking]` is now
  a hard dependency; the heavy `docling` converter, `easyocr`, and
  `sentence-transformers` stay optional under the new `doc-processing` extra.
  `docling_core.**` was also removed from `ty`'s `allowed-unresolved-imports`,
  which is what had hidden the breakage from the only static check in CI.
- **The `doc-processing` extra now exists.** Every install doc already told
  users to run `pip install "ontocast[doc-processing]"`, but no such extra was
  defined; pip only warns on unknown extras, so the documented path landed
  users in the failure above.
- **The wheel no longer ships a top-level `data` package.** It carried 15 MB of
  sample PDFs and installed an importable module named `data` into
  `site-packages`. Wheel size 13.2 MB -> 447 KB, and a base install 7.6 GB ->
  667 MB (see the dependency changes below).
- **`torch`, `hdbscan`, `umap-learn`, and `langchain-huggingface` moved to a new
  `semantic-chunking` extra, and `duckduckgo-search` to `web-search`.** All were
  already imported lazily with graceful fallbacks, and `sentence-transformers`
  (the default embedding path) pulls `torch` when installed.
- **Removed unused hard dependencies:** `asyncio` (the abandoned PyPI backport
  that shadows the stdlib module), `httpx2`, `numba`, `simsimd`, `rapidfuzz`,
  `owlready2`, `langchain-experimental`. None is imported anywhere in
  `ontocast/`.
- **Declared previously-transitive imports:** `pydantic-settings` (imported by
  `config/settings.py` and reaching the tree only via
  `langchain-experimental` -> `langchain-community`, so removing that unused dep
  would have broken `ontocast.config`), plus `numpy`, `scikit-learn`,
  `starlette`, and `python-dotenv`.
- **`EntityNormalizer.extract_entity_context` returns an `EntityContext`**
  instead of a 6-tuple (`tool/agg/normalizer.py`). External callers unpacking
  the tuple break at the call site. `EntityRepresentation` also gained three
  fields, and a `description_predicates` keyword was threaded through six
  `tool/sparql.py` helpers.
- **`FactsLoopAttempt.graph_hash` was removed** and replaced by
  `n_mandatory_findings` / `repair_failed`. The hash had no consumer anywhere
  and cost a URDNA2015 canonicalization per attempt (see Fixed).
- **CLI `serve` / `process` subcommands and batch TTL output dirs.** The
  `ontocast` console script is now a Click group: `ontocast serve` starts the API
  (was bare `ontocast`); `ontocast process --input-path …` runs local in-process
  batch extraction (was `ontocast --input-path …`). Batch dumps provenance-stripped
  `*.facts.ttl` and `*.ontology.ttl` next to each input by default, or under
  `--output-dir` (shared), with optional `--facts-output-dir` /
  `--ontology-output-dir` overrides. Filename → `dcterms:title` metadata fallback
  when `--document-metadata` is omitted is unchanged.
- **`VECTOR_STORE_CONSISTENCY_CRITIC_SIMILARITY_THRESHOLD` → `VECTOR_STORE_CONSISTENCY_CRITIC_MIN_FUSED_SCORE`, default `0.7 → 0.5`.** The value was never compared against a similarity: `search_patch_hits` returns a weighted reciprocal-rank score (sum of `weight/rank` across the core, neighborhood, and BM25 channels). On that scale a rank-1 core hit alone scores 0.583 and rank-2 scores 0.292, so the old `0.7` silently demanded rank-1 in two channels at once while reading as a cosine cutoff. `0.5` now means "top-ranked in the dominant dense channel".

### Added

- **`py.typed` marker.** The package is fully annotated and `ty check` is a CI
  gate, but without the PEP 561 marker every downstream type checker resolved
  `import ontocast` to `Any`.
- **Per-unit failure records and machine-repair provenance on the response.**
  `ProcessResultMetadata` gained `failed_units` (unit index, phase, stage,
  reason), `facts_repairs` (the deterministic rewrites the 0.5.0 changelog
  advertised as consumer-facing but never exposed), and
  `improvement_suggestions` from the structural check and consistency critic --
  all three were previously written to state and read by nothing.
- **Declared HTTP contract.** `/process`, `/process_unit`, `/flush` and
  `/health` now declare `response_model` and per-status response models, so
  OpenAPI documents them instead of showing parameterless POSTs returning
  `{}`. Every route renders errors in one shape: an app-level handler
  normalizes `HTTPException` (the `/ontologies` routes' `{"detail": ...}`) into
  `StatusErrorBody`. `/info` reports the input types the install actually
  supports rather than a hardcoded list claiming PDF support without the
  `doc-processing` extra.
- **Packaging metadata.** `authors`, `keywords`, `[project.urls]` and real
  trove classifiers. The classifier list had been sitting in `[build-system]`,
  a table where it is inert, so the published package advertised none.
- **`[tool.hatch.build.targets.sdist]`.** Without it hatchling shipped
  everything not VCS-ignored: `data/` (15 MB, plus an importable top-level
  `data` package), `docs/` and `demo/`. The 0.5.0 wheel fix did not cover the
  sdist, which `uv publish` uploads too. Source distribution: ~25 MB -> 591 KB.
- **Dependency upper bounds.** All 31 runtime dependencies were unbounded `>=`
  with floors that could not import the code (`langgraph>=0.2.35` predates the
  `CompiledStateGraph` import path; the resolved version is 1.2.x). Floors now
  name the major actually tested against and every range has a ceiling.
- **Startup validation bounds** on `PORT`, `BASE_RECURSION_LIMIT`,
  `ESTIMATED_CHUNKS`, `PARALLEL_WORKERS` and `ONTOLOGY_MAX_TRIPLES`, which
  previously accepted zero and negative values and failed deep in the pipeline.
- **BM25 case-mismatch penalty for symbol matches**
  (`VECTOR_STORE_SYMBOL_CASE_MISMATCH_POLICY`: `demote` (default) /
  `drop` / `off`, factor `VECTOR_STORE_SYMBOL_CASE_MISMATCH_DEMOTE_FACTOR`).
  The BM25/dense document text is case-folded, so prose `meV` also retrieved
  `unit:MegaEV` (symbol `MeV`) — one token from a 10^9 unit error. At merge
  time, an atom whose symbol surfaces match a query token only
  case-insensitively (with no exact-case match) is demoted or dropped;
  exact-case and label-only matches are untouched. Domain-agnostic: surfaces
  come from declared notation/symbol literals.
- **Machine-repair provenance.** Deterministic rewrites (property-alias
  repairs, `rdf:type` literal coercions) are recorded as `GraphRepairRecord`s
  on the unit state and surfaced document-level as
  `state.facts_repairs_applied`, so downstream consumers can tell
  machine-altered triples from what the LLM asserted.
- **Docs: pyoxigraph integer-subtype collapse documented as an accepted
  limitation** (user guide, triple stores): `xsd:nonNegativeInteger` on
  qualified cardinalities is served back as `xsd:integer`; keep authored
  Turtle as the source of truth when strict OWL 2 DL export matters.
- **A CI test job (`.github/workflows/tests.yml`).** The repository had no
  test job at all -- only pre-commit -- so the entire suite gated nothing. It
  now runs the non-slow suite on Python 3.12 and 3.13, plus an `import-smoke`
  job that builds the wheel, installs it with **no extras**, and imports the
  documented entry points. That job is what would have caught the
  unimportable-package defect above.
- **A tag-vs-version guard in the publish workflow.** `pypi-publish.yml` built
  and published on release with no check that the tag matched
  `pyproject.toml`; this is how 0.4.4 came to exist in `pyproject.toml` with no
  corresponding tag.
- **`FACTS_QUANTITY_FALLBACK_VOCABULARY`.** The facts prompt hardcoded QUDT as
  the fallback for bounded quantities when retrieval supplied no suitable
  class -- and the gate's own `NON_CATALOG_VOCABULARY` finding then reported
  that fallback as a retrieval miss, so the pipeline manufactured its own
  telemetry signal. The vocabulary is now configuration (QUDT remains the
  default; an empty mapping forbids reaching outside the provided context), and
  terms in a configured fallback namespace are reported as a deliberate
  fallback rather than as improvised vocabulary.
- **`CHUNK_CITATION_VOCABULARY`.** The citation-metadata prompt hardcoded
  `schema:ScholarlyArticle` and friends and -- because `CHUNK_BIBLIOGRAPHY_MODE`
  defaults to `citations_only` -- injected them into **every** deployment,
  along with the domain-specific phrase "materials, quantities, measurements,
  methods". Bibliographic terms are now a configurable role-to-term mapping
  (schema.org by default) and the domain nouns are gone.
- **`FACTS_ADDITIONAL_STANDARD_NAMESPACES`.** The "always legal" namespace
  allowlist mixed the RDF/OWL substrate with domain vocabularies (SOSA/SSN,
  CSVW, FOAF, schema.org), permanently exempting them from `UNKNOWN_TERM` with
  no way to change it. Only meta-vocabularies are built in now; domain
  vocabularies are exempted by configuration.
- **`rdfs:domain`/`rdfs:range` schema closure over retrieved seeds
  (`ONTOLOGY_PATCH_SCHEMA_CLOSURE_MAX_ENTITIES`, default 32;
  `_ANCESTOR_DEPTH`, default 2).** Admits the properties whose declared
  domain or range names an admitted class (or one of its ancestors), and the
  domain/range classes of admitted properties. Retrieval ranks terms by how
  prose reads, which favours nouns, so a snapshot could carry a class with no
  property able to link it — inert context the renderer works around by
  improvising a predicate. Deterministic: no embeddings, no LLM call, no
  domain knowledge. Additions rank by the score of the seed that induced
  them, properties first at equal score.
- **Predicate-role seed floor (`ONTOLOGY_PATCH_PER_ROLE_ATOM_FLOOR`, default
  12).** Reserves seed slots for predicate-role atoms before the global fill,
  in the same floor-not-ceiling shape as `PER_ONTOLOGY_ATOM_FLOOR`.
- **`NON_CATALOG_VOCABULARY` validation finding (warning severity).** Reports
  terms the facts graph uses that the ontology context never supplied. When
  the renderer cannot find a catalog term it takes the prompt's documented
  fallback to a well-known vocabulary — correct behaviour, but silent, and
  indistinguishable from success in the output. A fallback firing is evidence
  of a *retrieval* miss, so it now leaves a trace instead of shipping as a
  clean high-scoring graph. Telemetry only: it never drives the un-merge
  repair.
- **Post-aggregation validation gate (`VALIDATE_FACTS` node).** After
  `MERGE_FACTS`, deterministic invariants run over the aggregated graph:
  functional violations (`owl:FunctionalProperty` ∪ OWL max-cardinality-1
  harvest), suspect multi-values (≥ 2 distinct canonical numeric values on
  one predicate; ≥ 2 IRI objects on a dominantly single-valued predicate,
  severity `FACTS_SUSPECT_MULTI_VALUE_SEVERITY`), degenerate coreference
  (one object under ≥ 2 single-valued predicates — collapsed range bounds),
  and optional SHACL (`FACTS_SHAPES_DIR` or `sh:NodeShape` triples inline in
  the ontology context; new extra `shacl` installs `pyshacl`). Error findings
  on merged subjects become full-cluster pair vetoes and the retained facts
  units are re-aggregated (`FACTS_MERGE_REPAIR_PASSES`, default 1).

  This closes a gap that pairwise merge guards cannot close by construction.
  Entity coreference is decided one pair at a time, but the decisions are then
  combined transitively: if the aggregator judges entity A mergeable with B,
  and B mergeable with C, all three collapse into a single node — even when A
  and C carry contradictory values and would never have been merged had they
  been compared directly. No pairwise guard can reject that union, because no
  guard is ever asked about the A/C pair. Checking invariants on the *merged*
  graph catches it after the fact, and vetoing every pair in the offending
  cluster forces the aggregator to keep the entities apart on a second pass.

  Residual findings persist on `AgentState.facts_validation_findings` and in
  retrieval metrics.
- **Bibliography routing (`CHUNK_BIBLIOGRAPHY_MODE`, default
  `citations_only`).** Chunks detected as reference lists — by section label
  (`references`/`bibliography`) or citation-density heuristics (numbered
  citation runs, DOI density, venue tokens) — are extracted as citation
  metadata only (`schema:ScholarlyArticle` + `schema:citation`, no domain
  facts minted from citation titles) or dropped entirely (`skip`);
  `domain_facts` restores the legacy behavior. Numeric-coverage repair is
  suppressed on citation units so page/volume numbers are never pushed into
  the graph.
- **Per-source atom floor and small-module closure for snapshot assembly.**
  `ONTOLOGY_PATCH_PER_ONTOLOGY_ATOM_FLOOR` reserves seed slots per
  contributing ontology before the global fill (a floor, unlike the existing
  quota ceiling), so small modules (e.g. a qualified-quantity vocabulary)
  stop being starved at the atom cap by one dominant ontology.
  `ONTOLOGY_PATCH_SMALL_MODULE_CLOSURE_MAX_TRIPLES` includes a tiny module's
  whole header-stripped graph once any of its atoms is admitted. Both default
  off pending the recall-corpus sweep.
- **Query-side unit signals** (`VECTOR_STORE_QUERY_UNIT_SIGNALS_ENABLED`,
  default off): number-adjacent tokens in the unit text ("4-15 days",
  "200 kV", "0.5 %") match catalog surface forms (labels, `qudt:symbol`,
  UCUM codes) case-insensitively and plural-tolerantly; matched entities
  join the snapshot seeds at trigger score outside the semantic budget —
  so verbatim source units become expressible instead of being converted.
- **Referenced domain/range classes keep their symbol predicates and
  types.** The four induced-subgraph call sites that materialized referenced
  classes without `skos:notation`/`qudt:symbol` triples now pass the
  configured symbol predicates through, and the property-schema-links pass
  admits informative `rdf:type`s.
- **Facts loop telemetry.** Per-attempt records (render/critic/repair, critic
  score, actionable-fix and finding counts, graph hash, triple count) on
  `UnitFactsState.attempt_log`, merged into
  `AgentState.facts_loop_telemetry` with `facts_repair_visits_total` /
  `facts_findings_residual` summary metrics — critic-visit efficacy is now
  measurable instead of the scores being parsed and discarded.
- **Document-level provenance from payload metadata.** Optional `document_metadata`
  on `/process`, `/process_unit`, and `ontocast process --document-metadata '…'` attaches
  caller-asserted identity to the parent `doc_iri` as `prov:Entity` / `foaf:Document`.
  Bibliographic ids (`doi`, `isbn`, …) and `identifiers: [{scheme, value}]` remain
  literal / structured `dcterms:identifier` triples. Business-oriented keys
  (`author` / `creator` / `authors` → `schema:Person`; `project` and any other
  non-reserved key → `prov:Entity`) mint typed RDF entities under the document
  facts namespace with `rdfs:label` (optional `type` / `identifier` dict fields)
  so they are SPARQL-discoverable. Document identity survives chunk-level
  `strip_provenance`. In `ontocast process` batch mode, when no metadata is provided
  the filename (or `file:line` for JSONL records) is used as `dcterms:title`.
- **Retrieval-score preservation in the induced subgraph.** A retrieved individual now
  keeps its own score when its `rdf:type` classes are promoted into the seed set;
  previously the score was transferred to the type and the individual dropped to
  relevance 0, collapsing all typed individuals into a tie that was broken by raw IRI
  byte order (`http://…` before `https://…`) — under a tight triple budget this
  systematically starved high-ranked seeds (e.g. a rank-1 `millielectronvolt` lost to
  `unit:MegaEV` purely by URL scheme). Score ties now break by retrieval rank. New knobs:
  `VECTOR_STORE_INDUCED_SUBGRAPH_TYPE_PROMOTION_SCORE_FACTOR` (default `1.0`) and
  `VECTOR_STORE_INDUCED_SUBGRAPH_SEED_ORDER` (`score` | `ontology_round_robin`; round-robin
  measured neutral on the materials-science recall corpus, kept as an ablation arm).
- **Symbol surfacing for seed nodes** —
  `VECTOR_STORE_INDUCED_SUBGRAPH_SYMBOL_PREDICATES` (default: the lexical-trigger
  predicates `skos:notation` / `qudt:symbol` / `qudt:ucumCode`) admits notation literals
  as seed descriptions, ordered between names and glosses so tight budgets drop comments,
  not codes. Without this a unit individual reached the prompt label-only and the LLM
  could not map a surface token (`meV`) to its IRI.
- **Lexical-trigger score fusion** — trigger hits now carry a calibrated
  `VECTOR_STORE_LEXICAL_TRIGGER_SCORE` (default `0.35`) instead of a hardcoded `1.0`
  that outranked every semantic seed, and `VECTOR_STORE_LEXICAL_TRIGGER_FUSION=max_merge`
  (default) promotes an already-retrieved atom to `max(semantic, trigger)` score — the
  previous behavior silently discarded trigger evidence for any IRI the semantic lanes
  had already found, which was the one case-sensitive signal distinguishing `meV` from
  `MeV`. `append` retains the legacy additive-only mode. Promotion/append counts are
  reported in `last_retrieval_metrics`.
- **Object-property literal quarantine for rendered facts** — a deterministic post-check
  (`FACTS_OBJECT_PROPERTY_LITERAL_CHECK`, default on) quarantines string literals sitting
  on predicates whose schema declares a class range or `owl:ObjectProperty` (e.g.
  `qudt:unit "meV"` where an IRI like `unit:MilliEV` belongs). Quarantined triples flow
  through the existing invalid-literal channel to the facts critic with the declared
  range as a hint (`tool/validate.py::partition_object_property_literal_triples`).
  Facts guideline 8a now states the rule for the renderer (IRI objects resolved via
  case-exact label/notation matching), and the ontology-render prompt's
  `schema:unitCode "DAY"` string-literal example is replaced by unit-IRI-first guidance.
  `validate_predicates` also gains the missing literal-vs-class-range branch.
- **Author prefix persistence through the triple store** — prefix bindings are
  serialization metadata a SPARQL store never holds, so any export re-derived synthetic
  stem names (`dom_units:` where the author wrote `domunits:`) and prompt contexts
  rendered domain terms under names the source ontology never used. Ontology
  registration and store serialization now persist used, non-well-known author bindings
  as SHACL prefix declarations (`sh:declare`) on the ontology subject; fetch rebinds
  them before implicit-stem recovery, and snapshot prefix binding prefers
  canonical → author-declared → plainest candidate. Declarations are excluded from the
  ontology content hash, so stored identities are unchanged and existing indexes stay
  valid. The candidate-pushdown CONSTRUCT gained a header branch pulling the
  declaration blank nodes so both context paths recover identical names
  (`RDFGraph.materialize_prefix_declarations` / `bind_declared_prefixes` /
  `declared_prefix_map`).
- **Lexical-trigger retrieval lane** for notation-bearing vocabulary (unit symbols,
  chemical formulae, gene symbols, etc.). At index time each atom stores case-preserved
  `lexical_triggers` from `skos:notation`, `qudt:symbol`, `qudt:ucumCode`, and optionally
  code-shaped labels. At query time raw chunk text is scanned case-sensitively; matching
  atoms are injected as additive seeds (`VECTOR_STORE_LEXICAL_TRIGGER_MAX_ATOMS`, default
  16) outside the semantic atom budget. Configurable via `VECTOR_STORE_LEXICAL_TRIGGER_*`.
  Embedding contract bumps to `sf3` — reindex required. Matsci recall corpus: added
  a unit individual and a compound individual to one case's expected IRIs.
- **Ablation controls in the recall harness** (test-only; no production change). Asking
  whether indexing a large external vocabulary helps or dilutes previously required
  editing the corpus on disk, which changed the very baseline the arms were compared
  against.
  - `ONTOCAST_RECALL_EXTRA_ONTOLOGIES` (`test/retrieval_gt.py`) — `os.pathsep`-separated
    Turtle files or directories appended to the corpus catalog, so the *index* axis is a
    one-variable flip while the corpus stays byte-identical. A missing entry raises
    rather than being skipped: an arm that silently ran without the vocabulary it was
    meant to measure is indistinguishable from one where the vocabulary did not help.
  - `ONTOCAST_RECALL_COLLECTION_SUFFIX` / `ONTOCAST_RECALL_SKIP_INDEX`
    (`test/test_retrieval_recall.py`) — pin a Qdrant collection, skip teardown, and let a
    later arm score against it without re-embedding. Everything on the *retrieval* axis
    (`ONTOLOGY_PATCH_PER_ONTOLOGY_SEED_QUOTA`, merge mode, caps) is applied at merge
    time, so a whole sweep needs one index instead of one per arm — minutes rather than
    an hour when the catalog includes a 32k-triple vocabulary. Both default to the prior
    behaviour (per-run `uuid4` collection, deleted on teardown).
- CLI `--max-visits` flag for batch mode (`--input-path`) and to override the server default when starting the API.
- Docling converter configuration via `CONVERTER_*` settings, including a `born_digital` preset for text-selectable publisher PDFs.
- `OntologySnapshot` prompt view (graph + `source_iris` / assembly mode, no catalog id) with `from_ontology` / `from_graph` builders; `ontology_apply` complement subtract + namespace-ownership partition onto freshest catalog bases.
- Catalog alias resolution (`ontology_id`, author prefix, or IRI) via `OntologyManager.resolve_ontology_ref`.
- `OntologyManager.aadd_ontology` for non-blocking vector reindex from async callers (`merge_terminal_ontologies` updated).
- `VECTOR_STORE_REINDEX_CONCURRENCY` (default `2`) for bounded parallel ontology materialize/reindex during `ToolBox.initialize`.
- Vector-store init hygiene: `VECTOR_STORE_WIPE_ON_INIT` / CLI `--wipe-vector-store` for a clean-slate drop of the current partition; `VECTOR_STORE_PRUNE_ORPHAN_IRIS_ON_INIT` (default `true`) deletes indexed IRIs absent from the synchronized catalog (e.g. renamed ontology IRIs).
- `ONTOLOGY_PATCH_SEEDS_PER_WINDOW` / `ONTOLOGY_PATCH_MAX_ATOMS_BASE` so the effective seed cap scales with proposition window count.
- **Retrieval recall harness** (`test/test_retrieval_recall.py`, `test/retrieval_gt.py`) — the first measurement of ontology retrieval quality using real embeddings and a real Qdrant collection rather than hash-based fake vectors. Reports **seed recall** (expected term reached `atoms_final`) and **snapshot recall** (expected term is *defined* in the returned graph) plus a per-stage funnel, so a miss is attributable to vector filtering, budget truncation, or component pruning without bisecting. Ground truth comes from a Text2KGBench-style corpus (relation labels resolve to term IRIs by construction, no hand labelling) and from the in-repo anchor fixtures. Scale via `ONTOCAST_RECALL_ROOT` / `ONTOCAST_RECALL_ONTOLOGIES` / `ONTOCAST_RECALL_CASES`; skips when Qdrant is unreachable.
- **`EMBEDDING_QUERY_PREFIX` / `EMBEDDING_DOCUMENT_PREFIX`** — asymmetric retrieval models (BGE, E5) are trained with a distinct instruction on each side and underperform when query and document are encoded identically; there was previously no mechanism to supply one. Empty by default, matching the symmetric paraphrase model. Both are part of the stored embedding contract, so changing either fails loudly with `EmbeddingContractMismatchError` rather than quietly degrading retrieval.
- **`ONTOLOGY_PATCH_CROSS_QUERY_MERGE_MODE=sum_score`** — sums an entity's per-window fused scores instead of taking the best, so a term several windows agree on outranks one window's top hit. Measured **worse** than the `max_score` default under a tight atom cap (seed term recall 36.1% → 29.9% on a linked catalog at `max_atoms=48`) and neutral once the cap is generous (63.9% → 66.0% at 96): summing favours terms mentioned diffusely throughout a passage, and when few seeds survive those crowd out the sharp, specific matches. Kept as a documented non-default so it can be retested per corpus.
- **Term-level recall in the harness.** Case-level recall asks whether *any* expected term survived and saturates as soon as cases carry several — the materials-science corpus reported 100% snapshot recall while under half its expected terms were actually present. `seed TERM recall` / `snapshot TERM recall` count every expected term and are the numbers to compare configurations on.
- **Prebuilt-corpus tier for the recall harness** (`ONTOCAST_RECALL_CORPUS`) — points at a directory holding `ontologies/*.ttl` and `cases.jsonl` (`{"id", "text", "expected_iris", "ontology_iri"}`), so ground truth stays outside this repo and the core carries no domain data. Case text is now split into proposition windows exactly as production does, so multi-sentence passages issue several queries instead of one; single-sentence tiers are unaffected. Build a corpus with `ontocast-validation/run/build_recall_corpus.py`.
- `seed_iris` in `OntologyPatchRetriever.last_retrieval_metrics` (and in `state.retrieval_metrics["patch_retrieval"]`) — the seed entity IRIs behind `atoms_final`, needed to attribute a retrieval miss to a stage.
- **Targeted catalog reads on `TripleStoreManager`**: `aselect(query)` (SPARQL SELECT, rows of lexical values), `afetch_ontology_catalog()` (one `OntologyHeader` per stored version — lineage metadata without graphs), and `afetch_ontologies_by_iri(iris)` (materialize only the named graphs requested; empty means no restriction). All three have working base-class defaults expressed via `fetch_ontologies()`, so existing custom backends keep working unmodified; overriding `supports_sparql_select()` opts into the fast path. Implemented natively for Fuseki (HTTP SPARQL) and in-memory (pyoxigraph `Store.query`, previously unused).
- `OntologyHeader` (`ontocast/onto/ontology_header.py`) — graph-less ontology metadata. Deliberately not an `Ontology`, which recomputes its hash from the graph and would fabricate lineage when constructed without one. `dedupe_terminal_ontologies()` is now generic over both, so terminal-version selection runs identically on headers and materialized ontologies.
- `catalog_access_mode` / `catalog_select_queries` / `catalog_graphs_fetched` in `last_retrieval_metrics`, so a deployment silently degraded to the full-catalog fallback is visible rather than merely slow.
- **`OntologyManager` is now the catalog read path.** New `aget_catalog_headers()`, `aget_ontologies_by_iri(iris)`, `aget_merged_graph(ontologies)`, `register_triple_store()`, `reset_catalog()`, and `catalog_cache_stats()`. Graphs are cached by `versioned_iri` (`{iri}#{sha256}`) and the merged working graph by the `frozenset` of contributing versions. The cache cannot go stale: terminal selection always runs on freshly read headers, and a concurrent writer produces a *new* content address, so a shared-Fuseki deployment with several workers still sees every write. Documented in [Ontology Catalog](docs/architecture/ontology_catalog.md), which is the standing answer to which layer owns what.
- **`supports_sparql_construct()` / `aconstruct(query)` on `TripleStoreManager`** — the first triple-returning read on the interface. Separate from the SELECT pair because a backend can answer row queries without returning triples: Fuseki's SELECT path speaks `application/sparql-results+json` only and needs a different `Accept` header. Implemented for Fuseki (HTTP, Turtle) and in-memory (pyoxigraph `QueryTriples`); the base-class default raises, so custom backends are unaffected.
- **`VECTOR_STORE_INDUCED_SUBGRAPH_CANDIDATE_PUSHDOWN`** (default `false`) — builds the induced-subgraph working graph from one SPARQL `CONSTRUCT` of the seeds' bounded neighborhood rather than merging whole ontology graphs. Only *candidate generation* moves into the engine; the budgeted admission that follows stays in Python, because every triple it admits depends on how many have been admitted already. Off by default and worth measuring first: after the caches above, its remaining value is bounding memory and wire volume on a large catalog, and on a small one the neighborhood is essentially the whole ontology. Compare `catalog_context_triples` across modes. Known asymmetry: cross-component schema-path repair can search past the fetched neighborhood, so a rare connectivity bridge may be absent — a *smaller* snapshot, never a wrong one.
- `select_relevant_ontologies()` and `merge_ontology_graphs()` extracted from `SPARQLTool._build_induced_subgraph`, the former generic over headers and ontologies alike; `filter_overbroad_namespace_map()` is now public.
- **`qudt:symbol` / `qudt:ucumCode` are now retrieval surface forms.** A unit vocabulary is queried by symbol — prose reports `25 meV` and `W/cm2`, never "millielectronvolt" — but only `rdfs:label` / `skos:prefLabel` / `dcterms:title` / `skos:altLabel` were folded into an atom's text, so QUDT's own authoritative symbols were invisible and a corpus had to restate them as `skos:altLabel` by hand to be findable. Symbols are collected against a separate budget rather than appended to the label list: `_collect_literals` honours predicate priority, so a term declaring many labels would otherwise exhaust the surface-form cap before any symbol was reached (`unit:DEG_C` ships 23 labels against a default cap of 5). They lead in the sparse BM25 lane and follow the primary label in the core representation, so a unit keeps a readable name. Symbols also join `_ANNOTATION_PREDICATES`, so they stop being emitted as generic neighborhood clues once they name the entity. Changes stored vectors — requires a reindex.

### Changed

- **`CatalogSurfaceIndex` takes its symbol predicates from configuration.**
  `query_signals.py` hardcoded `qudt:symbol` and `qudt:ucumCode` fifteen lines
  below a docstring claiming "no unit vocabulary is hardcoded" -- and both
  already existed as overridable defaults
  (`VECTOR_STORE_INDUCED_SUBGRAPH_SYMBOL_PREDICATES`,
  `VECTOR_STORE_LEXICAL_TRIGGER_PREDICATES`). The retriever now passes the
  configured list.
- **`tool/chunk/bibliography.py` no longer claims unqualified
  domain-agnosticism.** It is subject-domain-agnostic but English- and
  Latin-script-bound: the section labels, venue abbreviations, and
  citation-marker patterns are all English/Western. The failure mode is a miss
  (the section is extracted as ordinary content), not a misfire; the docstring
  now says so.
- **`aggregate_graphs` / `postprocess_facts_units` return an
  `AggregationResult`** (graph + per-entity decisions + merged clusters +
  rejected-merge count) instead of a bare `RDFGraph`, and accept a
  `merge_vetoes` pair set — the targeted un-merge lever used by the
  validation gate. Call sites use `result.graph`.
- **Ontology sources atomize only IRIs they describe** (surface-form contract `sf3` →
  `sf4`; **every existing collection needs one reindex**). `GraphAtomizer` made an atom
  out of every `URIRef` in a graph — subject, predicate *or object* — so an IRI appearing
  only as the object of `qudt:hasDimensionVector` became a first-class searchable term
  whose entire text was its mangled local name (`a0e0l2i0m1h0t 3d0`). Meaningless token
  strings embed near the corpus centroid, making them *hubs*: near-equidistant from every
  query, so they rank against all of them. On the 8-module materials-science
  catalog 247 of 690 atoms (36%) were such references; on one document the six
  most frequently retrieved dense atoms were all dimension vectors (present in
  5 of 7 proposition windows, against 2–3 for real domain terms), taking 51 of
  140 dense slots and leaving four of the eight modules with no seeds at all.
  An ontology now mints an atom only for a term it describes — a
  subject-position triple, or a label. Measured end to end on that catalog:
  indexed atoms 669 → 443, noise seeds (dimension vector / system-of-units /
  prefix) 13 → 0, seeds from the main domain module 47 → 63, and a unit
  individual needed by the passage moved from seed rank 53 to 15. Referenced
  IRIs stay reachable through induced-subgraph expansion; they just stop being
  seeds. This also retires the junk atoms minted from `owl:versionIRI` and
  `dcterms:license` objects (`"1.3.0"`, `"4.0"`). New knobs, all requiring a reindex:
  `VECTOR_STORE_INDEX_UNDESCRIBED_IRIS` (default `false`) restores the previous scope;
  `VECTOR_STORE_EMBED_STANDARD_VOCAB_IRIS` and
  `VECTOR_STORE_EXTRA_EXCLUDED_NAMESPACE_PREFIXES` expose two `GraphAtomizer` fields that
  previously existed but reached no configuration path at all.
- **Sparse-lane fusion weight `0.2` → `0.8`, neighborhood `0.3` → `0.15`**
  (`VECTOR_STORE_FUSION_BM25_WEIGHT`, `VECTOR_STORE_FUSION_NEIGHBORHOOD_WEIGHT`; no
  reindex). A term whose surface form is a symbol rather than a phrase is frequently
  invisible to the dense lanes, so BM25 is its only evidence — but at `0.2` the
  normalized weights were `0.583 / 0.250 / 0.167`, so a rank-1 sparse hit was outvoted
  3.5:1 by a rank-1 dense hit. A unit individual was a rank-1 BM25 hit for a
  passage reporting `∼10−50 meV`, appeared in no dense lane in any window, and still
  ranked 32nd overall — reaching the prompt only because the lexical-trigger lane
  promoted it after the cap. On the materials-science recall corpus: seed term recall 57.1% → 65.3%,
  snapshot term recall 76.5% → 86.7%. Text2KGBench (the regression guard) improved too:
  seed term recall 77.0% → 79.1%, snapshot 92.0% → 95.1%. Known cost: one small
  module lost its last seed (1/13 → 0/13). Its terms are abstract modelling
  scaffolding rather than words that appear in prose, so they are hard for any
  lane to reach; snapshot recall holds at 10/13 because the graph-expansion stage
  still pulls them in.
  That regression is the missing per-source atom floor, tracked in `docs/PLANNING.md`.
- **Cross-ontology reference ownership is now deterministic.** Attributing a referenced IRI to an ontology previously returned the first catalog entry whose namespace matched, in whatever order the backend listed named graphs — so with nested namespaces (`…/domain` and `…/domain/sub`) the answer could differ between runs. The longest matching namespace now wins, and where several ontologies declare the same IRI the lexicographically smallest owner is chosen. Namespace containment is still tried before graph membership, which is what attributes *dangling* references — an IRI inside an ontology's namespace that the ontology declares no triples about.
- `SPARQLTool.get_induced_subgraph` / `aget_induced_subgraph` accept an optional trailing `ontologies` argument, letting a caller that already holds a catalog skip the fetch. Existing callers are unaffected.
- **`ONTOLOGY_PATCH_MAX_ATOMS` `48 → 96` and `ONTOLOGY_PATCH_MAX_ATOMS_BASE` `32 → 96`** — the largest single lever found. On multi-window input the candidate pool reaches ~167 atoms per passage, so a cap of 48 was discarding two thirds of what per-channel `top_k` had already paid to retrieve. Measured on 15 multi-sentence passages against a linked 6-ontology catalog: **seed term recall 36.1% → 63.9%**, snapshot term recall 47.4% → 56.7%, and on-topic precision *improved* 66.6% → 70.4% — this was not a recall/precision trade, it was a cap set below the useful signal. On the single-sentence 20-ontology corpus, seed term recall 53.0% → 59.6%. Snapshots grow from a mean 303 to 392 triples, still inside the 550 budget, which is the real cost: ~29% more ontology context per prompt.

  This relocates the bottleneck rather than removing it. On the linked catalog snapshot term recall (56.7%) is now *below* seed term recall (63.9%) — with 96 seeds the induced-subgraph triple budget, not the seed count, is what drops terms. Tuning `VECTOR_STORE_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES` and the per-seed quota is the next step, and it trades directly against prompt size.
- **`VECTOR_STORE_TOP_K` default `10 → 20`**, measured. The retained-atom cap stopped binding after the allocation retune, leaving the per-channel candidate pool as the constraint. Sweep on a 20-ontology catalog (160 cases):

  | `top_k` | seed recall | snapshot recall | on-topic precision | `atoms_after_dedupe` | `atoms_final` |
  |---|---|---|---|---|---|
  | 10 | 76.9% | 91.2% | 37.1% | 23.4 | 22.7 |
  | **20** | **80.0%** | **92.5%** | 34.4% | 42.5 | 31.4 |
  | 30 | 80.0% | 93.1% | 33.3% | 60.1 | 31.9 |

  20 captures the entire seed-recall gain; 30 adds nothing there because `ONTOLOGY_PATCH_MAX_ATOMS_BASE=32` binds again (`atoms_final` 31.9), so the extra candidates are fetched and discarded at 1.4× the search cost. Snapshots grow from a mean 190 to 204 triples, still inside the 550 budget. Caveat: the benchmark issues one query per case, so `effective_max_atoms` is 32 there; a 16-window production chunk gets 48 while its candidate pool grows ~16×, so on long inputs the cap binds first and this sweep does not predict the gain.
- **Seed allocation defaults retuned against measured recall** on a 20-ontology catalog: `ONTOLOGY_PATCH_PER_ONTOLOGY_SEED_QUOTA` `3 → 0` (global score order, no per-ontology cap) and `ONTOLOGY_PATCH_MAX_ATOMS_BASE` `16 → 32`. Seed recall **60.0% → 73.1%**, snapshot recall **85.6% → 89.4%**. Dropping the quota improved precision as well (33.8% → 37.4%): spreading a fixed seed budget across every ontology that scored *something* cost accuracy on both axes rather than trading between them. Raising the floor past 32 changes nothing — `atoms_final` saturates at the candidate pool, so per-channel `VECTOR_STORE_TOP_K` is now the binding constraint.

Cumulative effect of the retrieval fixes above, measured on a 20-ontology catalog (160 cases): **seed recall 61.5% → 80.0%**, **snapshot recall 41.5% → 92.5%**. Snapshots grow from a mean 85 to 204 triples, still inside the 550-triple budget, and on-topic precision moves 45.5% → 34.4% — a deliberate trade, since a term absent from the snapshot cannot be used at all while an extra one merely costs context. Both figures come from a corpus of mutually disjoint ontologies (0 cross-ontology schema edges) scored one sentence at a time, which overstates both the component-pruning loss and the precision cost relative to a linked stack, and cannot exercise cross-window merging at all; the direction is solid, the magnitude is corpus-specific.

A second corpus — 6 mutually referencing materials-science ontologies, 15 multi-sentence passages of real scientific prose — confirms that reading rather than contradicting it, and on-topic precision there is **70.6%** rather than the low thirties, because on a linked stack a pulled-in parent term belongs to a sibling ontology by design. Its ground truth is label-match derived, so it favours the lexical lane and reads optimistically in absolute terms.

Final state of both corpora, term level:

| corpus | seed term recall | snapshot term recall | on-topic precision | mean snapshot triples |
|---|---|---|---|---|
| 20 disjoint ontologies, 160 single-sentence cases | 59.6% | 84.3% | 31.1% | 208 |
| 6 linked ontologies, 15 multi-sentence passages | 62.9% | 55.7% | 70.6% | 391 |

Read those two rows together: on single-sentence input the graph stage still *recovers* far more terms than it drops (+128 cases' worth), while on multi-window input it is now net **−7 terms** — seed recall has outrun what a 550-triple snapshot can carry. Retrieval is no longer the limiting stage for realistic documents; the expansion budget is.

Run-to-run variation is roughly ±1 percentage point (approximate nearest-neighbour search is not bit-reproducible across index builds), so smaller differences in the tables above are not meaningful.
- **Per-ontology seed round-robin visits ontologies best-scoring first** instead of in IRI order. The previous ordering made seed allocation alphabetical whenever the atom cap bound before every contributing ontology was served — the common case, since the cap does not grow with catalog size. Still used when a quota is explicitly configured.
- **Ontology snapshot / writeback decoupling**: assemble `O* → OntologySnapshot`, propose complements on a `working_graph` scratchpad, apply `U \\ S` by namespace to catalog terminals. Removed `Ontology.from_working_context` identity lock. Ontology-update prompts use complement framing (single vs multi writable); critic flags restatement / undeclared prefixes.
- **Ontology patch retrieval simplified (defaults)**: default path is max-score IRI dedupe → per-ontology round-robin → window-scaled hard cap. Relative floors, hybrid tier merge, merged-score ratio, and MMR are off by default (`score_ratio=0`, `cross_query_merge_mode=max_score`, `merged_score_ratio=0`, `mmr_lambda=1.0`). `ONTOLOGY_PATCH_MAX_ATOMS` hard cap default is `48`; `VECTOR_STORE_INDUCED_SUBGRAPH_HUB_SEED_COUNT` default is `16`. Lean retrieval metrics: `atoms_after_dedupe`, `effective_max_atoms`, `seeds_by_ontology`.
- **Startup performance**: concatenate core+neighborhood dense embeds into one batched pass (LanceDB/Qdrant); overlap Qdrant BM25 sparse with dense; parallel LLM ontology-property enrich overlapping rematerialize; lazy Docling `DocumentConverter` (built on first convert); defer UMAP/torch/sklearn imports off the cold import path; slim `ontocast` package `__init__` lazy exports for unit loops. Document `VECTOR_STORE_EMBEDDING_BATCH_SIZE` and reindex concurrency in `.env.example` / user guide.
- **OntologyManager async I/O**: patch retrieval sync wrappers now delegate to the async path (same `asyncio.run` / refuse-in-loop pattern as `OntologyPatchRetriever`); vector reindex uses `aadd_ontology` + `asyncio.to_thread`, and sync `add_ontology` refuses reindex when an event loop is running. Fallback patch graphs use `RDFGraph.copy()` instead of nested `deepcopy`.
- **Ontology identity**: catalog key is the ontology IRI; `ontology_id` / author prefix are aliases (may differ). `derive_ontology_id` uses conventional rdflib prefix maps (SKOS → `skos`) and refuses pure-numeric IRI tails. Multi-ontology graphs no longer sync identity from an arbitrary `owl:Ontology` subject. Graph merges rename conflicting prefixes instead of silent override. `ontology_context_fixed_ontology_id` accepts IRI, `ontology_id`, or author prefix.
- **Docs**: align `.env.example` and user-guide defaults for `VECTOR_STORE_*` and `ONTOLOGY_PATCH_*` with `PatchRetrievalConfig` / `VectorStoreConfig` code defaults; document full patch-retrieval parameter table and tuning presets; document snapshot vs catalog and assemble/propose/apply; add [Ontology Catalog](docs/architecture/ontology_catalog.md) architecture page; document catalog I/O metrics, candidate pushdown, multi-component induced subgraphs, proposition window stride, BM25 IDF wipe requirement, consistency-critic fused-score rename, tenancy catalog reset, CLI `--wipe-vector-store` / `--max-visits`, and facts prefix hygiene.
- **Conversion**: `ConverterTool` now builds Docling's standard PDF pipeline from typed config, includes config-aware cache keys, and offers a temporary ligature-gap workaround for publisher-PDF text like `di ff usion`.
- **Prefix / namespace hygiene for facts prompts**:
  - Domain-ontology clause now excludes all rdflib default bindings (`brick`, `csvw`, `geo`, `xml`, …), not only the small `COMMON_PREFIXES` set.
  - Author-declared short prefixes (e.g. `dom:`) are kept canonical; IRI-tail `ontology_id` values no longer force a duplicate prefix binding. Degenerate `nsN` placeholders are still rebound and the old binding is removed.
  - Test fixtures updated to the short ontology IRI stems used by the evaluation catalog.
  - `sanitize_prefixes_namespaces` leaves rdflib reserved namespaces (notably `xml:`) untouched, so `xml1:` is no longer minted.
  - Facts operational guidelines state the domain-ontologies clause once (in the TWO-NAMESPACE CONTRACT) instead of repeating it four times.

### Removed

- **`ONTOLOGY_PATCH_CROSS_QUERY_MERGE_MODE=rrf`.** The mode re-ranked an *unsorted* concatenation of per-window fused hits, so an entity's "rank" was its position in the concatenation rather than its score — not a reciprocal-rank fusion at all. Non-default and unused. `max_score` (default) and `hybrid` are unaffected.

### Fixed

- **A transient Fuseki error could wipe the ontology vector index.** Listing
  named graphs degraded to an empty list on any HTTP error, which
  `ToolBox.initialize` read as "the catalog is empty" and, with
  `PRUNE_ORPHAN_IRIS_ON_INIT` (default on), used as grounds to delete every
  indexed ontology. Listing now raises `TripleStoreUnavailableError`, matching
  the contract already documented for `aselect`/`aconstruct`; a partial catalog
  fetch is reported through `last_catalog_was_complete()` and suppresses the
  prune; and `prune_orphan_ontology_iris` refuses an empty keep-set outright.
  Three failure modes each independently sufficient to destroy the index.
- **Concurrent tenancy switches could cross tenants.** `?tenant=` on a request
  triggers `update_tenancy_with_vector_mode`, which mutates ToolBox-wide state
  (dataset names, ontology catalog, vector-store handles) across awaits, with
  no concurrency cap. Two requests for different tenants could interleave and
  read or write each other's partition. Retargeting and `clean_tenancy_data`
  now hold a loop-bound lock.
- **Per-unit LLM budget attribution was arbitrary under parallelism.** The
  budget tracker was assigned to the shared `LLMTool` instance, so with
  `PARALLEL_WORKERS` units in flight whichever bound last collected every
  concurrent call's usage. Document totals were unaffected (the reduce merges
  all per-unit trackers) but the per-unit numbers reported to API clients were
  not. The active tracker is now a `ContextVar`, so each unit task charges its
  own; `summarize_chunks` and the ontology-selection call, which bypassed the
  accessor entirely, are now scoped too.
- **`AgentState(current_domain=...)` was silently discarded.** A custom
  `__init__` overwrote the caller's value from the environment after
  construction. `current_domain` is now an ordinary field defaulting to
  `CURRENT_DOMAIN` then `DEFAULT_DOMAIN`, so an explicit argument wins;
  `DomainConfig` reads the same variable instead of declaring an unrelated
  placeholder default that nothing consumed.
- **Configuring symbol predicates changed retrieval but not indexing.** The
  retrieval side read `INDUCED_SUBGRAPH_SYMBOL_PREDICATES` from config while
  the atomizer hardcoded the same three IRIs, so overriding the knob changed
  what surfaced without changing what was stored. Both label and symbol
  predicates are now configurable
  (`VECTOR_STORE_LABEL_PREDICATES` / `VECTOR_STORE_SYMBOL_PREDICATES`).
- **The embedding fingerprint ignored settings that change the stored payload.**
  `label_predicates`, `symbol_predicates`, `lexical_trigger_predicates` and the
  four other `lexical_trigger_*` knobs are pushed into the atomizer but were
  omitted from the fingerprint, so changing one silently served a stale index
  instead of raising. They now contribute when they diverge from the defaults;
  collections built at the defaults keep their existing fingerprint.
  `embedding_fingerprint_matches` accepts the same optional components as
  `embedding_model_fingerprint`, having previously disagreed with the validator
  for any non-default collection.
- **Public retrieval APIs contradicted the configured budget.** `retrieve`,
  `aretrieve`, `retrieve_ensemble`, `aretrieve_ensemble` and the six
  `OntologyManager.*patch_context*` wrappers carried literal defaults of
  `subgraph_depth=1` / `max_total_triples=300` against config defaults of
  `2` / `1200`. The pipeline passed config explicitly and was unaffected; every
  other caller silently got a 4x smaller snapshot. All ten now default to
  `None` and resolve from configuration.
- **`FUSEKI_AUTH=user:password` failed startup.** Only the slash form parsed,
  while `docs/user_guide/triple_stores.md` and the quickstart both recommended
  the colon form. Both are accepted now (the separator appearing first wins, so
  a password containing the other still round-trips), and the three docs pages
  were corrected.
- **`ontocast process` exited 0 when every input file failed.** The per-file
  loop logged and swallowed all exceptions with no failure count. It now
  returns the failed paths and the CLI raises a `ClickException`. Bad CLI
  parameters raise `click.BadParameter` instead of a traceback.
- **The detector for empty retrieval was disabled by empty retrieval.**
  `_non_catalog_vocabulary_findings` returned no findings when the ontology
  context was empty -- the exact case where every extracted term is outside the
  catalog. The check stays a pure graph-vs-graph comparison, and the condition
  is now reported by the validate node, which knows an empty context is
  unexpected; the ensemble resolver additionally records *why* a snapshot came
  back empty (`empty_snapshot_reason`: index empty/unreadable, everything below
  threshold, or no match).
- **No backend connection was ever closed.** There was no `ToolBox.close()`,
  `FusekiTripleStoreManager.close` was defined but never called, and the Qdrant
  client had neither a close path nor a timeout. `ToolBox.aclose()` (also an
  async context manager) now releases both, the FastAPI app closes them on
  shutdown, and `QDRANT_TIMEOUT_SECONDS` (default 30) bounds Qdrant calls.
- **Ontology delete writeback: GraphUpdate delete operations now propagate
  through map/reduce to catalog terminals.** Deletes were honored only inside
  the per-unit render loop and then dropped at the reduce stage (with a
  warning) — the catalog could grow but never shrink. The unit delta is now
  computed by replaying all GraphUpdates in order onto the prompt snapshot
  and diffing (so delete-then-reinsert nets out), deletes are reconciled
  across parallel units conservatively (any unit's insert vetoes another
  unit's delete of the same triple), partitioned by namespace ownership like
  inserts, and applied delete-first onto the catalog base. The applied update
  list now reaches `SERIALIZE`, so the existing delete-aware version-bump
  logic (`MAJOR` on deletions) fires — it was previously dead code on the map
  path.
- **Ontology consolidation no longer applies its delta to a stale base.** The
  consolidation pass asked the LLM for a complement of the map-stage artifact
  but applied the result onto the pre-run catalog terminal, then replaced
  `state.ontology_artifacts` wholesale — silently dropping every map-stage
  addition whenever `ENABLE_ONTOLOGY_CONSOLIDATION` fired. The apply now
  bases on the map-stage artifact via `base_overrides`, keeping one clean
  lineage (pre-run -> map -> consolidated).
- **Author-prefix collisions no longer block catalog ingest.** Two ontologies
  binding the same Turtle prefix (`onto:`, `ex:` — every Text2KGBench
  ontology binds `onto:`) raised `ValueError` on the second registration.
  Prefix aliases now degrade: the first binding keeps the alias, later
  ontologies register with a warning and stay addressable by IRI and
  `ontology_id`. Genuine `ontology_id` collisions still raise.
- **Facts: literal `rdf:type` objects are repaired deterministically.**
  Extraction sometimes emitted `a "prefix:Class"^^xsd:string` (JSON-LD
  bare-string type values parse the same way); such nodes are invisible to
  SPARQL class queries and to aggregation's URI minting/entity matching.
  Absolute and graph-bound compact IRIs are coerced to IRIs at the render
  chokepoint; unresolvable forms become MANDATORY `literal_type_object`
  findings driving the repair loop.
- **`facts_findings_residual` now measures the residual, not the workload.**
  Repair attempt records carried the finding counts computed *before* the
  repair render, so the document-level metric reported what repairs were
  asked to fix rather than what survived them. Records now carry the
  post-render residual.
- **Cross-document merge vetoes cover the whole cluster.** `merged_clusters`
  was keyed by final URI while one canonical spanning two documents mints one
  final URI per doc base, so the validation gate's veto dissolved only the
  flagged document's half of a bad merge. Clusters are now grouped by
  canonical identity, with every final URI keying the full member set.
- **JSON-LD prompt serialization no longer mints phantom prefixes from
  literal text.** `@context` filtering matched `prefix:`-shaped heads in
  *every* string value, so a plain literal like `"time: 10 minutes"` kept the
  unused `time:` vocabulary in front of the model. Prefixes are now collected
  only from IRI positions (`@id`, `@type`, predicate keys).
- **Lexical triggers match non-ASCII symbols.** `µm`, `μm`, `°C`, `%`, `Å`,
  `Ω`, `‰`, `Δ` tokenized to their ASCII tails (or nothing) and were
  unreachable — among the most common tokens in SI-reporting prose. Symbol
  characters are now valid token starts, and short non-ASCII triggers
  qualify for the substring lane.
- **Substring trigger scan respects token boundaries.** `mA/cm²` in prose
  fired the `A/cm²`, `W/cm²`, and `/cm` triggers (4 spurious seeds of 19 on
  one measured passage), consuming the additive trigger budget and
  mis-attributing units. Occurrences now count only at non-alphanumeric
  boundaries.
- **Schema-closure floor score no longer inverts at non-positive floors.**
  Closure additions were ranked at `min(seed score) * 0.5`, which ties with
  the weakest seed at `0.0` and *outranks* it for negative floors; closure
  now always scores strictly below every real seed.
- **Closed-range suggestions match case exactly.** The suggester's
  case-insensitive fallback could propose `unit:M` for `"m"`, contradicting
  the facts prompt's character-for-character rule.
- **Retrieval recall harness attributes versioned-header vocabularies.**
  `owner_of` was containment-only, so a vocabulary whose `owl:Ontology`
  header is not a prefix of its term IRIs (QUDT 2.1: header
  `…/2.1/vocab/unit`, terms `…/vocab/unit/`) read as "contributed nothing";
  it now falls back to subject membership like production, and unowned
  expected IRIs are reported as `(unattributed)` instead of being silently
  dropped. Production `patch_retriever` needed no change — the PLANNING
  entry's claim about the production path was wrong.
- **The deterministic facts repair no longer fires on advisory findings.**
  `_run_deterministic_repair` broke out of its loop on `if not findings`, but
  `findings` includes `NUMERIC_COVERAGE`, which is explicitly non-mandatory and
  fires whenever any number in the chunk is absent from the graph -- nearly
  always for numeric prose. At the defaults (`MAX_VISITS=1`,
  `FACTS_REPAIR_VISITS=1`) essentially every unit paid one extra
  `render_facts_update` LLM call. The loop now gates on
  `any(finding.mandatory)`; advisory findings still ride along in the prompt
  when a repair does run.
- **`graph.hash()` no longer runs inside the async parallel fan-out.** It was
  called 2-3x per unit from `_record_facts_attempt` to populate a `graph_hash`
  field that nothing read. `RDFGraph.hash` copies the graph, canonicalizes
  every literal, and runs pure-Python URDNA2015 -- inside `facts_loop`, an
  `async def` with no `to_thread`, so it blocked the event loop and serialized
  the `PARALLEL_WORKERS` fan-out.
- **`LLM_MAX_INFLIGHT` was a silent no-op.** The field sits in `LLMConfig`
  under `env_prefix="LLM_"`, so the real variable was `LLM_LLM_MAX_INFLIGHT`
  while the docs and `.env.example` documented `LLM_MAX_INFLIGHT`. Users
  setting the documented rate-limit control silently got the default 16. Now
  aliased via `AliasChoices`; both spellings work.
- **`DEGENERATE_COREFERENCE` could never drive the un-merge repair.** The
  finding reports the *pointing* node as `subject` and the over-merged endpoint
  in `values`, but `_vetoes_from_findings` read only `subject` -- which for a
  collapsed range is not a merged cluster at all, so the veto set came back
  empty and the loop broke immediately on an error-severity finding. Vetoes are
  now derived from IRI-valued `values` as well. The same fix covers the
  IRI-object branch of `SUSPECT_MULTI_VALUE`.
- **A repair pass that does not reduce errors is now reverted.** The gate
  assigned the re-aggregated graph unconditionally. Since a veto dissolves an
  entire cluster, a pass that failed to help still destroyed real coreference.
  The pass is kept only if the error count strictly decreases, and
  `facts_merge_repairs_rejected` records rejections.
- **`owl:sameAs` is excluded from the multi-value checks.** The rewriter emits
  `canonical owl:sameAs original` per remapped entity, so the predicate carries
  1 object for an unmerged entity and N-1 for a cluster. Left in, it profiled
  as dominantly single-valued and turned every large cluster into an error that
  drove the repair to dissolve a legitimate merge. Latent behind
  `blocked_sameas_namespaces` for single-document runs; live when
  re-aggregating already-merged graphs.
- **The validation gate's output reaches the batch path.**
  `facts_validation_findings` was written but read by nothing, and
  `_merge_workflow_state_into_agent_state` dropped both it and
  `retrieval_metrics` before serialization, making the gate log-only outside
  the HTTP API.
- **`/process_unit` runs the invariant gate.** The unit pipeline aggregated and
  serialized without any functional-violation, coreference, or SHACL check.
  Detection only -- the un-merge repair re-aggregates retained units against
  each other, which has no meaning for a single unit.
- **`FACTS_FUNCTIONAL_MIN_SINGLE_SUPPORT` is reachable.** The parameter was
  exposed on `validate_aggregated_facts` but never passed by its only caller,
  pinning it at a hardcoded 3 while every sibling knob was configurable.
- **SHACL no longer degrades silently.** With `FACTS_SHAPES_DIR` set but the
  `shacl` extra absent, the gate emitted `logger.debug` and returned no
  findings -- invisible at default log level and indistinguishable from
  conformance. A nonexistent or empty shapes directory was equally silent. All
  three now warn.
- **A transient catalog fetch error is no longer memoized as a permanent
  miss.** `_small_module_cache` caches `None` as a negative result, so one
  failed fetch stripped the small-module closure from every subsequent unit for
  the life of the process. The exception path no longer writes the cache.
- **A failed repair render is recorded instead of erased.** `clear_failure()`
  reset the unit to `Status.SUCCESS`, leaving nothing downstream able to
  distinguish "repair converged" from "repair crashed". The pre-repair graph is
  still kept (correct), but the failure is now on the attempt log.
- **`NON_CATALOG_VOCABULARY` and `UNKNOWN_TERM` agree on what "in the catalog"
  means.** The former used `ontology_graph.subjects()` while the latter used
  all three triple positions, so a term supplied only as an `rdfs:range` object
  was simultaneously in-catalog and non-catalog -- spurious warnings on exactly
  the terms the domain/range schema closure exists to admit.
- **Bibliography routing is logged.** In the default `citations_only` mode
  nothing was logged when the detector fired, so a misclassified Methods
  section was silently reduced to citation-metadata extraction. Every routing
  decision now leaves a trace.
- **Silent truncations now warn.** The sibling merge guard truncated
  co-object groups at 32 by IRI byte order, quietly leaving the tail
  unguarded; numeric-coverage mentions truncated at 30 without a word.
- **Config is read directly instead of via `getattr` with duplicated
  defaults.** `getattr(tools, "property_alias_min_ratio", 0.85)` and
  `getattr(atomic, "facts_repair_visits", 1)` would have silently reverted a
  tuned value to a hardcoded literal if either attribute were renamed.
- **Service-dependent tenancy tests skip when Qdrant is unreachable** instead
  of failing with a bare connection error.
- **`uv.lock` no longer fights pre-commit.** The `pretty-format-toml` and
  `toml-sort` hooks reformatted a file `uv lock` regenerates in its own format,
  turning CI red on every dependency change.
- **`ruff` configuration matches CI.** `E501`/`E722`/`W293` were ignored only
  in the pre-commit hook's arguments, so the documented `uv run ruff check .`
  reported 282 findings that CI ignored. The ignores now live in
  `pyproject.toml` and the hook passes `--fix` alone.
- **MkDocs build warnings** — workflow guide links now point at
  `docs/assets/graph.mmd` and the generated `stategraph/atomic` reference
  page; `docs/gen_pages.py` emits package pages at `reference/<pkg>.md`
  (not `reference/<pkg>/__init__.md`) so config/onto identifiers are not
  registered twice; doctest examples in `Ontology.to_lineage_node` no
  longer emit Markdown shortcut-link false positives for `'def456'`.
- **Atom `entity_role` is derived from property *declaration*, not from
  incidental predicate-position usage** (surface-form contract `sf4` → `sf5`,
  so existing vector collections must be reindexed). A TBox-only module never
  uses its own properties as predicates — it asserts them as subjects
  (`ex:hasResult a owl:ObjectProperty ; rdfs:domain … ; rdfs:range …`) — so on
  an 8-module materials-science catalog **85 of 89 declared properties were
  indexed as resources**, and only the one module shipping instance data
  scored at all. Resource-role property atoms get an empty neighborhood
  representation, and the empty string is what went into the vector: the
  neighborhood channel, the one meant to carry relational context, was
  structurally blind to properties (2 property hits across 140 slots).
- **The small-module closure fired only when the in-memory ontology manager
  was populated**, which it is not in any deployment that keeps its catalog in
  a triple store and fetches per query — the normal server configuration. It
  now falls back to `afetch_ontologies_by_iri`, with a per-tenancy cache so
  the extra catalog reads stay O(modules) rather than O(units × modules).
- **The small-module closure duplicated blank-node structures.** The induced
  subgraph had usually already taken a slice of the module; merging the whole
  module on top gave every blank node (OWL restriction shells, SHACL prefix
  declarations) a second identity. The slice is now dropped first, making the
  closure idempotent.
- **Prompt JSON-LD `@context` lists only referenced prefixes.** Every binding
  on the graph reached the namespace manager, so ~15 rdflib built-ins
  (`brick`, `csvw`, `dcat`, `odrl`, `qb`, `void`, `wgs`, …) were put in front
  of the model that nothing in the payload used.
- **Retrieval defaults retuned against measured snapshot adequacy.**
  `VECTOR_STORE_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES` 550 → 1200 (it, not the
  atom cap, is the binding constraint — every seed-side knob measured flat
  while it was low), `ONTOLOGY_PATCH_PER_ONTOLOGY_ATOM_FLOOR` 0 → 2,
  `ONTOLOGY_PATCH_SMALL_MODULE_CLOSURE_MAX_TRIPLES` 0 → 300. Together with the
  role fix and the schema closure this took needed-term recall from 0/11 to
  11/11 and declared-property coverage from 21% to 82% on the tuning passage,
  at roughly 2.7× the snapshot size — that is, a materially larger ontology
  context, and a correspondingly larger prompt, for every unit. Measured ranges
  per knob are tabulated in `docs/user_guide/configuration.md`. These values
  were fitted on a single corpus and should be re-swept for other catalogs.
- **Aggregator merge guards: distinct quantities/samples/conditions no longer
  collapse.** `EmbeddingBasedAggregator` now blocks identity merges between
  entities that provably describe distinct things: disjoint numeric/temporal
  literal values on one predicate (12.5 vs 96 meV red shifts, 30 vs 230
  μJ/cm² fluences), disjoint IRI objects on a functional-ish predicate —
  `owl:FunctionalProperty` ∪ OWL max-cardinality-1 restriction targets ∪
  empirically single-valued predicates such as `qudt:unit` (10 °C vs 10 kHz),
  co-objects of one subject (range bounds, sample series, grant/author
  lists), and fuzzy lexical merging of literal-bearing entities (only exact
  label/normal-form matches qualify; the label token-Jaccard tier tightened
  0.2 → 0.5). This addresses two observed over-merge failures: a single entity
  that had absorbed fourteen different numeric measurements, and a measured
  quality factor that had acquired a length unit by being merged with an
  unrelated entity. New `AGG_*` tunables; the two extraction outputs are
  vendored as regression fixtures (`test/data/case{4,5}`,
  `test/aggregation/test_case_regression.py`).
- **Deterministic findings are no longer dropped at `MAX_VISITS=1`.** The
  facts loop previously skipped the critic after the final render, so
  quarantined literals and namespace violations were logged and discarded —
  at the default settings the critic never ran at all. A bounded
  deterministic repair loop (`FACTS_REPAIR_VISITS`, default 1) now feeds
  machine-found MANDATORY fixes (quarantined literals with closed-range
  individual suggestions, `example.org`/doc-namespace predicates,
  unresolved catalog near-misses) plus a numeric-coverage checklist
  (source-text numbers missing from the graph, verbatim-unit policy) into
  `render_facts_update`, even when the LLM critic is skipped; findings are
  also injected into the critic prompt when it does run.
- **Parse-time literal and property-alias normalization.** Untyped numeric
  literals are retyped against declared numeric `rdfs:range`s (plain `230`
  next to `"230"^^xsd:decimal`), and near-miss predicates in catalog
  namespaces are rewritten deterministically when unambiguous
  (for example `qudt:value` → `qudt:numericValue`, or a `lowerBound` property
  written without the `has` prefix its catalog declares); ambiguous cases become
  findings with suggestions.
  The facts prompt now mandates verbatim source values and units (no LLM
  unit conversion — canonicalization is downstream code's job) and no longer
  cites the nonexistent `unit:MilliEV` IRI in its example.
- **Vector store initialization bypass during tenancy bootstrapping.** Fixed an ordering issue where `update_tenancy_with_vector_mode` initialized the vector store too early, causing an `EmbeddingContractMismatchError` when launching the server or processing files even when `--wipe-vector-store` was set. The vector store initialization is now properly delayed to the `initialize` method where the `wipe_vector_store` flag is checked and executed first.
- **Serialization of oxigraph-backed graphs holding RDF 1.2 triple terms.**
  `RDFGraph.serialize()` delegates Turtle output to pyoxigraph for oxigraph stores
  because oxrdflib surfaces a `pyoxigraph.Triple` as a plain Python tuple, which
  rdflib's Turtle writer cannot label. The delegation allowlist covered only
  `turtle` / `ttl`, so once `serialize_canonical_turtle()` switched to the lossless
  `ontocast-turtle` flavour every merged facts graph (built by `GraphRewriter` as
  `RDFGraph(store="oxigraph")`) fell through to rdflib and the `Serialize` node died
  with `'tuple' object has no attribute 'n3'`. `ontocast-turtle` is now routed to
  pyoxigraph as well; its writer is value-preserving for floating-point literals, so
  the precision the lossless serializer exists for is not given up.
- **Ontology content hash is now stable across a triple-store round trip.**
  `RDFGraph.hash()` hashed literal *lexical* forms — URDNA2015 canonicalizes blank node
  labels only — while triple stores normalize literals into their value space on insert.
  Measured against pyoxigraph: `"10.0"^^xsd:decimal` comes back as `"10"^^xsd:decimal`,
  and integer subtypes collapse (`"1"^^xsd:nonNegativeInteger`, the datatype OWL 2
  requires on qualified cardinality axioms, becomes `"1"^^xsd:integer`). Six of the eight
  ontologies in the evaluation catalog re-hashed differently after a round trip, so the hash written
  into `dcterms:identifier "hash:…"` and the named graph URI `<iri>#<hash>` disagreed
  with the hash recomputed on read. Consequences, all reproduced: the
  `catalog identity drift` warning fired on every retrieval touching those ontologies;
  `OntologyManager`'s catalog graph cache missed 100% of the time (written under the
  recomputed `versioned_iri`, read under the header's `graph_uri`); and each restart
  wrote a second named graph for the same ontology that still advertised the stale hash.
  Hashing now runs over the RDF value space via `canonical_literal()` (integer family,
  `xsd:decimal`, `xsd:boolean`), applied to a throwaway copy so stored and prompt-facing
  graphs are untouched. Hashes are also now backend-independent. The graded relaxation
  ladder in `select_relevant_ontologies` stays as defense in depth.
  **Upgrade note:** every ontology's hash changes, so `versioned_iri` changes. Existing
  Fuseki catalogs keep their old named graphs alongside the new ones (a one-time clean is
  simplest), and because `atom_id` folds `ontology_hash` the vector index must be rebuilt
  — run once with `--wipe-vector-store` (or `VECTOR_STORE_WIPE_ON_INIT=true`).
- **Turtle serialization no longer rounds floating-point literals.** rdflib's Turtle
  writer renders `xsd:double`/`xsd:float`/`xsd:decimal` through its plain-literal
  shorthand (`"%e" % float(x)`), truncating to 7 significant digits —
  `1.602176634e-22` was written out as `1.602177e-22`. This reached the triple store,
  the TTL returned by `/process`, and the graph rendered into the LLM prompt, so the
  model was shown rounded physical constants. `serialize_canonical_turtle()` now uses a
  serializer that emits explicit typed literals for those datatypes.
- **Ontology version/hash filters no longer silently empty the prompt context.** Graph
  hashes are not stable under serialization round-trips (URDNA2015 canonicalizes blank
  nodes, not literal lexical forms, so float-bearing vocabularies re-hash on every
  Turtle round trip). The exact-hash requirement in `select_relevant_ontologies` then
  dropped whole ontologies whose atoms were retrieved in another process — seeds became
  silent no-ops with no log line. Filters now relax per IRI (exact → same-version → any
  catalog entry) with a warning. Measured on the materials-science recall corpus in the
  ablation arm where the catalog is diluted with a large external unit vocabulary, this
  took snapshot term recall from 49% to 72% and flipped the graph-stage net from −4 to
  +19 terms; Text2KGBench stayed at 99% snapshot term recall.
- **Snapshot prefixes now reflect content.** The induced subgraph binds one prefix per
  namespace actually used by its triples (canonical vocabulary names preferred over
  stem-derived aliases) instead of eagerly binding every merged ontology's prefix map —
  prompts no longer advertise namespaces with zero visible terms. Graphs fetched from a
  triple store (which strips author `@prefix` bindings) rebuild implicit stem prefixes on
  fetch, on both the merge and candidate-pushdown context paths; an ontology's own
  namespace binds as its plain stem (`dom`, not `dom_dom`).
- `_SEED_DESCRIPTION_PREDICATES` is now an ordered tuple: it is iterated at the
  triple-budget cut point, and per-process set-order salting made snapshot admission
  nondeterministic across runs.
- Recall-harness `ONTOCAST_RECALL_SKIP_INDEX=1` reuse mode wrote nothing to the run's
  in-memory triple store, so every induced subgraph in a reused-index arm was silently
  empty (snapshot recall 0%). The skip path now serializes ontologies to the triple
  store while still skipping re-embedding.
- Documented that `ONTOCAST_RECALL_ONTOLOGIES` / `ONTOCAST_RECALL_CASES` apply to the
  Text2KGBench tier only. The module docstring implied they bounded the prebuilt-corpus
  tier as well; `load_corpus` has never consulted either.
- **A multi-language vocabulary was named in an arbitrary language.** Literals are sorted before truncation so atomization is reproducible, but that sort also picks the display name and the surface forms that survive the cap — so a term declaring one `rdfs:label` per language was named by whichever language sorted first alphabetically. QUDT's `unit:DEG_C` (23 labels) came out as the Hungarian "Celsius Fok", and three of its five surviving surface forms were translations no English-language query will match. Literals are now ranked by language before value, putting untagged and `en*` forms first; other languages are demoted rather than dropped, so a non-English corpus keeps its aliases. The ordering stays total and deterministic, so reproducibility is unaffected. Changes stored vectors — requires a reindex.
- **A tenant switch left the previous tenant's ontologies in memory.** `ToolBox.update_tenancy_with_vector_mode` retargeted the triple store and vector store but never touched `ontology_manager`, so after a per-request `?tenant=` switch the in-memory catalog still served the previous tenant's ontologies — and its alias-collision ledger still rejected a legitimately distinct ontology that reused an `ontology_id`. The catalog is now reset and reloaded from the retargeted store whenever tenancy actually changes; the first assignment is skipped because it precedes `initialize()`, which populates the catalog itself. Seed TTLs are deliberately not replayed into a different tenant.
- **Retrieval re-read and re-merged the catalog once per content unit.** Even with by-IRI fetching, each unit in the `PARALLEL_WORKERS` fan-out downloaded the selected ontology graphs again and paid a fresh rdflib union over them. Both are now served from `OntologyManager`, so a document pays per ontology rather than per unit. Snapshot content is unchanged: `test_catalog_read_path_matches_direct_store_reads` asserts byte-identical Turtle with and without the new read path, and `test_induced_subgraph_quality.py` pins the builder's output. (The recall harness cannot establish this — approximate nearest-neighbour search is not bit-reproducible across index builds, and its snapshot sizes move by several triples run to run on identical code. Its stable figure, seed TERM recall, is unchanged at 84.0% on Text2KGBench and 100% on the anchor fixtures.)
- Two remaining full-catalog fetches: the synchronous `SPARQLTool.get_induced_subgraph` still called `fetch_ontologies()` while the async path fetched by IRI, and `merge_terminal_ontologies` fetched every ontology to filter one IRI in Python.
- JSON-LD ingest no longer emits RDFLib's `Dataset.default_context` `DeprecationWarning` on every URDNA2015 → N-Quads parse (upstream still uses the deprecated attribute in `nquads.py`; see RDFLib/rdflib#3409).
- LanceDB vector indexes use the unified `create_index(column, config=IvfPq(...))` API instead of the deprecated metric/`vector_column_name` form (removes `DeprecationWarning` on index create).
- LanceDB FTS index creation uses `create_index(..., config=FTS())` instead of the deprecated `create_fts_index`.
- Pytest config lives only in `pyproject.toml` (`env_files`, markers, `addopts`); removed the shadowing `pytest.ini` so `integration`/`slow`/`unit` markers register and `-m "not slow"` works.
- **Ensemble retrieval materialized the entire ontology catalog twice per content unit.** `aretrieve_ensemble` called `afetch_ontologies()` to answer which ontologies the seeds reference through `rdfs:subClassOf` / `domain` / `range`, and `aget_induced_subgraph` then called it again and discarded everything outside `ontology_iris`. Since retrieval runs once per unit in a `PARALLEL_WORKERS` fan-out, a document with `N` units and `C` ontologies paid `2N` graph-listing queries and `2NC` named-graph downloads-and-parses — on Fuseki, over HTTP. Reference expansion is now two small SPARQL SELECTs against a header catalog, and the induced-subgraph step fetches only the ontologies that survive the filter: `~3N` tiny SELECTs and `C` graph fetches. Snapshot content is unchanged.
- **Induced-subgraph BFS kept whichever triples sorted first alphabetically.** When a seed's expansion quota could not hold everything at a BFS level, `sorted(..., key=str)` decided what survived, so a term could arrive labelled but unplaced in the hierarchy, or placed but unnamed. Triples are now admitted by predicate role — label, `rdf:type`, `subClassOf`/`equivalentClass`, `domain`/`range`/`subPropertyOf`, then descriptions — with lexicographic order only as a stable tiebreak.
- **Individual seeds were replaced by their classes instead of joined by them.** `_classify_and_promote_seeds` substituted an individual's `rdf:type` classes for the individual, discarding the node retrieval had actually matched; the facts two-namespace contract expects pre-declared reference individuals to stay reusable, and a snapshot naming only their class cannot support that. The individual is now kept alongside its classes.
- **Long chunks never queried their own tail.** `split_proposition_windows` kept the first `proposition_max_windows` (16) windows, so a 100-sentence chunk contributed queries only from its opening ~32 sentences. Windows are now sampled at an even stride spanning both endpoints, so the whole chunk is represented at the same query budget.
- **`ONTOLOGY_PATCH_MIN_MERGED_MAX_SCORE` is evaluated before the cross-window merge**, against the best per-window fused score. Under max-score merging that is the same number, so behaviour is unchanged; it stops the gate from being silently defeated by merge modes whose scores grow with window count.
- **Relative score floors inverted on negative similarity scores.** `_filter_hits_by_relative_floor` computed `floor = best * score_ratio`, which is correct only while `best > 0`. Qdrant cosine can return negative scores, and there the multiplicative floor lands *above* the best hit (at `best = -0.2`, `ratio = 0.8` the floor is `-0.16`), so the channel returned nothing at all. Worse, the documented "0 disables" default computed a floor of exactly `0.0` and silently dropped every negative-scoring hit. The floor is now expressed as a distance below the best score (`best - (1 - ratio) * |best|`, identical for positive scores) and `score_ratio <= 0` short-circuits to no filtering. All four `ONTOLOGY_PATCH_PER_QUERY_*_SCORE_RATIO` knobs default to 0, so this changes behaviour only for negative-scoring hits and for anyone who had enabled a ratio.
- **BM25 lexical retrieval was scoring without IDF.** Three defects compounded: queries were encoded with fastembed's *document* encoder (`embed`) rather than `query_embed`, so query terms carried term-frequency saturation weights instead of flat ones; the Qdrant sparse vector was created with `modifier=None` and collections carrying a modifier were actively rejected, so the IDF factor `Qdrant/bm25` requires was never applied; and the indexed text was the IRI local name alone, which for opaque identifiers (Wikidata-style `Q36834`) contains no searchable token at all. Queries now use `embed_sparse_query`, collections declare `Modifier.IDF`, and `minimal_representation` indexes the split local name plus `rdfs:label` / `skos:prefLabel` / `dcterms:title` / `skos:altLabel`. **Existing Qdrant collections must be recreated** — a stale collection now fails with `EmbeddingContractMismatchError` pointing at `VECTOR_STORE_WIPE_ON_INIT` / `--wipe-vector-store`.
- **Induced subgraph no longer collapses to a single connected component.** `_prune_disconnected_uri_entities` kept only the seed-bearing component with the most seeds and deleted every other component's subjects, so seeds that legitimately spanned unrelated ontologies were discarded after retrieval had correctly found them. All seed-bearing components are now kept (seedless ones are still dropped), protected seeds survive even with no schema edge, and references *to* a dropped IRI are removed along with its definition so the snapshot never names a term it does not define. Measured on a 20-ontology catalog with seed recall held constant: snapshot recall **41.5% → 85.0%**. Snapshots grow (mean 85 → 169 triples, still well inside the 550-triple budget) and admit more off-ontology material; this trades precision for recall deliberately.

## [0.4.3] - 2026-06-08

### Added
- **LanceDB** embedded vector store (`LANCEDB_ENABLED`, `LANCEDB_DATA_DIR`) as a local alternative to Qdrant.

### Changed
- **BREAKING**: Backend-agnostic vector retrieval settings moved from `QDRANT_*` to `VECTOR_STORE_*` (`top_k`, induced-subgraph limits, proposition windows, fusion weights, dedup mode, embedding batch size). `QDRANT_*` now covers connection/transport only (`URI`, `API_KEY`, collections, gRPC, `VECTOR_SIZE`, `DISTANCE`, `UPSERT_BATCH_SIZE`). Old `QDRANT_TOP_K`, `QDRANT_INDUCED_SUBGRAPH_*`, etc. are **ignored**.
- **BREAKING**: Configure **either** `QDRANT_URI` **or** `LANCEDB_ENABLED=true`, not both.

## [0.4.2] - 2026-06-08

### Added
- **In-memory triple store** — default pyoxigraph backend when Fuseki is not configured.

### Changed
- **BREAKING**: **Neo4j triple store removed** (`NEO4J_*` env vars no longer select a backend). Without Fuseki, OntoCast now uses the in-memory pyoxigraph store automatically.

### Removed
- `Neo4jTripleStoreManager` and `NEO4J_*` configuration.

## [0.4.1] - 2026-06-07

### Added
- **Structured documents** — section label catalog, section-aligned chunk prepare (segment → tag → filter → size), optional summarization (`target_sections`, `summarize_sections`, `section_schema_id`, `document_type_hint`).
- **Facts precision/recall/F1** on `POST /match/evaluate` (`fact_precision`, `fact_recall`, `fact_f1` and counts): relational triples only, excluding schema predicates and triples with ontological class/concept nodes in subject or object position.
- **Anthropic (Claude) and Google (Gemini) LLM providers** via `LLM_PROVIDER=anthropic|google`, with `ClaudeModel` and `GeminiModel` config enums.
- **Token usage reporting** in `BudgetTracker` when providers return `usage_metadata` on LLM responses (character counts remain the universal fallback).
- **LLM disk cache controls** on `LLMConfig`: `LLM_CACHE_ENABLED` (default on), `LLM_CACHE_READ_ONLY`, and in-memory plus on-disk stats via `LLMTool.get_cache_stats()`; `GET /info` exposes `llm_cache`.
- **Global LLM in-flight limit** (`LLM_MAX_INFLIGHT`, default 16) — shared semaphore caps concurrent provider requests across parallel unit workers.
- **Optional process concurrency cap** (`MAX_CONCURRENT_PROCESSES`) — limits simultaneous `/process` and `/process_unit` handlers (additional requests wait for a slot).
- **OpenAI Batch API helpers** (`ontocast.tool.llm_batch`) to export chat batch JSONL and import completed results into the LLM disk cache for offline benchmark pre-warming.
- **`BudgetTracker.cache_hits`** — disk-cache hits count toward character totals but not `calls_count`; included in budget summaries when non-zero.

### Changed
- **JSON-LD reinforced as internal exchange format** — compact JSON-LD (`@context` + `@graph`) when `LLM_GRAPH_FORMAT=jsonld`; prompt context, graph format instructions, and schema bindings share one format profile while runtime models stay canonical.
- **Section pipeline layout** — span detection and LLM backfill under `ontocast.tool.chunk`; section-label YAML in `ontocast.config.section_labels`.
- **Chunk prepare** — coalescing, section tagging, allowlist filtering, and size normalization in one pipeline inside the Chunk node.
- **LLM caching path** — `complete`, `extract`, `__call__`, and `acall` share one `_invoke_cached` implementation with consistent cache keys, optional disable/read-only modes, and provider calls gated by the global in-flight semaphore.
- **Facts extraction prompts** (`facts_guidelines.py`): clearer two-namespace contract — domain ontology is read-only schema plus optional **reference individuals**; all text-derived occurrences use `cd:` with `lowercase_snake_case` local names.

### Fixed
- **Entity alignment** (`EntityAligner`): identical `URIRef` across graphs always form a compatibility edge (score 1.0).
- **Match / evaluate API** (`match_models`, `triple_evaluator`, `match_common`): entity fields stay `URIRef` through Pydantic validation; triple projection and entity precision/recall use set-based unmatched counts.

### Documentation
- Structured documents, facts two-namespace model, entity alignment, LLM cache, and evaluate semantics (`concepts.md`, `workflow.md`, `api.md`, `configuration.md`, `aggregation.md`, `llm_caching.md`, `user_instructions.md`).

## [0.4.0] - 2026-05-26

### Added
- **Parallel map/reduce pipeline** for document processing: per-unit ontology and facts loops run concurrently with configurable `PARALLEL_WORKERS`, retry budgets (`PARALLEL_ONTOLOGY_RETRIES`, `PARALLEL_FACTS_RETRIES`), and a dedicated `/process_unit` endpoint for single-unit runs.
- **Robust semantic disambiguation across chunks**: embedding- and symbolic-aware entity alignment during aggregation (`EntityAligner`, connected-component clustering, `skos:altName` handling) with improved cross-unit identity resolution.
- **RDF 1.2 provenance support**: quoted-triple / reification syntax via `pyoxigraph`; provenance and alignment triples are split into a side artifact during ontology normalization; optional `strip_provenance` on `/process` and `/process_unit` omits reification scaffolding from API Turtle output.
- **Enhanced ontology update consolidation**: normalize → consolidate → structural check → consistency critic pipeline replaces the legacy sublimation stage; optional post-normalization consolidation pass via `ENABLE_ONTOLOGY_CONSOLIDATION`.
- **JSON-LD as LLM wire format**: `LLM_GRAPH_FORMAT=jsonld` emits compact JSON-LD (`@context` + `@graph`) for graph payloads while keeping canonical domain models (`GraphUpdate`, critique reports, etc.) at runtime; Turtle remains the default.
- Per-unit **ontology catalog selection** (`select_ontology_catalog`) with optional `ontology_selection_user_instruction`.
- **Ontology context modes**: `selected_single_ontology`, `selected_vector_search_ontology` (Qdrant stitched ensemble), and `fixed_single_ontology` (`ONTOLOGY_CONTEXT_FIXED_ONTOLOGY_ID`).
- **Qdrant vector retrieval** with dual-vector + BM25 hybrid fusion, patch-retrieval scoring/MMR caps (`ONTOLOGY_PATCH_*`), and induced-subgraph triple budgets (`VECTOR_STORE_INDUCED_SUBGRAPH_*` since 0.4.3; was `QDRANT_INDUCED_SUBGRAPH_*` in 0.4.0–0.4.2).
- **Embedding configuration** surface (`EMBEDDING_*`) and embedding-ready representation contracts for atomizer/retrieval pipelines.
- **Tenancy-aware storage**: `tenant` / `project` request parameters partition Fuseki datasets and Qdrant collections (`{tenant}--{project}--facts|ontologies`); defaults derive from built-in `ontocast` / `test`.
- REST **ontology management** routes: `POST/PUT/DELETE /ontologies` for catalog upload, replace, and delete.
- **Graph matching API**: `POST /match/entities`, `POST /match/derive-matches`, and `POST /match/evaluate` for entity alignment and triple/entity precision-recall evaluation.
- `match-dirs` standalone CLI client for batch benchmark evaluation against the match endpoints.

### Changed
- **BREAKING**: Ontology post-render processing now uses `normalize_ontology_units()` instead of `sublimate_ontology()`; provenance is extracted rather than inlined in the working ontology graph.
- **BREAKING**: CLI server module is `ontocast.cli.server` (entry point unchanged: `ontocast`); legacy `serve` module removed.
- Workflow graph restructured around parallel unit rendering, normalization, and optional consolidation before facts extraction.
- Fuseki/Qdrant dataset and collection names default from tenant/project naming when unset (explicit `FUSEKI_DATASET` / `FUSEKI_ONTOLOGIES_DATASET` still supported).
- Default `max_visits_per_node` is now `1` (override via `MAX_VISITS` or per-request `max_visits`).
- Graph format instructions, JSON Schema bindings, and prompt context chapters are driven by a shared format profile (`LLM_GRAPH_FORMAT`).
- Improved IRI policy, ontology access helpers, and atomizer coverage for facts and ontology cores.

### Removed
- `sublimate_ontology` agent stage and module (superseded by normalize + consolidate).
- Top-level `tool/aggregate` module path (aggregation lives under `tool/agg/`).

### Fixed
- GraphUpdate parsing and alignment edge cases across Turtle and JSON-LD encodings.
- Graceful initialization when vector store or optional backends are unavailable.
- Match endpoint robustness and evaluation semantics (label triples excluded from triple metrics).

### Documentation
- User guides updated for 0.4.0 pipeline (workflow, API, tenancy, ontology context, aggregation).
- API reference pages are generated at build time via `docs/gen_pages.py` (stale committed stubs removed).
- Workflow diagrams: `docs/assets/graph.png` (TB), `graph.lr.png` (LR); regenerate with `uv run plot-graph`.
- Configuration defaults aligned with `config.py` and `.env.example`.

## [0.3.0] - 2026-03-10

### Added
- `updated_at` timestamp field in Ontology properties for tracking last update time.
- Automatic semantic versioning with intelligent MAJOR/MINOR/PATCH increment analysis.
- Version analysis based on ontology changes (classes, properties, and instances).
- Hash-based versioning with parent hashes for git-style lineage tracking.
- `mark_as_updated()` in Ontology for version/timestamp management.
- `sync_properties_to_graph()` to persist `version` and `updated_at` in RDF.
- `versioned_iri` support for storing multiple ontology versions in triple stores.
- URL encoding for versioned IRIs in Fuseki to preserve `#` in named graph URIs.
- Multi-version ontology storage in Fuseki using separate named graphs.
- Automatic ontology synchronization from filesystem to triple store during initialization.
- `render_mode` processing options: `ontology`, `facts`, `ontology_and_facts`.
- Dedicated `serialize` workflow node; separated aggregation and serialization stages.
- API support for `render_mode` as a query parameter.
- **GraphUpdate** system with structured SPARQL insert/delete operations.
- `GraphUpdate`/`TripleOp` models for incremental graph modifications.
- `render_ontology_update()` and `render_facts_update()` GraphUpdate-based rendering.
- Automatic SPARQL generation from GraphUpdate operations.
- Budget tracking integrated in `AgentState`, including ontology/facts generation metrics.
- End-of-run budget summary reporting.
- Dependency-injected budget tracking for LLM calls.
- Shared caching architecture with a single `Cacher` instance and `ToolCacher` wrapper.
- `ONTOCAST_CACHE_DIR` environment variable for cache location.
- `serialize()` as a primary triple-manager interface for `Ontology` and `RDFGraph` objects.
- `ONTOLOGY_MAX_TRIPLES` guardrail to prevent unbounded ontology growth.
- Limit checks in `render_updated_graph()` and `sublimate_ontology()`.
- Parallel unit/chunk processing with configurable worker concurrency and retry behavior.
- More robust entity/property disambiguation across units/chunks during aggregation.
- Optional ontology consolidation switch via `ENABLE_ONTOLOGY_CONSOLIDATION`.
- Aggregation configuration via `AGG_EMBEDDING_MODEL` and `AGG_SIMILARITY_THRESHOLD`.
- Web grounding configuration surface (`WEB_SEARCH_*`) with planner, retry, evidence-budget, and domain filtering controls.
- `FUSEKI_ONTOLOGIES_DATASET` for separate ontology dataset configuration.

### Changed
- **BREAKING**: `serialize()` is now the primary interface for storing data in triple stores.
- **BREAKING**: `serialize()` now accepts `Ontology | RDFGraph` objects instead of raw `Graph` objects.
- **BREAKING**: `serialize_graph()` signature now uses `**kwargs` for backend-specific parameters.
- All triple store managers now implement both `serialize()` and `serialize_graph()`.
- **BREAKING**: Environment variables now use `ONTOCAST_` prefix:
  - `WORKING_DIRECTORY` → `ONTOCAST_WORKING_DIRECTORY`
  - `ONTOLOGY_DIRECTORY` → `ONTOCAST_ONTOLOGY_DIRECTORY`
  - `LLM_CACHE_DIR` → `ONTOCAST_CACHE_DIR`
- **BREAKING**: Ontology and facts rendering now use GraphUpdate/SPARQL operations instead of full TTL generation.
- LLM output now uses structured `GraphUpdate` + `TripleOp`, reducing token usage.
- Ontology version increments now derive from detected ontology diffs.
- Version updates now happen once at end of processing (`serialize`).
- LLM tool budget tracking refactored to dependency injection.
- Global `LLMBudgetTracker` replaced by AgentState-contained tracker.
- Agent functions updated to use injection-based budget plumbing.
- Server recursion control renamed to `BASE_RECURSION_LIMIT` (instead of `RECURSION_LIMIT`).
- `MAX_VISITS` remains supported as alias for `max_visits_per_node`.
- Default `ONTOLOGY_MAX_TRIPLES` increased to `50000`.
- Docs updated for new configuration sections and defaults (`Server`, `Aggregation`, and `Web Search`).

### Removed
- Global budget tracker state management.
- Manual budget tracker update calls inside agent functions.
- `set_budget_tracker()` and `get_budget_tracker()` functions.

## [0.1.7] - 2025-10

### Added
- Automatic LLM response caching for improved performance and cost reduction
- Platform-aware default cache directory selection
- Transparent caching with no configuration required

- Environment variable `SKIP_ONTOLOGY_DEVELOPMENT` to skip ontology critique step
- Environment variable `LLM_API_KEY` for LLM authentication (replaces `OPENAI_API_KEY`)
- Environment variable `MAX_VISITS` for controlling workflow behavior
- Environment variable `WORKING_DIRECTORY` for specifying working directory
- Environment variable `ONTOLOGY_DIRECTORY` for specifying ontology files
- Hierarchical configuration system with environment variable support
- Support for `.env` file configuration
- Python 3.12 type hint support (`str | None` syntax)
- `pathlib.Path` support for directory configurations
- Improved RDF graph operations with proper prefix binding

### Changed
- `OPENAI_API_KEY` environment variable renamed to `LLM_API_KEY`
- Configuration system refactored to use dependency injection
- `ToolBox` now accepts configuration objects directly
- `LLMTool` now accepts configuration objects directly
- Type annotations updated to Python 3.12 standards
- Path handling updated to use `pathlib.Path` objects
- Triple store configuration moved to environment variables

### Fixed
- RDF graph prefix binding issues
- Configuration validation errors
- Triple store initialization errors
- API key handling in LLM configuration
- Type annotation compatibility issues

### Removed
- Global configuration variable
- Support for `OPENAI_API_KEY` environment variable
- Individual parameter passing in tool initialization

### Security
- API keys now handled with secure string types
- Configuration validation prevents data exposure

## [0.1.5] - 2025-01-XX

### Added
- Automatic LLM response caching for improved performance and cost reduction
- Platform-aware default cache directory selection (avoids /tmp)
- Transparent caching with no configuration required

- Version bump to 0.1.5
- Various stability improvements

---

## Migration Guide

### Upgrading to 0.4.3

**Vector store env vars** — retrieval/indexing settings are backend-agnostic; rename `QDRANT_` → `VECTOR_STORE_` for:

| Old (ignored) | New |
|---------------|-----|
| `QDRANT_TOP_K` | `VECTOR_STORE_TOP_K` |
| `QDRANT_INDUCED_SUBGRAPH_DEPTH` | `VECTOR_STORE_INDUCED_SUBGRAPH_DEPTH` |
| `QDRANT_INDUCED_SUBGRAPH_HUB_SEED_COUNT` | `VECTOR_STORE_INDUCED_SUBGRAPH_HUB_SEED_COUNT` |
| `QDRANT_INDUCED_SUBGRAPH_ANCESTOR_CLOSURE_DEPTH` | `VECTOR_STORE_INDUCED_SUBGRAPH_ANCESTOR_CLOSURE_DEPTH` |
| `QDRANT_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES` | `VECTOR_STORE_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES` |
| `QDRANT_INDUCED_SUBGRAPH_ESTIMATED_TRIPLES_PER_QUERY` | `VECTOR_STORE_INDUCED_SUBGRAPH_ESTIMATED_TRIPLES_PER_QUERY` |
| `QDRANT_PROPOSITION_*` | `VECTOR_STORE_PROPOSITION_*` |
| `QDRANT_FUSION_*` | `VECTOR_STORE_FUSION_*` |
| `QDRANT_DEDUP_*` | `VECTOR_STORE_DEDUP_*` |
| `QDRANT_EMBEDDING_BATCH_SIZE` | `VECTOR_STORE_EMBEDDING_BATCH_SIZE` |
| `QDRANT_CONSISTENCY_CRITIC_SIMILARITY_THRESHOLD` | `VECTOR_STORE_CONSISTENCY_CRITIC_SIMILARITY_THRESHOLD` |

**Unchanged under `QDRANT_`:** `URI`, `API_KEY`, `ONTOLOGY_COLLECTION`, `FACTS_COLLECTION`, `GRPC_PORT`, `USE_GRPC`, `VECTOR_SIZE`, `DISTANCE`, `UPSERT_BATCH_SIZE`.

**LanceDB (optional):** `LANCEDB_ENABLED=true` and `LANCEDB_DATA_DIR=~/.lancedb_data` (`uv sync --extra lancedb`). Do not set `QDRANT_URI` at the same time.

### Upgrading to 0.4.2

**Triple store:**

```bash
# Old — Neo4j backend (removed)
NEO4J_URI=bolt://localhost:7687
NEO4J_AUTH=neo4j/password

# New — omit Fuseki for zero-config dev, or use Fuseki for persistence
FUSEKI_URI=http://localhost:3032
FUSEKI_AUTH=admin:password
# (no triple-store env vars → in-memory pyoxigraph)
```

### Upgrading to 0.4.0

**Environment variables:**

```bash
# Old (ignored in 0.4.0)
RECURSION_LIMIT=1000

# New
BASE_RECURSION_LIMIT=1000
```

**Defaults changed:**

| Setting | 0.3.x docs / `.env.example` | 0.4.0 code default |
|---------|----------------------------|-------------------|
| `MAX_VISITS` | often documented as `3` | `1` |
| `ONTOLOGY_MAX_TRIPLES` | sometimes `10000` | `50000` |
| Fuseki datasets | explicit `FUSEKI_DATASET` | derive `ontocast--test--facts` when unset |

**Removed APIs:**

- `ontocast.agent.sublimate_ontology` — use `normalize_ontology_units()` and optional consolidation instead.
- `ontocast.cli.serve` — server is `ontocast.cli.server` (CLI command `ontocast` unchanged).

**New request parameters:**

- `tenant`, `project` — partition Fuseki/Qdrant (query string on `/process`, `/ontologies`, etc.)
- `strip_provenance` — omit reification from API Turtle output
- `ontology_context_mode`, `ontology_context_fixed_ontology_id` — per-request ontology context

See [docs/user_guide/](docs/user_guide/) for full guides.

### Upgrading from 0.1.x / 0.3.x (general)
```bash
# Old
OPENAI_API_KEY=your_key_here

# New  
LLM_API_KEY=your_key_here
```

### Configuration Usage

```python
# Old way (no longer supported)
from ontocast.config import config

llm_provider = config.llm_config.provider

# New way
from ontocast.config import Config

config = Config()
llm_provider = config.tool_config.llm_config.provider
```

### ToolBox Initialization
```python
# Old way (no longer supported)
tools = ToolBox(
    llm_provider="openai",
    model_name="gpt-4",
    # ... many individual parameters
)

# New way
tools = ToolBox(config)
```

### CLI Parameters

### LLM Caching
```python
# Caching is now automatic - no configuration needed
```

```bash
# Skip ontology critique step
ontocast --skip-ontology-critique

# Or set environment variable
export SKIP_ONTOLOGY_DEVELOPMENT=true
ontocast --env-path .env
```
