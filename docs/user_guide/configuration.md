# Configuration System

OntoCast configuration is powered by Pydantic `BaseSettings` and is loaded from environment variables (typically via `.env`).

## Overview

- Typed config sections with defaults
- Environment variable parsing (including lists and booleans)
- Validation for provider/model compatibility
- Unified `Config` object shared across tools and server

## Configuration Shape

```python
Config
├── tool_config: ToolConfig
│   ├── llm_config: LLMConfig
│   ├── chunk_config: ChunkConfig
│   ├── converter_config: ConverterConfig
│   ├── path_config: PathConfig
│   ├── fuseki: FusekiConfig
│   ├── domain: DomainConfig
│   ├── web_search: WebSearchConfig
│   ├── aggregation: AggregationConfig
│   ├── embedding: EmbeddingConfig
│   ├── patch_retrieval: PatchRetrievalConfig
│   ├── vector_store: VectorStoreConfig
│   ├── qdrant: QdrantConfig
│   └── lancedb: LanceDBConfig
├── server: ServerConfig
├── logging_level: str | None
└── clean: bool
```

## Environment Variables

### LLM

```bash
LLM_PROVIDER=openai                     # openai | ollama | anthropic | google
LLM_MODEL_NAME=gpt-4o-mini
LLM_TEMPERATURE=0.0
LLM_API_KEY=your_api_key_here           # required for openai, anthropic, google
LLM_BASE_URL=http://localhost:11434     # optional (ollama; anthropic proxy URL)
```

| Provider | Example `LLM_MODEL_NAME` | `LLM_API_KEY` |
|----------|--------------------------|---------------|
| `openai` | `gpt-4o-mini` | Required |
| `ollama` | `llama3.1` | Not used (`LLM_BASE_URL` required) |
| `anthropic` | `claude-sonnet-4-20250514` | Required |
| `google` | `gemini-2.0-flash` | Required |

OntoCast uses `LLM_API_KEY` for all cloud providers (not `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY`).

**Disk cache and provider concurrency** (see [LLM Caching](llm_caching.md)):

```bash
LLM_CACHE_ENABLED=true          # read/write disk cache (default true)
LLM_CACHE_READ_ONLY=false       # use cache without writing new entries
LLM_MAX_INFLIGHT=16             # max concurrent provider requests (all documents)
```

**Ollama-specific generation controls** (ignored by other providers):

```bash
LLM_THINK=                      # true/false: thinking mode for qwen3, deepseek-r1, ...
                                # unset uses the model default
LLM_NUM_PREDICT=                # max tokens to generate; unset = Ollama default
LLM_NUM_CTX=                    # context window (prompt + output). Ollama defaults to
                                # 2048-4096; raise to 16384+ for large prompts
```

`LLM_THINK=false` guarantees a non-empty `content` response; `true` captures
reasoning separately. Leaving it unset can yield an empty response when a
thinking model spends its whole budget reasoning — raise `LLM_NUM_PREDICT`
alongside it.

```bash
# Anthropic Claude
LLM_PROVIDER=anthropic
LLM_MODEL_NAME=claude-sonnet-4-20250514
LLM_API_KEY=your_anthropic_api_key_here

# Google Gemini
LLM_PROVIDER=google
LLM_MODEL_NAME=gemini-2.0-flash
LLM_API_KEY=your_google_api_key_here
```

### Server

```bash
PORT=8999
BASE_RECURSION_LIMIT=1000
ESTIMATED_CHUNKS=30
MAX_VISITS=1                             # alias for max_visits_per_node
RENDER_MODE=ontology_and_facts           # ontology | facts | ontology_and_facts
LLM_GRAPH_FORMAT=turtle                  # turtle | jsonld
ONTOLOGY_CONTEXT_MODE=selected_single_ontology
#ONTOLOGY_CONTEXT_FIXED_ONTOLOGY_ID=catalog_iri_or_id_or_prefix
ONTOLOGY_MAX_TRIPLES=50000               # empty/unset for unlimited
PARALLEL_WORKERS=4
PARALLEL_FACTS_RETRIES=3
PARALLEL_ONTOLOGY_RETRIES=3
ENABLE_ONTOLOGY_CONSOLIDATION=false
# MAX_CONCURRENT_PROCESSES=4      # optional cap on simultaneous /process handlers
```

### Chunking

```bash
CHUNK_BREAKPOINT_THRESHOLD_TYPE=percentile  # percentile | standard_deviation | interquartile | gradient
CHUNK_BREAKPOINT_THRESHOLD_AMOUNT=95.0
CHUNK_MIN_SIZE=3000
CHUNK_MAX_SIZE=12000
CHUNK_SECTION_TAG_MIN_CHARS=80   # min size for LLM section backfill; smaller hybrid segments coalesce first
CHUNK_BIBLIOGRAPHY_MODE=citations_only  # citations_only | skip | domain_facts
```

`CHUNK_BIBLIOGRAPHY_MODE` routes chunks detected as bibliography/reference
lists (via section label or citation-density heuristics): `citations_only`
(default) extracts bibliographic metadata only (`schema:ScholarlyArticle` +
`schema:citation`, no domain facts from citation titles), `skip` drops the
chunks before extraction, `domain_facts` restores the legacy behavior.

Semantic chunking is configured here. **Section-aligned labels** and filtering are not chunker settings: they run when `/process` or CLI file mode passes `target_sections` and/or `summarize_sections` (see [Structured documents](concepts.md#structured-documents-optional)).

### Docling converter

Use these settings to tune Docling's standard document-conversion pipeline, especially for born-digital publisher PDFs where embedded ligatures can be split into patterns like `di ff usion`.

```bash
CONVERTER_PROFILE=default               # default | born_digital
# CONVERTER_PDF_BACKEND=docling_parse   # docling_parse | pypdfium2
# CONVERTER_DO_OCR=true
# CONVERTER_DO_TABLE_STRUCTURE=true
# CONVERTER_FORCE_BACKEND_TEXT=false
# CONVERTER_TABLE_CELL_MATCHING=true
# CONVERTER_LAYOUT_MODEL=heron          # heron | heron_101 | egret_medium | egret_large | egret_xlarge | v2
# CONVERTER_OCR_ENGINE=auto             # auto | easyocr | rapidocr | tesseract_cli | tesseract
# CONVERTER_OCR_LANG=
# CONVERTER_FORCE_FULL_PAGE_OCR=false
# CONVERTER_OCR_BITMAP_AREA_THRESHOLD=0.05
# CONVERTER_REPAIR_LIGATURE_GAPS=false  # TEMP workaround
```

Recommended preset for publisher PDFs with selectable text:

```bash
CONVERTER_PROFILE=born_digital
```

That preset currently implies:

| Setting | Value |
|---------|-------|
| `CONVERTER_PDF_BACKEND` | `pypdfium2` |
| `CONVERTER_DO_OCR` | `false` |
| `CONVERTER_FORCE_BACKEND_TEXT` | `true` |
| `CONVERTER_REPAIR_LIGATURE_GAPS` | `true` |

Notes:

- `CONVERTER_REPAIR_LIGATURE_GAPS` is a **temporary workaround** in OntoCast for ASCII `fi` / `fl` / `ff` gap patterns that Docling still passes through on some publisher PDFs.
- Prefer `CONVERTER_PROFILE=born_digital` for text-selectable PDFs before trying heavier OCR settings.
- If OCR remains enabled and you pick `rapidocr`, set `CONVERTER_OCR_LANG=english` for English scans; RapidOCR's upstream default language is Chinese.

### Structured documents (per request)

No environment variables. Pass on `POST /process`, multipart form, JSON body, or CLI batch mode:

| Parameter | CLI flag | Description |
|-----------|----------|-------------|
| `target_sections` | `--target-sections` | Comma-separated or JSON list; enables tagging and keeps only these sections |
| `summarize_sections` | `--summarize-sections` | Enables tagging + summarization; `*` or empty = all chunks |
| `summary_max_sentences` | `--summary-max-sentences` | Max sentences per summary (default `5`) |
| `max_visits` | `--max-visits` | Render/critic retry budget per loop (default from `MAX_VISITS`) |
| `section_schema_id` | `--section-schema-id` | Section label schema (`academic`, `financial`, `legal`, …) |
| `document_type_hint` | `--document-type-hint` | Free-text hint to resolve schema when `section_schema_id` is omitted |

```bash
ontocast process --input-path ./papers/ \
  --output-dir ./out \
  --target-sections results,methods \
  --summarize-sections results \
  --summary-max-sentences 5
```

Details: [API Endpoints](api.md#post-process), [Workflow](workflow.md#2-chunking-and-optional-structured-preprocessing).

### Triple Stores

```bash
# Fuseki — dataset names default to ontocast--test--facts / ontocast--test--ontologies
FUSEKI_URI=http://localhost:3030
FUSEKI_AUTH=admin/admin
#FUSEKI_DATASET=custom--project--facts
#FUSEKI_ONTOLOGIES_DATASET=custom--project--ontologies
```

See [Tenancy](tenancy.md) for how tenant/project names relate to dataset, collection, and table names.

### Embeddings

```bash
EMBEDDING_PROVIDER=huggingface          # huggingface | openai | ollama
EMBEDDING_MODEL_NAME=paraphrase-multilingual-MiniLM-L12-v2
# EMBEDDING_API_KEY=
# EMBEDDING_BASE_URL=http://localhost:11434
EMBEDDING_DIMENSION=384
# EMBEDDING_QUERY_PREFIX=
# EMBEDDING_DOCUMENT_PREFIX=
```

**Query/document prefixes.** Asymmetric retrieval models are trained with a distinct
instruction on each side and lose accuracy when query and document are encoded
identically. The default paraphrase model is symmetric and wants neither prefix.

| Model family | `EMBEDDING_QUERY_PREFIX` | `EMBEDDING_DOCUMENT_PREFIX` |
|---|---|---|
| `paraphrase-*` (default) | *(empty)* | *(empty)* |
| `BAAI/bge-*` | `Represent this sentence for searching relevant passages: ` | *(empty)* |
| `intfloat/e5-*` | `query: ` | `passage: ` |

Both prefixes are part of the stored embedding contract, so changing either invalidates
an existing index: rerun with `--wipe-vector-store` (or `VECTOR_STORE_WIPE_ON_INIT=true`).
A mismatch fails loudly with `EmbeddingContractMismatchError` rather than quietly
degrading retrieval.

### Qdrant

```bash
QDRANT_URI=http://localhost:6333
QDRANT_API_KEY=abc123-qwe
QDRANT_GRPC_PORT=6334
QDRANT_USE_GRPC=false
# QDRANT_ONTOLOGY_COLLECTION=ontocast--test--ontologies
# QDRANT_FACTS_COLLECTION=ontocast--test--facts
```

### Vector store (backend-agnostic)

Applies to both Qdrant and LanceDB:

```bash
VECTOR_STORE_TOP_K=20
VECTOR_STORE_INDUCED_SUBGRAPH_DEPTH=2
VECTOR_STORE_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES=1200
VECTOR_STORE_INDUCED_SUBGRAPH_ESTIMATED_TRIPLES_PER_QUERY=24
# VECTOR_STORE_INDUCED_SUBGRAPH_HUB_SEED_COUNT=16
# VECTOR_STORE_INDUCED_SUBGRAPH_ANCESTOR_CLOSURE_DEPTH=3
# VECTOR_STORE_INDUCED_SUBGRAPH_CANDIDATE_PUSHDOWN=false
# VECTOR_STORE_INDUCED_SUBGRAPH_TYPE_PROMOTION_SCORE_FACTOR=1.0
# VECTOR_STORE_INDUCED_SUBGRAPH_SEED_ORDER=score
# VECTOR_STORE_PROPOSITION_WINDOW_SENTENCES=2
# VECTOR_STORE_PROPOSITION_MAX_WINDOWS=16
# VECTOR_STORE_PROPOSITION_RETRIEVAL_ENABLED=true
# VECTOR_STORE_CONSISTENCY_CRITIC_MIN_FUSED_SCORE=0.5
# VECTOR_STORE_FUSION_CORE_WEIGHT=0.7
# VECTOR_STORE_FUSION_NEIGHBORHOOD_WEIGHT=0.15
# VECTOR_STORE_FUSION_BM25_WEIGHT=0.8
# VECTOR_STORE_INDEX_UNDESCRIBED_IRIS=false
# VECTOR_STORE_EMBED_STANDARD_VOCAB_IRIS=false
# VECTOR_STORE_EXTRA_EXCLUDED_NAMESPACE_PREFIXES=
# VECTOR_STORE_DEDUP_MODE=iri
# VECTOR_STORE_EMBEDDING_BATCH_SIZE=64
# VECTOR_STORE_REINDEX_CONCURRENCY=2
# VECTOR_STORE_WIPE_ON_INIT=false
# VECTOR_STORE_PRUNE_ORPHAN_IRIS_ON_INIT=true
```

| Variable | Default | Role |
|----------|---------|------|
| `VECTOR_STORE_TOP_K` | `20` | Vector hits per channel per proposition window |
| `VECTOR_STORE_INDUCED_SUBGRAPH_DEPTH` | `2` | BFS depth for hub seed expansion |
| `VECTOR_STORE_INDUCED_SUBGRAPH_HUB_SEED_COUNT` | `16` | Top seeds that receive full BFS budget (`0` = all seeds) |
| `VECTOR_STORE_INDUCED_SUBGRAPH_ANCESTOR_CLOSURE_DEPTH` | `3` | `rdfs:subClassOf` hops in the schema shell |
| `VECTOR_STORE_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES` | `1200` | Global triple cap returned to the LLM. The binding constraint in practice — every seed-side knob is flat while this is low |
| `VECTOR_STORE_INDUCED_SUBGRAPH_ESTIMATED_TRIPLES_PER_QUERY` | `24` | Per-entity BFS quota hint during retrieval |
| `VECTOR_STORE_INDUCED_SUBGRAPH_CANDIDATE_PUSHDOWN` | `false` | Opt-in SPARQL `CONSTRUCT` neighborhood instead of merging whole ontology graphs (see [Ontology Context](ontology_context.md#candidate-pushdown-opt-in)) |
| `VECTOR_STORE_INDUCED_SUBGRAPH_TYPE_PROMOTION_SCORE_FACTOR` | `1.0` | Fraction of a retrieved seed's score inherited by its promoted `rdf:type` IRIs; the seed always keeps its own score |
| `VECTOR_STORE_INDUCED_SUBGRAPH_SEED_ORDER` | `score` | Seed expansion order under the triple budget: `score` (global relevance) or `ontology_round_robin` (interleave source ontologies) |
| `VECTOR_STORE_INDUCED_SUBGRAPH_SYMBOL_PREDICATES` | trigger predicates | Symbol/notation predicates admitted as seed descriptions between names and glosses (empty list disables) |
| `VECTOR_STORE_PROPOSITION_WINDOW_SENTENCES` | `2` | Sentences per proposition window for multi-query retrieval |
| `VECTOR_STORE_PROPOSITION_MAX_WINDOWS` | `16` | Cap on windows per excerpt; when a chunk has more, windows are sampled at an even stride spanning both endpoints (not “first N only”) |
| `VECTOR_STORE_PROPOSITION_RETRIEVAL_ENABLED` | `true` | Multi-query proposition retrieval for induced-graph mode |
| `VECTOR_STORE_CONSISTENCY_CRITIC_MIN_FUSED_SCORE` | `0.5` | Min weighted reciprocal-rank score for the consistency critic to flag a cross-ontology conflict (not cosine). Renamed from `VECTOR_STORE_CONSISTENCY_CRITIC_SIMILARITY_THRESHOLD` (old default `0.7`) |
| `VECTOR_STORE_FUSION_CORE_WEIGHT` | `0.7` | Dense core-vector weight in rank fusion (weights are normalized, so only ratios matter) |
| `VECTOR_STORE_FUSION_NEIGHBORHOOD_WEIGHT` | `0.15` | Dense neighborhood-vector weight; the neighborhood text describes a term's edges, so it corroborates the core lane more than it adds to it |
| `VECTOR_STORE_FUSION_BM25_WEIGHT` | `0.8` | Sparse BM25 weight. A term whose surface form is a symbol (`meV`, a chemical formula) is often invisible to the dense lanes, so the sparse lane is its only evidence — see [Ontology Context](ontology_context.md#bm25-index-recreate) |
| `VECTOR_STORE_INDEX_UNDESCRIBED_IRIS` | `false` | Atomize IRIs an ontology only *references* (object/predicate position) in addition to ones it describes. Reindex on change |
| `VECTOR_STORE_EMBED_STANDARD_VOCAB_IRIS` | `false` | Atomize RDF/OWL/SKOS/DC/SHACL/schema.org IRIs instead of skipping them. Reindex on change |
| `VECTOR_STORE_EXTRA_EXCLUDED_NAMESPACE_PREFIXES` | *(empty)* | Extra IRI prefixes never atomized from ontology sources, on top of the standard-vocabulary set. Reindex on change |
| `VECTOR_STORE_EMBEDDING_BATCH_SIZE` | `64` | Texts per embedding request during ontology indexing (raise for remote APIs; lower if VRAM-bound) |
| `VECTOR_STORE_REINDEX_CONCURRENCY` | `2` | Max ontologies materialized/reindexed in parallel at `ToolBox.initialize` |
| `VECTOR_STORE_WIPE_ON_INIT` | `false` | Drop the current tenant/project vector partition before recreate+reindex (clean slate; also CLI `--wipe-vector-store`) |
| `VECTOR_STORE_PRUNE_ORPHAN_IRIS_ON_INIT` | `true` | Delete indexed ontology IRIs absent from the synchronized catalog (covers IRI renames without a full wipe) |
| `VECTOR_STORE_LEXICAL_TRIGGER_ENABLED` | `true` | Exact-match lane for notation/symbol tokens in raw chunk text |
| `VECTOR_STORE_LEXICAL_TRIGGER_PREDICATES` | `skos:notation`, `qudt:symbol`, `qudt:ucumCode` | Predicate IRIs whose literals become case-preserved triggers |
| `VECTOR_STORE_LEXICAL_TRIGGER_HEURISTIC_ENABLED` | `true` | Promote code-shaped labels/altLabels when no notation is declared |
| `VECTOR_STORE_LEXICAL_TRIGGER_MIN_LEN` | `2` | Minimum length for heuristic promotion |
| `VECTOR_STORE_LEXICAL_TRIGGER_MAX_LEN` | `24` | Maximum length for heuristic promotion |
| `VECTOR_STORE_LEXICAL_TRIGGER_HEURISTIC_MAX_PER_ENTITY` | `2` | Cap on heuristic triggers per entity |
| `VECTOR_STORE_LEXICAL_TRIGGER_MAX_ATOMS` | `16` | Additive cap on trigger seeds per retrieval call (outside semantic budget) |
| `VECTOR_STORE_LEXICAL_TRIGGER_SCORE` | `0.35` | Score assigned to trigger hits (calibrated against fused rank scores: rank-1 core = 0.583, merged floor = 0.18) |
| `VECTOR_STORE_LEXICAL_TRIGGER_FUSION` | `max_merge` | `max_merge` promotes an already-retrieved atom to `max(semantic, trigger)` score; `append` (legacy) only adds unseen atoms |
| `VECTOR_STORE_QUERY_UNIT_SIGNALS_ENABLED` | `false` | Match number-adjacent tokens ("4-15 days", "200 kV", "0.5 %") case-insensitively and plural-tolerantly against catalog labels/symbols/UCUM codes; matched entities join the snapshot seeds at trigger score, outside the semantic budget |
| `FACTS_OBJECT_PROPERTY_LITERAL_CHECK` | `true` | Quarantine string literals on predicates whose schema range is a class (e.g. `qudt:unit`); surfaced to the facts critic and the deterministic repair loop |
| `FACTS_REPAIR_VISITS` | `1` | Deterministic repair budget per unit: extra update renders fed with machine-found MANDATORY fixes (quarantined literals, unknown/near-miss terms) and numeric-coverage candidates. Applies even at `MAX_VISITS=1`, where the LLM critic never runs |
| `FACTS_PROPERTY_ALIAS_MIN_RATIO` | `0.85` | SequenceMatcher cutoff for deterministic near-miss property rewrites in catalog namespaces (token containment always qualifies, e.g. `qudt:value` → `qudt:numericValue`) |
| `FACTS_MERGE_REPAIR_PASSES` | `1` | Un-merge budget at the post-aggregation `VALIDATE_FACTS` gate: error findings on merged subjects become full-cluster pair vetoes and the facts units are re-aggregated. `0` records findings without repairing |
| `FACTS_SUSPECT_MULTI_VALUE_SEVERITY` | `error` | Severity of SUSPECT_MULTI_VALUE gate findings (multiple distinct numeric values on one predicate, or multiple objects on a dominantly single-valued predicate); only `error` findings drive the un-merge repair |
| `FACTS_SHAPES_DIR` | — | Directory of SHACL shape files for the gate; `sh:NodeShape` triples inlined in the ontology context are picked up automatically. SHACL runs only when `pyshacl` is installed (`uv sync --extra shacl`). Setting this without the extra, or pointing it at a missing/empty directory, logs a **warning** — it never passes silently |
| `FACTS_FUNCTIONAL_MIN_SINGLE_SUPPORT` | `3` | Distinct single-valued subjects a predicate needs before the gate treats it as empirically functional. Below this the evidence is too thin to call a second value a violation |
| `FACTS_QUANTITY_FALLBACK_VOCABULARY` | QUDT | Role → term mapping the facts prompt names as the fallback for bounded/approximate quantities when retrieval supplied no suitable class. Roles: `value_class`, `numeric_value`, `unit`. Override for catalogs modelling quantities with another vocabulary; set to `{}` to forbid the fallback entirely and keep the renderer inside the provided context. Terms in a configured fallback namespace are reported by `NON_CATALOG_VOCABULARY` as a *deliberate* fallback |
| `FACTS_ADDITIONAL_STANDARD_NAMESPACES` | schema.org | Namespaces exempt from `UNKNOWN_TERM` beyond the RDF/OWL substrate and annotation/provenance terms. Only meta-vocabularies are built in; a domain vocabulary shared across catalogs (SOSA/SSN, CSVW, FOAF, Dublin Core profiles) is exempted here. schema.org is the default because the shipped citation vocabulary uses it |
| `CHUNK_CITATION_VOCABULARY` | schema.org | Role → term mapping used by the citation-metadata prompt in `citations_only` mode. Bibliographic entries are not domain content, so unlike the rest of the pipeline these terms are configuration rather than retrieval. Roles: `work_class`, `fallback_class`, `title`, `author`, `author_name`, `date_published`, `venue`, `identifier`, `cites` |

**Migration note:** retrieval knobs formerly named `QDRANT_TOP_K`, `QDRANT_INDUCED_SUBGRAPH_*`, etc. are **ignored**. Use `VECTOR_STORE_*`. `QDRANT_*` covers connection/transport only.

**BM25 / sparse schema:** Qdrant collections must declare IDF sparse vectors and index label-enriched text. A stale collection fails with `EmbeddingContractMismatchError` — recreate via `VECTOR_STORE_WIPE_ON_INIT=true` or `--wipe-vector-store`. The surface-form contract is now `sf4` (ontology sources atomize only IRIs they describe; `sf3` added `lexical_triggers` for the exact-match lane). Every collection built under `sf3` or earlier needs one reindex.

Catalog graphs are served through `OntologyManager` (see [Ontology Catalog](../architecture/ontology_catalog.md)).

### LanceDB (embedded alternative)

Enable when `QDRANT_URI` is unset. Requires the optional extra: `uv sync --extra lancedb`.

```bash
LANCEDB_ENABLED=true
# LANCEDB_DATA_DIR=~/.lancedb_data
# LANCEDB_ONTOLOGY_TABLE=ontologies
# LANCEDB_FACTS_TABLE=facts
```

`QDRANT_URI` and `LANCEDB_ENABLED=true` cannot both be set.

### Backend storage details

Rarely changed; defaults suit both backends.

```bash
# Qdrant collection geometry — must match the embedding model
# QDRANT_VECTOR_SIZE=384
# QDRANT_DISTANCE=Cosine
# QDRANT_UPSERT_BATCH_SIZE=64

# Partition (table/collection) names within the configured backend
# VECTOR_STORE_ONTOLOGY_TABLE=ontologies
# VECTOR_STORE_FACTS_TABLE=facts

# Sparse (BM25) model for the lexical lane
# EMBEDDING_BM25_MODEL_NAME=Qdrant/bm25

# Atom de-duplication identity
# VECTOR_STORE_DEDUP_INCLUDE_VERSION=false   # treat ontology versions as distinct
# VECTOR_STORE_DEDUP_INCLUDE_HASH=false      # treat content hashes as distinct
# VECTOR_STORE_DEDUP_QUERY_HITS_BY_IRI=true  # collapse repeat hits on one IRI

# Prompt/diagnostic shaping
# VECTOR_STORE_MINIMAL_LABEL_LIMIT=...       # cap on label-only atom text
# ONTOLOGY_PATCH_DUMP_ONTOLOGY_RANKS=false   # dump per-ontology retrieval ranks
```

Changing `QDRANT_VECTOR_SIZE` or `EMBEDDING_BM25_MODEL_NAME` invalidates an
existing collection — reindex with `VECTOR_STORE_WIPE_ON_INIT=true`.

Budget behavior:

- `VECTOR_STORE_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES` is the global upper bound returned to the LLM.
- `VECTOR_STORE_INDUCED_SUBGRAPH_ESTIMATED_TRIPLES_PER_QUERY` shapes per-entity allocation during retrieval.

!!! warning "Retrieval defaults are single-corpus fits"

    The shipped retrieval defaults — notably
    `VECTOR_STORE_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES` (raised 550 → 1200),
    `PER_ONTOLOGY_ATOM_FLOOR`, `PER_ROLE_ATOM_FLOOR`, and
    `SCHEMA_CLOSURE_MAX_ENTITIES` — were tuned by a one-axis-at-a-time sweep on
    a single 8-module materials-science catalog. They have **not** been
    re-measured against Text2KGBench or other corpora, and the triple budget
    more than doubles prompt-context cost per unit. Treat them as a starting
    point and re-sweep for your catalog; the per-knob provenance is recorded in
    each field's description in `ontocast/config/settings.py`.

See [Ontology Context](ontology_context.md) for vector-search mode requirements.

### Ontology Patch Retrieval

Post-vector scoring and capping (backend-agnostic; prefix `ONTOLOGY_PATCH_`). Applied after hybrid dense + BM25 retrieval, before induced-subgraph expansion.

**Default path** (simple): max-score IRI dedupe → global score order → window-scaled hard cap. A non-zero `ONTOLOGY_PATCH_PER_ONTOLOGY_SEED_QUOTA` inserts per-ontology round-robin (best-scoring ontologies first) instead of plain score order. Relative floors, hybrid tier merge, merged-score ratio, and MMR are advanced opt-in.

```bash
ONTOLOGY_PATCH_CROSS_QUERY_MERGE_MODE=max_score
ONTOLOGY_PATCH_PER_ONTOLOGY_SEED_QUOTA=0
ONTOLOGY_PATCH_SEEDS_PER_WINDOW=4
ONTOLOGY_PATCH_MAX_ATOMS_BASE=96
ONTOLOGY_PATCH_MAX_ATOMS=96
ONTOLOGY_PATCH_MIN_MERGED_MAX_SCORE=0.18
ONTOLOGY_PATCH_MMR_LAMBDA=1.0
# Advanced (off by default):
# ONTOLOGY_PATCH_PER_QUERY_CORE_SCORE_RATIO=0.0
# ONTOLOGY_PATCH_PER_QUERY_NEIGHBORHOOD_SCORE_RATIO=0.0
# ONTOLOGY_PATCH_PER_QUERY_BM25_SCORE_RATIO=0.0
# ONTOLOGY_PATCH_MERGED_SCORE_RATIO=0.0
# ONTOLOGY_PATCH_CROSS_QUERY_MERGE_MODE=hybrid
# ONTOLOGY_PATCH_MAX_ATOMS_TIER1=12
# ONTOLOGY_PATCH_MIN_ENTITY_SCORE=0.3
```

| Variable | Default | Role |
|----------|---------|------|
| `ONTOLOGY_PATCH_CROSS_QUERY_MERGE_MODE` | `max_score` | Default merge; `sum_score` sums per-window scores; `hybrid` is tier-1 + tier-2 |
| `ONTOLOGY_PATCH_PER_ONTOLOGY_SEED_QUOTA` | `0` | Max seeds per ontology in round-robin fill; `0` (default) means global score order, which measured better on both recall and precision |
| `ONTOLOGY_PATCH_SEEDS_PER_WINDOW` | `4` | Scales effective cap with proposition window count |
| `ONTOLOGY_PATCH_MAX_ATOMS_BASE` | `96` | Floor for effective atom cap; below the candidate pool it silently clips seeds on multi-ontology catalogs |
| `ONTOLOGY_PATCH_MAX_ATOMS` | `96` | Hard cap: `min(max_atoms, max(base, seeds_per_window × n_queries))` (`0` = unlimited) |
| `ONTOLOGY_PATCH_MIN_MERGED_MAX_SCORE` | `0.18` | Empty patch when the best **per-window** fused score is below this (`0` disables); evaluated before cross-window merge |
| `ONTOLOGY_PATCH_MMR_LAMBDA` | `1.0` | `1.0` skips MMR; lower values enable diversity rerank |
| `ONTOLOGY_PATCH_PER_QUERY_CORE_SCORE_RATIO` | `0.0` | Advanced: per-window core relative floor (`0` disables) |
| `ONTOLOGY_PATCH_PER_QUERY_NEIGHBORHOOD_SCORE_RATIO` | `0.0` | Advanced: per-window neighborhood relative floor |
| `ONTOLOGY_PATCH_PER_QUERY_BM25_SCORE_RATIO` | `0.0` | Advanced: per-window BM25 relative floor |
| `ONTOLOGY_PATCH_MIN_CORE_QUERY_BEST_SCORE` | `0.0` | If `> 0`, windows whose top core score is below this contribute no core hits |
| `ONTOLOGY_PATCH_MIN_NEIGHBORHOOD_QUERY_BEST_SCORE` | `0.0` | If `> 0`, windows whose top neighborhood score is below this contribute no neighborhood hits |
| `ONTOLOGY_PATCH_MIN_BM25_QUERY_BEST_SCORE` | `0.0` | If `> 0`, windows whose top BM25 score is below this contribute no BM25 hits |
| `ONTOLOGY_PATCH_MERGED_SCORE_RATIO` | `0.0` | Advanced: drop seeds below `top_score × ratio` (`0` disables) |
| `ONTOLOGY_PATCH_MAX_ATOMS_TIER1` | `12` | Hybrid only: global tier-1 cap (`0` = no cap) |
| `ONTOLOGY_PATCH_MIN_ENTITY_SCORE` | `0.3` | Hybrid only: tier-2 minimum fused score |
| `ONTOLOGY_PATCH_PER_ONTOLOGY_ATOM_FLOOR` | `2` | Reserve pass before the global fill: every contributing ontology is guaranteed `min(floor, its candidates)` seed slots (round-robin). Unlike the quota (a ceiling), the floor protects small modules from starvation at the atom cap. `0` disables |
| `ONTOLOGY_PATCH_SMALL_MODULE_CLOSURE_MAX_TRIPLES` | `300` | Include a source ontology's whole header-stripped graph in the snapshot when it has ≥ 1 admitted atom and at most this many triples (prevents near-miss property improvisation on tiny vocabularies). `0` disables |
| `ONTOLOGY_PATCH_PER_ROLE_ATOM_FLOOR` | `12` | Reserve pass for predicate-role atoms before the global fill. Prose reads as noun phrases, so classes out-score the properties that link them in a shared ranking. `0` disables |
| `ONTOLOGY_PATCH_SCHEMA_CLOSURE_MAX_ENTITIES` | `32` | Cap on terms admitted by `rdfs:domain`/`rdfs:range` closure over the seeds: properties whose domain/range names an admitted class (or an ancestor), plus the domain/range classes of admitted properties. `0` disables |
| `ONTOLOGY_PATCH_SCHEMA_CLOSURE_ANCESTOR_DEPTH` | `2` | How far to walk `rdfs:subClassOf` upward when matching a property's declared domain/range against an admitted class |

#### Retrieval tuning: measured ranges

Measured one axis at a time on an 8-module materials-science catalog against a
four-paragraph measurement-heavy passage, scoring the snapshot the renderer would
receive. "Needed terms" is recall over the 11 range / epistemic-qualifier /
observation-result terms that passage required.

| Variable | Default | Useful range | Effect |
|---|---|---|---|
| `VECTOR_STORE_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES` | `1200` | 1000–1600 | **Gates everything.** At `550` every other knob below is flat, because the snapshot is already pinned at the cap. `550 → 1200` moved declared-property coverage 21% → 36%. Flattens past ~1600 |
| `ONTOLOGY_PATCH_SMALL_MODULE_CLOSURE_MAX_TRIPLES` | `300` | 250–400 | Largest single lever: with the atom floor at 2, `0 → 300` took needed-term recall 3/11 → 11/11 and property coverage 37% → 74%. Set it above the largest module you need whole; going past that adds triples for little gain |
| `ONTOLOGY_PATCH_PER_ONTOLOGY_ATOM_FLOOR` | `2` | 2–4 | Saturates at 3–4 (`0 → 3` moved needed terms 1/11 → 5/11). **The closure above is inert without this** — a module must win ≥ 1 seed before its graph is considered |
| `ONTOLOGY_PATCH_SCHEMA_CLOSURE_MAX_ENTITIES` | `32` | 16–48 | Property coverage 36% → 47% at 32; saturated by 64 |
| `ONTOLOGY_PATCH_PER_ROLE_ATOM_FLOOR` | `12` | 8–16 | Weak on its own (flat to 12, useful only past ~24 in isolation); contributes once the closure is on — property coverage 74% → 82% in the combined setting |
| `ONTOLOGY_PATCH_MAX_ATOMS` / `_BASE` | `96` | 96–192 | Now trades directly against context size once the triple budget is not binding: 96 → 192 gives 36% → 46% coverage for ~40% more triples |
| `VECTOR_STORE_TOP_K` | `20` | — | **Insensitive** (10–40 all within 1 point). Effectively capped by `MAX_ATOMS_BASE`; leave alone |
| `ONTOLOGY_PATCH_MMR_LAMBDA` | `1.0` | — | **Insensitive** (0.5–1.0 identical on this corpus). Leave alone unless you see near-duplicate terms crowding the snapshot |

Combined at the defaults: needed-term recall 11/11, declared-property coverage
82%, at roughly 2.7× the snapshot size of the old settings. That size increase is
the cost of the fix — if prompt budget matters more than recall for your corpus,
lower `MAX_TOTAL_TRIPLES` first and accept the coverage loss.

**Tighter preset** (optional precision knobs for noisy catalogs — see [Ontology Context](ontology_context.md)):

```bash
ONTOLOGY_PATCH_MAX_ATOMS=32
ONTOLOGY_PATCH_MERGED_SCORE_RATIO=0.5
ONTOLOGY_PATCH_MMR_LAMBDA=0.85
VECTOR_STORE_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES=600
```

### Paths and Domain

```bash
CURRENT_DOMAIN=https://example.com
ONTOCAST_WORKING_DIRECTORY=/path/to/working/directory
ONTOCAST_ONTOLOGY_DIRECTORY=/path/to/ontology/files
ONTOCAST_CACHE_DIR=/path/to/cache/directory
```

### Aggregation

```bash
AGG_EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
AGG_SIMILARITY_THRESHOLD=0.80
AGG_CANDIDATE_SIMILARITY_THRESHOLD=0.70
AGG_LEXICAL_LABEL_JACCARD=0.5
AGG_LEXICAL_SEQUENCE_RATIO=0.90
AGG_LEXICAL_TOKEN_JACCARD=0.75
AGG_FUNCTIONAL_MIN_EMPIRICAL_SUPPORT=2
AGG_SIBLING_GUARD_SCOPE=subject
```

See [Aggregation](aggregation.md) for what each threshold does.

### Web Search

```bash
WEB_SEARCH_ENABLED=false
WEB_SEARCH_PROVIDER=duckduckgo
WEB_SEARCH_TOP_K=3
WEB_SEARCH_TIMEOUT_SECONDS=8.0
WEB_SEARCH_MAX_SNIPPET_CHARS=400
WEB_SEARCH_MAX_TOTAL_CHARS=1800
WEB_SEARCH_ONTOLOGY_RENDER_ENABLED=true
WEB_SEARCH_ONTOLOGY_CRITIC_ENABLED=true
WEB_SEARCH_FACTS_RENDER_ENABLED=false
WEB_SEARCH_FACTS_CRITIC_ENABLED=false
WEB_SEARCH_PLANNER_ENABLED=true
WEB_SEARCH_PLANNER_MAX_QUERIES=3
WEB_SEARCH_PLANNER_MIN_QUERY_CHARS=12
WEB_SEARCH_PLANNER_MIN_CONFIDENCE=0.35
WEB_SEARCH_REUSE_EVIDENCE_ACROSS_ATTEMPT=true
WEB_SEARCH_MIN_SNIPPET_CHARS=40
WEB_SEARCH_ALLOWED_DOMAINS=
WEB_SEARCH_BLOCKED_DOMAINS=
WEB_SEARCH_REGION=wt-wt
WEB_SEARCH_SAFESEARCH=moderate
```

Search is "search-later": nodes run without search first, and only request external evidence when needed.

### Other

```bash
CLEAN=false                              # flush triple store before `ontocast process` batch
LOGGING_LEVEL=info                       # debug | info | warning | error
```

## LLM Graph Format (`LLM_GRAPH_FORMAT`)

- `turtle` (default): the LLM emits RDF graph fields as Turtle strings; prompt context chapters use `` ```ttl `` blocks.
- `jsonld`: the LLM emits compact JSON-LD objects (`@context` + `@graph`); prompt context uses `` ```json `` blocks.
- Domain models (`GraphUpdate`, critique reports, etc.) are **single canonical classes** at runtime. The format affects only LLM wire encoding, not duplicate Pydantic types.

## Ontology Context Mode

- `selected_single_ontology` (default): LLM picks one catalog ontology per content unit; no vector store required.
- `selected_vector_search_ontology`: hybrid vector retrieval + induced subgraph; requires `QDRANT_URI` **or** `LANCEDB_ENABLED=true` plus embedding settings.
- `fixed_single_ontology`: pin one catalog ontology via `ONTOLOGY_CONTEXT_FIXED_ONTOLOGY_ID` — ontology **IRI**, short `ontology_id`, or author **prefix**.

If vector mode is requested while no vector backend is available, the API returns `409` with `error_code: VECTOR_STORE_UNAVAILABLE`.

Details: [Ontology Context](ontology_context.md). Catalog read path: [Ontology Catalog](../architecture/ontology_catalog.md).

## Usage

```python
from ontocast.config import Config

config = Config()
tool_config = config.get_tool_config()

print(config.server.port)
print(config.server.max_visits_per_node)
print(tool_config.llm_config.provider)
print(tool_config.path_config.cache_dir)
```

## Graph Matching API

Entity alignment and evaluation endpoints are documented in [API Endpoints](api.md#graph-matching).

## Validation Notes

- `LLM_PROVIDER=openai`, `anthropic`, or `google` requires `LLM_API_KEY`.
- `LLM_MODEL_NAME` must match the selected provider family.
- `MAX_VISITS` is supported as an alias for `max_visits_per_node`.
- `RECURSION_LIMIT` was renamed to `BASE_RECURSION_LIMIT`.
- `WEB_SEARCH_ALLOWED_DOMAINS` and `WEB_SEARCH_BLOCKED_DOMAINS` accept comma-separated values.
- `LLM_CACHE_ENABLED` and `LLM_CACHE_READ_ONLY` control disk cache read/write behavior.
- `LLM_MAX_INFLIGHT` must be ≥ 1; `MAX_CONCURRENT_PROCESSES` must be ≥ 1 when set.

## Recommended Workflow

1. Copy `.env.example` to `.env`.
2. Fill in LLM credentials and backend settings.
3. Start with defaults for chunking, search, and aggregation.
4. Tune only after inspecting extraction quality and runtime.
