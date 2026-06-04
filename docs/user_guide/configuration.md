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
│   ├── path_config: PathConfig
│   ├── neo4j: Neo4jConfig
│   ├── fuseki: FusekiConfig
│   ├── domain: DomainConfig
│   ├── web_search: WebSearchConfig
│   ├── aggregation: AggregationConfig
│   ├── embedding: EmbeddingConfig
│   ├── patch_retrieval: PatchRetrievalConfig
│   └── qdrant: QdrantConfig
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
#ONTOLOGY_CONTEXT_FIXED_ONTOLOGY_ID=catalog_id
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
```

Semantic chunking is configured here. **Section-aligned labels** and filtering are not chunker settings: they run when `/process` or CLI file mode passes `target_sections` and/or `summarize_sections` (see [Structured documents](concepts.md#structured-documents-optional)).

### Structured documents (per request)

No environment variables. Pass on `POST /process`, multipart form, JSON body, or CLI batch mode:

| Parameter | CLI flag | Description |
|-----------|----------|-------------|
| `target_sections` | `--target-sections` | Comma-separated or JSON list; enables tagging and keeps only these sections |
| `summarize_sections` | `--summarize-sections` | Enables tagging + summarization; `*` or empty = all chunks |
| `summary_max_sentences` | `--summary-max-sentences` | Max sentences per summary (default `5`) |

```bash
ontocast --input-path ./papers/ \
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

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_AUTH=neo4j/test
NEO4J_PORT=7476
NEO4J_BOLT_PORT=7689
```

See [Tenancy](tenancy.md) for how tenant/project names relate to dataset and collection names.

### Embeddings

```bash
EMBEDDING_PROVIDER=huggingface          # huggingface | openai | ollama
EMBEDDING_MODEL_NAME=paraphrase-multilingual-MiniLM-L12-v2
# EMBEDDING_API_KEY=
# EMBEDDING_BASE_URL=http://localhost:11434
EMBEDDING_DIMENSION=384
```

### Qdrant

```bash
QDRANT_URI=http://localhost:6333
QDRANT_API_KEY=abc123-qwe
QDRANT_TOP_K=10
QDRANT_GRPC_PORT=6334
QDRANT_USE_GRPC=false
QDRANT_INDUCED_SUBGRAPH_DEPTH=1
QDRANT_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES=300
QDRANT_INDUCED_SUBGRAPH_ESTIMATED_TRIPLES_PER_QUERY=24
# QDRANT_ONTOLOGY_COLLECTION=ontocast--test--ontologies
# QDRANT_FACTS_COLLECTION=ontocast--test--facts
# QDRANT_FUSION_CORE_WEIGHT=0.7
# QDRANT_FUSION_NEIGHBORHOOD_WEIGHT=0.3
# QDRANT_FUSION_BM25_WEIGHT=0.2
# QDRANT_DEDUP_MODE=iri
```

Budget behavior:

- `QDRANT_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES` is the global upper bound returned to the LLM.
- `QDRANT_INDUCED_SUBGRAPH_ESTIMATED_TRIPLES_PER_QUERY` shapes per-entity allocation during retrieval.

See [Ontology Context](ontology_context.md) for vector-search mode requirements.

### Ontology Patch Retrieval

Post-vector scoring and capping (backend-agnostic; prefix `ONTOLOGY_PATCH_`):

```bash
ONTOLOGY_PATCH_PER_QUERY_CORE_SCORE_RATIO=0.85
ONTOLOGY_PATCH_PER_QUERY_NEIGHBORHOOD_SCORE_RATIO=0.85
ONTOLOGY_PATCH_MIN_MERGED_MAX_SCORE=0.18
# ONTOLOGY_PATCH_MMR_LAMBDA=0.7
# ONTOLOGY_PATCH_MAX_ATOMS=0
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
```

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
CLEAN=false                              # flush triple store before --input-path batch
LOGGING_LEVEL=info                       # debug | info | warning | error
```

## LLM Graph Format (`LLM_GRAPH_FORMAT`)

- `turtle` (default): the LLM emits RDF graph fields as Turtle strings; prompt context chapters use `` ```ttl `` blocks.
- `jsonld`: the LLM emits compact JSON-LD objects (`@context` + `@graph`); prompt context uses `` ```json `` blocks.
- Domain models (`GraphUpdate`, critique reports, etc.) are **single canonical classes** at runtime. The format affects only LLM wire encoding, not duplicate Pydantic types.

## Ontology Context Mode

- `selected_single_ontology` (default): LLM picks one catalog ontology per content unit; no Qdrant required.
- `selected_vector_search_ontology`: Qdrant stitched ensemble; requires `QDRANT_URI` and embedding settings.
- `fixed_single_ontology`: pin one catalog `ontology_id` via `ONTOLOGY_CONTEXT_FIXED_ONTOLOGY_ID`.

If vector mode is requested while Qdrant is unavailable, the API returns `409` with `error_code: VECTOR_STORE_UNAVAILABLE`.

Details: [Ontology Context](ontology_context.md).

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
