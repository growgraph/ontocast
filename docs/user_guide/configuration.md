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
│   └── aggregation: AggregationConfig
└── server: ServerConfig
```

## Environment Variables

### LLM

```bash
LLM_PROVIDER=openai                     # openai | ollama
LLM_MODEL_NAME=gpt-4o-mini
LLM_TEMPERATURE=0.0
LLM_API_KEY=your_openai_api_key_here    # required for openai provider
LLM_BASE_URL=http://localhost:11434     # optional (mainly for ollama)
```

### Server

```bash
PORT=8999
BASE_RECURSION_LIMIT=1000
ESTIMATED_CHUNKS=30
MAX_VISITS=3                             # alias for max_visits_per_node
RENDER_MODE=ontology_and_facts           # ontology | facts | ontology_and_facts
LLM_GRAPH_FORMAT=turtle                  # turtle | jsonld — controls LLM output encoding and prompt context graphs
ONTOLOGY_CONTEXT_MODE=selected_single_ontology   # selected_single_ontology | selected_vector_search_ontology | fixed_single_ontology
#ONTOLOGY_CONTEXT_FIXED_ONTOLOGY_ID=catalog_id  # required for fixed_single_ontology
ONTOLOGY_MAX_TRIPLES=50000               # empty/unset for unlimited
PARALLEL_WORKERS=4
PARALLEL_FACTS_RETRIES=3
PARALLEL_ONTOLOGY_RETRIES=3
ENABLE_ONTOLOGY_CONSOLIDATION=false
```

### Chunking

```bash
CHUNK_BREAKPOINT_THRESHOLD_TYPE=percentile  # percentile | standard_deviation | interquartile | gradient
CHUNK_BREAKPOINT_THRESHOLD_AMOUNT=95.0
CHUNK_MIN_SIZE=3000
CHUNK_MAX_SIZE=12000
```

### Triple Stores

```bash
# Fuseki
FUSEKI_URI=http://localhost:3030
FUSEKI_AUTH=admin/admin
FUSEKI_DATASET=dataset_name
FUSEKI_ONTOLOGIES_DATASET=ontologies

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_AUTH=neo4j/test
NEO4J_PORT=7476
NEO4J_BOLT_PORT=7689
```

### Qdrant Retrieval Budgets

```bash
QDRANT_URI=http://localhost:6333
QDRANT_API_KEY=abc123-qwe
QDRANT_TOP_K=10
QDRANT_INDUCED_SUBGRAPH_DEPTH=1
# Hard cap for total stitched context triples
QDRANT_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES=300
# Estimated budget per query window used to distribute triples across ranked entities
QDRANT_INDUCED_SUBGRAPH_ESTIMATED_TRIPLES_PER_QUERY=24
```

Budget behavior:

- `QDRANT_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES` is the global upper bound returned to the LLM.
- `QDRANT_INDUCED_SUBGRAPH_ESTIMATED_TRIPLES_PER_QUERY` shapes per-entity allocation during retrieval.
- Retrieval guarantees broad seed coverage when feasible, then allocates remaining budget by entity relevance.

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
WEB_SEARCH_ALLOWED_DOMAINS=              # comma-separated
WEB_SEARCH_BLOCKED_DOMAINS=              # comma-separated
WEB_SEARCH_REGION=wt-wt
WEB_SEARCH_SAFESEARCH=moderate
```

Search is "search-later": nodes run without search first, and only request external evidence when needed.

## LLM graph format (`LLM_GRAPH_FORMAT`)

- `turtle` (default): the LLM emits RDF graph fields as Turtle strings; prompt context chapters use `` ```ttl `` blocks.
- `jsonld`: the LLM emits compact JSON-LD objects (`@context` + `@graph`); prompt context uses `` ```json `` blocks.
- Domain models (`GraphUpdate`, `FactsRenderReport`, critique reports, etc.) are **single canonical classes** at runtime. The format affects only LLM wire encoding (parse validators + JSON Schema in format instructions), not duplicate Pydantic types.
- The setting applies consistently to render and critique agents (output instructions, format-bound JSON Schema, prompt context chapters, and `llm_graph_format_ctx` during parsing).

## Ontology Context Mode Behavior

- `ONTOLOGY_CONTEXT_MODE=selected_single_ontology` is the default (LLM-chosen catalog TTL per unit); it does not require Qdrant.
- `selected_single_ontology` skips vector-store initialization when running the server or file batch processing unless you select vector mode.
- `ontology_context_mode=selected_vector_search_ontology` requires configured and initialized vector infrastructure (`QDRANT_URI` and compatible embedding settings).
- If a request asks for `selected_vector_search_ontology` while vector store is unavailable, API returns `409` with `error_code: VECTOR_STORE_UNAVAILABLE`.
- `fixed_single_ontology` uses the catalog ontology whose `ontology_id` is `ONTOLOGY_CONTEXT_FIXED_ONTOLOGY_ID` (or per-request `ontology_context_fixed_ontology_id` query/form/JSON field). Omitting the id when mode is fixed returns HTTP 400 from the API.

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

## RDF Graph Matching

Matching is split into entity alignment (global, across many graphs) and evaluation
(predicted vs ground truth, using explicit entity mappings).

### `POST /match/entities`

Align entities globally across a list of graphs (connected-component clustering over
embedding + symbolic compatibility).

```json
{
  "graphs": [
    {"id": "gt:doc1.ttl", "graph": "@prefix ex: <https://gt.example/> . ..."},
    {"id": "predicted:doc1.ttl", "graph": "@prefix ex: <https://pred.example/> . ..."}
  ],
  "regime": "ontology_loose",
  "similarity_threshold": 0.8
}
```

### `POST /match/derive-matches`

Derive 1:1 predicted↔gt entity matches for one graph pair from alignment clusters.

```json
{
  "clusters": [],
  "predicted_graph_id": "predicted:doc1.ttl",
  "gt_graph_id": "gt:doc1.ttl",
  "similarity_threshold": 0.8
}
```

### `POST /match/evaluate`

Compute triple and entity precision/recall/F1 given graphs and entity matches.
Label triples (`rdfs:label`) are excluded from triple metrics.

```json
{
  "predicted_graph": "@prefix ex: <https://predicted.example/> . ...",
  "gt_graph": "@prefix ex: <https://gt.example/> . ...",
  "entity_matches": [
    {"predicted_entity": "https://pred.example/a", "gt_entity": "https://gt.example/a", "similarity": 0.95}
  ]
}
```

Precision and recall use the same semantics for triples and entities:
precision = TP / |predicted|, recall = TP / |ground truth|.

### Standalone CLI

`match-dirs` is a standalone HTTP client (no ontocast imports). It calls all three
endpoints per paired TTL file: align (gt + predicted only), derive, evaluate.

```bash
uv run match-dirs \
  --gt ./benchmark \
  --predicted ./extracted \
  --url http://localhost:8999 \
  --regime ontology_strict \
  --similarity-threshold 0.8
```

## Validation Notes

- `LLM_PROVIDER=openai` requires `LLM_API_KEY`.
- `LLM_MODEL_NAME` must match the selected provider family.
- `MAX_VISITS` is supported as an alias for `max_visits_per_node`.
- `WEB_SEARCH_ALLOWED_DOMAINS` and `WEB_SEARCH_BLOCKED_DOMAINS` accept comma-separated values.

## Recommended Workflow

1. Copy `.env.example` to `.env`.
2. Fill in LLM credentials and backend settings.
3. Start with defaults for chunking/search/aggregation.
4. Tune only after inspecting extraction quality and runtime.
