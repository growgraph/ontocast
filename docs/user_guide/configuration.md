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
FUSEKI_URI=http://localhost:3030/test
FUSEKI_AUTH=admin/admin
FUSEKI_DATASET=dataset_name
FUSEKI_ONTOLOGIES_DATASET=ontologies

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_AUTH=neo4j/test
NEO4J_PORT=7476
NEO4J_BOLT_PORT=7689
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
WEB_SEARCH_ALLOWED_DOMAINS=              # comma-separated
WEB_SEARCH_BLOCKED_DOMAINS=              # comma-separated
WEB_SEARCH_REGION=wt-wt
WEB_SEARCH_SAFESEARCH=moderate
```

Search is "search-later": nodes run without search first, and only request external evidence when needed.

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
