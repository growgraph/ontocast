# Docker services for OntoCast

This directory contains Docker Compose stacks for **optional** infrastructure OntoCast can use at runtime.

| Service | Directory | Purpose |
|---------|-----------|---------|
| **Apache Fuseki** | `fuseki/` | Persistent RDF triple store (production) |
| **Qdrant** | `qdrant/` | Vector store for ontology patch retrieval |

OntoCast does **not** require Docker to run. When `FUSEKI_URI` / `FUSEKI_AUTH` are unset, the server uses an **in-memory pyoxigraph** triple store automatically (data is not persisted across restarts).

---

## Triple store backend selection

OntoCast picks the triple store in `ToolBox` at startup:

1. **Fuseki** — when both `FUSEKI_URI` and `FUSEKI_AUTH` are set
2. **In-memory** — otherwise (zero git config, process-local only)

There is no filesystem or Neo4j fallback anymore. Use Fuseki (below) when you need durable RDF storage.

### OntoCast `.env` (Fuseki client)

Point OntoCast at a running Fuseki instance:

```bash
# Fuseki HTTP service root (not a dataset path or UI fragment URL)
FUSEKI_URI=http://localhost:3032
FUSEKI_AUTH=admin/abc123-qwe

# Optional: override default tenant/project dataset names
#FUSEKI_DATASET=ontocast--test--facts
#FUSEKI_ONTOLOGIES_DATASET=ontocast--test--ontologies
```

When dataset env vars are unset, names default to `ontocast--test--facts` and `ontocast--test--ontologies`. Per-request `?tenant=` / `?project=` retarget partitions at runtime. See [Tenancy](../docs/user_guide/tenancy.md).

### Seed ontologies (optional)

`ONTOCAST_ONTOLOGY_DIRECTORY` is independent of Docker. On `ToolBox.initialize()`, OntoCast scans that directory for `*.ttl` files and materializes any ontologies not already in the triple store. Ongoing persistence is through the triple store only.

---

## Apache Fuseki

**1. Prepare the environment file:**

```bash
cd docker/fuseki
cp .env.example .env
# Edit with your values
```

**Example `docker/fuseki/.env.example`:**

```bash
IMAGE_VERSION=conceptkernel/jena-fuseki:6.0
SPEC=ontocast
CONTAINER_NAME="${SPEC}.fuseki"
TS_PORT=3032
TS_PASSWORD="abc123-qwe"
TS_USERNAME="admin"
```

**2. Start / stop:**

```bash
cd docker/fuseki
docker compose --env-file .env up fuseki -d

# Stop (use container name from .env, e.g. ontocast.fuseki)
docker compose stop ontocast.fuseki
```

**3. Access:**

- Web UI: http://localhost:3032
- SPARQL (per dataset): `http://localhost:3032/{dataset}/sparql`

**4. Wire OntoCast** — set `FUSEKI_URI` and `FUSEKI_AUTH` in your main `.env` to match the compose credentials and port.

---

## Qdrant (optional vector store)

Required only when using vector-backed ontology context (`SELECTED_VECTOR_SEARCH_ONTOLOGY`).

**1. Prepare:**

```bash
cd docker/qdrant
cp .env.example .env
```

**Example `docker/qdrant/.env.example`:**

```bash
IMAGE_VERSION=qdrant/qdrant:v1.18.1
SPEC=ontocast
CONTAINER_NAME="${SPEC}.qdrant"
QDRANT_HTTP_PORT=6333
QDRANT_GRPC_PORT=6334
QDRANT_API_KEY="abc123-qwe"
QDRANT_LOG_LEVEL=INFO
```

**2. Start / stop:**

```bash
cd docker/qdrant
docker compose --env-file .env up qdrant -d
docker compose stop ontocast.qdrant
```

**3. Wire OntoCast:**

```bash
QDRANT_URI=http://localhost:6333
QDRANT_API_KEY=abc123-qwe
```

Collection names follow the same `{tenant}--{project}--{facts|ontologies}` pattern when unset.

---

## LanceDB (embedded vector store alternative)

Use when you want vector search without running Qdrant. Configure **either** `QDRANT_URI` **or** `LANCEDB_ENABLED=true`, not both.

**1. Install the optional extra:**

```bash
cd ontocast
uv sync --extra lancedb
```

**2. Wire OntoCast:**

```bash
LANCEDB_ENABLED=true
LANCEDB_DATA_DIR=~/.lancedb_data
```

OntoCast calls `lancedb.connect(LANCEDB_DATA_DIR)` once. Tenant/project isolation uses Lance **table names** (`{tenant}--{project}--ontologies` / `--facts`), matching Qdrant collection naming.

Shared retrieval settings use the `VECTOR_STORE_*` prefix (fusion weights, `top_k`, dedup mode, induced subgraph limits, proposition windows).

---

## Backend comparison

| Feature | Fuseki (Docker) | In-memory (default) |
|---------|-----------------|---------------------|
| **Persistence** | Yes | No (process lifetime) |
| **SPARQL** | Full 1.1 | Internal only |
| **Tenancy partitions** | Yes | Yes |
| **Docker required** | Yes | No |

| Feature | Qdrant (Docker) | LanceDB (embedded) |
|---------|-----------------|---------------------|
| **Persistence** | Yes | Yes (local directory) |
| **Tenancy partitions** | Yes (collections) | Yes (table names) |
| **Env var** | `QDRANT_URI` | `LANCEDB_ENABLED` + `LANCEDB_DATA_DIR` |
| **Docker required** | Yes | No |

---

## Troubleshooting

### Fuseki

```bash
curl http://localhost:3032/$/ping
curl http://localhost:3032/$/datasets
docker compose -f docker/fuseki/docker-compose.yml restart
```

### Qdrant

```bash
curl http://localhost:6333/healthz
```

### Common problems

- **Connection refused** — container not running or wrong port in OntoCast `.env`
- **Authentication failed** — `FUSEKI_AUTH` must be `user/password` (slash-separated)
- **Wrong URI** — use the Fuseki HTTP root (`http://host:port`), not `/#/dataset/...` UI links
- **Data gone after restart** — expected with in-memory backend; run Fuseki for persistence

---

## Further reading

- [Triple store guide](../docs/user_guide/triple_stores.md)
- [Tenancy](../docs/user_guide/tenancy.md)
- [Configuration](../docs/user_guide/configuration.md)
