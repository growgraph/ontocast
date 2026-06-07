# Triple Store Configuration

OntoCast stores ontologies and facts through a unified **TripleStoreManager** interface. Two backends are supported today:

1. **Apache Fuseki** (production) — persistent RDF store with SPARQL
2. **In-Memory (pyoxigraph)** (default) — zero-config backend for development and tests

When `FUSEKI_URI` and `FUSEKI_AUTH` are set, Fuseki is used. Otherwise OntoCast uses the in-memory backend automatically.

---

## Configuration

### Environment Variables

```bash
# Fuseki (optional — production)
FUSEKI_URI=http://localhost:3032
FUSEKI_AUTH=admin:password
#FUSEKI_DATASET=ontocast--test--facts
#FUSEKI_ONTOLOGIES_DATASET=ontocast--test--ontologies

# Seed ontologies (optional — bootstrap only, not persistence)
ONTOCAST_ONTOLOGY_DIRECTORY=/path/to/seed/ttl
```

Persistence is handled by the triple store only. Local TTL export to `working_directory` is no longer used.

### Tenancy and Partitions

Both Fuseki and the in-memory backend isolate data by tenant/project:

- `{tenant}--{project}--facts` — extracted facts graphs
- `{tenant}--{project}--ontologies` — catalog / versioned ontologies

When dataset env vars are unset, OntoCast derives names from the default tenant `ontocast` and project `test`. Per-request `?tenant=` / `?project=` retarget the active partition at runtime. See [Tenancy](tenancy.md).

### Detecting the Active Backend

```python
from ontocast.config import Config

config = Config()
tool_config = config.get_tool_config()

if tool_config.fuseki.uri and tool_config.fuseki.auth:
    print("Using Fuseki triple store")
else:
    print("Using in-memory triple store")
```

---

## Apache Fuseki Setup

Sample Docker configs: [ontocast/docker](https://github.com/growgraph/ontocast/tree/main/docker).

```bash
cd docker/fuseki
cp .env.example .env
docker compose --env-file .env fuseki up -d
```

Configure OntoCast:

```bash
FUSEKI_URI=http://localhost:3032
FUSEKI_AUTH=admin:your-password
```

---

## In-Memory Backend

No setup required. Data lives in process memory (pyoxigraph) and is lost on restart.

Use Fuseki for production deployments. The in-memory backend supports the same tenancy partition model as Fuseki.

---

## Seed Ontologies

Place `.ttl` files in `ONTOCAST_ONTOLOGY_DIRECTORY`. On startup, `ToolBox` scans that directory and materializes any ontologies not already present in the triple store. This is a one-way bootstrap path — ongoing persistence is through the triple store.

---

## Backend Comparison

| Feature | Fuseki | In-Memory |
|---------|--------|-----------|
| **Persistence** | Yes | No (process lifetime) |
| **SPARQL** | Full 1.1 | Internal only |
| **Tenancy partitions** | Yes | Yes |
| **Setup** | Docker + env | Automatic |

---

## Flushing Data

```bash
# Clean active partition
curl -X POST http://localhost:8999/flush

# Clean a specific tenant/project partition
curl -X POST "http://localhost:8999/flush?tenant=acme&project=demo"
```

**Warning:** Flush is irreversible.
