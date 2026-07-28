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
FUSEKI_URI=http://localhost:3030
FUSEKI_AUTH=admin:password
#FUSEKI_DATASET=ontocast--test--facts
#FUSEKI_ONTOLOGIES_DATASET=ontocast--test--ontologies

# Seed ontologies (optional — bootstrap only, not persistence)
ONTOCAST_ONTOLOGY_DIRECTORY=/path/to/seed/ttl
```

Persistence is handled by the triple store only. Local TTL export to `working_directory` is no longer used.

Ontology **catalog reads** (headers, by-IRI graphs, merged working graphs) go through `OntologyManager`, not ad-hoc `fetch_ontologies()` from callers — see [Ontology Catalog](../architecture/ontology_catalog.md).

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

## Targeted Catalog Reads

`fetch_ontologies()` materializes every stored ontology into rdflib. That is the right call at startup, but it is far too much for the per-content-unit retrieval path, which only needs to know *which* ontologies to pull. `TripleStoreManager` therefore exposes three narrower reads:

| Method | Returns | Cost |
|---|---|---|
| `aselect(query, *, use_ontologies_dataset=True)` | `list[dict[str, str]]` — one dict per SPARQL SELECT solution | One query |
| `aconstruct(query, *, use_ontologies_dataset=True)` | `RDFGraph` — real RDF terms, no prefix bindings | One query |
| `afetch_ontology_catalog()` | `list[OntologyHeader]` — `iri`, `version`, `hash`, `parent_hashes`, `created_at`, `graph_uri` per stored version | One query, no graphs |
| `afetch_ontologies_by_iri(iris)` | `list[Ontology]` with graphs, restricted to `iris` (empty means no restriction) | Only the named graphs requested |

`aselect` rows carry each term's **lexical value** only — term kind and datatype are dropped, so constrain kinds in the query itself (`FILTER(isIRI(?x))`). Unbound variables are simply absent from the row. `aconstruct` has no such loss: blank nodes and datatypes survive. What it *cannot* carry is prefix bindings — those are serialization metadata rather than triples, so a caller that needs them must source them elsewhere.

Both raise rather than returning an empty result on failure, because empty is indistinguishable from "nothing matched".

`OntologyHeader` is deliberately not an `Ontology`: constructing an `Ontology` recomputes its hash from the graph, so a graph-less one would carry fabricated lineage. Run `dedupe_terminal_ontologies()` over headers to pick terminal versions without downloading anything — it accepts headers and ontologies alike, as does `select_relevant_ontologies()`.

### Custom Backends

Implementing a `TripleStoreManager` subclass still requires only `fetch_ontologies()`. Every method above has a working base-class default expressed in terms of it, so a custom backend keeps working unchanged — it just fetches more than it needs.

Two independent opt-ins into the fast paths, both dispatched on a predicate rather than on the concrete type:

- `supports_sparql_select()` → `True` plus `aselect()` — enables targeted catalog reads and reference expansion.
- `supports_sparql_construct()` → `True` plus `aconstruct()` — enables the optional [candidate pushdown](ontology_context.md#catalog-io).

They are separate because a backend can answer row queries without returning triples: Fuseki's SELECT path speaks `application/sparql-results+json` only, and needs a different `Accept` header for CONSTRUCT.

---

## Backend Comparison

| Feature | Fuseki | In-Memory |
|---------|--------|-----------|
| **Persistence** | Yes | No (process lifetime) |
| **SPARQL** | Full 1.1 | Full 1.1 (pyoxigraph) |
| **`aselect` fast path** | Yes | Yes |
| **`aconstruct` fast path** | Yes | Yes |
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
