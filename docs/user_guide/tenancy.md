# Tenancy

OntoCast partitions triple-store datasets and vector-store partitions (Qdrant collections or LanceDB tables) by **tenant** and **project**. This enables multiple logical workspaces on shared infrastructure.

## Naming Convention

```
{tenant}--{project}--facts
{tenant}--{project}--ontologies
{tenant}--{project}--shapes
```

The shapes partition is a triple-store dataset only — SHACL shapes are never
retrieved by similarity, so there is no vector-store counterpart.

Separator default: `--` (double hyphen).

**Built-in defaults** when parameters are omitted:

| Parameter | Default |
|-----------|---------|
| `tenant` | `ontocast` |
| `project` | `test` |

Default Fuseki datasets: `ontocast--test--facts`, `ontocast--test--ontologies`,
`ontocast--test--shapes`.

## How Tenancy Is Resolved

Tenant and project are **runtime parameters**, not environment variables. They may appear as:

- HTTP query parameters: `?tenant=acme&project=reports`
- Multipart form fields
- JSON body fields on `/process` and `/process_unit`

When `tenant` or `project` appears in the **query string**, the request is served by a `ToolBox` bound to that scope. Requests without tenancy query parameters use the server's active tenant/project from startup (defaults: `ontocast` / `test`).

Seed TTLs from `ONTOCAST_ONTOLOGY_DIRECTORY` are replayed into a new tenant, but only for ontologies its partition serves no terms for — an ontology the new scope already defines is never overwritten by the on-disk copy. Details: [Ontology Catalog](../architecture/ontology_catalog.md#why-it-resets-on-a-tenancy-switch). `FACTS_SHAPES_DIR` is seeded on the same switch; a tenant with neither a shapes partition nor a shapes directory has no shapes, which reads downstream as "SHACL never checked" rather than "conforms". See [Validation](validation.md#where-shapes-come-from).

### One ToolBox per scope

Each scope gets its own `ToolBox` over its own deep copy of the configuration, so isolation is structural: two tenants cannot see each other's datasets, collections or ontology catalog because they do not share the objects that name them.

This replaced retargeting a single process-wide `ToolBox` per request. That approach worked but had two costs: every switch rebuilt the ontology catalog, and the lock protecting the mutation serialized **all** multi-tenant traffic — two tenants could not be served concurrently. Scoped ToolBoxes need no lock, so different tenants now run in parallel.

The expensive tools stay shared. The LLM client and its response cache, the document converter, the chunker, the aggregator and the embedding model live on a `ToolBoxRuntime` that every scope reuses; only the triple store, ontology catalog, SPARQL tool and vector store are per-scope. A second tenant therefore costs a store connection and a catalog, not another embedding model.

Resident scopes are bounded by `MAX_TENANCY_SCOPES` (default 16) in a least-recently-used cache. Evicting a scope closes its backend connections. The bound exists because scopes come from request parameters: without it, a client iterating tenant names would grow the process without limit.

### From your own code

```python
scoped = await tools.for_scope("acme", "reports")
```

Returns the same ToolBox when the scope already matches, and otherwise builds (and caches) one. A single-tenant application never allocates a registry at all. `await tools.aclose()` closes every scope the ToolBox spawned.

## Configuration Interaction

When `FUSEKI_DATASET`, `FUSEKI_ONTOLOGIES_DATASET` or `FUSEKI_SHAPES_DATASET` are **unset**, Fuseki config derives names from the default tenant/project at startup. Per-request `tenant`/`project` overrides route to the corresponding datasets at runtime.

When explicit dataset names are set in `.env`, they apply as the configured default scope; per-request tenancy still switches the active partition when supported by the store layer.

Qdrant collection names follow the same pattern (`QDRANT_ONTOLOGY_COLLECTION`, `QDRANT_FACTS_COLLECTION` derive when unset).

LanceDB table names follow the same `{tenant}--{project}--ontologies` / `--facts` pattern under `LANCEDB_DATA_DIR` when `LANCEDB_ENABLED=true`.

## API Usage

All document and ontology routes accept optional tenancy parameters:

```bash
# Process into acme/reports partition
curl -X POST "http://localhost:8999/process?tenant=acme&project=reports" \
  -F "file=@document.pdf"

# Upload ontology to the same partition
curl -X POST "http://localhost:8999/ontologies?tenant=acme&project=reports" \
  -F "file=@domain.ttl"

# Upload SHACL shapes to the same partition
curl -X POST "http://localhost:8999/shapes?tenant=acme&project=reports" \
  -F "file=@domain-shapes.ttl"

# Flush partition data (shapes are retained; add &include_shapes=true to drop them)
curl -X POST "http://localhost:8999/flush?tenant=acme&project=reports"
```

## In-Memory Mode

When Fuseki is not configured, OntoCast uses an in-memory pyoxigraph backend with the same tenant/project partition model. Data is not persisted across process restarts.

## Related

- [Triple Store Configuration](triple_stores.md)
- [Ontology Catalog](../architecture/ontology_catalog.md)
- [API Endpoints](api.md)
- [Ontology Context](ontology_context.md)
