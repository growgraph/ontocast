# Tenancy

OntoCast partitions triple-store datasets and vector-store partitions (Qdrant collections or LanceDB tables) by **tenant** and **project**. This enables multiple logical workspaces on shared infrastructure.

## Naming Convention

```
{tenant}--{project}--facts
{tenant}--{project}--ontologies
```

Separator default: `--` (double hyphen).

**Built-in defaults** when parameters are omitted:

| Parameter | Default |
|-----------|---------|
| `tenant` | `ontocast` |
| `project` | `test` |

Default Fuseki datasets: `ontocast--test--facts`, `ontocast--test--ontologies`.

## How Tenancy Is Resolved

Tenant and project are **runtime parameters**, not environment variables. They may appear as:

- HTTP query parameters: `?tenant=acme&project=reports`
- Multipart form fields
- JSON body fields on `/process` and `/process_unit`

When `tenant` or `project` appears in the **query string**, the server retargets Fuseki datasets and vector-store partitions to the resolved scope. Requests without tenancy query parameters use the server's active tenant/project from startup (defaults: `ontocast` / `test`).

## Configuration Interaction

When `FUSEKI_DATASET` or `FUSEKI_ONTOLOGIES_DATASET` are **unset**, Fuseki config derives names from the default tenant/project at startup. Per-request `tenant`/`project` overrides route to the corresponding datasets at runtime.

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

# Flush partition data
curl -X POST "http://localhost:8999/flush?tenant=acme&project=reports"
```

## In-Memory Mode

When Fuseki is not configured, OntoCast uses an in-memory pyoxigraph backend with the same tenant/project partition model. Data is not persisted across process restarts.

## Related

- [Triple Store Configuration](triple_stores.md)
- [API Endpoints](api.md)
- [Ontology Context](ontology_context.md)
