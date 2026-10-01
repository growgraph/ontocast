# Tenancy

One OntoCast server can keep the data of several users or projects apart. This
page explains how a tenant and a project name a partition of the stores, how a
request chooses one, and what each partition costs.

## How partitions are named

Each tenant and project pair owns three partitions:

```
{tenant}--{project}--facts
{tenant}--{project}--ontologies
{tenant}--{project}--shapes
```

In Fuseki these are datasets; in the in-memory store they are separate stores
in the process. The facts and ontologies names are also used for the vector
collections in Qdrant and the tables in LanceDB. Shapes have no vector
counterpart, because they are never retrieved by similarity; [Validation and
SHACL](validation.md) explains why they get a partition of their own.

Without a tenant or project, OntoCast uses tenant `ontocast` and project
`test`, so the default Fuseki datasets are `ontocast--test--facts`,
`ontocast--test--ontologies` and `ontocast--test--shapes`.

## Choosing a partition

Tenant and project are not settings. They come from two places:

- **At startup**, `ontocast serve` and `ontocast process` take `--tenant` and `--project`. This is the partition the command uses by default, and the one the seed ontologies and shapes are loaded into.
- **Per request**, the `tenant` and `project` query parameters. A request that has either one is served from that partition, and the one it leaves out takes its default. A request with neither uses the startup partition.

Only the query string selects a partition: `tenant` or `project` in a form
field or JSON body is ignored.

```bash
# Process a document into acme/reports
curl -X POST "http://127.0.0.1:8999/process?tenant=acme&project=reports" \
  -F "file=@document.pdf"

# Upload an ontology to the same partition
curl -X POST "http://127.0.0.1:8999/ontologies?tenant=acme&project=reports" \
  -F "file=@domain.ttl"

# Upload SHACL shapes to the same partition
curl -X POST "http://127.0.0.1:8999/shapes?tenant=acme&project=reports" \
  -F "file=@domain-shapes.ttl"

# Delete the partition's facts and ontologies; add &include_shapes=true to drop shapes too
curl -X POST "http://127.0.0.1:8999/flush?tenant=acme&project=reports"
```

The server does not authenticate callers, so any caller can name any
partition, including in `/flush`. Keep it behind a proxy that decides who may
use which tenant.

Because the commands always name the partitions after the tenant and project,
`FUSEKI_DATASET`, `FUSEKI_ONTOLOGIES_DATASET`, `FUSEKI_SHAPES_DATASET` and the
Qdrant collection and LanceDB table settings take effect only when you build a
`ToolBox` in your own code.

## A new partition and the seed directories

The first time a request names a partition, OntoCast fills its catalog from
`ONTOCAST_ONTOLOGY_DIRECTORY`, but only with ontologies the partition does not
already hold, so it never overwrites an ontology a previous run extended. It
loads the shapes in `FACTS_SHAPES_DIR` the same way. A partition with no
shapes, and no shapes directory configured, is not validated against SHACL:
its responses report `shacl_evaluated: null` rather than a pass.

## One ToolBox per partition

Each partition is served by its own `ToolBox`, over its own copy of the
configuration, so one tenant's requests never see another's datasets,
collections or catalog, and requests for different tenants run concurrently.

The expensive tools are shared: the LLM client and its cache, the document
converter, the chunker and the embedding model. A second partition costs a
store connection and a catalog, not another model.

[`MAX_TENANCY_SCOPES`](../reference/configuration/pipeline.md#max_tenancy_scopes)
caps how many partitions stay open. Beyond it, the least recently used one is
closed, and reopened on its next request. The cap exists because partitions
come from request parameters: without it, a client cycling through tenant names
would grow the process without limit.

In your own code, `await tools.for_scope("acme", "reports")` returns the
`ToolBox` for a partition; see [Embedding OntoCast in your
agent](embedding.md#several-tenants-in-one-process).

## Related

- [Triple stores](triple_stores.md): where the partitions are stored.
- [Ontology catalog](../internals/ontology_catalog.md#tenancy): how the catalog is rebuilt for a partition.
- [HTTP API](../reference/http_api.md): every route and its parameters.
