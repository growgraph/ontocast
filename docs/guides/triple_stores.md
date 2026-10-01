# Triple stores

OntoCast keeps ontologies, facts and SHACL shapes in a triple store. This page
shows how to choose between the in-memory store and Apache Jena Fuseki, how to
run Fuseki, and how seed files reach the store.

## Choosing a backend

| | In-memory (default) | Apache Jena Fuseki |
|---|---|---|
| Persistence | None: data is gone when the process exits | Kept across restarts |
| SPARQL | Full SPARQL 1.1 (pyoxigraph) | Full SPARQL 1.1 |
| Tenancy partitions | Yes | Yes, one dataset per partition |
| Setup | None | A Fuseki server and two settings |

OntoCast uses Fuseki only when both
[`FUSEKI_URI`](../reference/configuration/storage.md#fuseki_uri) and
[`FUSEKI_AUTH`](../reference/configuration/storage.md#fuseki_auth) are set.
With either one missing it uses the in-memory store, without a warning, so a
run you meant to keep is lost when the process exits.

## Running Fuseki

The repository ships a Docker Compose file for Fuseki in
[`docker/fuseki`](https://github.com/growgraph/ontocast/tree/main/docker/fuseki).
It publishes Fuseki on port 3032 of the host and takes the admin credentials
from its own settings file:

```bash
cd docker/fuseki
cp .env.example .env        # set TS_USERNAME and TS_PASSWORD
docker compose --env-file .env up fuseki -d
```

Then point OntoCast at it:

```bash
export FUSEKI_URI=http://localhost:3032
export FUSEKI_AUTH=admin/your-password
```

`FUSEKI_URI` is the server root, not a dataset URL; a link copied from the
Fuseki web interface (`.../#/dataset/...`) is trimmed to the root.
`FUSEKI_AUTH` is `user/password` or `user:password`.

OntoCast creates the datasets it needs (TDB2) at startup, and when a request
first names a new partition. Their names come from the tenant and project; see
[Tenancy](tenancy.md).

## Seed ontologies and shapes

Seed directories put your files into the store at startup. They are read, never
written: the store is where data is kept.

| | Ontologies | SHACL shapes |
|---|---|---|
| Setting | [`ONTOCAST_ONTOLOGY_DIRECTORY`](../reference/configuration/storage.md#ontocast_ontology_directory) | [`FACTS_SHAPES_DIR`](../reference/configuration/facts-validation.md#facts_shapes_dir) |
| Flag on `serve` and `process` | `--ontology-dir` | `--shapes-dir` |
| Files read | `*.ttl` in the directory | `*.ttl` in the directory and its subdirectories |
| When a file is written to the store | Only if the store holds no version of that ontology | At every startup, so an edited file takes effect on restart |

Because an ontology already in the store is never replaced by its seed file,
an ontology that OntoCast extended in Fuseki keeps its extensions across
restarts. To start over from your files, flush the partition.

A directory that does not exist stops `ontocast process` and only warns under
`ontocast serve`. To run with no seed ontologies when the setting is in your
environment, pass `--ontology-dir ''`.

`POST /ontologies`, `PUT /ontologies/...`, `POST /shapes` and the matching
`DELETE` routes change the store only; your files are left as they are. Shapes
are stored in their own partition, apart from the ontologies; [Validation and
SHACL](validation.md) explains why.

## Flushing data

```bash
# Delete facts and ontologies in the startup partition
curl -X POST http://127.0.0.1:8999/flush

# Delete them in another partition, shapes included
curl -X POST "http://127.0.0.1:8999/flush?tenant=acme&project=demo&include_shapes=true"
```

A flush cannot be undone, and the server does not check who asks for it.
Shapes are kept unless you pass `include_shapes=true`. For batch runs,
`CLEAN=true` makes `ontocast process` flush the startup partition before it
loads the seed ontologies.

## Integer subtypes in the in-memory store

The in-memory store normalizes literals when it stores them: a value typed
`xsd:nonNegativeInteger` is stored, and served back, as `xsd:integer`. OWL 2
requires `xsd:nonNegativeInteger` on qualified cardinality restrictions, so an
ontology read back from the in-memory store is not OWL 2 DL on those axioms,
and a reasoner may reject or ignore them. OntoCast's own processing is not
affected. If you need strict OWL 2 DL output, keep your authored Turtle as the
source of truth rather than exporting it from the store.
