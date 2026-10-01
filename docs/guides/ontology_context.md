---
search:
  boost: 3
---

# Choosing ontology context

For each part of a document, OntoCast shows the model a slice of your
ontologies and asks it to extract in those terms. This page helps you choose
how that slice is picked, set up vector retrieval when you need it, and check
what each part was shown.

## What the ontology context is

OntoCast cuts a document into parts (content units) and renders each one
separately. Every render carries the unit's *ontology context*: the classes,
properties and individuals the unit should be expressed in, taken from the
catalog, the set of ontologies OntoCast knows. A term missing from the context
is one the model either has to create (in an ontology unit) or cannot use (in a
facts unit), so the context bounds what an extraction can say.

Ontology units resolve their own context, using the mode you choose below.
Facts units, once an ontology stage has run, share one context: the union of
the ontologies the document's ontology units produced. Facts units resolve
their own context only when no ontology stage runs, that is under
`RENDER_MODE=facts`.

## What if the context is empty?

An empty context means different things to the two kinds of unit:

| Unit | What an empty context means | What happens |
|---|---|---|
| Ontology unit | No catalog ontology fits this text yet | OntoCast creates a new ontology from the text, with IRIs under [`CURRENT_DOMAIN`](../reference/configuration/pipeline.md#current_domain). This is how a run that starts with an empty catalog builds its first ontology |
| Facts unit | There is nothing to extract against | The model falls back on generic vocabulary. The result is well-formed and passes validation, because no shape targets its terms, so the pass tells you nothing |

To stop on the second case, set
[`ONTOLOGY_CONTEXT_REQUIRED=true`](../reference/configuration/pipeline.md#ontology_context_required):
a facts unit with an empty context then stops the run with an error. The
setting never applies to ontology units. Leave it off when the catalog is meant
to start empty, as it does when the default render mode builds the ontology;
turn it on when you extract against a curated catalog, where an empty context
can only mean the catalog did not load. With it on, `ontocast process` under
`RENDER_MODE=facts` also refuses to start on an empty catalog, before any
document is converted.

With it off, the run continues; in vector mode
`retrieval_metrics.empty_snapshot_reason` names the likely cause. Whatever the
setting, vector mode refuses to start when the index holds ontologies but the
catalog none, or the catalog holds ontologies but the index stays empty.

## The three modes

[`ONTOLOGY_CONTEXT_MODE`](../reference/configuration/pipeline.md#ontology_context_mode)
decides how an ontology unit's context is picked. It can also be set per
request; see [Configuring OntoCast](configuration.md#how-settings-are-read).

| | `selected_single_ontology` (default) | `selected_vector_search_ontology` | `fixed_single_ontology` |
|---|---|---|---|
| What a unit sees | The one catalog ontology the model picks for it, or none | The terms retrieval finds relevant, from every catalog ontology, with their surroundings | The same ontology for every unit |
| Extra cost | One LLM call per unit | Embedding and vector search; no LLM call | None |
| Needs | Ontologies with a title and description to choose by | A vector store and an embedding model | [`ONTOLOGY_CONTEXT_FIXED_ONTOLOGY_ID`](../reference/configuration/pipeline.md#ontology_context_fixed_ontology_id) |
| Pick it when | The catalog holds a few ontologies, each covering its own topic | The catalog is large, or one unit needs terms from several ontologies | You already know which ontology every document belongs to |

**Selection.** The model sees each ontology's id, prefix and description, and
picks one or answers that none fits. Guide its choice with an
`ontology_selection_user_instruction`; see [Writing user
instructions](user_instructions.md).

**Fixed.** The id may be the ontology IRI, its `ontology_id`, or the prefix its
Turtle binds to its namespace. A non-empty id selects fixed mode whatever
`ONTOLOGY_CONTEXT_MODE` says, on a request as well as in the environment, so
send an empty id to use another mode. `ontocast serve` and `ontocast process`
refuse to start in fixed mode without an id, and a request in fixed mode
without one gets `400`. An id that matches no catalog ontology is not an
error: the unit renders against an empty context, and the server log names the
id that matched nothing.

**Vector retrieval.** Only this mode runs the consistency critic, which
reports new terms that resemble terms in other catalog ontologies in
`metadata.improvement_suggestions`.

## Set up vector retrieval {#selected_vector_search_ontology}

Vector retrieval needs one backend, an embedding model and the mode:

1. **Backend.** For an embedded store, install the `lancedb` extra and set
   [`LANCEDB_ENABLED=true`](../reference/configuration/embeddings.md#lancedb_enabled);
   tables live under
   [`LANCEDB_DATA_DIR`](../reference/configuration/embeddings.md#lancedb_data_dir).
   For a Qdrant server, install the `qdrant` extra and set
   [`QDRANT_URI`](../reference/configuration/embeddings.md#qdrant_uri) and, if
   it needs one,
   [`QDRANT_API_KEY`](../reference/configuration/embeddings.md#qdrant_api_key).
   Configure one or the other; with both, OntoCast refuses to start.
2. **Embedding model.** The default
   [`EMBEDDING_PROVIDER`](../reference/configuration/embeddings.md#embedding_provider)
   runs a local Hugging Face model, which needs `sentence-transformers` (the
   `doc-processing` extra installs it); `openai` and `ollama` call a service
   instead. Set
   [`EMBEDDING_DIMENSION`](../reference/configuration/embeddings.md#embedding_dimension)
   to the model's output size. Models trained with a query or passage
   instruction need
   [`EMBEDDING_QUERY_PREFIX`](../reference/configuration/embeddings.md#embedding_query_prefix)
   and
   [`EMBEDDING_DOCUMENT_PREFIX`](../reference/configuration/embeddings.md#embedding_document_prefix).
   [Performance tuning](performance.md) covers local models and their memory.
3. **Mode.** Set `ONTOLOGY_CONTEXT_MODE=selected_vector_search_ontology`.

```bash
pip install "ontocast[server,openai,doc-processing,lancedb]"
export LANCEDB_ENABLED=true
export ONTOLOGY_CONTEXT_MODE=selected_vector_search_ontology
ontocast serve --ontology-dir ./my-ontologies
```

At every startup OntoCast indexes each catalog ontology into the vector
partition of the active tenant and project, and removes indexed ontologies
missing from the catalog
([`VECTOR_STORE_PRUNE_ORPHAN_IRIS_ON_INIT`](../reference/configuration/ontology-retrieval.md#vector_store_prune_orphan_iris_on_init)).
An uploaded ontology is indexed on upload. If the vector store cannot be
reached, a request in this mode gets `409` with `error_code:
VECTOR_STORE_UNAVAILABLE`.

### When to rebuild the index

The index records the settings that shaped it. When they differ from the
configuration, startup fails with `EmbeddingContractMismatchError`. Rebuild
the partition by starting once with `--wipe-vector-store`, or with
[`VECTOR_STORE_WIPE_ON_INIT=true`](../reference/configuration/ontology-retrieval.md#vector_store_wipe_on_init):

```bash
ontocast serve --wipe-vector-store
```

The wipe runs in every mode, but only vector mode refills the partition, so
wipe with `ONTOLOGY_CONTEXT_MODE=selected_vector_search_ontology` set.

A rebuild is needed after changing the embedding provider, model, dimension or
prefixes, the BM25 model, or the settings that decide what is indexed; [How
ontology retrieval works](../internals/retrieval.md#what-needs-a-reindex) lists
them. Retrieval settings such as `VECTOR_STORE_TOP_K` apply at query time and
need no rebuild.

## Seeding the catalog

Put Turtle (`.ttl`) files in one directory and pass it as `--ontology-dir`, or
set
[`ONTOCAST_ONTOLOGY_DIRECTORY`](../reference/configuration/storage.md#ontocast_ontology_directory).
OntoCast reads the files directly in it, not in subdirectories, at startup.
To change the catalog of a running server, use the HTTP API:

```bash
curl -X POST http://127.0.0.1:8999/ontologies -F "file=@my-ontology.ttl"
curl -X PUT "http://127.0.0.1:8999/ontologies/<url-encoded IRI>" -F "file=@my-ontology.ttl"
curl -X DELETE "http://127.0.0.1:8999/ontologies/<url-encoded IRI>"
```

Each file needs an `owl:Ontology` declaration whose subject is the ontology
IRI. Give that declaration a title (`rdfs:label` or `dcterms:title`) and a
description (`dcterms:description` or `rdfs:comment`): selection mode chooses
by them, and for an ontology without them OntoCast asks the model for both at
startup.

Uploads are never written to the directory. With the in-memory triple store
they are gone when the process stops. With Fuseki, the stored version of an
ontology wins over a file with the same IRI at startup, so change a stored
ontology with `PUT` rather than by editing its file. Each tenant and project
has its own catalog; see [Tenancy](tenancy.md).

## Context size and scope

Every mode is capped by
[`ONTOLOGY_CONTEXT_MAX_TRIPLES`](../reference/configuration/pipeline.md#ontology_context_max_triples);
[Keep the prompt in bounds](configuration.md#keep-the-prompt-in-bounds)
explains what is dropped first.

[`ONTOLOGY_CONTEXT_SCOPE`](../reference/configuration/pipeline.md#ontology_context_scope)
decides whether facts units that resolve their own context each get their own
(`unit`, the default) or all get the union of every unit's context
(`document`). A shared context is larger, but identical across the document,
so a provider's prompt cache can serve every call after the first. It matters
only where facts units resolve their own context, under `RENDER_MODE=facts`;
after an ontology stage they share one already. See [Performance
tuning](performance.md).

## Retrieval settings worth tuning

In vector mode, try these three first; [How ontology
retrieval works](../internals/retrieval.md) explains the rest.

| Setting | Raise it to | Cost |
|---|---|---|
| [`VECTOR_STORE_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES`](../reference/configuration/ontology-retrieval.md#vector_store_induced_subgraph_max_total_triples) | Show more of the neighborhood around each retrieved term. It caps the retrieved context, so while it binds the other two change little | Prompt size on every call |
| [`ONTOLOGY_PATCH_MAX_ATOMS_BASE`](../reference/configuration/ontology-retrieval.md#ontology_patch_max_atoms_base), [`ONTOLOGY_PATCH_MAX_ATOMS`](../reference/configuration/ontology-retrieval.md#ontology_patch_max_atoms) | Keep more retrieved terms per unit. Raise the base first: the hard cap only matters above it | Prompt size, once the triple cap is not binding |
| [`VECTOR_STORE_TOP_K`](../reference/configuration/ontology-retrieval.md#vector_store_top_k) | Search deeper for candidates. The number kept stays the same; they are chosen from a wider field | Search time, not prompt size |

## See what a unit was shown {#diagnostics}

The `/process` response carries `metadata.retrieval_metrics`, and a batch run
writes the same object into its run manifest. The keys to read:

| Key | What it tells you |
|---|---|
| `ontology_context_mode` | The mode the run used |
| `ontology_snapshot_triples` | How many triples of ontology the unit was shown, in every mode |
| `empty_snapshot_reason` | Why a unit's context came back empty (vector mode) |
| `patch_retrieval` | Vector mode: how many queries ran, how many candidates were found and how many a threshold rejected, the terms kept per ontology (`seeds_by_ontology`), and the final context size |

These values are recorded per unit and merged, so on a multi-unit document
the last unit's values win. To see one unit's context, send its text to
`POST /process_unit`. The server log also has one `Patch retrieval:` line per
unit at `INFO` level.

To find out why an ontology contributed nothing, set
[`ONTOLOGY_PATCH_DUMP_ONTOLOGY_RANKS=true`](../reference/configuration/ontology-retrieval.md#ontology_patch_dump_ontology_ranks).
`patch_retrieval.ontology_rank_diagnostics` then shows, per ontology, where
its best term ranked in each retrieval lane, its rank after fusion, and
whether it survived the cut. [Reading run telemetry](telemetry.md) covers the
other metrics.

## What to read next

- [How ontology retrieval works](../internals/retrieval.md): the mechanism behind vector mode.
- [Ontology retrieval settings](../reference/configuration/ontology-retrieval.md) and [Embeddings and vector stores](../reference/configuration/embeddings.md): every setting.
- [Tenancy](tenancy.md): how catalogs and vector partitions are named per tenant and project.
