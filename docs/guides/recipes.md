---
search:
  boost: 3
---

# Recipes

Each recipe here is a complete starting configuration for one job: trying
OntoCast on a document, building an ontology, extracting facts against one you
trust, scaling to a large catalog, and running a shared server. Pick the one
that matches what you are doing.

| You want to | Recipe |
|---|---|
| See whether OntoCast works on your documents | [Evaluate on one document](#evaluate-on-one-document) |
| Build an ontology from documents that have none | [Build an ontology](#build-an-ontology) |
| Extract facts in terms of an ontology you have settled | [Populate facts](#populate-facts-against-a-settled-catalog) |
| Work against many or overlapping ontologies | [Scale to a large catalog](#scale-to-a-large-catalog) |
| Run OntoCast as a service for other applications | [Serve it](#serve-it) |

Each settings block lists only what differs from the defaults. Start from
`.env.example.minimal`, add the block, and load it into the environment as
[Configuring OntoCast](configuration.md#how-settings-are-read) shows. Every
recipe also needs your model settings (`LLM_PROVIDER`, `LLM_API_KEY`, ...).

## Evaluate on one document

Use this first. Everything stays in memory, nothing needs to be indexed, and
you see within one short run whether the output is worth tuning.

```bash
CURRENT_DOMAIN=https://example.com/kg          # base of every IRI the run creates
ONTOCAST_ONTOLOGY_DIRECTORY=./my-ontologies    # optional: a few seed ontologies help
```

```bash
ontocast process --input-path ./paper.pdf --head-chunks 3 --output-dir ./out
```

`--head-chunks 3` stops after the first three parts of the document, which is
enough to judge the output and costs a fraction of the whole document.

**What to check.** Open `out/paper.run.json` (the run manifest) before the
graphs. It records the settings the run actually used and its `budget`: LLM
calls, cache hits and tokens. That tells "the model did badly" apart from "that
stage never ran". Then read `paper.facts.ttl`, the
`paper.<ontology>.ontology.ttl` files, and `paper.facts.validation.json` for
validation findings and parts that failed.

LLM responses are cached on disk, so running the same document again after a
change that does not alter the prompts costs nothing; see [LLM
caching](llm_caching.md).

If the ontology is the problem, go on to [Build an
ontology](#build-an-ontology). If the ontology is right but the facts are
wrong, go to [Populate facts](#populate-facts-against-a-settled-catalog).

## Build an ontology

You have documents and little or no ontology. Run the ontology half alone,
inspect what it produces, and fold the good parts back into your seed
directory.

```bash
RENDER_MODE=ontology          # build ontologies only; no facts are extracted
ONTOLOGY_CRITIC_PASSES=1      # one review call per part; the ontology is the deliverable
```

The ontology critic is off by default. Here it earns its call: a wrong term
carries into every fact extracted with it later. Each pass applies its fixes as
a patch, and only to the part's own changes, so it cannot delete terms that
came from your catalog.

Starting with no ontology at all is supported. When a part's ontology context
is empty, OntoCast creates a new ontology from the text, under
`CURRENT_DOMAIN`. To declare "no seed ontologies" for one run, even when
`ONTOCAST_ONTOLOGY_DIRECTORY` is set, pass an empty directory:

```bash
ontocast process --input-path ./docs --ontology-dir '' --ontology-output-dir ./onto
```

A directory that does not exist is an error, not an empty catalog:
`ontocast process` refuses to start.

**What to check.** The `*.ontology.ttl` files in the output directory, and the
`ontology_critic` block of each run manifest: how many reviews ran and how many
fixes they applied. Iterate until a run against your updated seed directory
stops adding terms you would reject, then switch to populating facts.

## Populate facts against a settled catalog

The ontology is settled. You want facts that use its terms rather than
near-duplicates of them.

```bash
RENDER_MODE=facts                                  # no new ontology terms are created
ONTOLOGY_CONTEXT_MODE=fixed_single_ontology        # the same ontology for every part
ONTOLOGY_CONTEXT_FIXED_ONTOLOGY_ID=my_schema       # its IRI, ontology id or prefix
ONTOLOGY_CONTEXT_REQUIRED=true                     # stop instead of extracting with no ontology
```

Fixing the ontology removes the per-part selection call and shows every part
the same ontology, so facts from different parts use the same terms. On the
server, pass `ontology_context_fixed_ontology_id` with each request; a request
that names an ontology this way runs in fixed mode. If your catalog has several
ontologies that parts need in different combinations, use [vector
retrieval](#scale-to-a-large-catalog) instead.

`ONTOLOGY_CONTEXT_REQUIRED=true` matters here. In a facts-only run nothing
creates ontology terms, so an empty catalog or a fixed id that matches no
ontology leaves the model with nothing to extract against. By default
OntoCast logs a warning and extracts in generic vocabulary; with this setting
`ontocast process` refuses to start on an empty catalog, and a part with no
ontology context stops the run.

On a facts-only run the prompt shows the ontology as a term sheet, one line per
term, which is the shortest form; see [Performance tuning](performance.md).

**What to check.** In the run manifest, `validation_config.context_from_units`
should be `true`, and `retrieval_metrics.ontology_snapshot_triples` should be
well above zero. Read `paper.facts.validation.json` for findings; if distinct
entities were merged or duplicates survived, see
[Troubleshooting](troubleshooting.md#duplicate-entities-or-wrong-merges). To
validate against SHACL shapes, add `FACTS_SHAPES_DIR`; see [Validation and
SHACL](validation.md).

## Scale to a large catalog

By default the model picks one ontology from the catalog for each part. That
breaks down when the catalog holds many ontologies, when their vocabularies
overlap, or when a part needs terms from several ontologies at once. Vector
retrieval then selects the relevant terms from all of them.

```bash
ONTOLOGY_CONTEXT_MODE=selected_vector_search_ontology   # retrieve terms instead of picking one ontology
LANCEDB_ENABLED=true                                    # embedded index on disk (lancedb extra)
```

For a Qdrant server, install the `qdrant` extra and set `QDRANT_URI` instead of
`LANCEDB_ENABLED`. The default embedding model runs locally and needs
`sentence-transformers` (in the `doc-processing` extra); set
`EMBEDDING_PROVIDER` to `openai` or `ollama` to embed through an API instead.

The catalog is indexed at startup. The embedding settings (`EMBEDDING_*`) are
recorded with the index, and a mismatch fails at startup rather than degrading
retrieval: after changing them, rebuild the index with `--wipe-vector-store`.
This mode also runs the consistency critic, which checks the document's
ontology against what retrieval found; the other modes skip it.

If parts get the wrong terms, change one setting per run, in this order:

1. [`VECTOR_STORE_TOP_K`](../reference/configuration/ontology-retrieval.md#vector_store_top_k): more candidates per query.
2. [`VECTOR_STORE_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES`](../reference/configuration/ontology-retrieval.md#vector_store_induced_subgraph_max_total_triples): more of the ontology around each candidate.
3. [`ONTOLOGY_PATCH_MAX_ATOMS`](../reference/configuration/ontology-retrieval.md#ontology_patch_max_atoms): lower it when irrelevant terms crowd out the right ones.

[How retrieval works](../internals/retrieval.md) explains what each one does.

Retrieval gives every part its own slice of the ontology, so no two prompts
share a prefix that a provider's prompt cache could serve.
`ONTOLOGY_CONTEXT_SCOPE=document` with `FANOUT_WARMUP_UNITS=1` makes the slice
the same for every part of a document; [Performance
tuning](performance.md) explains when that pays.

**What to check.** `retrieval_metrics.patch_retrieval` in the response or run
manifest shows what retrieval selected for each part. An empty selection means
the index is empty or does not match the catalog; see
[Troubleshooting](troubleshooting.md#empty-ontology-context).

## Serve it

```bash
FUSEKI_URI=http://localhost:3032     # keep graphs in Fuseki
FUSEKI_AUTH=admin/your-password
MAX_CONCURRENT_PROCESSES=4           # documents processed at once; more requests wait
```

`LLM_MAX_INFLIGHT` caps the provider calls in flight across all documents;
lower it if the provider rate-limits a busy server.

```bash
ontocast serve --ontology-dir ./my-ontologies --shapes-dir ./my-shapes
```

!!! warning "The server has no authentication"
    No route checks who is calling, and `POST /flush` deletes data in any
    partition named in the request. The server binds to `127.0.0.1` by
    default. Set `HOST=0.0.0.0` only when a proxy that authenticates sits in
    front of it.

Callers choose a partition per request with `?tenant=` and `?project=`; see
[Tenancy](tenancy.md). They can also override `render_mode`,
`ontology_context_mode`, `ontology_context_fixed_ontology_id`,
`llm_graph_format` and `max_visits` per request, so one server can run several
of the recipes above. A request that switches to
`selected_vector_search_ontology` needs a server started in that mode, which is
when the index is built. See the [HTTP API](../reference/http_api.md).

**What to check.** `GET /health` for liveness and `GET /info` for the server's
capabilities and LLM cache counters. Each `/process` response carries `budget`
and `failed_units` in its `metadata`; [Reading run telemetry](telemetry.md)
explains them.

## Local models

The defaults load local sentence-transformer models for chunking and for
entity disambiguation, and a third under vector retrieval. With Ollama as the
LLM provider, raise `LLM_NUM_CTX`, since Ollama's default context window is too
small for most prompts. [Performance tuning](performance.md) explains the memory
the local models take and how to load one model instead of two.

## What to read next

- [Troubleshooting](troubleshooting.md): what to change when the output is wrong.
- [Configuring OntoCast](configuration.md): the settings behind each recipe.
