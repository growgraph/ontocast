---
search:
  boost: 3
---

# Configuring OntoCast

OntoCast has a few hundred settings, and almost all of them have defaults you
can keep. This page covers the ones that change what a run does, in the order
you usually decide them. The [configuration
reference](../reference/configuration/index.md) lists every setting with its
default.

## How settings are read

Every setting is an environment variable, such as `LLM_PROVIDER` or
`RENDER_MODE`; case does not matter. Booleans are `true` or `false`; lists and
mappings are written as JSON.

OntoCast does not read a settings file itself. To keep your settings in one,
start from `.env.example.minimal` (the settings worth a decision) or
`.env.example` (all of them) in the repository, and load it into the
environment of the command you run:

```bash
set -a; source .env; set +a
ontocast serve
```

The server reads its settings once, at startup. A few can also be set per
request on `/process`: `render_mode`, `ontology_context_mode`,
`ontology_context_fixed_ontology_id`, `llm_graph_format` and `max_visits`. A
request value wins over the environment, and an unrecognised value is rejected
with `400` rather than replaced by the default. See the [HTTP
API](../reference/http_api.md).

When you embed OntoCast in Python, `Config()` reads the same variables, and
`Config.in_memory()` builds a configuration that needs no external services;
see [Embedding OntoCast in your agent](embedding.md).

## Choose the model

| Setting | What to set |
|---|---|
| [`LLM_PROVIDER`](../reference/configuration/llm.md#llm_provider) | `openai`, `anthropic`, `google` or `ollama`, with the matching install extra |
| [`LLM_MODEL_NAME`](../reference/configuration/llm.md#llm_model_name) | Any model name the provider accepts. Names OntoCast does not know are passed through with a warning |
| [`LLM_API_KEY`](../reference/configuration/llm.md#llm_api_key) | The key for OpenAI, Anthropic or Google. OntoCast reads only this variable, not the providers' own ones |
| [`LLM_BASE_URL`](../reference/configuration/llm.md#llm_base_url) | The Ollama server, or the endpoint of any OpenAI-compatible service (with `LLM_PROVIDER=openai`) |

Leave [`LLM_TEMPERATURE`](../reference/configuration/llm.md#llm_temperature) at
`0`: extraction should be repeatable. Responses are cached on disk, so a
document you run twice with the same prompts costs nothing the second time; see
[LLM caching](llm_caching.md).

For reasoning models,
[`LLM_REASONING_EFFORT`](../reference/configuration/llm.md#llm_reasoning_effort)
sets how much the model thinks before answering (OpenAI, Gemini 3 and later);
reasoning tokens are billed as output. For local models through Ollama, raise
[`LLM_NUM_CTX`](../reference/configuration/llm.md#llm_num_ctx): Ollama's default
context window is too small for most OntoCast prompts.

## Choose what a run produces

[`RENDER_MODE`](../reference/configuration/pipeline.md#render_mode) decides
which half of the pipeline runs:

| Value | What runs | Use it to |
|---|---|---|
| `ontology_and_facts` (default) | Ontology, then facts | Build an ontology and extract facts in its terms |
| `ontology` | Ontology only; no facts are written | Build or extend an ontology from a set of documents |
| `facts` | Facts only, against the existing catalog; no new terms are added | Extract facts in terms of an ontology you have settled |

!!! warning "Facts-only runs need a catalog"
    With `RENDER_MODE=facts`, nothing creates ontology terms, so an empty
    catalog leaves the model nothing to extract against; OntoCast warns at
    startup. Seed the catalog first, or run `ontology_and_facts` once. To make
    an empty catalog stop the run instead, set
    [`ONTOLOGY_CONTEXT_REQUIRED=true`](../reference/configuration/pipeline.md#ontology_context_required).

## Choose where each part's ontology comes from

OntoCast cuts a document into parts (content units) and shows the model, for
each part, the slice of your ontologies it needs.
[`ONTOLOGY_CONTEXT_MODE`](../reference/configuration/pipeline.md#ontology_context_mode)
decides how that slice is chosen:

| Value | How the ontology is chosen | Needs |
|---|---|---|
| `selected_single_ontology` (default) | The model picks one ontology from the catalog for each part | Nothing extra; one more LLM call per part |
| `selected_vector_search_ontology` | Retrieval picks the relevant terms from all ontologies | A vector store: LanceDB (`lancedb` extra) or Qdrant (`qdrant` extra) |
| `fixed_single_ontology` | The same ontology for every part | [`ONTOLOGY_CONTEXT_FIXED_ONTOLOGY_ID`](../reference/configuration/pipeline.md#ontology_context_fixed_ontology_id): its IRI, id or prefix |

For fixed mode, set both `ONTOLOGY_CONTEXT_MODE` and
`ONTOLOGY_CONTEXT_FIXED_ONTOLOGY_ID`. A server request in fixed mode that names
no ontology uses the configured one. A request can also pass
`ontology_context_fixed_ontology_id` itself, which selects fixed mode whatever
`ontology_context_mode` says.

Retrieval is the right choice once the catalog holds more ontologies than a
model can sensibly choose between; [Choosing ontology
context](ontology_context.md) explains the trade-offs.

## Give it your ontologies and shapes

| Setting | What it does |
|---|---|
| [`ONTOCAST_ONTOLOGY_DIRECTORY`](../reference/configuration/storage.md#ontocast_ontology_directory) | Turtle files loaded into the catalog at startup. Same as `--ontology-dir` |
| [`FACTS_SHAPES_DIR`](../reference/configuration/facts-validation.md#facts_shapes_dir) | SHACL shapes that extracted facts are validated against (`shacl` extra). Same as `--shapes-dir` |
| [`CURRENT_DOMAIN`](../reference/configuration/pipeline.md#current_domain) | The base of the document IRIs OntoCast creates (`<domain>/doc/<hash>`). Set it once per project: it ends up in every graph you write. Extracted entities always use the fixed facts namespace `cd:`; see [Ontologies and facts](../concepts/ontologies_and_facts.md) |

Ontologies and shapes can also be uploaded to a running server; see the [HTTP
API](../reference/http_api.md).

## Keep the prompt in bounds

Each prompt carries the part's slice of the ontology.
[`ONTOLOGY_CONTEXT_MAX_TRIPLES`](../reference/configuration/pipeline.md#ontology_context_max_triples)
caps it, in every mode. Over the cap, OntoCast drops the least useful triples
first (header metadata, then redundant structure, then comments and
definitions) and never drops labels, types, hierarchy or domain and range. A
warning that the ontology still does not fit means the catalog is too large to
show whole: split it, or switch to `selected_vector_search_ontology`.

The format of the ontology in the prompt matters as much as its size; see
[Performance tuning](performance.md).

## Balance quality and cost

Each part of a document costs at least one LLM call. These settings add calls
in exchange for quality:

| Setting | What one more costs and buys |
|---|---|
| [`FACTS_CRITIC_PASSES`](../reference/configuration/facts-validation.md#facts_critic_passes) | One call per part: a review of the extracted facts, with fixes applied as a patch |
| [`ONTOLOGY_CRITIC_PASSES`](../reference/configuration/ontology-validation.md#ontology_critic_passes) | The same for ontology changes; off by default |
| [`FACTS_COMPLETION_PASSES`](../reference/configuration/facts-validation.md#facts_completion_passes) | One call per part that still misses measurements stated in the text; off by default |
| [`MAX_VISITS_PER_NODE`](../reference/configuration/pipeline.md#max_visits_per_node) | Retries of an extraction that failed outright. A successful extraction is never repeated |

Two settings decide how fast a document goes through:
[`PARALLEL_WORKERS`](../reference/configuration/pipeline.md#parallel_workers)
is how many parts of one document are processed at once, and
[`LLM_MAX_INFLIGHT`](../reference/configuration/llm.md#llm_max_inflight) caps the
calls in flight across all documents. When the provider rate-limits you, lower
`LLM_MAX_INFLIGHT`.

## Choose where graphs are stored

By default graphs live in memory and are gone when the process ends. To keep
them in Apache Jena Fuseki, set
[`FUSEKI_URI`](../reference/configuration/storage.md#fuseki_uri), and
[`FUSEKI_AUTH`](../reference/configuration/storage.md#fuseki_auth) if the server
requires credentials. Datasets are named per tenant and project; see
[Triple stores](triple_stores.md) and [Tenancy](tenancy.md).

The server listens on
[`HOST`](../reference/configuration/pipeline.md#host) and
[`PORT`](../reference/configuration/pipeline.md#port), `127.0.0.1:8999` by
default. It has no authentication, so keep it on the loopback interface unless
a proxy that authenticates sits in front of it.

## Prepare the documents

| Setting | When to change it |
|---|---|
| [`CONVERTER_PROFILE`](../reference/configuration/conversion.md#converter_profile) | `born_digital` for PDFs with selectable text, such as publisher PDFs |
| [`CHUNK_MIN_SIZE`](../reference/configuration/chunking.md#chunk_min_size), [`CHUNK_MAX_SIZE`](../reference/configuration/chunking.md#chunk_max_size) | Size of each part in characters: larger parts give the model more context per call |
| [`CHUNK_BIBLIOGRAPHY_MODE`](../reference/configuration/chunking.md#chunk_bibliography_mode) | Reference lists are skipped by default; `citations_only` extracts them as bibliographic records |

To process only some sections of a document, such as the methods and results
of a paper, pass `target_sections` or `exclude_sections` with the request; see
the [HTTP API](../reference/http_api.md).

## What to read next

- [Recipes](recipes.md): complete settings for evaluating, building an ontology, extracting facts, and serving.
- [Configuration reference](../reference/configuration/index.md): every setting.
