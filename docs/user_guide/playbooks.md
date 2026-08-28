---
search:
  boost: 3
---

# Configuration Playbooks

OntoCast exposes around 200 environment variables. Roughly 45 of them are worth
a decision; the rest have defaults that are fine until you have a measured
reason to move them, and about a third are inert at their defaults anyway (the
whole `WEB_SEARCH_*` block does nothing while `WEB_SEARCH_ENABLED=false`).

Start from [`.env.example.minimal`](https://github.com/growgraph/ontocast/blob/main/.env.example.minimal),
not `.env.example`. Then pick the playbook that matches what you are doing.

| You want to | Playbook |
|---|---|
| See whether this works on your documents at all | [Evaluate](#1-evaluate) |
| Build a schema from a corpus that has none | [Build an ontology](#2-build-an-ontology) |
| Extract instances against a schema you have settled | [Populate facts](#3-populate-facts) |
| Work against a large or heterogeneous ontology catalog | [Scale the catalog](#4-scale-the-catalog) |
| Run it as a shared service | [Serve it](#5-serve-it) |

Each playbook lists only what it *changes* from the minimal file.

---

## 1. Evaluate

Smallest possible setup: no triple store, no vector store, no indexing step.
Everything runs in memory and is gone when the process exits — which is what you
want while you are still deciding whether the output is any good.

```bash
CURRENT_DOMAIN=https://example.com
ONTOCAST_ONTOLOGY_DIRECTORY=./my-ontologies   # even 2-3 seed .ttl files help a lot
LLM_API_KEY=...
# everything else: defaults
```

```bash
ontocast process --input-path ./doc.pdf --head-chunks 5 --output-dir ./out
```

`--head-chunks 5` caps the run at five content units. Use it. A 60-page PDF is
~60 LLM calls at defaults, and you do not need 60 to see whether the extraction
is sane.

Read `out/<stem>.run.json` before reading the graphs — it records the effective
configuration and the budget (calls, cache hits, triples), which is how you tell
"the model did badly" from "the pipeline was configured to skip that stage".

!!! tip "Repeat runs are nearly free"

    LLM responses are cached on disk (`~/.cache/ontocast/llm/`), so re-running
    the same document after changing a *downstream* setting costs almost
    nothing. Changing the prompt, the model, or the wire format invalidates it.

**Next:** if the schema is the problem, go to [Build an ontology](#2-build-an-ontology).
If the schema is right but instances are wrong, go to [Populate facts](#3-populate-facts).

---

## 2. Build an ontology

You have documents and little or no schema. Run the ontology half alone, look at
what comes out, iterate on the seed catalog.

```bash
RENDER_MODE=ontology              # writes NO facts — this is the point
MAX_VISITS_PER_NODE=2             # enable the LLM critic; schema quality is the deliverable
ONTOLOGY_CONTEXT_MODE=selected_single_ontology
```

Why these:

- **`RENDER_MODE=ontology`** skips the entire facts block, so you are not paying
  to instantiate against a schema you are still changing. Nothing is written to
  the facts partition at all.
- **`MAX_VISITS_PER_NODE=2`** turns on `criticise_ontology`, which is off at the
  default of `1`. This is the one place the extra cost is usually justified —
  a bad term propagates into every downstream fact. It costs roughly one extra
  call per unit, not the squared worst case the nested loops suggest, because
  the critic exits immediately when it fails without requesting external
  evidence. See [Configuration](configuration.md#server).

Iterate: run, inspect the emitted ontology, fold the good parts back into
`ONTOCAST_ONTOLOGY_DIRECTORY`, run again. The catalog is the input that most
improves the next run.

**Next:** once the schema stops changing, switch to [Populate facts](#3-populate-facts).

---

## 3. Populate facts

The schema is settled. You want instances, and you want them to use the terms
you already have rather than inventing near-duplicates.

```bash
RENDER_MODE=facts
ONTOLOGY_CONTEXT_MODE=fixed_single_ontology
ONTOLOGY_CONTEXT_FIXED_ONTOLOGY_ID=my_schema   # IRI, ontology_id, or author prefix
FACTS_SHACL_AUTOFIX=prune
AGG_CANDIDATE_SIMILARITY_THRESHOLD=0.70
```

Why these:

- **`RENDER_MODE=facts`** skips the ontology block entirely, so no new terms are
  minted. Extraction depends **wholly** on the existing catalog.
- **`fixed_single_ontology`** removes the per-unit selection call, which is both
  cheaper and more consistent than letting the model re-pick a schema for every
  chunk.

!!! warning "Two failure modes that look identical to bad extraction"

    **An empty catalog.** `RENDER_MODE=facts` creates no ontology terms, so if
    the catalog is empty or badly matched the renderer has nothing to
    instantiate against and returns almost nothing. Seed it first.

    **A fixed id that matches nothing.** A *missing* id is a clean 400. An id
    that matches no catalog entry is **not** an error — it logs a warning and
    renders against an empty snapshot. Check `GET /ontologies` if output goes
    sparse after a rename.

Tune `AGG_CANDIDATE_SIMILARITY_THRESHOLD` on what you actually see: raise it
when distinct entities are being merged, lower it when duplicates survive
(`AGG_SIMILARITY_THRESHOLD` only affects the cross-graph aligner). Details in
[Entity Disambiguation](aggregation.md), and the validation gate is described in
[Facts Validation](validation.md).

---

## 4. Scale the catalog

Catalog selection asks the LLM to pick one ontology per unit. That works while
the catalog is small and the ontologies are clearly distinct. Once it is large
or overlapping, the selection call becomes both the cost and the error, and
retrieval does better.

**Switch when:** more than roughly a dozen ontologies, heavy vocabulary overlap
between them, or units that legitimately need terms from several at once —
catalog selection returns exactly one, so cross-ontology documents are where it
visibly fails.

```bash
ONTOLOGY_CONTEXT_MODE=selected_vector_search_ontology

LANCEDB_ENABLED=true                 # embedded; or QDRANT_URI for a server. Never both.
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DIMENSION=384

VECTOR_STORE_TOP_K=20
VECTOR_STORE_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES=1200
```

This mode is also the **only** one that runs the consistency critic — switching
away from it silently drops that check.

Costs you are taking on: an embedding model resident in memory, an indexing pass
over the catalog, and an index that must be rebuilt whenever the embedding
contract changes. `EMBEDDING_MODEL_NAME`, `EMBEDDING_DIMENSION` and the query
and document prefixes are all part of that contract; a mismatch fails loudly
with `EmbeddingContractMismatchError` rather than quietly degrading recall.
Reindex with `--wipe-vector-store`.

Tune in this order, one at a time:

1. `VECTOR_STORE_TOP_K` — more candidates per query window.
2. `VECTOR_STORE_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES` — more schema per unit.
3. `ONTOLOGY_PATCH_MAX_ATOMS` — **lower** it for noisy catalogs, where the
   problem is irrelevant terms crowding out correct ones.

!!! warning "The retrieval defaults are fitted to one corpus"

    They are a starting point, not a recommendation. The full budget surface is
    in [Ontology Context](ontology_context.md) and
    [Configuration](configuration.md#ontology-patch-retrieval); change one knob
    per run and measure, because several of them interact.

---

## 5. Serve it

```bash
FUSEKI_URI=http://localhost:3030
FUSEKI_AUTH=admin/admin

HOST=0.0.0.0                      # ONLY behind an authenticating proxy — see below
MAX_CONCURRENT_PROCESSES=4
LLM_MAX_INFLIGHT=16
LOGGING_LEVEL=info
```

!!! danger "There is no authentication on any route"

    No auth, no authorization, no request-size limit. `POST /flush` is
    destructive and can target any tenancy partition by query parameter. `HOST`
    defaults to loopback for this reason; the server logs a warning when you
    widen it. Put a proxy in front or leave it on `127.0.0.1`.

Callers pass `tenant` and `project` per request, which partition both the Fuseki
datasets and the vector collections — see [Tenancy](tenancy.md). Per-request
parameters (`render_mode`, `ontology_context_mode`, `llm_graph_format`,
`max_visits`) override the environment for a single call, so one server can
serve several of the playbooks above without a restart. Precedence and the
400-on-typo contract are in [API Endpoints](api.md).

`MAX_CONCURRENT_PROCESSES` **queues** requests beyond the limit rather than
rejecting them.

---

## Cross-cutting: local models and memory

Independent of which playbook you are on. Three settings each name a local
sentence-transformer, and they **default to two different checkpoints**, so a
default run holds two resident models:

| Setting | Used by | Default |
|---|---|---|
| `CHUNK_EMBEDDING_MODEL` | semantic chunking, schema detection | `…/paraphrase-multilingual-mpnet-base-v2` (~1.1 GB) |
| `EMBEDDING_MODEL_NAME` | dense retrieval | `…/paraphrase-multilingual-MiniLM-L12-v2` (~458 MB) |
| `AGG_EMBEDDING_MODEL` | entity disambiguation | same MiniLM as above |

Setting all three to one string drops it to a single resident model — ~650 MB of
peak RSS, measured. The cache key is the **literal string**, so the spellings
must match character for character: `paraphrase-multilingual-MiniLM-L12-v2` and
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` resolve to the
same files on the hub and still load twice.

```bash
CHUNK_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
AGG_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

!!! warning "Changing `CHUNK_EMBEDDING_MODEL` is not a free win"

    It invalidates the on-disk chunk cache, shifts chunk boundaries — which
    changes what each unit extracts — and moves the
    `CHUNK_SECTION_SCHEMA_DETECT_*` thresholds, which are calibrated against the
    default model's score distribution. Worth it on a memory-constrained box;
    re-check extraction quality after. Full numbers in
    [Performance](performance.md#local-embedding-models).

Only `AGG_EMBEDDING_MODEL` matters in every configuration.
`EMBEDDING_MODEL_NAME` is read only under vector retrieval, and
`CHUNK_EMBEDDING_MODEL` only under `CHUNK_SEGMENTER=semantic` (the default).

## When output is wrong

Read the run manifest first. It records the effective configuration, so it
separates "the model did badly" from "that stage never ran".

| Symptom | Likely cause | Look at |
|---|---|---|
| No facts at all | `RENDER_MODE=ontology` | [Render Mode](configuration.md#render-mode-render_mode) |
| Facts are sparse, schema looks fine | Empty/mismatched catalog under `RENDER_MODE=facts`, or a fixed id matching nothing | [Populate facts](#3-populate-facts) |
| Schema invented instead of reused | Catalog not seeded, or the wrong ontology selected per unit | `ONTOCAST_ONTOLOGY_DIRECTORY`, [Ontology Context](ontology_context.md) |
| Right ontology, wrong terms within it | Not enough schema in context | `VECTOR_STORE_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES`, or `ONTOLOGY_PATCH_MAX_ATOMS` if the catalog is noisy |
| Structurally malformed output | Critic disabled | `MAX_VISITS_PER_NODE=2` |
| Duplicate entities across chunks | Disambiguation too strict | `AGG_CANDIDATE_SIMILARITY_THRESHOLD` lower — [Aggregation](aggregation.md) |
| Distinct entities merged | Disambiguation too loose | `AGG_CANDIDATE_SIMILARITY_THRESHOLD` higher, and check the guard flags (`AGG_LITERAL_CONFLICT_GUARD`, `AGG_INITIALS_DISTINCT_GUARD`) are on |
| Vector mode returns nothing | Index empty, or scores under the floor | `EMBEDDING_*` contract, `--wipe-vector-store`, [Observability](observability.md) |
| Consistency critic never reports | It only runs in vector mode | [Ontology Context](ontology_context.md#selected_vector_search_ontology) |
| Slower than expected | Provider concurrency, not worker count | `LLM_MAX_INFLIGHT` — [Performance](performance.md) |
| Memory higher than expected | Two resident encoders at defaults | [Local models and memory](#cross-cutting-local-models-and-memory) |
| Prompt too large / context-limit errors | Catalog-selection inlines the whole ontology | `ONTOLOGY_CONTEXT_MAX_TRIPLES`; `LLM_GRAPH_FORMAT=turtle` halves chars/triple; or vector mode |
| Log says "still exceeds the prompt budget after condensing" | Catalog too big to fit without cutting schema | Split the catalog or switch to vector mode — raising the budget defeats the point |
| Chunks split mid-argument, or are too coarse | Chunk size bounds, or the segmenter | `CHUNK_MIN_SIZE` / `CHUNK_MAX_SIZE`, `CHUNK_SEGMENTER` |
| Bibliography extracted as domain facts | Reference routing | `CHUNK_BIBLIOGRAPHY_MODE=skip` |
| Ligature gaps in PDF text (`di ff usion`) | Converter profile | `CONVERTER_PROFILE=born_digital` |

## Where the other ~175 variables are

[Configuration System](configuration.md) is the complete reference, and
`.env.example` documents every variable with its default. Areas deliberately
left out of the minimal file, each with a page of its own: chunking and section
detection, the Docling converter, LLM caching, web-search grounding, SHACL
validation detail, and the advanced retrieval-fusion knobs.
