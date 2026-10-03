# LLM caching

OntoCast stores every LLM response on disk and answers an identical request
from the disk instead of the provider. Running a document again, after a change
that does not alter the prompts, costs no tokens. This page explains what makes
a request identical, where the cache lives, and how to keep it in bounds.

!!! note "Not the provider's prompt cache"
    A cache hit here never reaches the provider. A provider's prompt cache is
    a different mechanism: the request is sent and billed, with a discount on a
    repeated prompt prefix. [Performance tuning](performance.md) covers how to
    make prefixes repeat.

## What is cached

The cache directory holds one subdirectory per tool:

| Subdirectory | Holds | An entry is reused when these match |
|---|---|---|
| `llm/` | Provider responses | The full prompt text and the LLM settings listed below |
| `converter/` | Converted documents | The file's bytes, the `CONVERTER_*` settings and the profile the document resolved to |
| `chunker/` | Chunked text | The text, `CHUNK_EMBEDDING_MODEL`, `CHUNK_MIN_SIZE`, `CHUNK_MAX_SIZE`, and whether semantic chunking ran |

The LLM settings in the key are those that change the answer:
`LLM_PROVIDER`, `LLM_MODEL_NAME`, `LLM_TEMPERATURE`, `LLM_BASE_URL`, the Ollama
settings `LLM_THINK`, `LLM_NUM_PREDICT` and `LLM_NUM_CTX`, and
`LLM_REASONING_EFFORT` and `LLM_THINKING_BUDGET` when they are set. Structured
calls also key on the name of the expected output schema. Each key also holds a
format version, so an OntoCast release that changes what is stored misses old
entries instead of misreading them.

Because the whole prompt is part of the key, anything that changes the prompt
misses the cache: a different document part, a changed ontology in the
catalog, a user instruction, `LLM_GRAPH_FORMAT`, `ONTOLOGY_CHAPTER_FORMAT`, or
an OntoCast upgrade that changes prompt wording. Settings that act only after
the last LLM call, such as entity disambiguation (`AGG_*`), leave the cache
warm.
`LLM_PROMPT_CACHE_KEY` and `LLM_JSON_MODE` are not part of the key.

Each entry is a JSON file holding the prompt, the response, the provider's
response metadata and its token usage, so a hit reports the same usage as the
original call. Entries are written to a temporary file and renamed into place,
so concurrent workers never read a half-written entry.

!!! warning "Entries contain your documents"
    The cached prompts include the document text. Restrict access to the cache
    directory as you would to the documents.

## Turning it off, or reading only

| Setting | Effect |
|---|---|
| [`LLM_CACHE_ENABLED`](../reference/configuration/llm.md#llm_cache_enabled) | `false` neither reads nor writes LLM responses |
| [`LLM_CACHE_READ_ONLY`](../reference/configuration/llm.md#llm_cache_read_only) | `true` answers from the cache when it can, and sends a miss to the provider without storing the answer |

Read-only mode keeps a cache fixed, for example a cache shared by several
runs that should all see the same responses. A miss still calls the provider,
so it still needs a key.

## Where the cache lives

The first of these that applies:

1. [`ONTOCAST_CACHE_DIR`](../reference/configuration/storage.md#ontocast_cache_dir).
2. `$XDG_CACHE_HOME/ontocast` when `XDG_CACHE_HOME` is set.
3. `~/.cache/ontocast`, or `%USERPROFILE%\AppData\Local\ontocast` on Windows.

Under pytest, the cache is `.test_cache/` in the working directory, so tests
never touch your own cache. There is no command-line flag; to use another
directory for one command, set the variable for it:

```bash
ONTOCAST_CACHE_DIR=/data/ontocast-cache ontocast serve
```

## Size and eviction

The cache bounds itself. Once the directory grows past
[`ONTOCAST_CACHE_MAX_BYTES`](../reference/configuration/storage.md#ontocast_cache_max_bytes)
(a byte count, or a size such as `500MB`), the least recently used entries are
deleted until it fits. Recency is the file's access time, so an entry read
often outlives one written recently and never read. Set the ceiling to `0` to
turn eviction off.

[`ONTOCAST_CACHE_TTL_DAYS`](../reference/configuration/storage.md#ontocast_cache_ttl_days)
also deletes entries unused for that many days, before the size check. The
cache is trimmed when a process starts and again after every
[`ONTOCAST_CACHE_PRUNE_EVERY`](../reference/configuration/storage.md#ontocast_cache_prune_every)
writes, so a long-running server stays bounded.

## The `ontocast cache` commands

```bash
ontocast cache stats                                    # size per subdirectory
ontocast cache prune                                    # trim to the configured ceiling at once
ontocast cache prune --max-bytes 500000000 --ttl-days 30
ontocast cache prune --orphaned                         # remove subdirectories no tool uses
ontocast cache clear --subdir llm                       # delete every LLM response
```

`stats` marks subdirectories that no current tool writes to as orphaned;
`prune --orphaned` removes them. `clear` without `--subdir` deletes the whole
cache. It asks for confirmation; pass `--yes` in scripts.

## Checking that the cache is used

- `budget.cache_hits` in a `/process` response or run manifest counts the calls answered from the cache, next to `calls_count`.
- `GET /info` reports `llm_cache`: hits and misses since the server started, and the size on disk.

[Performance tuning](performance.md) shows how to use a warm cache to time a
change without spending tokens.

## Pre-filling the cache with the OpenAI Batch API

For a large first run, you can fill the cache through OpenAI's Batch API,
which is cheaper per token and slower to answer. The helpers in
[`ontocast.tool.llm_batch`](../reference/python/tool/llm_batch.md) write the
batch input file and import the results into the cache. There is no command
for this. You supply the exact prompt text of each request as its cache key,
and the import must use the same LLM settings the later run uses, or the
entries are never read.
