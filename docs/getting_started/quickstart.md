# Quick start

This page takes one document all the way to a knowledge graph: you install
OntoCast, start its server, send it a PDF, and read back the ontology and facts
it extracted.

## 1. Install

```bash
pip install "ontocast[server,openai,doc-processing]"
```

`server` provides the `ontocast` command, `openai` the model provider, and
`doc-processing` the document converter (PDF, Office, HTML, Markdown, images). For plain text or JSON input
you can leave out `doc-processing`. [Installation](installation.md) lists the
other extras.

## 2. Give it a model

OntoCast reads its settings from environment variables. With the defaults it
calls OpenAI, so the only setting you need is the key:

```bash
export LLM_API_KEY=sk-...
```

To use another provider, install its extra and set three variables:

| Provider | Extra | Settings |
|---|---|---|
| Anthropic | `anthropic` | `LLM_PROVIDER=anthropic`, `LLM_MODEL_NAME`, `LLM_API_KEY` |
| Google | `google` | `LLM_PROVIDER=google`, `LLM_MODEL_NAME`, `LLM_API_KEY` |
| Ollama | `ollama` | `LLM_PROVIDER=ollama`, `LLM_MODEL_NAME`, `LLM_BASE_URL` (for example `http://localhost:11434`) |

!!! tip "Keeping settings in a file"
    The repository ships `.env.example.minimal`, the settings worth a decision,
    and `.env.example`, all of them. Copy one to `.env` and edit it. OntoCast
    does not read the file itself, so load it into the environment of the
    command you run:

    ```bash
    set -a; source .env; set +a
    ```

## 3. Start the server

```bash
ontocast serve
```

The server listens on `http://127.0.0.1:8999`. It keeps graphs in memory, so
they are gone when it stops; [Triple stores](../guides/triple_stores.md) shows
how to keep them in Apache Jena Fuseki. Check that it is up:

```bash
curl http://127.0.0.1:8999/health
```

## 4. Send a document

```bash
curl -X POST http://127.0.0.1:8999/process -F "file=@paper.pdf" -o result.json
```

OntoCast accepts `.pdf`, `.pptx`, `.txt`, `.json` and `.jsonl` files, or text
in a JSON body:

```bash
curl -X POST http://127.0.0.1:8999/process \
  -H "Content-Type: application/json" \
  -d '{"text": "The sample was annealed at 150 °C for 10 minutes."}'
```

You gave OntoCast no ontology, so it builds one from the document and then
extracts the facts in its terms. The request returns when the whole document
is done, and each part of the document goes through the model separately, so a
long document takes a while.

## 5. Read the result

The response is JSON. The graphs are Turtle strings:

```json
{
  "data": {
    "facts": "@prefix cd: <...> . ...",
    "ontology_artifacts": [
      {"iri": "https://...", "ontology_id": "...", "title": "...", "triples": 42, "ttl": "..."}
    ]
  },
  "metadata": {"status": "...", "chunks_processed": 7, "budget": {...}, "failed_units": [], ...}
}
```

Save the facts to a file you can load into any RDF tool:

```bash
jq -r .data.facts result.json > facts.ttl
jq -r '.data.ontology_artifacts[0].ttl' result.json > ontology.ttl
```

`metadata` says what the run cost and where it struggled: `budget` counts LLM
calls, `failed_units` lists any part of the document that produced nothing, and
`facts_conformance` summarizes validation. If no part of the document produced
output, the server answers `422` instead. [Reading run
telemetry](../guides/telemetry.md) explains the rest.

## Process files without a server

`ontocast process` runs the same pipeline on local files and writes the graphs
next to them, or into one directory:

```bash
ontocast process --input-path ./papers --output-dir ./out
```

`--input-path` takes a file or a directory, searched recursively. For each
document the command writes `<name>.facts.ttl`, plus one
`<name>.<ontology>.ontology.ttl` for every ontology the document changed. To
try settings on part of a document first, add `--head-chunks 3`.

## Start from your own ontologies

Extraction is better when OntoCast works in terms of an ontology you trust.
Put your Turtle files in a directory and pass it at startup:

```bash
ontocast serve --ontology-dir ./my-ontologies
```

Or upload one to a running server:

```bash
curl -X POST http://127.0.0.1:8999/ontologies -F "file=@my-ontology.ttl"
```

## What to read next

- [How OntoCast works](../concepts/index.md): what happens to the document between steps 4 and 5.
- [Configuring OntoCast](../guides/configuration.md): the settings that change what a run does.
- [Recipes](../guides/recipes.md): settings for evaluating, building an ontology, or extracting facts at scale.
- [HTTP API](../reference/http_api.md): every route and parameter.
