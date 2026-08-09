# Using OntoCast from your own agent

OntoCast is not only a server. The extraction pipeline, the ontology tooling and
the triple store are importable, so you can call them from your own LangChain
agent or splice them into an existing LangGraph workflow.

There are three ways in, in increasing order of how much of OntoCast you take on:

| You want | Use |
|---|---|
| Give an agent tools to read and edit ontologies | [`ontocast_tools`](#tools-for-any-agent) |
| Extract from one passage of text | [`run_unit_pipeline`](#extracting-from-text) |
| Run the whole document pipeline inside your graph | [`make_ontocast_node`](#the-pipeline-as-a-langgraph-node) |

## Install

The base install is deliberately light so OntoCast can be embedded without
dragging a gRPC stack, an ONNX runtime and a document-conversion pipeline into
your process. Everything heavier sits behind an extra.

```bash
pip install "ontocast[openai]"     # light core + one LLM provider
```

You always need one provider extra — OntoCast does not pick one for you.

| Extra | Adds | Needed for |
|---|---|---|
| `openai` / `anthropic` / `google` / `ollama` | the matching `langchain-*` | talking to that provider |
| `documents` | `docling-core` | representing and chunking converted documents |
| `doc-processing` | `docling`, `easyocr`, `sentence-transformers` | converting PDFs, local embeddings |
| `qdrant` | `qdrant-client`, `fastembed` | the Qdrant vector backend |
| `lancedb` | `lancedb`, `fastembed` | the LanceDB vector backend |
| `sparse` | `fastembed` | BM25 sparse embeddings on their own |
| `graph` | `networkx` | ontology lineage graphs |
| `server` | FastAPI, uvicorn, click, rich | the HTTP server and every console script |
| `all` | every runtime extra (excludes `plot`, `dev`, `docs`) | |

This table lists the extras relevant to embedding; the complete table
(`semantic-chunking`, `shacl`, `web-search`, `plot`, …) is in
[Installation](../getting_started/installation.md).

The base install carries the pipeline, the RDF stack, the in-memory triple and
vector stores, and the ontology tooling.

!!! warning "What a base install cannot do"
    **Convert or chunk documents.** Both need `docling-core`, and chunking
    additionally downloads a HuggingFace tokenizer at runtime. Use
    [`run_unit_pipeline`](#extracting-from-text), which treats its input as a
    single unit, or install `ontocast[documents]`.

    **Run the server or any console script.** `ontocast serve` needs
    `ontocast[server]`; a base install prints an install hint and exits.

    **Embed locally.** The default `EMBEDDING_PROVIDER=huggingface` needs
    `sentence-transformers`. Set `EMBEDDING_PROVIDER=openai` or `=ollama` to
    embed through an API instead.

## Constructing a ToolBox

`ToolBox` owns every stateful tool — the LLM, the triple store, the ontology
manager, the vector store. Build it **once**, at startup, and reuse it.

```python
from ontocast import Config, ToolBox

tools = await ToolBox.acreate(Config.in_memory())
await tools.initialize()
```

Two things matter here.

**Use `acreate`, not `ToolBox(config)`.** The plain constructor drives LLM
provider setup through `asyncio.run`, which is illegal inside a running event
loop — exactly where an embedder calls it from. `acreate` awaits that setup
instead. The synchronous constructor still exists for scripts and the CLI, and
now raises a directive error rather than an opaque one if you call it from a
coroutine.

**Close it when you are done.** `ToolBox` is an async context manager:

```python
async with await ToolBox.acreate(config) as tools:
    await tools.initialize()
    ...
# Fuseki's HTTP client and the Qdrant client are released here.
```

`Config.in_memory()` pins the process-local backends so nothing external is
required. Environment variables still populate every other setting; only the
store selection is forced. For a real deployment build a `Config()` normally
and point it at Fuseki and Qdrant.

## Tools for any agent

`ontocast_tools(tools)` returns LangChain `BaseTool` objects:

```python
from langchain.agents import create_agent
from ontocast import Config, ToolBox, ontocast_tools

tools = await ToolBox.acreate(Config.in_memory())
await tools.initialize()

agent = create_agent(
    model,
    tools=[*ontocast_tools(tools)],
    prompt="You are a helpful agent that edits the ontology based on input.",
)
```

### The tools

| Name | Default? | Does |
|---|---|---|
| `ontocast_list_ontologies` | yes | List every ontology with IRI, title, version |
| `ontocast_get_ontology` | yes | Fetch one ontology as Turtle |
| `ontocast_search_ontology_terms` | yes † | Find classes and properties by meaning |
| `ontocast_retrieve_ontology_context` | yes † | Retrieve the relevant ontology subgraph |
| `ontocast_sparql_select` | yes † | Read-only SELECT/ASK, returns JSON rows |
| `ontocast_sparql_construct` | yes † | Read-only CONSTRUCT/DESCRIBE, returns Turtle |
| `ontocast_chunk_text` | yes † | Split a document into size-bounded chunks |
| `ontocast_extract` | yes | Run the extraction pipeline over a passage |
| `ontocast_apply_graph_update` | `mutating=True` | Apply an insert/delete patch |
| `ontocast_ingest_ontology_ttl` | `mutating=True` † | Register a new ontology |
| `ontocast_delete_ontology` | `mutating=True` | Delete an ontology and its derivatives |
| `ontocast_convert_document` | `include=` only | Convert a file to markdown |
| `ontocast_align_entities` | `include=` only | Match equivalent entities across graphs |

† Offered only when its backend is available — see below.

### Capability gating

A tool whose backend is missing is **not returned**, rather than returned and
made to fail on first call. An agent handed a tool that always errors will keep
retrying it.

That means the list you get depends on your install and configuration. On a bare
`pip install "ontocast[openai]"` with `Config.in_memory()` you get five tools —
enough for an ontology-editing agent, because the in-memory triple store is a
full SPARQL engine rather than a degraded one.

Ask why something is missing:

```python
from ontocast import ontocast_tool_diagnostics

for name, reason in ontocast_tool_diagnostics(tools).items():
    print(f"{name}: {reason}")
```

```text
ontocast_search_ontology_terms: no vector store is configured (set VECTOR_STORE_BACKEND=memory)
ontocast_chunk_text: requires docling-core; install with pip install "ontocast[documents]"
ontocast_ingest_ontology_ttl: ontology_directory is not configured
```

`ontocast_tool_names(tools)` returns the same list without building the tools.

### Mutating tools are opt-in

Pass `mutating=True` to include the write tools. They are off by default because
each changes stored state irreversibly — `ontocast_delete_ontology` drops a
named graph, unlinks a file from disk, and deletes vectors, all from one
model-chosen IRI.

The SPARQL tools are read-only by contract and refuse `INSERT`, `DELETE`,
`DROP`, `CLEAR` and friends. Deliberate writes go through
`ontocast_apply_graph_update`, which is validated, namespace-partitioned and
triple-capped. There is no tool that hands an agent an unguarded SPARQL UPDATE.

### Selecting tools

```python
ontocast_tools(tools, include=["ontocast_sparql_select", "ontocast_get_ontology"])
ontocast_tools(tools, exclude=["ontocast_extract"])
ontocast_tools(tools, mutating=True, max_chars=50_000)
```

`max_chars` bounds each tool's rendered result. Output past the budget is cut
**and marked** — truncated Turtle parses as a syntax error and would otherwise
look like a complete graph.

!!! note "Async only"
    Every tool is a coroutine. Agents must call `ainvoke`; a synchronous
    `invoke` raises `NotImplementedError`. Several underlying calls are already
    async, and the rest are CPU-heavy enough that running them on your event
    loop would stall it.

## Vector search without a service

Term search and context retrieval need a vector store. The in-memory backend
needs no external service:

```python
config = Config.in_memory()  # already selects it
config.tool_config.embedding.provider = "openai"  # avoid local model weights
```

It keeps vectors in a numpy array and scores BM25 itself, so it needs neither
`fastembed` nor an ONNX runtime. Exact search over a few thousand ontology atoms
is faster than building an approximate index.

State lives in the process: it is lost on exit and not shared between workers.
Use Qdrant or LanceDB when the index must outlive the process.

Set the backend explicitly with `VECTOR_STORE_BACKEND`: `memory`, `qdrant`,
`lancedb`, `none`, or `auto` (the default, which infers from `QDRANT_URI` /
`LANCEDB_ENABLED` and otherwise disables vector retrieval).

## Extracting from text

`run_unit_pipeline` is the lightest way to run extraction. It is a plain
coroutine with pydantic in and out — no LangGraph, no recursion limits:

```python
from ontocast import AgentState, run_unit_pipeline
from ontocast.onto.enum import RenderMode

state = AgentState(
    raw_input={"note.txt": text.encode()},
    render_mode=RenderMode.ONTOLOGY_AND_FACTS,
)
ontology_result, facts_result = await run_unit_pipeline(state, tools)
```

It treats the whole input as **one** content unit, which is why it works on a
base install. Be aware of what it skips: chunking, section tagging, bibliography
routing, summarization, normalization and the validation gate. It also ignores
`max_chunks`, `target_sections` and `summarize_sections`. For a full document,
use the graph.

## The pipeline as a LangGraph node

`AgentState` declares no annotated reducer channels and every node returns the
whole state, so adding the compiled graph straight into your `StateGraph` only
works if your state happens to carry `raw_input`, `docling_doc`,
`aggregated_facts` and the rest. `input_schema` and `output_schema` narrow which
of `AgentState`'s keys cross the boundary but cannot rename them.

So the mapping is explicit:

```python
from langgraph.graph import StateGraph
from ontocast import make_ontocast_node, text_in_turtle_out

to_state, from_state = text_in_turtle_out()

builder = StateGraph(MyState)
builder.add_node(
    "extract",
    make_ontocast_node(tools, to_agent_state=to_state, from_agent_state=from_state),
)
```

`text_in_turtle_out()` covers the common case: read a string off your state,
write back `ontology_ttl` and `facts_ttl`. Pass `text_key`, `ontology_key` and
`facts_key` to use your own names, or write the two callables yourself for
anything more involved.

!!! warning "The recursion-limit trap"
    LangGraph's default recursion limit is 25, which a multi-chunk document
    exceeds — the most likely first-run failure when embedding the pipeline.
    `make_ontocast_node` derives a limit from your chunk budget instead. Leave
    `recursion_limit` unset unless you have a reason.

The node compiles the graph once at construction, not per invocation, and merges
your `RunnableConfig` rather than replacing it, so callbacks and tracing
metadata survive.

### Building the graph yourself

```python
from ontocast import build_agent_graph, create_agent_graph

compiled = create_agent_graph(tools, checkpointer=saver, name="ontocast")
builder = build_agent_graph(tools)  # uncompiled, for splicing nodes
```

`create_agent_graph` takes optional `checkpointer`, `store` and `name`. Set
`name` when embedding as a subgraph — an unnamed one shows up as `LangGraph` in
traces. `build_agent_graph` returns the uncompiled `StateGraph` when you need to
inspect the topology or add your own nodes before compiling.

## Tenancy

A `ToolBox` is bound to one tenant/project partition. For a single-tenant
application this is invisible: build one ToolBox and use it.

To serve several partitions from one process:

```python
scoped = await tools.for_scope("acme", "reports")
```

Each scope gets its own triple store, ontology catalog and vector store over a
deep copy of the configuration, so they cannot see each other. The expensive
tools — LLM client and cache, converter, chunker, embedding model — are shared
across scopes, so a second tenant costs a store connection rather than another
model.

Scopes are cached in a bounded LRU (`MAX_TENANCY_SCOPES`, default 16); evicting
one closes its connections, and `await tools.aclose()` closes them all. Nothing
is allocated until you first call `for_scope`. See [Tenancy](tenancy.md).
