# Embedding OntoCast in your agent

OntoCast is a library as well as a server. This page shows how to call the
extraction pipeline, the ontology tools and the triple store from your own
LangChain agent or LangGraph workflow.

There are three ways in, in increasing order of how much of OntoCast you take
on:

| You want to | Use |
|---|---|
| Give an agent tools to read and edit ontologies | [`ontocast_tools`](#tools-for-any-agent) |
| Extract from one passage of text | [`run_unit_pipeline`](#extract-from-one-passage) |
| Run the whole document pipeline inside your graph | [`make_ontocast_node`](#the-pipeline-as-a-langgraph-node) |

Install the core with one LLM provider extra, for example
`pip install "ontocast[openai,documents]"`. The `documents` extra is needed by
everything that extracts, including `run_unit_pipeline`; [Installation](../getting_started/installation.md)
lists the others.

## Construct a ToolBox

`ToolBox` owns every stateful tool: the LLM client, the triple store, the
ontology catalog and the vector store. Build it once, at startup, and reuse it:

```python
from ontocast import Config, ToolBox

async with await ToolBox.acreate(Config.in_memory()) as tools:
    await tools.initialize()
    ...
# Backend connections (Fuseki, Qdrant) are closed here.
```

Use `ToolBox.acreate`, not `ToolBox(config)`, from async code: the plain
constructor sets up the LLM provider through `asyncio.run`, which fails inside
a running event loop. `initialize()` loads the ontology catalog and the SHACL
shapes. Without `async with`, call `await tools.aclose()` when you are done.

`Config.in_memory()` uses the in-memory triple store, a full SPARQL engine, and
turns vector retrieval off, so nothing external is needed. Every other setting
still comes from the environment. For a deployment, build `Config()` and point
it at Fuseki; see [Triple stores](triple_stores.md).

## Tools for any agent

`ontocast_tools(tools)` returns LangChain tools any agent can call:

```python
from langchain.agents import create_agent
from ontocast import ontocast_tools

agent = create_agent(model, tools=ontocast_tools(tools))
```

| Tool | Offered | Does |
|---|---|---|
| `ontocast_list_ontologies` | by default | List every ontology with its IRI, title and version |
| `ontocast_get_ontology` | by default | Fetch one ontology as Turtle |
| `ontocast_search_ontology_terms` | by default, with a vector store | Find classes and properties by meaning |
| `ontocast_retrieve_ontology_context` | by default, with a vector store | Retrieve the part of the ontology relevant to a text |
| `ontocast_sparql_select` | by default | Read-only SELECT or ASK, returned as JSON rows |
| `ontocast_sparql_construct` | by default | Read-only CONSTRUCT or DESCRIBE, returned as Turtle |
| `ontocast_chunk_text` | by default, with `documents` | Split a document into parts |
| `ontocast_extract` | by default | Run extraction over a passage, as one part |
| `ontocast_apply_graph_update` | `mutating=True` | Apply an insert/delete patch |
| `ontocast_ingest_ontology_ttl` | `mutating=True` | Register a new ontology |
| `ontocast_delete_ontology` | `mutating=True` | Delete an ontology from the triple store and the vector store |
| `ontocast_convert_document` | named in `include` | Convert a file on the host to text |
| `ontocast_align_entities` | named in `include` | Match equivalent entities across graphs |

Every tool is a coroutine: agents must call `ainvoke`, and `invoke` raises
`NotImplementedError`.

### Tools whose backend is missing are left out

A tool whose backend is missing is not returned at all, because an agent
handed a tool that always fails keeps retrying it. The list you get therefore
depends on your install and configuration. To see why a tool is missing:

```python
from ontocast import ontocast_tool_diagnostics

for name, reason in ontocast_tool_diagnostics(tools).items():
    print(f"{name}: {reason}")
```

```text
ontocast_search_ontology_terms: no vector store is configured (set QDRANT_URI or LANCEDB_ENABLED)
ontocast_chunk_text: requires docling-core; install with pip install "ontocast[documents]"
```

`ontocast_tool_names(tools)` returns the names `ontocast_tools` would build,
without building them.

### Choosing tools

```python
ontocast_tools(tools, include=["ontocast_sparql_select", "ontocast_get_ontology"])
ontocast_tools(tools, exclude=["ontocast_extract"])
ontocast_tools(tools, mutating=True, max_chars=50_000)
```

The write tools need `mutating=True` because each changes stored state for
good: `ontocast_delete_ontology` drops every stored version of an ontology and
its vectors, from one IRI the model chose. It leaves your seed directory alone,
so an ontology that is still there returns at the next start. The SPARQL tools refuse
`INSERT`, `DELETE`, `DROP`, `CLEAR` and the other update keywords; writes go
through `ontocast_apply_graph_update`, which validates the patch and caps its
size.

`max_chars` bounds each tool's result. A result cut to fit is marked as cut, so
the model does not read truncated Turtle as a complete graph.

### Vector search

The two search tools need a vector store, and `Config.in_memory()` has none.
Without one, each part is shown one ontology from the catalog, which is a
complete extraction path; vector retrieval exists to combine terms from several
ontologies. To add the embedded LanceDB store (`lancedb` extra) to an in-memory
configuration:

```python
from ontocast.onto.enum import OntologyContextMode

config = Config.in_memory()
config.tool_config.lancedb.enabled = True
config.tool_config.embedding.provider = (
    "openai"  # embed through the API, no local model
)
config.server.ontology_context_mode = (
    OntologyContextMode.SELECTED_VECTOR_SEARCH_ONTOLOGY
)

tools = await ToolBox.acreate(config)
await tools.initialize()
```

`initialize()` builds the index when the configured mode, or the
`ontology_context_mode` passed to it, is vector retrieval; for extraction, set
the same mode on the `AgentState` (`ontology_context_mode`). For a Qdrant server, install the
`qdrant` extra and set `QDRANT_URI`. The
default embedding provider runs a local model and needs `sentence-transformers`
(in `doc-processing`). [Choosing ontology context](ontology_context.md) explains
the trade-offs.

## Extract from one passage

`run_unit_pipeline` is the lightest way to run extraction: a plain coroutine,
pydantic in and out, with no LangGraph graph and so no recursion limit to set.
Each loop inside it is bounded by the critic and retry budgets.

```python
from ontocast import AgentState, run_unit_pipeline
from ontocast.onto.enum import RenderMode

state = AgentState(
    raw_input={"note.txt": text.encode()},
    render_mode=RenderMode.ONTOLOGY_AND_FACTS,
    facts_user_instruction="Extract every measurement with its unit.",
)
ontology_result, facts_result = await run_unit_pipeline(state, tools)
```

It treats the whole input as one part (content unit). It skips chunking,
section tagging, bibliography routing, summarization, ontology normalization,
entity disambiguation and the validation gate, and ignores `max_chunks`,
`target_sections` and `summarize_sections`. For a whole document, use the
graph.

## The pipeline as a LangGraph node

The pipeline's state type has no reducer channels and every node returns the
whole state, so you cannot add the compiled graph to your `StateGraph`
directly unless your state has the same keys. `make_ontocast_node` maps
between the two states explicitly:

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

`text_in_turtle_out()` reads a string from your state's `input` key and writes back
`ontology_ttl` and `facts_ttl`; pass `text_key`, `ontology_key` and `facts_key`
to use your own names, or write the two mapping functions yourself.

!!! warning "Leave `recursion_limit` unset"
    LangGraph's default recursion limit is 25, which a document of more than a
    few parts exceeds. `make_ontocast_node` derives a limit from your chunk and
    retry budgets instead.

The node compiles the graph once, when you create it, and merges your
`RunnableConfig` into its own, so callbacks and tracing metadata reach the
inner run.

### Building the graph yourself

```python
from ontocast import build_agent_graph, create_agent_graph

compiled = create_agent_graph(tools, checkpointer=saver, name="ontocast")
builder = build_agent_graph(tools)  # uncompiled, to add your own nodes
```

`create_agent_graph` takes optional `checkpointer`, `store` and `name`. Set
`name` when you embed the graph as a subgraph; otherwise it appears as
`LangGraph` in traces.

## Several tenants in one process

A `ToolBox` serves one partition of the stores. To serve another:

```python
scoped = await tools.for_scope("acme", "reports")
```

Each scope has its own triple store connection, ontology catalog and vector
store, over its own copy of the configuration, so scopes cannot see each
other's data. The LLM client and its cache, the converter, the chunker and the
embedding model are shared. At most
[`MAX_TENANCY_SCOPES`](../reference/configuration/pipeline.md#max_tenancy_scopes)
scopes stay open; the least recently used is closed first. `await
tools.aclose()` closes them all. [Tenancy](tenancy.md) explains how partitions
are named.
