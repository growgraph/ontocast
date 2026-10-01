# Python API

These pages are generated from the docstrings at build time, one page per
public module. Start from the package that matches what you are doing:

| Package | What it holds |
|---|---|
| [`ontocast.toolbox`](toolbox.md) | `ToolBox`: builds and owns every tool a run needs (LLM, converter, chunker, triple store, vector store, caches) |
| [`ontocast.stategraph`](stategraph.md) | The LangGraph pipeline: graph construction, routing and the per-unit render/critic loop |
| [`ontocast.onto`](onto.md) | State and domain models: `AgentState`, per-unit states, ontologies, graph updates, budgets |
| [`ontocast.agent`](agent.md) | One function per pipeline step (render, criticise, consolidate, serialize, ...) |
| [`ontocast.tool`](tool.md) | The tools themselves: LLM, chunking, triple stores, vector stores, aggregation, validation |
| [`ontocast.config`](config.md) | Settings, read from environment variables |
| [`ontocast.api`](api.md) | The HTTP API (FastAPI) and request parsing |
| [`ontocast.integrations`](integrations.md) | Adapters that expose OntoCast to other agent frameworks |
| [`ontocast.cli`](cli.md) | Console scripts |
| [`ontocast.registry`](registry.md), [`ontocast.runtime`](runtime.md) | Scope-bound ToolBoxes over one shared runtime (tenancy) |
| [`ontocast.util`](util.md) | Helpers |
