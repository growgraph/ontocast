# Examples

Runnable scripts showing how to use OntoCast as a library. Each runs against the
in-memory triple and vector stores, so **no external services are required** —
no Fuseki, no Qdrant, no Docker.

## Setup

```bash
pip install "ontocast[openai]"
export OPENAI_API_KEY=sk-...
```

Any provider extra works; set `LLM_PROVIDER` and the matching key if you are not
using OpenAI.

## The scripts

| Script | Shows |
|---|---|
| `unit_pipeline.py` | The lightest entry point: extract an ontology and facts from a passage of text |
| `langchain_agent.py` | Handing OntoCast's tools to a LangChain agent |
| `langgraph_subgraph.py` | Running the full pipeline as a node inside your own LangGraph |

```bash
python examples/unit_pipeline.py
python examples/langchain_agent.py
python examples/langgraph_subgraph.py
```

`langchain_agent.py` prints the available tools and their gating reasons without
calling the model, so it is also the quickest way to check what your install
supports.

Background and the full tool table:
[Embedding OntoCast](https://growgraph.github.io/ontocast/user_guide/embedding/).
