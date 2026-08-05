"""Run the OntoCast pipeline as one node inside your own LangGraph.

    pip install "ontocast[openai,documents]"
    export OPENAI_API_KEY=sk-...
    python examples/langgraph_subgraph.py

The `documents` extra is needed here but not in the other examples: the full
graph chunks its input, and chunking needs docling-core.

The parent state below has nothing in common with `AgentState`, which is the
point. `AgentState` declares no annotated reducer channels and every node
returns the whole state, so LangGraph cannot merge it into a foreign schema on
its own -- `make_ontocast_node` takes the mapping explicitly instead.
"""

import asyncio
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from ontocast import Config, ToolBox, make_ontocast_node, text_in_turtle_out

REPORT = """
The Thames Water abstraction licence covers the Farmoor reservoir, which
supplies Oxford. Peak daily abstraction is capped at 145 megalitres. During
the 2022 drought the Environment Agency granted a temporary variation
raising the cap to 160 megalitres for eight weeks.
"""


class ReviewState(TypedDict):
    """Parent state -- deliberately unlike AgentState."""

    input: str
    ontology_ttl: str
    facts_ttl: str
    verdict: str


def summarize(state: ReviewState) -> dict[str, str]:
    """A downstream node consuming what OntoCast produced."""
    facts = state["facts_ttl"]
    triples = sum(1 for line in facts.splitlines() if line.strip().endswith("."))
    return {"verdict": f"extracted roughly {triples} statements"}


async def main() -> None:
    async with await ToolBox.acreate(Config.in_memory()) as tools:
        await tools.initialize()

        # text_in_turtle_out reads `input` and writes `ontology_ttl` / `facts_ttl`.
        # Pass text_key / ontology_key / facts_key to use your own names, or
        # write the two callables yourself for anything more involved.
        to_state, from_state = text_in_turtle_out()

        extract = make_ontocast_node(
            tools,
            to_agent_state=to_state,
            from_agent_state=from_state,
            # Leave recursion_limit unset: the node derives one from the chunk
            # budget. LangGraph's own default of 25 dies on a multi-chunk
            # document, and that is the usual first-run failure here.
        )

        builder = StateGraph(ReviewState)
        builder.add_node("extract", extract)
        builder.add_node("summarize", summarize)
        builder.add_edge(START, "extract")
        builder.add_edge("extract", "summarize")
        builder.add_edge("summarize", END)
        graph = builder.compile()

        result = await graph.ainvoke(
            {"input": REPORT, "ontology_ttl": "", "facts_ttl": "", "verdict": ""}
        )

        print("=== ontology ===")
        print(result["ontology_ttl"] or "(none produced)")
        print("=== facts ===")
        print(result["facts_ttl"] or "(none produced)")
        print("=== downstream node ===")
        print(result["verdict"])


if __name__ == "__main__":
    asyncio.run(main())
