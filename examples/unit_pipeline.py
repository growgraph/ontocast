"""Extract an ontology and facts from a passage of text.

The lightest way to use OntoCast: one coroutine, pydantic in and out, no
LangGraph. Runs against the in-memory triple store, so nothing external is
needed.

    pip install "ontocast[openai]"
    export OPENAI_API_KEY=sk-...
    python examples/unit_pipeline.py

`run_unit_pipeline` treats its input as a single content unit. That is why it
works on a base install -- chunking needs the `documents` extra -- but it also
means it skips section tagging, summarization, normalization and the validation
gate. Use the full graph for real documents; see examples/langgraph_subgraph.py.
"""

import asyncio

from ontocast import AgentState, Config, ToolBox, run_unit_pipeline
from ontocast.onto.enum import RenderMode

TEXT = """
The Curiosity rover landed in Gale Crater on Mars in August 2012. Its
ChemCam instrument uses laser-induced breakdown spectroscopy to determine
the elemental composition of rock targets from up to seven metres away.
In 2013 the rover drilled into a mudstone outcrop named John Klein, which
yielded evidence of an ancient freshwater lake.
"""


async def main() -> None:
    # in_memory() pins the process-local stores; every other setting still
    # comes from the environment.
    async with await ToolBox.acreate(Config.in_memory()) as tools:
        await tools.initialize()

        state = AgentState(
            raw_input={"curiosity.txt": TEXT.encode("utf-8")},
            render_mode=RenderMode.ONTOLOGY_AND_FACTS,
        )
        ontology_result, facts_result = await run_unit_pipeline(state, tools)

        if ontology_result is not None:
            graph = (
                ontology_result.fresh_ontology.graph
                if ontology_result.fresh_ontology is not None
                and not ontology_result.fresh_ontology.is_null()
                else ontology_result.working_graph
            )
            print("=== ontology ===")
            print(graph.serialize_canonical_turtle())

        if facts_result is not None:
            print("=== facts ===")
            print(facts_result.content_unit.graph.serialize_canonical_turtle())

        budget = state.budget_tracker
        print(
            f"=== budget ===\n"
            f"LLM calls: {budget.calls_count} "
            f"(cache hits: {budget.cache_hits}), "
            f"chars sent: {budget.chars_sent}"
        )


if __name__ == "__main__":
    asyncio.run(main())
