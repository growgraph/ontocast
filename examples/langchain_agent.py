"""Hand OntoCast's tools to a LangChain agent.

    pip install "ontocast[openai]"
    export OPENAI_API_KEY=sk-...
    python examples/langchain_agent.py

The first half of this script needs no model call: it prints which tools your
install exposes and why the rest are missing, which is the quickest way to see
what capability gating decided.
"""

import asyncio

from ontocast import (
    Config,
    ToolBox,
    ontocast_tool_diagnostics,
    ontocast_tools,
)

SEED_ONTOLOGY = """
@prefix ex:   <http://example.org/instrument#> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

<http://example.org/instrument> a owl:Ontology .

ex:Instrument a owl:Class ;
    rdfs:label "Instrument" ;
    rdfs:comment "A scientific device that performs a measurement." .

ex:Spectrometer a owl:Class ;
    rdfs:subClassOf ex:Instrument ;
    rdfs:label "Spectrometer" .

ex:measures a owl:ObjectProperty ;
    rdfs:domain ex:Instrument ;
    rdfs:label "measures" .
"""


async def main() -> None:
    config = Config.in_memory()
    # Embed through an API rather than downloading local model weights, so this
    # runs on a base install.
    config.tool_config.embedding.provider = "openai"

    async with await ToolBox.acreate(config) as tools:
        await tools.initialize()

        available = ontocast_tools(tools)
        print(f"=== {len(available)} tools available ===")
        for tool in available:
            print(f"  {tool.name}")

        omitted = ontocast_tool_diagnostics(tools)
        if omitted:
            print("\n=== omitted ===")
            for name, reason in omitted.items():
                print(f"  {name}: {reason}")

        # Tools are coroutines: agents must use `ainvoke`, never `invoke`.
        print("\n=== calling a tool directly ===")
        by_name = {tool.name: tool for tool in available}
        print(await by_name["ontocast_list_ontologies"].ainvoke({}))

        # Write tools are opt-in, because each one changes stored state
        # irreversibly.
        writable = ontocast_tools(tools, mutating=True)
        ingest = {t.name: t for t in writable}.get("ontocast_ingest_ontology_ttl")
        if ingest is not None:
            print("\n=== ingesting a seed ontology ===")
            print(await ingest.ainvoke({"ttl": SEED_ONTOLOGY}))

        # Wiring the tools into an agent. Requires `pip install langchain`.
        try:
            from langchain.agents import create_agent
            from langchain.chat_models import init_chat_model
        except ImportError:
            print("\n(install `langchain` to run the agent half of this example)")
            return

        model = init_chat_model("openai:gpt-4o-mini")
        agent = create_agent(
            model,
            tools=[*ontocast_tools(tools, mutating=True)],
            prompt=(
                "You are a helpful agent that edits an ontology based on input. "
                "Inspect the existing ontology before proposing changes, and "
                "reuse existing terms rather than minting near-duplicates."
            ),
        )

        print("\n=== agent ===")
        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "What classes already exist? Then add a Camera class "
                            "as a kind of Instrument."
                        ),
                    }
                ]
            }
        )
        print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
