import logging
import re
from pathlib import Path

from ontocast.config import (
    Config,
    LLMConfig,
    LLMProvider,
    OllamaModel,
    PathConfig,
    ToolConfig,
)
from ontocast.stategraph import create_agent_graph
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def update_mermaid_graph_in_markdown(file_path: str, new_graph: str):
    md_path = Path(file_path)
    content = md_path.read_text()

    # Regex pattern to find "### Agent graph" followed by a mermaid block
    pattern = r"(### Agent graph\s+```mermaid\n)(.*?)(\n```)"
    replacement = r"\1" + new_graph + r"\3"

    if re.search(pattern, content, flags=re.DOTALL):
        new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        print("✅ Replaced existing Mermaid block.")
    else:
        # Append new section at the end
        new_section = f"\n\n### Agent graph\n\n```mermaid\n{new_graph}\n```"
        new_content = content + new_section
        print("➕ Appended new Mermaid block at the end.")

    md_path.write_text(new_content)
    print(f"📄 Updated {file_path}")


frontmatter_config = {
    "config": {
        "theme": "base",
        "look": "handDrawn",
        "themeVariables": {
            "primaryColor": "#FFF3E0",
            "primaryBorderColor": "#143642",
            "primaryTextColor": "#372237",
            "lineColor": "#FFAB91",
            "fontFamily": "'Architects Daughter', cursive",
            "fontSize": "20px",
        },
        "flowchart": {"curve": "basis", "htmlLabels": True, "useMaxWidth": True},
    }
}


def main():
    # Create a minimal config for plotting (no API keys needed)
    config = Config(
        tool_config=ToolConfig(
            path_config=PathConfig(
                ontology_directory=None, working_directory=Path("/tmp")
            ),
            llm_config=LLMConfig(
                provider=LLMProvider.OLLAMA,
                model_name=OllamaModel.LLAMA3_1,
                base_url="http://localhost:11434",
            ),
        )
    )
    toolbox = ToolBox(config)

    # Get the graph and save it as PNG
    app = create_agent_graph(toolbox)
    graph = app.get_graph()
    mmd_data = graph.draw_mermaid(frontmatter_config=frontmatter_config)

    # Save the PNG data to a file
    with open("graph.mmd", "w") as f:
        f.write(mmd_data)
    mmd_data = mmd_data.replace("__start__", "START").replace("__end__", "END")
    # update_mermaid_graph_in_markdown("README.md", mmd_data)

    labels = {
        "nodes": {"__end__": "END", "__start__": "START"},
    }

    def tweak_draw(fname, extensions: tuple[str, ...]):
        fontname = "'Architects Daughter'"

        subtle_green = "#a9cca9"
        subtle_orange = "#ffdb99"
        viz = pgv.AGraph(directed=True, nodesep=0.7, ranksep=0.5)
        for node in graph.nodes:
            viz.add_node(
                node,
                label=labels.get("nodes", {}).get(node, node),
                style="filled",
                fillcolor=subtle_green,
                fontsize=12,
                fontname=fontname,
            )
        for start, end, data, conditional in graph.edges:
            label = str(data) if data is not None else ""
            label = labels.get("edges", {}).get(label, label)
            viz.add_edge(
                start,
                end,
                label=label,
                fontsize=10,
                fontname=fontname,
                style="dashed" if conditional else "solid",
            )
        if first := graph.first_node():
            viz.get_node(first.id).attr.update(fillcolor=subtle_orange)
        if last := graph.last_node():
            viz.get_node(last.id).attr.update(fillcolor=subtle_orange)
        for ext in extensions:
            if ext == "svg":
                viz.draw(fname + ".svg", format="svg:cairo", prog="dot")
            elif ext == "png":
                viz.draw(fname + ".png", format="png", prog="dot", args="-Gdpi=300")

    try:
        import pygraphviz as pgv  # type: ignore

        tweak_draw("docs/assets/graph", extensions=("svg", "png"))
    except ImportError as e:
        logger.info(f"Could not import graphviz: {e}")

    try:
        from langchain_core.runnables.graph import MermaidDrawMethod

        png_data = graph.draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
            frontmatter_config=frontmatter_config,
            padding=20,
        )

        with open("docs/assets/graph.mmd", "wb") as f:
            f.write(png_data)
    except ImportError as e:
        logger.info(f"Could not import MermaidDrawMethod: {e}")


if __name__ == "__main__":
    main()
