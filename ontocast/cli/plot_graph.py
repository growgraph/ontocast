import importlib
import logging
import re
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from langchain_core.runnables.graph import Graph

logger = logging.getLogger(__name__)

# Warm palette matching the Mermaid hand-drawn theme
_NODE_FILL = "#FFF3E0"
_NODE_BORDER = "#143642"
_NODE_FONT = "#372237"
_ACCENT_FILL = "#FFCCBC"
_ACCENT_BORDER = "#BF360C"
_EDGE_COLOR = "#8D6E63"
_COND_EDGE_COLOR = "#E64A19"
_BG_COLOR = "#FFFCF7"
_FONTNAME = "Helvetica"

_NODE_LABELS: dict[str, str] = {"__end__": "END", "__start__": "START"}


def _wrap_label(text: str, width: int = 8) -> str:
    """Break a multi-word label into lines so nodes stay narrow in LR layouts."""
    return "\n".join(
        textwrap.wrap(
            text,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


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


def update_mermaid_graph_in_markdown(file_path: str, new_graph: str) -> None:
    md_path = Path(file_path)
    content = md_path.read_text()

    pattern = r"(### Agent graph\s+```mermaid\n)(.*?)(\n```)"
    replacement = r"\1" + new_graph + r"\3"

    if re.search(pattern, content, flags=re.DOTALL):
        new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        print("✅ Replaced existing Mermaid block.")
    else:
        new_section = f"\n\n### Agent graph\n\n```mermaid\n{new_graph}\n```"
        new_content = content + new_section
        print("➕ Appended new Mermaid block at the end.")

    md_path.write_text(new_content)
    print(f"📄 Updated {file_path}")


def draw_graphviz(
    pgv_module: Any,
    graph: "Graph",
    fname: str,
    extensions: tuple[str, ...],
    rankdir: str = "TB",
) -> None:
    is_lr = rankdir == "LR"
    splines = "spline" if not is_lr else "ortho"

    viz: Any = pgv_module.AGraph(directed=True, strict=False)
    viz.graph_attr.update(
        rankdir=rankdir,
        bgcolor=_BG_COLOR,
        pad="0.3" if is_lr else "0.6",
        nodesep="0.35" if is_lr else "0.7",
        ranksep="0.5" if is_lr else "0.9",
        fontname=_FONTNAME,
        splines=splines,
    )
    viz.node_attr.update(
        shape="box",
        style="rounded,filled",
        fillcolor=_NODE_FILL,
        color=_NODE_BORDER,
        fontcolor=_NODE_FONT,
        fontsize="11" if is_lr else "13",
        fontname=_FONTNAME,
        margin="0.15,0.08" if is_lr else "0.25,0.12",
        penwidth="1.5" if is_lr else "1.8",
    )
    viz.edge_attr.update(
        fontname=_FONTNAME,
        fontsize="9" if is_lr else "10",
        fontcolor=_EDGE_COLOR,
        color=_EDGE_COLOR,
        penwidth="1.2" if is_lr else "1.5",
        arrowsize="0.7" if is_lr else "0.9",
    )

    hidden_nodes = {"__start__", "__end__"} if is_lr else set()

    for node in graph.nodes:
        if node in hidden_nodes:
            continue
        label = _NODE_LABELS.get(node, node)
        if is_lr:
            label = _wrap_label(label)
        viz.add_node(node, label=label)

    for start, end, data, conditional in graph.edges:
        if start in hidden_nodes or end in hidden_nodes:
            continue
        raw_label = str(data) if data is not None else ""
        label = _NODE_LABELS.get(raw_label, raw_label)
        if conditional:
            viz.add_edge(
                start,
                end,
                label=label,
                style="dashed",
                color=_COND_EDGE_COLOR,
                fontcolor=_COND_EDGE_COLOR,
            )
        else:
            viz.add_edge(start, end, label=label)

    accent_attrs = {
        "fillcolor": _ACCENT_FILL,
        "color": _ACCENT_BORDER,
        "fontcolor": _ACCENT_BORDER,
        "penwidth": "2.5",
    }
    if first := graph.first_node():
        if first.id not in hidden_nodes:
            viz.get_node(first.id).attr.update(**accent_attrs)
    if last := graph.last_node():
        if last.id not in hidden_nodes:
            viz.get_node(last.id).attr.update(**accent_attrs)

    out = fname + f".{rankdir.lower()}" if rankdir.lower() == "lr" else fname
    for ext in extensions:
        if ext == "svg":
            viz.draw(out + ".svg", format="svg:cairo", prog="dot")
            print(f"📄 Wrote {out}.svg")
        elif ext == "png":
            viz.draw(out + ".png", format="png", prog="dot", args="-Gdpi=300")
            print(f"📄 Wrote {out}.png")


def main() -> None:
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

    app = create_agent_graph(toolbox)
    graph = app.get_graph()
    mmd_data = graph.draw_mermaid(frontmatter_config=frontmatter_config)

    with open("graph.mmd", "w") as f:
        f.write(mmd_data)

    try:
        # import pygraphviz as pgv_module
        pgv_module = importlib.import_module("pygraphviz")

        draw_graphviz(
            pgv_module, graph, "docs/assets/graph", ("svg", "png"), rankdir="TB"
        )
        draw_graphviz(
            pgv_module, graph, "docs/assets/graph", ("svg", "png"), rankdir="LR"
        )
    except ImportError as e:
        logger.info(f"pygraphviz not available, skipping graphviz output: {e}")

    try:
        from langchain_core.runnables.graph import MermaidDrawMethod

        png_data = graph.draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
            frontmatter_config=frontmatter_config,
            padding=20,
        )

        with open("docs/assets/graph.preview.png", "wb") as f:
            f.write(png_data)
    except ImportError as e:
        logger.info(f"MermaidDrawMethod not available, skipping mermaid PNG: {e}")


if __name__ == "__main__":
    main()
