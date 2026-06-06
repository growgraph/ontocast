import importlib
import logging
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

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
_ACCENT_FILL = "#BCCCFF"
_ACCENT_BORDER = "#0C369F"
_EDGE_COLOR = "#8D6E63"
_COND_EDGE_COLOR = "#E64A19"
_BG_COLOR = "#FFFCF7"
_FONTNAME = "Helvetica"

_NODE_LABELS: dict[str, str] = {"__end__": "END", "__start__": "START"}

NodeShape = Literal["process", "decision", "terminal"]


@dataclass(frozen=True)
class FlowNode:
    node_id: str
    label: str
    shape: NodeShape = "process"


@dataclass(frozen=True)
class FlowEdge:
    start: str
    end: str
    label: str = ""
    conditional: bool = False


@dataclass(frozen=True)
class FlowGraph:
    nodes: tuple[FlowNode, ...]
    edges: tuple[FlowEdge, ...]
    start_node: str
    end_node: str


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


def _mermaid_frontmatter() -> str:
    return """---
config:
  flowchart:
    curve: basis
    htmlLabels: true
    useMaxWidth: true
  look: handDrawn
  theme: base
  themeVariables:
    fontFamily: '''Architects Daughter'', cursive'
    fontSize: 20px
    lineColor: '#FFAB91'
    primaryBorderColor: '#143642'
    primaryColor: '#FFF3E0'
    primaryTextColor: '#372237'
---
"""


def _mermaid_node(node: FlowNode) -> str:
    label = node.label.replace("\n", "<br/>")
    if node.shape == "terminal":
        return f"\t{node.node_id}([{label}])"
    if node.shape == "decision":
        return f"\t{node.node_id}{{{label}}}"
    return f"\t{node.node_id}[{label}]"


def flow_graph_to_mermaid(flow: FlowGraph) -> str:
    lines = [_mermaid_frontmatter(), "flowchart TD;"]
    for node in flow.nodes:
        lines.append(_mermaid_node(node))
    for edge in flow.edges:
        arrow = "-.->" if edge.conditional else "-->"
        suffix = f"|{edge.label}|" if edge.label else ""
        lines.append(f"\t{edge.start} {arrow}{suffix} {edge.end};")
    return "\n".join(lines) + "\n"


def _atomic_loop_core_edges(
    *, render_node: str, critic_node: str
) -> tuple[FlowEdge, ...]:
    """Render/critic retry loop without optional web-evidence branches."""
    return (
        FlowEdge("render_loop", render_node),
        FlowEdge(render_node, "final_render", "success", conditional=True),
        FlowEdge(render_node, "render_loop", "fail", conditional=True),
        FlowEdge("final_render", critic_node, "no", conditional=True),
        FlowEdge(critic_node, "done", "success", conditional=True),
        FlowEdge(critic_node, "render_loop", "fail", conditional=True),
        FlowEdge("render_loop", "exhausted", "exhausted", conditional=True),
    )


def _atomic_loop_evidence_edges(
    *, render_node: str, critic_node: str
) -> tuple[FlowEdge, ...]:
    """Full loop including optional plan/fetch web-evidence on render and critic failure."""
    return (
        FlowEdge("render_loop", render_node),
        FlowEdge(render_node, "final_render", "success", conditional=True),
        FlowEdge(render_node, "render_fail_search", "fail", conditional=True),
        FlowEdge("render_fail_search", "evid_r", "yes", conditional=True),
        FlowEdge("evid_r", "rerender"),
        FlowEdge("rerender", "final_render", "success", conditional=True),
        FlowEdge("rerender", "render_loop", "fail", conditional=True),
        FlowEdge("render_fail_search", "render_loop", "no", conditional=True),
        FlowEdge("final_render", "critic_loop", "no", conditional=True),
        FlowEdge("critic_loop", critic_node),
        FlowEdge(critic_node, "done", "success", conditional=True),
        FlowEdge(critic_node, "critic_fail_search", "fail", conditional=True),
        FlowEdge("critic_fail_search", "evid_c", "yes", conditional=True),
        FlowEdge("evid_c", "recritic"),
        FlowEdge("recritic", "done", "success", conditional=True),
        FlowEdge("recritic", "critic_loop", "fail", conditional=True),
        FlowEdge("critic_fail_search", "render_loop", "no", conditional=True),
        FlowEdge("critic_loop", "render_loop", "exhausted", conditional=True),
        FlowEdge("render_loop", "exhausted", "exhausted", conditional=True),
    )


def facts_loop_flow(*, include_evidence: bool = False) -> FlowGraph:
    render_node = "render_facts"
    critic_node = "criticise_facts"
    if include_evidence:
        nodes = (
            FlowNode("start", "Unit start", "terminal"),
            FlowNode("ctx", "Resolve / apply<br/>ontology context"),
            FlowNode("render_loop", "render attempt<br/>1 … max_visits", "decision"),
            FlowNode(render_node, "Render facts"),
            FlowNode("render_fail_search", "initiate_search?", "decision"),
            FlowNode("evid_r", "Plan + fetch<br/>web evidence"),
            FlowNode("rerender", "Re-render facts"),
            FlowNode("final_render", "final render<br/>attempt?", "decision"),
            FlowNode("quarantine", "Surface unresolved<br/>quarantine"),
            FlowNode("done", "Return unit state", "terminal"),
            FlowNode("critic_loop", "critic attempt<br/>1 … max_visits", "decision"),
            FlowNode(critic_node, "Criticise facts"),
            FlowNode("critic_fail_search", "initiate_search?", "decision"),
            FlowNode("evid_c", "Plan + fetch<br/>web evidence"),
            FlowNode("recritic", "Re-criticise facts"),
            FlowNode("exhausted", "Return (retries exhausted)", "terminal"),
        )
        loop_edges = _atomic_loop_evidence_edges(
            render_node=render_node, critic_node=critic_node
        )
    else:
        nodes = (
            FlowNode("start", "Unit start", "terminal"),
            FlowNode("ctx", "Resolve / apply<br/>ontology context"),
            FlowNode("render_loop", "render attempt<br/>1 … max_visits", "decision"),
            FlowNode(render_node, "Render facts"),
            FlowNode("final_render", "final render<br/>attempt?", "decision"),
            FlowNode("quarantine", "Surface unresolved<br/>quarantine"),
            FlowNode("done", "Return unit state", "terminal"),
            FlowNode(critic_node, "Criticise facts"),
            FlowNode("exhausted", "Return (retries exhausted)", "terminal"),
        )
        loop_edges = _atomic_loop_core_edges(
            render_node=render_node, critic_node=critic_node
        )
    edges = (
        FlowEdge("start", "ctx"),
        FlowEdge("ctx", "render_loop"),
        *loop_edges,
        FlowEdge("final_render", "quarantine", "yes", conditional=True),
        FlowEdge("quarantine", "done"),
    )
    return FlowGraph(
        nodes=nodes,
        edges=edges,
        start_node="start",
        end_node="done",
    )


def ontology_loop_flow(*, include_evidence: bool = False) -> FlowGraph:
    render_node = "render_ontology"
    critic_node = "criticise_ontology"
    if include_evidence:
        nodes = (
            FlowNode("start", "Unit start", "terminal"),
            FlowNode("ctx", "Resolve unit<br/>ontology context"),
            FlowNode("render_loop", "render attempt<br/>1 … max_visits", "decision"),
            FlowNode(render_node, "Render ontology"),
            FlowNode("render_fail_search", "initiate_search?", "decision"),
            FlowNode("evid_r", "Plan + fetch<br/>web evidence"),
            FlowNode("rerender", "Re-render ontology"),
            FlowNode("final_render", "final render<br/>attempt?", "decision"),
            FlowNode("done", "Return unit state", "terminal"),
            FlowNode("critic_loop", "critic attempt<br/>1 … max_visits", "decision"),
            FlowNode(critic_node, "Criticise ontology"),
            FlowNode("critic_fail_search", "initiate_search?", "decision"),
            FlowNode("evid_c", "Plan + fetch<br/>web evidence"),
            FlowNode("recritic", "Re-criticise ontology"),
            FlowNode("exhausted", "Return (retries exhausted)", "terminal"),
        )
        loop_edges = _atomic_loop_evidence_edges(
            render_node=render_node, critic_node=critic_node
        )
    else:
        nodes = (
            FlowNode("start", "Unit start", "terminal"),
            FlowNode("ctx", "Resolve unit<br/>ontology context"),
            FlowNode("render_loop", "render attempt<br/>1 … max_visits", "decision"),
            FlowNode(render_node, "Render ontology"),
            FlowNode("final_render", "final render<br/>attempt?", "decision"),
            FlowNode("done", "Return unit state", "terminal"),
            FlowNode(critic_node, "Criticise ontology"),
            FlowNode("exhausted", "Return (retries exhausted)", "terminal"),
        )
        loop_edges = _atomic_loop_core_edges(
            render_node=render_node, critic_node=critic_node
        )
    edges = (
        FlowEdge("start", "ctx"),
        FlowEdge("ctx", "render_loop"),
        *loop_edges,
        FlowEdge("final_render", "done", "yes", conditional=True),
    )
    return FlowGraph(
        nodes=nodes,
        edges=edges,
        start_node="start",
        end_node="done",
    )


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


def _flow_label_for_graphviz(label: str, is_lr: bool) -> str:
    text = label.replace("<br/>", "\n")
    return _wrap_label(text) if is_lr else text


def draw_flow_graphviz(
    pgv_module: Any,
    flow: FlowGraph,
    fname: str,
    extensions: tuple[str, ...],
    rankdir: str = "TB",
) -> None:
    is_lr = rankdir == "LR"

    viz: Any = pgv_module.AGraph(directed=True, strict=False)
    viz.graph_attr.update(
        rankdir=rankdir,
        bgcolor=_BG_COLOR,
        pad="0.3" if is_lr else "0.6",
        nodesep="0.35" if is_lr else "0.7",
        ranksep="0.5" if is_lr else "0.9",
        fontname=_FONTNAME,
        splines="spline",
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

    for node in flow.nodes:
        label = _flow_label_for_graphviz(node.label, is_lr)
        if node.shape == "decision":
            viz.add_node(
                node.node_id,
                label=label,
                shape="diamond",
                fontsize="10" if is_lr else "11",
                margin="0.12,0.06" if is_lr else "0.18,0.10",
            )
        else:
            viz.add_node(node.node_id, label=label)

    for edge in flow.edges:
        if edge.conditional:
            viz.add_edge(
                edge.start,
                edge.end,
                label=edge.label,
                style="dashed",
                color=_COND_EDGE_COLOR,
                fontcolor=_COND_EDGE_COLOR,
            )
        else:
            viz.add_edge(edge.start, edge.end, label=edge.label)

    accent_attrs = {
        "fillcolor": _ACCENT_FILL,
        "color": _ACCENT_BORDER,
        "fontcolor": _ACCENT_BORDER,
        "penwidth": "2.5",
    }
    viz.get_node(flow.start_node).attr.update(**accent_attrs)
    viz.get_node(flow.end_node).attr.update(**accent_attrs)

    out = fname + f".{rankdir.lower()}" if rankdir.lower() == "lr" else fname
    for ext in extensions:
        if ext == "svg":
            viz.draw(out + ".svg", format="svg:cairo", prog="dot")
            print(f"📄 Wrote {out}.svg")
        elif ext == "png":
            viz.draw(out + ".png", format="png", prog="dot", args="-Gdpi=300")
            print(f"📄 Wrote {out}.png")


def write_atomic_loop_diagrams(pgv_module: Any) -> None:
    assets = Path("docs/assets")
    assets.mkdir(parents=True, exist_ok=True)
    for name, builder in (
        ("facts_loop", lambda: facts_loop_flow(include_evidence=False)),
        ("facts_loop_evidence", lambda: facts_loop_flow(include_evidence=True)),
        ("ontology_loop", lambda: ontology_loop_flow(include_evidence=False)),
        ("ontology_loop_evidence", lambda: ontology_loop_flow(include_evidence=True)),
    ):
        flow = builder()
        mmd_path = assets / f"{name}.mmd"
        mmd_path.write_text(flow_graph_to_mermaid(flow))
        print(f"📄 Wrote {mmd_path}")
        base = assets / name
        draw_flow_graphviz(pgv_module, flow, str(base), ("svg", "png"), rankdir="TB")
        draw_flow_graphviz(pgv_module, flow, str(base), ("svg", "png"), rankdir="LR")


def draw_graphviz(
    pgv_module: Any,
    graph: "Graph",
    fname: str,
    extensions: tuple[str, ...],
    rankdir: str = "TB",
) -> None:
    is_lr = rankdir == "LR"
    splines = "spline"

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
        write_atomic_loop_diagrams(pgv_module)
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
