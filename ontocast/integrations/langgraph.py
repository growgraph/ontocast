"""Embed the OntoCast pipeline as a node in someone else's LangGraph.

``AgentState`` declares no ``Annotated[..., reducer]`` channels and every node
returns the whole state, so adding the compiled graph directly to a parent
``StateGraph`` only works when the parent's state literally has ``raw_input``,
``docling_doc``, ``aggregated_facts`` and the rest. ``input_schema`` and
``output_schema`` narrow which of ``AgentState``'s own keys cross the boundary
but cannot rename them, so they do not bridge a foreign state either.

:func:`make_ontocast_node` therefore asks for the mapping explicitly. That is
30 lines of adapter instead of a reducer refactor, and it is honest about where
the boundary is.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from ontocast.onto.state import AgentState
from ontocast.stategraph.create import create_agent_graph

if TYPE_CHECKING:
    from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)


def make_ontocast_node(
    tools: "ToolBox",
    *,
    to_agent_state: Callable[[Any], AgentState],
    from_agent_state: Callable[[AgentState, Any], dict[str, Any]],
    recursion_limit: int | None = None,
    graph: CompiledStateGraph | None = None,
) -> Callable[[Any, RunnableConfig], Awaitable[dict[str, Any]]]:
    """Build a node that runs the OntoCast pipeline inside another graph.

    ```python
    to_state, from_state = text_in_turtle_out()
    node = make_ontocast_node(tools, to_agent_state=to_state, from_agent_state=from_state)

    builder = StateGraph(MyState)
    builder.add_node("extract", node)
    ```

    Args:
        tools: The dependency container. The graph is compiled once here, not
            per invocation.
        to_agent_state: Maps the parent state to a fresh ``AgentState``.
        from_agent_state: Maps the finished ``AgentState`` and the original
            parent state to a parent-state delta.
        recursion_limit: LangGraph recursion limit for the inner run. Defaults
            to a value derived from the configured chunk budget. **Leaving this
            unset is safer than passing LangGraph's default of 25**, which a
            multi-chunk document exceeds.
        graph: A pre-compiled OntoCast graph to reuse instead of compiling one.

    Returns:
        An async node callable suitable for ``StateGraph.add_node``.
    """
    compiled = graph or create_agent_graph(tools, name="ontocast")
    limit = recursion_limit if recursion_limit is not None else _default_limit(tools)

    async def ontocast_node(state: Any, config: RunnableConfig) -> dict[str, Any]:
        initial = to_agent_state(state)
        # Merge rather than replace: the caller's config carries callbacks,
        # tags and run metadata that tracing depends on.
        merged: RunnableConfig = {**(config or {})}
        merged["recursion_limit"] = merged.get("recursion_limit") or limit

        # `ainvoke`, not `astream`: the HTTP layer streams because it wants
        # intermediate node output, and an embedded node does not.
        result = await compiled.ainvoke(initial, merged)

        # LangGraph hands back a plain dict for a pydantic state schema.
        final = (
            result
            if isinstance(result, AgentState)
            else AgentState.model_validate(result)
        )
        return from_agent_state(final, state)

    return ontocast_node


def text_in_turtle_out(
    *,
    text_key: str = "input",
    ontology_key: str = "ontology_ttl",
    facts_key: str = "facts_ttl",
) -> tuple[Callable[[Any], AgentState], Callable[[AgentState, Any], dict[str, Any]]]:
    """Return a ready-made mapping pair for the common text-to-Turtle case.

    Reads a string off the parent state and writes back two Turtle strings, so
    a parent state needs only those three plain keys.

    Args:
        text_key: Parent-state key holding the source text.
        ontology_key: Parent-state key to write the ontology Turtle to.
        facts_key: Parent-state key to write the facts Turtle to.

    Returns:
        The ``(to_agent_state, from_agent_state)`` pair.
    """

    def to_agent_state(state: Any) -> AgentState:
        text = _read_key(state, text_key)
        if not isinstance(text, str):
            raise TypeError(
                f"Expected a string at {text_key!r}, got {type(text).__name__}"
            )
        return AgentState(raw_input={f"{text_key}.txt": text.encode("utf-8")})

    def from_agent_state(final: AgentState, _parent: Any) -> dict[str, Any]:
        ontology_ttl = ""
        artifacts = [o for o in final.reduced_ontology_artifacts if not o.is_null()]
        if artifacts:
            ontology_ttl = artifacts[0].graph.serialize_canonical_turtle()
        return {
            ontology_key: ontology_ttl,
            facts_key: final.aggregated_facts.serialize_canonical_turtle(),
        }

    return to_agent_state, from_agent_state


def _read_key(state: Any, key: str) -> Any:
    """Read ``key`` from a dict-like or attribute-style state object."""
    if isinstance(state, dict):
        return state.get(key)
    return getattr(state, key, None)


def _default_limit(tools: "ToolBox") -> int:
    """Derive a recursion limit from the configured chunk budget."""
    from ontocast.api.process_helpers import calculate_recursion_limit

    server_config = tools.config.server
    return calculate_recursion_limit(
        None,
        server_config,
        max_visits_per_node=server_config.max_visits_per_node,
    )
