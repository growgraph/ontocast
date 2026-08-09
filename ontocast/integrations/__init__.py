"""Adapters that expose OntoCast to other agent frameworks.

Kept out of ``ontocast/tool/``, which holds the stateful tools the ToolBox owns
and injects. These are outward-facing wrappers around those.

Nothing here is imported by the pipeline itself, so ``langchain_core.tools``
stays off the cold import path of ``import ontocast``.
"""

from typing import Any

__all__ = [
    "make_ontocast_node",
    "ontocast_tool_diagnostics",
    "ontocast_tool_names",
    "ontocast_tools",
    "text_in_turtle_out",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ontocast_tools": ("ontocast.integrations.langchain", "ontocast_tools"),
    "ontocast_tool_names": ("ontocast.integrations.langchain", "ontocast_tool_names"),
    "ontocast_tool_diagnostics": (
        "ontocast.integrations.langchain",
        "ontocast_tool_diagnostics",
    ),
    "make_ontocast_node": ("ontocast.integrations.langgraph", "make_ontocast_node"),
    "text_in_turtle_out": ("ontocast.integrations.langgraph", "text_in_turtle_out"),
}


def __getattr__(name: str) -> Any:
    """Resolve the integration entry points on first access."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_name, attribute = target
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
