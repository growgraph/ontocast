"""Expose OntoCast capabilities as LangChain tools.

``ontocast_tools(tools)`` returns a list of ``BaseTool`` objects that any
LangChain or LangGraph agent can call:

```python
from langchain.agents import create_agent
from ontocast import Config, ToolBox, ontocast_tools

tools = await ToolBox.acreate(Config.in_memory())
await tools.initialize()

agent = create_agent(model, tools=[*ontocast_tools(tools)])
```

Two design rules run through this module.

**Capability gating.** A tool whose backend is missing is not returned, rather
than returned and made to fail on first call. A base install has no Qdrant, no
docling, and possibly no SPARQL-capable store, and an agent handed a tool that
always errors will keep retrying it. :func:`ontocast_tool_diagnostics` explains
each omission, since "my agent has six tools instead of eleven" is otherwise
unbreakable.

**Mutation is opt-in.** The write tools are excluded unless ``mutating=True``.
``ontocast_delete_ontology`` drops a named graph, unlinks a file from disk, and
deletes vectors -- three irreversible effects from one model-chosen string.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel

from ontocast.integrations.schemas import (
    AlignEntitiesArgs,
    ApplyGraphUpdateArgs,
    ChunkTextArgs,
    ConvertDocumentArgs,
    DeleteOntologyArgs,
    ExtractArgs,
    GetOntologyArgs,
    IngestOntologyArgs,
    NoArgs,
    RetrieveOntologyContextArgs,
    SearchOntologyTermsArgs,
    SparqlQueryArgs,
)
from ontocast.integrations.serialize import (
    graph_to_llm_text,
    json_to_llm_text,
    models_to_llm_text,
    truncate,
)
from ontocast.onto.enum import LLMGraphFormat
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import GraphUpdate, TripleOp
from ontocast.onto.state import AgentState
from ontocast.onto.tenancy import StoreKind
from ontocast.util.optional import is_available

if TYPE_CHECKING:
    from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)

#: Read-only tools, offered whenever their backend is available.
READ_TOOLS = (
    "ontocast_list_ontologies",
    "ontocast_get_ontology",
    "ontocast_search_ontology_terms",
    "ontocast_retrieve_ontology_context",
    "ontocast_sparql_select",
    "ontocast_sparql_construct",
    "ontocast_chunk_text",
    "ontocast_extract",
)

#: Tools that change stored state. Excluded unless ``mutating=True``.
MUTATING_TOOLS = (
    "ontocast_apply_graph_update",
    "ontocast_ingest_ontology_ttl",
    "ontocast_delete_ontology",
)

#: Tools returned only when named in ``include``. ``convert_document`` takes a
#: model-chosen filesystem path, which is a read primitive on the host;
#: ``align_entities`` is a specialist operation that would dilute tool choice.
OPT_IN_TOOLS = (
    "ontocast_convert_document",
    "ontocast_align_entities",
)

ALL_TOOL_NAMES = READ_TOOLS + MUTATING_TOOLS + OPT_IN_TOOLS

# Read-only SPARQL keywords; anything else is refused before reaching the store.
_UPDATE_KEYWORDS = (
    "insert",
    "delete",
    "drop",
    "clear",
    "load",
    "create",
    "add",
    "move",
    "copy",
)


def ontocast_tools(
    tools: "ToolBox",
    *,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    mutating: bool = False,
    max_chars: int = 20_000,
) -> list[BaseTool]:
    """Wrap OntoCast capabilities as LangChain structured tools.

    Only tools whose backend is installed and configured are returned; call
    :func:`ontocast_tool_diagnostics` to see why something is missing.

    All tools are async-only. Agents must invoke them with ``ainvoke``; a
    synchronous ``invoke`` raises ``NotImplementedError``. Several of the
    underlying calls are coroutines already, and the rest are CPU-heavy enough
    that running them on the caller's event loop would stall it.

    Args:
        tools: A constructed ToolBox. Call ``await tools.initialize()`` first if
            you want the ontology catalog populated.
        include: Restrict to these tool names. ``None`` selects the default set
            (read tools, plus mutating tools when ``mutating`` is true).
            Naming a tool here also opts into the ``OPT_IN_TOOLS``.
        exclude: Drop these names from whatever ``include`` selected.
        mutating: Include the write tools. Off by default; each one changes
            stored state irreversibly.
        max_chars: Truncation budget applied to each tool's rendered result.

    Returns:
        Available tools in a stable order.

    Raises:
        ValueError: If ``include`` or ``exclude`` names an unknown tool.
    """
    requested = _resolve_requested(include, exclude, mutating)
    builders = _builders(tools, max_chars=max_chars)

    built: list[BaseTool] = []
    for name in ALL_TOOL_NAMES:
        if name not in requested:
            continue
        reason = _unavailable_reason(name, tools)
        if reason is not None:
            logger.info("Skipping %s: %s", name, reason)
            continue
        built.append(builders[name]())
    return built


def ontocast_tool_names(
    tools: "ToolBox",
    *,
    mutating: bool = False,
) -> list[str]:
    """Return the names :func:`ontocast_tools` would produce, without building them."""
    requested = _resolve_requested(None, None, mutating)
    return [
        name
        for name in ALL_TOOL_NAMES
        if name in requested and _unavailable_reason(name, tools) is None
    ]


def ontocast_tool_diagnostics(tools: "ToolBox") -> dict[str, str]:
    """Explain why each unavailable tool is unavailable.

    Args:
        tools: The ToolBox the tools would be built against.

    Returns:
        Mapping of tool name to the reason it would be skipped. Tools that are
        available are absent from the mapping.
    """
    reasons: dict[str, str] = {}
    for name in ALL_TOOL_NAMES:
        reason = _unavailable_reason(name, tools)
        if reason is not None:
            reasons[name] = reason
    return reasons


def _resolve_requested(
    include: Iterable[str] | None,
    exclude: Iterable[str] | None,
    mutating: bool,
) -> set[str]:
    known = set(ALL_TOOL_NAMES)
    if include is None:
        requested = set(READ_TOOLS)
        if mutating:
            requested |= set(MUTATING_TOOLS)
    else:
        requested = set(include)
        unknown = requested - known
        if unknown:
            raise ValueError(
                f"Unknown tool name(s): {', '.join(sorted(unknown))}. "
                f"Available: {', '.join(ALL_TOOL_NAMES)}"
            )
    if exclude is not None:
        dropped = set(exclude)
        unknown = dropped - known
        if unknown:
            raise ValueError(
                f"Unknown tool name(s) in exclude: {', '.join(sorted(unknown))}"
            )
        requested -= dropped
    return requested


def _unavailable_reason(name: str, tools: "ToolBox") -> str | None:
    """Return why ``name`` cannot be offered, or None if it can."""
    store = tools.triple_store_manager

    if name == "ontocast_sparql_select" and not store.supports_sparql_select():
        return (
            f"{type(store).__name__} does not support SPARQL SELECT; "
            "use the in-memory or Fuseki triple store"
        )
    if name == "ontocast_sparql_construct" and not store.supports_sparql_construct():
        return (
            f"{type(store).__name__} does not support SPARQL CONSTRUCT; "
            "use the in-memory or Fuseki triple store"
        )
    if name == "ontocast_search_ontology_terms" and tools.vector_store is None:
        return "no vector store is configured (set QDRANT_URI or LANCEDB_ENABLED)"
    if name == "ontocast_retrieve_ontology_context" and tools.patch_retriever is None:
        return "no ontology patch retriever is configured (needs a vector store)"
    if name in (
        "ontocast_convert_document",
        "ontocast_chunk_text",
    ) and not is_available("docling_core"):
        return 'requires docling-core; install with pip install "ontocast[documents]"'
    return None


def _builders(tools: "ToolBox", *, max_chars: int) -> dict[str, Callable[[], BaseTool]]:
    """Return a lazy builder per tool name.

    Builders are thunks so that only the selected tools are constructed.
    """
    return {
        "ontocast_list_ontologies": lambda: _list_ontologies(tools, max_chars),
        "ontocast_get_ontology": lambda: _get_ontology(tools, max_chars),
        "ontocast_search_ontology_terms": lambda: _search_terms(tools, max_chars),
        "ontocast_retrieve_ontology_context": lambda: _retrieve_context(
            tools, max_chars
        ),
        "ontocast_sparql_select": lambda: _sparql_select(tools, max_chars),
        "ontocast_sparql_construct": lambda: _sparql_construct(tools, max_chars),
        "ontocast_chunk_text": lambda: _chunk_text(tools, max_chars),
        "ontocast_extract": lambda: _extract(tools, max_chars),
        "ontocast_apply_graph_update": lambda: _apply_graph_update(tools, max_chars),
        "ontocast_ingest_ontology_ttl": lambda: _ingest_ontology(tools, max_chars),
        "ontocast_delete_ontology": lambda: _delete_ontology(tools, max_chars),
        "ontocast_convert_document": lambda: _convert_document(tools, max_chars),
        "ontocast_align_entities": lambda: _align_entities(tools, max_chars),
    }


def _tool(
    *,
    name: str,
    description: str,
    args_schema: type[BaseModel],
    coroutine: Callable[..., Any],
) -> BaseTool:
    return StructuredTool.from_function(
        coroutine=coroutine,
        name=name,
        description=description,
        args_schema=args_schema,
    )


# -- read tools ------------------------------------------------------------


def _list_ontologies(tools: "ToolBox", max_chars: int) -> BaseTool:
    async def run() -> str:
        headers = await tools.ontology_manager.aget_catalog_headers()
        return models_to_llm_text(
            headers,
            max_chars=max_chars,
            fields=(
                "iri",
                "ontology_id",
                "title",
                "description",
                "version",
                "hash",
            ),
        )

    return _tool(
        name="ontocast_list_ontologies",
        description=(
            "List every ontology available in the knowledge base, with its IRI, "
            "short id, title, description and version. Call this first to "
            "discover what vocabularies exist before querying or editing them."
        ),
        args_schema=NoArgs,
        coroutine=run,
    )


def _get_ontology(tools: "ToolBox", max_chars: int) -> BaseTool:
    async def run(iri: str) -> str:
        ontologies = await tools.ontology_manager.aget_ontologies_by_iri([iri])
        if not ontologies:
            return f"# no ontology found with IRI {iri}"
        return graph_to_llm_text(
            ontologies[0].graph, max_chars=max_chars, sources=[iri]
        )

    return _tool(
        name="ontocast_get_ontology",
        description=(
            "Fetch the full definition of one ontology as Turtle, given its IRI. "
            "Use ontocast_list_ontologies to find valid IRIs. For a large "
            "ontology prefer ontocast_retrieve_ontology_context, which returns "
            "only the relevant part."
        ),
        args_schema=GetOntologyArgs,
        coroutine=run,
    )


def _search_terms(tools: "ToolBox", max_chars: int) -> BaseTool:
    async def run(
        query: str, top_k: int | None = None, filter_iri: str | None = None
    ) -> str:
        store = tools.require_vector_store()
        hits = await asyncio.to_thread(
            store.search_patch_hits, query, top_k, filter_iri
        )
        rows = [
            {
                "iri": hit.atom.iri,
                "ontology_iri": hit.atom.ontology_iri,
                "label": hit.atom.core_representation,
                "entity_role": hit.atom.entity_role,
                "score": round(hit.score, 4),
            }
            for hit in hits
        ]
        return json_to_llm_text(rows, max_chars=max_chars)

    return _tool(
        name="ontocast_search_ontology_terms",
        description=(
            "Find ontology classes and properties matching a natural-language "
            "description, ranked by relevance. Returns individual terms with "
            "their IRIs. Use this to discover which existing terms to reuse "
            "before minting new ones."
        ),
        args_schema=SearchOntologyTermsArgs,
        coroutine=run,
    )


def _retrieve_context(tools: "ToolBox", max_chars: int) -> BaseTool:
    async def run(
        query: str,
        top_k: int | None = None,
        subgraph_depth: int | None = None,
        max_total_triples: int | None = None,
    ) -> str:
        retriever = tools.require_patch_retriever()
        graph, sources = await retriever.aretrieve(
            query,
            top_k=top_k,
            subgraph_depth=subgraph_depth,
            max_total_triples=max_total_triples,
        )
        return graph_to_llm_text(graph, max_chars=max_chars, sources=sources)

    return _tool(
        name="ontocast_retrieve_ontology_context",
        description=(
            "Retrieve the region of the ontology relevant to a natural-language "
            "query, as Turtle, including the neighbourhood around each matching "
            "term. This is the best way to understand existing modelling before "
            "extending it."
        ),
        args_schema=RetrieveOntologyContextArgs,
        coroutine=run,
    )


def _reject_update_query(query: str) -> None:
    """Refuse anything that is not a read-only SPARQL form.

    The SPARQL tools are read-only by contract. Letting an UPDATE through here
    would hand every agent an unguarded ``DELETE WHERE { ?s ?p ?o }``, which is
    exactly the shape a prompt injection needs; deliberate writes go through
    ontocast_apply_graph_update, which is validated and triple-capped.
    """
    stripped = "\n".join(
        line for line in query.splitlines() if not line.strip().startswith("#")
    )
    lowered = stripped.lower()
    for keyword in _UPDATE_KEYWORDS:
        if f"{keyword} " in lowered or f"{keyword}\n" in lowered:
            raise ValueError(
                f"Refusing to run a SPARQL {keyword.upper()}: this tool is "
                "read-only. Use ontocast_apply_graph_update to change data."
            )


def _sparql_select(tools: "ToolBox", max_chars: int) -> BaseTool:
    async def run(query: str, store: StoreKind = "ontologies") -> str:
        _reject_update_query(query)
        rows = await tools.triple_store_manager.aselect(query, store=store)
        return json_to_llm_text(rows, max_chars=max_chars)

    return _tool(
        name="ontocast_sparql_select",
        description=(
            "Run a read-only SPARQL SELECT or ASK query and return the result "
            "rows as JSON. Rows carry lexical values only -- term kind and "
            "datatype are not preserved, so constrain kinds in the query itself "
            "(for example FILTER(isIRI(?x))). Update queries are rejected."
        ),
        args_schema=SparqlQueryArgs,
        coroutine=run,
    )


def _sparql_construct(tools: "ToolBox", max_chars: int) -> BaseTool:
    async def run(query: str, store: StoreKind = "ontologies") -> str:
        _reject_update_query(query)
        graph = await tools.triple_store_manager.aconstruct(query, store=store)
        return graph_to_llm_text(graph, max_chars=max_chars)

    return _tool(
        name="ontocast_sparql_construct",
        description=(
            "Run a read-only SPARQL CONSTRUCT or DESCRIBE query and return the "
            "resulting graph as Turtle. Unlike SELECT, this preserves real RDF "
            "terms including blank nodes and datatypes. Update queries are "
            "rejected."
        ),
        args_schema=SparqlQueryArgs,
        coroutine=run,
    )


def _chunk_text(tools: "ToolBox", max_chars: int) -> BaseTool:
    async def run(text: str) -> str:
        chunks = await asyncio.to_thread(tools.chunker.size_text, text)
        return json_to_llm_text(chunks, max_chars=max_chars)

    return _tool(
        name="ontocast_chunk_text",
        description=(
            "Split a long document into size-bounded chunks suitable for "
            "extraction. Use this before calling ontocast_extract on text that "
            "is too large to process at once."
        ),
        args_schema=ChunkTextArgs,
        coroutine=run,
    )


def _extract(tools: "ToolBox", max_chars: int) -> BaseTool:
    async def run(
        text: str,
        render_mode: str | None = None,
        instruction: str = "",
        domain: str | None = None,
    ) -> str:
        from ontocast.api.parse import parse_render_mode_param
        from ontocast.api.process_helpers import select_unit_facts_ontology_graph
        from ontocast.stategraph.unit_pipeline import run_unit_pipeline

        state = AgentState(
            raw_input={"input.txt": text.encode("utf-8")},
            render_mode=parse_render_mode_param(
                render_mode, tools.config.server.render_mode
            ),
            ontology_user_instruction=instruction,
            facts_user_instruction=instruction,
            **({"current_domain": domain} if domain else {}),
        )
        onto_result, facts_result = await run_unit_pipeline(state, tools)

        parts: list[str] = []
        if onto_result is not None:
            ontology_graph = (
                onto_result.fresh_ontology.graph
                if onto_result.fresh_ontology is not None
                and not onto_result.fresh_ontology.is_null()
                else onto_result.working_graph
            )
            parts.append("# --- ontology ---")
            parts.append(graph_to_llm_text(ontology_graph, max_chars=max_chars))
        if facts_result is not None:
            # Reuse the aggregation the HTTP path applies, rather than reading
            # the raw per-unit graph: it is what reconciles entities and mints
            # document-scoped IRIs.
            facts_graph = tools.aggregator.postprocess_facts_units(
                units=[facts_result.content_unit],
                ontology_graph=select_unit_facts_ontology_graph(
                    onto_result, facts_result
                ),
                doc_iri=state.doc_iri,
                document_metadata={},
                doc_namespace=state.doc_namespace,
            ).graph
            parts.append("# --- facts ---")
            parts.append(graph_to_llm_text(facts_graph, max_chars=max_chars))
        parts.append("# --- budget ---")
        parts.append(
            json_to_llm_text(
                state.budget_tracker.model_dump(mode="json"), max_chars=2_000
            )
        )
        return "\n".join(parts)

    return _tool(
        name="ontocast_extract",
        description=(
            "Run the OntoCast extraction pipeline over a passage of text and "
            "return the ontology and/or facts it produces as Turtle. This is the "
            "main entry point: it handles ontology context retrieval, extraction "
            "and criticism internally. Text is treated as a single unit, so "
            "chunk long documents first."
        ),
        args_schema=ExtractArgs,
        coroutine=run,
    )


# -- mutating tools --------------------------------------------------------


def _apply_graph_update(tools: "ToolBox", max_chars: int) -> BaseTool:
    async def run(
        insert_ttl: str = "",
        delete_ttl: str = "",
        base_ttl: str | None = None,
        target: str = "ontology",
        persist: bool = False,
    ) -> str:
        if not insert_ttl.strip() and not delete_ttl.strip():
            raise ValueError("Provide at least one of insert_ttl or delete_ttl.")

        base = RDFGraph()
        if base_ttl:
            base.parse(data=base_ttl, format="turtle")

        # This tool's own interface is Turtle-in (`insert_ttl`/`delete_ttl`),
        # independent of LLM_GRAPH_FORMAT: the caller is an agent writing
        # Turtle by hand, not a provider emitting a structured payload. Pin the
        # coercion format explicitly so it does not track the wire default.
        ops: list[TripleOp] = []
        for op_type, ttl in (("delete", delete_ttl), ("insert", insert_ttl)):
            if ttl.strip():
                ops.append(
                    TripleOp.model_validate(
                        {"type": op_type, "graph": ttl},
                        context={"llm_graph_format": LLMGraphFormat.TURTLE},
                    )
                )

        before = len(base)
        updated, applied = AgentState.render_updated_graph(
            base,
            [GraphUpdate(triple_operations=ops)],
            max_triples=tools.config.server.ontology_max_triples,
        )

        persisted = False
        if persist and applied and tools.triple_store_manager is not None:
            await tools.triple_store_manager.aserialize(updated)
            persisted = True

        summary = {
            "applied": applied,
            "persisted": persisted,
            "target": target,
            "triples_before": before,
            "triples_after": len(updated),
        }
        if not applied:
            summary["reason"] = (
                "update would exceed the configured ontology triple limit; "
                "nothing was changed"
            )
        return "\n".join(
            [
                graph_to_llm_text(updated, max_chars=max_chars),
                "# --- result ---",
                json_to_llm_text(summary, max_chars=1_000),
            ]
        )

    return _tool(
        name="ontocast_apply_graph_update",
        description=(
            "Apply an insert and/or delete patch to a graph and return the "
            "result as Turtle. Supply Turtle fragments, including any prefixes "
            "you use. Set persist=true to write the result to the triple store; "
            "otherwise the patch is applied and returned without being stored."
        ),
        args_schema=ApplyGraphUpdateArgs,
        coroutine=run,
    )


def _ingest_ontology(tools: "ToolBox", max_chars: int) -> BaseTool:
    async def run(ttl: str, filename: str | None = None) -> str:
        ontology = await tools.ingest_ontology_ttl(
            ttl.encode("utf-8"), filename=filename
        )
        return json_to_llm_text(
            {
                "iri": ontology.iri,
                "ontology_id": ontology.ontology_id,
                "title": ontology.title,
                "version": ontology.version,
                "hash": ontology.hash,
                "triples": len(ontology.graph),
            },
            max_chars=max_chars,
        )

    return _tool(
        name="ontocast_ingest_ontology_ttl",
        description=(
            "Register a new ontology from Turtle: writes it to the ontology "
            "directory, stores it in the triple store, and indexes it for "
            "search. Use this to add a vocabulary, not to patch an existing "
            "one -- for that use ontocast_apply_graph_update."
        ),
        args_schema=IngestOntologyArgs,
        coroutine=run,
    )


def _delete_ontology(tools: "ToolBox", max_chars: int) -> BaseTool:
    async def run(iri: str) -> str:
        await tools.delete_ontology_by_iri(iri)
        return json_to_llm_text({"deleted": iri}, max_chars=max_chars)

    return _tool(
        name="ontocast_delete_ontology",
        description=(
            "Permanently delete an ontology: removes its named graph from the "
            "triple store, deletes its file from the ontology directory, and "
            "drops its search vectors. This cannot be undone. Confirm the IRI "
            "with ontocast_list_ontologies before calling."
        ),
        args_schema=DeleteOntologyArgs,
        coroutine=run,
    )


# -- opt-in tools ----------------------------------------------------------


def _convert_document(tools: "ToolBox", max_chars: int) -> BaseTool:
    async def run(path: str) -> str:
        import pathlib

        doc = await asyncio.to_thread(tools.converter, pathlib.Path(path))
        return truncate(doc.export_to_markdown(), max_chars=max_chars)

    return _tool(
        name="ontocast_convert_document",
        description=(
            "Convert a document file (PDF, DOCX, HTML) at a filesystem path into "
            "markdown text."
        ),
        args_schema=ConvertDocumentArgs,
        coroutine=run,
    )


def _align_entities(tools: "ToolBox", max_chars: int) -> BaseTool:
    async def run(graphs: list[dict[str, str]], regime: str = "ontology_loose") -> str:
        from ontocast.tool.agg.match_models import MatchRegime, TaggedGraph

        tagged = []
        for entry in graphs:
            graph = RDFGraph()
            graph.parse(data=entry["ttl"], format="turtle")
            tagged.append(TaggedGraph(id=entry["name"], graph=graph))

        aligner = tools.get_entity_aligner()
        result = await asyncio.to_thread(
            aligner.align_graphs, tagged, regime=MatchRegime(regime)
        )
        return json_to_llm_text(
            result.model_dump(mode="json"),
            max_chars=max_chars,
        )

    return _tool(
        name="ontocast_align_entities",
        description=(
            "Find equivalent entities across two or more RDF graphs, returning "
            "the matched pairs and their similarity evidence."
        ),
        args_schema=AlignEntitiesArgs,
        coroutine=run,
    )
