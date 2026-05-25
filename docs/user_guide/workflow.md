# OntoCast Workflow

This document describes the document processing pipeline implemented in `stategraph/create.py`.

## Overview

OntoCast transforms input documents into RDF ontology and facts graphs through a **parallel map/reduce** pipeline:

1. **Document conversion** — PDF, DOCX, TXT, MD, or JSON → Markdown
2. **Semantic chunking** — split into content units (optionally limited with `--head-chunks`)
3. **Ontology map/reduce** (when `render_mode` includes ontology):
   - Per-unit context assembly (catalog selection or vector retrieval)
   - Render/critic loops with optional web evidence
   - Global normalize (provenance split) → optional consolidate → structural check → consistency critic
4. **Facts map/reduce** (when `render_mode` includes facts):
   - Per-unit render/critic loops
   - Merge facts across units with entity disambiguation
5. **Serialize** — write to triple store and return Turtle in the API response

## Document-Level Graph

The LangGraph compiled by `create_agent_graph()` is rendered from the live workflow. Regenerate after graph changes:

```bash
uv run plot-graph
```

Outputs (under `docs/assets/`):

| File | Layout | Description |
|------|--------|-------------|
| [graph.png](../assets/graph.png) | Top-to-bottom | Full document pipeline (default) |
| [graph.lr.png](../assets/graph.lr.png) | Left-to-right | Same graph, landscape layout |
| [graph.svg](../assets/graph.svg) / [graph.lr.svg](../assets/graph.lr.svg) | Vector | Scalable versions |
| [graph.preview.png](../assets/graph.preview.png) | Mermaid API | Small hand-drawn preview (optional) |
| [graph.mmd](../../graph.mmd) | Mermaid source | Editable source at repo root |

![Document workflow (TB)](../assets/graph.png)

<details>
<summary>Landscape layout (LR)</summary>

![Document workflow (LR)](../assets/graph.lr.png)

</details>

Nodes such as **Update Ontology** and **Render Facts** each run the per-unit atomic loop below (in parallel across content units).

## Per-Unit Atomic Loop

Inside `stategraph/atomic.py`, each content unit runs an independent **render → critic** loop with optional web evidence. The same pattern applies to ontology (`ontology_loop`) and facts (`facts_loop`).

```mermaid
flowchart TD
  START([Unit start]) --> CTX[Resolve ontology context]
  CTX --> RLOOP{render attempt<br/>1 … max_visits}
  RLOOP --> RENDER[Render GraphUpdate]
  RENDER -->|success| FINAL{final render<br/>attempt?}
  RENDER -->|fail| SEARCH_R{initiate_search?}
  SEARCH_R -->|yes| EVID_R[Plan + fetch web evidence]
  EVID_R --> RENDER2[Re-render]
  RENDER2 -->|success| FINAL
  RENDER2 -->|fail| RLOOP
  SEARCH_R -->|no| RLOOP
  FINAL -->|yes| DONE([Return unit state])
  FINAL -->|no| CLOOP{critic attempt<br/>1 … max_visits}
  CLOOP --> CRITIC[Criticise output]
  CRITIC -->|success| DONE
  CRITIC -->|fail| SEARCH_C{initiate_search?}
  SEARCH_C -->|yes| EVID_C[Plan + fetch web evidence]
  EVID_C --> CRITIC2[Re-criticise]
  CRITIC2 -->|success| DONE
  CRITIC2 -->|fail| CLOOP
  SEARCH_C -->|no| CLOOP
  CLOOP -->|exhausted| RLOOP
  RLOOP -->|exhausted| FAIL([Return with failure])
```

Notes:

- First render/critic pass always runs **without** web search; search runs only when the node sets `initiate_search`.
- On the **last allowed render attempt**, the critic is skipped (no further extract to critique).
- `/process_unit` runs this loop on a single unit via `unit_pipeline.py` (no chunking or document-level reduce).

Implementation: [`stategraph/atomic.py`](../../ontocast/stategraph/atomic.py).

## Stage Details

### 1. Document Input

- Accepts text, JSON (`text` field), or file uploads via `/process`
- Converts supported formats to Markdown while preserving structure

### 2. Chunking

- Semantic chunking splits the document into **content units**
- Units are processed **in parallel** up to `PARALLEL_WORKERS`
- Use `--head-chunks N` on the CLI to process only the first N units (testing)

### 3. Per-Unit Ontology Loop

Each content unit runs an independent **ontology loop** (`stategraph/atomic.py`):

1. **Context assembly** — pick or retrieve ontology context for the unit:
   - LLM catalog selection (`selected_single_ontology`)
   - Qdrant vector ensemble (`selected_vector_search_ontology`)
   - Fixed catalog ontology (`fixed_single_ontology`)
2. **Render** — LLM emits `GraphUpdate` operations (Turtle or JSON-LD wire format)
3. **Critic** — validate structure; retry up to `max_visits` (config or per-request override)
4. **External evidence** (optional) — web search on retry when the node requests it

See [Ontology Context](ontology_context.md) and [User Instructions](user_instructions.md).

### 4. Ontology Reduce (Document Level)

After all units finish:

| Stage | Purpose |
|-------|---------|
| **Normalize** | Merge unit deltas; split RDF 1.2 provenance/reification into a side artifact |
| **Consolidate** (optional) | Single-pass refinement when `ENABLE_ONTOLOGY_CONSOLIDATION=true` |
| **Structural check** | Connectivity and schema validation |
| **Consistency critic** | Cross-unit ontology consistency |

Provenance triples (`prov:`, reification, chunk metadata) are kept in `ontology_provenance_artifact`, not in the working ontology graph passed to consolidation.

### 5. Per-Unit Facts Loop

When facts rendering is enabled, each unit runs a **facts loop** (render → critic, with optional web evidence), then **merge facts** applies cross-chunk entity disambiguation and aggregation.

### 6. Output

- Ontology and facts serialized to the configured triple store
- API returns Turtle (optionally with `strip_provenance=true` to omit reification scaffolding)
- Budget summary logged (LLM calls, characters, triple counts)

## Configuration

| Setting / parameter | Effect |
|---------------------|--------|
| `RENDER_MODE` | `ontology`, `facts`, or `ontology_and_facts` |
| `PARALLEL_WORKERS` | Max concurrent unit workers |
| `MAX_VISITS` / `max_visits` | Render/critic retry budget per loop |
| `ENABLE_ONTOLOGY_CONSOLIDATION` | Optional post-normalization consolidation |
| `ONTOLOGY_CONTEXT_MODE` | How per-unit ontology context is sourced |
| `LLM_GRAPH_FORMAT` | `turtle` or `jsonld` LLM wire encoding |
| `--head-chunks` | CLI limit on units processed |

Full reference: [Configuration System](configuration.md).

## Best Practices

1. **Start with defaults** — `MAX_VISITS=1`, `ontology_and_facts`, consolidation off; tune after inspecting output.
2. **Use `--head-chunks`** for large documents during development.
3. **Monitor budget summaries** to estimate LLM cost at scale.
4. **Provide seed ontologies** in `ONTOCAST_ONTOLOGY_DIRECTORY` for catalog selection modes.
5. **Enable vector mode** only when Qdrant and embeddings are configured.

## Next Steps

- [Core Concepts](concepts.md) — GraphUpdate, provenance, disambiguation
- [API Endpoints](api.md) — `/process`, `/process_unit`, parameters
- [API Reference](../reference/onto/state.md) — `AgentState` and workflow types
