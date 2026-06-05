# Concepts

Main concepts in OntoCast, a framework for transforming documents into semantic triples.

## Ontology Management

OntoCast manages ontologies with automatic versioning and timestamp tracking:

- **Semantic Versioning**: MAJOR/MINOR/PATCH increments from change analysis
- **Hash-Based Lineage**: Parent hashes track ontology evolution
- **Multiple Versions**: Stored as separate named graphs in Fuseki
- **Timestamp Tracking**: `updated_at` synced as `dcterms:modified`
- **Versioned IRIs**: Unique IRIs with hash fragments for storage

## GraphUpdate System

Token-efficient incremental graph modifications:

- **Structured Operations**: LLM outputs `GraphUpdate` with ordered `TripleOp` insert/delete patches
- **Wire Formats**: Turtle strings or compact JSON-LD (`LLM_GRAPH_FORMAT`); canonical runtime models are the same
- **Internal compilation**: Triple patches compile to rdflib UPDATE queries at apply time
- **Token Savings**: Typically 80–95% fewer output tokens vs full graph regeneration

## RDF 1.2 Provenance

OntoCast uses **pyoxigraph** for RDF 1.2 quoted-triple syntax and separates provenance from the working ontology:

- During **normalization**, reification triples, `prov:wasDerivedFrom`, chunk metadata, and alignment artifacts (`owl:sameAs`) move to a **provenance artifact**
- The clean ontology graph feeds consolidation and serialization
- API clients can pass `strip_provenance=true` to omit reification scaffolding from returned Turtle

See [Workflow](workflow.md#4-ontology-reduce-document-level).

## Structured documents (optional)

For papers and other heading-structured Markdown text, `/process` and `ontocast --input-path` accept optional parameters. When both `target_sections` and `summarize_sections` are omitted, the pipeline stays `convert → chunk → extract` with no extra graph nodes.

### Section tagging and section-aligned chunks

When `target_sections` or `summarize_sections` is set, the **Chunk** node runs a single prepare pipeline:

1. **Segment** — Docling `HybridChunker` segments for layout-aware PDFs/DOCX; if none, semantic chunking on exported markdown (plain or weak structure).
2. **Coalesce** — undersized segments merge into the right neighbor (trailing tiny segments merge left); short abstract headings are preserved; section boundaries come from heading lines and Docling breadcrumbs.
3. **Tag** — heading regex on exported markdown (`ontocast.config.section_labels` YAML), optional front-matter abstract span, overlap labeling, then parallel LLM backfill for unlabeled segments at or above `CHUNK_SECTION_TAG_MIN_CHARS` (`PARALLEL_WORKERS`).
4. **Filter** — `target_sections` allowlist, or `summarize_sections` allowlist when `target_sections` is omitted (not `*`).
5. **Size** — split oversized segments (semantic when available), merge undersized consecutive same-label chunks to `min_size` / `max_size`.

**Schema selection:** `section_schema_id` (e.g. `academic`, `financial`, `legal`, `clinical`, `manual`, `fiction`, `general`) or `document_type_hint` (substring match in `manifest.yaml`, e.g. `10-Q` → financial). Default is `academic`.

Recognized labels are canonical ids from the active schema (underscore form), e.g. `results`, `md_and_a`, `risk_factors`.

### Optional summarization

When `summarize_sections` is present (including empty or `*` for all units), the **Summarize Chunks** node runs an LLM pass per selected unit (bounded by `PARALLEL_WORKERS`). Summaries are stored on `ContentUnit.summary`; render and critic agents read `extraction_text`, which prefers the summary over the raw chunk.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `target_sections` | omitted | Section prepare + keep only listed sections (e.g. `results,methods`) |
| `summarize_sections` | omitted | Section prepare + summarization node; omit to skip summaries. `*` or empty = all chunks after prepare |
| `summary_max_sentences` | `5` | Max sentences per summary when summarization runs |
| `section_schema_id` | omitted (`academic`) | Section label YAML schema (`financial`, `legal`, `clinical`, `manual`, `fiction`, `general`) |
| `document_type_hint` | omitted | Free-text hint to resolve schema when `section_schema_id` is not set |

Section lists accept comma-separated values or a JSON array in query, form, or JSON body fields.

## Parallel Map/Reduce

Document processing uses a **parallel map/reduce** architecture:

- **Map**: each content unit runs an independent ontology or facts loop (bounded by `PARALLEL_WORKERS`)
- **Reduce**: normalize merged ontology updates; merge and disambiguate facts across units
- Per-request `max_visits` overrides the server default for render/critic retry budgets

## Facts Extraction Model

Facts rendering follows a **two-namespace contract** baked into the operational guidelines (supplement any `facts_user_instruction` you pass on `/process`):

| Namespace | Role |
|-----------|------|
| Domain ontology prefix | Schema only: classes (`rdf:type` targets), properties, and **reference individuals** that already exist verbatim in the catalog (e.g. controlled vocabulary entries) |
| `cd:` (`FACTS_NAMESPACE`) | Every **new** instance extracted from the source text, even when typed with an ontology class |

Rules the model is steered to follow:

- Mint `cd:` instances with `lowercase_snake_case` local names and an `rdfs:label` from the source text.
- Never invent IRIs under the domain ontology namespace; reuse a reference individual’s canonical IRI only when it is explicitly declared in the provided ontology.
- A matching **class** does not mean a matching **individual** — text occurrences become new `cd:` nodes typed with that class.
- Do not place ontology class IRIs in subject/object slots; do not type `cd:` entities as `rdfs:Class` or `rdf:Property`.

Details and examples: [User Instructions](user_instructions.md#facts-extraction-guidelines).

## Entity Disambiguation

Cross-chunk identity resolution during facts aggregation:

- Embedding similarity + symbolic compatibility (`EntityAligner`)
- Identical `URIRef` across unit graphs always merge (independent of embedding score)
- Connected-component clustering with configurable `AGG_SIMILARITY_THRESHOLD`
- `skos:altName` and label-aware matching
- Provenance annotations on merged triples

The same aligner backs benchmark **graph matching** (`/match/entities`, `/match/evaluate`). See [Aggregation](aggregation.md) for configuration and evaluation notes.

## Ontology Context

Before rendering, each unit receives ontology context from one of three modes:

| Mode | Source |
|------|--------|
| `selected_single_ontology` | LLM picks a catalog TTL per unit |
| `selected_vector_search_ontology` | Qdrant hybrid retrieval + induced subgraph |
| `fixed_single_ontology` | Pinned catalog `ontology_id` |

Details: [Ontology Context](ontology_context.md).

## Tenancy

Runtime **tenant** and **project** parameters (HTTP query/form/JSON) partition triple-store datasets and Qdrant collections:

```
{tenant}--{project}--facts
{tenant}--{project}--ontologies
```

Defaults: `ontocast` / `test`. Not read from environment variables.

Details: [Tenancy](tenancy.md).

## Budget Tracking

- **LLM Statistics**: API calls, characters sent/received; optional token counts when the provider reports usage metadata
- **Cache hits**: Disk-cache hits increment `cache_hits` and character totals but **not** `calls_count` (no provider tokens)
- **Triple Metrics**: Ontology and facts triples per operation
- **Summary Reports**: Logged at end of processing:
  ```
  LLM: X calls, Y sent, Z received, N cache hits | Triples: A ontology, B facts
  ```
- **BudgetTracker** lives on `AgentState` and per-unit states; merged at reduce stages

## Key Components

| Component | Role |
|-----------|------|
| `Ontology` | Versioned RDF graph with metadata (id, hash, lineage) |
| `RDFGraph` | RDF 1.2-aware graph wrapper (Turtle + JSON-LD) |
| `AgentState` | Document-level workflow state |
| `UnitOntologyState` / `UnitFactsState` | Per-unit loop state |
| `ToolBox` | LLM, triple store, chunking, vector store, cache |
| `GraphUpdate` | Structured insert/delete triple patches from the LLM |
| `ContentUnit` | One chunk's text, optional `section_label` / `summary`, and ontology/facts outputs (`extraction_text` for LLM prompts) |

## Next Steps

- [Workflow](workflow.md) — full pipeline stages
- [Configuration](configuration.md) — environment variables
- [API Endpoints](api.md) — REST interface
