# Concepts

Main concepts in OntoCast, a framework for transforming documents into semantic triples.

## Ontology Management

OntoCast manages ontologies with automatic versioning and timestamp tracking:

- **Canonical identity**: catalog entries are keyed by ontology **IRI**; short `ontology_id` and author Turtle **prefix** are aliases (they may differ, e.g. `observation` / `obs`)
- **Semantic Versioning**: MAJOR/MINOR/PATCH increments from change analysis
- **Hash-Based Lineage**: Parent hashes track ontology evolution
- **Multiple Versions**: Stored as separate named graphs (Fuseki or in-memory pyoxigraph)
- **Timestamp Tracking**: `updated_at` synced as `dcterms:modified`
- **Versioned IRIs**: Unique IRIs with hash fragments for storage
- **Working context**: multi-ontology prompt snapshots use `OntologySnapshot` (graph + `source_iris`, no catalog id) instead of inventing a single catalog identity from a union graph
- **Writeback**: ontology complements are applied to catalog terminals by namespace ownership after subtracting triples already present in the prompt snapshot
- **Prefixes**: graph merges rename conflicting prefixes rather than silently overriding bindings; serialization aliases remain sugar over absolute IRIs
- **Catalog reads**: `OntologyManager` owns identity/aliases, terminal selection, and the content-addressed graph cache — see [Ontology Catalog](../architecture/ontology_catalog.md)

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

## Document-level identity metadata

Chunk provenance links facts → chunk → parent `doc_iri` (content-hash IRI). Callers can also assert **document identity** that is independent of body text via optional `document_metadata`:

| Kind | Examples |
|------|----------|
| Bibliographic | `doi`, `isbn`, `pmid`, `arxiv_id`, `handle` (literal `dcterms:identifier`) |
| Structured ids | `identifiers: [{ "scheme": "erp:doc", "value": "INV-…" }]` (blank-node `dcterms:Identifier`) |
| Descriptive scalars | `title`, `published` / `issued`, `source_system` |
| Typed entities | `author` / `creator` / `authors` → `schema:Person`; `project` and any other key → `prov:Entity` (SPARQL-discoverable) |
| Stable URI | `stable_source_iri` (`owl:sameAs`), `source_uri` / `source_url` (`dcterms:source`) |

Scalar / identifier example:

```turtle
<doc_iri> a prov:Entity, foaf:Document ;
    dcterms:title "Annual Report" ;
    dcterms:identifier "10.1234/example" ;
    dcterms:identifier [
        a dcterms:Identifier ;
        dcterms:type "erp:doc" ;
        rdf:value "INV-2024-001"
    ] .
```

Business-oriented keys mint **typed RDF entities** under the document facts namespace (not literals), so they are discoverable via SPARQL (`?x a schema:Person`):

```json
{
  "author": ["Jane Doe"],
  "project": {"name": "Perovskite Survey", "identifier": "PRJ-2024-07"}
}
```

```turtle
<doc_iri> dcterms:creator <doc_iri/janeDoe> ;
    dcterms:relation <doc_iri/perovskiteSurvey> .

<doc_iri/janeDoe> a schema:Person ;
    rdfs:label "Jane Doe" .

<doc_iri/perovskiteSurvey> a prov:Entity ;
    rdfs:label "Perovskite Survey" ;
    dcterms:identifier "PRJ-2024-07" .
```

A bare string is enough (`{"project": "Perovskite Survey"}`). Override the class with a dict: `{"type": "schema:Project"}` (CURIE or absolute IRI). Entities are per-document (no cross-document identity claim).

Rules:

- All fields optional; omit → no document identity triples (except local `ontocast process` filename fallback below).
- Payload values are guaranteed in the graph even when absent from the text.
- Document identity and minted metadata entities stay on the facts graph under chunk-level `strip_provenance`.
- `graph_uri_override` remains storage partitioning, not identity.

**Local batch (`ontocast process --input-path`):**

- `--document-metadata '{"doi":"…"}'` JSON object, or omit to default `dcterms:title` to the filename (`file:line` for JSONL records).
- After each successful run, dumps provenance-stripped Turtle:
  - Facts: `doc.facts.ttl` (or `doc.L3.facts.ttl` for JSONL line 3)
  - Ontology artifacts: `doc.ontology.ttl` (or `doc.<id>.ontology.ttl` when multiple)
- Dump destination defaults to siblings of each input. Override with `--output-dir` (shared), or separately with `--facts-output-dir` / `--ontology-output-dir`.

**HTTP:** pass `document_metadata` as a JSON object field (JSON body) or stringified JSON (multipart / query).

## Structured documents (optional)

For papers and other heading-structured Markdown text, `/process` and `ontocast process --input-path` accept optional parameters. When both `target_sections` and `summarize_sections` are omitted, the pipeline stays `convert → chunk → extract` with no extra graph nodes.

### Section tagging and section-aligned chunks

When `target_sections` or `summarize_sections` is set, the **Chunk** node runs a single prepare pipeline:

1. **Segment** — Docling `HybridChunker` segments for layout-aware PDFs/DOCX; if none, semantic chunking on exported markdown (plain or weak structure).
2. **Coalesce** — undersized segments merge into the right neighbor (trailing tiny segments merge left); short abstract headings are preserved; section boundaries come from heading lines and Docling breadcrumbs.
3. **Tag** — heading regex on exported markdown (`ontocast.config.section_labels` YAML), optional front-matter abstract span, overlap labeling, then parallel LLM backfill for unlabeled segments at or above `CHUNK_SECTION_TAG_MIN_CHARS` (`PARALLEL_WORKERS`).
4. **Filter** — `target_sections` allowlist, or `summarize_sections` allowlist when `target_sections` is omitted (not `*`).
5. **Size** — split oversized segments (semantic when available), merge undersized consecutive same-label chunks to `min_size` / `max_size`.

PDF extraction behavior before chunking is configurable through `CONVERTER_*` settings. For born-digital publisher PDFs, prefer `CONVERTER_PROFILE=born_digital` to favor embedded text and enable OntoCast's temporary ligature-gap workaround.

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
| `cd:` | Every **new** instance extracted from the source text, even when typed with an ontology class |

The `cd:` namespace is the fixed constant `DEFAULT_IRI`
(`ontocast/onto/constants.py`), not a configurable setting — there is no
`FACTS_NAMESPACE` environment variable.

Rules the model is steered to follow:

- Mint `cd:` instances with `lowercase_snake_case` local names and an `rdfs:label` from the source text.
- Never invent IRIs under the domain ontology namespace; reuse a reference individual’s canonical IRI only when it is explicitly declared in the provided ontology.
- A matching **class** does not mean a matching **individual** — text occurrences become new `cd:` nodes typed with that class.
- Do not place ontology class IRIs in subject/object slots; do not type `cd:` entities as `rdfs:Class` or `rdf:Property`.

**Prefix hygiene in facts prompts:** the domain-ontologies clause excludes rdflib’s default bindings (`brick`, `csvw`, `geo`, `xml`, …). Author-declared short prefixes (e.g. `matsci:`) stay canonical; IRI-tail `ontology_id` values do not force a duplicate prefix. Reserved namespaces such as `xml:` are left untouched (no `xml1:` minting).

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
| `selected_vector_search_ontology` | Qdrant or LanceDB hybrid retrieval + induced subgraph |
| `fixed_single_ontology` | Pinned catalog ontology (IRI, `ontology_id`, or author prefix) |

Details: [Ontology Context](ontology_context.md).

## Tenancy

Runtime **tenant** and **project** parameters (HTTP query/form/JSON) partition triple-store datasets and vector-store collections (Qdrant or LanceDB):

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
| `Ontology` | Versioned RDF graph with metadata (IRI, id, hash, lineage) |
| `OntologyHeader` | Graph-less catalog metadata (iri, version, hash, graph_uri) |
| `OntologySnapshot` | Prompt view of one or more catalog graphs (`source_iris`, no catalog id) |
| `OntologyManager` | Catalog identity, aliases, terminal selection, content-addressed graph cache |
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
