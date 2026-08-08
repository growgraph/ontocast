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

Keys are matched **case-insensitively** with camelCase / snake_case / kebab-case tolerance. Bibliographic identifier and source keys also accept an optional leading or trailing `id` affix (so `DOI`, `doi_id`, `arxivId`, and bare `arxiv` all resolve to the bibliographic identifiers above).

Unregistered keys whose last or first token is an **identifier-shaped affix** — `id`, `uid`, `uuid`, `guid`, `ref`, `reference`, `no`, `num`, `number`, `code`, `slug`, `handle`, `accession`, or `key` — become structured `dcterms:identifier` blank nodes with `dcterms:type` set to the stem (e.g. `department_id: "D-42"` → scheme `department`). When the stem matches a typed entity-link key present in the same payload (e.g. `project` + `project_id`), the value attaches as `dcterms:identifier` on that minted entity instead. Bare unknown keys without an affix still mint typed entities. Including `key` in the affix set is an accepted trade-off (`key_finding` is treated as a structured identifier with scheme `finding`).

Canonical names in the table remain the documented contract.

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

## Structured documents

Section tagging is **on by default** for every document and, since 0.6, costs
nothing: each chunk carries a `section_label` from the active schema, decided by
a cascade of deterministic tiers (`CHUNK_SECTION_CLASSIFIER=heuristic`). Set
`CHUNK_SECTION_CLASSIFIER=off` to restore untagged `convert → chunk → extract`
(this also disables section filters and schema default exclusions), or `llm` to
add a batched model pass over whatever the free tiers could not name.

### The classification cascade

Classification runs over the document's **outline**, not over chunks. Every
detected heading closes the preceding section, whether or not the heading maps
to a known label — an unrecognised heading opens an explicitly *unresolved*
section rather than letting the previous label run on. Each tier only sees what
the previous one left unlabeled:

| Tier | Cost | What it decides |
|---|---|---|
| **Outline** | free | Where sections begin and end; which headings are generic section names versus descriptive subsection titles (the latter inherit their parent's label) |
| **Heading** | free | Label from anchored patterns, then from keywords — so `Results and Discussion`, `Experimental Section` and `■ REFERENCES` resolve, not just the canonical spellings |
| **Order** | free | Fills a gap only when the schema's canonical section order leaves one candidate, and never backwards |
| **Density** | free | Labels heading-free regions from content features (`CHUNK_SECTION_DENSITY`) |
| **LLM** | ~1 call | One batched call over the remainder (`CHUNK_SECTION_CLASSIFIER=llm`) |

Ambiguity resolves to *no label*, deliberately: an unlabeled chunk is merely
unselectable, whereas a wrongly labeled one is silently dropped by a filter or
extracted as the wrong section. Each chunk records which tier decided its label
(`section_label_source`) and how confident that tier was
(`section_label_confidence`).

### Section tagging and section-aligned chunks

The **Chunk** node runs a single prepare pipeline:

1. **Segment** — sections-first by default (`CHUNK_SEGMENTER=semantic`): the outline partitions the exported markdown into section spans, the text is split at section boundaries, and the semantic chunker splits *within* each oversized section block — so no chunk straddles a section boundary and chunks from detected sections inherit their label deterministically. `CHUNK_SEGMENTER=docling` selects Docling `HybridChunker` structural segments instead (its tokenizer is budgeted from `CHUNK_MAX_SIZE`).
2. **Coalesce** — undersized segments merge into the right neighbor (trailing tiny segments merge left); short abstract headings are preserved; merges never cross section structure.
3. **Tag** — the cascade above, plus an optional front-matter abstract span. Segments at or above `CHUNK_SECTION_TAG_MIN_CHARS` reach the LLM tier when it is enabled; segments in an explicitly unresolved section are never filled from a neighbour.
4. **Filter** — `target_sections` allowlist (or `summarize_sections` allowlist when `target_sections` is omitted and not `*`), then the `exclude_sections` denylist. When `exclude_sections` is unset, the active schema's `default_exclude` applies — the academic schema drops `acknowledgements` and `appendix`; pass an explicit empty `exclude_sections` to keep everything.
5. **Size** — split oversized segments (semantic when available), merge undersized consecutive same-label chunks to `min_size` / `max_size`. Unlabeled chunks are never merged together, since they are not known to share a section.

### Inspecting what the classifier decided

Section labels drive which text is extracted, so a wrong label changes the
output without appearing anywhere in it. `ontocast sections` shows the
decisions before any extraction cost:

```bash
ontocast sections --input-path ./paper.pdf
ontocast sections --input-path ./paper.pdf --target-sections results --as-json
```

It prints the resolved schema (with the tier that detected it and its margin
over the runner-up), the detected outline, and every chunk's label, deciding
tier and confidence. Unless `--section-classifier llm` is passed it makes no LLM
calls and needs no provider credentials.

Reference lists are handled separately by `CHUNK_BIBLIOGRAPHY_MODE` (default `skip`: dropped before extraction; `citations_only` extracts bibliographic metadata instead).

PDF extraction behavior before chunking is configurable through `CONVERTER_*` settings. For born-digital publisher PDFs, prefer `CONVERTER_PROFILE=born_digital` to favor embedded text and enable OntoCast's temporary ligature-gap workaround.

Recognized labels are canonical ids from the active schema (underscore form), e.g. `results`, `md_and_a`, `risk_factors`.

### Which schema a document is scored against

Every label above is meaningful only relative to a **schema**. A 10-Q scored
against the academic schema recognises almost nothing and comes back entirely
unlabeled, so choosing the schema matters as much as the cascade that uses it.

The catalog is a **partition**: every document belongs to exactly one cell.
Each cell is defined by a one-sentence `document_profile` stating what makes it
exclusive — if two profiles could describe the same document, the partition is
broken and the vocabulary is wrong.

| Cell | What makes it exclusive |
|---|---|
| `academic` | Scholarly reporting of original experiments, or a review of them |
| `financial` | Corporate disclosure of results to shareholders or regulators |
| `clinical` | A protocol or report governing treatment of human participants |
| `legal` | A binding agreement *between parties* |
| `patent` | An invention disclosure with numbered claims — legal, but not an agreement |
| `standard` | Normative requirements *for implementers* — prescriptive, not instructional |
| `manual` | Instructions for a *user* operating a system — instructional, not normative |
| `fiction` | Narrative prose with characters and an unfolding story |
| `news` | An announcement of a recent event, with dateline and quotations |
| `general` | **Residual cell** — everything else |

`general` deliberately carries no `document_profile`, which is what keeps it out
of detection entirely: most of its labels are subsets of other schemas, so it
can be *asked for* but never *inferred*. There is no `thesis` cell — a thesis
has the same IMRaD body as a paper and differs only in front and back matter, so
it is a subtype of `academic` rather than a sibling, and belongs to the planned
`academic → paper → experimental` funnel rather than to a flat partition.
`thesis` and `dissertation` hints therefore resolve to `academic`.

**Precedence** — caller intent is never overridden; detection only fills a gap:

1. explicit `section_schema_id`
2. a `document_type_hint` matching a manifest needle (`10-Q` → `financial`).
   Needles match on **whole words**, so "quarterly widget report" is not a
   patent because of `epo` inside "report"
3. automatic detection from the document itself
4. the manifest default (`academic`)

Detection runs three tiers, cheapest first, each seeing only what the previous
could not decide (`CHUNK_SECTION_SCHEMA_DETECT`):

| Tier | Cost | Signal |
|---|---|---|
| **Lexical** | free | Headings **only one** candidate schema recognises |
| **Semantic** | reuses the chunker's embedding model | Each heading votes for its nearest schema by label-name similarity |
| **Content** | same model, off by default | Body paragraphs against `document_profile` sentences |

The lexical tier scores on *exclusive* evidence: a heading several schemas
recognise counts **zero**, not a fraction. `References` genuinely says nothing
about which cell a document is in, so weighting it only adds noise — and
dropping it is what produces the margins that make the tier safe to act on.
Measured over `test/data/schema_corpus.json` (one real document per cell) it
classifies all nine correctly on headings alone, with no model loaded.

Every tier **abstains** rather than guessing, falling back to the manifest
default. This is the same trade the label cascade makes and for the same reason:
a wrong schema silently relabels an entire document, while an abstention merely
leaves `document_type_hint` in charge.

### Optional summarization

When `summarize_sections` is present (including empty or `*` for all units), an LLM compression pass runs for each selected unit. Summaries are stored on `ContentUnit.summary`; render and critic agents read `extraction_text`, which prefers the summary over the raw chunk.

The pass runs **inside the extraction fan-out** (bounded by `PARALLEL_WORKERS`), immediately before the unit is rendered, rather than as a separate pipeline stage. A summary depends only on its own unit, so a stage would have been a barrier: every unit waiting on the slowest summary before any extraction could begin. It is computed once per unit even when both ontology and facts extraction run.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `target_sections` | omitted | Keep only listed sections (e.g. `results,methods`) |
| `exclude_sections` | omitted (schema `default_exclude`) | Drop listed sections; explicit empty value disables all exclusion |
| `summarize_sections` | omitted | Per-unit summarization; omit to skip summaries. `*` or empty = all chunks after prepare |
| `summary_max_sentences` | `5` | Max sentences per summary when summarization runs |
| `section_schema_id` | omitted (detected, else `academic`) | Section label YAML schema (`academic`, `financial`, `legal`, `clinical`, `manual`, `fiction`, `patent`, `standard`, `news`, `general`) |
| `document_type_hint` | omitted | Free-text hint to resolve schema when `section_schema_id` is not set; overrides detection |

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
