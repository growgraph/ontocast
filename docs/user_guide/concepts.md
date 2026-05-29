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

- **Structured Operations**: LLM outputs `GraphUpdate` with `TripleOp` insert/delete ops
- **Wire Formats**: Turtle strings or compact JSON-LD (`LLM_GRAPH_FORMAT`); canonical runtime models are the same
- **SPARQL Generation**: Operations convert to executable SPARQL
- **Token Savings**: Typically 80–95% fewer output tokens vs full graph regeneration

## RDF 1.2 Provenance

OntoCast uses **pyoxigraph** for RDF 1.2 quoted-triple syntax and separates provenance from the working ontology:

- During **normalization**, reification triples, `prov:wasDerivedFrom`, chunk metadata, and alignment artifacts (`owl:sameAs`) move to a **provenance artifact**
- The clean ontology graph feeds consolidation and serialization
- API clients can pass `strip_provenance=true` to omit reification scaffolding from returned Turtle

See [Workflow](workflow.md#4-ontology-reduce-document-level).

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

- **LLM Statistics**: API calls, characters sent/received
- **Triple Metrics**: Ontology and facts triples per operation
- **Summary Reports**: Logged at end of processing:
  ```
  LLM: X calls, Y sent, Z received | Triples: A ontology, B facts
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
| `GraphUpdate` | Structured SPARQL operations from the LLM |
| `ContentUnit` | One chunk's ontology/facts outputs |

## Next Steps

- [Workflow](workflow.md) — full pipeline stages
- [Configuration](configuration.md) — environment variables
- [API Endpoints](api.md) — REST interface
