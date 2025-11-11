# Concepts

Here we introduce the main concepts of OntoCast, a framework for transforming data into semantic triples.

## Ontology Management

OntoCast manages ontologies with automatic versioning and timestamp tracking:

- **Semantic Versioning**: Automatic version increments (MAJOR/MINOR/PATCH) based on change analysis
- **Hash-Based Lineage**: Git-style versioning with parent hashes for tracking ontology evolution
- **Multiple Versions**: Versions stored as separate named graphs in Fuseki triple stores
- **Timestamp Tracking**: `updated_at` field tracks when ontology was last modified
- **Smart Analysis**: Analyzes ontology changes (classes, properties, instances) to determine appropriate version bump:
  - **MAJOR**: Substantial breaking changes (deletions of classes/properties)
  - **MINOR**: New features (new classes/properties) or any deletions
  - **PATCH**: Updates to existing structures (instances, descriptions, small changes)
- **Property Syncing**: Version and timestamp are synced to the RDF graph as `owl:versionInfo` and `dcterms:modified`
- **Versioned IRIs**: Each version gets a unique IRI with hash fragment for storage organization

## GraphUpdate System

OntoCast uses a token-efficient GraphUpdate system for incremental graph modifications:

- **Structured Operations**: LLM outputs `GraphUpdate` objects containing `TripleOp` operations (insert/delete) instead of full TTL graphs
- **Token Efficiency**: Only changes are generated, dramatically reducing LLM token usage compared to full graph regeneration
- **SPARQL Generation**: Operations are automatically converted to executable SPARQL queries
- **Incremental Updates**: Graph updates are applied incrementally, allowing for precise modifications
- **Operation Types**: Supports both `insert` and `delete` operations with explicit prefix declarations
- **Custom Queries**: Also supports `GenericSparqlQuery` for complex custom SPARQL operations

### How GraphUpdate Saves Tokens

Instead of generating the entire graph in Turtle format (which can be thousands of tokens), the LLM now outputs only the changes:

- **Before**: Full TTL graph with all triples (e.g., 5000 tokens)
- **After**: Structured operations with only changes (e.g., 200 tokens)
- **Savings**: Typically 80-95% reduction in output tokens

## Budget Tracking

OntoCast provides comprehensive budget tracking for LLM usage and triple generation:

- **LLM Statistics**: Tracks API calls, characters sent/received for cost monitoring
- **Triple Metrics**: Tracks ontology and facts triples generated per operation
- **Operation Counts**: Tracks number of update operations for both ontology and facts
- **Summary Reports**: Budget summaries logged at end of processing with format:
  ```
  LLM: X calls, Y sent, Z received | Triples: A ontology, B facts
  ```
- **Integrated Tracking**: Budget tracker integrated into AgentState for clean dependency injection
- **Automatic Updates**: Budget tracker automatically updated when LLM calls are made or triples are generated

## Key Components

- **Ontology**: RDF graph with properties (id, title, description, version, timestamp, hash, parent_hashes)
- **AgentState**: Central state management with budget tracking and GraphUpdate operations
- **ToolBox**: Collection of tools for processing and caching
- **Triple Stores**: Support for filesystem, Fuseki, and Neo4j storage
- **GraphUpdate**: Structured representation of graph modifications as SPARQL operations
- **BudgetTracker**: Lightweight tracker for LLM usage and triple generation statistics

