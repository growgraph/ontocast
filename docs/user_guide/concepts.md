# Concepts

Here we introduce the main concepts of OntoCast, a framework for transforming data into semantic triples.

## Ontology Management

OntoCast manages ontologies with automatic versioning and timestamp tracking:

- **Semantic Versioning**: Automatic version increments (MAJOR/MINOR/PATCH) based on change analysis
- **Timestamp Tracking**: `updated_at` field tracks when ontology was last modified
- **Smart Analysis**: Analyzes ontology changes (classes, properties, instances) to determine appropriate version bump
- **Property Syncing**: Version and timestamp are synced to the RDF graph as `owl:versionInfo` and `dcterms:modified`

## Budget Tracking

OntoCast provides comprehensive budget tracking for LLM usage and triple generation:

- **LLM Statistics**: Tracks API calls, characters sent/received
- **Triple Metrics**: Tracks ontology and facts triples generated
- **Summary Reports**: Budget summaries logged at end of processing
- **Integrated Tracking**: Budget tracker integrated into AgentState for clean dependency injection

## Key Components

- **Ontology**: RDF graph with properties (id, title, description, version, timestamp)
- **AgentState**: Central state management with budget tracking
- **ToolBox**: Collection of tools for processing and caching
- **Triple Stores**: Support for filesystem, Fuseki, and Neo4j storage

