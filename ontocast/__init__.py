"""OntoCast: Agentic ontology-assisted framework for semantic triple extraction.

OntoCast is a comprehensive framework for extracting semantic triples from
documents using ontology assistance. It provides a complete pipeline for
document processing, ontology management, and knowledge graph construction.

The framework includes:
- Document conversion and chunking
- Ontology selection and management
- Fact extraction and validation
- Triple store integration (Fuseki, In-Memory)
- LLM-powered semantic analysis
- REST API server for document processing

For more information, see the documentation at https://growgraph.github.io/ontocast/
"""

from ontocast.stategraph.atomic import facts_loop, ontology_loop

__all__ = ["facts_loop", "ontology_loop"]
