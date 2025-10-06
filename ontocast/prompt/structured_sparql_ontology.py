"""Structured SPARQL prompts for ontology updates.

This module provides prompt templates for generating structured SPARQL queries
with explicit ADD, UPDATE, and REMOVE sections for ontology modifications.
"""

# Structured SPARQL instruction for ontology updates
structured_sparql_instruction = """
You are an expert in SPARQL and ontology engineering. Your task is to generate structured SPARQL operations to update an existing ontology based on new document content.

IMPORTANT: You must generate SPARQL operations that modify the existing ontology, not replace it entirely.

Generate operations in the following structured format:

## ADD Section
- INSERT new ontology classes, properties, and relationships
- Add new domain-specific concepts
- Extend existing hierarchies

## UPDATE Section
- MODIFY existing ontology elements
- Update descriptions, labels, or relationships
- Refine existing concepts

## REMOVE Section
- DELETE obsolete ontology elements
- Remove outdated concepts or relationships
- Clean up redundant elements

Focus on incremental improvements rather than complete rewrites.
"""

# Structured SPARQL template for ontology updates
structured_sparql_template = """
{structured_sparql_instruction}

Document to process:
{document}

Current Ontology:
{ontology_desc}

Generate structured SPARQL operations to update the ontology based on the document content.

{pydantic_format_instructions}
"""

# Pydantic output format instructions
pydantic_format_instructions = """
IMPORTANT: You must respond with a valid JSON object that matches the StructuredSPARQLQueryModel schema.

The JSON should have the following structure:
{
    "operations": [
        {
            "operation_type": "INSERT",
            "query": "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\nINSERT DATA { ... }",
            "description": "Add new ontology classes",
            "metadata": {}
        },
        {
            "operation_type": "UPDATE",
            "query": "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\nUPDATE DATA { ... }",
            "description": "Update existing ontology properties",
            "metadata": {}
        },
        {
            "operation_type": "DELETE",
            "query": "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\nDELETE DATA { ... }",
            "description": "Remove obsolete ontology elements",
            "metadata": {}
        }
    ],
    "namespaces": {
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "co": "http://example.org/ns#"
    }
}

Each query should be a complete, valid SPARQL operation with proper syntax.
"""
