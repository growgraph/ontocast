"""Structured SPARQL prompts for facts updates.

This module provides prompt templates for generating structured SPARQL queries
with explicit ADD, UPDATE, and REMOVE sections for facts modifications.
"""

# Structured SPARQL instruction for facts updates
structured_sparql_instruction = """
You are an expert in SPARQL and knowledge graph engineering. Your task is to generate structured SPARQL operations to update existing facts based on new document content.

IMPORTANT: You must generate SPARQL operations that modify the existing facts graph, not replace it entirely.

Generate operations in the following structured format:

## ADD Section
- INSERT new facts about entities and relationships
- Add new instances and their properties
- Extend existing knowledge with new information

## UPDATE Section
- MODIFY existing facts
- Update entity properties or relationships
- Refine existing knowledge

## REMOVE Section
- DELETE outdated facts
- Remove incorrect or obsolete information
- Clean up redundant data

Focus on incremental knowledge updates rather than complete rewrites.
"""

# Structured SPARQL template for facts updates
structured_sparql_template = """
{structured_sparql_instruction}

Document to process:
{document}

Current Facts:
{facts_desc}

Generate structured SPARQL operations to update the facts based on the document content.

{pydantic_facts_format_instructions}
"""

# Pydantic output format instructions for facts
pydantic_facts_format_instructions = """
IMPORTANT: You must respond with a valid JSON object that matches the StructuredSPARQLQueryModel schema.

The JSON should have the following structure:
{
    "operations": [
        {
            "operation_type": "INSERT",
            "query": "PREFIX cd: <http://example.org/doc/>\nPREFIX schema: <https://schema.org/>\nINSERT DATA { ... }",
            "description": "Add new facts about entities",
            "metadata": {}
        },
        {
            "operation_type": "UPDATE",
            "query": "PREFIX cd: <http://example.org/doc/>\nUPDATE DATA { ... }",
            "description": "Update existing fact relationships",
            "metadata": {}
        },
        {
            "operation_type": "DELETE",
            "query": "PREFIX cd: <http://example.org/doc/>\nDELETE DATA { ... }",
            "description": "Remove outdated facts",
            "metadata": {}
        }
    ],
    "namespaces": {
        "cd": "http://example.org/doc/",
        "schema": "https://schema.org/",
        "co": "http://example.org/ns#"
    }
}

Each query should be a complete, valid SPARQL operation with proper syntax.
"""
