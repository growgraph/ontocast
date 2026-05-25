"""Prompt templates for OntoCast.

This package contains prompt templates used by the OntoCast framework
for LLM interactions. These prompts are designed to guide language
models in performing specific tasks within the ontology processing
workflow.

Available prompts:
- render_ontology: Generate ontology triples from text
- render_facts: Extract facts from text using ontologies
- criticise_ontology: Evaluate and critique ontology quality
- criticise_facts: Validate and critique extracted facts
- common: Shared prompt templates and components
- graph_format: GraphFormatProfile (prompt + format instructions for llm_graph_format)
- llm_json_schema: Format-bound JSON Schema for canonical report models
- facts_guidelines: Format-specific facts operational guidelines
"""
