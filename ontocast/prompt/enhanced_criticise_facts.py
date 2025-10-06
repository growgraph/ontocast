"""Enhanced facts criticism prompts with context passing and memory support.

This module provides enhanced prompt templates for facts criticism that support
context passing, memory, and improved critique quality.
"""

# Enhanced prompt for facts criticism with context
prompt_enhanced = """
You are a helpful assistant that criticises the knowledge graph of facts derived from a document using a supporting ontology.
You need to decide whether the derived knowledge graph of facts is a faithful representation of the document.
It is considered satisfactory if the knowledge graph captures all facts (dates, numeric values, etc) that are present in the document.
Provide an itemized list improvements in case the graph is missing some facts.

PREVIOUS CONTEXT:
{previous_context}

CONTEXT-AWARE EVALUATION:
- Consider previous critiques when evaluating the current facts
- Build upon previous feedback rather than starting fresh
- Maintain consistency with previous evaluation criteria
- Focus on incremental improvements based on past feedback

Here is the supporting ontology:
```ttl
{ontology}
```

Here is the document from which the facts were derived:
{document}

Here's the knowledge graph of facts derived from the document:
```ttl
{knowledge_graph}
```

{format_instructions}
"""

# SPARQL-focused critique prompts for facts
sparql_facts_critique_instruction = """
You are a helpful assistant that criticises SPARQL operations for facts updates.

You need to decide whether the proposed SPARQL operations are appropriate and complete, also providing a score between 0 and 100.
The operations are considered satisfactory if they capture all necessary changes to the facts based on the document.

PREVIOUS CONTEXT:
{previous_context}

ONTOLOGY CONTEXT:
{ontology_context}

CURRENT FACTS:
{current_facts}

PROPOSED SPARQL OPERATIONS:
{sparql_operations}

EVALUATION CRITERIA:
1. Appropriateness: Are the proposed changes appropriate given the document and current facts?
2. Completeness: Do the changes capture all necessary updates from the document?
3. Correctness: Are the SPARQL operations syntactically and semantically correct?
4. Consistency: Are the changes consistent with the existing facts and ontology context?
5. Minimality: Are the changes minimal and focused on what actually changed?

CONTEXT-AWARE EVALUATION:
- Consider previous critiques when evaluating the current operations
- Build upon previous feedback rather than starting fresh
- Maintain consistency with previous evaluation criteria
- Focus on incremental improvements based on past feedback

{format_instructions}
"""
