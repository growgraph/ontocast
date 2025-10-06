"""Enhanced ontology criticism prompts with context passing and memory support.

This module provides enhanced prompt templates for ontology criticism that support
context passing, memory, and improved critique quality.
"""

# Enhanced prompt for fresh ontology criticism with context
prompt_fresh_enhanced = """
You are a helpful assistant that criticises a newly proposed ontology.

You need to decide whether the updated ontology is sufficiently complete and comprehensive, also providing a score between 0 and 100.
The ontology is considered complete and comprehensive if it captures the most important abstract classes and properties that are present explicitly or implicitly in the document.
If is not not complete and comprehensive, provide a very concrete itemized explanation of why can be improved.
As we are working on an ontology, ONLY abstract classes and properties are considered, concrete entities are not important.

PREVIOUS CONTEXT:
{previous_context}

CONTEXT-AWARE EVALUATION:
- Consider previous critiques when evaluating the current ontology
- Build upon previous feedback rather than starting fresh
- Maintain consistency with previous evaluation criteria
- Focus on incremental improvements based on past feedback

{ontology_original_str}

Here is the document from which the ontology was derived:
{document}

Here is the proposed ontology:
```ttl
{ontology_update}
```

{format_instructions}
"""

# Enhanced prompt for ontology update criticism with context
prompt_update_enhanced = """
You are a helpful assistant that criticises an ontology update.

You need to decide whether the updated ontology is sufficiently complete and comprehensive, also providing a score between 0 and 100.
The ontology is considered complete and comprehensive if it captures the most important abstract classes and properties that are present explicitly or implicitly in the document.
If is not not complete and comprehensive, provide a very concrete itemized explanation of why can be improved.
As we are working on an ontology, ONLY abstract classes and properties are considered, concrete entities are not important.

PREVIOUS CONTEXT:
{previous_context}

CONTEXT-AWARE EVALUATION:
- Consider previous critiques when evaluating the current ontology
- Build upon previous feedback rather than starting fresh
- Maintain consistency with previous evaluation criteria
- Focus on incremental improvements based on past feedback
- Evaluate whether the changes are appropriate and complete

{ontology_original_str}

Here is the document from which the ontology update was derived:
{document}

Here is the ontology update:
```ttl
{ontology_update}
```

{format_instructions}
"""

# SPARQL-focused critique prompts
sparql_critique_instruction = """
You are a helpful assistant that criticises SPARQL operations for ontology updates.

You need to decide whether the proposed SPARQL operations are appropriate and complete, also providing a score between 0 and 100.
The operations are considered satisfactory if they capture all necessary changes to the ontology based on the document.

PREVIOUS CONTEXT:
{previous_context}

CURRENT ONTOLOGY:
{current_ontology}

PROPOSED SPARQL OPERATIONS:
{sparql_operations}

EVALUATION CRITERIA:
1. Appropriateness: Are the proposed changes appropriate given the document and current ontology?
2. Completeness: Do the changes capture all necessary updates from the document?
3. Correctness: Are the SPARQL operations syntactically and semantically correct?
4. Consistency: Are the changes consistent with the existing ontology?
5. Minimality: Are the changes minimal and focused on what actually changed?

CONTEXT-AWARE EVALUATION:
- Consider previous critiques when evaluating the current operations
- Build upon previous feedback rather than starting fresh
- Maintain consistency with previous evaluation criteria
- Focus on incremental improvements based on past feedback

{format_instructions}
"""
