from .common import system_preamble_semantic

template_prompt = """
{preamble}

{facts_instruction}

{user_instruction}

{ontology_chapter}

{text_chapter}

{fact_chapter}

{improvement_instruction}

{output_instruction}

{format_instructions}
"""

preamble = f"""
{system_preamble_semantic}
Generate semantic triples representing facts (not abstract entities) based on provided domain ontology.
"""

_CITATION_METADATA_HEADER = """
# CITATION-METADATA UNIT
This unit is a bibliography/reference list, not document content. Extract ONLY
bibliographic citation metadata.
Do NOT mint domain facts of any kind from citation titles, and do NOT use the
domain ontologies for these entries.
"""

_CITATION_VOCABULARY_TEMPLATE = """Use these terms:
- one individual per cited work, typed {work_class}{fallback_clause};
- attach title ({title}), authors ({author} with {author_name}),
  publication year ({date_published}), venue ({venue}),
  DOI/identifier ({identifier}) when present;
- link each cited work from the document node via {cites}.
"""


def build_citation_metadata_instruction(vocabulary: dict[str, str]) -> str:
    """Render the citation-metadata prompt block for a configured vocabulary.

    The bibliographic terms are configuration, not retrieval: a reference list
    is not domain content, so its vocabulary never reaches the catalog. Keeping
    them out of the prompt literal is what lets a non-schema.org catalog (bibo,
    FaBiO, DCMI) describe citations in its own terms.

    Args:
        vocabulary: Role -> term mapping (``CHUNK_CITATION_VOCABULARY``). An
            empty mapping emits the routing guidance with no term list.

    Returns:
        str: The prompt block, or the header alone when no terms are configured.
    """
    if not vocabulary:
        return _CITATION_METADATA_HEADER
    fallback = vocabulary.get("fallback_class", "")
    filled = {
        "work_class": vocabulary.get("work_class", "the cited-work class"),
        "fallback_clause": (
            f" (or {fallback} when clearly not an article)" if fallback else ""
        ),
        "title": vocabulary.get("title", "the title property"),
        "author": vocabulary.get("author", "the author property"),
        "author_name": vocabulary.get("author_name", "the name property"),
        "date_published": vocabulary.get(
            "date_published", "the publication-date property"
        ),
        "venue": vocabulary.get("venue", "the venue property"),
        "identifier": vocabulary.get("identifier", "the identifier property"),
        "cites": vocabulary.get("cites", "the citation property"),
    }
    return _CITATION_METADATA_HEADER + _CITATION_VOCABULARY_TEMPLATE.format(**filled)


improvement_instruction_template = """\n\n
# IMPROVEMENT INSTRUCTION

The current iteration of the graph of factual triples has been reviewed by Critic, who provided suggestions for improvement.

CRITICAL: You are the final decision-maker. Critic's suggestions are advisory, not mandatory. Think independently.

Your task is to critically evaluate and improve the triples:

1. Independently verify each suggestion - Before implementing ANY suggestion, verify it against:
   - The original source text (does it accurately reflect what's written?)
   - The OPERATIONAL GUIDELINES (does it follow the rules?)
   - The domain ontology (does it use entities correctly?)
   - Logical consistency (does it make semantic sense?)

2. Implement only valid improvements - Apply suggestions that are demonstrably correct and enhance accuracy or completeness. If uncertain, prioritize faithfulness to the source text.

3. Actively reject flawed suggestions - If a suggestion is:
   - Factually incorrect (contradicts the source text)
   - Violates OPERATIONAL GUIDELINES
   - Would introduce errors or degrade quality
   - Based on misunderstanding of the ontology
   
   Then REJECT it and briefly explain why in your response.

4. Think beyond the critique - Critic may have:
   - Missed issues entirely
   - Identified patterns but not all instances
   - Focused on some aspects while overlooking others
   
   Proactively identify and fix additional problems not mentioned in the critique.

5. Verify every change - Before finalizing, double-check that:
   - Each triple accurately represents information from the source text
   - Existing ontology entities are used instead of creating new cd: entities
   - No ontology-prefixed entity was invented or renamed
   - All OPERATIONAL GUIDELINES are satisfied
   - The overall graph is more complete and accurate than before

Your goal: Produce the most accurate representation of the source text, not to satisfy Critic.
{suggestions_instruction}
"""
