from .common import system_preamble_semantic

template_prompt = """
{preamble}

{facts_instruction}

{user_instruction}

{ontology_chapter}

{conformance_chapter}

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

The current graph of factual triples has been reviewed. The items below are the corrections to apply.

This is a CORRECTION PASS, not a re-extraction. Apply the items and nothing else.

1. Fix each item by rewriting the offending term or value IN PLACE.
   - Do not delete the statement and do not drop extracted data. A response
     that only removes triples has resolved nothing: the item is gone because
     the data is gone, which is the failure this pass exists to avoid.
   - Every corrected statement must survive with its subject and its value intact.

2. Do NOT add statements that no item asks for. Leaving correct triples exactly
   as they are is the expected outcome for every part of the graph no item
   mentions.

3. If an item is contradicted by the source text, do not apply it, and say why
   in `explanation`. Never delete or alter other statements as a consequence -
   a wrong item licenses skipping that item, nothing more.

4. Before finalizing, check that:
   - Each triple still accurately represents information from the source text
   - Existing ontology entities are used instead of new cd: entities
   - No ontology-prefixed entity was invented or renamed
   - The graph holds at least as much correct data as it did before
{suggestions_instruction}
"""
