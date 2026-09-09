from .common import system_preamble_semantic

# Chapter order is a cost lever, not a stylistic choice. The ontology chapter
# is most of a facts prompt and identical between the render and the critic
# call on a unit, so everything up to its end -- preamble, conformance
# contract, ontology -- is kept byte-identical across the two templates and
# everything phase-specific (task, guidelines, user instruction, text) follows
# it. A provider's prefix cache can then serve the critic the chapter the
# render already paid for. The critic template opens the same way; a test
# pins the two heads to each other.
template_prompt = """
{preamble}

{conformance_chapter}

{ontology_chapter}

# TASK

Generate semantic triples representing facts (not abstract entities) based on provided domain ontology.

{facts_instruction}

{user_instruction}

{text_chapter}

{output_instruction}

{format_instructions}
"""

# Shared verbatim with the critic: the cacheable prefix starts at byte zero,
# so the task statement lives in the template after the ontology chapter
# rather than here.
preamble = system_preamble_semantic

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
