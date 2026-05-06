template_prompt = """
You are a helpful assistant that decides which catalog ontology to use for a text segment.
You are given a numbered list of ontologies and an excerpt. Choose one ontology, or
indicate that none apply (use answer index {none_index} for no suitable ontology;
use indices 1..{num_ontologies} for a listed ontology).

{ontologies_list}

Here is the text excerpt:
{excerpt}

Additional user instruction for ontology selection:
{ontology_selection_user_instruction}

{format_instructions}
"""
