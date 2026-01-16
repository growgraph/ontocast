template_prompt = """
You are a helpful assistant that decides which ontology to use for a given document.
You are given a numbered list of ontologies and a document excerpt.
You need to select which ontology can be used for the document to create a semantic graph.

Select from the following options:

0. No suitable ontology available

{ontologies_list}


Here is an excerpt from the document:
{excerpt}

{format_instructions}
"""
