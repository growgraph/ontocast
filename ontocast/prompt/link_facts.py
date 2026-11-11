"""Prompt for the link pass: links the text states for isolated entities."""

preamble = """\
You complete a knowledge graph extracted from one document. The document was \
read in parts, so some entities in the graph are connected to no other entity, \
though the document may state how they relate to one."""

instruction = """\
# TASK

Each ISOLATED entity below is shown with the passage of the document that \
mentions it. Propose a link from or to it only where its passage states the \
relation.

- Link an isolated entity to one of the listed entities, by number.
- Use only a property from PROPERTIES, written exactly as listed, in the \
direction its `domain -> range` gives: the subject is the domain side.
- Quote the words of the passage that state the link, copied exactly.
- Never refer to an entity that is not listed, never invent a property, and \
never add a link the passage only suggests or that general knowledge supplies.
- An empty list is a valid answer: a missing link costs less than a wrong one."""

entities_template = """\
# ENTITIES

{entities}"""

properties_template = """\
# PROPERTIES

{properties}"""

template_prompt = """{preamble}

{instruction}
{user_instruction}

{entities}

{properties}

{format_instructions}
"""
