"""Prompt for coreference resolution across the units of one document."""

preamble = """\
You resolve coreference in a knowledge graph extracted from one document. The \
document was read in parts; each part named the things it mentions on its own, \
so one thing may appear under several names, and two different things may have \
been given similar names."""

instruction = """\
# TASK

For each numbered pair below, decide whether entity A and entity B denote the \
same real-world thing. Each entity is shown as the part of the document that \
named it stated it: its id, labels, types, values, relations, and a sentence \
from that part when one names it.

- Answer "same" when both refer to one thing, even under different names: a \
role ("the first applicant") and the name of whoever holds it, an abbreviation \
and its expansion, a short and a full name.
- Answer "different" when they refer to distinct things: a different place, \
time, period, quantity, value, member of a series, or holder of a role. Shared \
words do not make two things the same.
- Answer "unsure" when the evidence shown does not decide. A wrong "same" \
corrupts the graph; a wrong "different" only leaves a duplicate, so prefer \
"unsure" or "different" when in doubt.

Give one verdict per pair, with its number and one short reason."""

pairs_template = """\
# PAIRS

{pairs}"""

template_prompt = """{preamble}

{instruction}
{user_instruction}

{pairs}

{format_instructions}
"""
