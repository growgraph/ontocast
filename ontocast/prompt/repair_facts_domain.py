"""Prompt for the facts domain-repair call.

Runs after the facts render/critic loop, only on a unit holding a statement
whose property declares an ``rdfs:domain`` the subject's asserted types
contradict (``domain_violation`` findings). The call sees each such statement
under a citable id, what its subject and object are, the classes its property
applies to, and the context properties whose domain fits the subject; it may
only replace the property. Chapter order follows the gap pass: preamble ->
task -> guidelines -> user instruction -> text -> violations.
"""

from __future__ import annotations

from collections.abc import Sequence

from rdflib import RDF, Literal, URIRef
from rdflib.term import Node

from ontocast.onto.enum import LLMGraphFormat
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.triple_index import Triple, TripleIndex
from ontocast.prompt.complete_facts import _first_label, _qname

from .common import system_preamble_semantic

preamble = system_preamble_semantic

template_prompt = """
{preamble}

# TASK

Each statement under DOMAIN VIOLATIONS links its subject through a property
that does not apply to the subject's kind of thing: the property's declared
domain and the subject's type are unrelated. Repair the statements the text
supports.

{repair_instruction}

{user_instruction}

{text_chapter}

{violations_chapter}

{output_instruction}

{format_instructions}
"""

repair_instruction = """\n\n
# DOMAIN REPAIR GUIDELINES

1. For a statement whose relation the text states, write one `REPLACE` fix:
   cite the statement's id in `triple_ids`, and give in `correct_value` the
   same subject and the same object linked by a property that applies to the
   subject's type and says what the text says. Take it from the statement's
   candidates when one fits; otherwise use a term of a well-known vocabulary
   (W3C, Dublin Core, FOAF, schema.org, QUDT) that you are certain exists and
   applies to that type.
2. Change only the property. Never retype the subject or the object, never
   change the object, and never add any other statement.
3. When no property fits, or the text does not state the relation, leave the
   statement out of your answer; it is removed.
4. Declare every namespace prefix your fixes use that is not bound above.
"""

_OUTPUT_INSTRUCTION_TURTLE = """\n\n
# OUTPUT INSTRUCTION

Provide `proposed_fixes` as a JSON array of fix objects, every `action` equal
to `REPLACE`, each citing one statement id in `triple_ids`. `correct_value` is
a **string** containing valid Turtle: any `@prefix` declarations not already
bound above, then the one replacement statement.
Example: "cd:subject_1 onto:fittingProperty cd:object_1 ."
"""

_OUTPUT_INSTRUCTION_JSONLD = """\n\n
# OUTPUT INSTRUCTION

Provide `proposed_fixes` as a JSON array of fix objects, every `action` equal
to `REPLACE`, each citing one statement id in `triple_ids`. `correct_value` is
a **string** containing valid JSON for the one replacement statement: an
inline `@context` for any prefix not already bound above, the subject's `@id`
and the fitting property.
Example: "{\\"@id\\": \\"cd:subject_1\\", \\"onto:fittingProperty\\": {\\"@id\\": \\"cd:object_1\\"}}"
"""


def output_instruction_for(fmt: LLMGraphFormat) -> str:
    """The GRAPH FORMAT INSTRUCTION for ``correct_value``, by wire format."""
    if fmt == LLMGraphFormat.JSONLD:
        return _OUTPUT_INSTRUCTION_JSONLD
    return _OUTPUT_INSTRUCTION_TURTLE


def _describe(graph: RDFGraph, node: Node) -> str:
    if isinstance(node, Literal):
        return f'"{node}"'
    if not isinstance(node, URIRef):
        return "(a blank node)"
    row = _qname(graph, node)
    types = ", ".join(
        sorted(
            _qname(graph, t)
            for t in graph.objects(node, RDF.type)
            if isinstance(t, URIRef)
        )
    )
    if types:
        row += f"  a {types}"
    label = _first_label(graph, node)
    if label:
        row += f'  rdfs:label "{label}"'
    return row


def build_violations_chapter(
    fact_graph: RDFGraph,
    index: TripleIndex,
    domains: dict[Triple, set[URIRef]],
    candidates: dict[Triple, Sequence[tuple[URIRef, set[URIRef]]]],
) -> str:
    """The DOMAIN VIOLATIONS chapter: each statement by id, with its options.

    Args:
        fact_graph: The unit's current facts graph, for names and types.
        index: Ids over exactly the listed statements.
        domains: Per statement, the classes its property's domain declares.
        candidates: Per statement, the context properties whose domain fits
            its subject, each with its domain classes.

    Returns:
        The chapter text, or "" when nothing is listed.
    """
    if not index.by_id:
        return ""
    lines = ["\n\n# DOMAIN VIOLATIONS"]
    for triple_id in sorted(index.by_id):
        triple = index.by_id[triple_id]
        subject, predicate, obj = triple
        predicate_name = _qname(fact_graph, URIRef(str(predicate)))
        lines.append(
            f"\n[{triple_id}] {_describe(fact_graph, subject).split('  ')[0]} "
            f"{predicate_name} {_describe(fact_graph, obj).split('  ')[0]}"
        )
        lines.append(f"    subject: {_describe(fact_graph, subject)}")
        lines.append(f"    object:  {_describe(fact_graph, obj)}")
        applies = ", ".join(
            sorted(_qname(fact_graph, c) for c in domains.get(triple, set()))
        )
        lines.append(f"    {predicate_name} applies to: {applies}")
        options = candidates.get(triple) or ()
        if options:
            lines.append("    candidate properties for this subject:")
            for candidate, declared in options:
                domain = ", ".join(sorted(_qname(fact_graph, c) for c in declared))
                lines.append(
                    f"      {_qname(fact_graph, candidate)}  (applies to {domain})"
                )
    return "\n".join(lines) + "\n"
