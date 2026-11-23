"""Messages of the facts check, one per kind, written for a person.

Every argument is already in display form (a compact IRI such as
``ex:worksFor``, or ``<iri>`` when no prefix fits).
"""

from collections.abc import Sequence


def _did_you_mean(suggestions: Sequence[str]) -> str:
    if not suggestions:
        return ""
    return f" Did you mean {' or '.join(suggestions)}?"


def _uses(count: int) -> str:
    return "once" if count == 1 else f"{count} times"


def non_catalog_vocabulary(term: str, uses: int, suggestions: Sequence[str]) -> str:
    """A term no ontology of the project defines."""
    return (
        f"{term} is not defined by any ontology in this project "
        f"(used {_uses(uses)})." + _did_you_mean(suggestions)
    )


def unknown_well_known_term(
    term: str, vocabulary_title: str, suggestions: Sequence[str]
) -> str:
    """A term missing from a well-known vocabulary."""
    return f"{term} is not a term of {vocabulary_title}." + _did_you_mean(suggestions)


def unknown_closed_range_term(term: str, suggestions: Sequence[str]) -> str:
    """An individual a value names where its property takes a known one."""
    return (
        f"{term} is neither an individual of the project's ontologies nor a "
        "term of a listed vocabulary, and its property takes one of those."
        + _did_you_mean(suggestions)
    )


def unknown_placeholder_term(term: str, suggestions: Sequence[str]) -> str:
    """A term in a placeholder namespace such as ``example.org``."""
    return (
        f"{term} is in a placeholder namespace; use a term from the "
        "project's ontologies instead." + _did_you_mean(suggestions)
    )


def domain_violation(
    subject: str, predicate: str, types: Sequence[str], expected: Sequence[str]
) -> str:
    """A subject whose type contradicts a property's ``rdfs:domain``."""
    return (
        f"{subject} is typed {', '.join(types)} but has {predicate}, "
        f"which applies to {', '.join(expected)}."
    )


def untyped_value(subject: str, predicate: str, lexical: str, datatype: str) -> str:
    """A plain or ``xsd:string`` value that reads as the property's datatype."""
    return f'"{lexical}" on {subject} {predicate} is given as text; it reads as {datatype}.'


def invalid_literal(
    subject: str,
    predicate: str,
    lexical: str,
    datatype: str,
    language: str | None = None,
) -> str:
    """A value that is not a value of its datatype or the declared one."""
    value = f'"{lexical}"' + (f"@{language}" if language else "")
    return f"{value} on {subject} {predicate} is not a valid {datatype}."


def temporal_range_mismatch(
    subject: str, predicate: str, lexical: str, datatype: str, declared: str
) -> str:
    """A value of another datatype where a date, time or duration belongs."""
    return f'{predicate} takes {declared}, but "{lexical}" on {subject} is {datatype}.'


def functional_violation(subject: str, predicate: str, count: int) -> str:
    """Several values on a property that allows one."""
    return f"{subject} has {count} values for {predicate}, which allows only one."


def catalog_subject(subject: str) -> str:
    """Statements about a term of the project's ontologies."""
    return (
        f"{subject} is a term of the project's ontologies; statements about "
        "it describe the schema, not facts."
    )


def schema_in_facts(term: str, role: str) -> str:
    """A class or property the upload declares itself."""
    return (
        f"{term} is declared as a {role} in this graph; a facts graph does "
        "not extend the project's ontologies, so its uses are undefined "
        "vocabulary."
    )


def shacl_violation(subject: str, message: str) -> str:
    """A result of the project's SHACL shapes."""
    return f"{subject}: {message}"


def untyped_instance(subject: str, suggestions: Sequence[str]) -> str:
    """An instance with no class of the project's ontologies."""
    return f"{subject} has no class from the project's ontologies." + _did_you_mean(
        suggestions
    )


def isolated_instance(subject: str) -> str:
    """An instance linked to no other instance."""
    return f"{subject} is not linked to any other instance."


def silent_instance(subject: str) -> str:
    """An instance that is linked to nothing and states no value."""
    return f"{subject} is not linked to any other instance and states no value."


def dangling_reference(obj: str, predicates: Sequence[str]) -> str:
    """An object that is referenced but never described."""
    return (
        f"{obj} is referenced via {', '.join(predicates)} but is never "
        "described: it has no statements of its own, no type and no label."
    )


def twin_labels(label: str, count: int) -> str:
    """One label carried by several instances."""
    return f'{count} instances share the label "{label}"; they may be duplicates.'
