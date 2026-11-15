"""What makes a good ontology term: one statement shared by every ontology prompt.

The plan, the render, consolidation and the ontology critic judge new terms by
the same test, so it is written once here and quoted by each.
"""

INSTANCE_CRITERION = (
    "**A term is justified by its instances.** A class is a kind whose instances "
    "are things of the domain, named in this text or not. If you cannot say what "
    "its instances are, it is not a class: one particular thing is an individual "
    "(a fact), and a distinction between members (a kind, a status, a format) is "
    "a property value. A property relates instances of its domain to instances "
    "of its range, and a side is a claim about every use of the property: give "
    "each side the class of everything the property can apply to in the domain, "
    "which is usually broader than this document's subject. `owl:Thing` or "
    "`skos:Concept` almost never fits; values from a fixed list are the "
    "instances of a class of their own under `skos:Concept`."
)

__all__ = ["INSTANCE_CRITERION"]
