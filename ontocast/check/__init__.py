"""Model-free checks of a user-supplied facts graph."""

from ontocast.check.facts import check_facts
from ontocast.check.report import (
    FactsCheckFinding,
    FactsCheckKind,
    FactsCheckMetrics,
    FactsCheckReport,
)
from ontocast.check.scope import catalog_schema_terms, declared_terms, instances

__all__ = [
    "FactsCheckFinding",
    "FactsCheckKind",
    "FactsCheckMetrics",
    "FactsCheckReport",
    "catalog_schema_terms",
    "check_facts",
    "declared_terms",
    "instances",
]
