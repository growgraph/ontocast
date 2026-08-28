"""Deterministic validation, findings, and LLM-free repair for rendered facts.

Split by concern: ``terms`` (catalog inventory, namespace closure,
``ValidationPolicy``), ``literal_repair`` (parse-time rewrites),
``unit_findings`` (per-unit findings for repair renders), ``shacl``
(execution, autofix, catalog lint), ``gate`` (document-level validation).
This package is the public surface; import from here.
"""

from ontocast.tool.facts_validation.gate import (
    FactsValidationReport,
    record_facts_gate_metrics,
    summarize_conformance,
    validate_aggregated_facts,
)
from ontocast.tool.facts_validation.literal_repair import (
    normalize_literals_against_schema,
    promote_degenerate_bounds,
    promote_degenerate_bounds_from_vocabulary,
    repair_literal_type_objects,
    repair_property_aliases,
    resolve_code_literals,
)
from ontocast.tool.facts_validation.shacl import (
    ShaclRepairResult,
    ShaclViolation,
    apply_shacl_repairs,
    collect_shacl_shapes,
    run_shacl,
    shacl_catalog_contradictions,
)
from ontocast.tool.facts_validation.terms import (
    ValidationPolicy,
    build_surface_index,
    collect_catalog_terms,
    collect_declared_namespaces,
    expand_vocabulary_terms,
    resolve_unique_surface,
)
from ontocast.tool.facts_validation.unit_findings import (
    collect_unit_findings,
    domain_violation_findings,
    format_findings_for_prompt,
)

__all__ = [
    "FactsValidationReport",
    "ShaclRepairResult",
    "ShaclViolation",
    "ValidationPolicy",
    "apply_shacl_repairs",
    "build_surface_index",
    "collect_catalog_terms",
    "collect_declared_namespaces",
    "collect_shacl_shapes",
    "collect_unit_findings",
    "domain_violation_findings",
    "expand_vocabulary_terms",
    "format_findings_for_prompt",
    "normalize_literals_against_schema",
    "promote_degenerate_bounds",
    "promote_degenerate_bounds_from_vocabulary",
    "record_facts_gate_metrics",
    "repair_literal_type_objects",
    "repair_property_aliases",
    "resolve_code_literals",
    "resolve_unique_surface",
    "run_shacl",
    "shacl_catalog_contradictions",
    "summarize_conformance",
    "validate_aggregated_facts",
]
