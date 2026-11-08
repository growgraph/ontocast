"""Deterministic validation of ontology deltas.

Two lanes with different authorities: ``unit_findings`` validates a unit's
delta against its own (possibly partial) prompt snapshot inside the loop;
``reconcile`` checks the merged document delta against the FULL catalog
terminals at reduce time, where duplicates invisible to a retrieved snapshot
become detectable.
"""

from ontocast.tool.ontology_validation.reconcile import (
    MintedDuplicate,
    apply_minted_duplicate_rewrites,
    detect_minted_duplicates,
)
from ontocast.tool.ontology_validation.shape import (
    CatalogSurfaceIndex,
    build_catalog_surface_index,
)
from ontocast.tool.ontology_validation.unit_findings import (
    collect_ontology_unit_findings,
    count_fixes_targeting_snapshot,
)

__all__ = [
    "CatalogSurfaceIndex",
    "MintedDuplicate",
    "apply_minted_duplicate_rewrites",
    "build_catalog_surface_index",
    "collect_ontology_unit_findings",
    "count_fixes_targeting_snapshot",
    "detect_minted_duplicates",
]
