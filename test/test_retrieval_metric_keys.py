"""The retrieval-metric registry is a wire contract, not an internal name list.

``AgentState.retrieval_metrics`` is serialized verbatim into
``ProcessResultMetadata.retrieval_metrics`` and into the batch run manifest, so
a member's *value* is public. Renaming a member is free; changing its value is
a breaking API change. This module pins the values so the second cannot happen
by accident, and checks that no bare string literals have crept back into the
writers the registry replaced.
"""

import json
import re
from pathlib import Path

from ontocast.onto.enum import RetrievalMetric

# Every wire name, pinned. Add a line when a metric is added; change one only
# in a release that documents the break.
EXPECTED_WIRE_NAMES = {
    "ONTOLOGY_CONTEXT_MODE": "ontology_context_mode",
    "PATCH_RETRIEVAL": "patch_retrieval",
    "EMPTY_SNAPSHOT_REASON": "empty_snapshot_reason",
    "ONTOLOGY_WRITABLE_COUNT": "ontology_writable_count",
    "ONTOLOGY_PRIMARY_UNITS": "ontology_primary_units",
    "ONTOLOGY_SNAPSHOT_TRIPLES": "ontology_snapshot_triples",
    "FACTS_ANCHOR_COUNT": "facts_anchor_count",
    "FACTS_ANCHOR_UNITS": "facts_anchor_units",
    "FACTS_LLM_REPAIR_RENDERS_TOTAL": "facts_llm_repair_renders_total",
    "FACTS_LLM_REPAIR_RENDERS_FAILED": "facts_llm_repair_renders_failed",
    "FACTS_REPAIR_DELETE_ONLY": "facts_repair_delete_only",
    "FACTS_FINDINGS_RESIDUAL": "facts_findings_residual",
    "FACTS_REJECTED_MERGES": "facts_rejected_merges",
    "FACTS_MERGE_REPAIR_PASSES": "facts_merge_repair_passes",
    "FACTS_MERGE_VETOES": "facts_merge_vetoes",
    "FACTS_MERGE_REPAIRS_REJECTED": "facts_merge_repairs_rejected",
    "VALIDATED_WITHOUT_ONTOLOGY_CONTEXT": "validated_without_ontology_context",
    "FACTS_VALIDATION_FINDINGS": "facts_validation_findings",
    "FACTS_VALIDATION_ERRORS": "facts_validation_errors",
    "FACTS_SHACL_VIOLATIONS_BEFORE": "facts_shacl_violations_before",
    "FACTS_SHACL_VIOLATIONS_AFTER": "facts_shacl_violations_after",
    "FACTS_SHACL_REPAIRS": "facts_shacl_repairs",
    "FACTS_SHACL_AUTOFIX_PASSES": "facts_shacl_autofix_passes",
    "FACTS_SHACL_AUTOFIX_REVERTED": "facts_shacl_autofix_reverted",
    "STRUCTURAL_ONTOLOGY_COMPONENTS_MAX": "structural_ontology_components_max",
    "CONSISTENCY_CONFLICTS": "consistency_conflicts",
}

# Modules that write top-level retrieval metrics. patch_retriever.py is not
# here on purpose: its keys land nested under `patch_retrieval` and belong to
# the retriever's own namespace.
_WRITER_MODULES = (
    "ontocast/stategraph/node_factories.py",
    "ontocast/stategraph/context_resolver.py",
    "ontocast/api/process_helpers.py",
    "ontocast/tool/facts_validation",
)


def test_wire_names_are_pinned() -> None:
    actual = {member.name: member.value for member in RetrievalMetric}
    assert actual == EXPECTED_WIRE_NAMES


def test_members_serialize_as_their_wire_name() -> None:
    """A StrEnum key must round-trip through JSON as the bare string.

    This is the whole reason the registry can be a StrEnum rather than a table
    of module constants: the API response is unchanged by using it.
    """
    payload = {RetrievalMetric.FACTS_SHACL_REPAIRS: 3}
    assert json.loads(json.dumps(payload)) == {"facts_shacl_repairs": 3}


def test_no_bare_string_metric_keys_remain() -> None:
    """The writers must go through the registry, not string literals.

    A typo in a bare key is a silently missing metric, which is what the
    registry exists to prevent -- so a new literal creeping back in is a
    regression even though it runs fine.
    """
    repo_root = Path(__file__).resolve().parents[1]
    pattern = re.compile(r"""retrieval_metrics\[\s*["']""")
    offenders = []
    for module in _WRITER_MODULES:
        path = repo_root / module
        files = sorted(path.glob("*.py")) if path.is_dir() else [path]
        if any(pattern.search(file.read_text()) for file in files):
            offenders.append(module)
    assert offenders == []
