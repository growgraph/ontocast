"""Shared helpers for tests that need OntologySnapshot seeds."""

from __future__ import annotations

from ontocast.onto.enum import OntologyAssemblyMode
from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_snapshot import OntologySnapshot


def snapshot_from_ontology(
    ontology: Ontology,
    *,
    assembly_mode: OntologyAssemblyMode = OntologyAssemblyMode.FIXED_SINGLE_ONTOLOGY,
) -> OntologySnapshot:
    """Wrap a catalog Ontology as a prompt snapshot for unit-state tests."""
    return OntologySnapshot.from_ontology(ontology, assembly_mode=assembly_mode)


def empty_snapshot(
    *,
    assembly_mode: OntologyAssemblyMode = OntologyAssemblyMode.SELECTED_SINGLE_ONTOLOGY_LLM,
) -> OntologySnapshot:
    return OntologySnapshot.empty(assembly_mode=assembly_mode)
