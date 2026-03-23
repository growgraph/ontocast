"""Tenant/project naming helpers for triple-store datasets and vector collections.

Naming convention (separator default ``--``)::

    {tenant}{sep}{project}{sep}facts
    {tenant}{sep}{project}{sep}ontologies

Runtime tenant and project are taken from CLI flags or HTTP request parameters only
(not from environment variables). :data:`DEFAULT_TENANT` / :data:`DEFAULT_PROJECT`
are used when a parameter is omitted and for deriving initial Fuseki/Qdrant names
in configuration.
"""

from __future__ import annotations

from typing import Literal

TENANCY_SEP = "--"
DEFAULT_TENANT = "ontocast"
DEFAULT_PROJECT = "test"

StoreKind = Literal["facts", "ontologies"]


def tenant_project_store_name(
    tenant: str,
    project: str,
    kind: StoreKind,
    *,
    sep: str = TENANCY_SEP,
) -> str:
    """Return Fuseki dataset or Qdrant collection name for the given kind."""
    t = tenant.strip()
    p = project.strip()
    if not t or not p:
        raise ValueError("tenant and project must be non-empty")
    return f"{t}{sep}{p}{sep}{kind}"


def tenant_project_facts_name(
    tenant: str, project: str, *, sep: str = TENANCY_SEP
) -> str:
    """Facts dataset (Fuseki) or facts collection (Qdrant)."""
    return tenant_project_store_name(tenant, project, "facts", sep=sep)


def tenant_project_ontologies_name(
    tenant: str, project: str, *, sep: str = TENANCY_SEP
) -> str:
    """Ontologies dataset (Fuseki) or ontologies collection (Qdrant)."""
    return tenant_project_store_name(tenant, project, "ontologies", sep=sep)
