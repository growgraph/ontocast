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

from pydantic import BaseModel, ConfigDict

TENANCY_SEP = "--"
DEFAULT_TENANT = "ontocast"
DEFAULT_PROJECT = "test"

StoreKind = Literal["facts", "ontologies"]


class TenancyScope(BaseModel):
    """A tenant/project partition and the backend names it resolves to.

    Frozen: a scope identifies a ToolBox in the registry, and a mutable key
    would let a rename silently point two scopes at the same entry.
    """

    model_config = ConfigDict(frozen=True)

    tenant: str
    project: str
    facts_name: str
    ontologies_name: str

    @classmethod
    def build(
        cls, tenant: str, project: str, *, sep: str = TENANCY_SEP
    ) -> "TenancyScope":
        """Resolve a tenant/project pair to its backend names.

        Args:
            tenant: Tenant identifier.
            project: Project identifier within the tenant.
            sep: Separator used in derived names.

        Returns:
            The resolved scope.

        Raises:
            ValueError: If either identifier is blank.
        """
        t, p = tenant.strip(), project.strip()
        if not t or not p:
            raise ValueError("tenant and project must be non-empty")
        return cls(
            tenant=t,
            project=p,
            facts_name=tenant_project_facts_name(t, p, sep=sep),
            ontologies_name=tenant_project_ontologies_name(t, p, sep=sep),
        )

    @property
    def key(self) -> tuple[str, str]:
        """Registry key for this scope."""
        return (self.tenant, self.project)


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
