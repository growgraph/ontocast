"""Triple store management tools for OntoCast.

This module provides functionality for managing RDF triple stores, including
abstract interfaces and concrete implementations for different triple store backends.
"""

import abc
import asyncio
import os
from collections.abc import Sequence
from typing import Any

from pydantic import Field
from rdflib import RDF, Graph

from ontocast.onto.constants import PROV, RDF_REIFIES, SCHEMA
from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_header import OntologyHeader
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.tenancy import TENANCY_SEP, StoreKind
from ontocast.tool import Tool


class TripleStoreUnavailableError(RuntimeError):
    """The triple store could not answer a read that must not degrade silently.

    Raised instead of returning an empty result when the difference between
    "the store is unreachable" and "the store is empty" is load-bearing --
    catalog listing being the case that matters, since an empty catalog is
    grounds for pruning the vector index.
    """


class TripleStoreManager(Tool):
    """Base class for managing RDF triple stores.

    This class defines the interface for triple store management operations,
    including fetching and storing ontologies and their graphs. All concrete
    triple store implementations should inherit from this class.

    This is an abstract base class that must be implemented by specific
    triple store backends (e.g., Fuseki, In-Memory).
    """

    def __init__(self, **kwargs: Any):
        """Initialize the triple store manager.

        Args:
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)

    @abc.abstractmethod
    def fetch_ontologies(self) -> list[Ontology]:
        """Fetch all available ontologies from the triple store.

        This method should retrieve all ontologies stored in the triple store
        and return them as Ontology objects with their associated RDF graphs.

        Returns:
            list[Ontology]: List of available ontologies with their graphs.
        """
        return []

    async def afetch_ontologies(self) -> list[Ontology]:
        """Async fetch helper for backends without native async I/O."""
        return await asyncio.to_thread(self.fetch_ontologies)

    @abc.abstractmethod
    def serialize_graph(self, graph: Graph, **kwargs: Any) -> bool:
        """Store an RDF graph in the triple store."""
        pass

    async def aserialize_graph(self, graph: Graph, **kwargs: Any) -> bool:
        """Async serialize helper for backends without native async I/O."""
        return await asyncio.to_thread(self.serialize_graph, graph, **kwargs)

    @abc.abstractmethod
    def serialize(self, o: Ontology | RDFGraph, **kwargs: Any) -> bool:
        """Store an Ontology or RDFGraph in the triple store."""
        pass

    async def aserialize(self, o: Ontology | RDFGraph, **kwargs: Any) -> bool:
        """Async serialize helper for backends without native async I/O."""
        return await asyncio.to_thread(self.serialize, o, **kwargs)

    async def async_init(self) -> None:
        """Backend warmup (e.g. ensure datasets exist). No-op by default."""

    async def update_tenancy(
        self,
        tenant: str,
        project: str,
        *,
        sep: str = TENANCY_SEP,
    ) -> None:
        """Switch the active tenant/project partition when supported."""
        if not self.supports_tenancy_partition():
            raise NotImplementedError(
                f"{type(self).__name__} does not isolate data by tenant/project"
            )
        raise NotImplementedError(
            f"{type(self).__name__} must implement update_tenancy()"
        )

    async def drop_named_graph(
        self, graph_uri: str, *, store: StoreKind = "ontologies"
    ) -> None:
        """Drop a single named graph."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support drop_named_graph()"
        )

    async def drop_all_ontology_graphs_for_iri(
        self, ontology_iri: str, *, store: StoreKind = "ontologies"
    ) -> None:
        """Remove named graphs for ``ontology_iri`` (base and versioned).

        Args:
            ontology_iri: Base IRI whose ``iri`` and ``iri#...`` graphs are dropped.
            store: Partition to drop from. Shapes documents are addressed the same
                way and live in ``"shapes"``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support drop_all_ontology_graphs_for_iri()"
        )

    @classmethod
    def _provenance_source_nodes(cls, graph: Graph) -> set:
        """Return chunk/source nodes whose triples are provenance scaffolding."""
        derived_from = set(graph.objects(None, PROV.wasDerivedFrom))
        entity_nodes = set(graph.subjects(RDF.type, PROV.Entity))
        text_chunk_nodes = set(graph.subjects(RDF.type, SCHEMA.Text))
        chunk_metadata_nodes = entity_nodes & text_chunk_nodes
        return derived_from | chunk_metadata_nodes

    @classmethod
    def strip_provenance(cls, graph: Graph) -> RDFGraph:
        """Return a graph without reification/provenance scaffolding triples."""
        clean = RDFGraph()
        for prefix, namespace in graph.namespaces():
            clean.bind(prefix, namespace)

        reifier_nodes = set(graph.subjects(RDF_REIFIES, None))
        source_nodes = cls._provenance_source_nodes(graph)

        for subject, predicate, object_ in graph:
            if predicate in {RDF_REIFIES, PROV.wasDerivedFrom}:
                continue
            if subject in reifier_nodes:
                continue
            if subject in source_nodes:
                continue
            clean.add((subject, predicate, object_))

        return clean

    @abc.abstractmethod
    async def clean(self, *, include_shapes: bool = False) -> None:
        """Clean/flush data managed by this store (backend-specific scope).

        The shapes partition is **retained by default**. Facts and ontologies are
        reproducible from a rerun; shapes are the deployment's validation
        contract, and dropping them turns the SHACL gate off silently -- a
        cleared run then reports ``shacl_evaluated: null`` rather than failing.

        Args:
            include_shapes: Also drop the shapes partition. Opt in explicitly.

        Warning: This operation is irreversible and will delete data.

        Raises:
            NotImplementedError: If the triple store doesn't support cleaning.
        """
        raise NotImplementedError("clean() method must be implemented by subclasses")

    def supports_tenancy_partition(self) -> bool:
        """True if this backend isolates facts/ontologies by :func:`tenant_project_*` names."""
        return False

    async def close(self) -> None:
        """Release any connection held by this backend.

        Default is a no-op for in-process backends.
        """
        return None

    def last_catalog_was_complete(self) -> bool:
        """True when the most recent full catalog fetch returned every graph.

        Consulted before destructive reconciliation (vector-store orphan
        pruning): a backend that fetched only part of its catalog reports False
        so callers treat the result as non-authoritative rather than concluding
        that the missing ontologies were deleted. Backends that cannot fetch
        partially always report True.
        """
        return True

    def supports_sparql_select(self) -> bool:
        """True when :meth:`aselect` reaches a real SPARQL engine.

        Callers branch on this to choose targeted queries over materializing the
        whole catalog. Backends returning ``False`` still answer every catalog
        method correctly, just by fetching more than they need.
        """
        return False

    async def aselect(
        self, query: str, *, store: StoreKind = "ontologies"
    ) -> list[dict[str, str]]:
        """Run a SPARQL SELECT against the active partition.

        Rows map variable name to the term's **lexical value** only; term kind and
        datatype are not preserved, so constrain kinds in the query itself
        (``FILTER(isIRI(?x))``). Unbound variables are absent from the row dict.

        Implementations must raise rather than return an empty list on failure --
        an empty result set is indistinguishable from "nothing matched", which
        would silently disable callers that treat no-rows as a valid answer.

        Args:
            query: A SPARQL SELECT query.
            store: Which partition to query -- ``"ontologies"``, ``"facts"`` or
                ``"shapes"``.

        Returns:
            list[dict[str, str]]: One dict per solution.

        Raises:
            NotImplementedError: If the backend has no SPARQL engine.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support aselect()")

    def supports_sparql_construct(self) -> bool:
        """True when :meth:`aconstruct` reaches a real SPARQL engine.

        Separate from :meth:`supports_sparql_select` because a backend can answer
        row queries without being able to return triples: the Fuseki SELECT path
        speaks ``application/sparql-results+json`` only.
        """
        return False

    async def aconstruct(
        self, query: str, *, store: StoreKind = "ontologies"
    ) -> RDFGraph:
        """Run a SPARQL CONSTRUCT against the active partition.

        Unlike :meth:`aselect`, the result carries real RDF terms, so blank nodes
        and datatypes survive. Prefix bindings do **not** -- they are serialization
        metadata rather than triples, and must be re-sourced by the caller.

        Implementations must raise rather than return an empty graph on failure,
        for the same reason :meth:`aselect` must raise: an empty result is
        indistinguishable from "nothing matched".

        Args:
            query: A SPARQL CONSTRUCT (or DESCRIBE) query.
            store: Which partition to query -- ``"ontologies"``, ``"facts"`` or
                ``"shapes"``.

        Returns:
            RDFGraph: The constructed triples, without prefix bindings.

        Raises:
            NotImplementedError: If the backend has no SPARQL engine.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support aconstruct()"
        )

    async def afetch_ontology_catalog(self) -> list[OntologyHeader]:
        """Fetch per-named-graph ontology header metadata.

        Headers carry the lineage fields terminal-version selection needs without
        the graphs themselves. The default implementation materializes the catalog
        and derives headers from it; SPARQL-capable backends should override with a
        single SELECT.

        Note the default returns one header per *terminal* ontology (whatever
        :meth:`afetch_ontologies` returns), while a native implementation returns
        one per *stored version*. Callers that re-run terminal selection over the
        result are correct either way; that is why they should.

        Returns:
            list[OntologyHeader]: Header metadata for stored ontologies.
        """
        return [
            OntologyHeader.from_ontology(onto)
            for onto in await self.afetch_ontologies()
        ]

    async def afetch_ontologies_by_iri(self, iris: Sequence[str]) -> list[Ontology]:
        """Fetch terminal ontologies restricted to ``iris``.

        Args:
            iris: Ontology IRIs to fetch. Empty means "no restriction", matching
                how :meth:`ontocast.tool.sparql.SPARQLTool._build_induced_subgraph`
                treats an empty ontology filter.

        Returns:
            list[Ontology]: The requested ontologies, with graphs.
        """
        if not iris:
            return await self.afetch_ontologies()
        wanted = set(iris)
        return [onto for onto in await self.afetch_ontologies() if onto.iri in wanted]

    async def clean_tenancy(
        self, tenant: str, project: str, *, include_shapes: bool = False
    ) -> None:
        """Remove all triples for datasets derived from ``tenant`` / ``project``.

        Shapes are retained unless ``include_shapes`` is set -- see :meth:`clean`.

        Backends without per-tenant partitions raise :class:`NotImplementedError`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not isolate data by tenant/project"
        )


class TripleStoreManagerWithAuth(TripleStoreManager):
    """Base class for triple store managers that require authentication.

    This class provides common functionality for triple store managers that
    need URI and authentication credentials. It handles environment variable
    loading and credential parsing.

    Attributes:
        uri: The connection URI for the triple store.
        auth: Authentication tuple (username, password) for the triple store.
    """

    uri: str | None = Field(default=None, description="Triple store connection URI")
    auth: tuple | None = Field(
        default=None, description="Triple store authentication tuple (user, password)"
    )

    def __init__(
        self,
        uri: str | None = None,
        auth: tuple[str, str] | str | None = None,
        env_uri: str | None = None,
        env_auth: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the triple store manager with authentication.

        This method handles loading URI and authentication credentials from
        either direct parameters or environment variables. It also parses
        authentication strings in the format "user/password".

        Args:
            uri: Direct URI for the triple store connection.
            auth: Direct authentication tuple or string in "user/password" format.
            env_uri: Environment variable name for the URI (e.g., "NEO4J_URI").
            env_auth: Environment variable name for authentication (e.g., "NEO4J_AUTH").
            **kwargs: Additional keyword arguments passed to the parent class.

        Raises:
            ValueError: If the authentication string is neither "user/password"
                nor "user:password".

        Example:
            >>> manager = TripleStoreManagerWithAuth(
            ...     env_uri="NEO4J_URI",
            ...     env_auth="NEO4J_AUTH"
            ... )
        """
        # Use env vars if not provided
        uri = uri or (os.getenv(env_uri) if env_uri else None)
        auth_env = auth or (os.getenv(env_auth) if env_auth else None)

        if auth_env and not isinstance(auth_env, tuple):
            # Both separators are accepted. The colon form is what Fuseki's own
            # documentation and most HTTP tooling use, and rejecting it meant
            # `FUSEKI_AUTH=admin:secret` -- recommended by several of our own
            # docs pages -- failed startup outright. Whichever separator comes
            # first wins, so a password containing the other still round-trips.
            positions = [
                (auth_env.index(sep), sep) for sep in ("/", ":") if sep in auth_env
            ]
            if positions:
                index, _ = min(positions)
                user, password = auth_env[:index], auth_env[index + 1 :]
                if not user:
                    raise ValueError(
                        f"{env_auth or 'TRIPLESTORE_AUTH'} has an empty username; "
                        "expected 'user/password' or 'user:password'"
                    )
                auth = (user, password)
            else:
                raise ValueError(
                    f"{env_auth or 'TRIPLESTORE_AUTH'} must be in 'user/password' "
                    "or 'user:password' format"
                )
        elif isinstance(auth_env, tuple):
            auth = auth_env
        # else: auth remains None

        super().__init__(uri=uri, auth=auth, **kwargs)
