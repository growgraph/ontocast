"""Cross-document identity: link each document's entities, resolve them on read.

Document graphs are never changed. For each document, :func:`links_for`
derives a links graph (``<graph>/identity-links``) from the document and the
catalog's declarations alone: one ``skos:closeMatch`` link from an entity to
a match node per declared key value, inverse-functional value, caller
identifier, or normalised name of a globally named class. The resolution
graph is the union of these links graphs; :func:`build`,
:func:`add_documents` and :func:`remove_documents` keep it in step with the
documents. :func:`cluster_of` reads which entities are one, with vetoes.
"""

from .clusters import ClusterMember, EntityCluster, KeptApart, cluster_of
from .declarations import IdentityDeclarations
from .links import identity_links_graph_uri, links_for, match_node_iri
from .resolution import (
    ResolutionReport,
    add_documents,
    build,
    document_graphs,
    remove_documents,
    write_document_links,
)

__all__ = [
    "ClusterMember",
    "EntityCluster",
    "IdentityDeclarations",
    "KeptApart",
    "ResolutionReport",
    "add_documents",
    "build",
    "cluster_of",
    "document_graphs",
    "identity_links_graph_uri",
    "links_for",
    "match_node_iri",
    "remove_documents",
    "write_document_links",
]
