"""Neo4j triple store management for OntoCast.

This module provides a concrete implementation of triple store management
using Neo4j with the n10s (neosemantics) plugin. It handles RDF data
faithfully by using both n10s property graph representation and raw RDF
triple storage for accurate reconstruction.
"""

import logging
from typing import Optional

from rdflib.namespace import OWL, RDF

from ontocast.tool.triple_manager.core import TripleStoreManagerWithAuth

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None

from pydantic import Field

from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.util import derive_ontology_id

logger = logging.getLogger(__name__)


class Neo4jTripleStoreManager(TripleStoreManagerWithAuth):
    """Neo4j-based triple store manager using n10s (neosemantics) plugin.

    This implementation handles RDF data more faithfully by using both the n10s
    property graph representation and raw RDF triple storage for accurate reconstruction.
    It provides comprehensive ontology management with namespace-based organization.

    The manager uses Neo4j's n10s plugin for RDF operations, including:
    - RDF import and export via n10s
    - Ontology metadata storage and retrieval
    - Namespace-based ontology organization
    - Faithful RDF graph reconstruction

    Attributes:
        clean: Whether to clean the database on initialization.
        _driver: Private Neo4j driver instance.
    """

    clean: bool = Field(
        default=False, description="If True, clean the database on init."
    )
    _driver = None  # private attribute, not a pydantic field

    def __init__(self, uri=None, auth=None, clean=False, **kwargs):
        """Initialize the Neo4j triple store manager.

        This method sets up the connection to Neo4j, initializes the n10s
        plugin configuration, creates necessary constraints and indexes,
        and optionally cleans all data from the database.

        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687").
            auth: Authentication tuple (username, password) or string in "user/password" format.
            clean: If True, delete all nodes in the database on initialization.
            **kwargs: Additional keyword arguments passed to the parent class.

        Raises:
            ImportError: If the neo4j Python driver is not installed.

        Example:
            >>> manager = Neo4jTripleStoreManager(
            ...     uri="bolt://localhost:7687",
            ...     auth="neo4j/password",
            ...     clean=True
            ... )
        """
        super().__init__(
            uri=uri, auth=auth, env_uri="NEO4J_URI", env_auth="NEO4J_AUTH", **kwargs
        )
        self.clean = clean
        if GraphDatabase is None:
            raise ImportError("neo4j Python driver is not installed.")
        self._driver = GraphDatabase.driver(self.uri, auth=self.auth)

        with self._driver.session() as session:
            # Clean database if requested
            if self.clean:
                try:
                    session.run("MATCH (n) DETACH DELETE n")
                    logger.debug("Neo4j database cleaned (all nodes deleted)")
                except Exception as e:
                    logger.debug(f"Neo4j cleanup failed: {e}")

            # Initialize n10s configuration
            self._init_n10s_config(session)

            # Create constraints and indexes
            self._create_constraints_and_indexes(session)

    def _init_n10s_config(self, session):
        """Initialize n10s configuration with better RDF handling.

        This method configures the n10s plugin for optimal RDF handling.
        It sets up the configuration to preserve vocabulary URIs, handle
        multivalued properties, and maintain RDF types as nodes.

        Args:
            session: Neo4j session for executing configuration commands.
        """
        try:
            # Check if already configured
            result = session.run("CALL n10s.graphconfig.show()")
            if result.single():
                logger.debug("n10s already configured")
        except:
            pass

        try:
            session.run("""
                CALL n10s.graphconfig.init({
                    handleVocabUris: "KEEP",
                    handleMultival: "OVERWRITE",
                    typesToLabels: false,
                    keepLangTag: false,
                    keepCustomDataTypes: true,
                    handleRDFTypes: "NODES"
                })
            """)
            logger.debug("n10s configuration initialized")
        except Exception as e:
            logger.warning(f"n10s configuration failed: {e}")

    def _create_constraints_and_indexes(self, session):
        """Create necessary constraints and indexes for optimal performance.

        This method creates Neo4j constraints and indexes that are needed
        for efficient ontology operations and data integrity.

        Args:
            session: Neo4j session for executing constraint/index creation commands.
        """
        constraints = [
            "CREATE CONSTRAINT n10s_unique_uri IF NOT EXISTS FOR (r:Resource) REQUIRE r.uri IS UNIQUE",
            "CREATE CONSTRAINT ontology_iri_unique IF NOT EXISTS FOR (o:Ontology) REQUIRE o.uri IS UNIQUE",
            "CREATE INDEX namespace_prefix IF NOT EXISTS FOR (ns:Namespace) ON (ns.prefix)",
        ]

        for constraint in constraints:
            try:
                session.run(constraint)
                logger.debug(f"Created constraint/index: {constraint.split()[-1]}")
            except Exception as e:
                logger.debug(f"Constraint/index creation (might already exist): {e}")

    def _extract_namespace_prefix(self, uri: str) -> tuple[str, str]:
        """Extract namespace and local name from URI.

        This method parses a URI to extract the namespace and local name
        using common separators (#, /, :).

        Args:
            uri: The URI to parse.

        Returns:
            tuple[str, str]: A tuple of (namespace, local_name).

        Example:
            >>> manager._extract_namespace_prefix("http://example.org/onto#Class")
            ("http://example.org/onto#", "Class")
        """
        common_separators = ["#", "/", ":"]
        for sep in common_separators:
            if sep in uri:
                parts = uri.rsplit(sep, 1)
                if len(parts) == 2:
                    return parts[0] + sep, parts[1]
        return uri, ""

    def _get_ontology_namespaces(self, session) -> dict:
        """Get all known ontology namespaces from the database.

        This method queries the Neo4j database to retrieve all known
        namespace prefixes and their corresponding URIs.

        Args:
            session: Neo4j session for executing queries.

        Returns:
            dict: Dictionary mapping namespace prefixes to URIs.
        """
        result = session.run("""
            MATCH (ns:Namespace)
            RETURN ns.prefix as prefix, ns.uri as uri
            UNION
            MATCH (o:Ontology)
            RETURN null as prefix, o.uri as uri
        """)

        namespaces = {}
        for record in result:
            uri = record.get("uri")
            prefix = record.get("prefix")
            if uri:
                if prefix:
                    namespaces[prefix] = uri
                else:
                    # Extract potential namespace from ontology URI
                    ns, _ = self._extract_namespace_prefix(uri)
                    if ns != uri:  # Only if we actually found a namespace
                        namespaces[ns] = ns

        return namespaces

    def fetch_ontologies(self) -> list[Ontology]:
        """Fetch ontologies from Neo4j with faithful RDF reconstruction.

        This method retrieves all ontologies from Neo4j and reconstructs
        their RDF graphs faithfully. It uses a multi-step process:

        1. Identifies distinct ontologies by their namespace URIs
        2. Fetches all entities belonging to each ontology
        3. Reconstructs the RDF graph faithfully using stored triples when available
        4. Falls back to n10s property graph conversion when needed

        Returns:
            list[Ontology]: List of all ontologies found in the database.

        Example:
            >>> ontologies = manager.fetch_ontologies()
            >>> for onto in ontologies:
            ...     print(f"Found ontology: {onto.iri}")
        """
        ontologies = []

        with self._driver.session() as session:
            try:
                # First, try to get explicitly stored ontology metadata
                ontology_iris = self._fetch_ontology_iris(session)

                if ontology_iris:
                    for ont_iri in ontology_iris:
                        ontology = self._reconstruct_ontology_from_metadata(
                            session, ont_iri
                        )
                        if ontology:
                            ontologies.append(ontology)

            except Exception as e:
                logger.error(f"Error in fetch_ontologies: {e}")

        logger.info(f"Successfully loaded {len(ontologies)} ontologies")
        return ontologies

    def _fetch_ontology_iris(self, session) -> list[str]:
        """Fetch explicit ontology metadata from Neo4j.

        This method queries Neo4j to find all entities that are explicitly
        typed as owl:Ontology.

        Args:
            session: Neo4j session for executing queries.

        Returns:
            list[str]: List of ontology IRIs found in the database.
        """
        result = session.run(f"""
            MATCH (o)-[:`{str(RDF.type)}`]->(t:Resource {{ uri: "{str(OWL.Ontology)}" }})
            WHERE o.uri IS NOT NULL
            RETURN
              o.uri AS iri
        """)

        iris = []
        for record in result:
            iri = record.get("iri", None)
            iris += [iri]
        iris = [iri for iri in iris if iri is not None]
        return iris

    def _reconstruct_ontology_from_metadata(self, session, iri) -> Optional[Ontology]:
        """Reconstruct an ontology from its metadata and related entities.

        This method takes an ontology IRI and reconstructs the complete
        ontology by fetching all related entities from the namespace.

        Args:
            session: Neo4j session for executing queries.
            iri: The ontology IRI to reconstruct.

        Returns:
            Optional[Ontology]: The reconstructed ontology, or None if failed.
        """
        namespace_uri, _ = self._extract_namespace_prefix(iri)

        logger.debug(f"Reconstructing ontology: {iri} with namespace: {namespace_uri}")

        # Fallback to n10s export for this namespace
        graph = self._export_namespace_via_n10s(session, namespace_uri)
        if graph and len(graph) > 0:
            return self._create_ontology_object(iri, iri, graph)

    def _export_namespace_via_n10s(
        self, session, namespace_uri: str
    ) -> Optional[RDFGraph]:
        """Export entities belonging to a namespace using n10s.

        This method uses Neo4j's n10s plugin to export all entities
        belonging to a specific namespace as RDF triples.

        Args:
            session: Neo4j session for executing queries.
            namespace_uri: The namespace URI to export.

        Returns:
            Optional[RDFGraph]: The exported RDF graph, or None if failed.
        """
        try:
            result = session.run(
                f"""
                CALL n10s.rdf.export.cypher(
                    'MATCH (n)-[r]->(m) WHERE n.uri STARTS WITH "{namespace_uri}" RETURN n,r,m',
                    {{format: 'Turtle'}}
                )
                YIELD subject, predicate, object, isLiteral, literalType, literalLang
                RETURN subject, predicate, object, isLiteral, literalType, literalLang
                """
            )

            # Process into Turtle format
            turtle_lines = []

            for record in result:
                subj = record["subject"]
                pred = record["predicate"]
                obj = record["object"]
                is_literal = record["isLiteral"]
                literal_type = record["literalType"]
                literal_lang = record["literalLang"]

                # Format object
                if is_literal:
                    # Escape special characters in literals
                    obj = obj.replace('"', r"\"")
                    obj_str = f'"{obj}"'

                    # Add datatype or language tag if present
                    if literal_lang:
                        obj_str += f"@{literal_lang}"
                    elif literal_type:
                        obj_str += f"^^<{literal_type}>"
                else:
                    obj_str = f"<{obj}>"

                # Format triple
                turtle_lines.append(f"<{subj}> <{pred}> {obj_str} .")

            # Combine into single string
            turtle_string = "\n".join(turtle_lines)

            if turtle_string.strip():
                graph = RDFGraph()
                graph.parse(data=turtle_string, format="turtle")
                logger.debug(
                    f"Exported {len(graph)} triples via n10s for namespace {namespace_uri}"
                )
                return graph
            return None

        except Exception as e:
            logger.debug(
                f"Failed to export via n10s for namespace {namespace_uri}: {e}"
            )

        return None

    def _create_ontology_object(
        self, iri: str, metadata: dict, graph: RDFGraph
    ) -> Ontology:
        """Create an Ontology object from IRI, metadata, and graph.

        Args:
            iri: The ontology IRI.
            metadata: Metadata dictionary (currently unused, kept for compatibility).
            graph: The RDF graph containing the ontology data.

        Returns:
            Ontology: The created ontology object.
        """
        ontology_id = derive_ontology_id(iri)
        return Ontology(graph=graph, iri=iri, ontology_id=ontology_id)

    def serialize_ontology(self, o: Ontology, **kwargs):
        """Serialize an ontology to Neo4j with both n10s and raw triple storage.

        This method stores the given ontology in Neo4j using the n10s plugin
        for RDF import. The ontology is stored as RDF triples that can be
        faithfully reconstructed later.

        Args:
            o: The ontology to store.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            Any: The result summary from n10s import operation.
        """
        turtle_data = o.graph.serialize(format="turtle")

        with self._driver.session() as session:
            # Store via n10s for graph queries
            result = session.run(
                "CALL n10s.rdf.import.inline($ttl, 'Turtle')", ttl=turtle_data
            )
            summary = result.single()

        return summary

    def serialize_facts(self, g: RDFGraph, **kwargs):
        """Serialize facts (RDF graph) to Neo4j.

        This method stores the given RDF graph containing facts in Neo4j
        using the n10s plugin for RDF import.

        Args:
            g: The RDF graph containing facts to store.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            Any: The result summary from n10s import operation.
        """
        turtle_data = g.serialize(format="turtle")

        with self._driver.session() as session:
            # Store via n10s
            result = session.run(
                "CALL n10s.rdf.import.inline($ttl, 'Turtle')", ttl=turtle_data
            )
            summary = result.single()

        return summary

    def close(self):
        """Close the Neo4j driver connection.

        This method should be called when the manager is no longer needed
        to properly close the database connection and free resources.
        """
        if self._driver:
            self._driver.close()
