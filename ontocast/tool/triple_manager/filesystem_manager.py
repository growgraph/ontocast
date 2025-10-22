"""Filesystem triple store management for OntoCast.

This module provides a concrete implementation of triple store management
using the local filesystem for storage. It supports reading and writing
ontologies and facts as Turtle files.
"""

import logging
import pathlib

from rdflib import Graph

from ontocast.onto.ontology import Ontology
from ontocast.tool.triple_manager.core import TripleStoreManager

logger = logging.getLogger(__name__)


class FilesystemTripleStoreManager(TripleStoreManager):
    """Filesystem-based implementation of triple store management.

    This class provides a concrete implementation of triple store management
    using the local filesystem for storage. It reads and writes ontologies
    and facts as Turtle (.ttl) files in specified directories.

    The manager supports:
    - Loading ontologies from a dedicated ontology directory
    - Storing ontologies with versioned filenames
    - Storing facts with customizable filenames based on specifications
    - Error handling for file operations

    Attributes:
        working_directory: Path to the working directory for storing data.
        ontology_path: Optional path to the ontology directory for loading ontologies.
    """

    working_directory: pathlib.Path | None
    ontology_path: pathlib.Path | None

    def __init__(self, **kwargs):
        """Initialize the filesystem triple store manager.

        This method sets up the filesystem manager with the specified
        working and ontology directories.

        Args:
            **kwargs: Additional keyword arguments passed to the parent class.
                working_directory: Path to the working directory for storing data.
                ontology_path: Path to the ontology directory for loading ontologies.

        Example:
            >>> manager = FilesystemTripleStoreManager(
            ...     working_directory="/path/to/work",
            ...     ontology_path="/path/to/ontologies"
            ... )
        """
        super().__init__(**kwargs)

    def fetch_ontologies(self) -> list[Ontology]:
        """Fetch all available ontologies from the filesystem.

        This method scans the ontology directory for Turtle (.ttl) files
        and loads each one as an Ontology object. Files are processed
        in sorted order for consistent results.

        Returns:
            list[Ontology]: List of all ontologies found in the ontology directory.

        Example:
            >>> ontologies = manager.fetch_ontologies()
            >>> for onto in ontologies:
            ...     print(f"Loaded ontology: {onto.ontology_id}")
        """
        ontologies = []
        if self.ontology_path is not None:
            sorted_files = sorted(self.ontology_path.glob("*.ttl"))
            for fname in sorted_files:
                try:
                    ontology = Ontology.from_file(fname)
                    ontologies.append(ontology)
                    logger.debug(f"Successfully loaded ontology from {fname}")
                except Exception as e:
                    logger.error(f"Failed to load ontology {fname}: {str(e)}")
        return ontologies

    def serialize_graph(
        self, graph: Graph, graph_uri: str | None = None
    ) -> bool | None:
        """Store an RDF graph in the filesystem.

        This method stores the given RDF graph as a Turtle file in the
        working directory. The filename is generated based on the graph_uri
        parameter or defaults to "current.ttl".

        Args:
            graph: The RDF graph to store.
            graph_uri: Optional URI to use for filename generation. If provided,
                      it will be processed to create a meaningful filename.

        Example:
            >>> graph = RDFGraph()
            >>> manager.serialize_graph(graph)
            # Creates: working_directory/current.ttl

            >>> manager.serialize_graph(graph, graph_uri="domain/subdomain")
            # Creates: working_directory/facts_domain_subdomain.ttl
        """
        if self.working_directory is None:
            return

        if graph_uri is None:
            fname = "current.ttl"
        else:
            s = graph_uri.split("/")[-2:]
            s = "_".join([x for x in s if x])
            fname = f"facts_{s}.ttl"

        output_path = self.working_directory / fname
        graph.serialize(format="turtle", destination=output_path)
        logger.info(f"Graph saved to {output_path}")
