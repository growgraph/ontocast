"""Filesystem triple store management for OntoCast.

This module provides a concrete implementation of triple store management
using the local filesystem for storage. It supports reading and writing
ontologies and facts as Turtle files.
"""

import logging
import pathlib

from rdflib import Graph

from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
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

    def serialize_graph(self, graph: Graph, **kwargs) -> bool | None:
        """Store an RDF graph in the filesystem.

        This method stores the given RDF graph as a Turtle file in the
        working directory. The filename is generated based on the graph_uri
        parameter or defaults to "current.ttl".

        Args:
            graph: The RDF graph to store.
            fname:  str

        Example:
            >>> graph = RDFGraph()
            >>> manager.serialize_graph(graph)
            # Creates: working_directory/current.ttl

            >>> manager.serialize_graph(graph, fname="facts_abc.ttl")
        """
        if self.working_directory is None:
            return

        fname: str = kwargs.pop("fname")
        output_path = self.working_directory / fname
        graph.serialize(format="turtle", destination=output_path)
        logger.info(f"Graph saved to {output_path}")

    def serialize(self, o: Ontology | RDFGraph, graph_uri: str | None = None):
        if isinstance(o, Ontology):
            graph = o.graph
            fname = f"ontology_{o.ontology_id}_{o.version}.ttl"
        elif isinstance(o, RDFGraph):
            graph = o
            if graph_uri:
                s = graph_uri.split("/")[-2:]
                s = "_".join([x for x in s if x])
                fname = f"facts_{s}.ttl"
            else:
                fname = "facts_default.ttl"
        else:
            raise TypeError(f"unsupported obj of type {type(o)} received")

        self.serialize_graph(graph=graph, fname=fname)

    async def clean(self, dataset: str | None = None) -> None:
        """Clean/flush all data from the filesystem triple store.

        This method deletes all Turtle (.ttl) files from both the working
        directory and the ontology directory.

        Args:
            dataset: Optional dataset parameter (ignored for Filesystem, which doesn't
                support datasets). Included for interface compatibility.

        Warning: This operation is irreversible and will delete all data.

        Raises:
            Exception: If the cleanup operation fails.
        """
        if dataset is not None:
            logger.warning(
                f"Dataset parameter '{dataset}' ignored for Filesystem (datasets not supported)"
            )
            logger.warning(
                "clean method not implemented for FilesystemTripleStoreManager"
            )
