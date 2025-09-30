"""Filesystem triple store management for OntoCast.

This module provides a concrete implementation of triple store management
using the local filesystem for storage. It supports reading and writing
ontologies and facts as Turtle files.
"""

import logging
import pathlib
from typing import Optional

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

    working_directory: Optional[pathlib.Path]
    ontology_path: Optional[pathlib.Path]

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

    def serialize_ontology(self, o: Ontology, **kwargs):
        """Store an ontology in the filesystem.

        This method stores the given ontology as a Turtle file in the
        working directory. The filename is generated using the ontology
        ID and version to ensure uniqueness.

        Args:
            o: The ontology to store.
            **kwargs: Additional keyword arguments for serialization (not used).

        Example:
            >>> ontology = Ontology(ontology_id="test", version="1.0", graph=graph)
            >>> manager.serialize_ontology(ontology)
            # Creates: working_directory/ontology_test_1.0.ttl
        """
        if self.working_directory is not None:
            fname = f"ontology_{o.ontology_id}_{o.version}"
            output_path = self.working_directory / f"{fname}.ttl"
            o.graph.serialize(format="turtle", destination=output_path)
            logger.info(f"Ontology saved to {output_path}")

    def serialize_facts(self, g: Graph, **kwargs):
        """Store a graph with facts in the filesystem.

        This method stores the given RDF graph containing facts as a
        Turtle file in the working directory. The filename can be
        customized using the spec parameter.

        Args:
            g: The RDF graph containing facts to store.
            **kwargs: Additional keyword arguments for serialization.
                spec: Optional specification for the filename. If provided as a string,
                      it will be processed to create a meaningful filename.

        Raises:
            TypeError: If spec is provided but not a string.

        Example:
            >>> facts = RDFGraph()
            >>> manager.serialize_facts(facts, spec="domain/subdomain")
            # Creates: working_directory/facts_domain_subdomain.ttl
        """
        spec = kwargs.pop("spec", None)
        if spec is None:
            fname = "current.ttl"
        elif isinstance(spec, str):
            s = spec.split("/")[-2:]
            s = "_".join([x for x in s if x])
            fname = f"facts_{s}.ttl"
        else:
            raise TypeError(f"string expected for spec {spec}")

        if self.working_directory is not None:
            filename = self.working_directory / fname
            g.serialize(format="turtle", destination=filename)
            logger.info(f"Facts saved to {filename}")
