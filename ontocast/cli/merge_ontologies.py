"""CLI script to merge terminal ontologies from Fuseki.

This script:
1. Fetches all terminal ontologies for a given IRI from Fuseki
2. Merges them pair-wise starting from the oldest pair
3. Continues until only one terminal ontology per IRI remains
4. Plots the ontology graph using pygraphviz
"""

import asyncio
import logging
from pathlib import Path

import click
from dotenv import load_dotenv

from ontocast.config import Config
from ontocast.onto.ontology_operations import (
    merge_terminal_ontologies,
    plot_ontology_graph,
)
from ontocast.tool.ontology_manager import OntologyManager
from ontocast.tool.triple_manager.fuseki import FusekiTripleStoreManager

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--env-file",
    type=click.Path(path_type=Path),
    required=True,
    default=".env",
    help="Path to .env file containing configuration",
)
@click.option(
    "--iri",
    type=str,
    required=True,
    help="IRI of the ontology to merge",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default="ontology_graph.png",
    help="Output path for the graph visualization",
)
def main(env_file: Path, iri: str, output: Path):
    """Merge terminal ontologies from Fuseki and plot the result."""
    # Load configuration
    load_dotenv(dotenv_path=env_file.expanduser())
    config = Config()

    # Validate configuration
    config.validate_llm_config()

    # Create Fuseki manager
    tool_config = config.get_tool_config()
    if not (tool_config.fuseki.uri and tool_config.fuseki.auth):
        raise ValueError("Fuseki configuration required (FUSEKI_URI and FUSEKI_AUTH)")

    fuseki_manager = FusekiTripleStoreManager(
        uri=tool_config.fuseki.uri,
        auth=tool_config.fuseki.auth,
        dataset=tool_config.fuseki.dataset,
        ontologies_dataset=tool_config.fuseki.ontologies_dataset,
    )

    # Create ontology manager
    ontology_manager = OntologyManager()

    # Merge ontologies
    async def run_merge():
        result = await merge_terminal_ontologies(fuseki_manager, ontology_manager, iri)
        if result:
            logger.info("Merge completed successfully")
            # Plot the graph
            plot_ontology_graph(ontology_manager, output, iri)
        else:
            logger.error("Merge failed - no result")

    asyncio.run(run_merge())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
