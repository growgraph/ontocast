"""Agent module for OntoCast.

This module provides a collection of agents that handle various aspects of ontology
processing, including document conversion, text chunking, fact aggregation, and
ontology management. Each agent is designed to perform a specific task in the
ontology processing pipeline.

The names re-exported here are the pipeline steps the graph and the unit loop
drive. Helper entry points that are only meaningful inside one agent -- the
``*_fresh`` render paths, the evidence planner/fetcher -- are reached through
their own module rather than from here.
"""

from .chunk_text import chunk_text
from .convert_document import convert_document
from .criticise_facts import criticise_facts
from .criticise_ontology import criticise_ontology
from .normalize_ontology import normalize_ontology_units
from .render_facts import render_facts
from .render_ontology import render_ontology
from .select_ontology_catalog import select_catalog_ontology_for_excerpt
from .serialize import serialize
from .summarize_chunks import ensure_unit_summary

__all__ = [
    "chunk_text",
    "convert_document",
    "criticise_facts",
    "criticise_ontology",
    "ensure_unit_summary",
    "normalize_ontology_units",
    "render_facts",
    "render_ontology",
    "select_catalog_ontology_for_excerpt",
    "serialize",
]
