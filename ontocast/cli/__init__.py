"""Command-line interface tools for OntoCast.

This package provides Click entry points for interacting with the OntoCast
framework. Non-executable server and HTTP helpers live under
:mod:`ontocast.api`; shared file I/O helpers live under :mod:`ontocast.util.files`.

Commands (pipeline order):

Preprocess
  - ``pdfs-to-markdown``: Convert PDFs to Markdown JSON

Serve / process
  - ``ontocast serve`` (``cli.server:cli``): Start the API server
  - ``ontocast process --input-path …``: Local in-process batch extraction
    (optional ``--output-dir`` / ``--facts-output-dir`` / ``--ontology-output-dir``)

API clients
  - ``test-api``: Smoke-test the ``/process`` endpoint

Dev / analysis
  - ``cmp-states``: Compare serialized agent state JSON files
  - ``match-graphs``: Match TTL graphs locally
  - ``plot-graph``: Generate workflow diagram images for docs
"""
