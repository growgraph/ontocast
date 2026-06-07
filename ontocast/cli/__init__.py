"""Command-line interface tools for OntoCast.

This package provides Click entry points for interacting with the OntoCast
framework. Non-executable server and HTTP helpers live under
:mod:`ontocast.api`; shared file I/O helpers live under :mod:`ontocast.util.files`.

Commands (pipeline order):

Preprocess
  - ``pdfs-to-markdown``: Convert PDFs to Markdown JSON
  - ``split-chunks``: Split documents into chunks

Serve / process
  - ``ontocast`` (``cli.server:run``): Start the API server or batch-process local files

API clients
  - ``test-api``: Smoke-test the ``/process`` endpoint
  - ``batch_process``: Batch POST files to a running server (no console script)

Dev / analysis
  - ``cmp-states``: Compare serialized agent state JSON files
  - ``match-graphs``: Match TTL graphs locally
  - ``merge_ontologies``: Merge terminal ontologies from Fuseki (no console script)
  - ``plot-graph``: Generate workflow diagram images for docs
"""
