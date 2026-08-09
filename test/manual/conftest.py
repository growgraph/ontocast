"""Keep the manual suite out of collection unless it is explicitly requested.

These modules drive live LLM providers, Fuseki, and Qdrant. They already guard
themselves with `skipif`, but a `skipif` still imports the module and reports a
skip on every run, so the opt-in was invisible in the summary line and paid for
in collection time. Ignoring them outright makes `ONTOCAST_RUN_MANUAL_TESTS=1`
the single switch that turns the suite on.
"""

import os

collect_ignore_glob = (
    [] if os.getenv("ONTOCAST_RUN_MANUAL_TESTS") == "1" else ["test_*.py"]
)
