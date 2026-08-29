"""Real malformed renders, pinned so the repair cannot silently regress.

These are verbatim `gpt-5-mini` responses in which single-operation ontology
updates over roughly 4k characters closed their brackets in the wrong *kind*
-- `] }` where `} ]` was due -- and were dropped, costing content units their
entire ontology output.

They carry the pre-flattening `graph_update.triple_operations` envelope on
purpose: what is pinned here is the JSON repair, which must keep working for
malformations the current wire cannot itself produce. The wire fix that removed
the singleton array is covered separately in test_graph_format_profile.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ontocast.agent.common import parse_json_object, repair_bracket_kinds

FIXTURE = Path(__file__).parent / "data" / "llm_malformed_graph_updates.json"


def _bodies() -> list[dict]:
    return json.loads(FIXTURE.read_text())


def test_fixture_bodies_are_genuinely_unparsable() -> None:
    bodies = _bodies()
    assert len(bodies) >= 13
    for body in bodies:
        with pytest.raises(json.JSONDecodeError):
            json.loads(body["content"], strict=False)


@pytest.mark.parametrize("body", _bodies(), ids=lambda b: b["id"])
def test_malformed_render_is_recovered_with_its_payload_intact(body: dict) -> None:
    parsed = parse_json_object(body["content"])

    operations = parsed["graph_update"]["triple_operations"]
    assert operations, "repair must not yield an empty patch"
    for op in operations:
        assert op["type"] in {"insert", "delete"}
        assert op["graph"]["@graph"], "the JSON-LD payload must survive intact"


@pytest.mark.parametrize("body", _bodies(), ids=lambda b: b["id"])
def test_repair_only_substitutes_characters(body: dict) -> None:
    """Never insert, delete, or reorder: same length, same non-bracket text."""
    original = body["content"]
    repaired, fixes = repair_bracket_kinds(original)

    assert fixes > 0
    assert len(repaired) == len(original)
    differing = [i for i, (a, b) in enumerate(zip(original, repaired)) if a != b]
    assert len(differing) == fixes
    for i in differing:
        assert {original[i], repaired[i]} in ({"}", "]"},)


def test_flat_wire_survives_the_full_sanitize_parse_compile_chain() -> None:
    """The repair must still cover the shape that replaced the failing one.

    Exercises the real path: sanitizers -> parse -> schema validation with the
    JSON-LD wire context -> patch compilation.
    """
    from ontocast.agent.common import (
        strip_json_comments,
        strip_trailing_commas,
        unescape_json_delimiters,
    )
    from ontocast.onto.enum import LLMGraphFormat
    from ontocast.onto.model import GraphUpdateRenderReport

    payload = {
        "insert_graph": {
            "@context": {
                "owl": "http://www.w3.org/2002/07/owl#",
                "ex": "https://example.org/onto#",
            },
            "@graph": [{"@id": "ex:Foo", "@type": "owl:Class"}],
        }
    }
    body = json.dumps(payload, indent=2)
    # The flat tail is `] } }` -- one array closer, then two object closers --
    # so the old wire's adjacent `] }` / `} ]` swap has nothing to swap here.
    # A single wrong kind is what remains possible, and must still be repaired.
    assert body.endswith("]\n  }\n}")
    broken = body[: -len("]\n  }\n}")] + "]\n  ]\n}"
    with pytest.raises(json.JSONDecodeError):
        json.loads(broken)

    sanitized = strip_trailing_commas(
        strip_json_comments(unescape_json_delimiters(broken))
    )
    report = GraphUpdateRenderReport.model_validate(
        parse_json_object(sanitized),
        context={"llm_graph_format": LLMGraphFormat.JSONLD},
    )

    queries = report.to_graph_update().generate_sparql_queries()
    assert len(queries) == 1
    assert "INSERT DATA" in queries[0]
    assert "Foo" in queries[0]
