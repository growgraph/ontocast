import logging

from ontocast.config import Config, ServerConfig, ToolConfig
from ontocast.onto.enum import OntologyContextScope


def _cfg(scope, contract):
    tc = ToolConfig()
    tc.facts_validation.shapes_prompt_contract = contract
    return Config(server=ServerConfig(ontology_context_scope=scope), tool_config=tc)


def test_context_contract_with_document_scope_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="ontocast.config.settings"):
        _cfg(OntologyContextScope.DOCUMENT, "context")
    assert "defeats" in caplog.text or "share a prefix" in caplog.text


def test_full_contract_with_document_scope_is_silent(caplog):
    with caplog.at_level(logging.INFO, logger="ontocast.config.settings"):
        _cfg(OntologyContextScope.DOCUMENT, "full")
    assert caplog.text == ""


def test_context_contract_without_document_scope_is_silent(caplog):
    with caplog.at_level(logging.INFO, logger="ontocast.config.settings"):
        _cfg(OntologyContextScope.UNIT, "context")
    assert caplog.text == ""


def test_unit_path_manifest_omits_fanout_only_settings(tmp_path):
    """A single-unit run must not claim a fan-out prompt regime it never ran.

    ``ONTOLOGY_CONTEXT_SCOPE`` and ``FANOUT_WARMUP_UNITS`` are read only by the
    document fan-out node, so on the ``/process_unit`` path they are inert --
    and echoing them into the manifest made a dump assert a prefix-sharing
    setup that never applied.
    """
    import pathlib

    from ontocast.api.process_helpers import dump_run_manifest
    from ontocast.config import Config
    from ontocast.onto.enum import OntologyContextScope
    from test.test_cli_server import _batch_state as batch_state

    config = Config()
    config.server.ontology_context_scope = OntologyContextScope.DOCUMENT
    config.server.fanout_warmup_units = 1
    src = pathlib.Path("doc.txt")

    doc = dump_run_manifest(
        batch_state(), src, config=config, output_dir=tmp_path / "doc"
    )
    unit = dump_run_manifest(
        batch_state(),
        src,
        config=config,
        output_dir=tmp_path / "unit",
        fanout_settings_apply=False,
    )

    import json

    assert doc is not None and unit is not None
    doc_prompting = json.loads(doc.read_text())["prompting"]
    unit_prompting = json.loads(unit.read_text())["prompting"]

    assert doc_prompting["ontology_context_scope"] == "document"
    assert doc_prompting["fanout_warmup_units"] == 1
    # The dump excludes nulls, so the unit manifest simply does not carry
    # the claim -- which is the point: absent, not asserted.
    assert unit_prompting.get("ontology_context_scope") is None
    assert unit_prompting.get("fanout_warmup_units") is None
    # Settings that do apply on both paths must still be recorded.
    assert (
        unit_prompting["ontology_chapter_format"]
        == doc_prompting["ontology_chapter_format"]
    )
