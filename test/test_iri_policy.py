from ontocast.onto.iri_policy import (
    is_in_namespace,
    join_namespace_local,
    normalize_namespace_iri,
    sanitize_prefix_map,
    split_namespace_local,
)


def test_normalize_namespace_iri_uses_context_hints() -> None:
    assert normalize_namespace_iri(
        "https://growgraph.dev/facts", context="auto"
    ).endswith("/")
    assert normalize_namespace_iri(
        "https://growgraph.dev/fcaont", context="ontology"
    ).endswith("#")
    assert normalize_namespace_iri(
        "https://growgraph.dev/legal_ontology", context="auto"
    ).endswith("#")


def test_join_and_split_namespace_local_round_trip() -> None:
    joined = join_namespace_local("https://growgraph.dev/facts", "imprisonment1")
    namespace, local = split_namespace_local(joined)
    assert namespace == "https://growgraph.dev/facts/"
    assert local == "imprisonment1"


def test_is_in_namespace_is_strict_on_boundary() -> None:
    assert is_in_namespace(
        "https://growgraph.dev/facts/Conviction1",
        "https://growgraph.dev/facts",
    )
    assert not is_in_namespace(
        "https://growgraph.dev/factsConviction1",
        "https://growgraph.dev/facts",
    )


def test_sanitize_prefix_map_normalizes_missing_delimiters() -> None:
    sanitized = sanitize_prefix_map(
        {"cd": "https://growgraph.dev/facts", "fcaont": "https://growgraph.dev/fcaont"},
        context="auto",
    )
    assert sanitized["cd"] == "https://growgraph.dev/facts/"
    assert sanitized["fcaont"] == "https://growgraph.dev/fcaont#"
