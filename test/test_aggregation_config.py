from ontocast.config import AggregationConfig, Config


def test_aggregation_config_defaults() -> None:
    config = AggregationConfig()
    assert config.embedding_model == "paraphrase-multilingual-MiniLM-L12-v2"
    assert config.similarity_threshold == 0.80


def test_aggregation_config_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("AGG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    monkeypatch.setenv("AGG_SIMILARITY_THRESHOLD", "0.73")

    config = Config()
    assert config.tool_config.aggregation.embedding_model == "all-MiniLM-L6-v2"
    assert config.tool_config.aggregation.similarity_threshold == 0.73
