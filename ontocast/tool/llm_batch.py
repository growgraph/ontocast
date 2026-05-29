"""Provider Batch API helpers for offline benchmark pre-warming.

OpenAI and Anthropic offer asynchronous batch endpoints (~50% lower cost,
multi-hour latency). This module supports exporting pending LLM prompts to
batch JSONL and importing completed results into the OntoCast disk cache so
subsequent server runs hit :class:`~ontocast.tool.llm.LLMTool` cache entries.

This is intended for validation / benchmark workflows, not interactive
``/process`` traffic.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ontocast.config import LLMConfig
from ontocast.tool.cache import Cacher, ToolCacher

logger = logging.getLogger(__name__)


def _llm_cache_config_dict(
    llm_config: LLMConfig,
) -> dict[str, str | int | float | bool]:
    """Build a cache config dict without optional None values."""
    cfg: dict[str, str | int | float | bool] = {
        "provider": llm_config.provider,
        "model_name": llm_config.model_name,
        "temperature": llm_config.temperature,
    }
    if llm_config.base_url is not None:
        cfg["base_url"] = llm_config.base_url
    return cfg


def write_openai_chat_batch_jsonl(
    requests: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write OpenAI Batch API input JSONL (one request object per line).

    Each item in ``requests`` should include:
    - ``custom_id``: stable id (e.g. cache key prefix)
    - ``body``: chat completions body with ``model``, ``messages``, etc.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in requests:
            line = {
                "custom_id": item["custom_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": item["body"],
            }
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
    logger.info("Wrote %s OpenAI batch request(s) to %s", len(requests), output_path)


def import_openai_batch_output_jsonl(
    output_path: Path,
    *,
    shared_cache: Cacher,
    llm_config: LLMConfig,
    custom_id_to_cache_key: dict[str, str],
) -> int:
    """Import OpenAI batch result JSONL lines into the LLM disk cache.

    Args:
        output_path: Path to the batch output JSONL from the provider.
        shared_cache: Shared :class:`Cacher` instance (same as the server uses).
        llm_config: LLM settings used when the batch was submitted.
        custom_id_to_cache_key: Maps each ``custom_id`` to the normalized prompt
            string used as cache content (see :meth:`LLMTool._cache_key_content`).

    Returns:
        Number of cache entries written.
    """
    tool_cache = ToolCacher(shared_cache, "llm")
    config_dict = _llm_cache_config_dict(llm_config)
    written = 0
    with output_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            custom_id = record.get("custom_id")
            if custom_id is None:
                continue
            cache_key = custom_id_to_cache_key.get(custom_id)
            if cache_key is None:
                logger.warning("No cache key mapping for custom_id=%s", custom_id)
                continue
            response = record.get("response", {})
            body = response.get("body", {})
            choices = body.get("choices", [])
            if not choices:
                logger.warning("Empty choices for custom_id=%s", custom_id)
                continue
            content = choices[0].get("message", {}).get("content", "")
            response_data = {
                "content": content,
                "prompt": cache_key[:200],
                "kwargs": {},
                "source": "openai_batch",
            }
            tool_cache.set(cache_key, response_data, config=config_dict)
            written += 1
    logger.info("Imported %s batch result(s) into LLM cache", written)
    return written
