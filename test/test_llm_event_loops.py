"""The LLM tool survives being driven from several event loops.

The CLI initializes under one ``asyncio.run`` and processes under another; an
initialization step may already call the model (an ontology without a
description gets one written). A provider client's pooled connections belong
to the loop that opened them, so the tool must not carry them across loops.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ontocast.config import LLMConfig, LLMProvider
from ontocast.tool.cache import Cacher
from ontocast.tool.llm import LLMTool


class _ChatCompletions(BaseHTTPRequestHandler):
    """A keep-alive OpenAI-compatible endpoint that always answers ``ok``."""

    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers["Content-Length"]))
        body = json.dumps(
            {
                "id": "stub",
                "object": "chat.completion",
                "created": 0,
                "model": "stub",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


@pytest.fixture
def stub_base_url() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ChatCompletions)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()


def test_calls_from_successive_event_loops_reach_the_provider(
    stub_base_url: str,
) -> None:
    tool = LLMTool.create(
        config=LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="stub-model",
            base_url=stub_base_url,
            api_key="test-key",
            cache_enabled=False,
        ),
        cache=Cacher(),
    )
    first = asyncio.run(tool("first"))
    second = asyncio.run(tool("second"))
    assert (first.content, second.content) == ("ok", "ok")


def test_a_model_assigned_from_outside_is_kept_across_loops() -> None:
    class _Model:
        async def ainvoke(self, *args: object, **kwds: object) -> object:
            from langchain_core.messages.ai import AIMessage

            return AIMessage(content="fake")

    tool = LLMTool(config=LLMConfig(cache_enabled=False), cache=Cacher())
    model = _Model()
    tool._llm = model
    asyncio.run(tool("first"))
    asyncio.run(tool("second"))
    assert tool._llm is model
