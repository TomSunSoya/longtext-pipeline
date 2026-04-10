import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import AsyncMock, patch

import pytest

from src.longtext_pipeline.errors import LLMResponseError, LLMTimeoutError
from src.longtext_pipeline.llm.openai_compatible import OpenAICompatibleClient


def _completion_response(content: str) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": content,
                }
            }
        ]
    }


class _LocalMockServer:
    def __init__(self, scenarios: list[dict]):
        self._scenarios = deque(scenarios)
        self.requests = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                owner._handle_request(self)

            def log_message(self, format, *args):
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)

    def _handle_request(self, handler: BaseHTTPRequestHandler) -> None:
        content_length = int(handler.headers.get("Content-Length", "0"))
        body = handler.rfile.read(content_length)
        scenario = (
            self._scenarios.popleft()
            if self._scenarios
            else {"status": 200, "json": _completion_response("ok")}
        )

        try:
            parsed_body = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            parsed_body = body.decode("utf-8")

        self.requests.append(
            {
                "path": handler.path,
                "body": parsed_body,
            }
        )

        delay = scenario.get("delay", 0)
        if delay:
            time.sleep(delay)

        payload = scenario.get("body")
        if payload is None:
            payload = json.dumps(scenario.get("json", {})).encode("utf-8")
        elif isinstance(payload, str):
            payload = payload.encode("utf-8")

        handler.send_response(scenario.get("status", 200))
        handler.send_header(
            "Content-Type", scenario.get("content_type", "application/json")
        )
        handler.end_headers()

        try:
            handler.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass


def test_complete_uses_local_mock_server_sync_path():
    with _LocalMockServer(
        [{"status": 200, "json": _completion_response("sync ok")}]
    ) as server:
        client = OpenAICompatibleClient(
            api_key="test-key", base_url=server.base_url, timeout=0.5
        )

        result = client.complete("hello sync")

    assert result == "sync ok"
    assert server.requests[0]["path"] == "/v1/chat/completions"
    assert server.requests[0]["body"]["messages"][-1]["content"] == "hello sync"


@pytest.mark.asyncio
async def test_acomplete_uses_local_mock_server_async_path():
    with _LocalMockServer(
        [{"status": 200, "json": _completion_response("async ok")}]
    ) as server:
        client = OpenAICompatibleClient(
            api_key="test-key", base_url=server.base_url, timeout=0.5
        )

        result = await client.acomplete("hello async", system_prompt="system")

    assert result == "async ok"
    assert server.requests[0]["path"] == "/v1/chat/completions"
    assert server.requests[0]["body"]["messages"][0]["content"] == "system"


def test_complete_retries_after_429_from_local_server():
    scenarios = [
        {"status": 429, "json": {"error": {"message": "slow down"}}},
        {"status": 200, "json": _completion_response("recovered")},
    ]

    with patch("src.longtext_pipeline.utils.retry.time.sleep", return_value=None):
        with _LocalMockServer(scenarios) as server:
            client = OpenAICompatibleClient(
                api_key="test-key", base_url=server.base_url, timeout=0.5
            )

            result = client.complete("retry me")

    assert result == "recovered"
    assert len(server.requests) == 2


@pytest.mark.asyncio
async def test_acomplete_retries_after_500_from_local_server():
    scenarios = [
        {"status": 500, "json": {"error": {"message": "upstream failure"}}},
        {"status": 200, "json": _completion_response("recovered async")},
    ]

    with patch(
        "src.longtext_pipeline.utils.retry.asyncio.sleep",
        new=AsyncMock(return_value=None),
    ):
        with _LocalMockServer(scenarios) as server:
            client = OpenAICompatibleClient(
                api_key="test-key", base_url=server.base_url, timeout=0.5
            )

            result = await client.acomplete("retry async")

    assert result == "recovered async"
    assert len(server.requests) == 2


def test_make_request_times_out_against_local_server():
    with _LocalMockServer(
        [{"status": 200, "json": _completion_response("too slow"), "delay": 0.2}]
    ) as server:
        client = OpenAICompatibleClient(
            api_key="test-key", base_url=server.base_url, timeout=0.05
        )

        with pytest.raises(LLMTimeoutError):
            client._make_request(client._build_payload("timeout"))


def test_make_request_rejects_bad_json_from_local_server():
    with _LocalMockServer([{"status": 200, "body": "{not-json"}]) as server:
        client = OpenAICompatibleClient(
            api_key="test-key", base_url=server.base_url, timeout=0.5
        )

        with pytest.raises(LLMResponseError):
            client._make_request(client._build_payload("bad json"))
