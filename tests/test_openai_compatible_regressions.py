"""Regression tests for OpenAI-compatible client and streaming helpers."""

import warnings

import pytest

from src.longtext_pipeline.llm.openai_compatible import (
    OpenAICompatibleClient,
    print_final_streaming_stats as client_print_final_streaming_stats,
)
from src.longtext_pipeline.llm.progress import (
    create_token_progress_callback,
    default_progress_callback,
    print_final_streaming_stats as shared_print_final_streaming_stats,
)


def test_progress_helpers_use_shared_implementations():
    """Progress helpers should resolve to the same shared functions."""
    assert create_token_progress_callback() is default_progress_callback
    assert OpenAICompatibleClient.default_progress_callback is default_progress_callback
    assert client_print_final_streaming_stats is shared_print_final_streaming_stats


def test_build_payload_does_not_warn_when_system_prompt_is_none_and_unchanged():
    """A missing system prompt should not look like a truncation event."""
    client = OpenAICompatibleClient(
        model="test-model",
        api_key="test-key",
        context_window=4096,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        payload = client._build_payload("Short prompt", system_prompt=None)

    truncation_warnings = [
        warning
        for warning in caught
        if "Content was truncated to fit context window" in str(warning.message)
    ]

    assert payload["messages"] == [{"role": "user", "content": "Short prompt"}]
    assert truncation_warnings == []


@pytest.mark.asyncio
async def test_complete_stream_decodes_error_response_before_handling():
    """Streaming errors should decode the response body before passing it on."""
    client = OpenAICompatibleClient(model="test-model", api_key="test-key")

    class SentinelError(Exception):
        """Stop the coroutine once the captured assertion is made."""

    class FakeResponse:
        status_code = 500

        async def aread(self) -> bytes:
            return '{"error":"stream failed"}'.encode("utf-8")

        async def aiter_lines(self):
            if False:
                yield ""

    class FakeStreamContext:
        async def __aenter__(self):
            return FakeResponse()

        async def __aexit__(self, exc_type, exc, tb):
            return None

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        def stream(self, *args, **kwargs):
            return FakeStreamContext()

    captured: list[tuple[int, str]] = []

    def capture_error(status_code: int, response_text: str) -> None:
        captured.append((status_code, response_text))
        raise SentinelError()

    client._handle_error = capture_error  # type: ignore[method-assign]

    with pytest.raises(SentinelError):
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                "src.longtext_pipeline.llm.openai_compatible.httpx.AsyncClient",
                FakeAsyncClient,
            )
            await OpenAICompatibleClient.complete_stream.__wrapped__(
                client, prompt="test prompt"
            )

    assert captured == [(500, '{"error":"stream failed"}')]
