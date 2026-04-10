"""Tests for runtime logging configuration."""

import json
import logging
from contextlib import contextmanager

import pytest

from src.longtext_pipeline.config import load_runtime_config
from src.longtext_pipeline.logging_utils import configure_logging


@contextmanager
def preserve_root_logger():
    """Restore the root logger after tests that reconfigure handlers."""
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level

    try:
        yield root_logger
    finally:
        current_handlers = list(root_logger.handlers)
        root_logger.handlers.clear()
        root_logger.setLevel(original_level)

        for handler in current_handlers:
            if handler not in original_handlers:
                handler.close()

        for handler in original_handlers:
            root_logger.addHandler(handler)


def test_load_runtime_config_applies_logging_env_overrides(monkeypatch, tmp_path):
    """Logging environment variables should override config defaults."""
    log_path = tmp_path / "runtime.log"
    monkeypatch.setenv("LONGTEXT_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LONGTEXT_LOG_FORMAT", "json")
    monkeypatch.setenv("LONGTEXT_LOG_FILE", str(log_path))

    config, _ = load_runtime_config()

    assert config["logging"]["level"] == "DEBUG"
    assert config["logging"]["format"] == "json"
    assert config["logging"]["file"] == str(log_path)


def test_configure_logging_writes_json_records_to_file(tmp_path):
    """Configured file logging should persist structured JSON records."""
    log_path = tmp_path / "pipeline.log"

    with preserve_root_logger():
        configure_logging(
            {
                "logging": {
                    "level": "INFO",
                    "format": "json",
                    "file": str(log_path),
                }
            }
        )

        logger = logging.getLogger("longtext_pipeline.test")
        logger.info("structured message")

        for handler in logging.getLogger().handlers:
            handler.flush()

    records = log_path.read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(records[-1])
    assert payload["level"] == "INFO"
    assert payload["logger"] == "longtext_pipeline.test"
    assert payload["message"] == "structured message"


def test_configure_logging_rejects_invalid_format():
    """Invalid logging formats should fail fast."""
    with preserve_root_logger():
        with pytest.raises(ValueError) as exc_info:
            configure_logging({"logging": {"format": "xml"}})

    assert "logging.format" in str(exc_info.value)
