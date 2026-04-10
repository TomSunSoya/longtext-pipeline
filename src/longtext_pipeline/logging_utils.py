"""Centralized logging configuration for longtext-pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


_TEXT_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_VALID_LOG_FORMATS = {"text", "json"}


class JsonFormatter(logging.Formatter):
    """Render log records as compact JSON for machine ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def _normalize_level(level_name: Any) -> int:
    """Convert a config-provided level name into a logging level constant."""
    if not isinstance(level_name, str):
        raise ValueError("logging.level must be a string.")

    normalized = level_name.upper()
    level = getattr(logging, normalized, None)
    if not isinstance(level, int):
        raise ValueError(
            "logging.level must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL."
        )
    return level


def _build_formatter(log_format: str) -> logging.Formatter:
    """Build a formatter for the requested output format."""
    if log_format == "json":
        return JsonFormatter(datefmt=_DATE_FORMAT)

    if log_format == "text":
        return logging.Formatter(_TEXT_LOG_FORMAT, datefmt=_DATE_FORMAT)

    raise ValueError("logging.format must be either 'text' or 'json'.")


def configure_logging(config: dict[str, Any] | None = None) -> None:
    """Configure root logging from runtime config."""
    logging_config = (config or {}).get("logging", {})
    if not isinstance(logging_config, dict):
        raise ValueError("logging config must be a dictionary.")

    level = _normalize_level(logging_config.get("level", "INFO"))
    log_format = logging_config.get("format", "text")
    if log_format not in _VALID_LOG_FORMATS:
        raise ValueError("logging.format must be either 'text' or 'json'.")

    formatter = _build_formatter(log_format)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for handler in list(root_logger.handlers):
        if getattr(handler, "_longtext_handler", False):
            root_logger.removeHandler(handler)
            handler.close()

    stream_handler = logging.StreamHandler()
    stream_handler._longtext_handler = True  # type: ignore[attr-defined]
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    log_file = logging_config.get("file")
    if log_file:
        log_path = Path(log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler._longtext_handler = True  # type: ignore[attr-defined]
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
