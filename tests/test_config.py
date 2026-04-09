"""Tests for configuration loading and local auto-discovery."""

from pathlib import Path

from src.longtext_pipeline.config import (
    AUTO_CONFIG_FILENAMES,
    find_auto_config_path,
    format_missing_settings_message,
    get_missing_required_settings,
    load_runtime_config,
)


def test_find_auto_config_path_searches_parent_directories(tmp_path, monkeypatch):
    """Auto config should be found from the current directory or its parents."""
    config_path = tmp_path / AUTO_CONFIG_FILENAMES[0]
    config_path.write_text("model:\n  api_key: test-key\n", encoding="utf-8")

    nested_dir = tmp_path / "nested" / "child"
    nested_dir.mkdir(parents=True)
    monkeypatch.chdir(nested_dir)

    discovered = find_auto_config_path()

    assert discovered == config_path


def test_load_runtime_config_merges_explicit_and_local_config(tmp_path, monkeypatch):
    """Explicit pipeline config should merge with local machine config."""
    explicit_config = tmp_path / "config.general.yaml"
    explicit_config.write_text(
        "stages:\n  ingest:\n    chunk_size: 1234\nmodel:\n  name: gpt-4o-mini\n",
        encoding="utf-8",
    )

    local_config = tmp_path / AUTO_CONFIG_FILENAMES[0]
    local_config.write_text(
        (
            "model:\n"
            "  name: deepseek-chat\n"
            "  base_url: https://api.deepseek.com/v1\n"
            "  api_key: local-test-key\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    config, sources = load_runtime_config(str(explicit_config))

    assert config["stages"]["ingest"]["chunk_size"] == 1234
    assert config["model"]["name"] == "deepseek-chat"
    assert config["model"]["base_url"] == "https://api.deepseek.com/v1"
    assert config["model"]["api_key"] == "local-test-key"
    assert str(explicit_config.resolve()) in sources
    assert str(local_config.resolve()) in sources


def test_get_missing_required_settings_reports_api_key():
    """API key is required at runtime and should be reported when missing."""
    config = {
        "model": {
            "provider": "openai",
            "name": "gpt-4o-mini",
            "api_key": "",
        }
    }

    assert get_missing_required_settings(config) == ["model.api_key"]


def test_format_missing_settings_message_mentions_local_config():
    """Missing-setting message should direct users to the auto-loaded local config."""
    message = format_missing_settings_message(["model.api_key"])

    assert "model.api_key" in message
    assert "longtext.local.yaml" in message
    assert "OPENAI_API_KEY" in message
