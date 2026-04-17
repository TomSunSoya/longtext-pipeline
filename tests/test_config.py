"""Tests for configuration loading and local auto-discovery."""

import pytest
import warnings

from src.longtext_pipeline.config import (
    AUTO_CONFIG_FILENAMES,
    ConfigError,
    find_auto_config_path,
    format_missing_settings_message,
    get_agent_model_config,
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
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

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


def test_agents_config_section_loads():
    """Multi-agent config with agents section should load correctly."""
    config, _ = load_runtime_config("examples/config.multi_agent.yaml")

    # Check agents section exists
    assert "agents" in config
    agents = config["agents"]

    # Check all agent types are present
    assert "summarizer" in agents
    assert "stage_synthesizer" in agents
    assert "analyst" in agents
    assert "auditor" in agents

    # Check agent has model config
    assert "model" in agents["summarizer"]
    assert agents["summarizer"]["model"]["name"] == "gpt-4o-mini"


def test_get_agent_model_config_returns_explicit_config():
    """get_agent_model_config should return agent-specific model config when present."""
    config, _ = load_runtime_config("examples/config.multi_agent.yaml")

    summarizer_config = get_agent_model_config(config, "summarizer")
    stage_synthesizer_config = get_agent_model_config(config, "stage_synthesizer")

    # Summarizer has explicit config
    assert summarizer_config["name"] == "gpt-4o-mini"
    assert summarizer_config["temperature"] == 0.7

    # Stage synthesizer has different explicit config
    assert stage_synthesizer_config["name"] == "gpt-4o"
    assert stage_synthesizer_config["temperature"] == 0.5


def test_get_agent_model_config_falls_back_to_top_level():
    """get_agent_model_config should fall back to top-level model when agent has no explicit config."""
    config, _ = load_runtime_config("examples/config.default.yaml")

    summarizer_config = get_agent_model_config(config, "summarizer")
    top_level_model = config.get("model", {})

    # Should be a deep copy of top-level model
    assert summarizer_config == top_level_model
    assert summarizer_config is not top_level_model  # Deep copy assertion


def test_get_agent_model_config_raises_for_unknown_agent():
    """get_agent_model_config should raise ConfigError for unknown agent type."""
    config = {"model": {"provider": "openai", "name": "gpt-4o-mini"}}

    with pytest.raises(ConfigError) as exc_info:
        get_agent_model_config(config, "unknown_agent")

    assert "unknown_agent" in str(exc_info.value)
    assert "summarizer" in str(exc_info.value)


def test_validate_config_warns_on_unknown_agent():
    """validate_config should warn about unknown agent types."""
    import warnings

    config = {
        "model": {"provider": "openai", "name": "gpt-4o-mini"},
        "agents": {
            "unknown_agent": {"model": {"name": "gpt-4o"}},
            "summarizer": {"model": {"name": "gpt-4o-mini"}},
        },
    }

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from src.longtext_pipeline.config import validate_config

        result = validate_config(config)

        assert result is True
        # Should have at least one warning for unknown_agent
        unknown_warnings = [
            warning for warning in w if "unknown_agent" in str(warning.message)
        ]
        assert len(unknown_warnings) >= 1


def test_validate_config_accepts_specialist_count_without_warning():
    """specialist_count should be a known pipeline config key."""
    import warnings

    config = {
        "model": {"provider": "openai", "name": "gpt-4o-mini"},
        "pipeline": {"max_workers": 4, "specialist_count": 2},
    }

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from src.longtext_pipeline.config import validate_config

        result = validate_config(config)

    assert result is True
    matching = [warning for warning in w if "specialist_count" in str(warning.message)]
    assert matching == []


# Timeout validation tests
def test_validate_config_accepts_valid_timeout_float():
    """timeout=120.0 (default) should pass validation."""
    import warnings

    config = {
        "model": {"provider": "openai", "name": "gpt-4o-mini", "timeout": 120.0},
    }

    with warnings.catch_warnings(record=True) as _:
        warnings.simplefilter("always")
        from src.longtext_pipeline.config import validate_config

        result = validate_config(config)

    assert result is True
    # Should not have any ConfigError


def test_validate_config_accepts_valid_timeout_int():
    """timeout as integer should pass validation."""
    import warnings

    config = {
        "model": {"provider": "openai", "name": "gpt-4o-mini", "timeout": 60},
    }

    with warnings.catch_warnings(record=True) as _:
        warnings.simplefilter("always")
        from src.longtext_pipeline.config import validate_config

        result = validate_config(config)

    assert result is True


def test_validate_config_rejects_invalid_timeout_type():
    """timeout must be int or float, not string."""
    config = {
        "model": {"provider": "openai", "name": "gpt-4o-mini", "timeout": "60"},
    }

    from src.longtext_pipeline.config import validate_config

    with pytest.raises(ConfigError) as exc_info:
        validate_config(config)

    assert "timeout" in str(exc_info.value).lower()


def test_validate_config_rejects_timeout_too_high():
    """timeout > 600.0 should raise ConfigError."""
    config = {
        "model": {"provider": "openai", "name": "gpt-4o-mini", "timeout": 999999},
    }

    from src.longtext_pipeline.config import validate_config

    with pytest.raises(ConfigError) as exc_info:
        validate_config(config)

    assert "timeout" in str(exc_info.value).lower()
    assert "600" in str(exc_info.value)


def test_validate_config_warnings_on_timeout_too_low():
    """timeout < 5.0 should log warning but NOT raise error."""
    config = {
        "model": {"provider": "openai", "name": "gpt-4o-mini", "timeout": 3.0},
    }

    from src.longtext_pipeline.config import validate_config

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = validate_config(config)

        # Should not raise error
        assert result is True
        # But should have a warning about timeout being too low
        timeout_warnings = [
            warning for warning in w if "timeout" in str(warning.message).lower()
        ]
        assert len(timeout_warnings) >= 1
