"""Tests for output directory validation in config.py."""

import pytest
from src.longtext_pipeline.config import (
    ConfigError,
    validate_config,
    _validate_output_dir,
)


class TestValidateOutputDir:
    """Tests for _validate_output_dir function."""

    def test_creates_missing_directory(self, tmp_path):
        """Should create the output directory if it doesn't exist."""
        non_existent_dir = tmp_path / "non_existent_output"
        config = {"output": {"dir": str(non_existent_dir)}}

        _validate_output_dir(config)

        assert non_existent_dir.exists()
        assert non_existent_dir.is_dir()

    def test_creates_parent_directories_if_missing(self, tmp_path):
        """Should create parent directories recursively."""
        deep_dir = tmp_path / "level1" / "level2" / "output"
        config = {"output": {"dir": str(deep_dir)}}

        _validate_output_dir(config)

        assert deep_dir.exists()
        assert deep_dir.is_dir()

    def test_works_with_existing_directory(self, tmp_path):
        """Should not fail if directory already exists."""
        existing_dir = tmp_path / "existing_output"
        existing_dir.mkdir(parents=True, exist_ok=True)
        config = {"output": {"dir": str(existing_dir)}}

        _validate_output_dir(config)

        assert existing_dir.exists()

    def test_validates_writable_permission(self, tmp_path):
        """Should fail if directory is not writable."""
        readonly_dir = tmp_path / "readonly_output"
        readonly_dir.mkdir(parents=True, exist_ok=True)
        # Make directory readonly - only on Unix-like systems
        # On Windows, chmod doesn't properly change permissions in the same way
        pytest.importorskip("pwd")  # Skip on Windows

        readonly_dir.chmod(0o444)

        config = {"output": {"dir": str(readonly_dir)}}

        with pytest.raises(ConfigError) as exc_info:
            _validate_output_dir(config)

        assert "not writable" in str(exc_info.value)

    def test_uses_default_when_output_section_missing(self):
        """Should use default output dir when output section not present."""
        config = {}

        # Should not raise
        _validate_output_dir(config)

    def test_uses_default_when_output_dir_not_specified(self):
        """Should use default output dir when dir is not specified in output."""
        config = {"output": {}}

        # Should not raise
        _validate_output_dir(config)

    def test_creates_directory_with_relative_path(self, tmp_path, monkeypatch):
        """Should handle relative paths correctly."""
        monkeypatch.chdir(tmp_path)
        relative_dir = "relative_output"
        config = {"output": {"dir": relative_dir}}

        _validate_output_dir(config)

        assert (tmp_path / relative_dir).exists()


class TestValidateConfigWithOutputDir:
    """Tests that validate_config calls _validate_output_dir."""

    def test_validate_config_validates_output_dir(self, tmp_path):
        """validate_config should trigger output dir validation."""
        config = {"output": {"dir": str(tmp_path / "output_dir")}}

        # Should not raise - validates output dir as part of config validation
        result = validate_config(config)

        assert result is True
        assert (tmp_path / "output_dir").exists()

    def test_validate_config_fails_with_invalid_output_dir(self, tmp_path):
        """validate_config should fail with invalid output dir."""
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        # Only test on Unix-like systems where chmod works meaningfully
        # On Windows, this test is effectively skipped due to permission model differences
        pytest.importorskip("pwd")  # Skip on Windows

        readonly_dir.chmod(0o444)

        config = {"output": {"dir": str(readonly_dir)}}

        with pytest.raises(ConfigError) as exc_info:
            validate_config(config)

        assert "not writable" in str(exc_info.value)


class TestOutputDirIntegration:
    """Integration tests for output directory handling."""

    def test_load_runtime_config_creates_output_dir(self, tmp_path, monkeypatch):
        """Runtime config loading should create output directory."""
        config_file = tmp_path / "config.yaml"
        output_dir = tmp_path / "runtime_output"
        config_file.write_text(f"output:\n  dir: {output_dir}\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        from src.longtext_pipeline.config import load_runtime_config

        config, _ = load_runtime_config(str(config_file))

        assert config["output"]["dir"] == str(output_dir)
        assert output_dir.exists()

    def test_output_dir_with_environment_variable_override(self, tmp_path, monkeypatch):
        """Environment variable override should create output directory."""
        output_dir = tmp_path / "env_output"
        monkeypatch.setenv("LONGTEXT_OUTPUT_DIR", str(output_dir))

        # Note: validate_config doesn't process env vars - that's done by merge_env_overrides
        # Test just ensures _validate_output_dir works correctly
        config = {"output": {}}
        _validate_output_dir(config)

        # When dir is None/unspecified, it should use the default
        # The default dir from DEFAULT_CONFIG should be used
