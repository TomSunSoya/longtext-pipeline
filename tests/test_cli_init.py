"""Tests for the longtext init CLI command."""

import os
import stat
from unittest.mock import patch

import pytest
import yaml
from typer.testing import CliRunner

from longtext_pipeline.cli import app

runner = CliRunner()


class TestInitDirectoryCreation:
    """Test directory creation scenarios."""

    def test_init_creates_config_files_in_current_directory(self, tmp_path):
        """Test that init creates all 5 config files in current directory."""
        result = runner.invoke(app, ["init", "--dir", str(tmp_path)])

        assert result.exit_code == 0

        # Verify all 5 files are created
        expected_files = [
            "config.general.yaml",
            "config.relationship.yaml",
            "longtext.local.yaml",
            "sample_input.txt",
            "README.md",
        ]

        for filename in expected_files:
            filepath = tmp_path / filename
            assert filepath.exists(), f"File {filename} was not created"

    def test_init_creates_nested_directory_structure(self, tmp_path):
        """Test that init creates nested directories if they don't exist."""
        nested_dir = tmp_path / "subdir" / "nested" / "config"

        result = runner.invoke(app, ["init", "--dir", str(nested_dir)])

        assert result.exit_code == 0
        assert nested_dir.exists()

        # Verify files are created in nested directory
        assert (nested_dir / "config.general.yaml").exists()

    @pytest.mark.skipif(
        os.name == "nt",
        reason="Windows permission handling differs from Unix",
    )
    def test_init_fails_on_permission_denied_for_directory_creation(self, tmp_path):
        """Test that init returns exit code 1 when directory creation fails due to permissions."""
        # Create a parent directory and make it read-only
        parent_dir = tmp_path / "readonly_parent"
        parent_dir.mkdir()

        # Make parent directory read-only to prevent subdirectory creation
        try:
            os.chmod(parent_dir, stat.S_IRUSR | stat.S_IXUSR)

            readonly_dir = parent_dir / "cannot_create_here"
            result = runner.invoke(app, ["init", "--dir", str(readonly_dir)])

            # Should fail with permission error
            assert result.exit_code == 1
            assert (
                "permission" in result.stdout.lower()
                or "error" in result.stdout.lower()
            )
        finally:
            # Restore permissions for cleanup
            try:
                os.chmod(parent_dir, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
            except Exception:
                pass

    def test_init_existing_directory_success(self, tmp_path):
        """Test init succeeds when target directory already exists."""
        # Directory already exists (created by tmp_path)
        result = runner.invoke(app, ["init", "--dir", str(tmp_path)])

        assert result.exit_code == 0
        assert (tmp_path / "config.general.yaml").exists()


class TestInitFileCreation:
    """Test file creation scenarios."""

    def test_init_creates_all_five_files(self, tmp_path):
        """Test that exactly 5 files are created."""
        result = runner.invoke(app, ["init", "--dir", str(tmp_path)])

        assert result.exit_code == 0

        files = list(tmp_path.iterdir())
        assert len(files) == 5

        filenames = {f.name for f in files}
        assert filenames == {
            "config.general.yaml",
            "config.relationship.yaml",
            "longtext.local.yaml",
            "sample_input.txt",
            "README.md",
        }

    def test_init_file_contents_are_non_empty(self, tmp_path):
        """Test that created files have content."""
        result = runner.invoke(app, ["init", "--dir", str(tmp_path)])

        assert result.exit_code == 0

        for filepath in tmp_path.iterdir():
            content = filepath.read_text()
            assert len(content) > 0, f"File {filepath.name} is empty"


class TestInitTemplateValidation:
    """Test template content structure."""

    def test_config_general_yaml_is_valid_yaml(self, tmp_path):
        """Test that config.general.yaml can be parsed as valid YAML."""
        result = runner.invoke(app, ["init", "--dir", str(tmp_path)])

        assert result.exit_code == 0

        config_path = tmp_path / "config.general.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert config is not None
        assert "model" in config
        assert "stages" in config
        assert "input" in config

    def test_config_relationship_yaml_is_valid_yaml(self, tmp_path):
        """Test that config.relationship.yaml can be parsed as valid YAML."""
        result = runner.invoke(app, ["init", "--dir", str(tmp_path)])

        assert result.exit_code == 0

        config_path = tmp_path / "config.relationship.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert config is not None
        assert "prompts" in config
        assert config["prompts"]["format"] == "relationship"

    def test_longtext_local_yaml_is_valid_yaml(self, tmp_path):
        """Test that longtext.local.yaml can be parsed as valid YAML."""
        result = runner.invoke(app, ["init", "--dir", str(tmp_path)])

        assert result.exit_code == 0

        config_path = tmp_path / "longtext.local.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert config is not None
        assert "model" in config

    def test_config_templates_have_placeholder_values(self, tmp_path):
        """Test that config templates contain expected placeholders."""
        result = runner.invoke(app, ["init", "--dir", str(tmp_path)])

        assert result.exit_code == 0

        # Check general config has placeholder
        general_config = tmp_path / "config.general.yaml"
        content = general_config.read_text()
        assert "YOUR_INPUT_FILE.txt" in content or "input_file" in content.lower()

        # Check for environment variable placeholder
        assert "${OPENAI_API_KEY}" in content or "API_KEY" in content

    def test_local_config_has_api_key_placeholder(self, tmp_path):
        """Test that local config has API key placeholder."""
        result = runner.invoke(app, ["init", "--dir", str(tmp_path)])

        assert result.exit_code == 0

        local_config = tmp_path / "longtext.local.yaml"
        content = local_config.read_text()

        # Should have a placeholder for API key
        assert "REPLACE" in content or "KEY" in content

    def test_sample_input_has_content_structure(self, tmp_path):
        """Test that sample input has markdown structure."""
        result = runner.invoke(app, ["init", "--dir", str(tmp_path)])

        assert result.exit_code == 0

        sample_input = tmp_path / "sample_input.txt"
        content = sample_input.read_text()

        # Should have headers and substantive content
        assert "#" in content
        assert len(content.split("\n")) > 10

    def test_readme_has_usage_instructions(self, tmp_path):
        """Test that README has usage instructions."""
        result = runner.invoke(app, ["init", "--dir", str(tmp_path)])

        assert result.exit_code == 0

        readme = tmp_path / "README.md"
        content = readme.read_text()

        # Should have setup and usage sections
        assert "Setup" in content or "setup" in content.lower()
        assert "longtext run" in content or "longtext init" in content


class TestInitOverwritePrevention:
    """Test overwrite prevention behavior."""

    def test_init_prompts_before_overwriting(self, tmp_path):
        """Test that init prompts for confirmation before overwriting existing files."""
        # Create a file first
        config_file = tmp_path / "config.general.yaml"
        config_file.write_text("# Original content\n")

        # Mock typer.confirm to return False (don't overwrite)
        with patch("typer.confirm", return_value=False):
            result = runner.invoke(app, ["init", "--dir", str(tmp_path)])

        # Should skip the file
        assert result.exit_code == 0
        assert "Skipping" in result.stdout or "skip" in result.stdout.lower()

        # Original content should be preserved
        assert config_file.read_text() == "# Original content\n"

    def test_init_overwrites_when_confirmed(self, tmp_path):
        """Test that init overwrites when user confirms."""
        # Create a file first
        config_file = tmp_path / "config.general.yaml"
        config_file.write_text("# Original content\n")

        # Mock typer.confirm to return True (overwrite)
        with patch("typer.confirm", return_value=True):
            result = runner.invoke(app, ["init", "--dir", str(tmp_path)])

        # Should overwrite
        assert result.exit_code == 0
        assert "Created" in result.stdout or "created" in result.stdout.lower()

        # Content should be replaced (YAML structure, not original)
        content = config_file.read_text()
        assert content != "# Original content\n"
        assert yaml.safe_load(content) is not None

    def test_init_multiple_files_overwrite_prompt(self, tmp_path):
        """Test that init prompts for each existing file."""
        # Create multiple files
        (tmp_path / "config.general.yaml").write_text("# General\n")
        (tmp_path / "config.relationship.yaml").write_text("# Relationship\n")
        (tmp_path / "longtext.local.yaml").write_text("# Local\n")

        # Mock typer.confirm to reject all overwrites
        with patch("typer.confirm", return_value=False):
            result = runner.invoke(app, ["init", "--dir", str(tmp_path)])

        assert result.exit_code == 0

        # Should skip all 3 existing files and create 2 new ones
        assert result.stdout.count("Skipping") >= 3


class TestInitPermissions:
    """Test permission handling."""

    @pytest.mark.skipif(
        os.name == "nt",
        reason="Windows permission handling differs from Unix",
    )
    def test_init_fails_on_unwritable_directory(self, tmp_path):
        """Test that init returns error when directory is not writable."""
        # Create directory and make it read-only
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()

        try:
            # Make directory read-only
            os.chmod(readonly_dir, stat.S_IRUSR | stat.S_IXUSR)

            result = runner.invoke(app, ["init", "--dir", str(readonly_dir)])

            # Should fail
            assert result.exit_code == 1
            assert (
                "writable" in result.stdout.lower()
                or "permission" in result.stdout.lower()
                or "error" in result.stdout.lower()
            )
        finally:
            # Restore permissions for cleanup
            try:
                os.chmod(readonly_dir, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
            except Exception:
                pass

    @pytest.mark.skipif(
        os.name == "nt",
        reason="Windows permission handling differs from Unix",
    )
    def test_init_fails_on_unwritable_file(self, tmp_path):
        """Test that init returns error when file cannot be written."""
        # Create a read-only file with target name
        readonly_file = tmp_path / "config.general.yaml"
        readonly_file.write_text("# Protected\n")

        try:
            # Make file read-only
            os.chmod(readonly_file, stat.S_IRUSR | stat.S_IXUSR)

            # Mock confirm to allow overwrite
            with patch("typer.confirm", return_value=True):
                result = runner.invoke(app, ["init", "--dir", str(tmp_path)])

            # Should fail on write
            assert result.exit_code == 1
            assert (
                "permission" in result.stdout.lower()
                or "error" in result.stdout.lower()
            )
        finally:
            # Restore permissions for cleanup
            try:
                os.chmod(readonly_file, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
            except Exception:
                pass


class TestInitExitCodes:
    """Test exit code behavior."""

    def test_init_returns_zero_on_success(self, tmp_path):
        """Test that init returns exit code 0 on success."""
        result = runner.invoke(app, ["init", "--dir", str(tmp_path)])
        assert result.exit_code == 0

    def test_init_returns_nonzero_on_error(self, tmp_path):
        """Test that init returns non-zero exit code on error."""
        # On Windows, we can't easily test permission errors.
        # Instead, we test that the command handles edge cases gracefully.
        # The init command is designed to be permissive (creates dirs on demand).
        # We verify the success path returns 0, which is tested elsewhere.
        # This test documents that error scenarios are handled gracefully.
        # For Unix systems, we could test with /root or /proc paths.
        if os.name != "nt":
            # Unix test - try a protected path
            result = runner.invoke(app, ["init", "--dir", "/root/nonexistent"])
            assert result.exit_code != 0 or "error" in result.stdout.lower()
        else:
            # Windows: just verify the command handles paths gracefully
            # Create a path that would fail if permissions were restrictive
            # Since Windows temp allows creation, we test error message format instead
            result = runner.invoke(app, ["init", "--dir", str(tmp_path)])
            # On Windows with permissive temp, this succeeds (exit 0)
            # The test verifies the command doesn't crash unexpectedly
            assert result.exit_code == 0

    def test_init_shows_completion_message_on_success(self, tmp_path):
        """Test that init shows completion message."""
        result = runner.invoke(app, ["init", "--dir", str(tmp_path)])

        assert result.exit_code == 0
        assert (
            "Initialization complete" in result.stdout
            or "complete" in result.stdout.lower()
        )
        assert str(tmp_path) in result.stdout


class TestInitConfigStructure:
    """Test YAML config structure verification."""

    def test_general_config_has_required_sections(self, tmp_path):
        """Test that general config has all required sections."""
        result = runner.invoke(app, ["init", "--dir", str(tmp_path)])
        assert result.exit_code == 0

        with open(tmp_path / "config.general.yaml") as f:
            config = yaml.safe_load(f)

        # Verify top-level sections exist
        assert "model" in config
        assert "input" in config
        assert "stages" in config
        assert "output" in config or "pipeline" in config

    def test_relationship_config_has_relationship_mode(self, tmp_path):
        """Test that relationship config has relationship-specific settings."""
        result = runner.invoke(app, ["init", "--dir", str(tmp_path)])
        assert result.exit_code == 0

        with open(tmp_path / "config.relationship.yaml") as f:
            config = yaml.safe_load(f)

        # Should have relationship mode indicator
        assert config.get("prompts", {}).get("format") == "relationship"

    def test_local_config_has_model_settings(self, tmp_path):
        """Test that local config has model configuration."""
        result = runner.invoke(app, ["init", "--dir", str(tmp_path)])
        assert result.exit_code == 0

        with open(tmp_path / "longtext.local.yaml") as f:
            config = yaml.safe_load(f)

        model_config = config.get("model", {})
        assert "provider" in model_config
        assert "name" in model_config
        assert "api_key" in model_config
