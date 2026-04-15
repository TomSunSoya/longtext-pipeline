"""Tests for relationship mode stability - verifying no experimental warnings remain."""

from pathlib import Path


# Get the prompts directory
_prompts_dir = Path(__file__).parent.parent / "src" / "longtext_pipeline" / "prompts"


def test_summary_relationship_no_experimental_warning():
    """summary_relationship.txt should not contain experimental warnings."""
    prompt_path = _prompts_dir / "summary_relationship.txt"
    content = prompt_path.read_text(encoding="utf-8")

    # Check that experimental warnings are removed
    assert "[EXPERIMENTAL MODE:" not in content
    assert "experimental" not in content.lower()
    # Should still contain relationship-focused instructions
    assert "relationship" in content.lower()
    assert "entity" in content.lower()


def test_stage_relationship_no_experimental_warning():
    """stage_relationship.txt should not contain experimental warnings."""
    prompt_path = _prompts_dir / "stage_relationship.txt"
    content = prompt_path.read_text(encoding="utf-8")

    # Check that experimental warnings are removed
    assert "[EXPERIMENTAL MODE:" not in content
    assert "experimental" not in content.lower()
    # Should still contain relationship-focused instructions
    assert "relationship" in content.lower()
    assert "entity" in content.lower()


def test_final_relationship_no_experimental_warning():
    """final_relationship.txt should not contain experimental warnings."""
    prompt_path = _prompts_dir / "final_relationship.txt"
    content = prompt_path.read_text(encoding="utf-8")

    # Check that experimental warnings are removed
    assert "[EXPERIMENTAL MODE:" not in content
    assert "experimental" not in content.lower()
    # Should still contain relationship-focused instructions
    assert "relationship" in content.lower()
    assert "network" in content.lower()
    assert "entity" in content.lower()


def test_audit_relationship_no_experimental_warning():
    """audit_relationship.txt should not contain experimental warnings."""
    prompt_path = _prompts_dir / "audit_relationship.txt"
    content = prompt_path.read_text(encoding="utf-8")

    # Check that experimental warnings are removed
    assert "[EXPERIMENTAL MODE:" not in content
    assert "experimental" not in content.lower()
    # Should still contain relationship-focused audit instructions
    assert "relationship" in content.lower()
    assert "entity" in content.lower()


def test_cli_mode_help_text_no_experimental():
    """CLI help text for relationship mode should not mention experimental."""
    from typer.testing import CliRunner
    from src.longtext_pipeline.cli import app

    # Get the help text for the run command
    runner = CliRunner()
    result = runner.invoke(app, ["run", "--help"])

    # Verify relationship mode help exists
    assert result.exit_code == 0
    help_text = result.stdout
    assert "relationship" in help_text.lower()


def test_prompts_still_have_relationship_format():
    """All relationship prompts should still format for relationship mode output."""
    relationship_prompts = [
        "summary_relationship.txt",
        "stage_relationship.txt",
        "final_relationship.txt",
        "audit_relationship.txt",
    ]

    for prompt_name in relationship_prompts:
        prompt_path = _prompts_dir / prompt_name
        content = prompt_path.read_text(encoding="utf-8")

        # Each prompt should have clear relationship-focused instructions
        assert "relationship" in content.lower(), (
            f"{prompt_name} missing relationship focus"
        )


def test_cli_generated_config_template_relationship():
    """CLI-generated config template should include relationship mode config."""
    from src.longtext_pipeline.cli import generate_config_relationship_template

    config_yaml = generate_config_relationship_template()

    # Should contain relationship mode settings
    assert "relationship" in config_yaml.lower()
    assert "format" in config_yaml.lower()
    # Should not mention experimental
    assert "experimental" not in config_yaml.lower()


def test_relationship_mode_config_exists():
    """Relationship config should exist in examples directory."""
    config_path = Path(__file__).parent.parent / "examples" / "config.relationship.yaml"
    assert config_path.exists(), "config.relationship.yaml should exist"

    content = config_path.read_text(encoding="utf-8")
    # Should not mention experimental
    assert "experimental" not in content.lower()
