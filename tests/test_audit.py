"""Tests for the AuditStage class."""

import pytest
import logging
from unittest.mock import Mock
from datetime import datetime

from src.longtext_pipeline.pipeline.audit import AuditStage
from src.longtext_pipeline.manifest import ManifestManager, Manifest
from src.longtext_pipeline.models import StageInfo


@pytest.fixture
def sample_manifest():
    """Create a sample manifest for testing."""
    stages = {
        "ingest": StageInfo(name="ingest", status="successful"),
        "summarize": StageInfo(name="summarize", status="successful"),
        "stage": StageInfo(name="stage", status="successful"),
        "final": StageInfo(name="final", status="successful"),
        "audit": StageInfo(name="audit", status="not_started"),
    }
    manifest = Manifest(
        session_id="test_session",
        input_path="/path/to/input.txt",
        input_hash="abc123",
        stages=stages,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        status="running",
        total_parts=10,
        total_stages=2,
        estimated_tokens=5000,
    )
    return manifest


@pytest.fixture
def sample_config():
    """Create a sample config dictionary."""
    return {
        "model": {
            "provider": "openai",
            "name": "gpt-4",
            "temperature": 0.7,
        },
        "stages": {
            "audit": {
                "enabled": True,
            }
        },
    }


@pytest.fixture
def mock_manifest_manager():
    """Create a mock manifest manager."""
    manager = Mock(spec=ManifestManager)
    manager.update_stage = Mock()
    return manager


class TestAuditStageInit:
    """Test AuditStage initialization."""

    def test_init_default_manifest_manager(self):
        """Test that AuditStage creates default ManifestManager when none provided."""
        stage = AuditStage()

        assert stage.manifest_manager is not None
        assert isinstance(stage.manifest_manager, ManifestManager)

    def test_init_custom_manifest_manager(self, mock_manifest_manager):
        """Test that AuditStage uses provided ManifestManager."""
        stage = AuditStage(manifest_manager=mock_manifest_manager)

        assert stage.manifest_manager is mock_manifest_manager


class TestAuditStageRun:
    """Test AuditStage.run method."""

    def test_run_returns_skipped_status(self, sample_manifest, sample_config):
        """Test that run returns status='skipped'."""
        stage = AuditStage()
        result = stage.run(
            analysis_objects=None,
            config=sample_config,
            manifest=sample_manifest,
            mode="general",
        )

        assert result["status"] == "skipped"

    def test_run_returns_correct_structure(self, sample_manifest, sample_config):
        """Test that run returns result with all expected keys."""
        stage = AuditStage()
        result = stage.run(
            analysis_objects=None,
            config=sample_config,
            manifest=sample_manifest,
            mode="general",
        )

        # Check all expected keys are present
        assert "status" in result
        assert "stage" in result
        assert "mode" in result
        assert "message" in result
        assert "checked_items" in result
        assert "issues_found" in result
        assert "confidence_score" in result
        assert "audited_files" in result
        assert "recommendations" in result

        # Check expected values
        assert result["stage"] == "audit"
        assert result["checked_items"] == 0
        assert result["issues_found"] == 0
        assert result["confidence_score"] is None
        assert result["audited_files"] == []
        assert result["recommendations"] == []

    def test_run_general_mode(self, sample_manifest, sample_config):
        """Test that run correctly handles general mode."""
        stage = AuditStage()
        result = stage.run(
            analysis_objects=None,
            config=sample_config,
            manifest=sample_manifest,
            mode="general",
        )

        assert result["mode"] == "general"
        assert "general" in result["message"].lower() or result["status"] == "skipped"

    def test_run_relationship_mode(self, sample_manifest, sample_config):
        """Test that run correctly handles relationship mode."""
        stage = AuditStage()
        result = stage.run(
            analysis_objects=None,
            config=sample_config,
            manifest=sample_manifest,
            mode="relationship",
        )

        assert result["mode"] == "relationship"

    def test_run_updates_manifest(self, sample_manifest, sample_config):
        """Test that run updates manifest with skipped status."""
        stage = AuditStage()
        stage.run(
            analysis_objects=None,
            config=sample_config,
            manifest=sample_manifest,
            mode="general",
        )

        # Check that manifest audit stage was updated
        assert "audit" in sample_manifest.stages
        assert sample_manifest.stages["audit"].status == "skipped"
        assert (
            sample_manifest.stages["audit"].error
            == "Audit functionality deferred to v2 - placeholder only"
        )

    def test_run_logs_warnings(self, sample_manifest, sample_config, caplog):
        """Test that run logs appropriate warnings."""
        stage = AuditStage()

        with caplog.at_level(logging.WARNING):
            stage.run(
                analysis_objects=None,
                config=sample_config,
                manifest=sample_manifest,
                mode="general",
            )

        # Check that warnings were logged
        assert "Audit functionality is experimental in MVP" in caplog.text
        assert "Skipping audit" in caplog.text
        assert "Analysis quality checks were not performed" in caplog.text

    def test_run_custom_manifest_manager(
        self, sample_manifest, sample_config, mock_manifest_manager
    ):
        """Test that run uses custom manifest manager."""
        stage = AuditStage(manifest_manager=mock_manifest_manager)
        stage.run(
            analysis_objects=None,
            config=sample_config,
            manifest=sample_manifest,
            mode="general",
        )

        # Verify update_stage was called on the custom manager
        mock_manifest_manager.update_stage.assert_called_once()

        # Check the call arguments
        call_args = mock_manifest_manager.update_stage.call_args
        assert call_args[0][0] is sample_manifest
        assert call_args[0][1] == "audit"
        assert call_args[0][2] == "skipped"

    def test_run_with_none_analysis_objects(self, sample_manifest, sample_config):
        """Test that run handles None analysis_objects gracefully."""
        stage = AuditStage()
        result = stage.run(
            analysis_objects=None,
            config=sample_config,
            manifest=sample_manifest,
            mode="general",
        )

        # Should still return placeholder result
        assert result["status"] == "skipped"
        assert result["stage"] == "audit"

    def test_run_with_empty_config(self, sample_manifest):
        """Test that run handles empty config gracefully."""
        stage = AuditStage()
        result = stage.run(
            analysis_objects=None,
            config={},
            manifest=sample_manifest,
            mode="general",
        )

        # Should still return placeholder result
        assert result["status"] == "skipped"

    def test_run_message_content(self, sample_manifest, sample_config):
        """Test that the result message explains the placeholder nature."""
        stage = AuditStage()
        result = stage.run(
            analysis_objects=None,
            config=sample_config,
            manifest=sample_manifest,
            mode="general",
        )

        assert "experimental" in result["message"].lower()
        assert "MVP" in result["message"]

    def test_run_preserves_manifest_status(self, sample_manifest, sample_config):
        """Test that run updates manifest overall status."""
        stage = AuditStage()
        stage.run(
            analysis_objects=None,
            config=sample_config,
            manifest=sample_manifest,
            mode="general",
        )

        # Manifest status should be updated to 'skipped'
        assert sample_manifest.status == "skipped"

    def test_run_error_message_in_manifest(self, sample_manifest, sample_config):
        """Test that error message is stored in manifest."""
        stage = AuditStage()
        stage.run(
            analysis_objects=None,
            config=sample_config,
            manifest=sample_manifest,
            mode="general",
        )

        audit_stage_info = sample_manifest.stages["audit"]
        assert audit_stage_info.error is not None
        assert "deferred to v2" in audit_stage_info.error


class TestAuditStagePlaceholderBehavior:
    """Test that AuditStage behaves correctly as a placeholder."""

    def test_no_actual_audit_logic(self, sample_manifest, sample_config, caplog):
        """Test that no actual audit logic is performed."""
        stage = AuditStage()

        with caplog.at_level(logging.WARNING):
            result = stage.run(
                analysis_objects=None,
                config=sample_config,
                manifest=sample_manifest,
                mode="general",
            )

        # Verify no actual checking happened
        assert result["checked_items"] == 0
        assert result["issues_found"] == 0
        assert result["confidence_score"] is None

    def test_placeholder_result_is_consistent(self, sample_manifest, sample_config):
        """Test that placeholder results are consistent across calls."""
        stage = AuditStage()

        result1 = stage.run(
            analysis_objects=None,
            config=sample_config,
            manifest=sample_manifest,
            mode="general",
        )
        result2 = stage.run(
            analysis_objects=None,
            config=sample_config,
            manifest=sample_manifest,
            mode="general",
        )

        # Results should be identical (except for potential timestamp differences)
        assert result1["status"] == result2["status"]
        assert result1["stage"] == result2["stage"]
        assert result1["checked_items"] == result2["checked_items"]
        assert result1["issues_found"] == result2["issues_found"]

    def test_different_modes_all_skipped(self, sample_manifest, sample_config):
        """Test that all modes return skipped status."""
        stage = AuditStage()

        for mode in ["general", "relationship", "custom"]:
            result = stage.run(
                analysis_objects=None,
                config=sample_config,
                manifest=sample_manifest,
                mode=mode,
            )
            assert result["status"] == "skipped"
            assert result["mode"] == mode
