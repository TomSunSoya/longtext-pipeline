"""Tests for the AuditStage runtime semantics."""

from datetime import datetime
import logging
from unittest.mock import Mock, patch

import pytest

from src.longtext_pipeline.manifest import Manifest, ManifestManager
from src.longtext_pipeline.models import FinalAnalysis, StageInfo
from src.longtext_pipeline.pipeline.audit import AuditStage


@pytest.fixture
def sample_manifest() -> Manifest:
    """Create a sample manifest for testing."""
    stages = {
        "ingest": StageInfo(name="ingest", status="successful"),
        "summarize": StageInfo(name="summarize", status="successful"),
        "stage": StageInfo(name="stage", status="successful"),
        "final": StageInfo(name="final", status="successful"),
        "audit": StageInfo(name="audit", status="not_started"),
    }
    return Manifest(
        session_id="test_session",
        input_path="/tmp/input.txt",
        input_hash="abc123",
        stages=stages,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        status="running",
        total_parts=10,
        total_stages=2,
        estimated_tokens=5000,
    )


@pytest.fixture
def sample_config() -> dict:
    """Create a minimal config dictionary."""
    return {
        "model": {
            "provider": "openai",
            "name": "gpt-4o-mini",
        },
        "stages": {
            "audit": {
                "enabled": True,
            }
        },
    }


@pytest.fixture
def sample_final_analysis() -> FinalAnalysis:
    """Create a minimal final analysis object for active audit tests."""
    return FinalAnalysis(
        status="completed",
        stages=[],
        final_result=(
            "Alice led the migration in January 2024. "
            "Bob shipped the follow-up release in March 2024."
        ),
        metadata={},
    )


@pytest.fixture
def mock_manifest_manager():
    """Create a mock manifest manager."""
    manager = Mock(spec=ManifestManager)
    manager.update_stage = Mock()
    return manager


def _make_mock_audit_client() -> Mock:
    """Create a minimal audit client for active-run tests."""
    client = Mock()
    client.context_window = 32000
    client.complete.return_value = "Detailed audit report"
    client.complete_json.return_value = {"supported": True, "confidence": "high"}
    return client


class TestAuditStageInit:
    """Test AuditStage initialization."""

    def test_init_default_manifest_manager(self):
        stage = AuditStage()

        assert stage.manifest_manager is not None
        assert isinstance(stage.manifest_manager, ManifestManager)

    def test_init_custom_manifest_manager(self, mock_manifest_manager):
        stage = AuditStage(manifest_manager=mock_manifest_manager)

        assert stage.manifest_manager is mock_manifest_manager

    def test_init_falls_back_to_offline_client_when_llm_client_creation_fails(self):
        with patch(
            "src.longtext_pipeline.pipeline.audit.OpenAICompatibleClient",
            side_effect=RuntimeError("missing API key"),
        ):
            stage = AuditStage()

        assert stage.client is not None
        assert stage.client.context_window == 32000


class TestAuditStageRunWithoutAnalysis:
    """Test failure behavior when audit is invoked without final analysis."""

    def test_run_without_final_analysis_returns_failed_result(
        self, sample_manifest, sample_config
    ):
        stage = AuditStage()

        result = stage.run(
            analysis_objects=None,
            config=sample_config,
            manifest=sample_manifest,
            mode="general",
        )

        assert result["status"] == "failed"
        assert result["stage"] == "audit"
        assert result["mode"] == "general"
        assert result["checked_items"] == 0
        assert result["issues_found"] == 0
        assert result["confidence_score"] is None
        assert "final analysis" in result["message"].lower()
        assert "experimental" not in result["message"].lower()

    def test_run_without_final_analysis_updates_manifest_failed(
        self, sample_manifest, sample_config
    ):
        stage = AuditStage()

        stage.run(
            analysis_objects=None,
            config=sample_config,
            manifest=sample_manifest,
            mode="general",
        )

        assert sample_manifest.stages["audit"].status == "failed"
        assert sample_manifest.stages["audit"].error is not None
        assert "final analysis" in sample_manifest.stages["audit"].error.lower()
        assert sample_manifest.status == "failed"

    def test_run_without_final_analysis_logs_error_not_experimental(
        self, sample_manifest, sample_config, caplog
    ):
        stage = AuditStage()

        with caplog.at_level(logging.ERROR):
            stage.run(
                analysis_objects=None,
                config=sample_config,
                manifest=sample_manifest,
                mode="general",
            )

        assert "requires final analysis output" in caplog.text
        assert "experimental" not in caplog.text.lower()

    def test_run_without_final_analysis_uses_custom_manifest_manager(
        self, sample_manifest, sample_config, mock_manifest_manager
    ):
        stage = AuditStage(manifest_manager=mock_manifest_manager)

        stage.run(
            analysis_objects=None,
            config=sample_config,
            manifest=sample_manifest,
            mode="general",
        )

        mock_manifest_manager.update_stage.assert_called_once()
        call_args = mock_manifest_manager.update_stage.call_args
        assert call_args[0][0] is sample_manifest
        assert call_args[0][1] == "audit"
        assert call_args[0][2] == "failed"
        assert "final analysis" in call_args.kwargs["error"].lower()


class TestAuditStageRunActive:
    """Test active audit execution semantics."""

    def test_run_returns_active_audit_payload(
        self, sample_manifest, sample_config, sample_final_analysis
    ):
        stage = AuditStage(llm_client=_make_mock_audit_client())

        with patch(
            "src.longtext_pipeline.pipeline.audit.read_file",
            return_value=(
                "Alice led the migration in January 2024. "
                "Bob shipped the follow-up release in March 2024."
            ),
        ):
            result = stage.run(
                analysis_objects=sample_final_analysis,
                config=sample_config,
                manifest=sample_manifest,
                mode="general",
            )

        assert result["status"] in {
            "successful",
            "successful_with_warnings",
            "failed",
        }
        assert result["stage"] == "audit"
        assert "hallucination_detection" in result
        assert "timeline_verification" in result
        assert "quality_scoring" in result
        assert result["source_document_available"] is True
        assert sample_manifest.stages["audit"].status in {
            "successful",
            "successful_with_warnings",
            "failed",
        }

    def test_run_handles_missing_source_document_with_offline_heuristics(
        self, sample_manifest, sample_config, sample_final_analysis
    ):
        stage = AuditStage(llm_client=_make_mock_audit_client())

        with patch(
            "src.longtext_pipeline.pipeline.audit.read_file",
            side_effect=FileNotFoundError("missing source"),
        ):
            result = stage.run(
                analysis_objects=sample_final_analysis,
                config=sample_config,
                manifest=sample_manifest,
                mode="relationship",
            )

        assert result["mode"] == "relationship"
        assert result["source_document_available"] is False
        assert result["status"] in {
            "successful",
            "successful_with_warnings",
            "failed",
        }
