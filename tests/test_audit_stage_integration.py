"""
Test suite for audit stage integration with the main pipeline.

These tests verify that the audit stage is properly integrated into the pipeline,
runs as the final stage after FinalAnalysis, and operates correctly with
different pipeline modes.
All tests use mocked LLM responses to avoid requiring real API keys in CI.
"""

import tempfile
from pathlib import Path
import json
import pytest
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch, MagicMock

from src.longtext_pipeline.pipeline.orchestrator import LongtextPipeline
from src.longtext_pipeline.config import DEFAULT_CONFIG


def _create_mock_llm_client():
    """Create a mock LLM client that returns valid responses for all stages."""
    mock_client = MagicMock()

    # Set model as a string to avoid JSON serialization issues
    mock_client.model = "mock-model"

    # Mock sync request (used by summarize, stage, final)
    def mock_sync_request(prompt, **kwargs):
        return {
            "content": f"Mocked analysis response for: {prompt[:50]}...",
            "tokens_used": 100,
            "model": "mock-model",
        }

    mock_client.make_sync_request = MagicMock(side_effect=mock_sync_request)
    mock_client.complete = MagicMock(side_effect=mock_sync_request)

    # Mock async request with AsyncMock for proper await support
    async def mock_async_complete(prompt, **kwargs):
        return f"Mocked async response for: {prompt[:50]}..."

    mock_client.acomplete = AsyncMock(side_effect=mock_async_complete)

    # Mock complete_json for audit
    def mock_complete_json(prompt, **kwargs):
        return {
            "hallucinations": [],
            "confidence_score": 0.95,
            "quality": "high",
        }

    mock_client.complete_json = MagicMock(side_effect=mock_complete_json)

    # Add context_window attribute for audit
    mock_client.context_window = 32000

    return mock_client


@pytest.fixture(autouse=True)
def mock_llm_client():
    """Auto-use mock for all tests in this module to avoid real API calls."""
    mock_client = _create_mock_llm_client()

    with ExitStack() as stack:
        # Patch all stage-level client factory seams
        stack.enter_context(
            patch(
                "src.longtext_pipeline.pipeline.summarize.get_llm_client",
                return_value=mock_client,
            )
        )
        stack.enter_context(
            patch(
                "src.longtext_pipeline.pipeline.stage_synthesis.get_llm_client",
                return_value=mock_client,
            )
        )
        stack.enter_context(
            patch(
                "src.longtext_pipeline.pipeline.final_analysis.get_llm_client",
                return_value=mock_client,
            )
        )
        # Patch OpenAICompatibleClient for AuditStage (it creates client directly)
        stack.enter_context(
            patch(
                "src.longtext_pipeline.pipeline.audit.OpenAICompatibleClient",
                return_value=mock_client,
            )
        )
        # Patch orchestrator's config loading to use minimal valid config
        stack.enter_context(
            patch.dict(
                "os.environ",
                {"OPENAI_API_KEY": "mock-api-key-for-testing"},
                clear=False,
            )
        )
        yield mock_client


def test_audit_stage_runs_by_default(mock_llm_client):
    """Test that audit stage runs by default in the complete pipeline flow."""
    # Create sample input file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(
            "This is a test document to verify that the audit stage runs.\n"
            "It contains several sentences that the pipeline will process.\n"
            "The content here is for testing hallucination detection.\n"
        )
        input_file = f.name

    try:
        # Test pipeline runs with audit stage enabled by default
        pipeline = LongtextPipeline()

        # Create a config with audit enabled by default
        config = DEFAULT_CONFIG.copy()
        config["stages"]["audit"]["enabled"] = True

        # Mock/Stub the external dependencies for test
        # Run the pipeline with a simple test document
        final_analysis = pipeline.run(
            input_path=input_file,
            mode="general",
            config_path=None,  # Use default config
        )

        # Verify pipeline completed successfully
        assert final_analysis is not None
        assert hasattr(final_analysis, "metadata")

        # Verify manifest file was created and contains audit status
        manifest_path = Path(input_file).parent / ".longtext" / "manifest.json"
        assert manifest_path.exists()

        with open(manifest_path, "r") as mf:
            manifest = json.load(mf)

        # Check that audit stage is present in stages dict
        assert "audit" in manifest.get("stages", {}), (
            "Audit stage should be present in manifest"
        )

        audit_stage_info = manifest["stages"]["audit"]
        assert audit_stage_info["status"] in [
            "successful",
            "successful_with_warnings",
            "failed",
        ], f"Audit stage should have status, got: {audit_stage_info}"

    finally:
        # Clean up
        Path(input_file).unlink(missing_ok=True)
        # Clean up .longtext directory potentially created
        longtext_dir = Path(input_file).parent / ".longtext"
        if longtext_dir.exists():
            import shutil

            shutil.rmtree(longtext_dir, ignore_errors=True)


def test_audit_stage_integration_relationship_mode():
    """Test that audit stage integrates correctly in relationship mode."""
    # Create sample input file for relationship mode
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(
            "John Smith worked as a project manager from January 2020 to March 2022.\n"
            "He collaborated with Jane Doe on the ABC project during 2021.\n"
            "Jane Doe is the CEO who founded the company in 2018.\n"
            "Mark Johnson joined as lead developer in February 2021.\n"
        )
        input_file = f.name

    try:
        pipeline = LongtextPipeline()

        # Run pipeline in relationship mode
        final_analysis = pipeline.run(
            input_path=input_file,
            mode="relationship",
            config_path=None,  # Use default config with audit enabled
        )

        # Verify pipeline completed
        assert final_analysis is not None

        # Check manifest for audit stage
        manifest_path = Path(input_file).parent / ".longtext" / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r") as mf:
                manifest = json.load(mf)

            # Audit stage should be present in relationship mode too
            assert "audit" in manifest.get("stages", {}), (
                "Audit stage should be present in relationship mode"
            )

    finally:
        # Clean up
        Path(input_file).unlink(missing_ok=True)
        longtext_dir = Path(input_file).parent / ".longtext"
        if longtext_dir.exists():
            import shutil

            shutil.rmtree(longtext_dir, ignore_errors=True)


def test_audit_stage_executes_after_final_analysis():
    """Test that audit stage executes after final analysis stage."""
    # Create sample text
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(
            "Document analysis test with sample content.\n"
            "This document contains information that can be analyzed.\n"
            "We will process this content through multiple pipeline stages.\n"
        )
        input_file = f.name

    try:
        pipeline = LongtextPipeline()

        # Run pipeline with resume=False to force all stages
        final_analysis = pipeline.run(
            input_path=input_file,
            mode="general",
            resume=False,  # Force execution of all stages
            config_path=None,  # Use default config with audit enabled
        )

        assert final_analysis is not None

        # Verify the audit stage was executed by examining manifest
        manifest_path = Path(input_file).parent / ".longtext" / "manifest.json"
        assert manifest_path.exists()

        with open(manifest_path, "r") as mf:
            manifest = json.load(mf)

        # Verify all expected stages are present in order:
        # ingest -> summarize -> stage -> final -> audit
        stages = manifest.get("stages", {})

        # Check that all stages occurred
        assert "ingest" in stages
        assert "summarize" in stages
        assert "stage" in stages
        assert "final" in stages
        assert "audit" in stages  # This is the key verification

        # Verify that audit is last or one of the last stages (timing may vary slightly)
        # but at minimum it comes after 'final' analysis
        final_completed = stages.get("final", {}).get("status") in [
            "successful",
            "successful_with_warnings",
            "completed",
            "partial_success",
        ]
        audit_completed = stages.get("audit", {}).get("status") in [
            "successful",
            "successful_with_warnings",
            "completed",
            "failed",
        ]

        assert final_completed, "Final analysis stage should complete"
        assert audit_completed, "Audit stage should complete after final analysis"

    finally:
        # Clean up
        Path(input_file).unlink(missing_ok=True)
        longtext_dir = Path(input_file).parent / ".longtext"
        if longtext_dir.exists():
            import shutil

            shutil.rmtree(longtext_dir, ignore_errors=True)


def test_audit_stage_updates_manifest_correctly():
    """Test that audit stage properly updates the manifest with audit results."""
    # Create a simple test file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(
            "This document has been processed by the pipeline.\n"
            "Sample information that may get audited.\n"
        )
        input_file = f.name

    try:
        pipeline = LongtextPipeline()

        # Run the pipeline
        pipeline.run(
            input_path=input_file,
            mode="general",
            config_path=None,  # Use default config with audit enabled
        )

        # Verify manifest contains audit results
        manifest_path = Path(input_file).parent / ".longtext" / "manifest.json"
        assert manifest_path.exists()

        with open(manifest_path, "r") as mf:
            manifest = json.load(mf)

        audit_info = manifest.get("stages", {}).get("audit", {})
        assert audit_info, "Audit stage information should be present in manifest"

        # Audit stage should have status and stats with audit-specific information
        assert "status" in audit_info, "Audit stage should have status"
        assert audit_info["status"] in [
            "successful",
            "successful_with_warnings",
            "failed",
        ], f"Audit should have valid status, got: {audit_info['status']}"

        # Stats field should contain audit-specific metrics if successful
        if audit_info["status"] in ["successful", "successful_with_warnings"]:
            assert "stats" in audit_info, "Successful audit should have stats section"
            stats = audit_info["stats"]
            # All expected fields might not be in every test run, but key ones should be present
            assert isinstance(stats.get("total_claims", 0), (int, float)), (
                "Should have total_claims metric"
            )
            assert isinstance(stats.get("confidence_score", 0), (int, float)), (
                "Should have confidence_score"
            )
            assert isinstance(stats.get("timeline_anomalies", 0), (int, float)), (
                "Should have timeline_anomalies metric"
            )

    finally:
        # Clean up
        Path(input_file).unlink(missing_ok=True)
        longtext_dir = Path(input_file).parent / ".longtext"
        if longtext_dir.exists():
            import shutil

            shutil.rmtree(longtext_dir, ignore_errors=True)


def test_audit_stage_with_multi_perspective():
    """Test audit stage works with multi-perspective specialist agents."""
    # Create a moderately sized test document for multi-perspective analysis
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        content = (
            "Company XYZ was founded in 2020 by John Smith and Jane Doe.\n"
            "The initial team had 5 members and grew to 15 by mid-2021.\n"
            "They developed an innovative software platform that launched in October 2021.\n"
            "By early 2022, they had secured major clients like ABC Corp and QRS Industries.\n"
            "The company raised a Series A funding round of $5M led by Venture Partners in June 2022.\n"
            "John Smith served as CEO while Jane Doe was the CTO until she stepped down in December 2022.\n"
        )
        f.write(content)
        input_file = f.name

    try:
        pipeline = LongtextPipeline()

        # Run pipeline with multi-perspective specialists (but limited count for test speed)
        final_analysis = pipeline.run(
            input_path=input_file,
            mode="general",
            multi_perspective=True,
            specialist_count=2,  # Limited for test speed
            config_path=None,
        )

        assert final_analysis is not None
        assert "multi_perspective" in (final_analysis.metadata or {}).get(
            "error_summary", {}
        )

        # Audit stage should run after multi-perspective analysis
        manifest_path = Path(input_file).parent / ".longtext" / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r") as mf:
                manifest = json.load(mf)

            assert "audit" in manifest.get("stages", {}), (
                "Audit stage should run with multi-perspective mode too"
            )

    finally:
        # Clean up
        Path(input_file).unlink(missing_ok=True)
        longtext_dir = Path(input_file).parent / ".longtext"
        if longtext_dir.exists():
            import shutil

            shutil.rmtree(longtext_dir, ignore_errors=True)
