"""Tests for batch resume from checkpoint functionality.

These tests cover the extension of resume capability to batch processing,
including checking per-file completion status and skip logic.
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch


from src.longtext_pipeline.utils.batch_processor import BatchProcessor


class TestBatchResumeCheckpoint:
    """Test batch resume from checkpoint functionality."""

    def test_check_file_completion_status_handles_missing_manifest(
        self, tmp_path: Path
    ):
        """Test that files with no manifest are treated as not completed."""
        batch_processor = BatchProcessor()
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = batch_processor._check_file_completion_status(str(test_file))

        assert result["is_completed"] is False

    def test_check_file_completion_status_handles_completed_file(self, tmp_path: Path):
        """Test that completed files are recognized as completed."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        # Create .longtext directory and manifest similar to actual pipeline
        manifest_dir = tmp_path / ".longtext"
        manifest_dir.mkdir()
        manifest_path = manifest_dir / "manifest.json"

        # Create a completed manifest
        manifest_content = {
            "session_id": "test_session_123456",
            "input_path": str(test_file),
            "input_hash": "sha256_hash",  # This would be computed in real scenario
            "created_at": "2026-01-01T10:00:00",
            "updated_at": "2026-01-01T10:05:00",
            "status": "completed",
            "stages": {
                "ingest": {
                    "status": "successful",
                    "input_file": None,
                    "output_file": None,
                    "timestamp": None,
                    "error": None,
                    "stats": None,
                },
                "summarize": {
                    "status": "successful",
                    "input_file": None,
                    "output_file": None,
                    "timestamp": None,
                    "error": None,
                    "stats": None,
                },
                "stage": {
                    "status": "successful",
                    "input_file": None,
                    "output_file": None,
                    "timestamp": None,
                    "error": None,
                    "stats": None,
                },
                "final": {
                    "status": "successful",
                    "input_file": None,
                    "output_file": None,
                    "timestamp": None,
                    "error": None,
                    "stats": None,
                },
                "audit": {
                    "status": "skipped",
                    "input_file": None,
                    "output_file": None,
                    "timestamp": None,
                    "error": None,
                    "stats": None,
                },
            },
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest_content, f)

        batch_processor = BatchProcessor()

        # Directly test the function by mocking the components that are instantiated internally
        with patch(
            "src.longtext_pipeline.utils.hashing.hash_content",
            return_value="sha256_hash",
        ):
            # Mock the manifest manager methods since the manifest manager is created in the constructor
            with patch.object(batch_processor.manifest_manager, "load_manifest"):
                from src.longtext_pipeline.models import Manifest, StageInfo
                from datetime import datetime

                mock_manifest = Manifest(
                    session_id="test_session_123456",
                    input_path=str(test_file),
                    input_hash="sha256_hash",
                    stages={
                        "ingest": StageInfo(name="ingest", status="successful"),
                        "summarize": StageInfo(name="summarize", status="successful"),
                        "stage": StageInfo(name="stage", status="successful"),
                        "final": StageInfo(name="final", status="successful"),
                        "audit": StageInfo(name="audit", status="skipped"),
                    },
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    status="completed",
                )

                batch_processor.manifest_manager.load_manifest.return_value = (
                    mock_manifest
                )

                with patch.object(
                    batch_processor.manifest_manager, "should_resume", return_value=True
                ):
                    with patch.object(
                        batch_processor.manifest_manager,
                        "is_pipeline_complete",
                        return_value=True,
                    ):
                        result = batch_processor._check_file_completion_status(
                            str(test_file)
                        )

        assert result["is_completed"] is True

    def test_check_file_completion_status_handles_changed_file(self, tmp_path: Path):
        """Test that files with changed content are treated as not completed."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("original content")

        # Create .longtext directory and manifest
        manifest_dir = tmp_path / ".longtext"
        manifest_dir.mkdir()
        manifest_path = manifest_dir / "manifest.json"

        # Create a completed manifest, but with a different hash
        manifest_content = {
            "session_id": "test_session_123456",
            "input_path": str(test_file),
            "input_hash": "different_hash_value",
            "created_at": "2026-01-01T10:00:00",
            "updated_at": "2026-01-01T10:05:00",
            "status": "completed",
            "stages": {
                "ingest": {
                    "status": "successful",
                    "input_file": None,
                    "output_file": None,
                    "timestamp": None,
                    "error": None,
                    "stats": None,
                },
                "summarize": {
                    "status": "successful",
                    "input_file": None,
                    "output_file": None,
                    "timestamp": None,
                    "error": None,
                    "stats": None,
                },
                "stage": {
                    "status": "successful",
                    "input_file": None,
                    "output_file": None,
                    "timestamp": None,
                    "error": None,
                    "stats": None,
                },
                "final": {
                    "status": "successful",
                    "input_file": None,
                    "output_file": None,
                    "timestamp": None,
                    "error": None,
                    "stats": None,
                },
                "audit": {
                    "status": "skipped",
                    "input_file": None,
                    "output_file": None,
                    "timestamp": None,
                    "error": None,
                    "stats": None,
                },
            },
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest_content, f)

        batch_processor = BatchProcessor()

        # Mock to set up the scenario where should_resume returns False due to hash mismatch
        with patch(
            "src.longtext_pipeline.utils.io.read_file", return_value="original content"
        ):
            with patch(
                "src.longtext_pipeline.utils.hashing.hash_content",
                return_value="CURRENT_HASH",
            ):
                # Simulate the manifest that has a different hash value
                from src.longtext_pipeline.models import Manifest, StageInfo
                from datetime import datetime

                mock_manifest = Manifest(
                    session_id="test_session_123456",
                    input_path=str(test_file),
                    input_hash="STORAGE_HASH_VALUE",  # Different from current content
                    stages={
                        "ingest": StageInfo(name="ingest", status="successful"),
                        "summarize": StageInfo(name="summarize", status="successful"),
                        "stage": StageInfo(name="stage", status="successful"),
                        "final": StageInfo(name="final", status="successful"),
                        "audit": StageInfo(name="audit", status="skipped"),
                    },
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    status="completed",
                )

                with patch.object(
                    batch_processor.manifest_manager,
                    "load_manifest",
                    return_value=mock_manifest,
                ):
                    with patch.object(
                        batch_processor.manifest_manager,
                        "should_resume",
                        return_value=False,
                    ):
                        result = batch_processor._check_file_completion_status(
                            str(test_file)
                        )

        assert result["is_completed"] is False

    def test_check_file_completion_status_handles_incomplete_file(self, tmp_path: Path):
        """Test that incomplete files are treated as not completed."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        # Create .longtext directory and manifest
        manifest_dir = tmp_path / ".longtext"
        manifest_dir.mkdir()
        manifest_path = manifest_dir / "manifest.json"

        # Create an incomplete manifest
        manifest_content = {
            "session_id": "test_session_123456",
            "input_path": str(test_file),
            "input_hash": "sha256_hash",
            "created_at": "2026-01-01T10:00:00",
            "updated_at": "2026-01-01T10:05:00",
            "status": "summarize",  # Not complete yet
            "stages": {
                "ingest": {
                    "status": "successful",
                    "input_file": None,
                    "output_file": None,
                    "timestamp": None,
                    "error": None,
                    "stats": None,
                },
                "summarize": {
                    "status": "successful",
                    "input_file": None,
                    "output_file": None,
                    "timestamp": None,
                    "error": None,
                    "stats": None,
                },
                "stage": {
                    "status": "running",
                    "input_file": None,
                    "output_file": None,
                    "timestamp": None,
                    "error": None,
                    "stats": None,
                },  # Still in progress
                "final": {
                    "status": "not_started",
                    "input_file": None,
                    "output_file": None,
                    "timestamp": None,
                    "error": None,
                    "stats": None,
                },
                "audit": {
                    "status": "not_started",
                    "input_file": None,
                    "output_file": None,
                    "timestamp": None,
                    "error": None,
                    "stats": None,
                },
            },
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest_content, f)

        batch_processor = BatchProcessor()

        # Mock the hash comparison to return True (no change)
        with patch(
            "src.longtext_pipeline.utils.io.read_file", return_value="test content"
        ):
            with patch(
                "src.longtext_pipeline.utils.hashing.hash_content",
                return_value="sha256_hash",
            ):
                from src.longtext_pipeline.models import Manifest, StageInfo
                from datetime import datetime

                # Create a mock manifest for an incomplete run
                mock_manifest = Manifest(
                    session_id="test_session_123456",
                    input_path=str(test_file),
                    input_hash="sha256_hash",
                    stages={
                        "ingest": StageInfo(name="ingest", status="successful"),
                        "summarize": StageInfo(name="summarize", status="successful"),
                        "stage": StageInfo(
                            name="stage", status="running"
                        ),  # Not completed
                        "final": StageInfo(name="final", status="not_started"),
                        "audit": StageInfo(name="audit", status="not_started"),
                    },
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    status="summarize",  # Processing stopped at 'summarize' stage
                )

                with patch.object(
                    batch_processor.manifest_manager,
                    "load_manifest",
                    return_value=mock_manifest,
                ):
                    with patch.object(
                        batch_processor.manifest_manager,
                        "should_resume",
                        return_value=True,
                    ):
                        with patch.object(
                            batch_processor.manifest_manager,
                            "is_pipeline_complete",
                            return_value=False,
                        ):
                            result = batch_processor._check_file_completion_status(
                                str(test_file)
                            )

        assert result["is_completed"] is False

    def test_process_single_file_skips_if_already_completed_when_resume_enabled(
        self, tmp_path: Path
    ):
        """Test that _process_single_file skips already completed files when resume is enabled."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        # Create .longtext directory and completed manifest
        manifest_dir = tmp_path / ".longtext"
        manifest_dir.mkdir()
        manifest_path = manifest_dir / "manifest.json"

        manifest_content = {
            "session_id": "test_session_123456",
            "input_path": str(test_file),
            "input_hash": "sha256_hash",
            "created_at": "2026-01-01T10:00:00",
            "updated_at": "2026-01-01T10:05:00",
            "status": "completed",
            "stages": {
                "ingest": {
                    "status": "successful",
                    "input_file": None,
                    "output_file": None,
                    "timestamp": None,
                    "error": None,
                    "stats": None,
                },
                "summarize": {
                    "status": "successful",
                    "input_file": None,
                    "output_file": None,
                    "timestamp": None,
                    "error": None,
                    "stats": None,
                },
                "stage": {
                    "status": "successful",
                    "input_file": None,
                    "output_file": None,
                    "timestamp": None,
                    "error": None,
                    "stats": None,
                },
                "final": {
                    "status": "successful",
                    "input_file": None,
                    "output_file": None,
                    "timestamp": None,
                    "error": None,
                    "stats": None,
                },
                "audit": {
                    "status": "skipped",
                    "input_file": None,
                    "output_file": None,
                    "timestamp": None,
                    "error": None,
                    "stats": None,
                },
            },
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest_content, f)

        # Configure for resume
        per_file_config = {
            "config": None,
            "mode": "general",
            "resume": True,
            "multi_perspective": False,
            "agent_count": None,
            "max_workers": None,
        }

        batch_processor = BatchProcessor()

        # We need to patch methods called from _check_file_completion_status
        with patch(
            "src.longtext_pipeline.utils.io.read_file", return_value="test content"
        ):
            with patch(
                "src.longtext_pipeline.utils.hashing.hash_content",
                return_value="sha256_hash",
            ):
                from src.longtext_pipeline.models import Manifest, StageInfo
                from datetime import datetime

                mock_manifest = Manifest(
                    session_id="test_session_123456",
                    input_path=str(test_file),
                    input_hash="sha256_hash",
                    stages={
                        "ingest": StageInfo(name="ingest", status="successful"),
                        "summarize": StageInfo(name="summarize", status="successful"),
                        "stage": StageInfo(name="stage", status="successful"),
                        "final": StageInfo(name="final", status="successful"),
                        "audit": StageInfo(name="audit", status="skipped"),
                    },
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    status="completed",
                )

                with patch.object(
                    batch_processor.manifest_manager,
                    "load_manifest",
                    return_value=mock_manifest,
                ):
                    with patch.object(
                        batch_processor.manifest_manager,
                        "should_resume",
                        return_value=True,
                    ):
                        with patch.object(
                            batch_processor.manifest_manager,
                            "is_pipeline_complete",
                            return_value=True,
                        ):
                            # Mock the pipeline run to verify it's not called when skipping
                            with patch(
                                "src.longtext_pipeline.pipeline.orchestrator.LongtextPipeline.run"
                            ) as mock_run:
                                result = batch_processor._process_single_file(
                                    str(test_file), per_file_config
                                )

        # Verify that pipeline.run was NOT called
        mock_run.assert_not_called()

        # Verify the correct result
        assert result["file"] == str(test_file)
        assert result["success"] is True
        assert result["status"] == "skipped_already_completed"
        assert result["error"] is None
        assert result["manifest_path"] == str(manifest_path)

    def test_process_single_file_calls_pipeline_if_not_completed_when_resume_enabled(
        self, tmp_path: Path
    ):
        """Test that _process_single_file calls pipeline when file is not completed with resume enabled."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        # Configure for resume with incomplete file manifest
        per_file_config = {
            "config": None,
            "mode": "general",
            "resume": True,
            "multi_perspective": False,
            "agent_count": None,
            "max_workers": None,
        }

        batch_processor = BatchProcessor()

        with patch(
            "src.longtext_pipeline.utils.hashing.hash_content", return_value="some_hash"
        ):
            with patch(
                "src.longtext_pipeline.utils.io.read_file", return_value="test content"
            ):
                # Mock a None manifest when no existing manifest exists
                with patch.object(
                    batch_processor.manifest_manager, "load_manifest", return_value=None
                ):
                    # Mock pipeline.run and expect it to be called
                    with patch(
                        "src.longtext_pipeline.pipeline.orchestrator.LongtextPipeline.run"
                    ) as mock_run:
                        # Create a mock FinalAnalysis
                        mock_final_analysis = Mock()
                        mock_final_analysis.status = "completed"

                        mock_run.return_value = mock_final_analysis

                        batch_processor._process_single_file(
                            str(test_file), per_file_config
                        )

        # Verify that pipeline.run was called
        mock_run.assert_called_once()

    def test_process_single_file_calls_pipeline_if_resume_disabled(
        self, tmp_path: Path
    ):
        """Test that _process_single_file calls pipeline when resume is disabled regardless of completion status."""
        # Create test file to satisfy actual file reading requirements
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        # Configure for no resume
        per_file_config = {
            "config": None,
            "mode": "general",
            "resume": False,  # Resume disabled
            "multi_perspective": False,
            "agent_count": None,
            "max_workers": None,
        }

        batch_processor = BatchProcessor()

        with patch(
            "src.longtext_pipeline.pipeline.orchestrator.LongtextPipeline"
        ) as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline_class.return_value = mock_pipeline

            mock_final_analysis = Mock()
            mock_final_analysis.status = "completed"

            mock_pipeline.run.return_value = mock_final_analysis

            batch_processor._process_single_file(str(test_file), per_file_config)

        # Verify that pipeline.run was called despite any potential completion status
        # When resume=False, the code path should go directly to pipeline execution
        mock_pipeline.run.assert_called_once()

    def test_run_sequential_skips_already_completed_files(self, tmp_path: Path):
        """Test that sequential batch run skips already completed files."""
        # Create test files with one already completed
        file1 = tmp_path / "test1.txt"
        file1.write_text("content 1")

        file2 = tmp_path / "test2.txt"
        file2.write_text("content 2")

        # Create manifest for file1 as completed to simulate resume scenario
        manifest_dir1 = tmp_path / ".longtext"
        manifest_dir1.mkdir(exist_ok=True)
        manifest_path1 = manifest_dir1 / "manifest.json"

        manifest_content1 = {
            "session_id": "test_session_123456",
            "input_path": str(file1),
            "input_hash": "sha256_hash1",
            "created_at": "2026-01-01T10:00:00",
            "updated_at": "2026-01-01T10:05:00",
            "status": "completed",
            "stages": {
                "ingest": {
                    "status": "successful",
                    "input_file": None,
                    "output_file": None,
                    "timestamp": None,
                    "error": None,
                    "stats": None,
                },
                "summarize": {
                    "status": "successful",
                    "input_file": None,
                    "output_file": None,
                    "timestamp": None,
                    "error": None,
                    "stats": None,
                },
                "stage": {
                    "status": "successful",
                    "input_file": None,
                    "output_file": None,
                    "timestamp": None,
                    "error": None,
                    "stats": None,
                },
                "final": {
                    "status": "successful",
                    "input_file": None,
                    "output_file": None,
                    "timestamp": None,
                    "error": None,
                    "stats": None,
                },
                "audit": {
                    "status": "skipped",
                    "input_file": None,
                    "output_file": None,
                    "timestamp": None,
                    "error": None,
                    "stats": None,
                },
            },
        }
        with open(manifest_path1, "w") as f:
            json.dump(manifest_content1, f)

        input_files = [str(file1), str(file2)]
        per_file_config = {
            "config": None,
            "mode": "general",
            "resume": True,  # Enable resume to activate skip logic
            "multi_perspective": False,
            "agent_count": None,
            "max_workers": None,
        }

        batch_processor = BatchProcessor(parallel=False, batch_max_workers=1)

        # Patch the hash content and should_resume to confirm files should be skipped if completed
        with patch(
            "src.longtext_pipeline.utils.hashing.hash_content",
            side_effect=["sha256_hash1", "sha256_hash2"],
        ):
            from src.longtext_pipeline.models import Manifest, StageInfo
            from datetime import datetime

            # Mock manifests for each file
            mock_manifest1 = Manifest(
                session_id="test_session_123456",
                input_path=str(file1),
                input_hash="sha256_hash1",
                stages={
                    "ingest": StageInfo(name="ingest", status="successful"),
                    "summarize": StageInfo(name="summarize", status="successful"),
                    "stage": StageInfo(name="stage", status="successful"),
                    "final": StageInfo(name="final", status="successful"),
                    "audit": StageInfo(name="audit", status="skipped"),
                },
                created_at=datetime.now(),
                updated_at=datetime.now(),
                status="completed",
            )
            mock_manifest2 = Manifest(
                session_id="test_session_123457",
                input_path=str(file2),
                input_hash="sha256_hash2",
                stages={
                    "ingest": StageInfo(name="ingest", status="not_started"),
                    "summarize": StageInfo(name="summarize", status="not_started"),
                    "stage": StageInfo(name="stage", status="not_started"),
                    "final": StageInfo(name="final", status="not_started"),
                    "audit": StageInfo(name="audit", status="not_started"),
                },
                created_at=datetime.now(),
                updated_at=datetime.now(),
                status="not_started",
            )

            # Configure mocking properly with the instance's manifest manager
            def load_manifest_side_effect(input_path):
                if input_path == str(file1):
                    return mock_manifest1
                elif input_path == str(file2):
                    return mock_manifest2
                return None

            with patch.object(
                batch_processor.manifest_manager,
                "load_manifest",
                side_effect=load_manifest_side_effect,
            ):
                with patch.object(
                    batch_processor.manifest_manager, "should_resume", return_value=True
                ):
                    with patch.object(
                        batch_processor.manifest_manager,
                        "is_pipeline_complete",
                        side_effect=[True, False],  # First is complete, second is not
                    ):
                        # Mock the pipeline run to track calls - only file2 should result in a pipeline call
                        with patch(
                            "src.longtext_pipeline.pipeline.orchestrator.LongtextPipeline.run"
                        ) as mock_run:
                            mock_final_analysis = Mock()
                            mock_final_analysis.status = "completed"
                            mock_run.return_value = mock_final_analysis

                            results = batch_processor._run_sequential(
                                input_files, per_file_config
                            )

        # Verify pipeline.run was called only once (for file2, not file1 which was skipped)
        assert mock_run.call_count == 1
        # Verify that first file was marked as skipped_already_completed and second as not skipped
        assert results[0]["status"] == "skipped_already_completed"
        assert results[1]["status"] != "skipped_already_completed"

    def test_run_parallel_skips_already_completed_files(self, tmp_path: Path):
        """Test that parallel batch run skips already completed files."""
        # This test focuses on verifying the skip logic works in parallel mode too
        # The actual parallel execution is harder to test precisely due to async nature
        # So we'll verify the underlying functions properly handle skipping

        # For this test focus on validating that _process_single_file would behave correctly
        # in our patched mock environment

        # Create test file
        file1 = tmp_path / "test1.txt"
        file1.write_text("content 1")

        # Create manifest for file1 as completed to simulate resume scenario
        manifest_dir1 = tmp_path / ".longtext"
        manifest_dir1.mkdir(exist_ok=True)
        manifest_path1 = manifest_dir1 / "manifest.json"

        manifest_content1 = {
            "session_id": "test_session_123456",
            "input_path": str(file1),
            "input_hash": "sha256_hash1",
            "created_at": "2026-01-01T10:00:00",
            "updated_at": "2026-01-01T10:05:00",
            "status": "completed",
            "stages": {
                "ingest": {
                    "status": "successful",
                    "input_file": None,
                    "output_file": None,
                    "timestamp": None,
                    "error": None,
                    "stats": None,
                },
                "summarize": {
                    "status": "successful",
                    "input_file": None,
                    "output_file": None,
                    "timestamp": None,
                    "error": None,
                    "stats": None,
                },
                "stage": {
                    "status": "successful",
                    "input_file": None,
                    "output_file": None,
                    "timestamp": None,
                    "error": None,
                    "stats": None,
                },
                "final": {
                    "status": "successful",
                    "input_file": None,
                    "output_file": None,
                    "timestamp": None,
                    "error": None,
                    "stats": None,
                },
                "audit": {
                    "status": "skipped",
                    "input_file": None,
                    "output_file": None,
                    "timestamp": None,
                    "error": None,
                    "stats": None,
                },
            },
        }
        with open(manifest_path1, "w") as f:
            json.dump(manifest_content1, f)

        batch_processor = BatchProcessor(parallel=True, batch_max_workers=2)

        # We'll patch the executor call in _run_parallel
        with patch(
            "src.longtext_pipeline.utils.hashing.hash_content",
            return_value="sha256_hash1",
        ):
            from src.longtext_pipeline.models import Manifest, StageInfo
            from datetime import datetime

            # Create completed manifest
            mock_manifest = Manifest(
                session_id="test_session_123456",
                input_path=str(file1),
                input_hash="sha256_hash1",
                stages={
                    "ingest": StageInfo(name="ingest", status="successful"),
                    "summarize": StageInfo(name="summarize", status="successful"),
                    "stage": StageInfo(name="stage", status="successful"),
                    "final": StageInfo(name="final", status="successful"),
                    "audit": StageInfo(name="audit", status="skipped"),
                },
                created_at=datetime.now(),
                updated_at=datetime.now(),
                status="completed",
            )

            # Mock the instance's manifest manager methods correctly
            with patch.object(
                batch_processor.manifest_manager,
                "load_manifest",
                return_value=mock_manifest,
            ):
                with patch.object(
                    batch_processor.manifest_manager, "should_resume", return_value=True
                ):
                    # Direct test approach for checking file completion,
                    # not relying on complex async internals.
                    # This confirms that the _check_file_completion_status method still works as expected
                    completion_result = batch_processor._check_file_completion_status(
                        str(file1)
                    )
                    assert completion_result["is_completed"]

        # Verify the file was properly identified as completed/skipped in the underlying check
        # The test has set up proper manifest in the same way as actual usage.
        # We're ensuring the check mechanism is working properly.
        with patch(
            "src.longtext_pipeline.utils.hashing.hash_content",
            return_value="sha256_hash1",
        ):
            completion_result = batch_processor._check_file_completion_status(
                str(file1)
            )
        assert completion_result["is_completed"]

    def test_run_batch_properly_sets_concurrency_with_resume_flag(self, tmp_path: Path):
        """Test that run_batch method handles resume flag properly in both modes."""
        # Create test files
        file1 = tmp_path / "test1.txt"
        file1.write_text("content 1")

        file2 = tmp_path / "test2.txt"
        file2.write_text("content 2")

        input_files = [str(file1), str(file2)]
        per_file_config = {
            "config": None,
            "mode": "general",
            "resume": True,  # Test with resume enabled
            "multi_perspective": False,
            "agent_count": None,
            "max_workers": None,
        }

        # Test sequential mode
        batch_processor_seq = BatchProcessor(parallel=False, batch_max_workers=1)

        from src.longtext_pipeline.models import Manifest, StageInfo
        from datetime import datetime

        # Mock manifest to represent completed files
        mock_manifest = Manifest(
            session_id="test_session_123",
            input_path=str(file1),
            input_hash="test_hash",
            stages={
                "ingest": StageInfo(name="ingest", status="successful"),
                "summarize": StageInfo(name="summarize", status="successful"),
                "stage": StageInfo(name="stage", status="successful"),
                "final": StageInfo(name="final", status="successful"),
                "audit": StageInfo(name="audit", status="skipped"),
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status="completed",
        )

        with patch.object(
            batch_processor_seq.manifest_manager,
            "load_manifest",
            return_value=mock_manifest,
        ):
            with patch(
                "src.longtext_pipeline.utils.hashing.hash_content",
                return_value="test_hash",
            ):
                # Mock should_resume to return false for this test
                with patch.object(
                    batch_processor_seq.manifest_manager,
                    "should_resume",
                    return_value=False,
                ):
                    with patch.object(
                        batch_processor_seq, "_run_sequential"
                    ) as mock_seq_method:
                        mock_seq_method.return_value = [
                            {
                                "file": str(file1),
                                "success": True,
                                "status": "completed",
                                "error": None,
                                "manifest_path": None,
                            },
                            {
                                "file": str(file2),
                                "success": True,
                                "status": "completed",
                                "error": None,
                                "manifest_path": None,
                            },
                        ]

                        results = batch_processor_seq.run_batch(
                            input_files, per_file_config
                        )

        # Verify _run_sequential was called with correct params
        mock_seq_method.assert_called()
        assert results[0]["file"] == str(file1)
        assert results[1]["file"] == str(file2)

        # Test parallel mode
        batch_processor_par = BatchProcessor(parallel=True, batch_max_workers=2)
        with patch("asyncio.run") as mock_async_run:
            mock_async_run.return_value = [
                {
                    "file": str(file1),
                    "success": True,
                    "status": "completed",
                    "error": None,
                    "manifest_path": None,
                },
                {
                    "file": str(file2),
                    "success": True,
                    "status": "completed",
                    "error": None,
                    "manifest_path": None,
                },
            ]

            results = batch_processor_par.run_batch(input_files, per_file_config)

        mock_async_run.assert_called()
        assert results[0]["file"] == str(file1)
        assert results[1]["file"] == str(file2)

        # Test parallel mode
        batch_processor_par = BatchProcessor(parallel=True, batch_max_workers=2)
        with patch("asyncio.run") as mock_async_run:
            mock_async_run.return_value = [
                {
                    "file": str(file1),
                    "success": True,
                    "status": "completed",
                    "error": None,
                    "manifest_path": None,
                },
                {
                    "file": str(file2),
                    "success": True,
                    "status": "completed",
                    "error": None,
                    "manifest_path": None,
                },
            ]

            results = batch_processor_par.run_batch(input_files, per_file_config)

        mock_async_run.assert_called()
        assert results[0]["file"] == str(file1)
        assert results[1]["file"] == str(file2)
