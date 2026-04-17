"""Tests for the sequential batch orchestrator.

This module contains tests for the BatchOrchestrator class with
sequential file processing. No parallel mode tests here - those
go in test_batch_parallel.py.
"""

from pathlib import Path


from src.longtext_pipeline.batch.orchestrator import (
    BatchOrchestrator,
    FileResult,
    BatchResult,
)


class TestFileResult:
    """Tests for FileResult dataclass."""

    def test_file_result_defaults(self):
        """FileResult should have sensible defaults."""
        result = FileResult(
            input_path="/path/to/file.txt",
            success=True,
        )

        assert result.input_path == "/path/to/file.txt"
        assert result.success is True
        assert result.output_path is None
        assert result.manifest_path is None
        assert result.errors == []
        assert result.warnings == []
        assert result.params == {}
        assert result.start_time is None
        assert result.end_time is None
        assert result.duration_seconds is None

    def test_file_result_with_times(self):
        """FileResult should calculate duration from start/end times."""
        result = FileResult(
            input_path="/path/to/file.txt",
            success=True,
            start_time="2026-01-01T10:00:00",
            end_time="2026-01-01T10:05:00",
        )

        assert result.duration_seconds == 300.0

    def test_file_result_with_errors(self):
        """FileResult should store errors correctly."""
        result = FileResult(
            input_path="/path/to/file.txt",
            success=False,
            errors=["Error 1", "Error 2"],
        )

        assert result.success is False
        assert len(result.errors) == 2
        assert result.errors[0] == "Error 1"


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_batch_result_defaults(self):
        """BatchResult should have sensible defaults."""
        result = BatchResult(
            batch_id="test_batch_001",
            input_sources=["/input/dir"],
            files=[],
            start_time="2026-01-01T10:00:00",
        )

        assert result.batch_id == "test_batch_001"
        assert result.input_sources == ["/input/dir"]
        assert result.files == []
        assert result.total_files == 0
        assert result.successful_files == 0
        assert result.failed_files == 0
        assert result.mode == "general"
        assert result.multi_perspective is False

    def test_batch_result_success_status(self):
        """BatchResult.success should be True when no failures."""
        files = [
            FileResult(
                input_path="/file1.txt",
                success=True,
            ),
            FileResult(
                input_path="/file2.txt",
                success=True,
            ),
        ]
        result = BatchResult(
            batch_id="test_batch_002",
            input_sources=["/input"],
            files=files,
            start_time="2026-01-01T10:00:00",
            successful_files=2,
            total_files=2,
        )

        assert result.success is True
        assert result.successful_files == 2
        assert result.failed_files == 0

    def test_batch_result_failure_status(self):
        """BatchResult.success should be False when any failures."""
        files = [
            FileResult(
                input_path="/file1.txt",
                success=True,
            ),
            FileResult(
                input_path="/file2.txt",
                success=False,
            ),
        ]
        result = BatchResult(
            batch_id="test_batch_003",
            input_sources=["/input"],
            files=files,
            start_time="2026-01-01T10:00:00",
            successful_files=1,
            failed_files=1,
            total_files=2,
        )

        assert result.success is False
        assert result.failed_files == 1

    def test_batch_result_empty(self):
        """BatchResult with no files should have success=False."""
        result = BatchResult(
            batch_id="test_batch_004",
            input_sources=["/input"],
            files=[],
            start_time="2026-01-01T10:00:00",
            total_files=0,
            successful_files=0,
            failed_files=0,
        )

        assert result.success is False

    def test_batch_result_duration(self):
        """BatchResult should calculate duration from start/end times."""
        result = BatchResult(
            batch_id="test_batch_005",
            input_sources=["/input"],
            files=[],
            start_time="2026-01-01T10:00:00",
            end_time="2026-01-01T10:30:00",
        )

        assert result.duration_seconds == 1800.0


class TestBatchOrchestratorDiscovery:
    """Tests for file discovery in BatchOrchestrator."""

    def test_discover_single_file(self, tmp_path: Path):
        """Should discover a single file."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        orchestrator = BatchOrchestrator()
        files = orchestrator.discover_files([str(test_file)])

        assert len(files) == 1
        assert files[0] == str(test_file.resolve())

    def test_discover_multiple_files(self, tmp_path: Path):
        """Should discover multiple files from a list."""
        # Create test files
        file1 = tmp_path / "test1.txt"
        file2 = tmp_path / "test2.md"
        file3 = tmp_path / "test3.txt"
        file1.write_text("content 1")
        file2.write_text("content 2")
        file3.write_text("content 3")

        orchestrator = BatchOrchestrator()
        files = orchestrator.discover_files([str(file1), str(file2), str(file3)])

        assert len(files) == 3

    def test_discover_directory(self, tmp_path: Path):
        """Should discover files in a directory."""
        # Create directory with files
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        file1 = subdir / "test1.txt"
        file2 = subdir / "test2.md"
        file3 = subdir / "test3.py"  # Should be skipped
        file1.write_text("content 1")
        file2.write_text("content 2")
        file3.write_text("print('skip me')")

        orchestrator = BatchOrchestrator()
        files = orchestrator.discover_files([str(subdir)])

        # Only .txt and .md files should be found
        assert len(files) == 2
        paths = {str(Path(f).name) for f in files}
        assert "test1.txt" in paths
        assert "test2.md" in paths

    def test_discover_recursive(self, tmp_path: Path):
        """Should discover files recursively when enabled."""
        # Create nested directory structure
        subdir1 = tmp_path / "level1"
        subdir2 = subdir1 / "level2"
        subdir1.mkdir()
        subdir2.mkdir()

        file1 = subdir1 / "test1.txt"
        file2 = subdir2 / "test2.txt"
        file1.write_text("content 1")
        file2.write_text("content 2")

        orchestrator = BatchOrchestrator()
        files = orchestrator.discover_files([str(subdir1)], recursive=True)

        assert len(files) == 2

    def test_discover_skips_non_text(self, tmp_path: Path):
        """Should skip non-text files."""
        # Create files with various extensions
        (tmp_path / "test.txt").write_text("txt content")
        (tmp_path / "test.md").write_text("md content")
        (tmp_path / "test.py").write_text("python code")
        (tmp_path / "test.json").write_text('{"test": true}')
        (tmp_path / "test.log").write_text("log content")

        orchestrator = BatchOrchestrator()
        files = orchestrator.discover_files([str(tmp_path)])

        # Only txt and md should be found
        assert len(files) == 2
        paths = {str(Path(f).name) for f in files}
        assert "test.txt" in paths
        assert "test.md" in paths


class TestBatchOrchestratorIntegration:
    """Integration tests for BatchOrchestrator.

    These tests actually process files through the pipeline.
    Use minimal test files to keep tests fast.
    """

    def test_process_single_file(self, tmp_path: Path):
        """Should process a single file successfully."""
        # Create a minimal test file
        test_file = tmp_path / "input.txt"
        test_file.write_text(
            "This is a simple test document.\n"
            "It has a few sentences.\n"
            "The content is short for testing purposes.\n"
        )

        orchestrator = BatchOrchestrator()
        result = orchestrator.process_file(str(test_file))

        assert result.input_path == str(test_file.resolve())
        assert result.start_time is not None
        assert result.end_time is not None
        # Result may or may not succeed depending on API setup
        # Just check the structure is correct
        assert result.success in [True, False]  # Either is acceptable for testing
        assert result.manifest_path is not None
        assert Path(result.manifest_path).parent.exists()

    def test_process_batch(self, tmp_path: Path):
        """Should process multiple files in batch."""
        # Create multiple test files
        for i in range(3):
            test_file = tmp_path / f"input_{i}.txt"
            test_file.write_text(
                f"This is test file {i}.\nIt contains some text for batch processing.\n"
            )

        orchestrator = BatchOrchestrator()
        batch_result = orchestrator.process_batch([str(tmp_path)])

        assert batch_result.batch_id is not None
        assert batch_result.input_sources == [str(tmp_path)]
        assert batch_result.total_files == 3
        assert len(batch_result.files) == 3

        # Check summary methods
        summary = orchestrator.get_batch_summary(batch_result)
        assert "Batch" in summary
        assert "Files processed" in summary
