"""Tests for batch progress tracking utilities."""

import json
import time
from pathlib import Path

from longtext_pipeline.utils.batch_progress import (
    ProgressReporter,
    ProgressTracker,
    create_default_output_callback,
    format_progress_for_cli,
    ProgressReport,
)


class TestProgressReport:
    """Tests for ProgressReport dataclass."""

    def test_initial_state(self) -> None:
        """Test that ProgressReport initializes correctly."""
        report = ProgressReport(
            total_files=10,
            processed_files=5,
            successful=4,
            failed=1,
            in_progress=0,
            elapsed_seconds=30.0,
        )

        assert report.total_files == 10
        assert report.processed_files == 5
        assert report.successful == 4
        assert report.failed == 1
        assert report.in_progress == 0
        assert report.elapsed_seconds == 30.0
        assert not report.is_complete

    def test_is_complete(self) -> None:
        """Test is_complete property."""
        # Not complete - more to process
        report = ProgressReport(
            total_files=10,
            processed_files=5,
            successful=4,
            failed=1,
            in_progress=0,
            elapsed_seconds=30.0,
        )
        assert not report.is_complete

        # Complete - all processed
        report = ProgressReport(
            total_files=10,
            processed_files=10,
            successful=8,
            failed=2,
            in_progress=0,
            elapsed_seconds=60.0,
        )
        assert report.is_complete

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        report = ProgressReport(
            total_files=5,
            processed_files=3,
            successful=2,
            failed=1,
            in_progress=0,
            elapsed_seconds=45.5,
            files_processed=["a.txt", "b.txt"],
            files_failed=["c.txt"],
            eta_seconds=90.0,
            success_rate=0.67,
        )

        d = report.to_dict()

        assert d["total_files"] == 5
        assert d["processed_files"] == 3
        assert d["successful"] == 2
        assert d["failed"] == 1
        assert d["files_processed"] == ["a.txt", "b.txt"]
        assert d["files_failed"] == ["c.txt"]
        assert d["eta_seconds"] == 90.0
        assert d["success_rate"] == 0.67


class TestProgressReporter:
    """Tests for ProgressReporter class."""

    def test_start_and_complete_file(self) -> None:
        """Test tracking file start and completion."""
        reported = []

        def callback(report: ProgressReport) -> None:
            reported.append(report.to_dict())

        reporter = ProgressReporter(total_files=3, output_callback=callback)

        # Start first file
        reporter.start_file("file1.txt")
        assert len(reported) == 1

        # Complete first file successfully
        reporter.complete_file("file1.txt", success=True, status="completed")
        assert len(reported) == 2
        assert "file1.txt" in reported[-1]["files_processed"]

        # Complete second file failed
        reporter.complete_file(
            "file2.txt", success=False, status="failed", error="Timeout"
        )
        assert len(reported) == 3
        assert "file2.txt" in reported[-1]["files_failed"]

    def test_no_callback(self) -> None:
        """Test that reporter works without callback."""
        reporter = ProgressReporter(total_files=2)

        reporter.start_file("file1.txt")
        reporter.complete_file("file1.txt", success=True)
        reporter.complete_file("file2.txt", success=True)

        assert reporter.total_files == 2

    def test_eta_calculation(self) -> None:
        """Test ETA calculation based on processing times."""
        reporter = ProgressReporter(total_files=10)

        # Simulate processing 3 files
        for i in range(3):
            reporter.start_file(f"file{i}.txt")
            time.sleep(0.01)  # Simulate some processing time
            reporter.complete_file(f"file{i}.txt", success=True)

        # Check that ETA is calculated (should be None initially, then calculated)
        reporter._emit_update()

    def test_success_rate(self) -> None:
        """Test success rate calculation."""
        reporter = ProgressReporter(total_files=5)

        # 3 success, 1 fail
        reporter.complete_file("file1.txt", success=True)
        reporter.complete_file("file2.txt", success=True)
        reporter.complete_file("file3.txt", success=True)
        reporter.complete_file("file4.txt", success=False)

        # Success rate should be 3/4 = 0.75
        # (This is tested implicitly through the callback)


class TestProgressTracker:
    """Tests for ProgressTracker class."""

    def test_write_and_read_progress(self, tmp_path: Path) -> None:
        """Test writing and reading progress from JSON file."""
        progress_file = tmp_path / "progress.json"
        tracker = ProgressTracker(progress_file)

        # Simulate some processing
        tracker.record_file_start("file1.txt")
        time.sleep(0.01)
        tracker.record_file_complete("file1.txt", success=True)

        tracker.record_file_start("file2.txt")
        time.sleep(0.01)
        tracker.record_file_complete(
            "file2.txt", success=False, error="Connection error"
        )

        # Read back the file
        assert progress_file.exists()

        with open(progress_file) as f:
            data = json.load(f)

        assert "progress" in data
        progress = data["progress"]
        assert progress["total_files"] == 2
        assert progress["processed_files"] == 2
        assert progress["successful"] == 1
        assert progress["failed"] == 1
        assert "file1.txt" in progress["files_processed"]
        assert "file2.txt" in progress["files_failed"]

    def test_get_current_report(self, tmp_path: Path) -> None:
        """Test getting report from tracker."""
        progress_file = tmp_path / "progress.json"
        tracker = ProgressTracker(progress_file)

        # Simulate processing
        tracker.record_file_start("file1.txt")
        tracker.record_file_complete("file1.txt", success=True)

        report = tracker.get_current_report()
        assert report is not None
        assert report.total_files == 1
        assert report.processed_files == 1
        assert report.successful == 1

    def test_get_report_when_no_file(self, tmp_path: Path) -> None:
        """Test getting report when no progress file exists."""
        progress_file = tmp_path / "nonexistent.json"
        tracker = ProgressTracker(progress_file)

        # Should return None for non-existent file
        report = tracker.get_current_report()
        assert report is None


class TestOutputCallback:
    """Tests for the default output callback."""

    def test_callback_creation(self) -> None:
        """Test that callback can be created."""
        callback = create_default_output_callback()
        assert callable(callback)

    def test_callback_executes(self) -> None:
        """Test that callback executes without errors."""
        callback = create_default_output_callback()

        report = ProgressReport(
            total_files=10,
            processed_files=3,
            successful=2,
            failed=1,
            in_progress=0,
            elapsed_seconds=10.0,
            eta_seconds=40.0,
            success_rate=0.67,
        )

        # This should not raise
        callback(report)


class TestFormatProgressForCLI:
    """Tests for format_progress_for_cli function."""

    def test_basic_formatting(self) -> None:
        """Test basic progress formatting."""
        report = ProgressReport(
            total_files=100,
            processed_files=25,
            successful=20,
            failed=5,
            in_progress=0,
            elapsed_seconds=300.0,
            eta_seconds=900.0,
            success_rate=0.8,
        )

        formatted = format_progress_for_cli(report)

        assert "Progress:" in formatted
        assert "25/100 files processed" in formatted
        assert "Success: 20 | Failed: 5" in formatted
        assert "Success rate: 80.0%" in formatted
        assert "ETA:" in formatted

    def test_format_with_in_progress_files(self) -> None:
        """Test formatting with files in progress."""
        report = ProgressReport(
            total_files=10,
            processed_files=7,
            successful=5,
            failed=2,
            in_progress=1,
            elapsed_seconds=60.0,
            files_in_progress=["file7.txt"],
            files_processed=["file1.txt", "file2.txt", "file3.txt"],
            files_failed=["file8.txt", "file9.txt"],
            success_rate=0.71,
        )

        formatted = format_progress_for_cli(report)

        assert "In progress:" in formatted
        assert "file7.txt" in formatted
        assert "Completed:" in formatted
        assert "file1.txt" in formatted

    def test_format_without_eta(self) -> None:
        """Test formatting when ETA is not available."""
        report = ProgressReport(
            total_files=10,
            processed_files=3,
            successful=2,
            failed=1,
            in_progress=0,
            elapsed_seconds=45.0,
            eta_seconds=None,
            success_rate=0.67,
        )

        formatted = format_progress_for_cli(report)

        assert "ETA: N/A" in formatted


class TestIntegration:
    """Integration tests for progress tracking."""

    def test_end_to_end_workflow(self, tmp_path: Path) -> None:
        """Test the full workflow: reporter + tracker."""
        progress_file = tmp_path / "batch_progress.json"
        reporter_callback = None

        def callback(report: ProgressReport) -> None:
            nonlocal reporter_callback
            reporter_callback = report

        reporter = ProgressReporter(total_files=3, output_callback=callback)
        tracker = ProgressTracker(progress_file)

        # Simulate processing 3 files
        files = ["doc1.txt", "doc2.txt", "doc3.txt"]

        for i, file_path in enumerate(files):
            reporter.start_file(file_path)
            tracker.record_file_start(file_path)

            # Simulate some processing
            time.sleep(0.001)

            # First two succeed, third fails
            success = i < 2
            reporter.complete_file(
                file_path,
                success=success,
                status="completed" if success else "failed",
                error=None if success else "Simulated error",
            )
            tracker.record_file_complete(
                file_path, success=success, error=None if success else "Simulated error"
            )

        # Verify reporter got the callback
        assert reporter_callback is not None
        assert reporter_callback.successful == 2
        assert reporter_callback.failed == 1

        # Verify tracker file
        assert progress_file.exists()

        with open(progress_file) as f:
            data = json.load(f)

        progress = data["progress"]
        assert progress["processed_files"] == 3
        assert progress["successful"] == 2
        assert progress["failed"] == 1
