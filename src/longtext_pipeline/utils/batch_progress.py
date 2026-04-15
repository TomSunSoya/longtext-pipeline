"""Batch progress tracking utilities for monitoring batch processing.

This module provides progress tracking functionality for batch processing operations,
including real-time progress display and JSON progress files for external monitoring.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ProgressReport:
    """Progress state at a point in time."""

    total_files: int
    processed_files: int
    successful: int
    failed: int
    in_progress: int
    elapsed_seconds: float
    files_processed: list[str] = field(default_factory=list)
    files_failed: list[str] = field(default_factory=list)
    files_in_progress: list[str] = field(default_factory=list)
    eta_seconds: float | None = None
    success_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "successful": self.successful,
            "failed": self.failed,
            "in_progress": self.in_progress,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "files_processed": self.files_processed,
            "files_failed": self.files_failed,
            "files_in_progress": self.files_in_progress,
            "eta_seconds": round(self.eta_seconds, 2) if self.eta_seconds else None,
            "success_rate": round(self.success_rate, 2),
        }

    @property
    def is_complete(self) -> bool:
        """Check if all files have been processed."""
        return self.processed_files >= self.total_files


class ProgressReporter:
    """Tracks and reports batch processing progress in real-time.

    This class maintains statistics about batch processing including:
    - Files processed, succeeded, failed
    - Elapsed time and estimated time to completion
    - Current state of each file being processed

    Args:
        total_files: Total number of files to process
        output_callback: Optional callback for progress updates (default: prints to console)
    """

    def __init__(
        self,
        total_files: int,
        output_callback: Callable[[ProgressReport], None] | None = None,
    ):
        """Initialize the progress reporter.

        Args:
            total_files: Total number of files to process
            output_callback: Optional callback function that receives ProgressReport
        """
        self.total_files = total_files
        self.output_callback = output_callback

        self._start_time = time.time()
        self._processed_files: list[str] = []
        self._failed_files: list[str] = []
        self._in_progress_files: set[str] = set()

        # Track timing for ETA calculation
        self._file_times: list[float] = []

    def start_file(self, file_path: str) -> None:
        """Track that a file processing has started.

        Args:
            file_path: Path to the file being processed
        """
        self._in_progress_files.add(file_path)
        self._emit_update()

    def complete_file(
        self,
        file_path: str,
        success: bool,
        status: str | None = None,
        error: str | None = None,
    ) -> None:
        """Track that a file processing has completed.

        Args:
            file_path: Path to the completed file
            success: Whether the processing was successful
            status: Status string from the processing
            error: Error message if failed
        """
        self._in_progress_files.discard(file_path)

        if success:
            self._processed_files.append(file_path)
        else:
            self._failed_files.append(file_path)

        # Record timing for ETA calculation
        elapsed = time.time() - self._start_time
        processed_count = len(self._processed_files) + len(self._failed_files)
        if processed_count > 1:
            time_per_file = elapsed / processed_count
            self._file_times.append(time_per_file)

        self._emit_update()

    def _emit_update(self) -> None:
        """Emit a progress report if callback is set."""
        if self.output_callback is None:
            return

        elapsed = time.time() - self._start_time
        processed = len(self._processed_files)
        failed = len(self._failed_files)
        in_progress = len(self._in_progress_files)

        # Calculate success rate
        total_attempted = processed + failed
        success_rate = processed / total_attempted if total_attempted > 0 else 0.0

        # Calculate ETA based on recent file processing times
        eta_seconds: float | None = None
        if processed > 0 and total_attempted < self.total_files:
            remaining = self.total_files - total_attempted
            if self._file_times:
                # Use median of recent times for more stable estimate
                avg_time = sum(self._file_times) / len(self._file_times)
                eta_seconds = avg_time * remaining

        report = ProgressReport(
            total_files=self.total_files,
            processed_files=processed + failed,
            successful=processed,
            failed=failed,
            in_progress=in_progress,
            elapsed_seconds=elapsed,
            files_processed=self._processed_files.copy(),
            files_failed=self._failed_files.copy(),
            files_in_progress=list(self._in_progress_files),
            eta_seconds=eta_seconds,
            success_rate=success_rate,
        )

        self.output_callback(report)


class ProgressTracker:
    """Writes progress updates to a JSON file for external monitoring.

    This class creates and maintains a JSON progress file that external tools
    can read to monitor batch processing progress.

    Args:
        progress_file: Path to the JSON progress file to write
        append_timestamp: If True, append timestamp to filename for versioning
    """

    def __init__(
        self,
        progress_file: str | Path,
        append_timestamp: bool = False,
    ):
        """Initialize the progress tracker.

        Args:
            progress_file: Path to the JSON progress file
            append_timestamp: If True, append timestamp to filename
        """
        self.progress_file = Path(progress_file)

        if append_timestamp:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.progress_file = (
                self.progress_file.parent
                / f"{self.progress_file.stem}_{timestamp}{self.progress_file.suffix}"
            )

        # Ensure parent directory exists
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)

    def update(self, report: ProgressReport) -> None:
        """Update the progress JSON file with current progress.

        Args:
            report: ProgressReport containing current state
        """
        progress_data = {
            "progress": report.to_dict(),
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "timestamp": time.time(),
        }

        try:
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump(progress_data, f, indent=2)
        except (IOError, OSError) as e:
            logger.warning("Failed to write progress file: %s", e)

    def record_file_start(self, file_path: str) -> None:
        """Record that a file has started processing.

        Args:
            file_path: Path to the file starting processing
        """
        progress_data = self._load_current()

        if "files_in_progress" not in progress_data:
            progress_data["files_in_progress"] = []
        if file_path not in progress_data["files_in_progress"]:
            progress_data["files_in_progress"].append(file_path)
        progress_data["total_files"] = self._infer_total_files(progress_data)
        progress_data["in_progress"] = len(progress_data["files_in_progress"])
        progress_data["last_updated"] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime()
        )
        progress_data["timestamp"] = time.time()

        self._save(progress_data)

    def record_file_complete(
        self,
        file_path: str,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Record that a file has finished processing.

        Args:
            file_path: Path to the completed file
            success: Whether processing was successful
            error: Error message if failed
        """
        progress_data = self._load_current()

        # Move from in_progress to appropriate list
        if file_path in progress_data.get("files_in_progress", []):
            progress_data["files_in_progress"].remove(file_path)

        if success:
            if "files_processed" not in progress_data:
                progress_data["files_processed"] = []
            if file_path not in progress_data["files_processed"]:
                progress_data["files_processed"].append(file_path)
        else:
            if "files_failed" not in progress_data:
                progress_data["files_failed"] = []
            if file_path not in progress_data["files_failed"]:
                progress_data["files_failed"].append(file_path)
            if error:
                failed_errors = progress_data.setdefault("failed_errors", {})
                failed_errors[file_path] = error

        # Update counts
        total = self._infer_total_files(progress_data)
        progress_data["total_files"] = total
        processed = len(progress_data.get("files_processed", []))
        failed_count = len(progress_data.get("files_failed", []))

        progress_data["processed_files"] = processed + failed_count
        progress_data["successful"] = processed
        progress_data["failed"] = failed_count
        progress_data["in_progress"] = len(progress_data.get("files_in_progress", []))
        progress_data["success_rate"] = round(processed / total if total > 0 else 0, 2)
        progress_data["last_updated"] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime()
        )
        progress_data["timestamp"] = time.time()

        self._save(progress_data)

    def _infer_total_files(self, progress_data: dict[str, Any]) -> int:
        """Infer total file count from the union of known file lists."""
        known_files = set(progress_data.get("files_in_progress", []))
        known_files.update(progress_data.get("files_processed", []))
        known_files.update(progress_data.get("files_failed", []))
        return max(progress_data.get("total_files", 0), len(known_files))

    def _load_current(self) -> dict[str, Any]:
        """Load current progress data, return empty structure if file doesn't exist."""
        if not self.progress_file.exists():
            return {
                "total_files": 0,
                "processed_files": 0,
                "successful": 0,
                "failed": 0,
                "in_progress": 0,
                "elapsed_seconds": 0,
                "files_in_progress": [],
                "files_processed": [],
                "files_failed": [],
                "success_rate": 0.0,
            }

        try:
            with open(self.progress_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Handle both structured and legacy format
                progress = data.get("progress", data)
                if "files_failed" in progress:
                    progress["files_failed"] = [
                        item["file"] if isinstance(item, dict) else item
                        for item in progress["files_failed"]
                    ]
                progress.setdefault("total_files", self._infer_total_files(progress))
                progress.setdefault("processed_files", len(progress.get("files_processed", [])) + len(progress.get("files_failed", [])))
                progress.setdefault("successful", len(progress.get("files_processed", [])))
                progress.setdefault("failed", len(progress.get("files_failed", [])))
                progress.setdefault("in_progress", len(progress.get("files_in_progress", [])))
                progress.setdefault("elapsed_seconds", 0)
                progress.setdefault("success_rate", 0.0)
                return progress
        except (json.JSONDecodeError, IOError):
            return {
                "total_files": 0,
                "processed_files": 0,
                "successful": 0,
                "failed": 0,
                "in_progress": 0,
                "elapsed_seconds": 0,
                "files_in_progress": [],
                "files_processed": [],
                "files_failed": [],
                "success_rate": 0.0,
            }

    def _save(self, data: dict[str, Any]) -> None:
        """Save progress data to file."""
        full_data = {
            "progress": data,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "timestamp": time.time(),
        }

        try:
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump(full_data, f, indent=2)
        except (IOError, OSError) as e:
            logger.warning("Failed to write progress file: %s", e)

    def get_current_report(self) -> ProgressReport | None:
        """Get current progress as a ProgressReport.

        Returns:
            ProgressReport if file exists and is valid, None otherwise
        """
        if not self.progress_file.exists():
            return None

        try:
            with open(self.progress_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                progress = data.get("progress", data)

                return ProgressReport(
                    total_files=progress.get("total_files", 0),
                    processed_files=progress.get("processed_files", 0),
                    successful=progress.get("successful", 0),
                    failed=progress.get("failed", 0),
                    in_progress=progress.get("in_progress", 0),
                    elapsed_seconds=progress.get("elapsed_seconds", 0),
                    files_processed=progress.get("files_processed", []),
                    files_failed=progress.get("files_failed", []),
                    files_in_progress=progress.get("files_in_progress", []),
                    eta_seconds=progress.get("eta_seconds"),
                    success_rate=progress.get("success_rate", 0),
                )
        except (json.JSONDecodeError, IOError):
            return None


def create_default_output_callback() -> Callable[[ProgressReport], None]:
    """Create a default console output callback for progress reports.

    Returns:
        A callback function that prints progress to console
    """
    last_reported = 0.0

    def callback(report: ProgressReport) -> None:
        nonlocal last_reported

        # Rate limit updates to ~per second
        if report.elapsed_seconds - last_reported < 1.0:
            return
        last_reported = report.elapsed_seconds

        # Calculate completion percentage
        completed = report.processed_files
        total = report.total_files
        percent = (completed / total * 100) if total > 0 else 0

        # Format elapsed time
        elapsed_minutes = int(report.elapsed_seconds // 60)
        elapsed_seconds = int(report.elapsed_seconds % 60)
        elapsed_str = f"{elapsed_minutes:02d}:{elapsed_seconds:02d}"

        # Format ETA if available
        eta_str = "N/A"
        if report.eta_seconds is not None:
            eta_minutes = int(report.eta_seconds // 60)
            eta_seconds = int(report.eta_seconds % 60)
            eta_str = f"{eta_minutes:02d}:{eta_seconds:02d}"

        # Build status line
        files_status = f"{completed}/{total}"
        success_status = f"✓ {report.successful} | ✗ {report.failed}"

        print(
            f"\r[{elapsed_str}] [{files_status}] [{success_status}] "
            f"[{percent:5.1f}%] [ETA: {eta_str}]",
            end="",
            flush=True,
        )

        # Print final newline on completion
        if report.is_complete:
            print()  # Newline after progress bar

    return callback


def format_progress_for_cli(report: ProgressReport) -> str:
    """Format a progress report for CLI output.

    Args:
        report: ProgressReport to format

    Returns:
        Formatted string suitable for CLI display
    """
    total = report.total_files
    completed = report.processed_files

    elapsed_minutes = int(report.elapsed_seconds // 60)
    elapsed_seconds = int(report.elapsed_seconds % 60)
    elapsed_str = f"{elapsed_minutes:02d}:{elapsed_seconds:02d}"

    eta_str = "N/A"
    if report.eta_seconds is not None:
        eta_minutes = int(report.eta_seconds // 60)
        eta_seconds = int(report.eta_seconds % 60)
        eta_str = f"{eta_minutes:02d}:{eta_seconds:02d}"

    success_rate = report.success_rate * 100

    lines = [
        f"Progress: [{elapsed_str}] | {completed}/{total} files processed",
        f"  Success: {report.successful} | Failed: {report.failed}",
        f"  Success rate: {success_rate:.1f}% | ETA: {eta_str}",
    ]

    # Show files in progress
    if report.files_in_progress:
        lines.append(f"  In progress: {', '.join(report.files_in_progress)}")

    # Show recent completions
    if report.files_processed:
        recent = report.files_processed[-5:]
        lines.append(f"  Completed: {', '.join(recent)}")

    # Show recent failures
    if report.files_failed:
        recent_failed = report.files_failed[-3:]
        lines.append(f"  Failed: {', '.join(str(f) for f in recent_failed)}")

    return "\n".join(lines)
