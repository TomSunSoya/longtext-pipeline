"""Batch orchestrator for sequential file processing.

This module provides the BatchOrchestrator class which coordinates
batch processing of multiple input files with sequential execution.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any

from ..pipeline.orchestrator import LongtextPipeline
from ..manifest import ManifestManager


logger = logging.getLogger(__name__)


@dataclass
class FileResult:
    """Result of processing a single file in a batch.

    Attributes:
        input_path: Path to the input file that was processed
        success: Whether the file was processed successfully
        output_path: Path to the output directory (e.g., .longtext/)
        manifest_path: Path to the manifest file
        start_time: When processing started
        end_time: When processing completed (or failed)
        errors: List of error messages if processing failed
        warnings: List of warning messages
        params: Processing parameters used (mode, multi_perspective, etc.)
    """

    input_path: str
    success: bool
    output_path: Optional[str] = None
    manifest_path: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate duration in seconds if both times are available."""
        if self.start_time and self.end_time:
            start_dt = datetime.fromisoformat(self.start_time)
            end_dt = datetime.fromisoformat(self.end_time)
            return (end_dt - start_dt).total_seconds()
        return None


@dataclass
class BatchResult:
    """Result of an entire batch processing run.

    Attributes:
        batch_id: Unique identifier for this batch run
        input_sources: List of input paths/files provided to the batch
        files: List of FileResult objects for each processed file
        start_time: When the batch started
        end_time: when the batch completed
        total_files: Total number of files in the batch
        successful_files: Number of files processed successfully
        failed_files: Number of files that failed
        mode: Processing mode (general or relationship)
        multi_perspective: Whether multi-perspective mode was enabled
    """

    batch_id: str
    input_sources: List[str]
    files: List[FileResult]
    start_time: str
    end_time: Optional[str] = None
    total_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    mode: str = "general"
    multi_perspective: bool = False

    @property
    def success(self) -> bool:
        """Whether all files processed successfully."""
        return self.failed_files == 0 and len(self.files) > 0

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate total batch duration in seconds."""
        if self.start_time and self.end_time:
            start_dt = datetime.fromisoformat(self.start_time)
            end_dt = datetime.fromisoformat(self.end_time)
            return (end_dt - start_dt).total_seconds()
        return None


class BatchOrchestrator:
    """Orchestrates batch processing of multiple files with sequential execution.

    This class handles:
    - File discovery from directories or explicit file lists
    - Sequential file processing (no parallelism yet)
    - Per-file manifest management
    - Result aggregation and reporting

    For parallel processing, use the batch_parallel module.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        mode: str = "general",
        multi_perspective: bool = False,
        specialist_count: Optional[int] = None,
        callback: Optional[Callable[[FileResult], None]] = None,
    ):
        """Initialize the batch orchestrator.

        Args:
            output_dir: Optional base output directory for results
            mode: Analysis mode ("general" or "relationship")
            multi_perspective: Whether to use multi-perspective mode
            specialist_count: Number of specialist agents (if multi_perspective)
            callback: Optional callback function called after each file completes
        """
        self.output_dir = output_dir
        self.mode = mode
        self.multi_perspective = multi_perspective
        self.specialist_count = specialist_count
        self.callback = callback
        self.pipeline = LongtextPipeline()
        self.manifest_manager = ManifestManager()

    def discover_files(
        self,
        inputs: List[str],
        recursive: bool = False,
    ) -> List[str]:
        """Discover input files from paths.

        Args:
            inputs: List of paths (files or directories)
            recursive: Whether to recurse into subdirectories

        Returns:
            List of discovered input file paths (txt/md only)
        """
        discovered = []

        for input_path in inputs:
            path = Path(input_path)

            if not path.exists():
                logger.warning("Input path does not exist: %s", input_path)
                continue

            if path.is_file():
                # Single file check
                if path.suffix.lower() in [".txt", ".md"]:
                    discovered.append(str(path.resolve()))
                else:
                    logger.warning(
                        "Skipping non-text file: %s (only .txt and .md supported)",
                        input_path,
                    )
            elif path.is_dir():
                # Directory traversal
                pattern = "**/*" if recursive else "*"
                for file_path in path.glob(pattern):
                    if file_path.is_file() and file_path.suffix.lower() in [
                        ".txt",
                        ".md",
                    ]:
                        discovered.append(str(file_path.resolve()))

        # Remove duplicates and preserve order
        seen = set()
        unique = []
        for p in discovered:
            if p not in seen:
                seen.add(p)
                unique.append(p)

        logger.info("Discovered %d files from %d input(s)", len(unique), len(inputs))
        return unique

    def process_file(
        self,
        input_path: str,
        resume: bool = False,
        config_path: Optional[str] = None,
    ) -> FileResult:
        """Process a single file sequentially.

        Args:
            input_path: Path to input file
            resume: Whether to resume from existing manifest
            config_path: Optional path to config file

        Returns:
            FileResult with processing results
        """
        start_time = datetime.now().isoformat()
        file_result = FileResult(
            input_path=input_path,
            success=False,
            start_time=start_time,
            params={
                "mode": self.mode,
                "multi_perspective": self.multi_perspective,
                "resume": resume,
                "config_path": config_path,
            },
        )

        try:
            logger.info("Processing file: %s", input_path)

            # Run the single-file pipeline
            final_analysis = self.pipeline.run(
                input_path=input_path,
                config_path=config_path,
                mode=self.mode,
                resume=resume,
                multi_perspective=self.multi_perspective,
                specialist_count=self.specialist_count,
            )

            # Determine output and manifest paths
            input_file = Path(input_path)
            output_dir = input_file.parent / ".longtext"
            manifest_path = output_dir / "manifest.json"

            if self.output_dir:
                # Custom output directory support (future enhancement)
                pass

            file_result.success = final_analysis.status in [
                "completed",
                "completed_with_issues",
                "partial_success",
            ]
            file_result.output_path = str(output_dir)
            file_result.manifest_path = str(manifest_path)
            file_result.end_time = datetime.now().isoformat()

            # Check for errors in the final analysis
            metadata_errors = final_analysis.metadata.get("errors", [])
            pipeline_errors = final_analysis.metadata.get("pipeline_errors", [])
            all_errors = metadata_errors + pipeline_errors

            if all_errors:
                file_result.errors = [str(e) for e in all_errors]

            logger.info(
                "File %s completed: success=%s, status=%s",
                input_path,
                file_result.success,
                final_analysis.status,
            )

        except Exception as e:
            logger.exception("Failed to process file: %s", input_path)
            file_result.success = False
            file_result.errors.append(str(e))
            file_result.end_time = datetime.now().isoformat()

        # Call callback if provided
        if self.callback:
            self.callback(file_result)

        return file_result

    def process_batch(
        self,
        inputs: List[str],
        resume: bool = False,
        config_path: Optional[str] = None,
        recursive: bool = False,
    ) -> BatchResult:
        """Process a batch of files sequentially.

        Args:
            inputs: List of input paths (files or directories)
            resume: Whether to resume from existing manifests
            config_path: Optional path to config file
            recursive: Whether to recurse into subdirectories

        Returns:
            BatchResult with results for all files
        """
        batch_id = self.manifest_manager._generate_session_id()
        batch_start = datetime.now().isoformat()

        # Discover files from inputs
        file_list = self.discover_files(inputs, recursive=recursive)

        batch_result = BatchResult(
            batch_id=batch_id,
            input_sources=inputs,
            files=[],
            start_time=batch_start,
            total_files=len(file_list),
            mode=self.mode,
            multi_perspective=self.multi_perspective,
        )

        logger.info(
            "Starting batch %s with %d files (mode=%s)",
            batch_id,
            len(file_list),
            self.mode,
        )

        # Process files sequentially (NO parallelism yet)
        for input_path in file_list:
            file_result = self.process_file(
                input_path=input_path,
                resume=resume,
                config_path=config_path,
            )
            batch_result.files.append(file_result)

            if file_result.success:
                batch_result.successful_files += 1
            else:
                batch_result.failed_files += 1

        batch_result.end_time = datetime.now().isoformat()

        logger.info(
            "Batch %s completed: %d/%d files processed successfully",
            batch_id,
            batch_result.successful_files,
            batch_result.total_files,
        )

        return batch_result

    def get_batch_summary(self, batch_result: BatchResult) -> str:
        """Generate a human-readable summary of batch results.

        Args:
            batch_result: BatchResult to summarize

        Returns:
            Formatted summary string
        """
        lines = [
            f"Batch {batch_result.batch_id} Summary",
            "=" * 40,
            f"Input sources: {', '.join(batch_result.input_sources)}",
            f"Mode: {batch_result.mode}",
            f"Multi-perspective: {batch_result.multi_perspective}",
            "",
            f"Files processed: {batch_result.total_files}",
            f"  Successful: {batch_result.successful_files}",
            f"  Failed: {batch_result.failed_files}",
            f"  Success rate: {batch_result.successful_files / batch_result.total_files * 100:.1f}%",
            "",
            f"Duration: {batch_result.duration_seconds:.2f} seconds"
            if batch_result.duration_seconds
            else "Duration: N/A",
            "",
            "File Details:",
        ]

        for file_result in batch_result.files:
            duration = (
                f"{file_result.duration_seconds:.2f}s"
                if file_result.duration_seconds
                else "N/A"
            )
            status = "✓" if file_result.success else "✗"
            lines.append(f"  {status} {Path(file_result.input_path).name}: {duration}")
            if file_result.errors:
                for error in file_result.errors[:3]:  # Show first 3 errors
                    lines.append(f"      Error: {error[:100]}...")

        return "\n".join(lines)
