"""Batch processing utility for parallel file processing."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .batch_progress import ProgressReporter, ProgressTracker

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..manifest import ManifestManager


@dataclass
class BatchResult:
    """Result of processing a single file in batch mode."""

    file: str
    success: bool
    status: str
    error: str | None = None
    manifest_path: str | None = None


class BatchProcessor:
    """
    Processes multiple files through the pipeline, either sequentially or in parallel.

    This class manages concurrent execution of pipeline runs across multiple input
    files using asyncio.TaskGroup for proper task lifecycle management and
    asyncio.Semaphore to control concurrency limits.

    Attributes:
        parallel: Enable parallel processing mode.
        batch_max_workers: Maximum number of concurrent file processes in parallel mode.
    """

    parallel: bool
    batch_max_workers: int

    def __init__(
        self,
        parallel: bool = False,
        batch_max_workers: int = 1,
        manifest_manager: ManifestManager | None = None,
    ):
        """Initialize the batch processor.

        Args:
            parallel: Enable parallel processing mode.
            batch_max_workers: Maximum concurrent files to process (only used if parallel=True).
            manifest_manager: Optional ManifestManager instance, defaults to new instance.
        """
        self.parallel = parallel
        # Ensure batch_max_workers is at least 1
        self.batch_max_workers = max(1, batch_max_workers)
        # Create manifest manager instance to manage manifests
        if manifest_manager is None:
            from ..manifest import ManifestManager

            self.manifest_manager = ManifestManager()
        else:
            self.manifest_manager = manifest_manager

    def run_batch(
        self,
        input_files: list[str],
        per_file_config: dict[str, Any],
        progress_reporter: ProgressReporter | None = None,
        progress_tracker: ProgressTracker | None = None,
    ) -> list[dict[str, Any]]:
        """Run batch processing on multiple files.

        Args:
            input_files: List of absolute paths to input files.
            per_file_config: Configuration dict to pass to each pipeline run.
            progress_reporter: Optional ProgressReporter for real-time progress updates.
            progress_tracker: Optional ProgressTracker for writing progress to JSON file.

        Returns:
            List of result dictionaries with keys: file, success, status, error, manifest_path
        """
        if progress_reporter:
            progress_reporter.total_files = len(input_files)

        if self.parallel and self.batch_max_workers > 1:
            parallel_run = self._run_parallel(
                input_files, per_file_config, progress_reporter, progress_tracker
            )
            try:
                return asyncio.run(parallel_run)
            finally:
                parallel_run.close()
        else:
            return self._run_sequential(
                input_files, per_file_config, progress_reporter, progress_tracker
            )

    def _run_sequential(
        self,
        input_files: list[str],
        per_file_config: dict[str, Any],
        progress_reporter: ProgressReporter | None = None,
        progress_tracker: ProgressTracker | None = None,
    ) -> list[dict[str, Any]]:
        """Process files sequentially, one at a time.

        Args:
            input_files: List of file paths to process.
            per_file_config: Configuration for each pipeline run.
            progress_reporter: Optional ProgressReporter for real-time updates.
            progress_tracker: Optional ProgressTracker for writing progress to JSON.

        Returns:
            List of result dictionaries.
        """
        results = []

        for i, file_path in enumerate(input_files, start=1):
            # Update progress before processing
            if progress_reporter:
                progress_reporter.start_file(file_path)
            if progress_tracker:
                progress_tracker.record_file_start(file_path)

            logger.info("Processing file %d/%d: %s", i, len(input_files), file_path)

            result = self._process_single_file(file_path, per_file_config)
            results.append(result)

            # Update progress after processing
            if progress_reporter:
                progress_reporter.complete_file(
                    file_path,
                    result["success"],
                    result.get("status"),
                    result.get("error"),
                )
            if progress_tracker:
                progress_tracker.record_file_complete(
                    file_path, result["success"], result.get("error")
                )

            if result["success"]:
                logger.info("Successfully processed: %s", file_path)
            else:
                logger.error("Failed to process %s: %s", file_path, result.get("error"))

        return results

    async def _run_parallel(
        self,
        input_files: list[str],
        per_file_config: dict[str, Any],
        progress_reporter: ProgressReporter | None = None,
        progress_tracker: ProgressTracker | None = None,
    ) -> list[dict[str, Any]]:
        """Process files in parallel using asyncio.TaskGroup and Semaphore.

        Uses asyncio.Semaphore to limit concurrent file processing and
        asyncio.TaskGroup to ensure proper task lifecycle management and cleanup.

        Args:
            input_files: List of file paths to process.
            per_file_config: Configuration for each pipeline run.
            progress_reporter: Optional ProgressReporter for real-time updates.
            progress_tracker: Optional ProgressTracker for writing progress to JSON.

        Returns:
            List of result dictionaries.
        """
        semaphore = asyncio.Semaphore(self.batch_max_workers)
        results: list[dict[str, Any]] = []

        async def process_with_semaphore(file_path: str) -> dict[str, Any]:
            """Process a single file while respecting semaphore limits."""
            async with semaphore:
                # Update progress before processing
                if progress_reporter:
                    progress_reporter.start_file(file_path)
                if progress_tracker:
                    progress_tracker.record_file_start(file_path)

                logger.info("Starting parallel processing: %s", file_path)
                # Run sync pipeline in executor to not block event loop
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, self._process_single_file, file_path, per_file_config
                )
                logger.info(
                    "Completed %s: success=%s",
                    file_path,
                    result.get("success", False),
                )
                # Update progress after processing
                if progress_reporter:
                    progress_reporter.complete_file(
                        file_path,
                        result["success"],
                        result.get("status"),
                        result.get("error"),
                    )
                if progress_tracker:
                    progress_tracker.record_file_complete(
                        file_path, result["success"], result.get("error")
                    )
                return result

        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(process_with_semaphore(file_path))
                for file_path in input_files
            ]

        # Collect results from completed tasks
        for task in tasks:
            results.append(task.result())

        return results

    def _process_single_file(
        self,
        file_path: str,
        per_file_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Process a single file through the pipeline.

        Args:
            file_path: Path to the input file.
            per_file_config: Configuration dict including:
                - config: Optional config file path
                - mode: Analysis mode
                - resume: Resume flag
                - multi_perspective: Multi-perspective flag
                - agent_count: Number of specialist agents
                - max_workers: Workers for internal async stages

        Returns:
            Result dictionary with success status and metadata.
        """
        try:
            # Check if resume is enabled and if file has already been fully processed
            resume_enabled = per_file_config.get("resume", False)
            if resume_enabled:
                completion_status = self._check_file_completion_status(file_path)
                is_completed = completion_status.get(
                    "is_completed", False
                )  # Extract value correctly
                if is_completed:
                    logger.info("Skipping %s - already completed", file_path)
                    manifest_path = str(
                        Path(file_path).parent / ".longtext" / "manifest.json"
                    )

                    return {
                        "file": file_path,
                        "success": True,
                        "status": "skipped_already_completed",
                        "error": None,
                        "manifest_path": manifest_path,
                    }

            from ..pipeline.orchestrator import LongtextPipeline

            pipeline = LongtextPipeline()

            final_analysis = pipeline.run(
                input_path=file_path,
                config_path=per_file_config.get("config"),
                mode=per_file_config.get("mode", "general"),
                resume=resume_enabled,  # Pass resume flag to individual pipeline runs
                multi_perspective=per_file_config.get("multi_perspective", False),
                specialist_count=per_file_config.get("agent_count"),
                max_workers=per_file_config.get("max_workers"),
            )

            # Construct manifest path
            manifest_path = str(Path(file_path).parent / ".longtext" / "manifest.json")

            status = (
                final_analysis.status
                if hasattr(final_analysis, "status")
                else "unknown"
            )

            return {
                "file": file_path,
                "success": status in ("completed", "completed_with_issues"),
                "status": status,
                "error": None,
                "manifest_path": manifest_path,
            }

        except KeyboardInterrupt:
            logger.warning("Processing interrupted for %s", file_path)
            return {
                "file": file_path,
                "success": False,
                "status": "interrupted",
                "error": "Processing interrupted by user",
                "manifest_path": None,
            }

        except Exception as e:
            logger.exception("Pipeline failed for %s", file_path)
            return {
                "file": file_path,
                "success": False,
                "status": "failed",
                "error": str(e),
                "manifest_path": None,
            }

    def _check_file_completion_status(self, file_path: str) -> dict[str, bool]:
        """Check completion status of a file based on its manifest.

        Args:
            file_path: Path to the input file.

        Returns:
            Dictionary with `is_completed` boolean indicating whether file processing is fully completed.
        """
        from ..utils.hashing import hash_content
        from ..utils.io import read_file

        manifest = self.manifest_manager.load_manifest(file_path)

        # Return default as not completed if manifest doesn't exist
        if manifest is None:
            return {"is_completed": False}

        # Check if input file has changed since last run
        current_hash = hash_content(read_file(file_path))
        if not self.manifest_manager.should_resume(manifest, current_hash):
            # Input file changed, so it's not safe to skip
            return {"is_completed": False}

        # Check if all required stages are complete
        is_pipeline_complete = self.manifest_manager.is_pipeline_complete(manifest)

        return {"is_completed": is_pipeline_complete}
