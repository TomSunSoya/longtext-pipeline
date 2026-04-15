"""
Pipeline orchestrator for the longtext pipeline.

This module provides the LongtextPipeline class which coordinates all stages
of processing: ingest → summarize → stage → final → audit. The orchestrator
handles input validation, configuration loading, manifest management, resume
capabilities, and error handling with the Continue-with-Partial strategy.
"""

import asyncio
import hashlib
import logging
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from ..config import (
    format_missing_settings_message,
    get_missing_required_settings,
    load_runtime_config,
)
from ..errors import StageFailedError
from ..errors.continuation import ErrorAggregator, PartialResult
from ..manifest import ManifestManager, Manifest
from ..models import FinalAnalysis, Part
from ..renderer import format_status
from ..utils.process_lock import InterProcessFileLock
from ..utils.metrics import write_metrics_to_file
from .ingest import IngestStage
from .summarize import SummarizeStage
from .stage_synthesis import StageSynthesisStage
from .final_analysis import FinalAnalysisStage
from .audit import AuditStage


logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result container for pipeline execution with error tracking."""

    success: bool
    final_analysis: Optional[FinalAnalysis]
    errors: List[str]


class LongtextPipeline:
    """
    Orchestrates all stages of the longtext processing pipeline.

    The pipeline flows through four main stages:
    1. Ingest: Load and split input file
    2. Summarize: Summarize individual parts
    3. Stage: Synthesize multiple summaries
    4. Final: Create comprehensive analysis
    5. Audit: Optional verification of results

    Async stages (summarize, stage synthesis, final) are run via asyncio.run()
    to enable concurrent LLM calls within each stage.
    """

    def __init__(self):
        """Initialize the pipeline orchestrator."""
        self.manifest_manager = ManifestManager()
        self.error_aggregator = ErrorAggregator()

    def run(
        self,
        input_path: str,
        config_path: Optional[str] = None,
        mode: str = "general",
        resume: bool = False,
        multi_perspective: bool = False,
        specialist_count: Optional[int] = None,
        max_workers: Optional[int] = None,
    ) -> FinalAnalysis:
        """
        Execute the entire processing pipeline from input to final analysis.

        Args:
            input_path: Path to input file (supports txt/md)
            config_path: Optional path to config file (uses defaults if None)
            mode: Analysis mode ("general" or "relationship")
            resume: Whether to resume from existing checkpoints
            multi_perspective: Whether to use multi-perspective parallel specialist agents
            specialist_count: Optional number of final-analysis specialist agents to run
            max_workers: Optional maximum concurrent workers for summarize and stage stages

        Returns:
            FinalAnalysis object with results and error tracking

        Raises:
            Exception: For fatal errors that prevent pipeline execution
        """
        # Reset error aggregator for this run
        self.error_aggregator = ErrorAggregator()
        all_errors = []
        all_stages = []
        final_analysis = None
        manifest = None
        run_lock = None

        try:
            # 1. Load/validate input file (exists, supported format)
            input_path = self._validate_input_file(input_path)

            # 1.1 Acquire cross-process lock for this input and fail fast on conflicts.
            run_lock = self._acquire_run_lock(input_path)
            logger.info("Acquired pipeline run lock: %s", run_lock.lock_path)

            # 2. Load config from file or use defaults with env overrides
            config = self._load_and_validate_config(config_path, mode)

            # 2.1 Add runtime flags to config based on parameters
            config["multi_perspective"] = multi_perspective
            if max_workers is not None:
                if not isinstance(max_workers, int):
                    raise ValueError("max_workers must be an integer.")
                if not 1 <= max_workers <= 256:
                    raise ValueError("max_workers must be between 1 and 256.")
                config.setdefault("pipeline", {})
                config["pipeline"]["max_workers"] = max_workers
            if specialist_count is not None:
                config.setdefault("pipeline", {})
                config["pipeline"]["specialist_count"] = specialist_count

            # 2.2 Determine output directory from config
            output_dir_config = config.get("output", {}).get("dir")
            if output_dir_config:
                output_dir = Path(output_dir_config)
            else:
                output_dir = None  # Will default to adjacent .longtext/

            # 3. Initialize manifest manager
            manifest_manager = self.manifest_manager

            # 4. Load or create manifest (with content hash for validation)
            input_hash = self._get_input_content_hash(input_path)
            manifest = self._load_or_create_manifest(
                manifest_manager, input_path, input_hash, resume
            )

            # 5. Validate manifest and check completed stages for resume
            if resume and manifest:
                # Verify the input hasn't changed since the previous run
                current_input_hash = self._get_input_content_hash(input_path)
                if not self.manifest_manager.should_resume(
                    manifest, current_input_hash
                ):
                    logger.warning(
                        "Input file has changed since last run; cannot resume from existing manifest"
                    )
                    # Recreate manifest since it's stale
                    manifest = self.manifest_manager.create_manifest(
                        input_path, current_input_hash
                    )
                    self.manifest_manager.save_manifest(manifest)
                    completed_stages = []
                    logger.info("Created fresh manifest; reprocessing all stages")
                else:
                    completed_stages = self.manifest_manager.get_completed_stages(
                        manifest
                    )
                    logger.info(
                        "Resume enabled; completed stages: %s", completed_stages
                    )
            else:
                completed_stages = []
                logger.info("Not resuming; processing all stages")

            # Track current stage for error reporting
            current_stage = None
            all_summaries = []
            final_analysis = None

            try:
                # 6. Execute stages sequentially:
                # ingest → summarize → stage_synthesis → final_analysis → audit

                # STAGE 1: INGEST
                current_stage = "ingest"
                if not resume or "ingest" not in completed_stages:
                    logger.info("Starting %s stage", current_stage)
                    parts_result = self._execute_stage_with_error_handling(
                        self._run_ingest_stage, [input_path, config, manifest], "ingest"
                    )

                    # Handle stage result
                    if parts_result.success and parts_result.data is not None:
                        parts = parts_result.data
                        # Handle warnings if available in result
                        if parts_result.warnings:
                            for warning in parts_result.warnings:
                                self.error_aggregator.add_warning(
                                    current_stage, warning
                                )
                        logger.info("Ingest stage completed with %s parts", len(parts))
                    else:
                        # Try to continue with partial result if available
                        if parts_result.data is not None:
                            parts = parts_result.data
                            logger.warning(
                                "Ingest stage partially completed with %s parts; errors: %s",
                                len(parts),
                                parts_result.errors,
                            )
                            self.error_aggregator.add_errors(
                                current_stage, parts_result.errors
                            )
                        else:
                            # No viable data to continue
                            all_errors.append(
                                "Ingest stage failed and no partial result available - "
                                "pipeline cannot continue"
                            )
                            raise Exception(
                                "Pipeline cannot continue after critical ingest failure"
                            )
                else:
                    # Load parts from existing files based on manifest info
                    parts = self._load_parts_from_existing_files(
                        manifest, input_path, output_dir
                    )
                    logger.info(
                        "Resume loaded %s parts from existing files", len(parts)
                    )
                    # Update manifest to ensure stage is marked as complete
                    self.manifest_manager.update_stage(
                        manifest,
                        "ingest",
                        "successful",
                        output_file=f"parts directory for {Path(input_path).name}",
                    )

                # STAGE 2: SUMMARIZE
                current_stage = "summarize"
                if not resume or "summarize" not in completed_stages:
                    logger.info("Starting %s stage", current_stage)
                    summaries_result = self._execute_stage_with_error_handling(
                        self._run_summarize_stage,
                        [parts, config, manifest, mode],
                        "summarize",
                    )

                    if summaries_result.success and summaries_result.data is not None:
                        all_summaries = summaries_result.data
                        logger.info(
                            "Summarize stage completed with %s summaries",
                            len(all_summaries),
                        )
                    else:
                        all_summaries = []
                        if (
                            summaries_result.data is not None
                            and len(summaries_result.data) > 0
                        ):
                            # We have partial summaries
                            all_summaries = summaries_result.data
                            logger.warning(
                                "Summarize stage partially completed with %s summaries; errors: %s",
                                len(all_summaries),
                                len(summaries_result.errors),
                            )
                        else:
                            logger.error(
                                "Summarize stage failed with no valid partial summaries"
                            )

                        self.error_aggregator.add_errors(
                            current_stage, summaries_result.errors
                        )
                        # Handle warnings if available in result
                        if summaries_result.warnings:
                            for warning in summaries_result.warnings:
                                self.error_aggregator.add_warning(
                                    current_stage, warning
                                )
                else:
                    # Load existing summaries
                    all_summaries = self._load_summaries_from_existing_files(
                        manifest, input_path, output_dir
                    )
                    logger.info(
                        "Resume loaded %s summaries from existing files",
                        len(all_summaries),
                    )
                    # Update manifest to ensure stage is marked as complete
                    self.manifest_manager.update_stage(
                        manifest,
                        "summarize",
                        "successful",
                        output_file=f"summaries directory for {Path(input_path).name}",
                    )

                # STAGE 3: STAGE SYNTHESIS
                current_stage = "stage"
                if not resume or "stage" not in completed_stages:
                    logger.info("Starting %s stage", current_stage)
                    synthesis_result = self._execute_stage_with_error_handling(
                        self._run_stage_synthesis_stage,
                        [all_summaries, config, manifest, mode],
                        "stage",
                    )

                    if synthesis_result.success and synthesis_result.data is not None:
                        all_stages = synthesis_result.data
                        logger.info(
                            "Stage synthesis completed with %s stages",
                            len(all_stages),
                        )
                    else:
                        all_stages = []
                        if (
                            synthesis_result.data is not None
                            and len(synthesis_result.data) > 0
                        ):
                            # We have partial stages
                            all_stages = synthesis_result.data
                            logger.warning(
                                "Stage synthesis partially completed with %s stages; errors: %s",
                                len(all_stages),
                                len(synthesis_result.errors),
                            )
                        else:
                            logger.error(
                                "Stage synthesis failed with no valid partial stages"
                            )

                        self.error_aggregator.add_errors(
                            current_stage, synthesis_result.errors
                        )
                        # Skip warnings since ErrorAggregator only supports errors for now
                        if synthesis_result.warnings:
                            for warning in synthesis_result.warnings:
                                self.error_aggregator.add_warning(
                                    current_stage, warning
                                )
                else:
                    # Load existing stages
                    all_stages = self._load_stages_from_existing_files(
                        manifest, input_path, output_dir
                    )
                    logger.info(
                        "Resume loaded %s stages from existing files", len(all_stages)
                    )
                    # Update manifest to ensure stage is marked as complete
                    self.manifest_manager.update_stage(
                        manifest,
                        "stage",
                        "successful",
                        output_file=f"stages directory for {Path(input_path).name}",
                    )

                # STAGE 4: FINAL ANALYSIS
                current_stage = "final"
                if not resume or "final" not in completed_stages:
                    logger.info("Starting %s stage", current_stage)
                    final_result = self._execute_stage_with_error_handling(
                        self._run_final_analysis_stage,
                        [all_stages, config, manifest, mode],
                        "final",
                    )

                    if final_result.success and final_result.data is not None:
                        final_analysis = final_result.data
                        logger.info("Final analysis stage completed successfully")

                        # Update manifest final metadata
                        final_analysis.metadata["completed_at"] = (
                            datetime.now().isoformat()
                        )
                        manifest.status = "completed"
                    else:
                        # Handle partial/failure case
                        if final_result.data is not None:
                            # Use partial result with error info in final analysis
                            final_analysis = final_result.data
                            final_analysis.metadata["errors"] = final_result.errors
                        else:
                            # Create a partial result structure
                            final_analysis = FinalAnalysis(
                                status="partial_success",
                                stages=all_stages,
                                final_result="Incomplete: Final analysis stage failed",
                                metadata={"errors": final_result.errors},
                            )

                        if final_result.errors:
                            self.error_aggregator.add_errors(
                                current_stage, final_result.errors
                            )
                        if final_result.warnings:
                            for warning in final_result.warnings:
                                self.error_aggregator.add_warning(
                                    current_stage, warning
                                )

                        # Flag as partial success or failed depending on severity
                        if final_result.success:
                            manifest.status = "completed_with_issues"
                        elif len(getattr(final_analysis, "stages", [])) > 0:
                            manifest.status = "partial_success"
                        else:
                            manifest.status = "failed"

                        logger.warning(
                            "Final analysis stage encountered errors but pipeline continues"
                        )

                else:
                    logger.info("Skipping final stage because it is already completed")
                    final_analysis = self._load_final_analysis_from_file(
                        input_path, output_dir
                    )
                    # Update manifest to ensure stage is marked as complete
                    self.manifest_manager.update_stage(
                        manifest,
                        "final",
                        "successful",
                        output_file=f"final_analysis.md for {Path(input_path).name}",
                    )
                    manifest.status = "completed"

                # STAGE 5: AUDIT (conditional)
                current_stage = "audit"
                audit_result = None
                audit_enabled = (
                    config.get("stages", {}).get("audit", {}).get("enabled", False)
                )
                if audit_enabled and (not resume or "audit" not in completed_stages):
                    logger.info("Starting %s stage", current_stage)
                    audit_result = self._execute_stage_with_error_handling(
                        self._run_audit_stage,
                        [final_analysis, config, manifest, mode],
                        "audit",
                    )

                    if audit_result.success and audit_result.data is not None:
                        logger.info("Audit stage completed successfully")
                        for warning in audit_result.warnings or []:
                            self.error_aggregator.add_warning(current_stage, warning)

                        # Store audit result in manifest with findings
                        audit_data = audit_result.data
                        self.manifest_manager.update_stage(
                            manifest,
                            "audit",
                            "successful",
                            output_file="audit_report.md",
                            stats={
                                "issues_found": audit_data.get("issues_found", 0),
                                "confidence_score": audit_data.get(
                                    "confidence_score", 0.0
                                ),
                                "recommendations": audit_data.get(
                                    "recommendations", []
                                ),
                                "checked_items": audit_data.get("checked_items", []),
                                "report_path": audit_data.get("report_path"),
                            },
                        )
                        manifest.status = "completed"
                    else:
                        logger.warning(
                            "Audit stage encountered errors or produced partial results"
                        )
                        if audit_result.errors:
                            self.error_aggregator.add_errors(
                                current_stage, audit_result.errors
                            )
                        # Still update manifest with audit failure info
                        self.manifest_manager.update_stage(
                            manifest,
                            "audit",
                            "failed",
                            error="; ".join(audit_result.errors)
                            if audit_result.errors
                            else "Audit stage failed",
                        )
                elif audit_enabled and resume and "audit" in completed_stages:
                    # Audit stage was previously completed - preserve existing audit findings
                    audit_stage_info = manifest.stages.get("audit")
                    if audit_stage_info and audit_stage_info.stats:
                        # Reload existing audit findings from manifest stats
                        logger.info(
                            "Resume: Audit stage already completed, preserving findings"
                        )
                        self.manifest_manager.update_stage(
                            manifest,
                            "audit",
                            "successful",
                            output_file="audit_report.md",
                            stats=audit_stage_info.stats,
                        )
                        manifest.status = "completed"
                    else:
                        self.manifest_manager.update_stage(
                            manifest,
                            "audit",
                            "successful",
                            output_file="audit_report.md",
                        )
                        manifest.status = "completed"

            except Exception as e:
                error_msg = f"{current_stage} stage failed with error: {str(e)}"
                error_traceback = traceback.format_exc()
                all_errors.extend([error_msg, error_traceback])
                self.error_aggregator.add_errors(
                    current_stage, [str(e), error_traceback]
                )

                # Update manifest with stage error information
                try:
                    self.manifest_manager.update_stage(
                        manifest, current_stage, "failed", error=error_msg
                    )
                    self.manifest_manager.save_manifest(manifest)
                except Exception:
                    pass  # Don't cascade manifest errors

                # Continue with partial result if we have data from earlier stages
                final_analysis = FinalAnalysis(
                    status="partial_success",
                    stages=all_stages,
                    final_result="Pipeline interrupted - partial results available",
                    metadata={
                        "errors": all_errors,
                        "error_summary": self.error_aggregator.get_full_summary(),
                    },
                )

        except KeyboardInterrupt:
            # Handle Ctrl-C interruption gracefully
            logger.warning("Pipeline interrupted by user")
            error_msg = "Pipeline interrupted by user"
            all_errors.append(error_msg)

            if manifest is not None:
                try:
                    self.manifest_manager.update_stage(
                        manifest, "pipeline", "failed", error=error_msg
                    )
                    self.manifest_manager.save_manifest(manifest)
                except Exception:
                    pass

            final_analysis = FinalAnalysis(
                status="interrupted",
                stages=all_stages,
                final_result="Pipeline was interrupted by user",
                metadata={"errors": all_errors},
            )

        except Exception as e:
            # Generic exception handling for critical errors
            error_msg = f"Critical error during pipeline execution: {str(e)}"
            error_traceback = traceback.format_exc()
            all_errors.extend([error_msg, error_traceback])

            final_analysis = FinalAnalysis(
                status="failed",
                stages=[],
                final_result="Pipeline failed due to critical error",
                metadata={"errors": all_errors},
            )

        finally:
            # 9. Generate final status report using renderer
            try:
                if manifest is not None:
                    status_report = format_status(manifest, show_details=True)
                    logger.info("Pipeline status report:\n%s", status_report)

                    # Always save the manifest to persist the current state
                    self.manifest_manager.save_manifest(manifest)

                    logger.info(
                        "Manifest saved: %s",
                        self.manifest_manager._get_manifest_path(input_path),
                    )

                    # Export metrics to Prometheus format after pipeline completion
                    # Use configured output directory or default to adjacent .longtext/
                    output_dir_config = config.get("output", {}).get("dir")
                    if output_dir_config:
                        metrics_output_dir = Path(output_dir_config) / ".longtext"
                    else:
                        metrics_output_dir = Path(input_path).parent / ".longtext"
                    write_metrics_to_file(metrics_output_dir)
                    logger.info(
                        "Metrics exported to %s/metrics.prom", metrics_output_dir
                    )
            except Exception:
                logger.exception(
                    "Failed to generate/render final status report or export metrics"
                )
            finally:
                if run_lock is not None:
                    run_lock.release()
                    logger.info("Released pipeline run lock: %s", run_lock.lock_path)

        # 10. Return FinalAnalysis or partial result with error tracking
        if final_analysis is None:
            final_analysis = FinalAnalysis(
                status="failed",
                stages=all_stages,
                final_result="Pipeline failed before producing output",
                metadata={},
            )
        final_analysis.metadata["pipeline_errors"] = all_errors
        return final_analysis

    def _validate_input_file(self, input_path: str) -> str:
        """Validate that input file exists and has supported format."""
        path = Path(input_path).resolve()

        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_path}")

        # Check if file is readable
        if not os.access(path, os.R_OK):
            raise PermissionError(f"Cannot read input file: {input_path}")

        # Check extension (only txt/md supported)
        ext = path.suffix.lower()
        if ext not in [".txt", ".md"]:
            raise ValueError(
                f"Unsupported file format. Only .txt and .md files are supported, got: {ext}"
            )

        return str(path)

    def _get_run_lock_path(self, input_path: str) -> Path:
        """Build a per-input lock path inside the adjacent .longtext directory."""
        resolved_input = Path(input_path).resolve()
        lock_id = hashlib.sha256(str(resolved_input).encode("utf-8")).hexdigest()[:16]
        return (
            resolved_input.parent
            / ".longtext"
            / ".locks"
            / f"{resolved_input.stem}.{lock_id}.lock"
        )

    def _acquire_run_lock(self, input_path: str) -> InterProcessFileLock:
        """Acquire the per-input cross-process run lock."""
        lock = InterProcessFileLock(self._get_run_lock_path(input_path))
        lock.acquire()
        return lock

    def _load_and_validate_config(self, config_path: Optional[str], mode: str) -> dict:
        """Load and validate configuration with environment overrides."""
        config, _ = load_runtime_config(config_path, search_dir=Path.cwd())

        missing_settings = get_missing_required_settings(config)
        if missing_settings:
            raise ValueError(format_missing_settings_message(missing_settings))

        # Update prompts based on mode
        if "prompts" in config and mode == "relationship":
            config["prompts"]["format"] = "relationship"

        return config

    def _get_input_content_hash(self, input_path: str) -> str:
        """Calculate SHA-256 hash of input file content."""
        from ..utils.hashing import hash_content
        from ..utils.io import read_file

        content = read_file(input_path)
        return hash_content(content)

    def _load_or_create_manifest(
        self,
        manifest_manager: "ManifestManager",
        input_path: str,
        input_hash: str,
        resume: bool,
    ) -> Manifest:
        """Load existing manifest if resume is enabled and input hasn't changed, otherwise create new."""
        if resume:
            # Try to load existing manifest
            existing_manifest = manifest_manager.load_manifest(input_path)
            if existing_manifest:
                # Validate hash to make sure input hasn't changed since last run
                if manifest_manager.should_resume(existing_manifest, input_hash):
                    logger.info("Using existing manifest for resume")
                    return existing_manifest
                else:
                    logger.warning(
                        "Input file has changed since last run; creating new manifest"
                    )

        # Create new manifest (either no existing or input changed)
        manifest = manifest_manager.create_manifest(input_path, input_hash)
        manifest_manager.save_manifest(manifest)
        logger.info("Created new manifest: %s", manifest.session_id)
        return manifest

    def _execute_stage_with_error_handling(
        self, stage_fn, args: List, stage_name: str
    ) -> PartialResult:
        """Normalize stage execution results into PartialResult."""
        try:
            result = stage_fn(*args)
        except StageFailedError as exc:
            errors = [str(error) for error in getattr(exc, "errors", [])] or [str(exc)]
            return PartialResult(
                success=False,
                data=getattr(exc, "partial_result", None),
                errors=errors,
            )
        except Exception as exc:
            return PartialResult(success=False, data=None, errors=[str(exc)])

        if isinstance(result, PartialResult):
            return result

        if result is None:
            return PartialResult(
                success=False,
                data=None,
                errors=[f"{stage_name} stage returned no usable result"],
            )

        return PartialResult(success=True, data=result, errors=[])

    def _run_ingest_stage(
        self, input_path: str, config: dict, manifest: Manifest
    ) -> List[Part]:
        """Run the ingest stage and return the generated parts."""
        stage = IngestStage(manifest_manager=self.manifest_manager)
        result = stage.run(input_path, config, manifest)
        logger.info("Successfully ran ingest stage for %s", input_path)
        return result

    def _run_summarize_stage(
        self,
        parts: List[Part],
        config: dict,
        manifest: Manifest,
        mode: str,
    ) -> list:  # Returns list of Summary objects
        """Run the summarize stage (async internally, sync interface)."""
        stage = SummarizeStage(manifest_manager=self.manifest_manager)
        result = asyncio.run(stage.run(parts, config, manifest, mode))
        logger.info("Successfully ran summarize stage for %s parts", len(parts))
        return result

    def _run_stage_synthesis_stage(
        self,
        summaries: list,  # List of Summary objects
        config: dict,
        manifest: Manifest,
        mode: str,
    ) -> list:  # Returns list of StageSummary objects
        """Run the stage synthesis stage (async internally, sync interface)."""
        stage = StageSynthesisStage(manifest_manager=self.manifest_manager)
        result = asyncio.run(stage.run(summaries, config, manifest, mode))
        logger.info(
            "Successfully ran stage synthesis stage for %s summaries",
            len(summaries),
        )
        return result

    def _run_final_analysis_stage(
        self,
        stages: list,  # List of StageSummary objects
        config: dict,
        manifest: Manifest,
        mode: str,
    ) -> FinalAnalysis:
        """Run the final analysis stage (async internally, sync interface)."""
        stage = FinalAnalysisStage(manifest_manager=self.manifest_manager)
        multi_perspective = config.get("multi_perspective", False)
        result = asyncio.run(
            stage.run(
                stages, config, manifest, mode, multi_perspective=multi_perspective
            )
        )
        logger.info("Successfully ran final analysis stage for %s stages", len(stages))
        return result

    def _run_audit_stage(
        self,
        final_analysis: FinalAnalysis,
        config: dict,
        manifest: Manifest,
        mode: str,
    ) -> dict:  # Returns audit results
        """Run the audit stage."""
        stage = AuditStage(manifest_manager=self.manifest_manager)
        result = stage.run(final_analysis, config, manifest, mode)
        logger.info("Successfully ran audit stage")
        return result

    def _load_parts_from_existing_files(
        self, manifest: Manifest, input_path: str, output_dir: Optional[Path] = None
    ) -> List[Part]:
        """Load parts from existing part files referenced in manifest."""
        try:
            # Use provided output_dir or default to adjacent .longtext/
            if output_dir:
                parts_dir = output_dir / ".longtext"
            else:
                parts_dir = Path(input_path).parent / ".longtext"
            parts = []

            i = 0
            while True:
                part_path = parts_dir / f"part_{i:02d}.txt"
                if not part_path.exists():
                    break

                # Read the part file and extract content following the format in DATA_MODEL.md
                content = ""
                with open(part_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                # Find the content after "---END---" marker
                content_start = -1
                for idx, line in enumerate(lines):
                    if line.strip() == "METADATA_END: ---END---":
                        content_start = idx + 1
                        break

                if content_start != -1 and content_start < len(lines):
                    # Join all lines from content_start onward
                    content = "\n".join(lines[content_start:]).strip()

                if content:
                    from ..models import Part

                    parts.append(
                        Part(
                            index=i,
                            content=content,
                            token_count=int(
                                len(content.split()) * 1.2
                            ),  # rough estimate
                            metadata={"source": str(part_path)},
                        )
                    )

                i += 1

            return parts
        except Exception:
            logger.exception("Failed to load existing parts")
            return []

    def _load_summaries_from_existing_files(
        self, manifest: Manifest, input_path: str, output_dir: Optional[Path] = None
    ) -> list:
        """Load existing summaries from summary files."""
        try:
            # Use provided output_dir or default to adjacent .longtext/
            if output_dir:
                summaries_dir = output_dir / ".longtext"
            else:
                summaries_dir = Path(input_path).parent / ".longtext"
            summaries = []
            from ..models import Summary

            i = 0
            while True:
                summary_path = summaries_dir / f"summary_{i:02d}.md"
                if not summary_path.exists():
                    break

                # Read the summary file content
                with open(summary_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Rehydrate the current Summary dataclass shape from the saved markdown file.
                summary_obj = Summary(
                    part_index=i,
                    content=content,
                    metadata={
                        "status": "loaded_from_file",
                        "path": str(summary_path),
                    },
                )
                summaries.append(summary_obj)
                i += 1

            logger.info("Loaded %s summaries from existing files", len(summaries))
            return summaries
        except Exception:
            logger.exception("Failed to load existing summaries")
            return []

    def _load_stages_from_existing_files(
        self, manifest: Manifest, input_path: str, output_dir: Optional[Path] = None
    ) -> list:
        """Load existing stage summaries from stage files."""
        try:
            # Use provided output_dir or default to adjacent .longtext/
            if output_dir:
                stages_dir = output_dir / ".longtext"
            else:
                stages_dir = Path(input_path).parent / ".longtext"
            stages = []
            from ..models import StageSummary

            i = 0
            while True:
                stage_path = stages_dir / f"stage_{i:02d}.md"
                if not stage_path.exists():
                    break

                # Read the stage file content
                with open(stage_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Rehydrate the current StageSummary dataclass shape from the saved markdown file.
                stage_obj = StageSummary(
                    stage_index=i,
                    summaries=[],
                    synthesis=content,
                    metadata={
                        "status": "loaded_from_file",
                        "path": str(stage_path),
                    },
                )
                stages.append(stage_obj)
                i += 1

            logger.info("Loaded %s stages from existing files", len(stages))
            return stages
        except Exception:
            logger.exception("Failed to load existing stages")
            return []

    def _load_final_analysis_from_file(
        self, input_path: str, output_dir: Optional[Path] = None
    ) -> FinalAnalysis:
        """Load existing final analysis from file."""
        try:
            # Use provided output_dir or default to adjacent .longtext/
            if output_dir:
                load_output_dir = output_dir / ".longtext"
            else:
                load_output_dir = Path(input_path).parent / ".longtext"
            final_path = load_output_dir / "final_analysis.md"
            if final_path.exists():
                from ..utils.io import read_file

                content = read_file(str(final_path))
                return FinalAnalysis(
                    status="completed",
                    stages=[],  # May need to reload based on context, but this ensures the object builds
                    final_result=f"Resumed: Loading existing complete analysis from {final_path}\n{content[:200]}...",
                    metadata={
                        "resumed_from": True,
                        "source_file": str(final_path),
                        "content_preview": content,
                    },
                )
            return FinalAnalysis(
                status="not_found",
                stages=[],
                final_result="Final analysis file not found for resume",
                metadata={},
            )
        except Exception as e:
            logger.exception("Failed to load existing final analysis")
            return FinalAnalysis(
                status="error",
                stages=[],
                final_result="Error loading existing final analysis",
                metadata={"error": str(e)},
            )

    def _save_summaries_to_files(
        self, summaries: list, input_path: str, output_dir: Optional[Path] = None
    ):
        """Save summaries to file system."""
        try:
            from ..renderer import render_summary

            # Use provided output_dir or default to adjacent .longtext/
            if output_dir:
                parts_dir = output_dir / ".longtext"
            else:
                parts_dir = Path(input_path).parent / ".longtext"

            for summary_obj in summaries:
                summary_path = parts_dir / f"summary_{summary_obj.part_index:02d}.md"
                try:
                    rendered_content = render_summary(
                        summary_obj, model="gpt-4o-mini"
                    )  # Use appropriate model from config

                    from ..utils.io import write_file

                    write_file(str(summary_path), rendered_content)
                except Exception:
                    logger.exception("Failed to save summary file %s", summary_path)
        except Exception:
            logger.exception("Failed to save summaries")

    def _save_stages_to_files(
        self, stages: list, input_path: str, output_dir: Optional[Path] = None
    ):
        """Save stages to file system."""
        try:
            from ..renderer import render_stage

            # Use provided output_dir or default to adjacent .longtext/
            if output_dir:
                parts_dir = output_dir / ".longtext"
            else:
                parts_dir = Path(input_path).parent / ".longtext"

            for stage_obj in stages:
                stage_path = parts_dir / f"stage_{stage_obj.stage_index:02d}.md"
                try:
                    rendered_content = render_stage(
                        stage_obj, model="gpt-4o-mini"
                    )  # Use appropriate model from config

                    from ..utils.io import write_file

                    write_file(str(stage_path), rendered_content)
                except Exception:
                    logger.exception("Failed to save stage file %s", stage_path)
        except Exception:
            logger.exception("Failed to save stages")

    def _save_final_analysis_to_file(
        self, final_analysis: FinalAnalysis, input_path: str
    ):
        """Save final analysis to file system."""
        try:
            from ..renderer import render_final

            parts_dir = Path(input_path).parent / ".longtext"
            final_path = parts_dir / "final_analysis.md"

            try:
                rendered_content = render_final(
                    final_analysis, input_path, model="gpt-4o-mini"
                )  # Use appropriate model from config

                from ..utils.io import write_file

                write_file(str(final_path), rendered_content)
            except Exception:
                logger.exception("Failed to save final analysis file %s", final_path)
        except Exception:
            logger.exception("Failed to save final analysis")
