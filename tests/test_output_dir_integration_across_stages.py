"""Tests for output directory integration across all pipeline stages.

This module tests that config.output.dir is respected by all stages:
- Ingest: parts written to output.dir/.longtext/
- Summarize: summaries written to output.dir/.longtext/
- Stage: stage files written to output.dir/.longtext/
- Final: final analysis written to output.dir/.longtext/
- Audit: audit results tracked with output.dir
"""

import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from longtext_pipeline.manifest import ManifestManager
from longtext_pipeline.pipeline.ingest import IngestStage
from longtext_pipeline.pipeline.summarize import SummarizeStage
from longtext_pipeline.pipeline.stage_synthesis import StageSynthesisStage
from longtext_pipeline.pipeline.final_analysis import FinalAnalysisStage


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test output."""
    dirpath = Path(tempfile.mkdtemp())
    yield dirpath
    shutil.rmtree(dirpath, ignore_errors=True)


@pytest.fixture
def sample_input_file(temp_dir: Path) -> Path:
    """Create a sample input file for testing."""
    input_file = temp_dir / "test_input.txt"
    input_file.write_text(
        "This is a test document for output directory integration.\n"
        "It contains multiple paragraphs to test the pipeline stages.\n"
        "\n"
        "The pipeline should respect config.output.dir setting.\n"
        "All output files should be written to the configured directory.\n"
        "\n"
        "This is the third paragraph with more content.\n"
        "Testing multiple parts generation and processing.\n"
        "\n"
        "Final paragraph to ensure sufficient content for splitting.\n"
        "The test verifies output directory enforcement.\n"
    )
    return input_file


@pytest.fixture
def sample_config(temp_dir: Path, output_dir_path: Path) -> dict:
    """Create a sample configuration with custom output directory."""
    config = {
        "model": {
            "provider": "openai",
            "name": "gpt-4o-mini",
            "api_key": "test-key",
            "temperature": 0.7,
            "timeout": 120.0,
        },
        "stages": {
            "ingest": {
                "chunk_size": 200,  # Small chunks for testing
                "overlap_rate": 0.1,
            },
            "summarize": {
                "prompt_template": "prompts/summary_general.txt",
                "batch_size": 2,
            },
            "stage": {
                "group_size": 2,
                "prompt_template": "prompts/stage_general.txt",
            },
            "final": {
                "prompt_template": "prompts/final_general.txt",
            },
            "audit": {
                "enabled": False,  # Disable for basic tests
            },
        },
        "output": {
            "dir": str(output_dir_path),
            "naming": {
                "summarize_prefix": "summary_",
                "stage_prefix": "stage_",
                "final_filename": "final_analysis.md",
            },
        },
        "pipeline": {
            "max_workers": 2,
            "specialist_count": 2,
        },
    }
    return config


@pytest.fixture
def output_dir_path(temp_dir: Path) -> Path:
    """Create a custom output directory path."""
    output_dir = temp_dir / "custom_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


class TestOutputDirIngestStage:
    """Tests for output directory integration in ingest stage."""

    def test_ingest_uses_custom_output_dir(
        self, sample_input_file: Path, sample_config: dict, output_dir_path: Path
    ) -> None:
        """Test that ingest stage writes parts to configured output directory."""
        manifest_manager = ManifestManager()
        manifest = manifest_manager.create_manifest(
            str(sample_input_file), "test_hash_123"
        )

        ingest_stage = IngestStage(manifest_manager=manifest_manager)
        parts = ingest_stage.run(sample_input_file, sample_config, manifest)

        # Verify parts were created
        assert len(parts) > 0

        # Verify output directory structure
        expected_parts_dir = output_dir_path / ".longtext"
        assert expected_parts_dir.exists(), (
            f"Expected parts dir {expected_parts_dir} not found"
        )

        # Verify part files exist in custom output directory
        part_files = list(expected_parts_dir.glob("part_*.txt"))
        assert len(part_files) > 0, "No part files found in custom output directory"

        # Verify manifest contains output_dir_used
        ingest_stage_info = manifest.stages.get("ingest")
        assert ingest_stage_info is not None
        assert ingest_stage_info.stats is not None
        assert "output_dir_used" in ingest_stage_info.stats
        assert ingest_stage_info.stats["output_dir_used"] == str(expected_parts_dir)


class TestOutputDirSummarizeStage:
    """Tests for output directory integration in summarize stage."""

    @pytest.mark.asyncio
    async def test_summarize_uses_custom_output_dir(
        self, sample_input_file: Path, sample_config: dict, output_dir_path: Path
    ) -> None:
        """Test that summarize stage writes summaries to configured output directory."""
        # First run ingest to create parts
        manifest_manager = ManifestManager()
        manifest = manifest_manager.create_manifest(
            str(sample_input_file), "test_hash_456"
        )

        ingest_stage = IngestStage(manifest_manager=manifest_manager)
        parts = ingest_stage.run(sample_input_file, sample_config, manifest)

        # Run summarize stage
        summarize_stage = SummarizeStage(manifest_manager=manifest_manager)
        summaries = await summarize_stage.run(parts, sample_config, manifest, "general")

        # Verify summaries were created
        assert len(summaries) > 0

        # Verify output directory structure
        expected_output_dir = output_dir_path / ".longtext"
        assert expected_output_dir.exists()

        # Verify summary files exist in custom output directory
        summary_files = list(expected_output_dir.glob("summary_*.md"))
        assert len(summary_files) > 0, (
            "No summary files found in custom output directory"
        )

        # Verify manifest contains output_dir_used
        summarize_stage_info = manifest.stages.get("summarize")
        assert summarize_stage_info is not None
        assert summarize_stage_info.stats is not None
        assert "output_dir_used" in summarize_stage_info.stats


class TestOutputDirStageSynthesis:
    """Tests for output directory integration in stage synthesis."""

    @pytest.mark.asyncio
    async def test_stage_synthesis_uses_custom_output_dir(
        self, sample_input_file: Path, sample_config: dict, output_dir_path: Path
    ) -> None:
        """Test that stage synthesis writes stage files to configured output directory."""
        # Run ingest
        manifest_manager = ManifestManager()
        manifest = manifest_manager.create_manifest(
            str(sample_input_file), "test_hash_789"
        )

        ingest_stage = IngestStage(manifest_manager=manifest_manager)
        parts = ingest_stage.run(sample_input_file, sample_config, manifest)

        # Run summarize
        summarize_stage = SummarizeStage(manifest_manager=manifest_manager)
        summaries = await summarize_stage.run(parts, sample_config, manifest, "general")

        # Run stage synthesis
        stage_stage = StageSynthesisStage(manifest_manager=manifest_manager)
        stages = await stage_stage.run(summaries, sample_config, manifest, "general")

        # Verify stages were created
        assert len(stages) > 0

        # Verify output directory structure
        expected_output_dir = output_dir_path / ".longtext"
        assert expected_output_dir.exists()

        # Verify stage files exist in custom output directory
        stage_files = list(expected_output_dir.glob("stage_*.md"))
        assert len(stage_files) > 0, "No stage files found in custom output directory"

        # Verify manifest contains output_dir_used
        stage_info = manifest.stages.get("stage")
        assert stage_info is not None
        assert stage_info.stats is not None
        assert "output_dir_used" in stage_info.stats


class TestOutputDirFinalAnalysis:
    """Tests for output directory integration in final analysis."""

    @pytest.mark.asyncio
    async def test_final_analysis_uses_custom_output_dir(
        self, sample_input_file: Path, sample_config: dict, output_dir_path: Path
    ) -> None:
        """Test that final analysis writes to configured output directory."""
        # Run ingest
        manifest_manager = ManifestManager()
        manifest = manifest_manager.create_manifest(
            str(sample_input_file), "test_hash_final"
        )

        ingest_stage = IngestStage(manifest_manager=manifest_manager)
        parts = ingest_stage.run(sample_input_file, sample_config, manifest)

        # Run summarize
        summarize_stage = SummarizeStage(manifest_manager=manifest_manager)
        summaries = await summarize_stage.run(parts, sample_config, manifest, "general")

        # Run stage synthesis
        stage_stage = StageSynthesisStage(manifest_manager=manifest_manager)
        stages = await stage_stage.run(summaries, sample_config, manifest, "general")

        # Run final analysis
        final_stage = FinalAnalysisStage(manifest_manager=manifest_manager)
        final_analysis = await final_stage.run(
            stages, sample_config, manifest, "general"
        )

        # Verify final analysis was created
        assert final_analysis is not None
        assert final_analysis.status == "completed"

        # Verify output directory structure
        expected_output_dir = output_dir_path / ".longtext"
        assert expected_output_dir.exists()

        # Verify final analysis files exist in custom output directory
        final_md = expected_output_dir / "final_analysis.md"
        final_json = expected_output_dir / "final_analysis.json"
        assert final_md.exists(), (
            "final_analysis.md not found in custom output directory"
        )
        assert final_json.exists(), (
            "final_analysis.json not found in custom output directory"
        )

        # Verify manifest contains output_dir_used
        final_info = manifest.stages.get("final")
        assert final_info is not None
        assert final_info.stats is not None
        assert "output_dir_used" in final_info.stats


class TestOutputDirDefaultBehavior:
    """Tests for default output directory behavior when config.output.dir not set."""

    def test_ingest_uses_default_adjacent_longtext(
        self, sample_input_file: Path, temp_dir: Path
    ) -> None:
        """Test that ingest uses adjacent .longtext/ when output.dir not configured."""
        # Config without output.dir
        config = {
            "model": {
                "provider": "openai",
                "name": "gpt-4o-mini",
                "api_key": "test-key",
                "temperature": 0.7,
                "timeout": 120.0,
            },
            "stages": {
                "ingest": {
                    "chunk_size": 200,
                    "overlap_rate": 0.1,
                },
            },
            "pipeline": {
                "max_workers": 2,
            },
        }

        manifest_manager = ManifestManager()
        manifest = manifest_manager.create_manifest(
            str(sample_input_file), "test_hash_default"
        )

        ingest_stage = IngestStage(manifest_manager=manifest_manager)
        parts = ingest_stage.run(sample_input_file, config, manifest)

        # Verify parts were created
        assert len(parts) > 0

        # Verify default adjacent .longtext/ directory was used
        default_parts_dir = sample_input_file.parent / ".longtext"
        assert default_parts_dir.exists(), (
            f"Expected default parts dir {default_parts_dir} not found"
        )

        # Verify part files exist
        part_files = list(default_parts_dir.glob("part_*.txt"))
        assert len(part_files) > 0


class TestOutputDirManifestTracking:
    """Tests for manifest tracking of output directory usage."""

    def test_manifest_tracks_output_dir_per_stage(
        self, sample_input_file: Path, sample_config: dict, output_dir_path: Path
    ) -> None:
        """Test that manifest tracks output_dir_used for each stage."""
        manifest_manager = ManifestManager()
        manifest = manifest_manager.create_manifest(
            str(sample_input_file), "test_hash_manifest"
        )

        # Run ingest
        ingest_stage = IngestStage(manifest_manager=manifest_manager)
        ingest_stage.run(sample_input_file, sample_config, manifest)

        # Verify manifest tracks output_dir for ingest
        ingest_info = manifest.stages.get("ingest")
        assert ingest_info is not None
        assert ingest_info.stats is not None
        assert "output_dir_used" in ingest_info.stats

        # Verify the tracked output_dir matches expected
        expected_dir = str(output_dir_path / ".longtext")
        assert ingest_info.stats["output_dir_used"] == expected_dir


class TestOutputDirResumeCapability:
    """Tests for output directory integration with resume functionality."""

    def test_resume_loads_from_custom_output_dir(
        self, sample_input_file: Path, sample_config: dict, output_dir_path: Path
    ) -> None:
        """Test that resume loads parts from custom output directory."""
        manifest_manager = ManifestManager()
        manifest = manifest_manager.create_manifest(
            str(sample_input_file), "test_hash_resume"
        )

        # First run - ingest
        ingest_stage = IngestStage(manifest_manager=manifest_manager)
        parts = ingest_stage.run(sample_input_file, sample_config, manifest)
        original_part_count = len(parts)

        # Verify custom output dir was used
        expected_parts_dir = output_dir_path / ".longtext"
        assert expected_parts_dir.exists()

        # Create new manifest manager to simulate resume
        manifest_manager2 = ManifestManager()
        manifest2 = manifest_manager2.load_manifest(str(sample_input_file))
        assert manifest2 is not None

        # Use the orchestrator helper to load from custom output dir
        from longtext_pipeline.pipeline.orchestrator import LongtextPipeline

        pipeline = LongtextPipeline()
        reloaded_parts = pipeline._load_parts_from_existing_files(
            manifest2, str(sample_input_file), output_dir_path
        )

        # Verify parts were loaded from custom directory
        assert len(reloaded_parts) == original_part_count
