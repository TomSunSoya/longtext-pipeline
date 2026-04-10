from pathlib import Path

from src.longtext_pipeline.manifest import ManifestManager
from src.longtext_pipeline.pipeline.orchestrator import LongtextPipeline


def _write_manifest(input_file: Path) -> None:
    manager = ManifestManager()
    manifest = manager.create_manifest(str(input_file), content_hash="x" * 64)
    manifest.stages["ingest"].status = "successful"
    manifest.stages["summarize"].status = "successful"
    manifest.status = "summarize"
    manager.save_manifest(manifest)


def test_load_summaries_from_existing_files_uses_current_summary_shape(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("input", encoding="utf-8")
    output_dir = tmp_path / ".longtext"
    output_dir.mkdir()
    (output_dir / "summary_00.md").write_text("# Summary 0\n\nBody", encoding="utf-8")
    (output_dir / "summary_01.md").write_text("# Summary 1\n\nBody", encoding="utf-8")
    _write_manifest(input_file)

    pipeline = LongtextPipeline()
    manifest = pipeline.manifest_manager.load_manifest(str(input_file))

    summaries = pipeline._load_summaries_from_existing_files(manifest, str(input_file))

    assert [summary.part_index for summary in summaries] == [0, 1]
    assert summaries[0].content.startswith("# Summary 0")
    assert summaries[0].metadata["status"] == "loaded_from_file"


def test_load_stages_from_existing_files_uses_current_stage_summary_shape(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("input", encoding="utf-8")
    output_dir = tmp_path / ".longtext"
    output_dir.mkdir()
    (output_dir / "stage_00.md").write_text("# Stage 0\n\nBody", encoding="utf-8")
    _write_manifest(input_file)

    pipeline = LongtextPipeline()
    manifest = pipeline.manifest_manager.load_manifest(str(input_file))

    stages = pipeline._load_stages_from_existing_files(manifest, str(input_file))

    assert [stage.stage_index for stage in stages] == [0]
    assert stages[0].synthesis.startswith("# Stage 0")
    assert stages[0].summaries == []
    assert stages[0].metadata["status"] == "loaded_from_file"
