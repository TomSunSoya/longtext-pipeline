from unittest.mock import patch

from src.longtext_pipeline.errors.continuation import PartialResult
from src.longtext_pipeline.models import Part, StageSummary, Summary
from src.longtext_pipeline.pipeline.orchestrator import LongtextPipeline


def _make_config() -> dict:
    return {
        "stages": {
            "audit": {
                "enabled": False,
            }
        }
    }


def _make_stage_objects():
    part = Part(index=0, content="body", token_count=10, metadata={})
    summary = Summary(part_index=0, content="summary", metadata={"token_count": 5})
    stage = StageSummary(
        stage_index=0,
        summaries=[summary],
        synthesis="stage synthesis",
        metadata={"token_count": 7},
    )
    return part, summary, stage


def test_run_fails_fast_when_input_lock_is_already_held(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("body", encoding="utf-8")

    holder = LongtextPipeline()._acquire_run_lock(str(input_file))
    try:
        result = LongtextPipeline().run(str(input_file))
    finally:
        holder.release()

    assert result.status == "failed"
    assert any("already running" in error for error in result.metadata["errors"])


def test_run_marks_manifest_completed_with_issues_for_warning_only_final_output(
    tmp_path,
):
    input_file = tmp_path / "input.txt"
    input_file.write_text("body", encoding="utf-8")
    part, summary, stage = _make_stage_objects()
    pipeline = LongtextPipeline()

    with patch.object(
        pipeline, "_load_and_validate_config", return_value=_make_config()
    ):
        with patch.object(pipeline, "_run_ingest_stage", return_value=[part]):
            with patch.object(pipeline, "_run_summarize_stage", return_value=[summary]):
                with patch.object(
                    pipeline, "_run_stage_synthesis_stage", return_value=[stage]
                ):
                    with patch.object(
                        pipeline,
                        "_run_final_analysis_stage",
                        return_value=PartialResult(
                            success=True,
                            data=None,
                            errors=["final output incomplete"],
                            warnings=["missing confidence block"],
                        ),
                    ):
                        result = pipeline.run(str(input_file))

    manifest = pipeline.manifest_manager.load_manifest(str(input_file))

    assert manifest.status == "completed_with_issues"
    assert result.final_result == "Incomplete: Final analysis stage failed"
    assert result.metadata["errors"] == ["final output incomplete"]
    assert pipeline.error_aggregator.warnings["final"] == ["missing confidence block"]


def test_run_marks_manifest_failed_when_final_stage_has_no_output(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("body", encoding="utf-8")
    part, _, _ = _make_stage_objects()
    pipeline = LongtextPipeline()

    with patch.object(
        pipeline, "_load_and_validate_config", return_value=_make_config()
    ):
        with patch.object(pipeline, "_run_ingest_stage", return_value=[part]):
            with patch.object(pipeline, "_run_summarize_stage", return_value=[]):
                with patch.object(
                    pipeline, "_run_stage_synthesis_stage", return_value=[]
                ):
                    with patch.object(
                        pipeline,
                        "_run_final_analysis_stage",
                        return_value=PartialResult(
                            success=False,
                            data=None,
                            errors=["final stage crashed"],
                        ),
                    ):
                        result = pipeline.run(str(input_file))

    manifest = pipeline.manifest_manager.load_manifest(str(input_file))

    assert manifest.status == "failed"
    assert result.stages == []
    assert result.metadata["errors"] == ["final stage crashed"]
    assert pipeline.error_aggregator.errors["final"] == ["final stage crashed"]
