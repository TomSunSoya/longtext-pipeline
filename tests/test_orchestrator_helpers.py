from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.longtext_pipeline.errors import StageFailedError
from src.longtext_pipeline.errors.continuation import PartialResult
from src.longtext_pipeline.models import (
    FinalAnalysis,
    Manifest,
    Part,
    StageInfo,
    StageSummary,
    Summary,
)
from src.longtext_pipeline.pipeline.orchestrator import LongtextPipeline


def _make_manifest(input_path: str) -> Manifest:
    now = datetime(2026, 4, 10, 9, 0, 0)
    return Manifest(
        session_id="session-123",
        input_path=str(Path(input_path).resolve()),
        input_hash="a" * 64,
        stages={
            "ingest": StageInfo(name="ingest", status="not_started"),
            "summarize": StageInfo(name="summarize", status="not_started"),
            "stage": StageInfo(name="stage", status="not_started"),
            "final": StageInfo(name="final", status="not_started"),
            "audit": StageInfo(name="audit", status="skipped"),
        },
        created_at=now,
        updated_at=now,
        status="not_started",
    )


def test_validate_input_file_accepts_txt_and_rejects_missing(tmp_path):
    pipeline = LongtextPipeline()
    input_file = tmp_path / "input.txt"
    input_file.write_text("body", encoding="utf-8")

    assert pipeline._validate_input_file(str(input_file)) == str(input_file.resolve())

    with pytest.raises(FileNotFoundError):
        pipeline._validate_input_file(str(tmp_path / "missing.txt"))


def test_validate_input_file_rejects_unsupported_extension(tmp_path):
    pipeline = LongtextPipeline()
    input_file = tmp_path / "input.pdf"
    input_file.write_text("body", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file format"):
        pipeline._validate_input_file(str(input_file))


def test_load_and_validate_config_updates_relationship_prompts():
    pipeline = LongtextPipeline()

    with patch(
        "src.longtext_pipeline.pipeline.orchestrator.load_runtime_config",
        return_value=({"model": {"api_key": "key"}, "prompts": {}}, []),
    ):
        config = pipeline._load_and_validate_config(None, "relationship")

    assert config["prompts"]["format"] == "relationship"


def test_load_and_validate_config_raises_for_missing_settings():
    pipeline = LongtextPipeline()

    with patch(
        "src.longtext_pipeline.pipeline.orchestrator.load_runtime_config",
        return_value=({"model": {}}, []),
    ):
        with patch(
            "src.longtext_pipeline.pipeline.orchestrator.get_missing_required_settings",
            return_value=["model.api_key"],
        ):
            with patch(
                "src.longtext_pipeline.pipeline.orchestrator.format_missing_settings_message",
                return_value="missing api key",
            ):
                with pytest.raises(ValueError, match="missing api key"):
                    pipeline._load_and_validate_config(None, "general")


def test_load_or_create_manifest_reuses_existing_resume_manifest(tmp_path):
    pipeline = LongtextPipeline()
    input_file = tmp_path / "input.txt"
    input_file.write_text("body", encoding="utf-8")
    existing = _make_manifest(str(input_file))
    manager = Mock()
    manager.load_manifest.return_value = existing
    manager.should_resume.return_value = True

    result = pipeline._load_or_create_manifest(
        manager, str(input_file), "a" * 64, resume=True
    )

    assert result is existing
    manager.create_manifest.assert_not_called()
    manager.save_manifest.assert_not_called()


def test_load_or_create_manifest_creates_new_manifest_when_resume_is_stale(tmp_path):
    pipeline = LongtextPipeline()
    input_file = tmp_path / "input.txt"
    input_file.write_text("body", encoding="utf-8")
    existing = _make_manifest(str(input_file))
    fresh = _make_manifest(str(input_file))
    manager = Mock()
    manager.load_manifest.return_value = existing
    manager.should_resume.return_value = False
    manager.create_manifest.return_value = fresh

    result = pipeline._load_or_create_manifest(
        manager, str(input_file), "b" * 64, resume=True
    )

    assert result is fresh
    manager.create_manifest.assert_called_once_with(str(input_file), "b" * 64)
    manager.save_manifest.assert_called_once_with(fresh)


@pytest.mark.parametrize(
    ("stage_fn", "expected_success", "expected_data", "expected_error"),
    [
        (lambda: "ok", True, "ok", None),
        (lambda: None, False, None, "stage stage returned no usable result"),
        (
            lambda: PartialResult(success=False, data=["partial"], errors=["warn"]),
            False,
            ["partial"],
            "warn",
        ),
    ],
)
def test_execute_stage_with_error_handling_normalizes_results(
    stage_fn, expected_success, expected_data, expected_error
):
    pipeline = LongtextPipeline()

    result = pipeline._execute_stage_with_error_handling(
        lambda: stage_fn(), [], "stage"
    )

    assert result.success is expected_success
    assert result.data == expected_data
    if expected_error is None:
        assert result.errors == []
    else:
        assert expected_error in result.errors


def test_execute_stage_with_error_handling_handles_stage_failed_error():
    pipeline = LongtextPipeline()

    def _raise():
        raise StageFailedError(
            "final", [ValueError("boom")], partial_result=["partial"]
        )

    result = pipeline._execute_stage_with_error_handling(_raise, [], "final")

    assert result.success is False
    assert result.data == ["partial"]
    assert result.errors == ["boom"]


def test_execute_stage_with_error_handling_handles_generic_exception():
    pipeline = LongtextPipeline()

    result = pipeline._execute_stage_with_error_handling(
        lambda: (_ for _ in ()).throw(RuntimeError("broken")),
        [],
        "final",
    )

    assert result.success is False
    assert result.data is None
    assert result.errors == ["broken"]


def test_stage_wrapper_methods_delegate_to_stage_classes(tmp_path):
    pipeline = LongtextPipeline()
    input_file = tmp_path / "input.txt"
    input_file.write_text("body", encoding="utf-8")
    manifest = _make_manifest(str(input_file))
    part = Part(index=0, content="body", token_count=10, metadata={})
    summary = Summary(part_index=0, content="summary", metadata={})
    stage_summary = StageSummary(
        stage_index=0, summaries=[summary], synthesis="stage", metadata={}
    )
    final_analysis = FinalAnalysis(
        status="completed", stages=[stage_summary], final_result="done", metadata={}
    )

    with patch("src.longtext_pipeline.pipeline.orchestrator.IngestStage") as ingest_cls:
        ingest_cls.return_value.run.return_value = [part]
        assert pipeline._run_ingest_stage(str(input_file), {}, manifest) == [part]

    with patch(
        "src.longtext_pipeline.pipeline.orchestrator.SummarizeStage"
    ) as summarize_cls:
        summarize_cls.return_value.run.return_value = "summarize-awaitable"
        with patch(
            "src.longtext_pipeline.pipeline.orchestrator.asyncio.run",
            return_value=[summary],
        ) as run_mock:
            assert pipeline._run_summarize_stage([part], {}, manifest, "general") == [
                summary
            ]
            run_mock.assert_called_once_with("summarize-awaitable")

    with patch(
        "src.longtext_pipeline.pipeline.orchestrator.StageSynthesisStage"
    ) as stage_cls:
        stage_cls.return_value.run.return_value = "stage-awaitable"
        with patch(
            "src.longtext_pipeline.pipeline.orchestrator.asyncio.run",
            return_value=[stage_summary],
        ) as run_mock:
            assert pipeline._run_stage_synthesis_stage(
                [summary], {}, manifest, "general"
            ) == [stage_summary]
            run_mock.assert_called_once_with("stage-awaitable")

    with patch(
        "src.longtext_pipeline.pipeline.orchestrator.FinalAnalysisStage"
    ) as final_cls:
        final_cls.return_value.run.return_value = "final-awaitable"
        with patch(
            "src.longtext_pipeline.pipeline.orchestrator.asyncio.run",
            return_value=final_analysis,
        ) as run_mock:
            assert (
                pipeline._run_final_analysis_stage(
                    [stage_summary], {}, manifest, "general"
                )
                is final_analysis
            )
            run_mock.assert_called_once_with("final-awaitable")

    with patch("src.longtext_pipeline.pipeline.orchestrator.AuditStage") as audit_cls:
        audit_cls.return_value.run.return_value = {"status": "ok"}
        assert pipeline._run_audit_stage(final_analysis, {}, manifest, "general") == {
            "status": "ok"
        }


def test_load_final_analysis_from_file_returns_completed_result(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("body", encoding="utf-8")
    output_dir = tmp_path / ".longtext"
    output_dir.mkdir()
    (output_dir / "final_analysis.md").write_text("full output", encoding="utf-8")
    pipeline = LongtextPipeline()

    result = pipeline._load_final_analysis_from_file(str(input_file))

    assert result.status == "completed"
    assert result.metadata["resumed_from"] is True
    assert "full output" in result.metadata["content_preview"]


def test_save_helpers_render_and_write_expected_files(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("body", encoding="utf-8")
    pipeline = LongtextPipeline()
    summary = Summary(part_index=0, content="summary", metadata={})
    stage_summary = StageSummary(
        stage_index=0, summaries=[summary], synthesis="stage", metadata={}
    )
    final_analysis = FinalAnalysis(
        status="completed", stages=[stage_summary], final_result="done", metadata={}
    )

    with patch(
        "src.longtext_pipeline.renderer.render_summary", return_value="summary-md"
    ):
        with patch("src.longtext_pipeline.utils.io.write_file") as write_file:
            pipeline._save_summaries_to_files([summary], str(input_file))
            write_file.assert_called_once()
            assert write_file.call_args.args[0].endswith("summary_00.md")

    with patch("src.longtext_pipeline.renderer.render_stage", return_value="stage-md"):
        with patch("src.longtext_pipeline.utils.io.write_file") as write_file:
            pipeline._save_stages_to_files([stage_summary], str(input_file))
            write_file.assert_called_once()
            assert write_file.call_args.args[0].endswith("stage_00.md")

    with patch("src.longtext_pipeline.renderer.render_final", return_value="final-md"):
        with patch("src.longtext_pipeline.utils.io.write_file") as write_file:
            pipeline._save_final_analysis_to_file(final_analysis, str(input_file))
            write_file.assert_called_once()
            assert write_file.call_args.args[0].endswith("final_analysis.md")
