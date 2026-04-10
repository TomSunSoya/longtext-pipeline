from datetime import datetime

import pytest

from src.longtext_pipeline.models import (
    FinalAnalysis,
    Manifest,
    StageInfo,
    StageSummary,
    Summary,
)
from src.longtext_pipeline.renderer import (
    _format_dict,
    _format_list,
    _format_stage_info,
    _safe_get,
    _determine_next_action,
    format_output_type,
    format_status,
    render_manifest_status,
    render_stage,
    render_summary,
    render_final,
)


def _make_manifest(status: str = "not_started") -> Manifest:
    now = datetime(2026, 4, 10, 9, 0, 0)
    return Manifest(
        session_id="session-123",
        input_path="/tmp/input.txt",
        input_hash="a" * 64,
        stages={
            "ingest": StageInfo(name="ingest", status="successful", timestamp=now),
            "summarize": StageInfo(
                name="summarize", status="successful", timestamp=now
            ),
            "stage": StageInfo(name="stage", status="successful", timestamp=now),
            "final": StageInfo(name="final", status="successful", timestamp=now),
            "audit": StageInfo(name="audit", status="skipped", timestamp=now),
        },
        created_at=now,
        updated_at=now,
        status=status,
        total_parts=2,
        total_stages=1,
        estimated_tokens=42,
    )


def test_render_manifest_status_includes_stage_errors():
    manifest = _make_manifest(status="failed")
    manifest.stages["summarize"].status = "failed"
    manifest.stages["summarize"].error = "provider timeout"

    output = render_manifest_status(manifest)

    assert "ERROR in summarize stage:" in output
    assert "provider timeout" in output


def test_format_status_handles_completed_with_issues():
    manifest = _make_manifest(status="completed_with_issues")

    output = format_status(manifest, show_details=True)

    assert "Overall Status: [⚠] completed_with_issues" in output
    assert "Pipeline completed with warnings" in output


@pytest.mark.parametrize(
    ("status", "stage_name", "stage_status", "expected"),
    [
        ("completed", None, None, "Pipeline completed successfully"),
        ("partial_success", None, None, "Pipeline completed with partial success"),
        ("running", "summarize", "failed", "Summarize stage failed"),
        (
            "running",
            "final",
            "not_started",
            "Continue with: --resume to proceed with final analysis",
        ),
    ],
)
def test_determine_next_action_covers_core_branches(
    status, stage_name, stage_status, expected
):
    manifest = _make_manifest(status=status)
    if stage_name is not None:
        manifest.stages[stage_name].status = stage_status

    action = _determine_next_action(manifest)

    assert expected in action


def test_render_final_calculates_processing_time_and_token_totals():
    summary = Summary(part_index=0, content="summary", metadata={"token_count": 11})
    stage = StageSummary(
        stage_index=0,
        summaries=[summary],
        synthesis="## Executive Summary\nDone",
        metadata={"token_count": 13},
    )
    final = FinalAnalysis(
        status="completed",
        stages=[stage],
        final_result="## Executive Summary\nAll done",
        metadata={
            "token_count": 17,
            "created_at": "2026-04-10T09:00:00",
            "completed_at": "2026-04-10T09:00:05",
            "models_used": ["model-a", "model-b"],
        },
    )

    output = render_final(final, input_path=r"C:\docs\input.txt", model="model-a")

    assert "**Source File:** [input.txt](C:\\docs\\input.txt)" in output
    assert "**Processing Time:** 5s" in output
    assert "**Tokens Analyzed:** 41" in output
    assert "**Models Used:** model-a, model-b" in output


def test_renderer_helpers_and_section_renderers_cover_dispatch_paths():
    now = datetime(2026, 4, 10, 9, 0, 0)
    stage_info = StageInfo(
        name="summarize",
        status="failed",
        input_file="input.txt",
        output_file="summary_00.md",
        timestamp=now,
        error="bad response",
    )
    summary = Summary(
        part_index=0,
        content=(
            "## Key Points\n- point\n## Entities\n- Alice\n## Themes\n- Risk\n"
            "## Action Items\n- follow up\n## Additional Notes\nnote"
        ),
        metadata={"token_count": 9},
    )
    stage = StageSummary(
        stage_index=0,
        summaries=[summary],
        synthesis=(
            "## Executive Summary\nOverview\n## Consolidated Points\n- point\n"
            "## Entity Synthesis\nAlice\n## Theme Evolution\nRisk\n"
            "## Consistency Checks\nOK\n## Action Items Tracking\nPending"
        ),
        metadata={"token_count": 12},
    )
    manifest = _make_manifest(status="running")

    assert _format_list(["a", "b"]) == "- a\n- b"
    assert _format_list([]) == "(none)"
    assert _format_dict({"a": {"b": 1}}) == "a:\n  b: 1"
    assert _safe_get({"a": {"b": "ok"}}, "a", "b") == "ok"
    assert _safe_get({"a": None}, "a", "b") == "(not available)"
    formatted_stage = _format_stage_info(stage_info)
    assert "Status: failed" in formatted_stage
    assert "Error: bad response" in formatted_stage

    summary_output = render_summary(summary, model="model-a")
    stage_output = render_stage(stage, model="model-a")
    manifest_output = format_output_type("manifest", manifest)
    status_output = format_output_type("status", manifest, show_details=True)

    assert "Summary for Part 00" in summary_output
    assert "Stage Summary 00" in stage_output
    assert "Pipeline Status Report" in manifest_output
    assert "PIPELINE STATUS REPORT" in status_output
    assert format_output_type("summary", "raw summary") == "raw summary"
    assert format_output_type("stage", "raw stage") == "raw stage"
    assert "Unknown output type: unknown" in format_output_type("unknown", "content")
