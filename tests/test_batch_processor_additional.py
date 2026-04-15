"""Additional unit coverage for batch processor edge paths."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from src.longtext_pipeline.utils.batch_processor import BatchProcessor
from src.longtext_pipeline.utils.batch_progress import ProgressReporter, ProgressTracker


def test_batch_processor_init_clamps_workers_and_preserves_manifest_manager():
    manifest_manager = object()
    processor = BatchProcessor(
        parallel=True, batch_max_workers=0, manifest_manager=manifest_manager
    )

    assert processor.parallel is True
    assert processor.batch_max_workers == 1
    assert processor.manifest_manager is manifest_manager


def test_process_single_file_treats_completed_with_issues_as_success(
    tmp_path, monkeypatch
):
    input_file = tmp_path / "input.txt"
    input_file.write_text("hello")

    class _Pipeline:
        def run(self, **kwargs):
            return SimpleNamespace(status="completed_with_issues")

    monkeypatch.setattr(
        "src.longtext_pipeline.pipeline.orchestrator.LongtextPipeline",
        _Pipeline,
    )

    processor = BatchProcessor()
    result = processor._process_single_file(str(input_file), {"resume": False})

    assert result["success"] is True
    assert result["status"] == "completed_with_issues"
    assert result["manifest_path"].endswith(".longtext\\manifest.json") or result[
        "manifest_path"
    ].endswith(".longtext/manifest.json")


def test_process_single_file_handles_keyboard_interrupt(tmp_path, monkeypatch):
    input_file = tmp_path / "input.txt"
    input_file.write_text("hello")

    class _Pipeline:
        def run(self, **kwargs):
            raise KeyboardInterrupt()

    monkeypatch.setattr(
        "src.longtext_pipeline.pipeline.orchestrator.LongtextPipeline",
        _Pipeline,
    )

    processor = BatchProcessor()
    result = processor._process_single_file(str(input_file), {"resume": False})

    assert result["success"] is False
    assert result["status"] == "interrupted"


def test_process_single_file_handles_unexpected_exception(tmp_path, monkeypatch):
    input_file = tmp_path / "input.txt"
    input_file.write_text("hello")

    class _Pipeline:
        def run(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "src.longtext_pipeline.pipeline.orchestrator.LongtextPipeline",
        _Pipeline,
    )

    processor = BatchProcessor()
    result = processor._process_single_file(str(input_file), {"resume": False})

    assert result["success"] is False
    assert result["status"] == "failed"
    assert result["error"] == "boom"


def test_run_parallel_updates_progress_reporter_and_tracker(tmp_path, monkeypatch):
    input_a = tmp_path / "a.txt"
    input_b = tmp_path / "b.txt"
    input_a.write_text("a")
    input_b.write_text("b")

    progress_events = []
    reporter = ProgressReporter(
        total_files=2,
        output_callback=lambda report: progress_events.append(report.to_dict()),
    )
    tracker = ProgressTracker(tmp_path / "progress.json")

    processor = BatchProcessor(parallel=True, batch_max_workers=2)
    monkeypatch.setattr(
        processor,
        "_process_single_file",
        lambda file_path, config: {
            "file": file_path,
            "success": True,
            "status": "completed",
            "error": None,
            "manifest_path": None,
        },
    )

    results = asyncio.run(
        processor._run_parallel(
            [str(input_a), str(input_b)],
            {},
            progress_reporter=reporter,
            progress_tracker=tracker,
        )
    )

    report = tracker.get_current_report()
    assert len(results) == 2
    assert report is not None
    assert report.total_files == 2
    assert report.processed_files == 2
    assert progress_events
