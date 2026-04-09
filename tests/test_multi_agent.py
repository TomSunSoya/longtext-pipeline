"""Tests for current multi-perspective final analysis behavior."""

from datetime import datetime
from unittest.mock import AsyncMock, call, patch

import pytest

from src.longtext_pipeline.models import FinalAnalysis, Manifest, StageInfo, StageSummary, Summary
from src.longtext_pipeline.pipeline.final_analysis import FinalAnalysisStage


class DummyClient:
    """Minimal async-capable client stub for final-analysis routing tests."""

    def __init__(self, model: str):
        self.model = model

    async def acomplete(self, prompt, system_prompt=None):
        return f"response from {self.model}"


def make_stage_summaries() -> list[StageSummary]:
    """Build stable stage summaries for multi-agent tests."""
    return [
        StageSummary(
            stage_index=0,
            summaries=[
                Summary(part_index=0, content="First summary", metadata={"test": True}),
                Summary(part_index=1, content="Second summary", metadata={"test": True}),
            ],
            synthesis="Stage 0 synthesis",
            metadata={"test": True},
        ),
        StageSummary(
            stage_index=1,
            summaries=[
                Summary(part_index=2, content="Third summary", metadata={"test": True}),
            ],
            synthesis="Stage 1 synthesis",
            metadata={"test": True},
        ),
    ]


def make_manifest(tmp_path) -> Manifest:
    """Build a manifest with all prerequisite stages completed."""
    return Manifest(
        session_id="multi-agent-test",
        input_path=str(tmp_path / "input.txt"),
        input_hash="abc123" * 10,
        stages={
            "ingest": StageInfo(name="ingest", status="successful"),
            "summarize": StageInfo(name="summarize", status="successful"),
            "stage": StageInfo(name="stage", status="successful"),
            "final": StageInfo(name="final", status="not_started"),
        },
        created_at=datetime.now(),
        updated_at=datetime.now(),
        status="in_progress",
        total_parts=3,
        total_stages=2,
    )


def make_config() -> dict:
    """Build a config that matches the current nested model schema."""
    return {
        "model": {
            "provider": "openai",
            "name": "gpt-4o-mini",
            "api_key": "fake-key",
        },
        "agents": {
            "topic_analyst": {"model": {"provider": "openai", "name": "topic-model"}},
            "entity_analyst": {"model": {"provider": "openai", "name": "entity-model"}},
            "sentiment_analyst": {"model": {"provider": "openai", "name": "sentiment-model"}},
            "timeline_analyst": {"model": {"provider": "openai", "name": "timeline-model"}},
            "analyst": {"model": {"provider": "openai", "name": "meta-model"}},
        },
    }


@pytest.mark.asyncio
async def test_multi_perspective_run_records_specialist_metadata(tmp_path):
    """run(..., multi_perspective=True) should execute the current specialist pipeline."""
    stage = FinalAnalysisStage()
    stage_summaries = make_stage_summaries()
    manifest = make_manifest(tmp_path)
    config = make_config()

    async def fake_generate(analyst_type, stage_summaries, prompt_template, client, model):
        return {
            "analyst_type": analyst_type,
            "model_used": model,
            "analysis": f"{analyst_type} analysis",
            "timestamp": "2024-01-01T00:00:00Z",
            "status": "completed",
        }

    def fake_client_factory(config, agent_type=None, **kwargs):
        model_name = config["agents"].get(agent_type, {}).get("model", {}).get("name", agent_type or "default")
        return DummyClient(model_name)

    with patch("src.longtext_pipeline.pipeline.final_analysis.get_llm_client", side_effect=fake_client_factory), \
         patch.object(stage, "_generate_specialist_analysis", side_effect=fake_generate), \
         patch.object(stage, "_aggregate_with_meta_agent", AsyncMock(return_value="Integrated analysis")), \
         patch.object(stage, "_save_final_analysis", return_value=("fake.md", "fake.json")), \
         patch.object(stage.manifest_manager, "update_stage"), \
         patch.object(stage.manifest_manager, "save_manifest"):
        result = await stage.run(
            stage_summaries=stage_summaries,
            config=config,
            manifest=manifest,
            mode="general",
            multi_perspective=True,
        )

    assert result.status == "completed"
    assert result.final_result == "Integrated analysis"
    assert result.metadata["multi_perspective_analysis"] is True
    assert result.metadata["specialist_counts"] == {
        "completed": 4,
        "failed": 0,
        "total_requested": 4,
    }
    assert result.metadata["specialist_success_threshold"] == 3
    assert result.metadata["selected_specialists"] == [
        "topic_analyst",
        "entity_analyst",
        "sentiment_analyst",
        "timeline_analyst",
    ]
    assert result.metadata["specialist_analyses_performed"] == [
        "topic_analyst",
        "entity_analyst",
        "sentiment_analyst",
        "timeline_analyst",
    ]


@pytest.mark.asyncio
async def test_multi_perspective_falls_back_below_success_threshold(tmp_path):
    """When fewer than three specialists succeed, final analysis should fall back to single-pass."""
    stage = FinalAnalysisStage()
    stage_summaries = make_stage_summaries()
    manifest = make_manifest(tmp_path)
    config = make_config()

    results = {
        "topic_analyst": "completed",
        "entity_analyst": "completed",
        "sentiment_analyst": "failed",
        "timeline_analyst": "failed",
    }

    async def fake_generate(analyst_type, stage_summaries, prompt_template, client, model):
        status = results[analyst_type]
        return {
            "analyst_type": analyst_type,
            "model_used": model,
            "analysis": f"{analyst_type} {status}",
            "timestamp": "2024-01-01T00:00:00Z",
            "status": status,
        }

    fallback = FinalAnalysis(
        status="completed",
        stages=stage_summaries,
        final_result="single-pass fallback",
        metadata={"multi_perspective_analysis": False, "fallback_used": True},
    )

    with patch("src.longtext_pipeline.pipeline.final_analysis.get_llm_client", return_value=DummyClient("fallback")), \
         patch.object(stage, "_generate_specialist_analysis", side_effect=fake_generate), \
         patch.object(stage, "_run_single_pass", AsyncMock(return_value=fallback)) as mock_fallback, \
         patch.object(stage, "_save_final_analysis", return_value=("fake.md", "fake.json")), \
         patch.object(stage.manifest_manager, "update_stage"), \
         patch.object(stage.manifest_manager, "save_manifest"):
        result = await stage.run(
            stage_summaries=stage_summaries,
            config=config,
            manifest=manifest,
            mode="general",
            multi_perspective=True,
        )

    mock_fallback.assert_awaited_once()
    assert result.final_result == "single-pass fallback"
    assert result.metadata["fallback_used"] is True


@pytest.mark.asyncio
async def test_multi_perspective_routes_specialist_and_meta_agent_types(tmp_path):
    """Specialist execution should request one client per specialist plus one meta-agent client."""
    stage = FinalAnalysisStage()
    stage_summaries = make_stage_summaries()
    manifest = make_manifest(tmp_path)
    config = make_config()

    requested_agent_types = []

    def fake_client_factory(config, agent_type=None, **kwargs):
        requested_agent_types.append(agent_type)
        model_name = config["agents"].get(agent_type, {}).get("model", {}).get("name", agent_type or "default")
        return DummyClient(model_name)

    async def fake_generate(analyst_type, stage_summaries, prompt_template, client, model):
        return {
            "analyst_type": analyst_type,
            "model_used": model,
            "analysis": f"analysis via {model}",
            "timestamp": "2024-01-01T00:00:00Z",
            "status": "completed",
        }

    with patch("src.longtext_pipeline.pipeline.final_analysis.get_llm_client", side_effect=fake_client_factory), \
         patch.object(stage, "_generate_specialist_analysis", side_effect=fake_generate), \
         patch.object(stage, "_aggregate_with_meta_agent", AsyncMock(return_value="Integrated analysis")):
        result = await stage._run_multi_perspective(
            stage_summaries=stage_summaries,
            config=config,
            manifest=manifest,
            mode="general",
        )

    assert requested_agent_types == [
        "topic_analyst",
        "entity_analyst",
        "sentiment_analyst",
        "timeline_analyst",
        "analyst",
    ]
    assert result.metadata["model"] == "meta-model"


@pytest.mark.asyncio
async def test_multi_perspective_respects_configured_agent_count(tmp_path):
    """Configured specialist_count should limit which specialists run."""
    stage = FinalAnalysisStage()
    stage_summaries = make_stage_summaries()
    manifest = make_manifest(tmp_path)
    config = make_config()
    config["pipeline"] = {"specialist_count": 2}

    requested_agent_types = []

    def fake_client_factory(config, agent_type=None, **kwargs):
        requested_agent_types.append(agent_type)
        model_name = config["agents"].get(agent_type, {}).get("model", {}).get("name", agent_type or "default")
        return DummyClient(model_name)

    async def fake_generate(analyst_type, stage_summaries, prompt_template, client, model):
        return {
            "analyst_type": analyst_type,
            "model_used": model,
            "analysis": f"{analyst_type} analysis",
            "timestamp": "2024-01-01T00:00:00Z",
            "status": "completed",
        }

    with patch("src.longtext_pipeline.pipeline.final_analysis.get_llm_client", side_effect=fake_client_factory), \
         patch.object(stage, "_generate_specialist_analysis", side_effect=fake_generate), \
         patch.object(stage, "_aggregate_with_meta_agent", AsyncMock(return_value="Integrated analysis")):
        result = await stage._run_multi_perspective(
            stage_summaries=stage_summaries,
            config=config,
            manifest=manifest,
            mode="general",
        )

    assert requested_agent_types == [
        "topic_analyst",
        "entity_analyst",
        "analyst",
    ]
    assert result.metadata["selected_specialists"] == [
        "topic_analyst",
        "entity_analyst",
    ]
    assert result.metadata["specialist_counts"]["total_requested"] == 2
    assert result.metadata["specialist_success_threshold"] == 2


@pytest.mark.asyncio
async def test_multi_perspective_threshold_scales_with_agent_count(tmp_path):
    """With two selected specialists, one failure should trigger single-pass fallback."""
    stage = FinalAnalysisStage()
    stage_summaries = make_stage_summaries()
    manifest = make_manifest(tmp_path)
    config = make_config()
    config["pipeline"] = {"specialist_count": 2}

    statuses = {
        "topic_analyst": "completed",
        "entity_analyst": "failed",
    }

    async def fake_generate(analyst_type, stage_summaries, prompt_template, client, model):
        return {
            "analyst_type": analyst_type,
            "model_used": model,
            "analysis": f"{analyst_type} {statuses[analyst_type]}",
            "timestamp": "2024-01-01T00:00:00Z",
            "status": statuses[analyst_type],
        }

    fallback = FinalAnalysis(
        status="completed",
        stages=stage_summaries,
        final_result="single-pass fallback",
        metadata={"fallback_used": True},
    )

    with patch("src.longtext_pipeline.pipeline.final_analysis.get_llm_client", return_value=DummyClient("fallback")), \
         patch.object(stage, "_generate_specialist_analysis", side_effect=fake_generate), \
         patch.object(stage, "_run_single_pass", AsyncMock(return_value=fallback)) as mock_fallback:
        result = await stage._run_multi_perspective(
            stage_summaries=stage_summaries,
            config=config,
            manifest=manifest,
            mode="general",
        )

    mock_fallback.assert_awaited_once()
    assert result.final_result == "single-pass fallback"


def test_multi_agent_stage_exposes_current_private_api_only():
    """Regression coverage for the current internal API shape after the refactor."""
    stage = FinalAnalysisStage()

    assert hasattr(stage, "_generate_specialist_analysis")
    assert hasattr(stage, "_run_multi_perspective")
    assert hasattr(stage, "_run_single_pass")
    assert not hasattr(stage, "run_multi_perspective_analysis")
    assert not hasattr(stage, "_generate_topic_analysis")
    assert not hasattr(stage, "_run_single_pass_analysis")
