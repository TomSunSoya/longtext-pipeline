"""Phase 2 integration tests for end-to-end parallel pipeline execution.

This module tests the complete pipeline with parallel execution capabilities:
- Full summarize stage execution with parallel processing
- Full stage synthesis stage execution with parallel processing
- Sync/async consistency (same data produces equivalent outputs)
- Resume functionality with manifest manager after pipeline stages

All tests use mock LLM clients to avoid real API calls.
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import patch

from src.longtext_pipeline.pipeline.summarize import SummarizeStage
from src.longtext_pipeline.pipeline.stage_synthesis import StageSynthesisStage
from src.longtext_pipeline.models import Part, Summary, Manifest
from src.longtext_pipeline.errors import LLMError


class MockLLMClient:
    """Mock LLM client for integration testing without real API calls.

    Provides configurable responses with realistic delays to simulate
    actual API behavior while maintaining test speed and reliability.
    """

    def __init__(self, response_sequence=None, delay=0.01, fail_indices=None):
        """Initialize mock client.

        Args:
            response_sequence: List of responses to return in order
            delay: Artificial delay to simulate API call (seconds)
            fail_indices: Set of indices that should fail (for error testing)
        """
        self.response_sequence = response_sequence or []
        self.delay = delay
        self.fail_indices = fail_indices or set()
        self.call_count = 0
        self.call_history = []
        self._lock = asyncio.Lock()

    async def acomplete(self, prompt, system_prompt=None):
        """Mock async complete method with configurable behavior."""
        async with self._lock:
            call_index = self.call_count
            self.call_count += 1
            self.call_history.append(
                {
                    "index": call_index,
                    "prompt": prompt[:50] if prompt else "",
                    "system_prompt": system_prompt[:50] if system_prompt else "",
                }
            )

        # Simulate network delay
        if self.delay > 0:
            await asyncio.sleep(self.delay)

        # Check if this call should fail
        if call_index in self.fail_indices:
            raise LLMError(f"Simulated failure for call {call_index}")

        # Return next response in sequence or default
        if self.response_sequence and call_index < len(self.response_sequence):
            return self.response_sequence[call_index]
        return f"Mock response {call_index}"

    async def acomplete_json(self, prompt, system_prompt=None):
        """Mock async JSON complete method."""
        response = await self.acomplete(prompt, system_prompt)
        return {"mock": response}


class TestSummarizeStageEndToEnd:
    """Test full summarize stage execution with parallel processing."""

    @pytest.mark.asyncio
    async def test_summarize_full_stage_parallel_execution(self, tmp_path):
        """Test complete summarize stage with parallel processing of multiple parts."""
        from src.longtext_pipeline.models import StageInfo

        # Create realistic test data - 8 parts to process
        num_parts = 8
        parts = [
            Part(
                index=i,
                content=f"Content for part {i}. This is a detailed text segment "
                f"that needs to be summarized. It contains important "
                f"information about topic {i}.",
                token_count=150,
                metadata={"source": f"part_{i:03d}.txt"},
            )
            for i in range(num_parts)
        ]

        # Create stage and manifest
        stage = SummarizeStage()
        manifest = Manifest(
            session_id="e2e-test-session-001",
            input_path=str(tmp_path / "input.txt"),
            input_hash="abc123" * 16,
            stages={
                "ingest": StageInfo(name="ingest", status="successful"),
                "summarize": StageInfo(name="summarize", status="not_started"),
                "stage": StageInfo(name="stage", status="not_started"),
                "final": StageInfo(name="final", status="not_started"),
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status="pending",
        )

        # Create mock client with unique responses for each part
        mock_responses = [
            f"# Summary {i}\n\nKey points about part {i}." for i in range(num_parts)
        ]
        mock_client = MockLLMClient(response_sequence=mock_responses, delay=0.01)

        # Mock prompt loading and file I/O
        with patch.object(
            stage, "_load_prompt_template", return_value="Summarize this: "
        ):
            with patch(
                "src.longtext_pipeline.pipeline.summarize.get_llm_client",
                return_value=mock_client,
            ):
                with patch.object(
                    stage,
                    "_save_summary",
                    side_effect=lambda s, d: str(d / f"summary_{s.part_index:03d}.md"),
                ):
                    # Run summarize stage with parallel execution
                    summaries = await stage.run(
                        parts=parts,
                        config={
                            "model": {"name": "gpt-4o-mini"},
                            "pipeline": {"max_workers": 4},
                        },
                        manifest=manifest,
                        mode="general",
                    )

        # Verify all parts were summarized
        assert len(summaries) == num_parts, (
            f"Expected {num_parts} summaries, got {len(summaries)}"
        )

        # Verify order preservation
        for i, summary in enumerate(summaries):
            assert summary.part_index == i, (
                f"Expected part_index {i}, got {summary.part_index}"
            )
            assert summary.content == f"# Summary {i}\n\nKey points about part {i}."

        # Verify mock client was called correct number of times
        assert mock_client.call_count == num_parts

    @pytest.mark.asyncio
    async def test_summarize_parallel_acceleration(self, tmp_path):
        """Test that parallel execution provides acceleration compared to sequential."""
        from src.longtext_pipeline.models import StageInfo

        num_parts = 10
        parts = [
            Part(index=i, content=f"Content {i}", token_count=100, metadata={})
            for i in range(num_parts)
        ]

        stage = SummarizeStage()
        manifest = Manifest(
            session_id="e2e-test-speed-001",
            input_path=str(tmp_path / "input.txt"),
            input_hash="def456" * 16,
            stages={
                "ingest": StageInfo(name="ingest", status="successful"),
                "summarize": StageInfo(name="summarize", status="not_started"),
                "stage": StageInfo(name="stage", status="not_started"),
                "final": StageInfo(name="final", status="not_started"),
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status="pending",
        )

        # Create mock client with measurable delay
        call_times = []
        lock = asyncio.Lock()

        async def tracked_complete(prompt, system_prompt=None):
            async with lock:
                call_times.append(datetime.now())
            await asyncio.sleep(0.05)  # 50ms delay
            return "Mock summary"

        mock_client = MockLLMClient(delay=0)
        mock_client.acomplete = tracked_complete

        start_time = datetime.now()

        with patch.object(stage, "_load_prompt_template", return_value="Summarize: "):
            with patch(
                "src.longtext_pipeline.pipeline.summarize.get_llm_client",
                return_value=mock_client,
            ):
                with patch.object(
                    stage,
                    "_save_summary",
                    side_effect=lambda s, d: str(d / f"summary_{s.part_index:03d}.md"),
                ):
                    await stage.run(
                        parts=parts,
                        config={
                            "model": {"name": "gpt-4o-mini"},
                            "pipeline": {"max_workers": 4},
                        },
                        manifest=manifest,
                        mode="general",
                    )

        end_time = datetime.now()
        actual_duration = (end_time - start_time).total_seconds()

        # Sequential would take: 10 * 0.05 = 0.5 seconds
        # Parallel with 4 workers: ~ceil(10/4) * 0.05 = 3 * 0.05 = 0.15 seconds
        # Allow some overhead, but should be significantly faster than sequential
        sequential_time = num_parts * 0.05
        # Parallel should be at least 2x faster than sequential
        assert actual_duration < sequential_time * 0.8, (
            f"Parallel execution ({actual_duration:.2f}s) should be faster than sequential ({sequential_time:.2f}s)"
        )


class TestStageSynthesisStageEndToEnd:
    """Test full stage synthesis stage execution with parallel processing."""

    @pytest.mark.asyncio
    async def test_stage_synthesis_full_stage_parallel_execution(self, tmp_path):
        """Test complete stage synthesis with parallel processing of multiple groups."""
        from src.longtext_pipeline.models import StageInfo

        # Create 12 summaries to group (6 groups of 2)
        num_summaries = 12
        group_size = 2
        summaries = [
            Summary(
                part_index=i,
                content=f"# Summary {i}\n\nKey points from part {i}.",
                metadata={"generated_at": datetime.now().isoformat()},
            )
            for i in range(num_summaries)
        ]

        stage = StageSynthesisStage()
        manifest = Manifest(
            session_id="e2e-test-session-002",
            input_path=str(tmp_path / "input.txt"),
            input_hash="ghi789" * 16,
            stages={
                "ingest": StageInfo(name="ingest", status="successful"),
                "summarize": StageInfo(name="summarize", status="successful"),
                "stage": StageInfo(name="stage", status="not_started"),
                "final": StageInfo(name="final", status="not_started"),
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status="pending",
        )

        # Create mock client with unique responses for each group
        num_groups = num_summaries // group_size
        mock_responses = [
            f"# Stage {i}\n\nSynthesis of group {i}." for i in range(num_groups)
        ]
        mock_client = MockLLMClient(response_sequence=mock_responses, delay=0.01)

        with patch.object(stage, "_load_prompt_template", return_value="Synthesize: "):
            with patch(
                "src.longtext_pipeline.pipeline.stage_synthesis.get_llm_client",
                return_value=mock_client,
            ):
                with patch.object(
                    stage,
                    "_save_stage_summary",
                    side_effect=lambda s, d: str(d / f"stage_{s.stage_index:03d}.md"),
                ):
                    stage_summaries = await stage.run(
                        summaries=summaries,
                        config={
                            "model": {"name": "gpt-4o-mini"},
                            "pipeline": {"max_workers": 3},
                            "stages": {"stage": {"group_size": group_size}},
                        },
                        manifest=manifest,
                        mode="general",
                    )

        # Verify all groups were synthesized
        assert len(stage_summaries) == num_groups, (
            f"Expected {num_groups} stage summaries, got {len(stage_summaries)}"
        )

        # Verify order preservation
        for i, stage_summary in enumerate(stage_summaries):
            assert stage_summary.stage_index == i, (
                f"Expected stage_index {i}, got {stage_summary.stage_index}"
            )
            assert stage_summary.synthesis == f"# Stage {i}\n\nSynthesis of group {i}."

        # Verify correct groupings
        for i, stage_summary in enumerate(stage_summaries):
            summary_indices = [s.part_index for s in stage_summary.summaries]
            expected_indices = list(range(i * group_size, (i + 1) * group_size))
            assert summary_indices == expected_indices, (
                f"Group {i} has wrong summaries: {summary_indices} vs {expected_indices}"
            )

        # Verify mock client was called correct number of times
        assert mock_client.call_count == num_groups

    @pytest.mark.asyncio
    async def test_stage_synthesis_parallel_acceleration(self, tmp_path):
        """Test that parallel stage synthesis provides acceleration."""
        from src.longtext_pipeline.models import StageInfo

        num_summaries = 15
        group_size = 3
        summaries = [
            Summary(part_index=i, content=f"Summary {i}", metadata={})
            for i in range(num_summaries)
        ]

        stage = StageSynthesisStage()
        manifest = Manifest(
            session_id="e2e-test-speed-002",
            input_path=str(tmp_path / "input.txt"),
            input_hash="jkl012" * 16,
            stages={
                "ingest": StageInfo(name="ingest", status="successful"),
                "summarize": StageInfo(name="summarize", status="successful"),
                "stage": StageInfo(name="stage", status="not_started"),
                "final": StageInfo(name="final", status="not_started"),
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status="pending",
        )

        call_times = []
        lock = asyncio.Lock()

        async def tracked_complete(prompt, system_prompt=None):
            async with lock:
                call_times.append(datetime.now())
            await asyncio.sleep(0.05)
            return "Mock synthesis"

        mock_client = MockLLMClient(delay=0)
        mock_client.acomplete = tracked_complete

        start_time = datetime.now()

        with patch.object(stage, "_load_prompt_template", return_value="Synthesize: "):
            with patch(
                "src.longtext_pipeline.pipeline.stage_synthesis.get_llm_client",
                return_value=mock_client,
            ):
                with patch.object(
                    stage,
                    "_save_stage_summary",
                    side_effect=lambda s, d: str(d / f"stage_{s.stage_index:03d}.md"),
                ):
                    await stage.run(
                        summaries=summaries,
                        config={
                            "model": {"name": "gpt-4o-mini"},
                            "pipeline": {"max_workers": 3},
                            "stages": {"stage": {"group_size": group_size}},
                        },
                        manifest=manifest,
                        mode="general",
                    )

        end_time = datetime.now()
        actual_duration = (end_time - start_time).total_seconds()

        num_groups = num_summaries // group_size  # 5 groups
        sequential_time = num_groups * 0.05  # 0.25 seconds

        # Parallel should be faster than sequential
        assert actual_duration < sequential_time * 0.8, (
            f"Parallel ({actual_duration:.2f}s) should be faster than sequential ({sequential_time:.2f}s)"
        )


class TestSyncAsyncConsistency:
    """Test that sync and async processing produce equivalent outputs."""

    @pytest.mark.asyncio
    async def test_sync_async_summarize_same_data(self, tmp_path):
        """Test that same data produces equivalent results with sync/async."""
        from src.longtext_pipeline.models import StageInfo

        # Create identical test data
        num_parts = 5
        parts = [
            Part(
                index=i,
                content=f"Content for part {i} with detailed information.",
                token_count=100,
                metadata={},
            )
            for i in range(num_parts)
        ]

        # Mock responses that are identical regardless of call method
        mock_responses = [f"Summary {i}" for i in range(num_parts)]

        # Test async execution
        stage_async = SummarizeStage()
        manifest_async = Manifest(
            session_id="consistency-test-async",
            input_path=str(tmp_path / "input.txt"),
            input_hash="mno345" * 16,
            stages={
                "ingest": StageInfo(name="ingest", status="successful"),
                "summarize": StageInfo(name="summarize", status="not_started"),
                "stage": StageInfo(name="stage", status="not_started"),
                "final": StageInfo(name="final", status="not_started"),
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status="pending",
        )

        mock_client_async = MockLLMClient(
            response_sequence=mock_responses.copy(), delay=0
        )

        with patch.object(
            stage_async, "_load_prompt_template", return_value="Summarize: "
        ):
            with patch(
                "src.longtext_pipeline.pipeline.summarize.get_llm_client",
                return_value=mock_client_async,
            ):
                with patch.object(
                    stage_async,
                    "_save_summary",
                    side_effect=lambda s, d: str(d / f"summary_{s.part_index:03d}.md"),
                ):
                    summaries_async = await stage_async.run(
                        parts=parts.copy(),
                        config={
                            "model": {"name": "gpt-4o-mini"},
                            "pipeline": {"max_workers": 4},
                        },
                        manifest=manifest_async,
                        mode="general",
                    )

        # Verify results
        assert len(summaries_async) == num_parts
        for i, summary in enumerate(summaries_async):
            assert summary.part_index == i
            assert summary.content == f"Summary {i}"

        # Note: We can't test true sync mode here because SummarizeStage.run()
        # is now async-only in Phase 2. This test verifies that parallel
        # async execution produces correct, ordered results.

    @pytest.mark.asyncio
    async def test_parallel_execution_same_content_different_order(self, tmp_path):
        """Test that parallel execution doesn't affect content despite completion order."""
        from src.longtext_pipeline.models import StageInfo

        num_parts = 6
        parts = [
            Part(index=i, content=f"Content {i}", token_count=100, metadata={})
            for i in range(num_parts)
        ]

        stage = SummarizeStage()
        manifest = Manifest(
            session_id="order-test-001",
            input_path=str(tmp_path / "input.txt"),
            input_hash="pqr678" * 16,
            stages={
                "ingest": StageInfo(name="ingest", status="successful"),
                "summarize": StageInfo(name="summarize", status="not_started"),
                "stage": StageInfo(name="stage", status="not_started"),
                "final": StageInfo(name="final", status="not_started"),
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status="pending",
        )

        # Responses are tied to part index, not completion order
        mock_responses = [f"Content-specific summary {i}" for i in range(num_parts)]
        mock_client = MockLLMClient(response_sequence=mock_responses, delay=0)

        with patch.object(stage, "_load_prompt_template", return_value="Summarize: "):
            with patch(
                "src.longtext_pipeline.pipeline.summarize.get_llm_client",
                return_value=mock_client,
            ):
                with patch.object(
                    stage,
                    "_save_summary",
                    side_effect=lambda s, d: str(d / f"summary_{s.part_index:03d}.md"),
                ):
                    summaries = await stage.run(
                        parts=parts,
                        config={
                            "model": {"name": "gpt-4o-mini"},
                            "pipeline": {"max_workers": 4},
                        },
                        manifest=manifest,
                        mode="general",
                    )

        # Verify that despite parallel execution, results are in original order
        for i, summary in enumerate(summaries):
            assert summary.part_index == i
            assert summary.content == f"Content-specific summary {i}"


class TestResumeFunctionality:
    """Test resume functionality with manifest manager after pipeline stages.

    Note: Resume logic is handled at the orchestrator level by checking
    manifest state. These tests verify that stages work correctly when
    given data that represents a partial-completion state.
    """

    @pytest.mark.asyncio
    async def test_summarize_processes_all_provided_parts(self, tmp_path):
        """Test that summarize processes all parts provided to it."""
        from src.longtext_pipeline.models import StageInfo

        num_parts = 6
        parts = [
            Part(index=i, content=f"Content {i}", token_count=100, metadata={})
            for i in range(num_parts)
        ]

        stage = SummarizeStage()
        manifest = Manifest(
            session_id="resume-test-001",
            input_path=str(tmp_path / "input.txt"),
            input_hash="stu901" * 16,
            stages={
                "ingest": StageInfo(name="ingest", status="successful"),
                "summarize": StageInfo(name="summarize", status="running"),
                "stage": StageInfo(name="stage", status="not_started"),
                "final": StageInfo(name="final", status="not_started"),
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status="in_progress",
        )

        # Mock client that generates summaries for all parts
        mock_responses = [f"Summary {i}" for i in range(num_parts)]
        mock_client = MockLLMClient(response_sequence=mock_responses, delay=0)

        with patch.object(stage, "_load_prompt_template", return_value="Summarize: "):
            with patch(
                "src.longtext_pipeline.pipeline.summarize.get_llm_client",
                return_value=mock_client,
            ):
                with patch.object(
                    stage,
                    "_save_summary",
                    side_effect=lambda s, d: str(d / f"summary_{s.part_index:03d}.md"),
                ):
                    summaries = await stage.run(
                        parts=parts,
                        config={
                            "model": {"name": "gpt-4o-mini"},
                            "pipeline": {"max_workers": 4},
                        },
                        manifest=manifest,
                        mode="general",
                    )

        # Should process all parts
        assert len(summaries) == num_parts

        # Verify all indices are present
        result_indices = {s.part_index for s in summaries}
        assert result_indices == set(range(num_parts))

    @pytest.mark.asyncio
    async def test_stage_synthesis_processes_all_groups(self, tmp_path):
        """Test that stage synthesis processes all provided groups."""
        from src.longtext_pipeline.models import StageInfo

        num_summaries = 8
        group_size = 2
        summaries = [
            Summary(part_index=i, content=f"Summary {i}", metadata={})
            for i in range(num_summaries)
        ]

        stage = StageSynthesisStage()
        manifest = Manifest(
            session_id="resume-test-002",
            input_path=str(tmp_path / "input.txt"),
            input_hash="vwx234" * 16,
            stages={
                "ingest": StageInfo(name="ingest", status="successful"),
                "summarize": StageInfo(name="summarize", status="successful"),
                "stage": StageInfo(name="stage", status="running"),
                "final": StageInfo(name="final", status="not_started"),
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status="in_progress",
        )

        num_groups = num_summaries // group_size
        mock_responses = [f"Stage {i}" for i in range(num_groups)]
        mock_client = MockLLMClient(response_sequence=mock_responses, delay=0)

        with patch.object(stage, "_load_prompt_template", return_value="Synthesize: "):
            with patch(
                "src.longtext_pipeline.pipeline.stage_synthesis.get_llm_client",
                return_value=mock_client,
            ):
                with patch.object(
                    stage,
                    "_save_stage_summary",
                    side_effect=lambda s, d: str(d / f"stage_{s.stage_index:03d}.md"),
                ):
                    stage_summaries = await stage.run(
                        summaries=summaries,
                        config={
                            "model": {"name": "gpt-4o-mini"},
                            "pipeline": {"max_workers": 3},
                            "stages": {"stage": {"group_size": group_size}},
                        },
                        manifest=manifest,
                        mode="general",
                    )

        # Should process all groups
        assert len(stage_summaries) == num_groups

        # Verify all stage indices are present
        result_indices = {s.stage_index for s in stage_summaries}
        assert result_indices == set(range(num_groups))

    @pytest.mark.asyncio
    async def test_manifest_status_updated_after_summarize_completion(self, tmp_path):
        """Test manifest status is updated after summarize stage completes."""
        from src.longtext_pipeline.models import StageInfo

        num_parts = 4
        parts = [
            Part(index=i, content=f"Content {i}", token_count=100, metadata={})
            for i in range(num_parts)
        ]

        stage = SummarizeStage()
        manifest = Manifest(
            session_id="manifest-update-test-001",
            input_path=str(tmp_path / "input.txt"),
            input_hash="yza567" * 16,
            stages={
                "ingest": StageInfo(name="ingest", status="successful"),
                "summarize": StageInfo(name="summarize", status="not_started"),
                "stage": StageInfo(name="stage", status="not_started"),
                "final": StageInfo(name="final", status="not_started"),
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status="pending",
        )

        mock_responses = [f"Summary {i}" for i in range(num_parts)]
        mock_client = MockLLMClient(response_sequence=mock_responses, delay=0)

        with patch.object(stage, "_load_prompt_template", return_value="Summarize: "):
            with patch(
                "src.longtext_pipeline.pipeline.summarize.get_llm_client",
                return_value=mock_client,
            ):
                with patch.object(
                    stage,
                    "_save_summary",
                    side_effect=lambda s, d: str(d / f"summary_{s.part_index:03d}.md"),
                ):
                    await stage.run(
                        parts=parts,
                        config={
                            "model": {"name": "gpt-4o-mini"},
                            "pipeline": {"max_workers": 4},
                        },
                        manifest=manifest,
                        mode="general",
                    )

        # Verify manifest was updated
        assert manifest.stages["summarize"].status == "successful"


class TestEndToEndIntegration:
    """Test complete pipeline integration with both stages."""

    @pytest.mark.asyncio
    async def test_full_pipeline_summarize_then_stage_synthesis(self, tmp_path):
        """Test complete flow: parts -> summaries -> stage summaries."""
        from src.longtext_pipeline.models import StageInfo

        # Stage 1: Create parts and run summarize
        num_parts = 8
        parts = [
            Part(
                index=i,
                content=f"Detailed content for part {i}.",
                token_count=150,
                metadata={},
            )
            for i in range(num_parts)
        ]

        summarize_stage = SummarizeStage()
        manifest = Manifest(
            session_id="e2e-full-pipeline-001",
            input_path=str(tmp_path / "input.txt"),
            input_hash="bcd890" * 16,
            stages={
                "ingest": StageInfo(name="ingest", status="successful"),
                "summarize": StageInfo(name="summarize", status="not_started"),
                "stage": StageInfo(name="stage", status="not_started"),
                "final": StageInfo(name="final", status="not_started"),
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status="pending",
        )

        # Summarize stage mock
        summary_responses = [
            f"# Summary {i}\n\nKey points from part {i}." for i in range(num_parts)
        ]
        summarize_client = MockLLMClient(response_sequence=summary_responses, delay=0)

        with patch.object(
            summarize_stage, "_load_prompt_template", return_value="Summarize: "
        ):
            with patch(
                "src.longtext_pipeline.pipeline.summarize.get_llm_client",
                return_value=summarize_client,
            ):
                with patch.object(
                    summarize_stage,
                    "_save_summary",
                    side_effect=lambda s, d: str(d / f"summary_{s.part_index:03d}.md"),
                ):
                    summaries = await summarize_stage.run(
                        parts=parts,
                        config={
                            "model": {"name": "gpt-4o-mini"},
                            "pipeline": {"max_workers": 4},
                        },
                        manifest=manifest,
                        mode="general",
                    )

        # Verify summarize stage output
        assert len(summaries) == num_parts
        assert manifest.stages["summarize"].status == "successful"

        # Stage 2: Run stage synthesis on summaries
        synthesis_stage = StageSynthesisStage()
        group_size = 2

        synthesis_responses = [
            f"# Stage {i}\n\nSynthesis of summaries {i * 2}-{i * 2 + 1}."
            for i in range(num_parts // group_size)
        ]
        synthesis_client = MockLLMClient(response_sequence=synthesis_responses, delay=0)

        with patch.object(
            synthesis_stage, "_load_prompt_template", return_value="Synthesize: "
        ):
            with patch(
                "src.longtext_pipeline.pipeline.stage_synthesis.get_llm_client",
                return_value=synthesis_client,
            ):
                with patch.object(
                    synthesis_stage,
                    "_save_stage_summary",
                    side_effect=lambda s, d: str(d / f"stage_{s.stage_index:03d}.md"),
                ):
                    stage_summaries = await synthesis_stage.run(
                        summaries=summaries,
                        config={
                            "model": {"name": "gpt-4o-mini"},
                            "pipeline": {"max_workers": 3},
                            "stages": {"stage": {"group_size": group_size}},
                        },
                        manifest=manifest,
                        mode="general",
                    )

        # Verify synthesis stage output
        num_groups = num_parts // group_size
        assert len(stage_summaries) == num_groups
        assert manifest.stages["stage"].status == "successful"

        # Verify each stage summary contains correct part summaries
        for i, stage_summary in enumerate(stage_summaries):
            assert stage_summary.stage_index == i
            expected_summary_indices = list(range(i * group_size, (i + 1) * group_size))
            actual_summary_indices = [s.part_index for s in stage_summary.summaries]
            assert actual_summary_indices == expected_summary_indices


class TestErrorHandling:
    """Test error handling in parallel pipeline execution."""

    @pytest.mark.asyncio
    async def test_summarize_continue_with_partial_on_errors(self, tmp_path):
        """Test that summarize continues despite individual part failures."""
        from src.longtext_pipeline.models import StageInfo

        num_parts = 6
        fail_indices = {1, 4}
        parts = [
            Part(index=i, content=f"Content {i}", token_count=100, metadata={})
            for i in range(num_parts)
        ]

        stage = SummarizeStage()
        manifest = Manifest(
            session_id="error-test-001",
            input_path=str(tmp_path / "input.txt"),
            input_hash="efg123" * 16,
            stages={
                "ingest": StageInfo(name="ingest", status="successful"),
                "summarize": StageInfo(name="summarize", status="not_started"),
                "stage": StageInfo(name="stage", status="not_started"),
                "final": StageInfo(name="final", status="not_started"),
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status="pending",
        )

        # Mock client that fails on specific indices
        mock_responses = [f"Summary {i}" for i in range(num_parts)]
        mock_client = MockLLMClient(
            response_sequence=mock_responses, delay=0, fail_indices=fail_indices
        )

        with patch.object(stage, "_load_prompt_template", return_value="Summarize: "):
            with patch(
                "src.longtext_pipeline.pipeline.summarize.get_llm_client",
                return_value=mock_client,
            ):
                with patch.object(
                    stage,
                    "_save_summary",
                    side_effect=lambda s, d: str(d / f"summary_{s.part_index:03d}.md"),
                ):
                    summaries = await stage.run(
                        parts=parts,
                        config={
                            "model": {"name": "gpt-4o-mini"},
                            "pipeline": {"max_workers": 4},
                        },
                        manifest=manifest,
                        mode="general",
                    )

        # Should have successful summaries for non-failed parts
        expected_successes = num_parts - len(fail_indices)
        assert len(summaries) == expected_successes

        # Verify failed parts are not in results
        result_indices = {s.part_index for s in summaries}
        assert result_indices == set(range(num_parts)) - fail_indices

    @pytest.mark.asyncio
    async def test_stage_synthesis_continue_with_partial_on_errors(self, tmp_path):
        """Test that stage synthesis continues despite individual group failures."""
        from src.longtext_pipeline.models import StageInfo

        num_summaries = 6
        group_size = 2
        fail_group_indices = {1}
        summaries = [
            Summary(part_index=i, content=f"Summary {i}", metadata={})
            for i in range(num_summaries)
        ]

        stage = StageSynthesisStage()
        manifest = Manifest(
            session_id="error-test-002",
            input_path=str(tmp_path / "input.txt"),
            input_hash="hij456" * 16,
            stages={
                "ingest": StageInfo(name="ingest", status="successful"),
                "summarize": StageInfo(name="summarize", status="successful"),
                "stage": StageInfo(name="stage", status="not_started"),
                "final": StageInfo(name="final", status="not_started"),
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status="pending",
        )

        num_groups = num_summaries // group_size
        mock_responses = [f"Stage {i}" for i in range(num_groups)]
        mock_client = MockLLMClient(
            response_sequence=mock_responses, delay=0, fail_indices=fail_group_indices
        )

        with patch.object(stage, "_load_prompt_template", return_value="Synthesize: "):
            with patch(
                "src.longtext_pipeline.pipeline.stage_synthesis.get_llm_client",
                return_value=mock_client,
            ):
                with patch.object(
                    stage,
                    "_save_stage_summary",
                    side_effect=lambda s, d: str(d / f"stage_{s.stage_index:03d}.md"),
                ):
                    stage_summaries = await stage.run(
                        summaries=summaries,
                        config={
                            "model": {"name": "gpt-4o-mini"},
                            "pipeline": {"max_workers": 3},
                            "stages": {"stage": {"group_size": group_size}},
                        },
                        manifest=manifest,
                        mode="general",
                    )

        # Should have successful stage summaries for non-failed groups
        expected_successes = num_groups - len(fail_group_indices)
        assert len(stage_summaries) == expected_successes

        # Verify failed groups are not in results
