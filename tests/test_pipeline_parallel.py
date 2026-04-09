"""Tests for parallel pipeline execution correctness.

This module tests the parallel execution behavior of pipeline stages:
- Order preservation when processing parts/summaries concurrently
- Semaphore concurrency limits (max_workers enforcement)
- Error isolation (one failure doesn't affect others)
- Uses mock LLM clients to prevent real API calls
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from datetime import datetime

from src.longtext_pipeline.pipeline.summarize import SummarizeStage
from src.longtext_pipeline.pipeline.stage_synthesis import StageSynthesisStage
from src.longtext_pipeline.models import Part, Summary, StageSummary, Manifest
from src.longtext_pipeline.llm.factory import get_llm_client
from src.longtext_pipeline.manifest import ManifestManager


class MockLLMClient:
    """Mock LLM client for testing parallel execution without real API calls."""
    
    def __init__(self, response_sequence=None, delay=0.01, fail_indices=None):
        """Initialize mock client.
        
        Args:
            response_sequence: List of responses to return in order
            delay: Artificial delay to simulate API call (seconds)
            fail_indices: Set of indices that should fail (for error isolation tests)
        """
        self.response_sequence = response_sequence or []
        self.delay = delay
        self.fail_indices = fail_indices or set()
        self.call_count = 0
        self.call_order = []
        self._lock = asyncio.Lock()
    
    async def acomplete(self, prompt, system_prompt=None):
        """Mock async complete method with configurable behavior."""
        async with self._lock:
            call_index = self.call_count
            self.call_count += 1
            self.call_order.append(call_index)
        
        # Simulate network delay
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        
        # Check if this call should fail
        if call_index in self.fail_indices:
            from src.longtext_pipeline.errors import LLMError
            raise LLMError(f"Simulated failure for call {call_index}")
        
        # Return next response in sequence or default
        if self.response_sequence and call_index < len(self.response_sequence):
            return self.response_sequence[call_index]
        return f"Mock response {call_index}"
    
    async def acomplete_json(self, prompt, system_prompt=None):
        """Mock async JSON complete method."""
        response = await self.acomplete(prompt, system_prompt)
        return {"mock": response}


class TestOrderPreservation:
    """Test that parallel execution preserves input order."""
    
    @pytest.mark.asyncio
    async def test_summarize_order_preservation_with_variable_delays(self):
        """Test that summaries maintain original part order despite variable processing times."""
        from src.longtext_pipeline.models import StageInfo
        
        # Create parts with different indices
        parts = [
            Part(index=i, content=f"Content for part {i}", token_count=100, metadata={})
            for i in range(5)
        ]
        
        # Create mock client - each call returns response based on call index
        mock_client = MockLLMClient(response_sequence=[f"Summary {i}" for i in range(5)], delay=0)
        
        # Create stage and manifest
        stage = SummarizeStage()
        manifest = Manifest(
            session_id="test-session-001",
            input_path="test_input.txt",
            input_hash="a" * 64,
            stages={
                "ingest": StageInfo(name="ingest", status="successful"),
                "summarize": StageInfo(name="summarize", status="not_started"),
                "stage": StageInfo(name="stage", status="not_started"),
                "final": StageInfo(name="final", status="not_started")
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status="pending"
        )
        
        # Track call order
        call_order = []
        original_complete = mock_client.acomplete
        
        async def tracked_complete(prompt, system_prompt=None):
            # Extract part index from prompt
            for i in range(5):
                if f"part {i}" in prompt:
                    call_order.append(i)
                    break
            return await original_complete(prompt, system_prompt)
        
        mock_client.acomplete = tracked_complete
        
        # Mock the LLM client factory and prompt loading
        with patch.object(stage, '_load_prompt_template', return_value="Summarize: "):
            with patch('src.longtext_pipeline.pipeline.summarize.get_llm_client', return_value=mock_client):
                with patch.object(stage, '_save_summary', side_effect=lambda s, d: f"summary_{s.part_index:02d}.md"):
                    summaries = await stage.run(
                        parts=parts,
                        config={'model': {'name': 'mock'}, 'pipeline': {'max_workers': 4}},
                        manifest=manifest,
                        mode='general'
                    )
        
        # Verify order is preserved - summaries should be in original part order
        assert len(summaries) == 5
        for i, summary in enumerate(summaries):
            assert summary.part_index == i, f"Expected part_index {i} but got {summary.part_index}"
        
        # Verify content is correct for each part
        assert summaries[0].content == "Summary 0"
        assert summaries[1].content == "Summary 1"
        assert summaries[2].content == "Summary 2"
        assert summaries[3].content == "Summary 3"
        assert summaries[4].content == "Summary 4"
    
    @pytest.mark.asyncio
    async def test_stage_synthesis_order_preservation(self):
        """Test that stage summaries maintain group order despite concurrent execution."""
        from src.longtext_pipeline.models import StageInfo
        
        # Create summaries to group
        summaries = [
            Summary(
                part_index=i,
                content=f"Summary content {i}",
                metadata={"generated_at": datetime.now().isoformat()}
            )
            for i in range(6)
        ]
        
        # Create stage and manifest
        stage = StageSynthesisStage()
        manifest = Manifest(
            session_id="test-session-002",
            input_path="test_input.txt",
            input_hash="b" * 64,
            stages={
                "ingest": StageInfo(name="ingest", status="successful"),
                "summarize": StageInfo(name="summarize", status="successful"),
                "stage": StageInfo(name="stage", status="not_started"),
                "final": StageInfo(name="final", status="not_started")
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status="pending"
        )
        
        # Mock client
        mock_client = MockLLMClient(delay=0)
        
        # Mock prompt loading
        with patch.object(stage, '_load_prompt_template', return_value="Synthesize: "):
            with patch('src.longtext_pipeline.pipeline.stage_synthesis.get_llm_client', return_value=mock_client):
                with patch.object(stage, '_save_stage_summary', side_effect=lambda s, d: f"stage_{s.stage_index:02d}.md"):
                    # Run with concurrent execution
                    stage_summaries = await stage.run(
                        summaries=summaries,
                        config={
                            'model': {'name': 'mock'},
                            'pipeline': {'max_workers': 3},
                            'stages': {'stage': {'group_size': 2}}
                        },
                        manifest=manifest,
                        mode='general'
                    )
        
        # Should have 3 stage summaries (6 summaries / 2 per group)
        assert len(stage_summaries) == 3
        
        # Verify order is preserved - stage summaries should be in group order
        for i, stage_summary in enumerate(stage_summaries):
            assert stage_summary.stage_index == i, f"Expected stage_index {i} but got {stage_summary.stage_index}"
        
        # Verify correct groupings
        assert stage_summaries[0].summaries[0].part_index == 0
        assert stage_summaries[0].summaries[1].part_index == 1
        assert stage_summaries[1].summaries[0].part_index == 2
        assert stage_summaries[1].summaries[1].part_index == 3
        assert stage_summaries[2].summaries[0].part_index == 4
        assert stage_summaries[2].summaries[1].part_index == 5


class TestSemaphoreConcurrency:
    """Test that Semaphore correctly limits concurrent execution."""
    
    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrent_execution(self):
        """Test that no more than max_workers execute concurrently."""
        max_workers = 2
        concurrent_count = 0
        max_concurrent_observed = 0
        lock = asyncio.Lock()
        
        # Track concurrency
        async def track_concurrency():
            nonlocal concurrent_count, max_concurrent_observed
            async with lock:
                concurrent_count += 1
                if concurrent_count > max_concurrent_observed:
                    max_concurrent_observed = concurrent_count
            
            await asyncio.sleep(0.05)  # Simulate work
            
            async with lock:
                concurrent_count -= 1
        
        # Create semaphore and tasks
        semaphore = asyncio.Semaphore(max_workers)
        
        async def worker():
            async with semaphore:
                await track_concurrency()
        
        # Run 5 concurrent tasks
        tasks = [worker() for _ in range(5)]
        await asyncio.gather(*tasks)
        
        # Verify max concurrent never exceeded limit
        assert max_concurrent_observed <= max_workers, \
            f"Observed {max_concurrent_observed} concurrent, expected <= {max_workers}"
        assert max_concurrent_observed == max_workers, \
            f"Expected exactly {max_workers} concurrent at peak, got {max_concurrent_observed}"
    
    @pytest.mark.asyncio
    async def test_summarize_respects_max_workers_config(self):
        """Test that SummarizeStage respects max_workers from config."""
        from src.longtext_pipeline.models import StageInfo
        
        max_workers = 2
        concurrent_count = 0
        max_concurrent_observed = 0
        lock = asyncio.Lock()
        
        parts = [
            Part(index=i, content=f"Content {i}", token_count=100, metadata={})
            for i in range(5)
        ]
        
        stage = SummarizeStage()
        manifest = Manifest(
            session_id="test-session-003",
            input_path="test_input.txt",
            input_hash="c" * 64,
            stages={
                "ingest": StageInfo(name="ingest", status="successful"),
                "summarize": StageInfo(name="summarize", status="not_started"),
                "stage": StageInfo(name="stage", status="not_started"),
                "final": StageInfo(name="final", status="not_started")
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status="pending"
        )
        
        # Create mock client that tracks concurrency
        async def tracked_complete(prompt, system_prompt=None):
            nonlocal concurrent_count, max_concurrent_observed
            async with lock:
                concurrent_count += 1
                if concurrent_count > max_concurrent_observed:
                    max_concurrent_observed = concurrent_count
            
            await asyncio.sleep(0.05)  # Simulate work
            
            async with lock:
                concurrent_count -= 1
            
            return "Mock summary"
        
        mock_client = MockLLMClient(delay=0)
        mock_client.acomplete = tracked_complete
        
        with patch.object(stage, '_load_prompt_template', return_value="Summarize: "):
            with patch('src.longtext_pipeline.pipeline.summarize.get_llm_client', return_value=mock_client):
                with patch.object(stage, '_save_summary', side_effect=lambda s, d: f"summary_{s.part_index:02d}.md"):
                    await stage.run(
                        parts=parts,
                        config={'model': {'name': 'mock'}, 'pipeline': {'max_workers': max_workers}},
                        manifest=manifest,
                        mode='general'
                    )
        
        # Verify semaphore limited concurrency
        assert max_concurrent_observed <= max_workers, \
            f"Observed {max_concurrent_observed} concurrent, expected <= {max_workers}"
    
    @pytest.mark.asyncio
    async def test_stage_synthesis_respects_max_workers_config(self):
        """Test that StageSynthesisStage respects max_workers from config."""
        from src.longtext_pipeline.models import StageInfo
        
        max_workers = 3
        concurrent_count = 0
        max_concurrent_observed = 0
        lock = asyncio.Lock()
        
        summaries = [
            Summary(part_index=i, content=f"Summary {i}", metadata={})
            for i in range(9)
        ]
        
        stage = StageSynthesisStage()
        manifest = Manifest(
            session_id="test-session-004",
            input_path="test_input.txt",
            input_hash="d" * 64,
            stages={
                "ingest": StageInfo(name="ingest", status="successful"),
                "summarize": StageInfo(name="summarize", status="successful"),
                "stage": StageInfo(name="stage", status="not_started"),
                "final": StageInfo(name="final", status="not_started")
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status="pending"
        )
        
        # Create mock client that tracks concurrency
        async def tracked_complete(prompt, system_prompt=None):
            nonlocal concurrent_count, max_concurrent_observed
            async with lock:
                concurrent_count += 1
                if concurrent_count > max_concurrent_observed:
                    max_concurrent_observed = concurrent_count
            
            await asyncio.sleep(0.05)  # Simulate work
            
            async with lock:
                concurrent_count -= 1
            
            return "Mock synthesis"
        
        mock_client = MockLLMClient(delay=0)
        mock_client.acomplete = tracked_complete
        
        with patch.object(stage, '_load_prompt_template', return_value="Synthesize: "):
            with patch('src.longtext_pipeline.pipeline.stage_synthesis.get_llm_client', return_value=mock_client):
                with patch.object(stage, '_save_stage_summary', side_effect=lambda s, d: f"stage_{s.stage_index:02d}.md"):
                    await stage.run(
                        summaries=summaries,
                        config={
                            'model': {'name': 'mock'},
                            'pipeline': {'max_workers': max_workers},
                            'stages': {'stage': {'group_size': 3}}
                        },
                        manifest=manifest,
                        mode='general'
                    )
        
        # Verify semaphore limited concurrency
        assert max_concurrent_observed <= max_workers, \
            f"Observed {max_concurrent_observed} concurrent, expected <= {max_workers}"


class TestErrorIsolation:
    """Test that errors in one task don't affect others."""
    
    @pytest.mark.asyncio
    async def test_summarize_error_isolation_single_failure(self):
        """Test that one part failure doesn't prevent others from completing."""
        from src.longtext_pipeline.models import StageInfo
        
        fail_index = 2
        total_parts = 5
        
        parts = [
            Part(index=i, content=f"Content {i}", token_count=100, metadata={})
            for i in range(total_parts)
        ]
        
        stage = SummarizeStage()
        manifest = Manifest(
            session_id="test-session-005",
            input_path="test_input.txt",
            input_hash="e" * 64,
            stages={
                "ingest": StageInfo(name="ingest", status="successful"),
                "summarize": StageInfo(name="summarize", status="not_started"),
                "stage": StageInfo(name="stage", status="not_started"),
                "final": StageInfo(name="final", status="not_started")
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status="pending"
        )
        
        # Create client that fails on specific index
        mock_client = MockLLMClient(
            response_sequence=[f"Response {i}" for i in range(total_parts)],
            delay=0,
            fail_indices={fail_index}
        )
        
        with patch.object(stage, '_load_prompt_template', return_value="Summarize: "):
            with patch('src.longtext_pipeline.pipeline.summarize.get_llm_client', return_value=mock_client):
                with patch.object(stage, '_save_summary', side_effect=lambda s, d: f"summary_{s.part_index:02d}.md"):
                    summaries = await stage.run(
                        parts=parts,
                        config={'model': {'name': 'mock'}, 'pipeline': {'max_workers': 4}},
                        manifest=manifest,
                        mode='general'
                    )
        
        # Should have 4 successful summaries (all except index 2)
        assert len(summaries) == total_parts - 1
        assert len([s for s in summaries if s.part_index != fail_index]) == total_parts - 1
        
        # Verify correct indices present
        present_indices = {s.part_index for s in summaries}
        assert present_indices == {0, 1, 3, 4}
        assert fail_index not in present_indices
    
    @pytest.mark.asyncio
    async def test_summarize_multiple_failures_isolation(self):
        """Test that multiple failures don't prevent remaining successes."""
        from src.longtext_pipeline.models import StageInfo
        
        fail_indices = {1, 3, 5}
        total_parts = 7
        
        parts = [
            Part(index=i, content=f"Content {i}", token_count=100, metadata={})
            for i in range(total_parts)
        ]
        
        stage = SummarizeStage()
        manifest = Manifest(
            session_id="test-session-006",
            input_path="test_input.txt",
            input_hash="f" * 64,
            stages={
                "ingest": StageInfo(name="ingest", status="successful"),
                "summarize": StageInfo(name="summarize", status="not_started"),
                "stage": StageInfo(name="stage", status="not_started"),
                "final": StageInfo(name="final", status="not_started")
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status="pending"
        )
        
        mock_client = MockLLMClient(
            response_sequence=[f"Response {i}" for i in range(total_parts)],
            delay=0,
            fail_indices=fail_indices
        )
        
        with patch.object(stage, '_load_prompt_template', return_value="Summarize: "):
            with patch('src.longtext_pipeline.pipeline.summarize.get_llm_client', return_value=mock_client):
                with patch.object(stage, '_save_summary', side_effect=lambda s, d: f"summary_{s.part_index:02d}.md"):
                    summaries = await stage.run(
                        parts=parts,
                        config={'model': {'name': 'mock'}, 'pipeline': {'max_workers': 4}},
                        manifest=manifest,
                        mode='general'
                    )
        
        # Should have 4 successful summaries
        expected_successes = total_parts - len(fail_indices)
        assert len(summaries) == expected_successes
        
        # Verify correct indices
        present_indices = {s.part_index for s in summaries}
        expected_indices = set(range(total_parts)) - fail_indices
        assert present_indices == expected_indices
    
    @pytest.mark.asyncio
    async def test_stage_synthesis_error_isolation(self):
        """Test that one group failure doesn't prevent others from completing."""
        from src.longtext_pipeline.models import StageInfo
        
        fail_group_index = 1
        total_groups = 3
        group_size = 2
        
        # Create 6 summaries (3 groups of 2)
        summaries = [
            Summary(part_index=i, content=f"Summary {i}", metadata={})
            for i in range(total_groups * group_size)
        ]
        
        stage = StageSynthesisStage()
        manifest = Manifest(
            session_id="test-session-007",
            input_path="test_input.txt",
            input_hash="g" * 64,
            stages={
                "ingest": StageInfo(name="ingest", status="successful"),
                "summarize": StageInfo(name="summarize", status="successful"),
                "stage": StageInfo(name="stage", status="not_started"),
                "final": StageInfo(name="final", status="not_started")
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status="pending"
        )
        
        # Calculate which calls should fail (group 1 = calls for indices 2,3)
        # Since we process groups concurrently, fail the second group's call
        fail_indices = {fail_group_index}  # Fail group index 1
        
        mock_client = MockLLMClient(
            response_sequence=[f"Stage response {i}" for i in range(total_groups)],
            delay=0,
            fail_indices=fail_indices
        )
        
        with patch.object(stage, '_load_prompt_template', return_value="Synthesize: "):
            with patch('src.longtext_pipeline.pipeline.stage_synthesis.get_llm_client', return_value=mock_client):
                with patch.object(stage, '_save_stage_summary', side_effect=lambda s, d: f"stage_{s.stage_index:02d}.md"):
                    stage_summaries = await stage.run(
                        summaries=summaries,
                        config={
                            'model': {'name': 'mock'},
                            'pipeline': {'max_workers': 3},
                            'stages': {'stage': {'group_size': group_size}}
                        },
                        manifest=manifest,
                        mode='general'
                    )
        
        # Should have 2 successful stage summaries (all except group 1)
        assert len(stage_summaries) == total_groups - 1
        
        # Verify correct group indices
        present_indices = {s.stage_index for s in stage_summaries}
        expected_indices = {0, 2}
        assert present_indices == expected_indices
        assert fail_group_index not in present_indices
    
    @pytest.mark.asyncio
    async def test_all_failures_raises_stage_failed_error(self):
        """Test that all failures raises StageFailedError."""
        from src.longtext_pipeline.errors import StageFailedError
        from src.longtext_pipeline.models import StageInfo
        
        total_parts = 3
        parts = [
            Part(index=i, content=f"Content {i}", token_count=100, metadata={})
            for i in range(total_parts)
        ]
        
        stage = SummarizeStage()
        manifest = Manifest(
            session_id="test-session-008",
            input_path="test_input.txt",
            input_hash="h" * 64,
            stages={
                "ingest": StageInfo(name="ingest", status="successful"),
                "summarize": StageInfo(name="summarize", status="not_started"),
                "stage": StageInfo(name="stage", status="not_started"),
                "final": StageInfo(name="final", status="not_started")
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status="pending"
        )
        
        # Make all calls fail
        mock_client = MockLLMClient(
            response_sequence=[f"Response {i}" for i in range(total_parts)],
            delay=0,
            fail_indices=set(range(total_parts))
        )
        
        with patch.object(stage, '_load_prompt_template', return_value="Summarize: "):
            with patch('src.longtext_pipeline.pipeline.summarize.get_llm_client', return_value=mock_client):
                with patch.object(stage, '_save_summary', side_effect=lambda s, d: f"summary_{s.part_index:02d}.md"):
                    with pytest.raises(StageFailedError) as exc_info:
                        await stage.run(
                            parts=parts,
                            config={'model': {'name': 'mock'}, 'pipeline': {'max_workers': 4}},
                            manifest=manifest,
                            mode='general'
                        )
        
        # Verify error message mentions all failures
        assert "all" in str(exc_info.value).lower()
        assert "failed" in str(exc_info.value).lower()


class TestConcurrencyPatterns:
    """Test specific concurrency patterns used in pipeline."""
    
    @pytest.mark.asyncio
    async def test_gather_preserves_result_order(self):
        """Test that asyncio.gather preserves result order regardless of completion order."""
        results = []
        
        async def timed_task(value, delay):
            await asyncio.sleep(delay)
            results.append(value)
            return value
        
        # Tasks complete in reverse order (5, 4, 3, 2, 1)
        delays = [0.05, 0.04, 0.03, 0.02, 0.01]
        values = [1, 2, 3, 4, 5]
        
        tasks = [timed_task(v, d) for v, d in zip(values, delays)]
        gathered = await asyncio.gather(*tasks)
        
        # Gather preserves order, not completion order
        assert gathered == [1, 2, 3, 4, 5]
        
        # But completion order is different
        assert results == [5, 4, 3, 2, 1]
    
    @pytest.mark.asyncio
    async def test_semaphore_with_exception_continues(self):
        """Test that semaphore allows other tasks to continue after exception."""
        semaphore = asyncio.Semaphore(2)
        completed = []
        
        async def task(index, should_fail=False):
            async with semaphore:
                await asyncio.sleep(0.01)
                if should_fail:
                    raise ValueError(f"Task {index} failed")
                completed.append(index)
                return index
        
        # Run tasks, some failing
        tasks = [
            task(0),
            task(1, should_fail=True),
            task(2),
            task(3, should_fail=True),
            task(4)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check results
        assert 0 in completed
        assert 2 in completed
        assert 4 in completed
        
        # Check exceptions were captured
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 2
    
    @pytest.mark.asyncio
    async def test_no_deadlock_with_semaphore_exhaustion(self):
        """Test that semaphore doesn't cause deadlock when all slots used."""
        max_workers = 3
        semaphore = asyncio.Semaphore(max_workers)
        completed_count = 0
        lock = asyncio.Lock()
        
        async def worker(workers_id):
            nonlocal completed_count
            async with semaphore:
                await asyncio.sleep(0.01)
                async with lock:
                    completed_count += 1
            return workers_id
        
        # Run more tasks than semaphore slots
        num_tasks = 10
        tasks = [worker(i) for i in range(num_tasks)]
        results = await asyncio.gather(*tasks)
        
        # All tasks should complete
        assert completed_count == num_tasks
        assert len(results) == num_tasks


class TestMockClientBehavior:
    """Test the MockLLMClient helper for testing."""
    
    @pytest.mark.asyncio
    async def test_mock_client_response_sequence(self):
        """Test that mock client returns responses in sequence."""
        responses = ["First", "Second", "Third"]
        client = MockLLMClient(response_sequence=responses)
        
        result1 = await client.acomplete("prompt1")
        result2 = await client.acomplete("prompt2")
        result3 = await client.acomplete("prompt3")
        
        assert result1 == "First"
        assert result2 == "Second"
        assert result3 == "Third"
    
    @pytest.mark.asyncio
    async def test_mock_client_fail_indices(self):
        """Test that mock client fails on specified indices."""
        from src.longtext_pipeline.errors import LLMError
        
        client = MockLLMClient(
            response_sequence=["Success"] * 5,
            fail_indices={1, 3}
        )
        
        # First should succeed
        result0 = await client.acomplete("prompt0")
        assert result0 == "Success"
        
        # Second should fail
        with pytest.raises(LLMError):
            await client.acomplete("prompt1")
        
        # Third should succeed
        result2 = await client.acomplete("prompt2")
        assert result2 == "Success"
        
        # Fourth should fail
        with pytest.raises(LLMError):
            await client.acomplete("prompt3")
    
    @pytest.mark.asyncio
    async def test_mock_client_call_tracking(self):
        """Test that mock client tracks call count and order."""
        client = MockLLMClient(response_sequence=["A", "B", "C"])
        
        await client.acomplete("prompt1")
        await client.acomplete("prompt2")
        await client.acomplete("prompt3")
        
        assert client.call_count == 3
        assert client.call_order == [0, 1, 2]
