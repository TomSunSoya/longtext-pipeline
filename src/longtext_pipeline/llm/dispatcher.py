"""
Multi-Provider Parallel Dispatcher for LLM clients.

This module provides the core parallel execution engine that dispatches
requests to multiple LLM providers simultaneously, supporting different execution modes
and result combination strategies.
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import time

from .registry import ProviderRegistry, get_default_registry
from .results import RankingStrategy, ResultRanker


class ParallelMode(Enum):
    """Execution modes for the parallel dispatcher."""

    SINGLE = "single"  # Use first provider only
    PARALLEL = "parallel"  # Get responses from all providers
    FASTEST = "fastest"  # Return first successful completion
    RANKED = "ranked"  # Rank results and return best based on heuristic


@dataclass
class ProviderResponse:
    """Response data structure for a single provider."""

    provider_name: str
    content: str
    latency: float  # in seconds
    tokens_used: int = 0
    cost_estimate: float = 0.0  # in dollars
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ParallelResult:
    """Result data structure for multi-provider parallel execution."""

    mode: ParallelMode
    responses: List[ProviderResponse]
    primary_content: str = ""  # Primary content from the best provider
    best_provider: Optional[str] = None
    execution_duration: float = 0.0  # Total execution time in seconds
    metadata: Dict[str, Any] = field(default_factory=dict)


class ParallelDispatcher:
    """Parallel LLM dispatcher that manages requests to multiple providers simultaneously."""

    def __init__(
        self,
        registry: Optional[ProviderRegistry] = None,
        max_concurrent_requests: int = 10,
        timeout_per_provider: float = 120.0,
    ) -> None:
        """
        Initialize the parallel dispatcher.

        Args:
            registry: Provider registry to use (defaults to global registry)
            max_concurrent_requests: Maximum number of concurrent requests per dispatch
            timeout_per_provider: Timeout for each individual provider request (seconds)
        """
        self.registry: ProviderRegistry = registry or get_default_registry()
        self.semaphore: asyncio.Semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.timeout_per_provider: float = timeout_per_provider

    async def dispatch(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        mode: ParallelMode = ParallelMode.PARALLEL,
        provider_configs: Optional[List[Dict[str, Any]]] = None,
        response_ranking_strategy: Optional[
            Callable[[List[ProviderResponse]], ProviderResponse]
        ] = None,
    ) -> ParallelResult:
        """
        Dispatch prompt to multiple providers based on the selected mode.

        Args:
            prompt: The user prompt to send to the providers
            system_prompt: Optional system prompt to set context/behavior
            mode: Execution mode (SINGLE, PARALLEL, FASTEST, RANKED)
            provider_configs: List of provider configurations to use
            response_ranking_strategy: Strategy to rank responses (used with RANKED mode)

        Returns:
            ParallelResult containing responses and metadata based on mode
        """
        start_time = time.time()

        # Default to simple quality heuristic for ranking if no strategy provided
        if not response_ranking_strategy and mode == ParallelMode.RANKED:
            response_ranking_strategy = self._default_quality_ranking_strategy

        # Use all registered providers if none specified
        if provider_configs is None:
            registered_providers = self.registry.list_providers()
            provider_configs = [{"provider": name} for name in registered_providers]

            if not provider_configs:
                raise ValueError("No providers configured in registry")

        if mode == ParallelMode.SINGLE:
            return await self._single_dispatch(
                prompt, system_prompt, provider_configs[0], start_time
            )
        elif mode == ParallelMode.FASTEST:
            return await self._fastest_dispatch(
                prompt, system_prompt, provider_configs, start_time
            )
        elif mode == ParallelMode.RANKED:
            return await self._ranked_dispatch(
                prompt,
                system_prompt,
                provider_configs,
                response_ranking_strategy or self._default_quality_ranking_strategy,
                start_time,
            )
        else:  # PARALLEL mode is default
            return await self._parallel_dispatch(
                prompt, system_prompt, provider_configs, start_time
            )

    async def _execute_single_request(
        self,
        provider_config: Dict[str, Any],
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> ProviderResponse:
        """
        Execute a single LLM request to a specific provider.

        Args:
            provider_config: Configuration for the provider
            prompt: The prompt to submit
            system_prompt: Optional system prompt

        Returns:
            ProviderResponse for this request
        """
        provider_name = provider_config["provider"]
        request_start = time.time()

        try:
            async with self.semaphore:
                client = self.registry.create_from_config(
                    provider_name, provider_config
                )

                if not hasattr(client, "acomplete"):
                    raise AttributeError(
                        f"Client for {provider_name} must support acomplete method"
                    )
                content = await client.acomplete(prompt, system_prompt)

                latency = time.time() - request_start
                response = ProviderResponse(
                    provider_name=provider_name,
                    content=content,
                    latency=latency,
                    success=True,
                    metadata=provider_config.get("metadata", {}),
                )
                return response

        except Exception as e:
            latency = time.time() - request_start
            return ProviderResponse(
                provider_name=provider_name,
                content="",
                latency=latency,
                success=False,
                error=str(e),
                metadata=provider_config.get("metadata", {}),
            )

    async def _single_dispatch(
        self,
        prompt: str,
        system_prompt: Optional[str],
        single_provider_config: Dict[str, Any],
        start_time: float,
    ) -> ParallelResult:
        """Execute request to only the first provider."""
        response = await self._execute_single_request(
            single_provider_config, prompt, system_prompt
        )
        execution_duration = time.time() - start_time

        return ParallelResult(
            mode=ParallelMode.SINGLE,
            responses=[response],
            primary_content=response.content,
            best_provider=response.provider_name,
            execution_duration=execution_duration,
        )

    async def _fastest_dispatch(
        self,
        prompt: str,
        system_prompt: Optional[str],
        provider_configs: List[Dict[str, Any]],
        start_time: float,
    ) -> ParallelResult:
        """Return the first successful completion from available providers."""

        # Create tasks for all providers
        tasks = [
            asyncio.create_task(
                self._execute_single_request(config, prompt, system_prompt)
            )
            for config in provider_configs
        ]

        # Get first successful response using asyncio.wait
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        # Get result from the first completed task
        first_task = done.pop()  # Take any task from the done set
        response = await first_task

        # If first response was successful, cancel others
        if response.success:
            for task in pending:
                task.cancel()

            # Wait briefly for cancellations and collect other responses
            if pending:
                for task in pending:
                    try:
                        await task  # Wait for potential errors during cancellation
                    except asyncio.CancelledError:
                        pass  # Expected error during cancelation
                    except Exception:
                        pass  # Other errors after task was cancelled

            execution_duration = time.time() - start_time
            return ParallelResult(
                mode=ParallelMode.FASTEST,
                responses=[response],  # Only return first successful
                primary_content=response.content,
                best_provider=response.provider_name,
                execution_duration=execution_duration,
            )
        else:
            # If the first response was not successful, wait for the next successful one
            # Keep checking remaining tasks for first success
            if pending:
                while pending:
                    try:
                        done_remaining, pending = await asyncio.wait(
                            pending,
                            return_when=asyncio.FIRST_COMPLETED,
                            timeout=self.timeout_per_provider,
                        )

                        # Check all tasks that completed in this round
                        for task in done_remaining:
                            try:
                                another_response = await task
                                if another_response.success:
                                    # Found a success response, cancel remaining tasks
                                    for remaining_task in pending:
                                        remaining_task.cancel()

                                    # Wait briefly for all cancellations
                                    if pending:
                                        for t in pending:
                                            try:
                                                await t
                                            except asyncio.CancelledError:
                                                pass
                                            except Exception:
                                                pass

                                    execution_duration = time.time() - start_time
                                    return ParallelResult(
                                        mode=ParallelMode.FASTEST,
                                        responses=[another_response],
                                        primary_content=another_response.content,
                                        best_provider=another_response.provider_name,
                                        execution_duration=execution_duration,
                                    )
                            except Exception:
                                # Skip failed tasks
                                continue

                    except asyncio.TimeoutError:
                        break  # Exit the loop if timeout is reached

            # If no successful responses found
            all_task_responses = [response]  # Include initially unsuccessful response
            if pending:
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        pass

            # Add response results that may have completed before cancellation
            for task in tasks:  # Process all tasks to collect results
                try:
                    result = await task
                    if result != response:  # Don't duplicate the initial response
                        all_task_responses.append(result)
                except Exception:
                    # Skip any remaining failed tasks
                    pass

            execution_duration = time.time() - start_time
            return ParallelResult(
                mode=ParallelMode.FASTEST,
                responses=all_task_responses,
                primary_content="",
                execution_duration=execution_duration,
            )

    async def _parallel_dispatch(
        self,
        prompt: str,
        system_prompt: Optional[str],
        provider_configs: List[Dict[str, Any]],
        start_time: float,
    ) -> ParallelResult:
        """Get responses from all providers in parallel."""

        responses: List[ProviderResponse] = []

        # Use asyncio TaskGroup if available (Python 3.11+), otherwise use gather with timeout
        if hasattr(asyncio, "TaskGroup"):
            # Helper function to execute and store results from each provider
            async def execute_request_and_store(config: Dict[str, Any]) -> None:
                try:
                    response = await asyncio.wait_for(
                        self._execute_single_request(config, prompt, system_prompt),
                        timeout=self.timeout_per_provider,
                    )
                    responses.append(response)
                except Exception as e:
                    # Even if request fails, we want to create a response entry
                    responses.append(
                        ProviderResponse(
                            provider_name=config["provider"],
                            content="",
                            latency=self.timeout_per_provider,
                            success=False,
                            error=str(e),
                        )
                    )

            try:
                async with asyncio.TaskGroup() as tg:
                    for config in provider_configs:
                        tg.create_task(execute_request_and_store(config))
            except Exception:
                # TaskGroup handles the exceptions internally but continues execution where possible
                pass
        else:
            # For older Python versions or fallback
            async def run_with_timeout(config: Dict[str, Any]) -> ProviderResponse:
                try:
                    return await asyncio.wait_for(
                        self._execute_single_request(config, prompt, system_prompt),
                        timeout=self.timeout_per_provider,
                    )
                except Exception as e:
                    return ProviderResponse(
                        provider_name=config["provider"],
                        content="",
                        latency=self.timeout_per_provider,
                        success=False,
                        error=str(e),
                    )

            tasks = [run_with_timeout(config) for config in provider_configs]
            task_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process the results
            for i, result in enumerate(task_results):
                if isinstance(result, BaseException):
                    responses.append(
                        ProviderResponse(
                            provider_name=provider_configs[i]["provider"],
                            content="",
                            latency=self.timeout_per_provider,
                            success=False,
                            error=str(result) if str(result) else repr(result),
                        )
                    )
                else:
                    responses.append(result)

        execution_duration = time.time() - start_time

        # For PARALLEL mode, return first successful response as primary if any
        successful_responses = [r for r in responses if r.success]
        if successful_responses:
            primary_content = successful_responses[0].content
            best_provider = successful_responses[0].provider_name
        else:
            primary_content = ""
            best_provider = None

        return ParallelResult(
            mode=ParallelMode.PARALLEL,
            responses=responses,
            primary_content=primary_content,
            best_provider=best_provider,
            execution_duration=execution_duration,
        )

    async def _ranked_dispatch(
        self,
        prompt: str,
        system_prompt: Optional[str],
        provider_configs: List[Dict[str, Any]],
        ranking_strategy: Callable[[List[ProviderResponse]], ProviderResponse],
        start_time: float,
    ) -> ParallelResult:
        """Rank responses from all providers and return the best one."""

        # For ranked dispatch, we need all responses before ranking
        responses: List[ProviderResponse] = []

        # Use asyncio TaskGroup if available (Python 3.11+), otherwise use gather with timeout
        if hasattr(asyncio, "TaskGroup"):
            # Helper function to execute and store results from each provider
            async def execute_request_and_store(config: Dict[str, Any]) -> None:
                try:
                    response = await asyncio.wait_for(
                        self._execute_single_request(config, prompt, system_prompt),
                        timeout=self.timeout_per_provider,
                    )
                    responses.append(response)
                except Exception as e:
                    # Even if request fails, we want to create a response entry
                    responses.append(
                        ProviderResponse(
                            provider_name=config["provider"],
                            content="",
                            latency=self.timeout_per_provider,
                            success=False,
                            error=str(e),
                        )
                    )

            try:
                async with asyncio.TaskGroup() as tg:
                    for config in provider_configs:
                        tg.create_task(execute_request_and_store(config))
            except Exception:
                # TaskGroup handles the exceptions internally but continues execution where possible
                pass
        else:
            # For older Python versions
            async def run_with_timeout(config: Dict[str, Any]) -> ProviderResponse:
                try:
                    return await asyncio.wait_for(
                        self._execute_single_request(config, prompt, system_prompt),
                        timeout=self.timeout_per_provider,
                    )
                except Exception as e:
                    return ProviderResponse(
                        provider_name=config["provider"],
                        content="",
                        latency=self.timeout_per_provider,
                        success=False,
                        error=str(e),
                    )

            tasks = [run_with_timeout(config) for config in provider_configs]
            task_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process the results
            for i, result in enumerate(task_results):
                if isinstance(result, BaseException):
                    responses.append(
                        ProviderResponse(
                            provider_name=provider_configs[i]["provider"],
                            content="",
                            latency=self.timeout_per_provider,
                            success=False,
                            error=str(result) if str(result) else repr(result),
                        )
                    )
                else:
                    responses.append(result)

        execution_duration = time.time() - start_time

        # Apply ranking strategy but only to successful responses
        successful_responses = [r for r in responses if r.success and r.content]

        if successful_responses:
            best_response = ranking_strategy(successful_responses)
            return ParallelResult(
                mode=ParallelMode.RANKED,
                responses=responses,
                primary_content=best_response.content,
                best_provider=best_response.provider_name,
                execution_duration=execution_duration,
            )
        else:
            # If no successful responses, return the original responses
            return ParallelResult(
                mode=ParallelMode.RANKED,
                responses=responses,
                primary_content="",
                execution_duration=execution_duration,
            )

    def _default_quality_ranking_strategy(
        self, responses: List[ProviderResponse]
    ) -> ProviderResponse:
        """
        Default strategy to rank responses based on a quality heuristic.

        Factors considered (in order of importance):
        1. Lower latency if within reasonable range
        2. Higher content length (non-empty content, not too verbose)
        3. Lower estimated cost

        Args:
            responses: List of successful responses to rank

        Returns:
            The 'best' ProviderResponse based on heuristic
        """
        if not responses:
            raise ValueError("No responses provided to ranking strategy")
        ranker = ResultRanker()
        return ranker.rank(responses, strategy=RankingStrategy.BEST_PRICE_QUALITY)
