"""Unified result ranking utilities for multi-provider LLM responses.

This module is the single supported ranking implementation for the ``llm``
package. Legacy modules such as ``ranker.py`` and ``result_ranker.py`` should
only re-export from here.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .dispatcher import ProviderResponse


class RankingStrategy(Enum):
    """Strategies for selecting the best response from multiple providers."""

    FASTEST = "fastest"
    CHEAPEST = "cheapest"
    BEST_QUALITY = "best_quality"
    BEST_PRICE_QUALITY = "best_price_quality"
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"


@dataclass
class QualityMetrics:
    """Heuristic quality metrics for a provider response."""

    length_score: float
    structure_score: float
    coherence_score: float
    vocabulary_score: float
    quality_score: float

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to a serializable mapping."""
        return {
            "length_score": self.length_score,
            "structure_score": self.structure_score,
            "coherence_score": self.coherence_score,
            "vocabulary_score": self.vocabulary_score,
            "quality_score": self.quality_score,
        }


@dataclass
class ProviderMetrics:
    """Summary metrics for a provider response."""

    provider_name: str
    latency: float = 0.0
    tokens_used: int = 0
    cost_estimate: float = 0.0
    success: bool = True
    content_length: int = 0
    quality_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert provider metrics to a serializable mapping."""
        return {
            "provider_name": self.provider_name,
            "latency": self.latency,
            "tokens_used": self.tokens_used,
            "cost_estimate": self.cost_estimate,
            "success": self.success,
            "content_length": self.content_length,
            "quality_score": self.quality_score,
        }


class ResultRanker:
    """Ranker used by dispatcher ranked mode and public tests."""

    def __init__(
        self,
        latency_weight: float = 0.3,
        cost_weight: float = 0.3,
        quality_weight: float = 0.4,
    ) -> None:
        total_weight = latency_weight + cost_weight + quality_weight
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError("Weights must sum to approximately 1.0")

        self.latency_weight = latency_weight
        self.cost_weight = cost_weight
        self.quality_weight = quality_weight
        self._round_robin_index = 0

    def rank(
        self,
        responses: list["ProviderResponse"],
        strategy: RankingStrategy = RankingStrategy.BEST_QUALITY,
    ) -> "ProviderResponse":
        """Return the best response according to the requested strategy."""
        if not responses:
            raise ValueError("No responses provided for ranking")

        successful = [response for response in responses if response.success]
        candidates = successful or responses

        if strategy == RankingStrategy.FASTEST:
            return max(candidates, key=self._calculate_latency_score)
        if strategy == RankingStrategy.CHEAPEST:
            return max(candidates, key=self._calculate_cost_score)
        if strategy == RankingStrategy.BEST_QUALITY:
            return max(
                candidates,
                key=lambda response: (
                    self._calculate_quality_score(response).quality_score
                ),
            )
        if strategy == RankingStrategy.BEST_PRICE_QUALITY:
            return max(candidates, key=self._calculate_combined_score)
        if strategy == RankingStrategy.ROUND_ROBIN:
            return self._select_round_robin(candidates)
        if strategy == RankingStrategy.RANDOM:
            return random.choice(candidates)

        return max(
            candidates,
            key=lambda response: self._calculate_quality_score(response).quality_score,
        )

    def _select_round_robin(
        self, responses: list["ProviderResponse"]
    ) -> "ProviderResponse":
        """Select a response deterministically across repeated calls."""
        ordered = sorted(responses, key=lambda response: response.provider_name)
        choice = ordered[self._round_robin_index % len(ordered)]
        self._round_robin_index += 1
        return choice

    def _calculate_combined_score(self, response: "ProviderResponse") -> float:
        """Calculate a blended score for latency, cost, and content quality."""
        quality_score = self._calculate_quality_score(response).quality_score
        return (
            self.latency_weight * self._calculate_latency_score(response)
            + self.cost_weight * self._calculate_cost_score(response)
            + self.quality_weight * quality_score
        )

    def _calculate_latency_score(self, response: "ProviderResponse") -> float:
        """Higher is better, with low-latency responses preferred."""
        latency = max(response.latency, 0.0)
        return 1.0 / (1.0 + latency)

    def _calculate_cost_score(self, response: "ProviderResponse") -> float:
        """Higher is better, with cheaper responses preferred."""
        cost = response.cost_estimate
        if cost is None or cost <= 0:
            return 1.0
        return 1.0 / (1.0 + (cost * 1000.0))

    def _calculate_quality_score(self, response: "ProviderResponse") -> QualityMetrics:
        """Estimate response quality using lightweight text heuristics."""
        content = (response.content or "").strip()
        if not content:
            return QualityMetrics(
                length_score=0.0,
                structure_score=0.0,
                coherence_score=0.0,
                vocabulary_score=0.0,
                quality_score=0.0,
            )

        length_score = self._calculate_length_score(content)
        structure_score = self._calculate_structure_score(content)
        coherence_score = self._calculate_coherence_score(content)
        vocabulary_score = self._calculate_vocabulary_score(content)

        quality_score = (
            (length_score * 0.25)
            + (structure_score * 0.25)
            + (coherence_score * 0.30)
            + (vocabulary_score * 0.20)
        )

        return QualityMetrics(
            length_score=length_score,
            structure_score=structure_score,
            coherence_score=coherence_score,
            vocabulary_score=vocabulary_score,
            quality_score=max(0.0, min(quality_score, 1.0)),
        )

    def _calculate_length_score(self, content: str) -> float:
        """Prefer substantive answers without over-rewarding verbosity."""
        content_length = len(content)
        if content_length < 50:
            return max(0.1, content_length / 100.0)
        if content_length <= 1200:
            return min(1.0, content_length / 1200.0)
        if content_length <= 4000:
            return 1.0
        return max(0.65, 1.0 - ((content_length - 4000) / 12000.0))

    def _calculate_structure_score(self, content: str) -> float:
        """Reward headings, paragraphs, and lists when present."""
        lines = content.splitlines()
        non_empty_lines = [line for line in lines if line.strip()]
        if not non_empty_lines:
            return 0.0

        paragraph_count = max(
            1, len([line for line in non_empty_lines if len(line) > 40])
        )
        heading_count = len(
            [
                line
                for line in non_empty_lines
                if line.strip().startswith(("#", "##", "###"))
            ]
        )
        list_count = len(
            [
                line
                for line in non_empty_lines
                if line.strip().startswith(("- ", "* ", "1. ", "2. ", "3. "))
            ]
        )

        paragraph_score = min(1.0, paragraph_count / 4.0)
        heading_score = min(1.0, heading_count / 3.0)
        list_score = min(1.0, list_count / 4.0)

        return min(
            1.0, (paragraph_score * 0.5) + (heading_score * 0.25) + (list_score * 0.25)
        )

    def _calculate_coherence_score(self, content: str) -> float:
        """Estimate coherence from sentence balance and discourse markers."""
        sentences = [
            sentence.strip()
            for sentence in re.split(r"[.!?]+\s*", content)
            if sentence.strip()
        ]
        if not sentences:
            return 0.0

        word_counts = [len(sentence.split()) for sentence in sentences]
        average_sentence_length = sum(word_counts) / len(word_counts)

        if 6 <= average_sentence_length <= 24:
            balance_score = 1.0
        elif 3 <= average_sentence_length <= 32:
            balance_score = 0.75
        else:
            balance_score = 0.5

        transitions = [
            "however",
            "therefore",
            "because",
            "meanwhile",
            "first",
            "second",
            "finally",
            "此外",
            "因此",
            "首先",
            "最后",
        ]
        transition_hits = sum(1 for marker in transitions if marker in content.lower())
        transition_score = min(1.0, transition_hits / 4.0)

        return min(1.0, (balance_score * 0.7) + (transition_score * 0.3))

    def _calculate_vocabulary_score(self, content: str) -> float:
        """Estimate lexical diversity without over-penalizing short responses."""
        words = re.findall(r"\b\w+\b", content.lower())
        if not words:
            return 0.0
        if len(words) < 10:
            return 0.4

        type_token_ratio = len(set(words)) / len(words)
        return max(0.0, min(1.0, (type_token_ratio - 0.3) / 0.5))

    def get_ranking_scoring_details(
        self,
        responses: list["ProviderResponse"],
        strategy: RankingStrategy,
    ) -> dict[str, Any]:
        """Return per-provider scoring details for debugging and tests."""
        ranked = self.rank(responses, strategy=strategy)
        provider_details: list[dict[str, Any]] = []

        for response in responses:
            quality_metrics = self._calculate_quality_score(response)
            provider_details.append(
                {
                    "provider_name": response.provider_name,
                    "latency_score": self._calculate_latency_score(response),
                    "cost_score": self._calculate_cost_score(response),
                    "quality_metrics": quality_metrics.to_dict(),
                    "combined_score": self._calculate_combined_score(response),
                    "success": response.success,
                }
            )

        return {
            "strategy": strategy.value,
            "selected_provider": ranked.provider_name,
            "providers": provider_details,
        }

    def reset_round_robin(self) -> None:
        """Reset round-robin state."""
        self._round_robin_index = 0

    def get_provider_metrics(
        self, responses: list["ProviderResponse"]
    ) -> list[ProviderMetrics]:
        """Return a compact provider metrics summary."""
        metrics: list[ProviderMetrics] = []
        for response in responses:
            quality = self._calculate_quality_score(response)
            metrics.append(
                ProviderMetrics(
                    provider_name=response.provider_name,
                    latency=response.latency,
                    tokens_used=response.tokens_used,
                    cost_estimate=response.cost_estimate,
                    success=response.success,
                    content_length=len(response.content or ""),
                    quality_score=quality.quality_score,
                )
            )
        return metrics


def rank_responses(
    responses: list["ProviderResponse"],
    strategy: RankingStrategy = RankingStrategy.BEST_QUALITY,
) -> "ProviderResponse":
    """Convenience wrapper for one-off ranking calls."""
    return ResultRanker().rank(responses, strategy=strategy)


QualityEstimate = QualityMetrics
