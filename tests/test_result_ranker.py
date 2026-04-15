"""Tests for the ResultRanker class and ranking strategies."""

import pytest

from src.longtext_pipeline.llm import (
    ResultRanker,
    RankingStrategy,
    QualityMetrics,
    rank_responses,
)
from src.longtext_pipeline.llm.dispatcher import ProviderResponse


class TestRankingStrategy:
    def test_strategy_enum_values(self):
        assert RankingStrategy.FASTEST.value == "fastest"
        assert RankingStrategy.CHEAPEST.value == "cheapest"
        assert RankingStrategy.BEST_QUALITY.value == "best_quality"
        assert RankingStrategy.BEST_PRICE_QUALITY.value == "best_price_quality"
        assert RankingStrategy.ROUND_ROBIN.value == "round_robin"
        assert RankingStrategy.RANDOM.value == "random"


class TestResultRankerInitialization:
    def test_ranker_initialization_with_defaults(self):
        ranker = ResultRanker()
        assert ranker.latency_weight == 0.3
        assert ranker.cost_weight == 0.3
        assert ranker.quality_weight == 0.4

    def test_ranker_initialization_with_invalid_weights_raises_error(self):
        with pytest.raises(ValueError, match="Weights must sum to approximately 1.0"):
            ResultRanker(latency_weight=0.5, cost_weight=0.3, quality_weight=0.1)


class TestRankingByLatency:
    def test_rank_by_latency_simple(self):
        ranker = ResultRanker()
        responses = [
            ProviderResponse(
                provider_name="slow", content="text", latency=2.0, cost_estimate=0.01
            ),
            ProviderResponse(
                provider_name="fast", content="text", latency=0.5, cost_estimate=0.02
            ),
        ]
        best = ranker.rank(responses, strategy=RankingStrategy.FASTEST)
        assert best.provider_name == "fast"
        assert best.latency == 0.5


class TestRankingByCost:
    def test_rank_by_cost_simple(self):
        ranker = ResultRanker()
        responses = [
            ProviderResponse(
                provider_name="expensive",
                content="text",
                latency=1.0,
                cost_estimate=0.1,
            ),
            ProviderResponse(
                provider_name="cheap", content="text", latency=2.0, cost_estimate=0.01
            ),
        ]
        best = ranker.rank(responses, strategy=RankingStrategy.CHEAPEST)
        assert best.provider_name == "cheap"


class TestRankingByQuality:
    def test_rank_by_quality_with_content(self):
        ranker = ResultRanker()
        responses = [
            ProviderResponse(
                provider_name="short", content="Hi", latency=0.5, cost_estimate=0.001
            ),
            ProviderResponse(
                provider_name="medium",
                content="A" * 1000,
                latency=1.0,
                cost_estimate=0.01,
            ),
        ]
        best = ranker.rank(responses, strategy=RankingStrategy.BEST_QUALITY)
        assert best.provider_name == "medium"


class TestRankingByPriceQuality:
    def test_rank_by_price_quality_simple(self):
        ranker = ResultRanker(latency_weight=0.2, cost_weight=0.3, quality_weight=0.5)
        responses = [
            ProviderResponse(
                provider_name="fast_cheap",
                content="Short",
                latency=0.1,
                cost_estimate=0.001,
            ),
            ProviderResponse(
                provider_name="slow_expensive",
                content="Long response with details.",
                latency=2.0,
                cost_estimate=0.05,
            ),
        ]
        best = ranker.rank(responses, strategy=RankingStrategy.BEST_PRICE_QUALITY)
        assert isinstance(best, ProviderResponse)


class TestRoundRobinRanking:
    def test_round_robin_selection(self):
        ranker = ResultRanker()
        r1 = [
            ProviderResponse(
                provider_name="a", content="x", latency=1.0, cost_estimate=0.01
            )
        ]
        r2 = [
            ProviderResponse(
                provider_name="b", content="x", latency=1.0, cost_estimate=0.01
            )
        ]
        b1 = ranker.rank(r1, strategy=RankingStrategy.ROUND_ROBIN)
        b2 = ranker.rank(r2, strategy=RankingStrategy.ROUND_ROBIN)
        assert b1.provider_name != b2.provider_name


class TestRandomRanking:
    def test_random_ranking_returns_valid_response(self):
        ranker = ResultRanker()
        responses = [
            ProviderResponse(
                provider_name="a", content="x", latency=1.0, cost_estimate=0.01
            ),
            ProviderResponse(
                provider_name="b", content="x", latency=1.0, cost_estimate=0.01
            ),
        ]
        best = ranker.rank(responses, strategy=RankingStrategy.RANDOM)
        assert best.provider_name in ["a", "b"]


class TestQualityMetrics:
    def test_calculate_latency_score(self):
        ranker = ResultRanker()
        resp_fast = ProviderResponse(
            provider_name="test", content="text", latency=0.1, cost_estimate=0.01
        )
        resp_slow = ProviderResponse(
            provider_name="test", content="text", latency=2.0, cost_estimate=0.01
        )
        score_fast = ranker._calculate_latency_score(resp_fast)
        score_slow = ranker._calculate_latency_score(resp_slow)
        assert score_fast > score_slow

    def test_calculate_cost_score(self):
        ranker = ResultRanker()
        resp_cheap = ProviderResponse(
            provider_name="test", content="text", latency=1.0, cost_estimate=0.001
        )
        resp_expensive = ProviderResponse(
            provider_name="test", content="text", latency=1.0, cost_estimate=0.1
        )
        score_cheap = ranker._calculate_cost_score(resp_cheap)
        score_expensive = ranker._calculate_cost_score(resp_expensive)
        assert score_cheap > score_expensive

    def test_calculate_quality_score_with_content(self):
        ranker = ResultRanker()
        content = "This is well-structured response with multiple paragraphs. First point. Second point. Finally, summary."
        resp = ProviderResponse(
            provider_name="test", content=content, latency=1.0, cost_estimate=0.01
        )
        metrics = ranker._calculate_quality_score(resp)
        assert isinstance(metrics, QualityMetrics)
        assert 0 <= metrics.quality_score <= 1


class TestRankResponsesConvenienceFunction:
    def test_convenience_function_ranking(self):
        responses = [
            ProviderResponse(
                provider_name="fast", content="c", latency=0.5, cost_estimate=0.01
            )
        ]
        best = rank_responses(responses, strategy=RankingStrategy.FASTEST)
        assert best.provider_name == "fast"


class TestGetRankingScoringDetails:
    def test_get_details_for_fastest_strategy(self):
        ranker = ResultRanker()
        responses = [
            ProviderResponse(
                provider_name="fast", content="t", latency=0.5, cost_estimate=0.01
            )
        ]
        details = ranker.get_ranking_scoring_details(responses, RankingStrategy.FASTEST)
        assert details["strategy"] == "fastest"


if __name__ == "__main__":
    pytest.main([__file__])
