"""Coverage tests for compatibility wrapper modules."""

from src.longtext_pipeline.llm import results
from src.longtext_pipeline.llm import ranker as legacy_ranker
from src.longtext_pipeline.llm import result_ranker as legacy_result_ranker


def test_result_ranker_wrapper_reexports_expected_symbols():
    assert legacy_result_ranker.ResultRanker is results.ResultRanker
    assert legacy_result_ranker.RankingStrategy is results.RankingStrategy
    assert legacy_result_ranker.rank_responses is results.rank_responses
    assert "ResultRanker" in legacy_result_ranker.__all__


def test_ranker_wrapper_reexports_expected_symbols():
    assert legacy_ranker.ResultRanker is results.ResultRanker
    assert legacy_ranker.ProviderMetrics is results.ProviderMetrics
    assert legacy_ranker.QualityEstimate is results.QualityEstimate
    assert "rank_responses" in legacy_ranker.__all__
