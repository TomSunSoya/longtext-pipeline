"""Test suite for Pipeline Multi-Provider Integration functionality.

This test covers integration points between the multi-provider dispatcher/ranker
and the pipeline orchestrator, stage processing components.
"""

from longtext_pipeline.config import get_agent_provider_configs, DEFAULT_CONFIG
from longtext_pipeline.pipeline.summarize import SummarizeStage
from longtext_pipeline.pipeline.stage_synthesis import StageSynthesisStage
from longtext_pipeline.pipeline.final_analysis import FinalAnalysisStage


def test_get_agent_provider_configs_single():
    """Test get_agent_provider_configs with single provider."""
    config = {
        "agents": {"summarizer": {"model": "default"}},
        "model": {
            "providers": {"default": {"provider": "openai", "name": "gpt-4o-mini"}}
        },
    }

    provider_configs = get_agent_provider_configs(config, "summarizer")
    assert len(provider_configs) == 1
    assert provider_configs[0]["name"] == "gpt-4o-mini"


def test_get_agent_provider_configs_multiple():
    """Test get_agent_provider_configs with multiple providers."""
    config = {
        "agents": {
            "summarizer": {
                "model": {
                    "providers": [
                        {"provider": "openai", "name": "gpt-4o"},
                        {"provider": "anthropic", "name": "claude-3.5-sonnet"},
                    ]
                }
            }
        }
    }

    provider_configs = get_agent_provider_configs(config, "summarizer")
    assert len(provider_configs) == 2
    assert provider_configs[0]["name"] == "gpt-4o"
    assert provider_configs[1]["name"] == "claude-3.5-sonnet"


def test_get_agent_provider_configs_from_main_providers_section():
    """Test get_agent_provider_configs when provider refers to the main model config."""
    config = {
        "agents": {
            "summarizer": {
                "model": "primary"  # Refers to primary in model.providers
            }
        },
        "model": {
            "providers": {
                "primary": {"provider": "openai", "name": "gpt-4o"},
                "backup": {"provider": "anthropic", "name": "claude-3.5-sonnet"},
            }
        },
    }

    provider_configs = get_agent_provider_configs(config, "summarizer")
    assert len(provider_configs) == 1
    assert provider_configs[0]["name"] == "gpt-4o"
    assert provider_configs[0]["provider"] == "openai"


def test_legacy_config_format():
    """Test that legacy config format is handled properly."""
    config = {
        "agents": {
            "summarizer": {"model": {"provider": "openai", "name": "gpt-4o-mini"}}
        }
    }

    provider_configs = get_agent_provider_configs(config, "summarizer")
    # Should return a single provider config
    assert len(provider_configs) == 1
    assert provider_configs[0]["provider"] == "openai"
    assert provider_configs[0]["name"] == "gpt-4o-mini"


def test_multi_provider_with_dispatch_mode():
    """Test multi-provider setup with specific dispatch mode."""
    config = {
        "model": {
            "dispatch_mode": "ranked",
            "providers": {
                "gpt": {"provider": "openai", "name": "gpt-4o"},
                "claude": {"provider": "anthropic", "name": "claude-3.5-sonnet"},
            },
        },
        "agents": {
            "analyst": {
                "providers": [
                    {"provider": "openai", "name": "gpt-4o"},
                    {"provider": "anthropic", "name": "claude-3.5-sonnet"},
                ]
            }
        },
    }

    # Get analyst configs
    analyst_provider_configs = get_agent_provider_configs(config, "analyst")
    assert len(analyst_provider_configs) == 2

    # Verify both providers are present
    provider_names = [pc["name"] for pc in analyst_provider_configs]
    assert "gpt-4o" in provider_names
    assert "claude-3.5-sonnet" in provider_names


def test_default_config_with_new_fields():
    """Test that DEFAULT_CONFIG includes the new fields."""
    assert "providers" in DEFAULT_CONFIG["model"]
    assert "dispatch_mode" in DEFAULT_CONFIG["model"]
    assert "default" in DEFAULT_CONFIG["model"]["providers"]


def test_summarize_stage_creation():
    """Test that SummarizeStage can be instantiated and has expected behavior."""
    summarize_stage = SummarizeStage()
    assert summarize_stage is not None


def test_stage_synthesis_stage_creation():
    """Test that StageSynthesisStage can be instantiated."""
    stage_synthesis_stage = StageSynthesisStage()
    assert stage_synthesis_stage is not None


def test_final_analysis_stage_creation():
    """Test that FinalAnalysisStage can be instantiated."""
    final_analysis_stage = FinalAnalysisStage()
    assert final_analysis_stage is not None


if __name__ == "__main__":
    test_get_agent_provider_configs_single()
    test_get_agent_provider_configs_multiple()
    test_get_agent_provider_configs_from_main_providers_section()
    test_legacy_config_format()
    test_multi_provider_with_dispatch_mode()
    test_default_config_with_new_fields()
    test_summarize_stage_creation()
    test_stage_synthesis_stage_creation()
    test_final_analysis_stage_creation()

    print("All tests passed!")
