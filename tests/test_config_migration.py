"""Tests for v1→v2 config migration functionality."""

import warnings
from pathlib import Path

from src.longtext_pipeline.config import migrate_config


class TestMigrateConfig:
    """Tests for the migrate_config function."""

    def test_v1_flat_format_detection(self):
        """v1 config with flat model.name is detected."""
        v1_config = {
            "model": {
                "provider": "openai",
                "name": "gpt-4o-mini",
            }
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            migrate_config(v1_config)

            # Should detect and warn about v1 format
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated flat model format" in str(w[0].message).lower()

    def test_v1_to_v2_conversion(self):
        """v1 flat config is converted to v2 nested structure."""
        v1_config = {
            "model": {
                "provider": "openai",
                "name": "gpt-4o-mini",
                "temperature": 0.7,
            }
        }

        result = migrate_config(v1_config)

        # Should have nested providers structure
        assert "providers" in result["model"]
        assert "default" in result["model"]["providers"]

        # Provider info should be preserved
        provider = result["model"]["providers"]["default"]
        assert provider["provider"] == "openai"
        assert provider["name"] == "gpt-4o-mini"

        # Other fields should be preserved at top level
        assert result["model"]["temperature"] == 0.7

    def test_v2_no_migration_needed(self):
        """v2 format with providers structure doesn't trigger migration."""
        # v2 format with providers structure
        v2_config = {
            "model": {
                "providers": {
                    "default": {
                        "provider": "openai",
                        "name": "gpt-4o-mini",
                    }
                },
                "temperature": 0.7,
            }
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = migrate_config(v2_config)

            # Should not warn since providers structure exists
            assert len(w) == 0

            # Should not modify existing providers
            assert result == v2_config

    def test_v2_with_providers_no_warning(self):
        """v2 format with providers structure passes cleanly."""
        v2_config = {
            "model": {
                "providers": {
                    "default": {
                        "provider": "openai",
                        "name": "gpt-4o-mini",
                    }
                },
                "temperature": 0.7,
            }
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = migrate_config(v2_config)

            # Should not trigger v1 warning
            assert len(w) == 0

            # Should not modify existing providers
            assert "providers" in result["model"]
            assert result["model"]["providers"]["default"]["provider"] == "openai"

    def test_original_not_mutated(self):
        """Original config dict is not modified."""
        v1_config = {
            "model": {
                "provider": "openai",
                "name": "gpt-4o-mini",
            }
        }

        import copy

        original = copy.deepcopy(v1_config)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = migrate_config(v1_config)

        # Original should be unchanged
        assert v1_config == original

        # Result should be different (migrated)
        assert result != original

    def test_with_source_path_in_warning(self):
        """Warning includes source path when provided."""
        v1_config = {"model": {"provider": "openai", "name": "gpt-4o-mini"}}
        source_path = Path("/test/config.yaml")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            migrate_config(v1_config, source_path=source_path)

            assert len(w) == 1
            # Path conversion may normalize separators
            warning_msg = str(w[0].message)
            assert (
                "/test/config.yaml" in warning_msg
                or "\\test\\config.yaml" in warning_msg
            )

    def test_empty_config(self):
        """Empty config passes through cleanly."""
        empty_config = {}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = migrate_config(empty_config)

        assert len(w) == 0
        assert result == {}

    def test_no_model_section(self):
        """Config without model section passes through cleanly."""
        no_model_config = {"stages": {"ingest": {"chunk_size": 4000}}}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = migrate_config(no_model_config)

        assert len(w) == 0
        assert result == no_model_config

    def test_preserves_all_model_fields(self):
        """All model fields are preserved during migration."""
        v1_config = {
            "model": {
                "provider": "openai",
                "name": "gpt-4o-mini",
                "base_url": "https://api.openai.com/v1",
                "api_key": "sk-xxx",
                "temperature": 0.7,
                "timeout": 120.0,
                "context_window": 128000,
            }
        }

        result = migrate_config(v1_config)

        # All original fields should be preserved
        model = result["model"]
        assert model["base_url"] == "https://api.openai.com/v1"
        assert model["api_key"] == "sk-xxx"
        assert model["temperature"] == 0.7
        assert model["timeout"] == 120.0
        assert model["context_window"] == 128000

        # And providers structure added
        assert "providers" in model
