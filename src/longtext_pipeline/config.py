"""YAML configuration loader for the longtext pipeline.

This module provides configuration loading, validation, and environment variable
override support following the hierarchical approach with defaults and env overrides.
"""

import os
import re
import warnings
from pathlib import Path
from typing import Any, Optional
import yaml  # type: ignore[import-untyped]


AUTO_CONFIG_FILENAMES = (
    "longtext.local.yaml",
    ".longtext.local.yaml",
)
DEFAULT_PROMPTS_DIR = str((Path(__file__).resolve().parent / "prompts").resolve())


def migrate_config(config: dict, source_path: Optional[str | Path] = None) -> dict:
    """Migrate config from v1 flat format to v2 nested format.

       This function detects v1-style configurations and converts them to v2 format
       while issuing deprecation warnings. The migration is applied at runtime only
       - user config files are NOT modified.

       Args:
           config: Configuration dictionary to migrate.
           source_path: Path to source config file for warnings (optional).

       Returns:
           Migrated configuration dictionary. The original config dict is not modified.

       Notes:
           - v1 format uses flat `model.name` structure
           - v2 format uses nested `model.providers` structure
        - All migrations create copies to avoid mutating original data
        - Deprecation warnings alert users to modernize their configs

    Examples:
        >>> v1_config = {"model": {"name": "gpt-4o", "provider": "openai"}}
        >>> migrated = migrate_config(v1_config)
        >>> "providers" in migrated["model"]
        True
    """

    # Create a deep copy to avoid modifying the original
    migrated = _deep_copy(config)

    source_hint = f" (from {source_path})" if source_path else ""

    # Check for v1 flat model format (model.name with no model.providers)
    if "model" in migrated:
        model_config = migrated["model"]

        # If model has a "name" field but NO nested "providers" structure, it's v1 format
        if "name" in model_config and "providers" not in model_config:
            warnings.warn(
                f"Config uses deprecated flat model format. "
                f"Convert to nested 'model.providers' structure for v2 compatibility.{source_hint}",
                DeprecationWarning,
                stacklevel=3,
            )

            # Extract current model fields for use in new structure
            current_provider = model_config.get("provider", "openai")
            current_name = model_config.get("name")

            # Build the nested providers structure
            # Keep all other model fields (base_url, api_key, temperature, etc.)
            providers_dict = {
                "default": {
                    "provider": current_provider,
                    "name": current_name,
                }
            }

            # Preserve other model fields that aren't part of provider spec
            preserved_fields = {}
            for key, value in model_config.items():
                if key not in ("provider", "name"):
                    preserved_fields[key] = value

            # Rebuild model config with providers structure
            model_config["providers"] = providers_dict
            # Add preserved fields back to top-level model
            for key, value in preserved_fields.items():
                model_config[key] = value

    # Check for v1 agent config format (agents with model: provider,name format)
    # v2 format supports nested "model.providers" structure
    if "agents" in migrated:
        agents_config = migrated["agents"]
        # Process each agent to migrate legacy model config to providers format if needed
        for agent_key, agent_config in agents_config.items():
            if (
                isinstance(agent_config, dict)
                and "model" in agent_config
                and isinstance(agent_config["model"], dict)
            ):
                model_spec = agent_config["model"]
                # Check if this looks like legacy format (has name/key without providers struct)
                # and no existing providers section (meaning it hasn't already been migrated)
                if "name" in model_spec and "providers" not in model_spec:
                    # Migrate this agent's model config to new format
                    source_name = f"'agents.{agent_key}'"
                    warnings.warn(
                        f"Agents configuration '{source_name}' uses deprecated format. "
                        f"Convert to nested 'providers' structure for v2 compatibility.{source_hint}",
                        DeprecationWarning,
                        stacklevel=3,
                    )

                    current_provider = model_spec.get("provider", "openai")
                    current_name = model_spec.get("name")

                    if current_name:  # Only migrate if there's a name
                        providers_dict = {
                            "default": {
                                "provider": current_provider,
                                "name": current_name,
                            }
                        }

                        # Preserve other model fields that aren't part of provider spec
                        preserved_fields = {}
                        for key, value in model_spec.items():
                            if key not in ("provider", "name"):
                                preserved_fields[key] = value

                        # Update model config with providers structure
                        new_model_spec = {**model_spec}
                        new_model_spec["providers"] = providers_dict

                        # Clean up the preserved fields
                        for key in ("provider", "name"):
                            new_model_spec.pop(key, None)  # Remove old fields

                        # Add preserved fields back
                        new_model_spec.update(preserved_fields)

                        # Replace the model configuration in agent config
                        agent_config["model"] = new_model_spec
                        migrated["agents"][agent_key] = (
                            agent_config  # Update the master copy
                        )

    return migrated


class ConfigError(Exception):
    """Raised when configuration loading or validation fails."""


# Default configuration values (MVP)
DEFAULT_CONFIG = {
    "model": {
        "provider": "openai",
        "name": "gpt-4o-mini",
        "base_url": None,  # Will default to provider-specific base URL
        "api_key": None,  # Will be overridden by env var
        "temperature": 0.7,
        "timeout": 120.0,
        "context_window": 128000,  # For GPT-4o-mini model as default
        "providers": {
            "default": {
                "provider": "openai",
                "name": "gpt-4o-mini",
                "base_url": None,
                "api_key": None,
                "temperature": 0.7,
                "timeout": 120.0,
            }
        },
        "dispatch_mode": "parallel",  # Options: 'single', 'parallel', 'fastest', 'ranked'
    },
    "stages": {
        "ingest": {
            "chunk_size": 4000,  # Approximately 1000-1500 tokens
            "overlap_rate": 0.1,
        },
        "summarize": {
            "prompt_template": "prompts/summary_general.txt",
            "batch_size": 4,
        },
        "stage": {
            "group_size": 5,  # Combine every 5 summaries into 1 stage file
            "prompt_template": "prompts/stage_general.txt",
        },
        "final": {
            "prompt_template": "prompts/final_general.txt",
        },
        "audit": {
            "enabled": True,
            "prompt_template": "prompts/audit_general.txt",
        },
    },
    "prompts": {
        "dir": DEFAULT_PROMPTS_DIR,
        "format": "general",  # Alternative: "relationship"
    },
    "output": {
        "dir": "./output",
        "naming": {
            "summarize_prefix": "summary_",
            "stage_prefix": "stage_",
            "final_filename": "final_analysis.md",
        },
        "save_intermediate": True,
    },
    "input": {
        "file_path": None,  # Must be provided in final config
        "encoding": "utf-8",
    },
    "pipeline": {
        "allow_resume": True,
        "audit_enabled": True,
        "max_workers": 4,
        "specialist_count": 4,
    },
    "ocr": {
        "enabled": False,
        "paddle_api_token": None,  # Will be overridden by env vars
        "paddle_api_url": "https://kbierdt4sav0zbee.aistudio-app.com/layout-parsing",
        "use_local_fallback": True,
        "api_failures_before_fallback": 1,
        "threshold_token_ratio": 0.05,
    },
    "logging": {
        "level": "INFO",
        "format": "text",
        "file": None,
    },
    "agents": {
        "summarizer": {
            "model": None,  # None means use top-level model config
        },
        "stage_synthesizer": {
            "model": None,
        },
        "analyst": {
            "model": None,
        },
        "auditor": {
            "model": None,
        },
        "topic_analyst": {
            "model": None,
        },
        "entity_analyst": {
            "model": None,
        },
        "sentiment_analyst": {
            "model": None,
        },
        "timeline_analyst": {
            "model": None,
        },
    },
}


def load_config(path: Optional[str] = None) -> dict:
    """Load configuration from YAML file or return defaults.

    Args:
        path: Path to YAML config file. If None, returns defaults only.

    Returns:
        Dict containing the merged configuration with defaults applied.

    Notes:
        - If path is provided, loads YAML and merges with defaults
        - If path is None, returns DEFAULT_CONFIG
        - Environment variable overrides are NOT applied by this function
          Callers should call merge_env_overrides() after loading.
    """
    if path is None:
        result = _deep_copy(DEFAULT_CONFIG)
        return result  # type: ignore[return-value,no-any-return]

    loaded = _load_yaml_file(path)
    # Migrate config from v1 to v2 format if needed
    migrated = migrate_config(loaded, source_path=path)
    # Merge migrated config with defaults (migrated takes precedence)
    return _deep_merge(DEFAULT_CONFIG, migrated)


def _load_yaml_file(path: str | Path) -> dict:
    """Load a YAML config file without applying defaults."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
    except FileNotFoundError:
        raise ConfigError(f"Configuration file not found: {path}")
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in configuration file {path}: {e}")

    if not isinstance(loaded, dict):
        raise ConfigError(
            f"Configuration file must contain a YAML object at the top level: {path}"
        )

    return loaded


def find_auto_config_path(start_dir: Optional[str | Path] = None) -> Optional[Path]:
    """Search current directory and its parents for a local auto-loaded config file."""
    current_dir = Path(start_dir or Path.cwd()).resolve()

    for directory in [current_dir, *current_dir.parents]:
        for filename in AUTO_CONFIG_FILENAMES:
            candidate = directory / filename
            if candidate.exists() and candidate.is_file():
                return candidate

    return None


def load_runtime_config(
    path: Optional[str] = None,
    search_dir: Optional[str | Path] = None,
) -> tuple[dict, list[str]]:
    """Load runtime config from defaults, explicit config, local config, and env vars.

    Precedence from low to high:
    1. Built-in defaults
    2. Explicit config passed via ``path``
    3. Auto-discovered local config (for machine-local settings and secrets)
    4. Environment variable overrides
    """
    config = _deep_copy(DEFAULT_CONFIG)
    loaded_sources: list[str] = []
    explicit_path = Path(path).resolve() if path else None

    if explicit_path is not None:
        config = _deep_merge(config, _load_yaml_file(explicit_path))
        loaded_sources.append(str(explicit_path))

    auto_path = find_auto_config_path(search_dir)
    if auto_path is not None and (
        explicit_path is None or auto_path.resolve() != explicit_path
    ):
        config = _deep_merge(config, _load_yaml_file(auto_path))
        loaded_sources.append(str(auto_path))

    config = merge_env_overrides(config)
    validate_config(config)
    return config, loaded_sources


def get_missing_required_settings(config: dict) -> list[str]:
    """Return required runtime settings still missing after config resolution."""
    missing: list[str] = []
    model_config = config.get("model", {})

    if not str(model_config.get("api_key") or "").strip():
        missing.append("model.api_key")

    return missing


def format_missing_settings_message(missing: list[str]) -> str:
    """Build a user-facing error message for missing required settings."""
    if not missing:
        return ""

    suggestions = {
        "model.api_key": "set `model.api_key` in `longtext.local.yaml` or export `OPENAI_API_KEY`",
    }
    detail = "; ".join(suggestions.get(item, item) for item in missing)
    return f"Missing required configuration: {', '.join(missing)}. Please {detail}."


def _validate_output_dir(config: dict) -> None:
    """Validate output directory configuration.

    Args:
        config: Configuration dictionary containing output section.

    Raises:
        ConfigError: If output.dir is not writable or cannot be created.
    """
    output = config.get("output", {})
    output_dir = output.get("dir", DEFAULT_CONFIG["output"]["dir"])

    if output_dir is None:
        # Use default if not specified
        output_dir = DEFAULT_CONFIG["output"]["dir"]

    output_path = Path(output_dir).resolve()

    # Create directory if it doesn't exist
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        raise ConfigError(f"Cannot create output directory '{output_dir}': {e}")

    # Check if directory is writable
    test_file = output_path / ".longtext_write_test"
    try:
        test_file.touch()
        test_file.unlink()
    except (OSError, PermissionError) as e:
        raise ConfigError(f"Output directory '{output_dir}' is not writable: {e}")


def validate_config(config: dict) -> bool:
    """Validate configuration structure and warn on unknown keys.

    Args:
        config: Configuration dictionary to validate.

    Returns:
        True if validation passes (warnings are non-fatal).

    Notes:
        - Unknown keys at any level generate warnings but do not fail
        - Missing required fields use defaults (no required fields in MVP)
        - Type checking is minimal (config is validated at usage time)
    """
    # Known top-level keys
    known_top_keys = {
        "model",
        "stages",
        "prompts",
        "output",
        "input",
        "pipeline",
        "ocr",  # Add ocr to the set of known top-level configuration keys
        "logging",
        "agents",
    }

    # Known nested keys per section
    known_model_keys = {
        "provider",
        "name",
        "base_url",
        "api_key",
        "temperature",
        "timeout",
        "context_window",
        "providers",
        "dispatch_mode",
    }
    known_ingest_keys = {"chunk_size", "overlap_rate"}
    known_summarize_keys = {"prompt_template", "batch_size"}
    known_stage_keys = {"group_size", "prompt_template"}
    known_final_keys = {"prompt_template"}
    known_audit_keys = {"enabled", "prompt_template"}
    known_prompts_keys = {"dir", "format"}
    known_output_keys = {"dir", "naming", "save_intermediate"}
    known_naming_keys = {"summarize_prefix", "stage_prefix", "final_filename"}

    # Validate output directory (must exist and be writable)
    _validate_output_dir(config)

    known_input_keys = {"file_path", "encoding"}
    known_pipeline_keys = {
        "allow_resume",
        "audit_enabled",
        "max_workers",
        "specialist_count",
    }
    known_ocr_keys = {
        "enabled",
        "paddle_api_token",
        "paddle_api_url",
        "use_local_fallback",
        "api_failures_before_fallback",
        "threshold_token_ratio",
    }
    known_logging_keys = {"level", "format", "file"}

    # Check top-level keys
    for key in config:
        if key not in known_top_keys:
            warnings.warn(f"Unknown configuration key at top level: '{key}'")

    # Validate model section
    if "model" in config:
        for key in config["model"]:
            if key not in known_model_keys:
                warnings.warn(f"Unknown configuration key in model: '{key}'")

            # Validate timeout if present
            if "timeout" in config["model"]:
                timeout = config["model"]["timeout"]
                # Type check: must be int or float
                if not isinstance(timeout, (int, float)):
                    raise ConfigError(
                        f"model.timeout must be int or float, got {type(timeout).__name__}"
                    )
                # Upper bound: error if > 600.0s
                if timeout > 600.0:
                    raise ConfigError(
                        f"model.timeout must not exceed 600.0 seconds (got {timeout})"
                    )
                # Lower bound: warning if < 5.0s
                if timeout < 5.0:
                    warnings.warn(
                        f"model.timeout is very low ({timeout}s). "
                        "Consider using at least 5.0 seconds for reliability."
                    )

            # Validate context_window if present
            if "context_window" in config["model"]:
                context_window = config["model"]["context_window"]
                # Type check: must be int
                if not isinstance(context_window, int):
                    raise ConfigError(
                        f"model.context_window must be int, got {type(context_window).__name__}"
                    )
                # Lower bound: warning if < 4096 (very small context)
                if context_window < 4096:
                    warnings.warn(
                        f"model.context_window is very small ({context_window}). "
                        "Consider using at least 8192 tokens for typical use cases."
                    )
                # Upper bound: no limit enforced since modern models can have large contexts

            # Validate providers if present
            if "providers" in config["model"]:
                providers = config["model"]["providers"]
                if not isinstance(providers, dict):
                    raise ConfigError("model.providers must be a dictionary")

                # Check each provider
                for provider_key, provider_config in providers.items():
                    if not isinstance(provider_config, dict):
                        raise ConfigError(
                            f"Each provider in model.providers must be a dictionary, found {type(provider_config).__name__}"
                        )

                    # Check known keys for provider configuration
                    known_provider_keys = {
                        "provider",
                        "name",
                        "base_url",
                        "api_key",
                        "temperature",
                        "timeout",
                        "context_window",
                    }
                    for key in provider_config:
                        if key not in known_provider_keys:
                            warnings.warn(
                                f"Unknown configuration key in model.providers.{provider_key}: '{key}'"
                            )

            # Validate dispatch_mode if present
            if "dispatch_mode" in config["model"]:
                dispatch_mode = config["model"]["dispatch_mode"]
                if dispatch_mode not in ("single", "parallel", "fastest", "ranked"):
                    warnings.warn(
                        f"model.dispatch_mode is '{dispatch_mode}', but should be one of: 'single', 'parallel', 'fastest', 'ranked'"
                    )

    # Validate stages section
    if "stages" in config:
        stages = config["stages"]

        if "ingest" in stages:
            for key in stages["ingest"]:
                if key not in known_ingest_keys:
                    warnings.warn(
                        f"Unknown configuration key in stages.ingest: '{key}'"
                    )

        if "summarize" in stages:
            for key in stages["summarize"]:
                if key not in known_summarize_keys:
                    warnings.warn(
                        f"Unknown configuration key in stages.summarize: '{key}'"
                    )

        if "stage" in stages:
            for key in stages["stage"]:
                if key not in known_stage_keys:
                    warnings.warn(f"Unknown configuration key in stages.stage: '{key}'")

        if "final" in stages:
            for key in stages["final"]:
                if key not in known_final_keys:
                    warnings.warn(f"Unknown configuration key in stages.final: '{key}'")

        if "audit" in stages:
            audit = stages["audit"]
            for key in audit:
                if key not in known_audit_keys:
                    warnings.warn(f"Unknown configuration key in stages.audit: '{key}'")

            # Validate audit.enabled type (should be boolean)
            if "enabled" in audit:
                enabled = audit["enabled"]
                if not isinstance(enabled, bool):
                    warnings.warn(
                        f"stages.audit.enabled must be boolean, got {type(enabled).__name__}"
                    )

            # Validate audit.prompt_template type and existence
            if "prompt_template" in audit:
                prompt_template = audit["prompt_template"]
                if not isinstance(prompt_template, str):
                    warnings.warn(
                        f"stages.audit.prompt_template must be a string path, got {type(prompt_template).__name__}"
                    )
                # If audit.enabled=true, check prompt_template file exists
                elif audit.get("enabled") is True:
                    prompts_dir = config.get("prompts", {}).get(
                        "dir", DEFAULT_PROMPTS_DIR
                    )
                    # Extract just the filename from prompt_template (e.g., "prompts/audit_general.txt" → "audit_general.txt")
                    template_filename = Path(prompt_template).name
                    template_path = Path(prompts_dir) / template_filename
                    if not template_path.exists():
                        warnings.warn(
                            f"stages.audit.prompt_template file not found: '{template_path}'"
                        )
            else:
                # Warning if enabled=true but no prompt_template specified
                if audit.get("enabled") is True:
                    warnings.warn(
                        "stages.audit.enabled=true but stages.audit.prompt_template is not specified"
                    )

    # Validate prompts section
    if "prompts" in config:
        for key in config["prompts"]:
            if key not in known_prompts_keys:
                warnings.warn(f"Unknown configuration key in prompts: '{key}'")

    # Validate output section
    if "output" in config:
        output = config["output"]

        for key in output:
            if key not in known_output_keys:
                warnings.warn(f"Unknown configuration key in output: '{key}'")

        if "naming" in output:
            for key in output["naming"]:
                if key not in known_naming_keys:
                    warnings.warn(
                        f"Unknown configuration key in output.naming: '{key}'"
                    )

    # Validate input section
    if "input" in config:
        for key in config["input"]:
            if key not in known_input_keys:
                warnings.warn(f"Unknown configuration key in input: '{key}'")

    # Validate pipeline section
    if "pipeline" in config:
        for key in config["pipeline"]:
            if key not in known_pipeline_keys:
                warnings.warn(f"Unknown configuration key in pipeline: '{key}'")

    # Validate OCR section
    if "ocr" in config:
        for key in config["ocr"]:
            if key not in known_ocr_keys:
                warnings.warn(f"Unknown configuration key in ocr: '{key}'")
        # Validate OCR enabled flag type
        if "enabled" in config["ocr"]:
            enabled = config["ocr"]["enabled"]
            if not isinstance(enabled, bool):
                warnings.warn(
                    f"ocr.enabled must be boolean, got {type(enabled).__name__}"
                )
        # Validate OCR use_local_fallback flag type
        if "use_local_fallback" in config["ocr"]:
            use_local = config["ocr"]["use_local_fallback"]
            if not isinstance(use_local, bool):
                warnings.warn(
                    f"ocr.use_local_fallback must be boolean, got {type(use_local).__name__}"
                )
        # Validate OCR api_failures_before_fallback type
        if "api_failures_before_fallback" in config["ocr"]:
            failures = config["ocr"]["api_failures_before_fallback"]
            if not isinstance(failures, int) or failures < 1:
                warnings.warn(
                    f"ocr.api_failures_before_fallback must be positive integer, got {failures}"
                )
        # Validate threshold_token_ratio type and range
        if "threshold_token_ratio" in config["ocr"]:
            threshold = config["ocr"]["threshold_token_ratio"]
            if (
                not isinstance(threshold, (int, float))
                or threshold < 0
                or threshold > 1
            ):
                warnings.warn(
                    f"ocr.threshold_token_ratio must be float between 0 and 1, got {threshold}"
                )

    # Validate logging section
    if "logging" in config:
        for key in config["logging"]:
            if key not in known_logging_keys:
                warnings.warn(f"Unknown configuration key in logging: '{key}'")

    # Validate agents section
    if "agents" in config:
        agents = config["agents"]
        known_agent_keys = {
            "summarizer",
            "stage_synthesizer",
            "analyst",
            "auditor",
            "topic_analyst",
            "entity_analyst",
            "sentiment_analyst",
            "timeline_analyst",
        }
        known_agent_model_keys = {"model"}

        for agent_name, agent_config in agents.items():
            if agent_name not in known_agent_keys:
                warnings.warn(
                    f"Unknown agent type in agents: '{agent_name}'. "
                    f"Supported: {', '.join(known_agent_keys)}"
                )
                continue

            if not isinstance(agent_config, dict):
                warnings.warn(
                    f"Agent '{agent_name}' config must be a dictionary, got {type(agent_config)}"
                )
                continue

            for key in agent_config:
                if key not in known_agent_model_keys:
                    warnings.warn(
                        f"Unknown configuration key in agents.{agent_name}: '{key}'"
                    )

    return True


def _substitute_env_vars(value: Any) -> Any:
    """Substitute environment variable placeholders in a value.

    Supports ${VAR_NAME} and ${VAR_NAME:-default} syntax.

    Args:
        value: Value that may contain environment variable references.

    Returns:
        Value with environment variables substituted.
    """
    if not isinstance(value, str):
        return value

    # Pattern for ${VAR_NAME:-default} or ${VAR_NAME}
    pattern = r"\$\{([^}:]+)(?::-([^}]*))?\}"

    def replace_match(match):
        var_name = match.group(1)
        default_value = match.group(2)
        env_value = os.environ.get(var_name)
        if env_value is not None:
            return env_value
        elif default_value is not None:
            return default_value
        else:
            # Return empty string if env var not found and no default
            return ""

    return re.sub(pattern, replace_match, value)


def get_agent_model_config(config: dict, agent_type: str) -> dict:
    """Get model configuration for a specific agent type.

    Agent-specific configurations take precedence over top-level model config.
    When agent-specific model is None or not provided, falls back to top-level
    model configuration.

    Args:
        config: Full configuration dictionary.
        agent_type: Agent type name (e.g., 'summarizer', 'analyst', 'topic_analyst').

    Returns:
        Dictionary with model configuration for the specified agent.
        If agent-specific config missing, returns deep copy of top-level model config.

    Raises:
        ConfigError: If agent_type is not a known agent type.
    """
    known_agents: set[str] = {
        "summarizer",
        "stage_synthesizer",
        "analyst",
        "auditor",
        "topic_analyst",
        "entity_analyst",
        "sentiment_analyst",
        "timeline_analyst",
    }

    if agent_type not in known_agents:
        raise ConfigError(
            f"Unknown agent type '{agent_type}'. Supported: {', '.join(known_agents)}"
        )

    agents_config = config.get("agents", {})
    agent_config = agents_config.get(agent_type, {})

    # Check for the new providers configuration format
    providers_config = config.get("model", {}).get("providers", {})

    # If agent has explicit model config using providers, return it
    if agent_config and "model" in agent_config and agent_config["model"] is not None:
        model_config = agent_config["model"]
        # If model_config has provider reference, resolve it from providers
        if isinstance(model_config, str) and model_config in providers_config:
            resolved_model_config = _deep_copy(providers_config[model_config])
        else:
            resolved_model_config = _deep_copy(model_config)
        return resolved_model_config  # type: ignore[return-value,no-any-return]

    # Check if there's default provider configuration for this agent type
    default_provider_key = agent_config.get("default_provider")
    if default_provider_key and default_provider_key in providers_config:
        resolved_model_config = _deep_copy(providers_config[default_provider_key])
        return resolved_model_config  # type: ignore[return-value,no-any-return]

    # Fall back to top-level model config
    model_config = config.get("model", {})
    result = _deep_copy(model_config)
    return result  # type: ignore[return-value,no-any-return]


def get_agent_provider_configs(config: dict, agent_type: str) -> list:
    """Get list of provider configurations for multi-provider mode per agent.

    Args:
        config: Full configuration dictionary.
        agent_type: Agent type name (e.g., 'summarizer', 'analyst', 'topic_analyst').

    Returns:
        List of provider configuration dictionaries to be used in parallel
    """
    known_agents: set[str] = {
        "summarizer",
        "stage_synthesizer",
        "analyst",
        "auditor",
        "topic_analyst",
        "entity_analyst",
        "sentiment_analyst",
        "timeline_analyst",
    }

    if agent_type not in known_agents:
        raise ConfigError(
            f"Unknown agent type '{agent_type}'. Supported: {', '.join(known_agents)}"
        )

    agents_config = config.get("agents", {})
    agent_config = agents_config.get(agent_type, {})

    # Check for multi-provider configuration for this agent
    provider_configs = agent_config.get("providers", [])

    if provider_configs and isinstance(provider_configs, list):
        # This is a multi-provider configuration
        return _deep_copy(provider_configs)

    # Check if agent has a provider reference that maps to a main providers section
    model_name = agent_config.get("model")
    if model_name and isinstance(model_name, str):
        main_providers = config.get("model", {}).get("providers", {})
        if model_name in main_providers:
            # Single provider specified in main providers section - return as single-item list for compatibility
            provider_config = main_providers[model_name].copy()
            provider_config["provider"] = provider_config.get("provider", "openai")
            return [provider_config]

    # Check if agent config has providers specified directly in its config as array
    if "model" in agent_config and isinstance(agent_config["model"], dict):
        agent_model_config = agent_config["model"]
        agent_providers = agent_model_config.get("providers", {})

        # If agents specify a specific provider name in their config
        agent_provider_ref = agent_model_config.get("provider_reference")
        if agent_provider_ref and agent_provider_ref in agent_providers:
            provider_config = agent_providers[agent_provider_ref].copy()
            return [provider_config]

        # If agent specifies "providers" as a list of configurations
        if "providers" in agent_model_config and isinstance(
            agent_model_config["providers"], list
        ):
            return _deep_copy(agent_model_config["providers"])

    # Fall back to single provider using top-level config and agent-specific override
    model_config = get_agent_model_config(config, agent_type)
    model_config["provider"] = model_config.get("provider", "openai")
    return [model_config]


def merge_env_overrides(config: dict) -> dict:
    """Apply environment variable overrides to configuration.

    Environment variables take precedence over YAML config values.
    Supports the following variables:

    - OPENAI_API_KEY: API key for OpenAI service
    - OPENAI_BASE_URL: Base URL for OpenAI API (proxy/custom endpoint)
    - LONGTEXT_MODEL_PROVIDER: Default model provider override
    - LONGTEXT_MODEL_NAME: Default model name override
    - LONGTEXT_OUTPUT_DIR: Default output directory override
    - LONGTEXT_PROMPTS_DIR: Default prompts directory override
    - LONGTEXT_LOG_LEVEL: Logging level override
    - LONGTEXT_LOG_FORMAT: Logging format override ('text' or 'json')
    - LONGTEXT_LOG_FILE: Log file path override

    Also performs ${VAR_NAME} and ${VAR_NAME:-default} substitution in string values.

    Args:
        config: Configuration dictionary to override.

    Returns:
        New configuration dict with environment overrides applied.
        The original config dict is not modified.
    """
    result = _deep_copy(config)

    # Model API configuration
    if "model" not in result:
        result["model"] = {}

    # OPENAI_API_KEY overrides model.api_key
    api_key_override = os.environ.get("OPENAI_API_KEY")
    if api_key_override:
        result["model"]["api_key"] = api_key_override

    # OPENAI_BASE_URL overrides model.base_url
    base_url_override = os.environ.get("OPENAI_BASE_URL")
    if base_url_override:
        result["model"]["base_url"] = base_url_override

    # Provider override
    provider_override = os.environ.get("LONGTEXT_MODEL_PROVIDER")
    if provider_override:
        if "model" not in result:
            result["model"] = {}
        result["model"]["provider"] = provider_override

    # Model name override
    name_override = os.environ.get("LONGTEXT_MODEL_NAME")
    if name_override:
        if "model" not in result:
            result["model"] = {}
        result["model"]["name"] = name_override

    # Output directory override
    output_dir_override = os.environ.get("LONGTEXT_OUTPUT_DIR")
    if output_dir_override:
        if "output" not in result:
            result["output"] = {}
        result["output"]["dir"] = output_dir_override

    # Prompts directory override
    prompts_dir_override = os.environ.get("LONGTEXT_PROMPTS_DIR")
    if prompts_dir_override:
        if "prompts" not in result:
            result["prompts"] = {}
        result["prompts"]["dir"] = prompts_dir_override

    # Logging overrides
    if "logging" not in result:
        result["logging"] = {}

    log_level_override = os.environ.get("LONGTEXT_LOG_LEVEL")
    if log_level_override:
        result["logging"]["level"] = log_level_override

    log_format_override = os.environ.get("LONGTEXT_LOG_FORMAT")
    if log_format_override:
        result["logging"]["format"] = log_format_override

    log_file_override = os.environ.get("LONGTEXT_LOG_FILE")
    if log_file_override:
        result["logging"]["file"] = log_file_override

    # Apply env var substitution to all string values
    result = _substitute_env_vars_recursive(result)

    return result  # type: ignore[return-value,no-any-return]


def _substitute_env_vars_recursive(obj: Any) -> Any:
    """Recursively substitute environment variables in a nested structure."""
    if isinstance(obj, dict):
        return {k: _substitute_env_vars_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_substitute_env_vars_recursive(item) for item in obj]
    elif isinstance(obj, str):
        return _substitute_env_vars(obj)
    else:
        return obj


def _deep_copy(obj: Any) -> Any:
    """Create a deep copy of a nested dict/list structure."""
    if isinstance(obj, dict):
        return {k: _deep_copy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_deep_copy(item) for item in obj]  # type: ignore[misc]
    else:
        return obj


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base (override takes precedence).

    Args:
        base: Base configuration dictionary.
        override: Override configuration dictionary.

    Returns:
        New dict with override values merged into base.
    """
    result: dict = _deep_copy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = _deep_copy(value)  # type: ignore[assignment]

    return result  # type: ignore[return-value,no-any-return]
