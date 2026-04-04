# Configuration Specification

## Overview

The `longtext-pipeline` uses YAML configuration files to define processing parameters, model settings, and pipeline behavior. Configuration follows a hierarchical approach with environment variable overrides available for key parameters.

## File Locations

Configuration files are typically located in the `examples/` directory:

- `examples/config.general.yaml` - General text analysis configuration
- `examples/config.relationship.yaml` - Relationship-focused analysis configuration

The dual configuration approach supports the different analysis types as designed in the pipeline architecture, enabling both general text summarization and entity relationship mapping as outlined in the planned prompt templates (summary_general/relationship, stage_general/relationship, etc.).

## Configuration Structure

The configuration file uses the following top-level schema:

```yaml
model:
  provider: string          # Model provider (e.g. "openai", "openrouter", "ollama")
  name: string              # Model name (e.g. "gpt-4o-mini", "gpt-4o")
  base_url: string          # API endpoint URL (defaults to provider standard)
  api_key: string           # API key reference (supports environment variable substitution)
  temperature: number       # Generation temperature (0.0-2.0, default: 0.7)

stages:
  ingest:
    chunk_size: integer     # Text split size for initial fragments
    overlap_rate: number    # Overlap ratio between chunks (0.0-1.0)
    
  summarize:
    prompt_template: string # Path to summary prompt template
    batch_size: integer     # Number of parts processed concurrently
    
  stage:
    group_size: integer     # Number of summaries to combine in a stage
    prompt_template: string # Path to stage aggregation prompt template
    
  final:
    prompt_template: string # Path to final analysis prompt template

  audit:                    # Optional stage
    enabled: bool           # Whether to run post-processing audit
    prompt_template: string # Path to audit prompt template

prompts:
  dir: string               # Root directory for prompt templates
  format: string            # Prompt format ("general", "relationship")
  
output:
  dir: string               # Output directory for results
  naming:
    summarize_prefix: string # Filename prefix for summary outputs
    stage_prefix: string     # Filename prefix for stage outputs
    final_filename: string   # Filename for final output
  save_intermediate: bool    # Whether to keep intermediate files

input:
  file_path: string         # Path to input file to process
  encoding: string          # Text encoding (default: "utf-8")

pipeline:
  allow_resume: boolean     # Enable resumable processing
  audit_enabled: boolean    # Enable post-processing audit phase
  max_workers: integer      # Max concurrent workers for parallel operations
```

## Default Values

Default configuration values are applied throughout the system, eliminating required fields for MVP usage:

```yaml
model:
  provider: "openai"
  name: "gpt-4o-mini"
  base_url: null           # Will default to provider-specific base URL
  api_key: "${OPENAI_API_KEY}"  # Environment variable reference
  temperature: 0.7

stages:
  ingest:
    chunk_size: 4000       # Approximately 1000-1500 tokens
    overlap_rate: 0.1
    
  summarize:
    prompt_template: "prompts/summary_general.txt"  # Relative to prompts.dir
    batch_size: 4
    
  stage:
    group_size: 5          # Combine every 5 summaries into 1 stage file
    prompt_template: "prompts/stage_general.txt"
    
  final:
    prompt_template: "prompts/final_general.txt"

prompts:
  dir: "./src/longtext_pipeline/prompts"
  format: "general"        # Alternative: "relationship"
  
output:
  dir: "./output"
  naming:
    summarize_prefix: "summary_"
    stage_prefix: "stage_"
    final_filename: "final_analysis.md"
  save_intermediate: true

input:
  file_path: null          # Must be provided in final config
  encoding: "utf-8"

pipeline:
  allow_resume: true
  audit_enabled: false
  max_workers: 4
```

## Environment Variable Overrides

The configuration system supports the following environment variables as secure alternatives to specifying values in YAML:

- `OPENAI_API_KEY`: API key for OpenAI service
- `OPENAI_BASE_URL`: Base URL for OpenAI API (use for proxy/custom endpoint)
- `LONGTEXT_MODEL_PROVIDER`: Default model provider override
- `LONGTEXT_MODEL_NAME`: Default model name override
- `LONGTEXT_OUTPUT_DIR`: Default output directory override
- `LONGTEXT_PROMPTS_DIR`: Default prompts directory override

### Variable Substitution

Environment variables are referenced in YAML using `"${VARIABLE_NAME}"` syntax:

```yaml
model:
  api_key: "${OPENAI_API_KEY}"           # Substituted at runtime
  base_url: "${OPENAI_BASE_URL:-https://api.openai.com/v1}"  # With fallback

output:
  dir: "${LONGTEXT_OUTPUT_DIR:-./output}"
```

## Configuration Validation

### Unknown Key Handling

The configuration system implements a "fail-safe" validation approach:

- Unknown keys are logged as warnings, not errors
- Configuration process continues with unknown keys ignored
- This allows configuration files to maintain backward compatibility
- Users receive feedback about potentially unused or misspelled keys

### Validation Examples

Valid extra keys (produce warnings):
```yaml
model:
  provider: "openai"
  unknown_setting: "ignored"  # Warning logged: "unknown key: unknown_setting"
  name: "gpt-4o-mini"
```

Invalid value types are handled gracefully with fallback to defaults when possible.

## Supported Formats

Configuration supports two primary analysis formats:

### General Analysis (`format: "general"`)

Used for general-purpose text summarization and analysis:
- `prompts/summary_general.txt` - Basic summary generation
- `prompts/stage_general.txt` - Section combination
- `prompts/final_general.txt` - Final synthesis

### Relationship Analysis (`format: "relationship"`)

Optimized for entity relationships and connection mapping:
- `prompts/summary_relationship.txt` - Relationships summary
- `prompts/stage_relationship.txt` - Relationship clustering
- `prompts/final_relationship.txt` - Final relations analysis