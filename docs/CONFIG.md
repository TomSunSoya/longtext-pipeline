# Configuration Reference

`longtext-pipeline` uses YAML configuration plus environment-variable overrides.

## Resolution order

Runtime config precedence, from lowest to highest:

1. Built-in defaults
2. Explicit config passed with `--config`
3. Auto-discovered local config
4. Environment variable overrides

Auto-discovered local config filenames:

- `longtext.local.yaml`
- `.longtext.local.yaml`

## Top-level sections

Supported sections in the current schema:

- `model`
- `stages`
- `prompts`
- `output`
- `input`
- `pipeline`
- `logging`
- `agents`

Unknown keys currently warn rather than fail.

## Example

```yaml
model:
  provider: openai
  name: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}
  temperature: 0.7
  timeout: 120.0

stages:
  ingest:
    chunk_size: 4000
    overlap_rate: 0.1
  summarize:
    batch_size: 4
  stage:
    group_size: 5
  audit:
    enabled: false

pipeline:
  allow_resume: true
  max_workers: 4
  specialist_count: 4

logging:
  level: INFO
  format: text
```

## Environment variables

The runtime recognizes these overrides:

- `OPENAI_API_KEY` — API key for authentication
- `OPENAI_BASE_URL` — Custom API endpoint (for non-OpenAI providers)
- `LONGTEXT_MODEL_NAME` — Model name (required if using non-OpenAI providers)
- `LONGTEXT_MODEL_PROVIDER` — Provider identifier
- `LONGTEXT_OUTPUT_DIR` — Output directory
- `LONGTEXT_PROMPTS_DIR` — Custom prompts directory
- `LONGTEXT_LOG_LEVEL` — Log verbosity
- `LONGTEXT_LOG_FORMAT` — Log format (`text` or `json`)
- `LONGTEXT_LOG_FILE` — Log file path

**Important**: The default model is `gpt-4o-mini`. If you set `OPENAI_BASE_URL` to a non-OpenAI endpoint (e.g., DeepSeek, Ollama), you must also set `LONGTEXT_MODEL_NAME` to a model supported by that provider.

**Example for DeepSeek**:
```bash
export OPENAI_API_KEY="your-deepseek-api-key"
export OPENAI_BASE_URL="https://api.deepseek.com"
export LONGTEXT_MODEL_NAME="deepseek-chat"
```

## Section notes

### `model`

Controls provider, model name, API endpoint, timeout, temperature, and context window.

### `stages`

Controls stage-specific behavior such as:

- ingest chunk size and overlap
- summarize batch size
- stage group size
- audit enablement

### `prompts`

Holds prompt-template metadata.

Important:

- The packaged runtime includes built-in prompt templates under `longtext_pipeline/prompts/`.
- The current pipeline stages load bundled templates by mode.
- `prompts.dir` and stage `prompt_template` fields are most useful for advanced overrides, validation, and repository-based development workflows.

### `output`

The schema includes output-related keys, but the current runtime still writes its working directory next to the input file in `.longtext/`.

Treat this block as partially implemented metadata rather than a guaranteed output router.

### `pipeline`

Current notable keys:

- `allow_resume`
- `audit_enabled`
- `max_workers`
- `specialist_count`

### `agents`

Lets advanced users override model settings per role, for example:

- `summarizer`
- `stage_synthesizer`
- `analyst`
- `auditor`
- `topic_analyst`
- `entity_analyst`
- `sentiment_analyst`
- `timeline_analyst`

## Practical guidance

- For normal local use, keep secrets in `longtext.local.yaml` and commit only reusable example configs.
- If you just want the default prompts, do not change `prompts.dir`.
- If you distribute a packaged install, verify the wheel includes prompt `.txt` files.
