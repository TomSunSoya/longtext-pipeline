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
- `ocr`
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
    enabled: true

output:
  dir: ./artifacts

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
- `OPENAI_BASE_URL` — Custom API endpoint for non-OpenAI providers
- `LONGTEXT_MODEL_NAME` — Model name
- `LONGTEXT_MODEL_PROVIDER` — Provider identifier
- `LONGTEXT_OUTPUT_DIR` — Output directory override
- `LONGTEXT_PROMPTS_DIR` — Custom prompts directory
- `LONGTEXT_LOG_LEVEL` — Log verbosity
- `LONGTEXT_LOG_FORMAT` — Log format (`text` or `json`)
- `LONGTEXT_LOG_FILE` — Log file path

The default model is `gpt-4o-mini`. If you set `OPENAI_BASE_URL` to a non-OpenAI endpoint such as DeepSeek or Ollama, also set `LONGTEXT_MODEL_NAME` to a model that provider actually serves.

Example for DeepSeek:

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
- audit enablement and prompt template selection

### `prompts`

Holds prompt-template metadata.

Important:

- The packaged runtime includes built-in prompt templates under `longtext_pipeline/prompts/`.
- Runtime stages load bundled templates by mode unless overridden.
- `audit_reporting.py` now builds token-budgeted audit prompts instead of concatenating unbounded source and analysis text.

### `output`

`output.dir` is validated and created during config loading.

The built-in default is `./output`.

In the standard single-file pipeline path:

- part, summary, stage, final-analysis, and metrics files are written under `<output.dir>/.longtext/`
- manifest and lock files remain next to the input file in its local `.longtext/`

For batch runs, generated artifacts are namespaced per input file under the configured base directory, for example `<output.dir>/report_a1b2c3d4/.longtext/`.

### `input`

Holds source-level settings such as encoding and related preprocessing controls.

### `pipeline`

Current notable keys:

- `allow_resume`
- `audit_enabled`
- `max_workers`
- `specialist_count`

### `ocr`

Holds OCR-related controls used by the extraction fallback path.

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
- `longtext run` supports `.txt`, `.md`, `.pdf`, and `.docx`; use `.pdf` and `.docx` when you want ingest to extract document text for you.
