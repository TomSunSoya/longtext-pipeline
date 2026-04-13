# longtext-pipeline

[![CI](https://github.com/TomSunSoya/longtext-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/TomSunSoya/longtext-pipeline/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A Python CLI tool for hierarchical analysis of super-long texts using LLMs.

**Problem**: Feeding massive documents directly to LLMs causes context overflow, hallucinations, and unauditable outputs.

**Solution**: A 5-stage pipeline that decomposes long texts into manageable chunks, processes them hierarchically, and synthesizes results with built-in traceability.

## Features

- **Hierarchical processing**: 5-stage pipeline (Ingest → Summarize → Stage → Final → Audit)
- **Resumable**: SHA-256-based checkpoint/resume — pick up where you left off
- **Continue-with-Partial**: Pipeline continues with available results when individual parts fail
- **Token budget management**: Automatic context window validation and prompt truncation
- **Streaming**: Real-time token streaming with progress callbacks
- **Cross-process locking**: File-level locking prevents concurrent runs on the same input
- **Observability**: Prometheus metrics, structured JSON logging, configurable log sinks
- **Dual modes**: General analysis and relationship-focused analysis
- **Multi-perspective**: Parallel specialist agents for richer final synthesis
- **Model-agnostic**: Any OpenAI-compatible API endpoint (OpenAI, OpenRouter, Ollama, vLLM, etc.)
- **Docker-ready**: Multi-stage Dockerfile with non-root user

## Prerequisites

- Python >= 3.10
- pip
- An OpenAI-compatible API key

## Installation

```bash
# Clone the repository
git clone https://github.com/TomSunSoya/longtext-pipeline.git
cd longtext-pipeline

# Install in editable mode (with dev dependencies)
pip install -e ".[dev]"

# Verify installation
longtext --version
```

## Quickstart

### 1. Set your API key

```bash
# Required
export OPENAI_API_KEY="sk-your-api-key-here"

# Optional: custom endpoint (Ollama, vLLM, DeepSeek, etc.)
export OPENAI_BASE_URL="https://your-endpoint/v1"

# Required if using non-OpenAI endpoints
export LONGTEXT_MODEL_NAME="your-model-name"
```

On Windows PowerShell:
```powershell
$env:OPENAI_API_KEY="sk-your-api-key-here"
$env:OPENAI_BASE_URL="https://your-endpoint/v1"
$env:LONGTEXT_MODEL_NAME="your-model-name"
```

**Note**: The default model is `gpt-4o-mini`. If you use a non-OpenAI provider (e.g., DeepSeek, Ollama), you must set `LONGTEXT_MODEL_NAME` to a model supported by your provider.

**Common providers**:
| Provider | OPENAI_BASE_URL | LONGTEXT_MODEL_NAME |
|----------|-----------------|---------------------|
| OpenAI | (default) | `gpt-4o-mini` (default) |
| DeepSeek | `https://api.deepseek.com` | `deepseek-chat` |
| Ollama | `http://localhost:11434/v1` | Your local model name |
| OpenRouter | `https://openrouter.ai/api/v1` | Provider-specific |

### 2. Run the pipeline

```bash
# Basic usage — analyze a text file
longtext run input.txt

# With a config file
longtext run input.txt --config examples/config.general.yaml

# Relationship analysis mode
longtext run input.txt --mode relationship

# Resume an interrupted run
longtext run input.txt --resume

# Multi-perspective analysis with 3 specialist agents
longtext run input.txt --multi-perspective --agent-count 3

# Control concurrency
longtext run input.txt --max-workers 2
```

### 3. Check results

```bash
# View processing status
longtext status input.txt

# Read the final analysis
cat .longtext/final_analysis.md
```

## CLI Commands

### `longtext run`

Execute the full pipeline on an input file.

```
longtext run <input-file> [OPTIONS]

Arguments:
  <input-file>                Path to input .txt or .md file

Options:
  --config, -c PATH           YAML config file
  --mode, -m TEXT              "general" (default) or "relationship"
  --resume, -r                Resume from checkpoint
  --multi-perspective, -mp    Enable parallel specialist agents
  --agent-count INT           Number of specialist agents (1-4, implies --multi-perspective)
  --max-workers INT           Max concurrent workers (1-256)
  --help                      Show all options
```

### `longtext status`

Check processing status and manifest state.

```bash
longtext status <input-file>
```

### `longtext init`

Initialize a new project with default configuration.

```bash
longtext init [--dir PATH]
```

## Output Structure

Processing creates a `.longtext/` directory alongside the input file:

```
.longtext/
├── part_001.txt           # Split input chunks
├── part_002.txt
├── summary_001.md         # Per-chunk LLM summaries
├── summary_002.md
├── stage_001.md           # Aggregated stage summaries
├── final_analysis.md      # Final synthesized analysis
├── manifest.json          # Processing state & checkpoint data
├── metrics.prom           # Prometheus metrics
└── .locks/                # Cross-process lock files
```

## Configuration

Config is YAML-based with layered precedence (highest wins):

**env vars > `longtext.local.yaml` (auto-discovered) > `--config` file > built-in defaults**

### Key sections

| Section | Purpose | Key settings |
|---------|---------|--------------|
| `model` | LLM provider and model | `provider`, `name`, `temperature`, `timeout` |
| `stages` | Per-stage parameters | `chunk_size`, `group_size`, `batch_size` |
| `prompts` | Prompt templates | `dir`, `format` |
| `output` | Output location | `dir`, `naming` conventions |
| `pipeline` | General behavior | `allow_resume`, `max_workers` |
| `logging` | Log configuration | `level`, `format` (text/json), `file` |

### Environment variable overrides

```bash
# Required
export OPENAI_API_KEY="sk-..."

# Model
export OPENAI_BASE_URL="https://custom-endpoint.com/v1"
export LONGTEXT_MODEL_NAME="gpt-4o"

# Output
export LONGTEXT_OUTPUT_DIR="./my-output"

# Logging
export LONGTEXT_LOG_LEVEL="DEBUG"
export LONGTEXT_LOG_FORMAT="json"
export LONGTEXT_LOG_FILE="./pipeline.log"
```

### Example configs

| File | Use case |
|------|----------|
| `examples/config.default.yaml` | All defaults with documentation |
| `examples/config.general.yaml` | General analysis |
| `examples/config.relationship.yaml` | Relationship analysis |
| `examples/config.multi_agent.yaml` | Multi-perspective analysis |
| `examples/config.performance_test.yaml` | Performance tuning |

### Local overrides

Create `longtext.local.yaml` in the working directory for secrets and local provider settings. This file is auto-discovered and should not be committed.

### Current limitation

The runtime currently writes working files next to the input file in `.longtext/`. The `output` section remains in the config schema, but it is not yet enforced uniformly by every stage.

## Modes

### General Mode (default)

Standard summarization and analysis. Best for meeting transcripts, project docs, knowledge bases, chat logs.

```bash
longtext run document.txt --mode general
```

### Relationship Mode

Entity and relationship-focused analysis for network mapping. Best for organizational networks, stakeholder mapping, communication flows.

```bash
longtext run transcript.txt --mode relationship
```

Relationship mode is available today, but some prompt sets and warnings still treat it as experimental.

## Resume

The pipeline supports checkpoint-based resumable processing via SHA-256 hash validation.

### How it works

1. Progress is saved to `.longtext/manifest.json` after each stage
2. On interruption, rerun with `--resume`
3. Input file hash is validated to detect changes
4. Completed stages are skipped; only remaining work is processed

```bash
# First run (interrupted)
longtext run large_document.txt

# Resume from checkpoint
longtext run large_document.txt --resume
```

### When resume helps

- Network timeouts during LLM calls
- API rate limit backoff
- Manual interruption (Ctrl+C)
- System crashes

## Docker

### Build and run

```bash
# Build the image
docker build -t longtext-pipeline .

# Run
docker run \
  -e OPENAI_API_KEY="your-key" \
  -v ./input:/data:ro \
  -v ./output:/output \
  longtext-pipeline run /data/input.txt
```

### Docker Compose

```bash
# Place input files in ./input/
# Set OPENAI_API_KEY in .env or environment
docker compose run longtext run /data/input.txt
```

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                          CLI Layer                           │
│                  (Typer-based entry point)                   │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                       Config Layer                           │
│          (YAML loading, env vars, validation)                │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                    Pipeline Orchestrator                      │
│         (sequential stages, continue-with-partial)           │
│                                                              │
│  ┌────────┐  ┌─────────┐  ┌───────┐  ┌───────┐  ┌───────┐  │
│  │ Ingest │→ │Summarize│→ │ Stage │→ │ Final │→ │ Audit │  │
│  │ split  │  │  async  │  │ async │  │ async │  │ check │  │
│  └────────┘  └─────────┘  └───────┘  └───────┘  └───────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ Manifest (SHA-256 state tracking & resume)           │    │
│  ├──────────────────────────────────────────────────────┤    │
│  │ Token Budget Manager (context window validation)     │    │
│  ├──────────────────────────────────────────────────────┤    │
│  │ File Lock (cross-process mutex)                      │    │
│  ├──────────────────────────────────────────────────────┤    │
│  │ Prometheus Metrics (retry, latency, rate limits)     │    │
│  └──────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│                 ┌─────────────────┐                          │
│                 │   LLM Client    │                          │
│                 │ (OpenAI-compat) │                          │
│                 │ sync/async/SSE  │                          │
│                 └─────────────────┘                          │
└──────────────────────────────────────────────────────────────┘
```

### Data flow

1. **CLI** parses arguments and loads layered config
2. **Ingest** reads input, splits into chunks (4000 chars, 10% overlap by default)
3. **Summarize** generates summaries for each chunk via LLM (async, concurrent workers)
4. **Stage** groups summaries (default 5 per group) and synthesizes (async)
5. **Final** synthesizes all stage summaries into one analysis (async, optional multi-perspective)
6. **Audit** post-processing quality check (placeholder in v1)
7. **Manifest** tracks state throughout for resume capability

## Troubleshooting

### API authentication errors

```bash
# Verify API key is set
echo $OPENAI_API_KEY

# Check endpoint is reachable
curl -s https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY" | head -1
```

### Resume not working

```bash
# Check manifest exists
cat .longtext/manifest.json | python -m json.tool

# If input file changed, hash mismatch will prevent resume
# Remove manifest and restart:
rm .longtext/manifest.json
```

### "Another pipeline process is already running"

The file lock prevents concurrent runs on the same input. If a previous run crashed without releasing the lock:

```bash
# Remove stale lock file
rm .longtext/.locks/*.lock
```

### Rate limits or timeouts

```yaml
# Reduce concurrency in config
pipeline:
  max_workers: 2     # Default is 4

stages:
  summarize:
    batch_size: 2    # Default is 4

model:
  timeout: 120       # Seconds (max 600)
```

### Context window exceeded

The token budget manager validates prompts before sending. If you see `ContextWindowExceededError`:

```yaml
# Reduce chunk size so individual prompts are smaller
stages:
  ingest:
    chunk_size: 3000   # Default is 4000
```

### Encoding errors

```yaml
input:
  encoding: "latin-1"   # Default is utf-8
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, testing, and PR guidelines.

```bash
# Quick start
pip install -e ".[dev]"
pytest tests/
ruff check .
```

## Documentation

- [Documentation index](docs/README.md)
- [CLI reference](docs/CLI.md)
- [Configuration reference](docs/CONFIG.md)
- [Architecture overview](docs/ARCHITECTURE.md)
- [Examples guide](examples/README.md)
- [Security policy](SECURITY.md)
- [Code of conduct](CODE_OF_CONDUCT.md)

## License

MIT License - see [LICENSE](LICENSE) for details.
