# longtext-pipeline

A Python CLI tool for hierarchical analysis of super-long texts using LLMs.

**Problem**: Feeding massive documents directly to LLMs causes context overflow, hallucinations, and unauditable outputs.

**Solution**: A 4-stage pipeline that decomposes long texts into manageable chunks, processes them hierarchically, and synthesizes results with built-in traceability.

## Features

- **Stratifiable processing**: 4-stage pipeline (Ingest → Summarize → Stage → Final)
- **Resumable**: Checkpoint-based resume with SHA-256 hash validation
- **Dual modes**: General analysis and relationship-focused (experimental)
- **Model-agnostic**: OpenAI-compatible API support (OpenAI, OpenRouter, Ollama, etc.)
- **Audit trail**: Intermediate files preserved for traceability

## Prerequisites

- Python 3.9+
- pip
- OpenAI API key (or compatible endpoint)

## Installation

```bash
# Clone or download the project
cd longtext-pipeline

# Install in editable mode
pip install -e .

# Verify installation
longtext --version
```

## Quickstart

### 1. Prepare your input file

Create a text file with content to analyze. For testing, use a sample document:

```text
# sample_input.txt

Chapter 1: The Beginning
The project started on January 15th when the team gathered for the kickoff meeting.
Sarah proposed the new architecture while John raised concerns about timeline...

Chapter 2: Development Phase
During the sprint, the team encountered several blockers. The database migration
took longer than expected, but the frontend team made significant progress...

Chapter 3: Results and Conclusions
The final delivery exceeded expectations. User engagement increased by 40% and
the system handled 10x the original load capacity...
```

### 2. Set your API key

```bash
# Set OpenAI API key (required)
export OPENAI_API_KEY="sk-your-api-key-here"

# Or for Windows PowerShell:
$env:OPENAI_API_KEY="sk-your-api-key-here"
```

### 3. Run the pipeline

```bash
# Using general analysis mode (default)
longtext run sample_input.txt --config examples/config.general.yaml
```

### 4. Check results

```bash
# Check processing status
longtext status sample_input.txt

# View the final analysis
cat output/final_analysis.md
```

## CLI Commands

### `longtext run`

Execute the full pipeline on an input file.

```bash
longtext run <input-file> [OPTIONS]

# Required
<input-file>              Path to input .txt or .md file

# Options
--config PATH             Path to YAML config file (default: examples/config.general.yaml)
--mode TEXT               Processing mode: "general" or "relationship"
--resume                  Resume from checkpoint if available
--output-dir PATH         Override output directory
--help                    Show all options
```

### `longtext status`

Check processing status and manifest state.

```bash
longtext status <input-file>

# Shows:
# - Processing stage (pending/in-progress/completed)
# - Completed parts and summaries
# - File hash validation
# - Timestamps for each operation
```

### `longtext init`

Initialize a new project with default configuration.

```bash
longtext init [OPTIONS]

--output-dir PATH         Output directory to initialize
--copy-config             Copy default config files
```

## Modes

### General Mode (default)

Standard summarization and analysis for most use cases.

```bash
longtext run document.txt --config examples/config.general.yaml
# or
longtext run document.txt --mode general
```

**Best for**: Meeting transcripts, project docs, knowledge bases, chat logs.

### Relationship Mode (experimental)

Entity and relationship-focused analysis for network mapping.

```bash
longtext run transcript.txt --config examples/config.relationship.yaml
# or
longtext run transcript.txt --mode relationship
```

**Best for**: Organizational networks, stakeholder mapping, communication flows.

**Note**: Uses gpt-4o (higher quality) with lower temperature for consistent entity naming.

## Resume Functionality

The pipeline supports resumable processing via `manifest.json` checkpointing.

### How it works

1. After each stage completes, progress is saved to `.longtext/manifest.json`
2. On interruption, rerun with `--resume` flag
3. System validates file hashes to detect input changes
4. Skips completed stages, processes only remaining work

### Example: Resume interrupted run

```bash
# First run (interrupted at 60%)
longtext run large_document.txt --config examples/config.general.yaml

# Resume from checkpoint
longtext run large_document.txt --config examples/config.general.yaml --resume

# System output:
# "Resuming from checkpoint... 12/20 parts completed"
# "Skipping completed stages, processing remaining 8 parts..."
```

### When to use resume

- Network timeouts during LLM calls
- Power failures or system crashes
- Manual interruption (Ctrl+C)
- API rate limit backoff

## Configuration

Configuration is YAML-based with environment variable overrides.

### Key sections

| Section | Purpose | Key settings |
|---------|---------|--------------|
| `model` | LLM provider and model | `provider`, `name`, `temperature` |
| `stages` | Per-stage parameters | `chunk_size`, `group_size`, `batch_size` |
| `prompts` | Prompt templates | `dir`, `format` |
| `output` | Output location | `dir`, `naming` conventions |
| `pipeline` | General behavior | `allow_resume`, `max_workers` |

### Example configs

- `examples/config.general.yaml` - General analysis (default)
- `examples/config.relationship.yaml` - Relationship analysis (experimental)
- `examples/config.default.yaml` - Default values reference

### Environment overrides

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://custom-endpoint.com/v1"
export LONGTEXT_MODEL_NAME="gpt-4o"
export LONGTEXT_OUTPUT_DIR="./my-output"
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                          CLI Layer                           │
│                     (cli.py entry point)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                       Config Layer                           │
│                  (validation, merging)                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ Ingest  │→ │Summarize│→ │  Stage  │→ │  Final  │        │
│  │  split  │  │  parts  │  │aggregate│  │synthesize│       │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        │
│       │            │            │            │              │
│       └────────────┴────────────┴────────────┘              │
│                         │                                   │
│                         ▼                                   │
│                  ┌──────────┐                               │
│                  │ Manifest │ ← State tracking & resume     │
│                  └──────────┘                               │
│                         │                                   │
│                         ▼                                   │
│                  ┌──────────┐                               │
│                  │   LLM    │ ← OpenAI-compatible clients   │
│                  └──────────┘                               │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                      Output Files                            │
│   parts/ → summaries/ → stages/ → final_analysis.md         │
└─────────────────────────────────────────────────────────────┘
```

### Data flow

1. **CLI** parses arguments and loads config
2. **Ingest** reads input, splits into chunks (4000 chars, 10% overlap)
3. **Summarize** generates summaries for each chunk via LLM
4. **Stage** aggregates 5 summaries into stage summaries
5. **Final** synthesizes all stage summaries into final analysis
6. **Manifest** tracks state throughout for resume capability

## Output Structure

After processing completes, the output directory contains:

```
output/
├── manifest.json          # State tracking, hashes, timestamps
├── status.log            # Processing log
├── parts/
│   ├── part_001.txt      # Split input chunks
│   ├── part_002.txt
│   └── ...
├── summaries/
│   ├── summary_001.md    # Per-chunk summaries
│   ├── summary_002.md
│   └── ...
├── stages/
│   ├── stage_001.md      # Aggregated stage summaries
│   └── ...
└── final/
    └── final_analysis.md # Final synthesized analysis (deliverable)
```

### manifest.json

The manifest file contains:

```json
{
  "input_hash": "sha256:abc123...",
  "stages": {
    "ingest": { "status": "completed", "timestamp": "..." },
    "summarize": { "status": "completed", "parts": [
      { "id": "part_001", "status": "completed", "hash": "..." },
      { "id": "part_002", "status": "completed", "hash": "..." }
    ]},
    "stage": { "status": "completed" },
    "final": { "status": "completed" }
  }
}
```

## Troubleshooting

### API authentication errors

```bash
# Verify API key is set
echo $OPENAI_API_KEY

# Check key format (should start with "sk-")
export OPENAI_API_KEY="sk-..."
```

### Resume not working

```bash
# Check manifest exists
ls -la output/manifest.json

# Verify input file hasn't changed (hash mismatch breaks resume)
# If input changed, remove manifest and restart fresh
rm output/manifest.json
```

### Out of memory errors

```yaml
# Reduce batch_size in config
stages:
  summarize:
    batch_size: 2  # Default is 4
  ingest:
    chunk_size: 3000  # Default is 4000
```

### LLM rate limits

```yaml
# Reduce concurrent workers
pipeline:
  max_workers: 2  # Default is 4

# Add delays between API calls (custom implementation may be needed)
```

### Empty or tiny input warnings

The system handles small inputs gracefully:

- Files < 1000 chars: Processed as single chunk with adjusted prompts
- Empty files: Warning logged, placeholder created

### Encoding errors

```yaml
# Specify encoding for non-UTF-8 files
input:
  encoding: "latin-1"  # or cp1252, utf-16, etc.
```

## Contributing

See comprehensive documentation in `docs/`:

- `docs/SPEC.md` - MVP specification and feature requirements
- `docs/ARCHITECTURE.md` - Detailed module architecture and interfaces
- `docs/development/` - Development workflow and testing guidelines

### Quick contribution guide

```bash
# Fork and clone
git clone https://github.com/your-username/longtext-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Submit PR
```

## License

MIT License - see LICENSE file in the project root.
