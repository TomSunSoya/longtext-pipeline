# Architecture Overview

This document describes the current runtime architecture, not the historical MVP plan.

## High-level flow

```text
CLI
  -> Config loading and validation
  -> Manifest and file lock setup
  -> Ingest
  -> Summarize
  -> Stage synthesis
  -> Final analysis
  -> Audit
  -> Metrics export
```

## Primary modules

### CLI and entry point

- `src/longtext_pipeline/cli.py`
- Typer-based command surface: `run`, `status`, `init`

### Configuration

- `src/longtext_pipeline/config.py`
- Merges defaults, explicit config, local config, and env overrides

### State and resumability

- `src/longtext_pipeline/manifest.py`
- Stores progress in `.longtext/manifest.json`
- Uses SHA-256 input hashing to decide whether resume is safe

### Pipeline orchestration

- `src/longtext_pipeline/pipeline/orchestrator.py`
- Owns stage ordering, resume logic, error aggregation, metrics export, and file locking

### Processing stages

- `pipeline/ingest.py`: split input into parts
- `pipeline/summarize.py`: async part summarization
- `pipeline/stage_synthesis.py`: async group-level synthesis
- `pipeline/final_analysis.py`: final synthesis plus optional specialist fan-out
- `pipeline/audit.py`: current placeholder audit stage

### LLM client layer

- `llm/base.py`: abstract client contract
- `llm/openai_compatible.py`: sync, async, JSON, and streaming OpenAI-compatible implementation
- `llm/progress.py`: shared streaming progress helpers

### Utilities

- `utils/retry.py`: retry policies
- `utils/token_budget.py`: context-window checks and truncation
- `utils/process_lock.py`: cross-process file locks
- `utils/metrics.py`: Prometheus export
- `utils/io.py`, `utils/hashing.py`, `utils/text_clean.py`, `utils/token_estimator.py`

## Storage model

The runtime currently writes all working files adjacent to the input file in `.longtext/`.

That directory contains:

- split parts
- intermediate summaries
- stage summaries
- final outputs
- manifest
- metrics export
- lock files

## Concurrency model

- `run()` is synchronous at the CLI boundary
- summarize and stage synthesis run async internally
- concurrency is bounded with semaphores
- final analysis can optionally fan out into multiple specialist analyses

## Current limitations

- Audit is intentionally a placeholder and records skipped status rather than doing full semantic verification.
- Relationship mode is supported but still carries some experimental warnings and conservative prompt assumptions.
- The configuration schema includes `output` controls that are not yet honored consistently by every stage.
- Editable installs can hide packaging bugs, so release validation must include a wheel or non-editable install path.
