# Architecture Overview

This document describes the current runtime architecture, not the historical MVP plan.

## High-level flows

### Single-file flow

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

### Batch flow

```text
CLI batch
  -> Input expansion and validation
  -> BatchProcessor
  -> N independent single-file pipeline runs
     -> sequential mode or parallel mode
```

## Primary modules

### CLI and entry points

- `src/longtext_pipeline/cli.py`
- Typer-based commands: `run`, `batch`, `status`, `init`

### Configuration

- `src/longtext_pipeline/config.py`
- Merges defaults, explicit config, auto-discovered local config, and env overrides
- Validates writable `output.dir`

### State and resumability

- `src/longtext_pipeline/manifest.py`
- Stores progress in `.longtext/manifest.json`
- Uses SHA-256 input hashing to decide whether resume is safe

### Pipeline orchestration

- `src/longtext_pipeline/pipeline/orchestrator.py`
- Owns stage ordering, resume logic, error aggregation, metrics export, and file locking

### Processing stages

- `pipeline/ingest.py`: input reading, cleaning, splitting, and extractor dispatch
- `pipeline/pdf_extraction.py`: PDF extraction and preprocessing helpers
- `pipeline/docx_extraction.py`: DOCX extraction and preprocessing helpers
- `pipeline/ocr_fallback.py`: OCR fallback for image-heavy documents
- `pipeline/summarize.py`: async part summarization
- `pipeline/stage_synthesis.py`: async group-level synthesis
- `pipeline/final_analysis.py`: final synthesis, tiny-input fallback, and optional specialist fan-out
- `pipeline/audit.py`: active audit stage with claim/timeline/quality checks
- `pipeline/audit_types.py`: audit result dataclasses
- `pipeline/audit_reporting.py`: prompt loading and token-budgeted audit prompt assembly

### LLM client layer

- `llm/base.py`: abstract client contract
- `llm/openai_compatible.py`: sync, async, JSON, and streaming OpenAI-compatible implementation
- `llm/registry.py`: provider registry and client creation
- `llm/dispatcher.py`: parallel provider dispatch modes
- `llm/results.py`: ranking models and `ResultRanker`
- `llm/ranker.py`, `llm/result_ranker.py`: legacy compatibility wrappers
- `llm/progress.py`: shared streaming progress helpers

### Batch execution

- `utils/batch_processor.py`: per-file execution in sequential or parallel mode
- `utils/batch_progress.py`: progress tracking and reporting
- `batch/orchestrator.py`: older sequential batch orchestrator retained in the repo

### Utilities

- `utils/retry.py`: retry policies
- `utils/token_budget.py`: context-window checks and truncation
- `utils/process_lock.py`: cross-process file locks
- `utils/metrics.py`: Prometheus export
- `utils/io.py`, `utils/hashing.py`, `utils/text_clean.py`, `utils/token_estimator.py`

## Storage model

### Default layout

When `output.dir` is not set, the runtime writes working files adjacent to the input in `.longtext/`.

That directory contains:

- split parts
- intermediate summaries
- stage summaries
- final outputs
- manifest
- metrics export
- lock files

### Custom output layout

When `output.dir` is set for the standard pipeline path:

- part, summary, stage, final-analysis, and metrics files are written to `<output.dir>/.longtext/`
- manifest and lock files still remain adjacent to the input file in its local `.longtext/`

That split is intentional in the current implementation because resume and status lookup still key off the input-local manifest.

## Concurrency model

- `run()` is synchronous at the CLI boundary
- summarize and stage synthesis run async internally
- concurrency is bounded with semaphores and worker limits
- final analysis can optionally fan out into multiple specialist analyses
- `longtext batch` can process files sequentially or in parallel across independent single-file runs

## Operational caveats

- The top-level `LongtextPipeline.run()` path still validates only `.txt` and `.md`, even though the ingest layer and CLI validators contain PDF/DOCX support. Treat text and markdown as the safest end-to-end inputs today.
- Audit is active, not a placeholder, but it may fall back to offline heuristics when no API-backed auditor can be created.
- `output.dir` works for standard stage artifacts, but batch users should avoid sharing one explicit output directory across many inputs unless mixed artifacts are acceptable.
- Editable installs can hide packaging bugs, so release validation should include a wheel or non-editable install path to confirm prompt files are packaged.
