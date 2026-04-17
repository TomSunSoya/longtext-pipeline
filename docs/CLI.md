# CLI Reference

This document describes the commands exposed by the `longtext` CLI.

## Global options

Available on the root command:

- `--help`
- `--version`

## `longtext run <input-file>`

Run the full analysis pipeline on a single input file.

### Supported inputs

`longtext run` supports `.txt`, `.md`, `.pdf`, and `.docx` inputs end to end.

PDF and DOCX inputs are extracted to text during ingest before the remaining pipeline stages run.

### Flags

- `--config`, `-c PATH`: Load an explicit YAML config file
- `--mode`, `-m TEXT`: `general` or `relationship`
- `--resume`, `-r`: Resume from an existing manifest
- `--multi-perspective`, `-mp`: Enable specialist-agent synthesis in the final stage
- `--agent-count INT`: Number of specialist agents to run, `1-4`
- `--max-workers INT`: Max concurrent summarize/stage workers, `1-256`

### Exit codes

- `0`: Success
- `1`: Failure
- `2`: Partial success or completed with issues

### Examples

```bash
longtext run input.txt
longtext run input.txt --config examples/config.general.yaml
longtext run input.txt --mode relationship
longtext run input.txt --resume
longtext run input.txt --multi-perspective --agent-count 3
longtext run input.txt --max-workers 2
```

## `longtext batch <input-pattern>`

Run the pipeline on multiple files.

Input can be provided as:

- a glob pattern such as `"inputs/*.txt"`
- a recursive glob such as `"docs/**/*.md"`
- a comma-separated list such as `"a.txt,b.txt,c.txt"`

### Flags

- `--config`, `-c PATH`: Load an explicit YAML config file
- `--mode`, `-m TEXT`: `general` or `relationship`
- `--resume`, `-r`: Resume each file from its manifest when possible
- `--multi-perspective`, `-mp`: Enable specialist-agent synthesis per file
- `--agent-count INT`: Number of specialist agents per file, `1-4`
- `--max-workers INT`: Max concurrent summarize/stage workers per file, `1-256`
- `--parallel`, `-p`: Process multiple files concurrently
- `--batch-max-workers INT`: Max concurrent files in parallel mode, `1-64`

### Exit codes

- `0`: All files succeeded
- `1`: All files failed
- `2`: Partial success

### Examples

```bash
longtext batch "inputs/*.txt"
longtext batch "inputs/*.txt" --parallel --batch-max-workers 4
longtext batch "doc1.txt,doc2.txt,doc3.txt" --config config.yaml
longtext batch "*.md" --parallel --batch-max-workers 2 --multi-perspective
```

### Batch output layout

If batch runs redirect generated artifacts through `output.dir`, each input file gets a namespaced subdirectory under the configured base directory. That keeps artifacts from different inputs from landing in the same `.longtext/` directory.

## `longtext status <input-file>`

Show the current manifest-driven status for a previous or in-progress run.

### Example

```bash
longtext status input.txt
```

The command reads the manifest associated with the input file. In the current implementation, manifest lookup still uses the input-local `.longtext/manifest.json`, even when generated stage artifacts were redirected with `output.dir`.

## `longtext init`

Generate starter files in a target directory.

### Flags

- `--dir`, `-d PATH`: Directory where starter files should be created

### Files generated

- `config.general.yaml`
- `config.relationship.yaml`
- `longtext.local.yaml`
- `sample_input.txt`
- `README.md`

### Example

```bash
longtext init --dir ./demo-project
```

## Output layout

### Built-in defaults

With the built-in defaults, generated artifacts are written under `./output/.longtext/`:

```text
output/.longtext/
├── part_00.txt
├── summary_00.md
├── stage_00.md
├── final_analysis.md
└── metrics.prom
```

### With `output.dir`

For standard single-file runs with a custom `output.dir`:

- part, summary, stage, final-analysis, and metrics files go to `<output.dir>/.longtext/`
- manifest and `.locks/` remain beside the input file in its local `.longtext/`

For batch runs with a custom `output.dir`:

- each input gets a namespaced output base such as `<output.dir>/report_a1b2c3d4/.longtext/`
- manifest and `.locks/` still remain beside each source input file

## Current caveats

- Audit is active and no longer just a skipped placeholder stage.
- `status` is manifest-based, so it follows the input-local manifest rather than the redirected artifact directory.
