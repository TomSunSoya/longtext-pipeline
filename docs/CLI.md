# CLI Reference

This document describes the commands exposed by the `longtext` CLI.

## Global options

Available on the root command:

- `--help`
- `--version`

## `longtext run <input-file>`

Run the full analysis pipeline on a `.txt` or `.md` file.

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

## `longtext status <input-file>`

Show the current manifest-driven status for a previous or in-progress run.

### Example

```bash
longtext status input.txt
```

The command reads `.longtext/manifest.json` next to the input file and renders a human-readable summary.

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

The live runtime writes working files next to the input file in a `.longtext/` directory:

```text
.longtext/
├── part_00.txt
├── summary_00.md
├── stage_00.md
├── final_analysis.md
├── manifest.json
├── metrics.prom
└── .locks/
```

## Current caveats

- Relationship mode is usable today, but some logs and prompt paths still describe it as experimental.
- The runtime output location is currently fixed to `.longtext/` beside the input file.
- The `output` block in config exists, but it is not yet enforced uniformly by every stage.
