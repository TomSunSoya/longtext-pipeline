# CLI Specification for longtext-pipeline

## Overview

The `longtext-pipeline` provides a comprehensive command-line interface to perform hierarchical analysis of super-long text files in an automated manner. The tool is designed to be used in both scripting environments and for ad hoc text analysis tasks.

## Global Options

All `longtext` commands accept the following global options:

- `--help`: Display help information for any command
- `--version`: Display the version of the tool

## Commands

### 1. `longtext run <input>`

Run the hierarchical analysis pipeline on a text file. This is the main command that orchestrates the four-stage processing flow.

#### Parameters

- `<input>`: (required) Path to the input text file (.txt or .md) to be analyzed
- `--config <path>`: Optional configuration file path. If not provided, uses default configuration or falls back to environment variables
- `--mode <general|relationship>`: Select analysis type (default: general). Experimental `relationship` mode optimizes for entity relationship discovery
- `--resume`: Resume from existing manifest checkpoint if processing was interrupted

#### Behavior

1. Validates presence and format of input file
2. Computes SHA-256 hash of input to detect changes since last run
3. Creates output directory based on input file name (e.g., `.input_basename/`)
4. Executes pipeline stages in sequence: Ingest → Summarize → Stage → Final
5. Maintains progress state in `manifest.json`
6. Creates part files (`part_*.txt`), summary files (`summary_*.md`), stage files (`stage_*.md`), and final analysis (`final_analysis.md`)

#### Exit Codes

- `0`: Success - Analysis completed without errors
- `1`: Error - Analysis failed due to critical error (e.g., invalid file, no API key)
- `2`: Partial Success - Some analyses completed, others failed (Continue-with-Partial strategy)

#### Examples

```bash
# Basic usage
longtext run input.txt

# With custom config
longtext run input.md --config /path/to/config.yaml

# Resume interrupted analysis
longtext run input.txt --resume

# Run with relationship-focused mode
longtext run input.md --mode relationship
```

### 2. `longtext init`

Creates default configuration files in the current directory for customization.

#### Parameters

- `--dir <path>`: Directory where to create configuration files (default: cwd)

#### Behavior

1. Creates example configuration files in specified directory:
   - `config.general.yaml` - General analysis configuration
   - `config.relationship.yaml` - Relationship-focused analysis configuration
2. Sets up default directory structure based on configuration defaults

#### Exit Codes

- `0`: Success - Configuration files created successfully
- `1`: Error - Unable to create configuration files (permissions, disk space, etc.)

#### Examples

```bash
# Create default configs in current directory
longtext init

# Create configs in a specific directory
longtext init --dir /path/to/config/location
```

### 3. `longtext status <input>`

Displays the progress and status of a pipeline run for a given input file.

#### Parameters

- `<input>`: (required) Path to the input file that was submitted to the pipeline

#### Behavior

1. Locates the corresponding manifest (usually at `.input_basename/manifest.json`)
2. Parses and displays the manifest contents with user-friendly formatting
3. Shows processing progress in table form for each pipeline stage
4. Displays aggregate metrics if available (part count, stage count, timestamps)

#### Output Format

The command outputs a formatted table displaying:

```
Session ID: 20260403_153045_ab7f2d
Input File: path/to/input.txt
Status:     summarizing
Created:    2026-04-03T15:30:45Z
Updated:    2026-04-03T15:45:22Z
Progress:
 └─ Ingest: ✓ Completed (15:32:12)
 └─ Summarize: → In Progress (3/5 complete)
 └─ Stage: ⏳ Waiting
 └─ Final: ⏳ Waiting
 └─ Audit: ⏸️ Skipped

Total Parts: 5 (2/5 failed)
Total Stages: 2
Estimated Tokens: 25000
```

#### Exit Codes

- `0`: Success - Status retrieved and displayed
- `1`: Error - Status unavailable (no manifest found, invalid manifest, etc.)

#### Error Behavior

- If input file is not found, outputs: "Error: Input file '<input>' does not exist."
- If no manifest exists for input, outputs: "Analysis status not found for file '<input>'. Has it been processed?"
- If manifest file is invalid JSON, outputs: "Error: Cannot parse manifest file. May be corrupted or incomplete."

#### Examples

```bash
# Check status of pipeline for input.txt
longtext status input.txt

# Check status of pipeline for markdown file
longtext status document.md
```

### 4. `longtext --version`

Display the version of the longtext-pipeline tool.

#### Exit Code

- `0`: Always succeeds

#### Output Format

Current version in format `v{major}.{minor}.{patch}`, for example: `v0.1.0`.

### 5. `longtext --help`

Display global help information and list all available commands.

#### Exit Code

- `0`: Always succeeds

#### Output Format

Shows usage and available commands in concise format.

## Input Validation Rules

The CLI performs the following validations for commands accepting input files:

### For ALL commands that accept input files:
- Input file must exist on local filesystem
- Path must use forward slashes (/) or valid Windows path separators
- File size must be greater than 0 bytes (empty files rejected)
- Path cannot contain control characters that could affect command execution

### For `longtext run` and `longtext status`:
- Input file must be readable by user
- File extension must be `.txt` or `.md` (case-insensitive)
- File content must be valid UTF-8

#### Rejected Inputs

- Files larger than available memory for reading (checked prior to processing)
- Symbolic links to files with circular references
- Directory paths (rejected when file path expected)

## Error Message Guidelines

### Format

Error messages follow a consistent pattern:
```
Error: [short description]. [context about resolution or next steps].
```

### Categories

- **Usage errors**: "Error: Missing required argument 'input'. Run 'longtext run --help' for usage information."
- **I/O errors**: "Error: Cannot access input file '/path'. Please verify permissions or file existence."
- **Validation errors**: "Error: Unsupported file type '.xyz'. Supported: .txt, .md."
- **API connectivity** (run command): "Error: Cannot reach LLM API. Check API key and network connectivity."

### Color Coding in Terminal

- All error messages use red text foreground
- Warning messages use yellow text foreground
- Informative messages use white/neutral color
- Success indicators use green color

## Exit Codes Convention

The CLI consistently implements these exit codes:

- `0`: Success (operation completed as expected)
- `1`: General error (validation failure, permission issues, I/O problems, API errors)
- `2`: Partial success (some operations succeeded, others failed - only when the primary goal had mixed outcomes)

This convention enables usage in complex scripts via standard bash error checking:
```bash
if longtext run input.txt ; then
  echo "Success with code $?"
else
  echo "Error with code $?"
fi
```