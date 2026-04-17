"""Command-line interface for longtext-pipeline.

This module provides the CLI entry point using Typer for creating a structured
command-line interface with proper help texts and command routing.
"""

import os
import logging
from pathlib import Path
from typing import Any

import typer
import yaml  # type: ignore[import-untyped]
from typing_extensions import Annotated

from longtext_pipeline import __version__
from longtext_pipeline.config import (
    AUTO_CONFIG_FILENAMES,
    DEFAULT_CONFIG,
    ConfigError,
    format_missing_settings_message,
    get_missing_required_settings,
    load_runtime_config,
)
from longtext_pipeline.logging_utils import configure_logging
from longtext_pipeline.utils.io import write_file
from longtext_pipeline.pipeline.orchestrator import LongtextPipeline
from longtext_pipeline.manifest import ManifestManager
from longtext_pipeline.renderer import format_status

app = typer.Typer(
    name="longtext",
    help="A Python CLI tool for hierarchical analysis of super-long texts.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)

logger = logging.getLogger(__name__)


def version_callback(value: bool) -> None:
    """Callback for --version flag to display version information."""
    if value:
        typer.echo(f"longtext-pipeline v{__version__}")
        raise typer.Exit()


@app.command()
def run(
    input_file: Annotated[
        str,
        typer.Argument(
            ...,
            help="Path to the input text file (.txt, .md, .pdf, or .docx) to be analyzed.",
        ),
    ],
    config: Annotated[
        str | None,
        typer.Option(
            "--config",
            "-c",
            help="Optional configuration file path. If not provided, uses default configuration.",
        ),
    ] = None,
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            "-m",
            help="Select analysis type. 'general' for standard analysis, 'relationship' for entity relationship discovery.",
        ),
    ] = "general",
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            "-r",
            help="Resume from existing manifest checkpoint if processing was interrupted.",
        ),
    ] = False,
    multi_perspective: Annotated[
        bool,
        typer.Option(
            "--multi-perspective",
            "-mp",
            help="Enable multi-perspective analysis with parallel specialist agents.",
        ),
    ] = False,
    agent_count: Annotated[
        int | None,
        typer.Option(
            "--agent-count",
            min=1,
            max=4,
            help="Number of final-analysis specialist agents to run (1-4). Implies --multi-perspective.",
        ),
    ] = None,
    max_workers: Annotated[
        int | None,
        typer.Option(
            "--max-workers",
            min=1,
            max=256,
            help="Maximum concurrent workers for summarize and stage-synthesis execution (1-256).",
        ),
    ] = None,
) -> int:
    """Run the hierarchical analysis pipeline on a single text file.

    This is the main command that orchestrates the five-stage processing flow:
    Ingest → Summarize → Stage → Final → Audit

    Args:
        input_file: Path to the input text file (.txt, .md, .pdf, or .docx) to be analyzed.
        config: Optional configuration file path. If not provided, uses default
            configuration or falls back to environment variables.
        mode: Select analysis type. 'general' for standard analysis, 'relationship'
            for entity relationship discovery.
        resume: Resume from existing manifest checkpoint if processing was interrupted.

    Returns:
        int: Exit code (0 for success, 1 for error, 2 for partial success)

    Examples:
        # Basic usage
        $ longtext run input.txt

        # With custom config
        $ longtext run input.md --config /path/to/config.yaml

        # PDF file support
        $ longtext run document.pdf

        # DOCX file support
        $ longtext run document.docx

        # Resume interrupted analysis
        $ longtext run input.txt --resume

        # Run with relationship-focused mode
        $ longtext run input.md --mode relationship
    """
    try:
        # Step 1: Validate input file (exists, valid extension .txt/.md)
        input_path = _validate_input_file(input_file)

        # Step 2: Load config from explicit config + auto-discovered local config + env overrides
        final_config, loaded_sources = load_runtime_config(
            config, search_dir=Path.cwd()
        )
        configure_logging(final_config)

        missing_settings = get_missing_required_settings(final_config)
        if missing_settings:
            typer.echo(format_missing_settings_message(missing_settings), err=True)
            return 1

        # Update prompts based on mode
        if "prompts" in final_config and mode == "relationship":
            final_config["prompts"]["format"] = "relationship"

        effective_multi_perspective = multi_perspective or agent_count is not None
        final_config["multi_perspective"] = effective_multi_perspective
        if max_workers is not None:
            final_config.setdefault("pipeline", {})
            final_config["pipeline"]["max_workers"] = max_workers
        if agent_count is not None:
            final_config.setdefault("pipeline", {})
            final_config["pipeline"]["specialist_count"] = agent_count

        # Step 3: Initialize LongtextPipeline from orchestrator
        pipeline = LongtextPipeline()

        # Step 4: Execute pipeline with appropriate parameters
        typer.echo(f"Starting pipeline for: {input_path}")
        typer.echo(f"Mode: {mode}")
        typer.echo(f"Resume: {resume}")
        typer.echo(f"Multi-perspective: {effective_multi_perspective}")
        if max_workers is not None:
            typer.echo(f"Max workers: {max_workers}")
        if agent_count is not None:
            typer.echo(f"Specialist agent count: {agent_count}")
        if loaded_sources:
            typer.echo(f"Config sources: {', '.join(loaded_sources)}")
        else:
            typer.echo("Config sources: built-in defaults")
        typer.echo()

        # Step 5: Run pipeline with error handling
        try:
            final_analysis = pipeline.run(
                input_path=input_path,
                config_path=config,
                mode=mode,
                resume=resume,
                multi_perspective=effective_multi_perspective,
                specialist_count=agent_count,
                max_workers=max_workers,
            )

            # Step 6: Determine exit code based on result
            status = (
                final_analysis.status
                if hasattr(final_analysis, "status")
                else "unknown"
            )

            if status == "completed":
                typer.echo("\n[PASS] Pipeline completed successfully")
                return 0
            elif status in ("partial_success", "completed_with_issues"):
                typer.echo(
                    "\n[PARTIAL] Pipeline completed with partial results or issues"
                )
                return 2
            else:
                typer.echo(f"\n[FAIL] Pipeline failed with status: {status}")
                return 1

        except KeyboardInterrupt:
            logger.warning("Pipeline interrupted by user")
            typer.echo("\nPipeline interrupted by user")
            return 1
        except Exception as e:
            logger.exception("Pipeline execution failed")
            typer.echo(f"\n[FAIL] Pipeline execution failed: {e}")
            return 1

    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        return 1
    except ConfigError as e:
        typer.echo(f"Error: {e}", err=True)
        return 1
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        return 1
    except PermissionError as e:
        typer.echo(f"Error: {e}", err=True)
        return 1
    except Exception as e:
        logger.exception("Unexpected CLI error")
        typer.echo(f"Unexpected error: {e}", err=True)
        return 1


def _validate_input_file(input_path: str) -> str:
    """Validate that input file exists and has supported format.

    Args:
        input_path: Path to the input file to validate

    Returns:
        Resolved absolute path to the input file

    Raises:
        FileNotFoundError: If file does not exist
        PermissionError: If file cannot be read
        ValueError: If file extension is not supported
    """
    path = Path(input_path).resolve()

    # Check if file exists
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    # Check if file is readable
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")

    # Check extension (now including PDF and DOCX)
    ext = path.suffix.lower()
    if ext not in [".txt", ".md", ".pdf", ".docx"]:
        raise ValueError(
            f"Unsupported file format. Only .txt, .md, .pdf, and .docx files are supported, got: {ext}"
        )

    return str(path)


@app.command()
def batch(
    input_pattern: Annotated[
        str,
        typer.Argument(
            ...,
            help="Glob pattern or comma-separated list of input files (.txt, .md, .docx). Example: 'inputs/*.txt' or 'file1.txt,file2.txt,file3.txt'",
        ),
    ],
    config: Annotated[
        str | None,
        typer.Option(
            "--config",
            "-c",
            help="Optional configuration file path. If not provided, uses default configuration.",
        ),
    ] = None,
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            "-m",
            help="Select analysis type. 'general' for standard analysis, 'relationship' for entity relationship discovery.",
        ),
    ] = "general",
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            "-r",
            help="Resume from existing manifest checkpoint if processing was interrupted.",
        ),
    ] = False,
    multi_perspective: Annotated[
        bool,
        typer.Option(
            "--multi-perspective",
            "-mp",
            help="Enable multi-perspective analysis with parallel specialist agents.",
        ),
    ] = False,
    agent_count: Annotated[
        int | None,
        typer.Option(
            "--agent-count",
            min=1,
            max=4,
            help="Number of final-analysis specialist agents to run (1-4). Implies --multi-perspective.",
        ),
    ] = None,
    max_workers: Annotated[
        int | None,
        typer.Option(
            "--max-workers",
            min=1,
            max=256,
            help="Maximum concurrent workers PER FILE for summarize and stage-synthesis (1-256).",
        ),
    ] = None,
    parallel: Annotated[
        bool,
        typer.Option(
            "--parallel",
            "-p",
            help="Enable parallel processing of multiple files. Without this flag, files are processed sequentially.",
        ),
    ] = False,
    batch_max_workers: Annotated[
        int | None,
        typer.Option(
            "--batch-max-workers",
            min=1,
            max=64,
            help="Maximum concurrent files to process in parallel batch mode (1-64). Only used with --parallel.",
        ),
    ] = None,
) -> int:
    """Run the hierarchical analysis pipeline on multiple text files.

    This command processes multiple input files in batch mode, either sequentially
    (default) or in parallel (with --parallel flag).

    Input files can be specified via:
    - Glob pattern: 'inputs/*.txt', 'docs/**/*.md'
    - Comma-separated list: 'file1.txt,file2.txt,file3.txt'
    - Single file (for convenience): 'single.txt'

    Args:
        input_pattern: Glob pattern or comma-separated list of input files.
        config: Optional configuration file path.
        mode: Analysis mode ('general' or 'relationship').
        resume: Resume from checkpoints for each file.
        multi_perspective: Enable specialist agents for each file.
        agent_count: Number of specialist agents per file.
        max_workers: Workers per file for internal async stages.
        parallel: Enable parallel file processing.
        batch_max_workers: Max concurrent files in parallel mode.

    Returns:
        int: Exit code (0 for all success, 1 for all failed, 2 for partial success)

    Examples:
        # Process all txt files sequentially
        $ longtext batch 'inputs/*.txt'

        # Process multiple files in parallel (up to 4 concurrent)
        $ longtext batch 'inputs/*.txt' --parallel --batch-max-workers 4

        # Process specific files with custom config
        $ longtext batch 'doc1.txt,doc2.txt,doc3.txt' --config config.yaml

        # Parallel batch with multi-perspective analysis
        $ longtext batch '*.md' --parallel --batch-max-workers 2 --multi-perspective
    """
    from longtext_pipeline.utils.batch_processor import (
        BatchProcessor,
        create_namespace_for_file,
    )

    try:
        # Expand input pattern to file list
        input_files = _expand_input_pattern(input_pattern)

        if not input_files:
            typer.echo(f"Error: No files found matching '{input_pattern}'", err=True)
            return 1

        typer.echo(f"Found {len(input_files)} file(s) to process:")
        for f in input_files:
            typer.echo(f"  - {f}")
        typer.echo()

        # Build per-file config with namespace isolation for batch mode
        # In batch mode, each file gets its own output subdirectory to prevent conflicts
        # First, determine the base output directory from config
        base_output_dir = None
        if config:
            try:
                from longtext_pipeline.config import load_config

                loaded_config = load_config(config)
                base_output_dir = loaded_config.get("output", {}).get("dir")
            except Exception:
                pass  # Use default behavior if config can't be loaded

        # If no explicit config output.dir, use default
        if base_output_dir is None:
            from longtext_pipeline.config import DEFAULT_CONFIG

            base_output_dir = DEFAULT_CONFIG["output"]["dir"]

        base_output_path = Path(base_output_dir).resolve()

        # Create per-file config with namespaced output directories
        # Note: per_file_config is now a dict keyed by file path
        per_file_config = {}
        for file_path in input_files:
            # Generate unique namespace for this file
            namespace = create_namespace_for_file(file_path)
            namespaced_output = str(base_output_path / namespace)

            per_file_config[file_path] = {
                "config": config,
                "mode": mode,
                "resume": resume,
                "multi_perspective": multi_perspective or agent_count is not None,
                "agent_count": agent_count,
                "max_workers": max_workers,
                "output_dir": namespaced_output,
            }

        # Create batch processor and run
        processor = BatchProcessor(
            parallel=parallel,
            batch_max_workers=batch_max_workers or 1,
        )

        typer.echo(f"Starting batch processing (parallel={parallel})...")
        if parallel and batch_max_workers:
            typer.echo(f"Max concurrent files: {batch_max_workers}")
        if max_workers:
            typer.echo(f"Max workers per file: {max_workers}")
        typer.echo(f"Output directory: {base_output_path}")
        typer.echo("Each file will have a namespaced subdirectory for artifacts")
        typer.echo()

        results = processor.run_batch(input_files, per_file_config)

        # Report results
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful

        typer.echo()
        typer.echo("=" * 60)
        typer.echo("BATCH PROCESSING COMPLETE")
        typer.echo("=" * 60)
        typer.echo(f"Total files: {len(results)}")
        typer.echo(f"Successful: {successful}")
        typer.echo(f"Failed: {failed}")
        typer.echo()

        if failed > 0:
            typer.echo("Failed files:", err=True)
            for r in results:
                if not r["success"]:
                    typer.echo(
                        f"  - {r['file']}: {r.get('error', 'Unknown error')}", err=True
                    )
            typer.echo()

        if successful == len(results):
            typer.echo("[PASS] All files processed successfully")
            return 0
        elif successful > 0:
            typer.echo("[PARTIAL] Some files processed successfully")
            return 2
        else:
            typer.echo("[FAIL] All files failed to process")
            return 1

    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        return 1
    except Exception as e:
        logger.exception("Batch processing failed")
        typer.echo(f"Batch processing failed: {e}", err=True)
        return 1


def _expand_input_pattern(pattern: str) -> list[str]:
    """Expand input pattern (glob or comma-separated) to list of validated file paths.

    Args:
        pattern: Glob pattern (e.g., 'inputs/*.txt') or comma-separated list of files.

    Returns:
        List of absolute paths to existing files.

    Raises:
        FileNotFoundError: If no files are found.
    """
    import glob as glob_module

    # Check if it's a comma-separated list
    if "," in pattern:
        files = [f.strip() for f in pattern.split(",") if f.strip()]
        validated = []
        for f in files:
            path = Path(f).resolve()
            if not path.exists():
                raise FileNotFoundError(f"File not found: {f}")
            if not path.is_file():
                raise ValueError(f"Not a file: {f}")
            ext = path.suffix.lower()
            if ext not in [".txt", ".md", ".pdf", ".docx"]:
                raise ValueError(f"Unsupported file type '{ext}': {f}")
            validated.append(str(path))
        return validated

    # Otherwise, treat as glob pattern
    matched = glob_module.glob(pattern, recursive=True)

    if not matched:
        return []

    validated = []
    for f in matched:
        path = Path(f).resolve()
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in [".txt", ".md", ".pdf", ".docx"]:
            continue
        validated.append(str(path))

    return validated


@app.command()
def status(
    input_file: Annotated[
        str,
        typer.Argument(
            ...,
            help="Path to the input file that was submitted to the pipeline.",
        ),
    ],
) -> int:
    """Display the progress and status of a pipeline run for a given input file.

    Args:
        input_file: Path to the input file that was submitted to the pipeline.

    Returns:
        int: Exit code (0 for success, 1 for error)

    Examples:
        # Check status of pipeline for input.txt
        $ longtext status input.txt

        # Check status of pipeline for markdown file
        $ longtext status document.md

        # Check status of pipeline for DOCX file
        $ longtext status document.docx
    """
    try:
        # Step 1: Validate input file exists
        input_path = Path(input_file).resolve()
        if not input_path.exists():
            typer.echo(f"Error: Input file '{input_file}' does not exist.", err=True)
            return 1

        if not input_path.is_file():
            typer.echo(f"Error: Input path is not a file: {input_file}", err=True)
            return 1

        # Check extension (now including PDF and DOCX)
        ext = input_path.suffix.lower()
        if ext not in [".txt", ".md", ".pdf", ".docx"]:
            typer.echo(
                f"Error: Unsupported file type '{ext}'. Supported: .txt, .md, .pdf, .docx.",
                err=True,
            )
            return 1

        # Step 2: Load manifest from .longtext/manifest.json
        manifest_manager = ManifestManager()
        manifest = manifest_manager.load_manifest(str(input_path))

        # Step 3: Check if manifest exists
        if manifest is None:
            typer.echo(
                f"Analysis status not found for file '{input_file}'. Has it been processed?",
                err=True,
            )
            return 1

        # Step 4: Format and display status using renderer
        status_output = format_status(manifest, show_details=True)
        typer.echo(status_output)

        return 0

    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        return 1
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        return 1
    except PermissionError as e:
        typer.echo(f"Error: {e}", err=True)
        return 1
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        return 1


@app.command()
def init(
    dir: Annotated[
        str,
        typer.Option(
            "--dir",
            "-d",
            help="Directory where to create configuration files. Default is current directory.",
        ),
    ] = ".",
) -> int:
    """Create default configuration files in the current directory for customization.

    Args:
        dir: Directory where to create configuration files. Default is current directory.

    Returns:
        int: Exit code (0 for success, 1 for error)

    Examples:
        # Create default configs in current directory
        $ longtext init

        # Create configs in a specific directory
        $ longtext init --dir /path/to/config/location
    """
    try:
        # Step 1: Validate target directory (check exists, is writable)
        target_path = Path(dir).resolve()

        # Create directory if needed
        if not target_path.exists():
            try:
                target_path.mkdir(parents=True, exist_ok=True)
                typer.echo(f"Created directory: {target_path}")
            except PermissionError:
                typer.echo(
                    f"Error: No permission to create directory: {target_path}", err=True
                )
                return 1
            except Exception as e:
                typer.echo(
                    f"Error: Failed to create directory {target_path}: {e}", err=True
                )
                return 1
        else:
            # Check if directory is writeable
            if not os.access(target_path, os.W_OK):
                typer.echo(f"Error: Directory {target_path} is not writable", err=True)
                return 1

        # Step 2: Create configuration templates
        config_general_content = generate_config_general_template()
        config_relationship_content = generate_config_relationship_template()

        # Step 3: Create files with confirmation if they already exist
        files_to_create = [
            ("config.general.yaml", config_general_content),
            ("config.relationship.yaml", config_relationship_content),
            ("longtext.local.yaml", generate_local_config_template()),
            ("sample_input.txt", generate_sample_input_content()),
            ("README.md", generate_quickstart_readme_content()),
        ]

        for filename, content in files_to_create:
            filepath = target_path / filename
            if filepath.exists():
                # Ask for overwrite confirmation using typer.confirm
                should_overwrite = typer.confirm(
                    f"File {filename} already exists. Overwrite?", default=False
                )

                if not should_overwrite:
                    typer.echo(f"Skipping {filename}...")
                    continue

            try:
                write_file(filepath, content)
                typer.echo(f"Created {filepath}")
            except PermissionError:
                typer.echo(f"Error: No permission to write file: {filepath}", err=True)
                return 1
            except Exception as e:
                typer.echo(f"Error: Failed to write {filepath}: {e}", err=True)
                return 1

        typer.echo(
            f"\nInitialization complete! Configuration files created in: {target_path}"
        )
        typer.echo("\nTo get started:")
        typer.echo("- Review config.general.yaml for general analysis settings")
        typer.echo(
            "- Review config.relationship.yaml for relationship-focused analysis"
        )
        typer.echo(
            f"- Put your local API/model settings in {AUTO_CONFIG_FILENAMES[0]} (auto-loaded on startup)"
        )
        typer.echo("- Create your own input text file (.txt or .md)")
        typer.echo(
            "- Run with: longtext run your_input.txt --config config.general.yaml"
        )

        return 0

    except Exception as e:
        logger.exception("Unexpected error during initialization")
        typer.echo(f"Unexpected error during initialization: {e}", err=True)
        return 1


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-v",
            help="Display the version of the longtext-pipeline tool.",
            callback=version_callback,
        ),
    ] = None,
) -> None:
    """longtext-pipeline - Hierarchical analysis of super-long texts.

    This tool processes long-text documents through a 5-stage pipeline:
    1. Ingest - Read and split input
    2. Summarize - Generate part summaries
    3. Stage - Aggregate summaries
    4. Final - Create final analysis
    5. Audit - Optional post-processing quality check
    """
    pass


def generate_config_general_template() -> str:
    """Generate default general analysis configuration template."""
    default_config = DEFAULT_CONFIG.copy()

    # Remove input-specific path requirement
    config_copy: dict[str, Any] = default_config.copy()
    config_copy["input"]["file_path"] = "YOUR_INPUT_FILE.txt"  # type: ignore[index]
    config_copy["model"]["api_key"] = (
        "${OPENAI_API_KEY}"  # Use environment variable  # type: ignore[index]
    )

    return yaml.dump(config_copy, default_flow_style=False, indent=2)  # type: ignore[no-any-return]


def generate_config_relationship_template() -> str:
    """Generate example relationship-focused analysis configuration template."""
    default_config = DEFAULT_CONFIG.copy()

    # Adapt for relationship analysis mode
    config_copy: dict[str, Any] = default_config.copy()
    config_copy["input"]["file_path"] = "YOUR_INPUT_FILE.txt"  # type: ignore[index]
    config_copy["model"]["api_key"] = (
        "${OPENAI_API_KEY}"  # Use environment variable  # type: ignore[index]
    )
    config_copy["prompts"]["format"] = "relationship"  # type: ignore[index]

    # Update prompt templates for relationship analysis
    config_copy["stages"]["summarize"]["prompt_template"] = (  # type: ignore[index]
        "prompts/summary_relationship.txt"
    )
    config_copy["stages"]["stage"]["prompt_template"] = "prompts/stage_relationship.txt"  # type: ignore[index]
    config_copy["stages"]["final"]["prompt_template"] = "prompts/final_relationship.txt"  # type: ignore[index]

    # Make relationship analysis specific changes
    config_copy["stages"]["audit"]["enabled"] = True  # type: ignore[index]
    config_copy["stages"]["audit"]["prompt_template"] = "prompts/audit_relationship.txt"  # type: ignore[index]

    return yaml.dump(config_copy, default_flow_style=False, indent=2)  # type: ignore[no-any-return]


def generate_local_config_template() -> str:
    """Generate a machine-local runtime config template that is auto-loaded on startup."""
    local_config: dict[str, Any] = {
        "model": {
            "provider": "openai",
            "name": "deepseek-chat",
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "REPLACE_WITH_API_KEY",
            "temperature": 0.7,
            "timeout": 120.0,
        }
    }
    return (  # type: ignore[no-any-return]
        "# Machine-local runtime config. This file is auto-loaded on startup\n"
        "# and should stay out of git.\n"
        + yaml.dump(local_config, default_flow_style=False, indent=2)  # type: ignore[no-any-return]
    )


def generate_sample_input_content() -> str:
    """Generate sample input text demonstrating the tool capabilities."""
    return """# Sample Text for longtext-pipeline Analysis

This is a demonstration text to showcase the capabilities of the longtext-pipeline tool.

## Introduction
The longtext-pipeline is designed to handle extensive text documents by breaking them down into manageable sections and performing hierarchical analysis.

## Topic: Advancements in Large Language Models

Large Language Models (LLMs) have revolutionized the natural language processing landscape. Since the introduction of transformer architecture, models have grown significantly in scale and capability.

### Historical Context
- Early transformer models (2017): Vaswani et al. introduced the seminal "Attention is All You Need" paper
- Scaling breakthroughs (2020): GPT-3 demonstrated emergent properties at scale with 175 billion parameters
- Foundation models era (2022-present): Development of generalist AI systems capable of diverse tasks

### Technical Capabilities
Modern LLMs excel at numerous tasks including:
- Natural language understanding and generation
- Code generation and explanation
- Question answering and reasoning
- Document analysis and summarization
- Content moderation and classification

### Applications
These models find applications across various domains:
- Academic research assistance
- Business document processing
- Educational support systems
- Content creation tools
- Customer service automation

### Challenges and Limitations
Despite their capabilities, LLMs face challenges:
- Computational resource requirements
- Potential for hallucination in outputs
- Bias inheritance from training data
- Interpretability of internal representations
- Energy consumption considerations

## Conclusion
Longtext analysis tools like this pipeline enable scalable insight extraction from extensive text corpora, supporting research, business intelligence, and content management workflows. This technology represents a crucial step toward managing the information explosion in digital formats."""


def generate_quickstart_readme_content() -> str:
    """Generate quickstart README with usage instructions."""
    return """# Longtext-Pipeline Quick Start Guide

Welcome to the longtext-pipeline! This tool helps you analyze super-long texts through hierarchical analysis.

## Setup and Initialization

You've successfully initialized the pipeline configuration using:
```bash
longtext init
```

This created several important files:
- `config.general.yaml` - General purpose text analysis configuration
- `config.relationship.yaml` - Relationship-focused analysis configuration  
- `sample_input.txt` - Demo content to test with
- This `README.md` file with instructions

## Before Running Analysis

1. **Set up your API key:**
   Preferred: put your API/model settings in `longtext.local.yaml`.

   Example:
   ```yaml
   model:
     provider: "openai"
     name: "deepseek-chat"
     base_url: "https://api.deepseek.com/v1"
     api_key: "your_api_key_here"
     timeout: 120.0
   ```

   Or export your API key as an environment variable:
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```

   `longtext.local.yaml` is auto-loaded on startup and is intended for local secrets.

2. **Customize configuration:**
   Review `config.general.yaml` or `config.relationship.yaml` to adjust settings:
   - Change model provider/name in the `model` section
   - Adjust processing parameters in the `stages` section
   - Set your desired output directory

3. **Prepare a text file:**
   - Add a new file in txt or md format
   - Make sure your text is substantial (several paragraphs or more) 

## Running Analysis

To analyze a file using general analysis mode:
```bash
longtext run your_text_file.txt
```

To run with a specific configuration:
```bash
longtext run your_text_file.txt --config config.general.yaml
```

To analyze using relationship-focused mode:
```bash
longtext run your_text_file.txt --mode relationship
```

## Checking Progress

To check status of an ongoing analysis:
```bash
longtext status your_text_file.txt
```

## Configuration Types

- **General Mode:** Standard analysis focused on understanding and summarizing content
- **Relationship Mode:** Entity and relationship discovery mode

## Output Overview

After running, look in the adjacent `.longtext/` working directory for:
- Part files: Individual chunk analysis (`part_*.txt`)
- Summary files: Intermediate summaries (`summary_*.md`)
- Stage files: Aggregated sections (`stage_*.md`)
- Final file: Complete analysis (`final_analysis.md`)
- Manifest: Progress tracking and state (`manifest.json`)
- Metrics: Prometheus export (`metrics.prom`)

## Troubleshooting

1. **API Errors:** Confirm your API key is correctly set and has sufficient quota
2. **Permission Issues:** Check that the output directory is writable
3. **Memory Issues:** For extremely large files, consider adjusting `chunk_size` in the config

## Learn More

- Review detailed configuration options in docs/CONFIG.md
- Check CLI usage information in docs/CLI.md

Ready to analyze your first long text document? Get started with:
```bash
longtext run sample_input.txt
```
"""


def main_callable() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    app()
