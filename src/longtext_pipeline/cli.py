"""Command-line interface for longtext-pipeline.

This module provides the CLI entry point using Typer for creating a structured
command-line interface with proper help texts and command routing.
"""

import os
import sys
from pathlib import Path

import typer
import yaml
from typing_extensions import Annotated

from longtext_pipeline.config import DEFAULT_CONFIG, load_config, merge_env_overrides
from longtext_pipeline.utils.io import ensure_dir, write_file
from longtext_pipeline.utils.io import ensure_dir, write_file
from longtext_pipeline.pipeline.orchestrator import LongtextPipeline
from longtext_pipeline.manifest import ManifestManager
from longtext_pipeline.renderer import format_status

app = typer.Typer(
    name="longtext",
    help="A Python CLI tool for hierarchical analysis of super-long texts.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


def version_callback(value: bool) -> None:
    """Callback for --version flag to display version information."""
    if value:
        typer.echo("longtext-pipeline v0.1.0")
        raise typer.Exit()


@app.command()
def run(
    input_file: Annotated[
        str,
        typer.Argument(
            ...,
            help="Path to the input text file (.txt or .md) to be analyzed.",
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
) -> int:
    """Run the hierarchical analysis pipeline on a text file.

    This is the main command that orchestrates the four-stage processing flow:
    Ingest → Summarize → Stage → Final

    Args:
        input_file: Path to the input text file (.txt or .md) to be analyzed.
        config: Optional configuration file path. If not provided, uses default
            configuration or falls back to environment variables.
        mode: Select analysis type. 'general' for standard analysis, 'relationship'
            for entity relationship discovery optimization.
        resume: Resume from existing manifest checkpoint if processing was interrupted.

    Returns:
        int: Exit code (0 for success, 1 for error, 2 for partial success)

    Examples:
        # Basic usage
        $ longtext run input.txt

        # With custom config
        $ longtext run input.md --config /path/to/config.yaml

        # Resume interrupted analysis
        $ longtext run input.txt --resume

        # Run with relationship-focused mode
        $ longtext run input.md --mode relationship
    """
    try:
        # Step 1: Validate input file (exists, valid extension .txt/.md)
        input_path = _validate_input_file(input_file)
        
        # Step 2: Load config from specified path or use defaults
        loaded_config = load_config(config)
        # Apply environment variable overrides
        final_config = merge_env_overrides(loaded_config)
        
        # Update prompts based on mode
        if "prompts" in final_config and mode == "relationship":
            final_config["prompts"]["format"] = "relationship"
        
        # Step 3: Initialize LongtextPipeline from orchestrator
        pipeline = LongtextPipeline()
        
        # Step 4: Execute pipeline with appropriate parameters
        print(f"Starting pipeline for: {input_path}")
        print(f"Mode: {mode}")
        print(f"Resume: {resume}")
        if config:
            print(f"Config file: {config}")
        print()
        
        # Step 5: Run pipeline with error handling
        try:
            final_analysis = pipeline.run(
                input_path=input_path,
                config_path=config,
                mode=mode,
                resume=resume
            )
            
            # Step 6: Determine exit code based on result
            status = final_analysis.status if hasattr(final_analysis, 'status') else "unknown"
            
            if status == "completed":
                print("\n[PASS] Pipeline completed successfully")
                return 0
            elif status in ("partial_success", "completed_with_issues"):
                print("\n[PARTIAL] Pipeline completed with partial results or issues")
                return 2
            else:
                print(f"\n[FAIL] Pipeline failed with status: {status}")
                return 1
                
        except KeyboardInterrupt:
            print("\nPipeline interrupted by user")
            return 1
        except Exception as e:
            print(f"\n[FAIL] Pipeline execution failed: {e}")
            return 1
            
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


def _validate_input_file(input_path: str) -> str:
    """Validate that input file exists and has supported format.
    
    Args:
        input_path: Path to the input file to validate
        
    Returns:
        Resolved absolute path to the input file
        
    Raises:
        FileNotFoundError: If file does not exist
        PermissionError: If file cannot be read
        ValueError: If file extension is not .txt or .md
    """
    path = Path(input_path).resolve()
    
    # Check if file exists
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    
    # Check if file is readable
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")
    
    # Check extension (only txt/md supported)
    ext = path.suffix.lower()
    if ext not in ['.txt', '.md']:
        raise ValueError(f"Unsupported file format. Only .txt and .md files are supported, got: {ext}")
    
    return str(path)


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
        
        # Check extension (only txt/md supported)
        ext = input_path.suffix.lower()
        if ext not in ['.txt', '.md']:
            typer.echo(f"Error: Unsupported file type '{ext}'. Supported: .txt, .md.", err=True)
            return 1
        
        # Step 2: Load manifest from .longtext/manifest.json
        manifest_manager = ManifestManager()
        manifest = manifest_manager.load_manifest(str(input_path))
        
        # Step 3: Check if manifest exists
        if manifest is None:
            typer.echo(f"Analysis status not found for file '{input_file}'. Has it been processed?", err=True)
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
                print(f"Created directory: {target_path}")
            except PermissionError:
                typer.echo(f"Error: No permission to create directory: {target_path}", err=True)
                return 1
            except Exception as e:
                typer.echo(f"Error: Failed to create directory {target_path}: {e}", err=True)
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
            ("sample_input.txt", generate_sample_input_content()),
            ("README.md", generate_quickstart_readme_content()),
        ]
        
        for filename, content in files_to_create:
            filepath = target_path / filename
            if filepath.exists():
                # Ask for overwrite confirmation using typer.confirm
                should_overwrite = typer.confirm(
                    f"File {filename} already exists. Overwrite?",
                    default=False
                )
                
                if not should_overwrite:
                    print(f"Skipping {filename}...")
                    continue
            
            try:
                write_file(filepath, content)
                print(f"Created {filepath}")
            except PermissionError:
                typer.echo(f"Error: No permission to write file: {filepath}", err=True)
                return 1
            except Exception as e:
                typer.echo(f"Error: Failed to write {filepath}: {e}", err=True)
                return 1
        
        print(f"\nInitialization complete! Configuration files created in: {target_path}")
        print("\nTo get started:")
        print("- Review config.general.yaml for general analysis settings")
        print("- Review config.relationship.yaml for relationship-focused analysis")
        print("- Customize API keys and model settings in configuration files")
        print("- Create your own input text file (.txt or .md)")
        print("- Run with: longtext run your_input.txt --config config.general.yaml")
        
        return 0
        
    except Exception as e:
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

    This tool processes long-text documents through a 4-stage pipeline:
    1. Ingest - Read and split input
    2. Summarize - Generate part summaries
    3. Stage - Aggregate summaries
    4. Final - Create final analysis
    """
    pass


def generate_config_general_template() -> str:
    """Generate default general analysis configuration template."""
    default_config = DEFAULT_CONFIG.copy()
    
    # Remove input-specific path requirement
    config_copy = default_config.copy()
    config_copy["input"]["file_path"] = "YOUR_INPUT_FILE.txt"
    config_copy["model"]["api_key"] = "${OPENAI_API_KEY}"  # Use environment variable
    
    return yaml.dump(config_copy, default_flow_style=False, indent=2)


def generate_config_relationship_template() -> str:
    """Generate example relationship-focused analysis configuration template."""
    default_config = DEFAULT_CONFIG.copy()
    
    # Adapt for relationship analysis mode
    config_copy = default_config.copy()
    config_copy["input"]["file_path"] = "YOUR_INPUT_FILE.txt"
    config_copy["model"]["api_key"] = "${OPENAI_API_KEY}"  # Use environment variable
    config_copy["prompts"]["format"] = "relationship"
    
    # Update prompt templates for relationship analysis
    config_copy["stages"]["summarize"]["prompt_template"] = "prompts/summary_relationship.txt"
    config_copy["stages"]["stage"]["prompt_template"] = "prompts/stage_relationship.txt"
    config_copy["stages"]["final"]["prompt_template"] = "prompts/final_relationship.txt"
    
    # Make relationship analysis specific changes
    config_copy["stages"]["audit"]["enabled"] = True
    config_copy["stages"]["audit"]["prompt_template"] = "prompts/audit_relationship.txt"
    
    return yaml.dump(config_copy, default_flow_style=False, indent=2)


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
   Export your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```

   Or edit the config files to directly specify your key.

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

To analyze using relationship-focused mode (experimental):
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
- **Relationship Mode:** Experimental mode focused on identifying connections and relationships between entities/concepts

## Output Overview

After running, look in the output directory (`./output` by default) for:
- Part files: Individual chunk analysis (`part_*.txt`)
- Summary files: Intermediate summaries (`summary_*.md`) 
- Stage files: Aggregated sections (`stage_*.md`)
- Final file: Complete analysis (`final_analysis.md`)
- Manifest: Progress tracking and state (`manifest.json`)

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
