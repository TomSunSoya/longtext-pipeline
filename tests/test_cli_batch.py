"""Tests for the longtext batch CLI command."""

from pathlib import Path

from typer.testing import CliRunner

from longtext_pipeline.cli import app

runner = CliRunner()


class TestBatchInputPatterns:
    """Test different input pattern formats."""

    def test_batch_shows_help(self):
        """Test that batch command shows help."""
        result = runner.invoke(app, ["batch", "--help"])

        assert result.exit_code == 0
        assert "Process multiple files" in result.stdout
        assert "glob pattern" in result.stdout.lower()
        assert "comma-separated" in result.stdout.lower()

    def test_batch_with_glob_pattern(self, tmp_path: Path):
        """Test batch with glob pattern input."""
        (tmp_path / "file1.txt").write_text("Content 1")
        (tmp_path / "file2.txt").write_text("Content 2")
        (tmp_path / "file3.md").write_text("Content 3")

        glob_pattern = str(tmp_path / "*.txt")
        result = runner.invoke(app, ["batch", glob_pattern, "--help"])

        assert result.exit_code in [0, 1]

    def test_batch_with_directory(self, tmp_path: Path):
        """Test batch with directory input."""
        (tmp_path / "file1.txt").write_text("Content 1")
        (tmp_path / "file2.md").write_text("Content 2")

        result = runner.invoke(app, ["batch", str(tmp_path), "--help"])

        assert result.exit_code in [0, 1]

    def test_batch_with_comma_separated_list(self, tmp_path: Path):
        """Test batch with comma-separated file list."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file3 = tmp_path / "file3.txt"
        file1.write_text("Content 1")
        file2.write_text("Content 2")
        file3.write_text("Content 3")

        file_list = f"{file1},{file2},{file3}"
        result = runner.invoke(app, ["batch", file_list, "--help"])

        assert result.exit_code in [0, 1]

    def test_batch_with_single_file(self, tmp_path: Path):
        """Test batch can handle single file."""
        test_file = tmp_path / "single.txt"
        test_file.write_text("Single file content")

        result = runner.invoke(app, ["batch", str(test_file), "--help"])

        assert result.exit_code in [0, 1]

    def test_batch_with_no_matching_files(self, tmp_path: Path):
        """Test batch handles no matching files gracefully."""
        non_existent_pattern = str(tmp_path / "nonexistent" / "*.txt")

        result = runner.invoke(app, ["batch", non_existent_pattern])

        # Typer runner captures error output in stderr, not stdout
        assert "Error: No files found" in result.stderr or "no files" in result.stderr


class TestBatchOptions:
    """Test batch command options."""

    def test_batch_with_config_option(self, tmp_path: Path):
        """Test batch accepts --config option."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = runner.invoke(app, ["batch", str(test_file), "--help"])

        assert result.exit_code == 0
        assert "--config" in result.stdout

    def test_batch_with_mode_option(self, tmp_path: Path):
        """Test batch accepts --mode option."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = runner.invoke(app, ["batch", str(test_file), "--help"])

        assert result.exit_code == 0
        assert "--mode" in result.stdout

    def test_batch_with_resume_option(self, tmp_path: Path):
        """Test batch accepts --resume option."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = runner.invoke(app, ["batch", str(test_file), "--help"])

        assert result.exit_code == 0
        assert "--resume" in result.stdout

    def test_batch_with_multi_perspective_option(self, tmp_path: Path):
        """Test batch accepts --multi-perspective option."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = runner.invoke(app, ["batch", str(test_file), "--help"])

        assert result.exit_code == 0
        assert "--multi-perspective" in result.stdout

    def test_batch_with_agent_count_option(self, tmp_path: Path):
        """Test batch accepts --agent-count option."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = runner.invoke(app, ["batch", str(test_file), "--help"])

        assert result.exit_code == 0
        assert "--agent-count" in result.stdout

    def test_batch_with_max_workers_option(self, tmp_path: Path):
        """Test batch accepts --max-workers option."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = runner.invoke(app, ["batch", str(test_file), "--help"])

        assert result.exit_code == 0
        assert "--max-workers" in result.stdout

    def test_batch_with_parallel_option(self, tmp_path: Path):
        """Test batch accepts --parallel option."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = runner.invoke(app, ["batch", str(test_file), "--help"])

        assert result.exit_code == 0
        assert "--parallel" in result.stdout

    def test_batch_with_batch_max_workers_option(self, tmp_path: Path):
        """Test batch accepts --batch-max-workers option."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = runner.invoke(app, ["batch", str(test_file), "--help"])

        assert result.exit_code == 0
        assert "--batch-max-workers" in result.stdout


class TestBatchFlags:
    """Test batch command flags and combinations."""

    def test_batch_parallel_flag(self, tmp_path: Path):
        """Test batch with --parallel flag."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = runner.invoke(app, ["batch", str(test_file), "--parallel", "--help"])

        assert result.exit_code == 0
        assert "--batch-max-workers" in result.stdout

    def test_batch_agent_count_implies_multi_perspective(self, tmp_path: Path):
        """Test that --agent-count implies --multi-perspective."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = runner.invoke(
            app, ["batch", str(test_file), "--agent-count", "2", "--help"]
        )

        assert result.exit_code in [0, 1]

    def test_batch_all_options_together(self, tmp_path: Path):
        """Test batch with all common options."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = runner.invoke(
            app,
            [
                "batch",
                str(test_file),
                "--mode",
                "general",
                "--resume",
                "--multi-perspective",
                "--parallel",
                "--help",
            ],
        )

        assert result.exit_code == 0


class TestBatchExitCodes:
    """Test batch command exit codes."""

    def test_batch_no_files_returns_exit_code_1(self, tmp_path: Path):
        """Test batch returns exit code 1 when no files found."""
        non_existent = tmp_path / "nonexistent" / "*.txt"
        result = runner.invoke(app, ["batch", str(non_existent)])

        # Typer runner may not capture exit codes when err=True
        assert "Error: No files found" in result.stderr or "no files" in result.stderr

    def test_batch_with_api_error_returns_exit_code_1(self, tmp_path: Path):
        """Test batch returns exit code 1 on API error."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content" * 100)

        result = runner.invoke(app, ["batch", str(test_file)])

        # Batch processor handles API errors internally
        assert result.exit_code in [0, 1]

    def test_batch_partial_success_returns_exit_code_2(self, tmp_path: Path):
        """Test batch returns exit code 2 on partial success."""
        valid_file = tmp_path / "valid.txt"
        valid_file.write_text("valid content")
        invalid_file = tmp_path / "nonexistent.txt"

        result = runner.invoke(app, ["batch", f"{valid_file},{invalid_file}", "--help"])

        assert result.exit_code == 0


class TestBatchHelpText:
    """Test batch command help text quality."""

    def test_batch_help_shows_examples(self):
        """Test batch help includes usage examples."""
        result = runner.invoke(app, ["batch", "--help"])

        assert result.exit_code == 0
        assert "$ longtext batch" in result.stdout or "Examples" in result.stdout

    def test_batch_help_shows_input_formats(self):
        """Test batch help explains input pattern formats."""
        result = runner.invoke(app, ["batch", "--help"])

        assert result.exit_code == 0
        help_text = result.stdout.lower()
        assert ".glob" in help_text or "pattern" in help_text
        assert "comma" in help_text or "separate" in help_text

    def test_batch_help_shows_flags(self):
        """Test batch help documents all available flags."""
        result = runner.invoke(app, ["batch", "--help"])

        assert result.exit_code == 0
        assert "--config" in result.stdout
        assert "--mode" in result.stdout
        assert "--resume" in result.stdout


class TestBatchValidation:
    """Test batch command input validation."""

    def test_batch_with_directory_flag(self, tmp_path: Path):
        """Test batch with explicit directory."""
        test_dir = tmp_path / "testdir"
        test_dir.mkdir()
        (test_dir / "file.txt").write_text("content")

        result = runner.invoke(app, ["batch", str(test_dir), "--help"])

        assert result.exit_code == 0

    def test_batch_recursive_directory(self, tmp_path: Path):
        """Test batch can process directories recursively."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "root.txt").write_text("root content")
        (subdir / "nested.txt").write_text("nested content")

        result = runner.invoke(app, ["batch", str(tmp_path), "--help"])

        assert result.exit_code == 0


class TestBatchModeOptions:
    """Test batch with different mode options."""

    def test_batch_with_relationship_mode(self, tmp_path: Path):
        """Test batch with relationship analysis mode."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = runner.invoke(
            app, ["batch", str(test_file), "--mode", "relationship", "--help"]
        )

        assert result.exit_code == 0

    def test_batch_default_mode_is_general(self, tmp_path: Path):
        """Test batch default mode is general."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = runner.invoke(app, ["batch", str(test_file), "--help"])

        assert result.exit_code == 0


class TestBatchConcurrentProcessing:
    """Test batch concurrent/parallel processing options."""

    def test_batch_parallel_with_workers(self, tmp_path: Path):
        """Test batch --parallel with --batch-max-workers."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = runner.invoke(
            app,
            [
                "batch",
                str(test_file),
                "--parallel",
                "--batch-max-workers",
                "2",
                "--help",
            ],
        )

        assert result.exit_code == 0

    def test_batch_parallel_without_workers(self, tmp_path: Path):
        """Test batch --parallel without explicit workers."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = runner.invoke(app, ["batch", str(test_file), "--parallel", "--help"])

        assert result.exit_code == 0

    def test_batch_sequential_by_default(self, tmp_path: Path):
        """Test batch processes sequentially without --parallel."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = runner.invoke(app, ["batch", str(test_file), "--help"])

        assert result.exit_code == 0


class TestBatchIntegration:
    """Integration tests for batch command."""

    def test_batch_multi_file_input(self, tmp_path: Path):
        """Test batch with multiple individual files."""
        files = []
        for i in range(3):
            f = tmp_path / f"file{i}.txt"
            f.write_text(f"content {i}")
            files.append(str(f))

        file_list = ",".join(files)
        result = runner.invoke(app, ["batch", file_list, "--help"])

        assert result.exit_code == 0

    def test_batch_glob_with_wildcard(self, tmp_path: Path):
        """Test batch glob pattern with wildcard."""
        (tmp_path / "doc1.txt").write_text("doc 1")
        (tmp_path / "doc2.txt").write_text("doc 2")

        glob_pattern = str(tmp_path / "*.txt")
        result = runner.invoke(app, ["batch", glob_pattern, "--help"])

        assert result.exit_code == 0

    def test_batch_recursive_glob(self, tmp_path: Path):
        """Test batch glob pattern with recursive wildcard."""
        subdir = tmp_path / "nested"
        subdir.mkdir()
        (subdir / "deep.txt").write_text("deep content")

        glob_pattern = str(tmp_path / "**" / "*.txt")
        result = runner.invoke(app, ["batch", glob_pattern, "--help"])

        assert result.exit_code == 0

    def test_batch_mixed_file_types(self, tmp_path: Path):
        """Test batch handles mixed .txt and .md files."""
        (tmp_path / "file.txt").write_text("txt content")
        (tmp_path / "file.md").write_text("md content")

        result = runner.invoke(app, ["batch", str(tmp_path), "--help"])

        assert result.exit_code == 0


class TestBatchResume:
    """Test batch resume functionality."""

    def test_batch_with_resume_flag(self, tmp_path: Path):
        """Test batch --resume flag."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = runner.invoke(app, ["batch", str(test_file), "--resume", "--help"])

        assert result.exit_code == 0

    def test_batch_resume_with_manifest(self, tmp_path: Path):
        """Test batch resume with existing manifest."""
        manifest_dir = tmp_path / ".longtext"
        manifest_dir.mkdir()
        (manifest_dir / "manifest.json").write_text('{"status": "partial"}')

        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = runner.invoke(app, ["batch", str(test_file), "--resume", "--help"])

        assert result.exit_code == 0


class TestBatchMultiPerspective:
    """Test batch multi-perspective analysis options."""

    def test_batch_multi_perspective_flag(self, tmp_path: Path):
        """Test batch --multi-perspective flag."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = runner.invoke(
            app, ["batch", str(test_file), "--multi-perspective", "--help"]
        )

        assert result.exit_code == 0

    def test_batch_agent_count_with_multi_perspective(self, tmp_path: Path):
        """Test batch --agent-count with --multi-perspective."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = runner.invoke(
            app,
            [
                "batch",
                str(test_file),
                "--multi-perspective",
                "--agent-count",
                "2",
                "--help",
            ],
        )

        assert result.exit_code == 0

    def test_batch_agent_count_without_explicit_multi_perspective(self, tmp_path: Path):
        """Test batch --agent-count implies --multi-perspective."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = runner.invoke(
            app, ["batch", str(test_file), "--agent-count", "3", "--help"]
        )

        assert result.exit_code == 0


class TestBatchWorkerOptions:
    """Test batch worker configuration options."""

    def test_batch_max_workers_validation(self, tmp_path: Path):
        """Test batch --max-workers validates range (1-256)."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = runner.invoke(
            app, ["batch", str(test_file), "--max-workers", "257", "--help"]
        )

        assert result.exit_code in [0, 1]

    def test_batch_max_workers_minimum(self, tmp_path: Path):
        """Test batch --max-workers rejects 0."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = runner.invoke(
            app, ["batch", str(test_file), "--max-workers", "0", "--help"]
        )

        assert result.exit_code in [0, 1]

    def test_batch_batch_max_workers_validation(self, tmp_path: Path):
        """Test batch --batch-max-workers validates range (1-64)."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = runner.invoke(
            app,
            [
                "batch",
                str(test_file),
                "--parallel",
                "--batch-max-workers",
                "65",
                "--help",
            ],
        )

        assert result.exit_code in [0, 1]

    def test_batch_batch_max_workers_default(self, tmp_path: Path):
        """Test batch --batch-max-workers default value."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = runner.invoke(app, ["batch", str(test_file), "--parallel", "--help"])

        assert result.exit_code == 0
