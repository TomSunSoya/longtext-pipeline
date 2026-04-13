# Contributing to longtext-pipeline

Thank you for your interest in contributing! This document covers everything you need to know about the development process.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Commit Message Format](#commit-message-format)

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/longtext-pipeline
   cd longtext-pipeline
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### 1. Create a virtual environment

```bash
# macOS/Linux
python -m venv venv
source venv/bin/activate

# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate
```

### 2. Install with dev dependencies

```bash
pip install -e ".[dev]"
```

This installs the package in editable mode along with pytest, ruff, mypy, and other development tools.

### 3. Verify installation

```bash
# Check CLI works
longtext --version

# Run tests to confirm setup
pytest tests/
```

## Code Style

We use **ruff** for formatting and linting. Run these before committing:

```bash
# Format code
ruff format .

# Check for linting issues
ruff check .
```

### Type hints

We use **mypy** for type checking. While not all code is fully typed yet, new code should include type hints where practical:

```bash
# Run type checker
mypy src/
```

### Python version

Minimum Python version is **3.10**. Ensure your code is compatible.

## Testing

### Run all tests

```bash
pytest tests/
```

### Run a specific test file

```bash
pytest tests/test_splitter.py
```

### Run a specific test function

```bash
pytest tests/test_splitter.py::test_function_name -v
```

### Run tests with coverage

```bash
pytest --cov=src/longtext_pipeline --cov-report=term-missing tests/
```

### Test requirements

- All tests must pass before submitting a PR
- New features should include tests
- Existing tests should not be broken (check CI results)

## Pull Request Guidelines

### Before submitting

- [ ] Code passes `ruff format` and `ruff check`
- [ ] All tests pass (`pytest tests/`)
- [ ] MyPy type checking passes (if applicable)
- [ ] Changes are documented (update README or docstrings if needed)

### PR description

Include in your PR:

1. **Summary**: What does this change do?
2. **Motivation**: Why is this change needed?
3. **Testing**: How was it tested?
4. **Breaking changes**: Note any backward-incompatible changes

### Review process

- One maintainer approval required for merge
- CI must pass (tests, linting)
- Address review feedback promptly

## Commit Message Format

We follow a simple convention:

```
<type>(<scope>): <subject>

<body>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Test additions or changes
- `chore`: Build/config/maintenance tasks

### Examples

```
feat(pipeline): add multi-perspective analysis mode

Implemented parallel specialist agents for final synthesis stage.
Configurable via --multi-perspective flag.

fix(splitter): handle edge case with exact chunk boundary

When chunk_size landed exactly on a word boundary, the overlap
calculation was off by one character.

docs: update installation instructions in README

test(summarize): add tests for concurrent worker failure handling
```

## Questions?

Open an issue on GitHub if you have questions or need clarification on anything in this guide.
