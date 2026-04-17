# Repository Guidelines

## Project Structure & Module Organization

Core code lives in `src/longtext_pipeline/`. Key areas include `pipeline/` for the five-stage workflow, `llm/` for provider/client logic, `utils/` for helpers such as locking and batch processing, and `prompts/` for packaged prompt templates. Tests live in `tests/` and follow the runtime structure closely. User-facing documentation is in `docs/`, example configs are in `examples/`, and packaging/runtime entry points are defined in `pyproject.toml` and `longtext.py`.

## Build, Test, and Development Commands

Set up a local environment with:

```bash
pip install -e ".[dev]"
```

Use these commands before opening a PR:

```bash
ruff format .
ruff check .
mypy src/longtext_pipeline
pytest tests/
python -m pip wheel . -w .tmp/wheel-check --no-deps
```

The wheel build matters because prompt templates are shipped as package data. For a quick smoke test, run `longtext run input.txt` or `longtext status input.txt`.

## Coding Style & Naming Conventions

Follow standard Python style with 4-space indentation, type hints where they improve clarity, and small focused functions. Use `snake_case` for modules, functions, and test files; use descriptive filenames such as `test_orchestrator_paths.py` or `pdf_extraction.py`. Keep CLI and config behavior synchronized with `README.md`, `docs/CLI.md`, and `docs/CONFIG.md`.

## Testing Guidelines

Pytest is the test runner; async tests use `pytest-asyncio`. Name tests `test_*.py` and prefer targeted regression coverage for pipeline state, resume, packaging, streaming, and config precedence. When behavior differs between editable and packaged installs, validate both paths. Add or update tests in the same PR as user-visible behavior changes.

## Commit & Pull Request Guidelines

Use short imperative commit messages with prefixes already used in the repo: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`. PRs should explain what changed, why it changed, how it was verified, and any caveats. If flags, config semantics, or operational behavior change, update docs and examples in the same PR.

## Security & Configuration Tips

Do not commit secrets or machine-local overrides. Keep credentials in `longtext.local.yaml`, `.longtext.local.yaml`, or environment variables such as `OPENAI_API_KEY`. Review `SECURITY.md` for vulnerability reporting.
