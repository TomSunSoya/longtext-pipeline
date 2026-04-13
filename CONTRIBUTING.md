# Contributing to longtext-pipeline

Thanks for contributing. This guide is written for first-time contributors and for maintainers doing release-quality changes.

## Getting started

1. Fork the repository on GitHub.
2. Clone your fork:

```bash
git clone https://github.com/TomSunSoya/longtext-pipeline.git
cd longtext-pipeline
```

3. Create a feature branch:

```bash
git checkout -b feature/your-change
```

## Development setup

Create a virtual environment:

```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate

# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project with development dependencies:

```bash
pip install -e ".[dev]"
```

Verify the local setup:

```bash
longtext --version
pytest tests/
ruff check .
```

## Local configuration

Runtime secrets should live in `longtext.local.yaml` or `.longtext.local.yaml`, or in environment variables such as `OPENAI_API_KEY`.

Do not commit local provider credentials or machine-specific overrides.

## Quality gates

Run these before opening a pull request:

```bash
ruff format .
ruff check .
mypy src/longtext_pipeline
pytest tests/
python -m pip wheel . -w .tmp/wheel-check --no-deps
```

The wheel build check matters: the project ships prompt templates as package data, so packaging regressions can break installed or containerized runs even when editable installs still work.

## Testing guidance

- Add or update tests for user-visible behavior changes.
- Prefer focused unit tests for parser, config, and pipeline helpers.
- Add regression tests for packaging, streaming, resume, or manifest bugs.
- If a behavior differs between editable installs and packaged installs, test the packaged path explicitly.

Useful commands:

```bash
pytest tests/test_streaming.py -q
pytest tests/test_token_budget.py -q
pytest tests/test_openai_compatible_regressions.py -q
```

## Documentation expectations

If behavior, flags, configuration, or operational caveats change, update the relevant docs in the same PR:

- `README.md` for installation, quickstart, and user-facing features
- `docs/CLI.md` for command semantics
- `docs/CONFIG.md` for configuration behavior
- `docs/ARCHITECTURE.md` for structural or operational changes
- `examples/README.md` when example configs change meaning

## Pull requests

Good PRs are small, explicit, and easy to verify. Please include:

1. A short summary of what changed
2. Why the change was needed
3. How you verified it
4. Any caveats or follow-up work

External pull requests require maintainer review before merge.

## Commit style

Use short imperative commit messages:

```text
feat: add multi-perspective final analysis
fix: decode streaming error bodies before handling
docs: update configuration and packaging guidance
chore: verify packaged prompt templates in CI
```

## Security and community

- For sensitive vulnerabilities, follow [SECURITY.md](SECURITY.md) instead of opening a public issue.
- Community participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
