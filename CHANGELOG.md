# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Audit prompt templates for general and relationship modes
- GitHub Actions CI/CD workflow (test, type-check, lint)
- Comprehensive .gitignore for Python projects

### Changed
- Fixed duplicate import in cli.py

## [0.1.0] - 2026-04-03

### Added
- Initial MVP release
- 4-stage processing pipeline (Ingest → Summarize → Stage → Final)
- CLI commands: `run`, `status`, `init`
- YAML-based configuration system with environment variable overrides
- Manifest-based checkpoint and resume functionality
- SHA-256 hash validation for input integrity
- OpenAI-compatible LLM client (supports OpenRouter, Ollama, vLLM)
- Dual analysis modes: general and relationship-focused
- Prompt templates for all pipeline stages
- Comprehensive error handling with Continue-with-Partial strategy
- Unit tests with 100% mock LLM responses (79 tests)
- Documentation: README, ARCHITECTURE, SPEC, CLI, CONFIG, DATA_MODEL

### Technical Details
- Python 3.10+ support
- Dependencies: httpx, pyyaml, typer
- Dev dependencies: pytest, pytest-cov
- Entry point: `longtext` CLI command

[Unreleased]: https://github.com/TomSunSoya/longtext-pipeline/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/TomSunSoya/longtext-pipeline/releases/tag/v0.1.0