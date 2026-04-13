# Example Configurations

This directory contains starter configurations for common longtext-pipeline workflows.

## Files

- `config.default.yaml` — fully annotated reference config
- `config.general.yaml` — standard summarization and analysis
- `config.relationship.yaml` — relationship-focused analysis
- `config.multi_agent.yaml` — multi-perspective final synthesis with specialist models
- `config.performance_test.yaml` — performance-oriented tuning

## Typical usage

```bash
longtext run input.txt --config examples/config.general.yaml
longtext run input.txt --config examples/config.relationship.yaml
longtext run input.txt --config examples/config.multi_agent.yaml --multi-perspective
```

## Notes

- Keep secrets out of committed example files. Use `longtext.local.yaml` or environment variables instead.
- The live runtime writes working artifacts next to the input file in `.longtext/`.
- Bundled prompt templates ship with the package; you generally do not need to edit `prompts.dir` unless you are testing custom prompts.
