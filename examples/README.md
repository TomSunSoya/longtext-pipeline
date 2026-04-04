# Example Configuration Templates

Two example configuration files are provided in this directory:

## `config.general.yaml`

Standard configuration for general-purpose text analysis and summarization.

**Use this for:**
- Meeting transcripts and minutes
- Project documentation synthesis
- Knowledge base summarization
- Chat log analysis
- General document processing

**Key settings:**
- Model: gpt-4o-mini (cost-effective, fast)
- Chunk size: 4000 characters
- Temperature: 0.7 (balanced creativity)
- Format: "general"

## `config.relationship.yaml` (EXPERIMENTAL)

Specialized configuration for entity relationship mapping and network analysis.

**Use this for:**
- Organizational network mapping
- Stakeholder relationship extraction
- Communication flow analysis
- Timeline reconstruction
- Conflict/opportunity identification

**Key differences from general:**
- Model: gpt-4o (higher quality for nuanced relationships)
- Smaller chunks: 3500 characters (preserve entity context)
- Lower temperature: 0.5 (more consistent entity naming)
- Higher overlap: 0.15 (capture spanning relationships)
- Format: "relationship"

## Usage

```bash
# General analysis
longtext run my_document.txt --config examples/config.general.yaml

# Relationship analysis (experimental)
longtext run meeting_transcript.txt --config examples/config.relationship.yaml
```

## Environment Variables

Both configurations support environment variable overrides:

- `OPENAI_API_KEY` - Your OpenAI API key (required)
- `OPENAI_BASE_URL` - Custom API endpoint
- `LONGTEXT_MODEL_PROVIDER` - Override provider
- `LONGTEXT_MODEL_NAME` - Override model name
- `LONGTEXT_OUTPUT_DIR` - Custom output directory
- `LONGTEXT_PROMPTS_DIR` - Custom prompts directory

## Configuration Reference

All configuration fields are documented inline with comments. Key sections:

- `model` - LLM provider and model settings
- `stages` - Per-stage processing parameters (ingest, summarize, stage, final, audit)
- `prompts` - Prompt template directory and format
- `output` - Output location and file naming
- `input` - Input file path and encoding
- `pipeline` - General pipeline behavior
