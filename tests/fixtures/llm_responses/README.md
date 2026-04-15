# Mock LLM Responses for Testing

This directory contains mock LLM API responses simulating various scenarios.

## Fixture Organization

### Response Types

| File | Purpose | Dimensions |
|------|---------|------------|
| `summarization_success.json` | Normal chunk summarization | 200-500 words |
| `summarization_streaming.json` | Chunk summarization with streaming | 200-500 words |
| `summarization_empty.json` | Edge case: empty/very short input | < 50 words |
| `summarization_error.json` | Error simulation for retry testing | N/A |
| `stage_synthesis_success.json` | Stage aggregation result | 500-1000 words |
| `stage_synthesis_streaming.json` | Streaming stage synthesis | 500-1000 words |
| `stage_synthesis_empty.json` | Edge case: no summaries to aggregate | < 100 words |
| `final_analysis_success.json` | Final synthesis result | 1000-2000 words |
| `final_analysis_streaming.json` | Streaming final analysis | 1000-2000 words |
| `final_analysis_multi_agent.json` | Multi-perspective analysis | 3000+ words |
| `relationship_mode_success.json` | Relationship-focused analysis | 1000-1500 words |
| `relationship_mode_streaming.json` | Streaming relationship analysis | 1000-1500 words |
| `audit_success.json` | Audit result | 200-500 words |
| `audit_streaming.json` | Streaming audit result | 200-500 words |

### Format

Each JSON file contains:
```json
{
    "request_id": "uuid-string",
    "model": "gpt-4o-mini",
    "prompt_tokens": 1234,
    "completion_tokens": 567,
    "total_tokens": 1801,
    "timestamp": "2026-04-14T12:00:00Z",
    "response": {
        "content": "The actual response text",
        "finish_reason": "stop",
        "logprobs": null
    }
}
```

### Streaming Responses

Streaming responses include multiple chunks:
```json
{
    "request_id": "uuid-string",
    "model": "gpt-4o-mini",
    "chunks": [
        {
            "delta": {"role": "assistant", "content": "This is "},
            "timestamp": "2026-04-14T12:00:00.000Z"
        },
        {
            "delta": {"role": "assistant", "content": "chunked "},
            "timestamp": "2026-04-14T12:00:00.100Z"
        },
        {
            "delta": {"role": "assistant", "content": "response."},
            "timestamp": "2026-04-14T12:00:00.200Z"
        }
    ]
}
```

## Usage in Tests

```python
import json
from pathlib import Path

def test_summarization_fixture():
    fixture_path = Path("tests/fixtures/llm_responses/summarization_success.json")
    data = json.loads(fixture_path.read_text())
    
    assert "response" in data
    assert "content" in data["response"]
    assert data["response"]["finish_reason"] == "stop"
```

## Creating New Fixtures

To add a new mock response:

1. Create JSON file following naming convention:
   - `<stage>_<scenario>_<mode>.json`
   - Stage: `summarization`, `stage_synthesis`, `final_analysis`, `audit`
   - Scenario: `success`, `error`, `empty`, `timeout`, `rate_limit`
   - Mode: `normal`, `streaming`, `batch`

2. Include realistic token counts based on content length

3. Add timestamp in ISO 8601 format

4. For streaming, include at least 3 chunks with realistic timing
