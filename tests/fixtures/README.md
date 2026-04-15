# Test Fixtures

This directory contains test fixtures for the longtext-pipeline project.

## Directory Structure

```
tests/fixtures/
├── pdfs/              # PDF document test fixtures
├── docx/              # DOCX document test fixtures
├── llm_responses/     # Mock LLM API responses
└── audit_cases/       # Hallucination detection test cases
```

## PDF Fixtures (`tests/fixtures/pdfs/`)

Contains 10 sample PDF files covering various document types and complexity levels.

### Text-Based PDFs (3 files)
| File | Size | Description |
|------|------|-------------|
| `01_simple.pdf` | ~0.00 MB | Single-column text document |
| `02_multi_column.pdf` | ~0.00 MB | Two-column layout for column parsing tests |
| `03_with_tables.pdf` | ~0.00 MB | Document with embedded data tables |

### Scanned PDFs (3 files)  
| File | Size | Description |
|------|------|-------------|
| `04_scanned_simple.pdf` | ~0.00 MB | Simulated scanned document, requires OCR |
| `05_scanned_multi_page.pdf` | ~0.00 MB | Multi-page scanned document with noise |
| `06_scanned_form.pdf` | ~0.00 MB | Scanned form/document with form elements |

### Hybrid PDFs (2 files)
| File | Size | Description |
|------|------|-------------|
| `07_hybrid_report.pdf` | ~0.00 MB | Mix of text and image placeholders |
| `08_hybrid_document.pdf` | ~0.00 MB | Complex layout with mixed content |

### Special Test PDFs (2 files)
| File | Size | Description |
|------|------|-------------|
| `09_encrypted_restrictions.pdf` | ~0.00 MB | Encrypted document for error testing |
| `10_large_document.pdf` | ~0.07 MB | 105-page document for pagination tests |

### PDF Usage

```python
from pathlib import Path

pdf_dir = Path("tests/fixtures/pdfs")

# Access specific PDFs
simple_pdf = pdf_dir / "01_simple.pdf"
large_pdf = pdf_dir / "10_large_document.pdf"

# Test various scenarios
assert simple_pdf.exists()
assert large_pdf.stat().st_size > 10000  # 10KB min
```

## DOCX Fixtures (`tests/fixtures/docx/`)

Contains 5 sample DOCX files covering different formatting scenarios.

### DOCX Files (5 files)
| File | Size | Description |
|------|------|-------------|
| `01_simple.docx` | ~35 KB | Basic document without special features |
| `02_with_tables.docx` | ~35 KB | Document with tables (simple and nested) |
| `03_tracked_changes.docx` | ~35 KB | Document simulating tracked changes |
| `04_with_images.docx` | ~35 KB | Document with image placeholders |
| `05_complex_formatting.docx` | ~35 KB | Document with styles, colors, lists |

### DOCX Usage

```python
from pathlib import Path

docx_dir = Path("tests/fixtures/docx")

# Load and parse
docx_file = docx_dir / "02_with_tables.docx"
assert docx_file.exists()

# Verify DOCX is valid ZIP (DOCX format)
import zipfile
with zipfile.ZipFile(docx_file) as zf:
    assert "document.xml" in zf.namelist()
```

## LLM Responses (`tests/fixtures/llm_responses/`)

Mock API responses simulating various LLM interactions.

### Response Types

| File | Purpose | Key Features |
|------|---------|--------------|
| `summarization_success.json` | Normal chunk summarization | 200-500 words |
| `summarization_streaming.json` | Streaming response | Multiple chunks |
| `summarization_empty.json` | Edge case handling | Minimal content |
| `stage_synthesis_success.json` | Stage aggregation | 500-1000 words |
| `stage_synthesis_empty.json` | Empty aggregation edge case | Minimal output |
| `final_analysis_success.json` | Final synthesis | 1000-2000 words |
| `final_analysis_multi_agent.json` | Multi-perspective analysis | 3000+ words, 3 agents |
| `relationship_mode_success.json` | Relationship analysis | Entity mapping |
| `audit_success.json` | Audit result | Quality metrics |

### LLM Response Format

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

### Streaming Response Format

```json
{
    "request_id": "uuid-string",
    "model": "gpt-4o-mini",
    "chunks": [
        {
            "delta": {"role": "assistant", "content": "chunk text"},
            "timestamp": "2026-04-14T12:00:00.100Z"
        }
    ]
}
```

## Audit Cases (`tests/fixtures/audit_cases/`)

Test cases for hallucination detection in LLM outputs.

### Case Types

| Type | Description | Severity Range |
|------|-------------|----------------|
| `factual` | Incorrect facts or figures | low, medium, high |
| `numerical` | Wrong numbers or scales | low, medium, high |
| `citation` | Fake or incorrect references | low, medium, high |
| `overreach` | Unsubstantiated claims | low, medium, high |
| `contradiction` | Internal inconsistencies | low, medium, high |

### Test Case Naming

`<type>_<severity>_<number>.json`

Examples:
- `factual_high_01.json` - High severity factual hallucination
- `citation_medium_01.json` - Medium severity citation issue
- `numerical_low_01.json` - Low severity numerical error

### Audit Case Structure

```json
{
    "test_case_id": "factual_high_01",
    "original_text": "原始内容...",
    "llm_generated_output": "生成内容...",
    "hallucination_type": "factual",
    "severity": "high",
    "expected_detection": true,
    "detection_reason": "Why this should be detected",
    "detection_method": "numerical_verification",
    "verification": {
        "original_value": 25,
        "generated_value": 2500,
        "discrepancy_ratio": 100.0
    }
}
```

### Expected Detection Rates

| Severity | Minimum | Target |
|----------|---------|--------|
| Low | 60% | 80% |
| Medium | 75% | 90% |
| High | 90% | 98% |

## Fixture Management

### Adding New Fixtures

1. **Naming**: Use sortable prefixes (`01_`, `02_`, etc.)
2. **Naming**: Be descriptive (`simple`, `multi_column`, `with_tables`)
3. **Size**: Keep most fixtures under 5MB
4. **Validation**: Ensure files are valid PDF/DOCX (not fake extensions)
5. **Documentation**: Update this README with new fixtures

### Git Ignore

Test fixtures are git-ignored via `.gitignore`:
```
# Longtext pipeline working directories
.longtext/
.sisyphus/
```

Generated fixtures go in `tests/fixtures/` which follows the `tests/` pattern in .gitignore.

### Validating Fixtures

```bash
# Check PDF files
python -c "from pathlib import Path; p = Path('tests/fixtures/pdfs/01_simple.pdf'); print(p.read_bytes()[:8])"
# Should output: b'%PDF-1.3\n%\xff\xff\xff\xff'

# Check DOCX files (they're ZIP archives)
python -c "from pathlib import Path; p = Path('tests/fixtures/docx/01_simple.docx'); print(p.read_bytes()[:4])"
# Should output: b'PK\x03\x04'

# Count fixtures
ls tests/fixtures/pdfs/*.pdf | wc -l  # Should be 10
ls tests/fixtures/docx/*.docx | wc -l  # Should be 5
```

## Related Documentation

- **README.md**: Project overview and quickstart
- **docs/ARCHITECTURE.md**: Technical architecture documentation
- **src/longtext_pipeline/pipeline/**: Pipeline stage implementations
