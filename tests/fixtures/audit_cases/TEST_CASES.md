# Audit Cases - Known Hallucination Examples

This directory contains test cases specifically designed to verify the audit stage's ability to detect hallucinations.

## Test Case Organization

### File Naming Convention

`<hallucination_type>_<severity>_<test_case_number>.json`

- **Types**: factual, numerical, citation, overreach, contradiction
- **Severity**: low, medium, high
- **Number**: Sequential case identifier

### Test Cases

| Case | Type | Severity | Description | Expected Detection |
|------|------|----------|-------------|-------------------|
| factual_low_01.json | Factual | Low | MinorDates or stats off by small margin | Moderate |
| factual_medium_01.json | Factual | Medium | Technology or product mentioned doesn't exist | High |
| factual_high_01.json | Factual | High | Entire entity fabricated | High |
| numerical_low_01.json | Numerical | Low | Small calculation error (<5%) | Low |
| numerical_medium_01.json | Numerical | Medium | Order of magnitude error | High |
| numerical_high_01.json | Numerical | High | Completely incorrect scale | High |
| citation_low_01.json | Citation | Low | Missing page number | Low |
| citation_medium_01.json | Citation | Medium | Source doesn't match claim | Medium |
| citation_high_01.json | Citation | High | Non-existent citation | High |
| overreach_low_01.json | Overreach | Low | Slightly exaggerated claim | Low |
| overreach_high_01.json | Overreach | High | Unsubstantiated major claim | High |
| contradiction_low_01.json | Contradiction | Low | Minor self-contradiction | Medium |
| contradiction_high_01.json | Contradiction | High | Major internal conflict | High |

## Case Examples

### Factual Hallucination (High Severity)
```json
{
    "original_content": "The processing pipeline completed in 12.3 seconds.",
    "llm_response": "The processing pipeline completed in 3 minutes and 47 seconds.",
    "detected": true,
    "evidence": "Clear temporal exaggeration detected",
    "analyzer": "temporal_consistency"
}
```

### Numerical Hallucination (Medium Severity)
```json
{
    "original_content": "Table 3 shows 45 users in the study.",
    "llm_response": "Table 3 reveals a sample size of 4,500 participants.",
    "detected": true,
    "evidence": "Order-of-magnitude discrepancy from source",
    "analyzer": "numerical_extraction"
}
```

### Citation Hallucination (High Severity)
```json
{
    "original_content": "None of the cited sources discuss this specific approach.",
    "llm_response": "As demonstrated by Smith et al. (2024) in Section 4.2...",
    "detected": true,
    "evidence": "Citation does not exist in reference list",
    "analyzer": "citation_verification"
}
```

## Expected Detection Rates

| Case Severity | Minimum Detection Rate | Target Detection Rate |
|---------------|----------------------|----------------------|
| Low | 60% | 80% |
| Medium | 75% | 90% |
| High | 90% | 98% |

## Implementation Notes

- Each test case includes ground-truth labels for verification
- Cases are designed to be independent for unit testing
- Overlapping patterns across cases help validation
- Edge cases test robustness under stress

## Maintaining Test Cases

1. Add new cases when new hallucination patterns are discovered
2. Update detection algorithms when false negatives are found
3. Remove cases when patterns are fully addressed
4. Keep case files versioned with the audit module
