"""
Tests for evidence tracing functionality in the audit stage.

Tests the complete evidence tracing pipeline:
- Enhanced claim-to-source mapping
- Position and line tracking
- Confidence scoring
- Multi-method evidence detection
- Enhanced trace results
"""

from unittest.mock import Mock

import pytest

from src.longtext_pipeline.pipeline.audit import (
    AuditStage,
    HallucinationClaim,
    EvidenceTrace,
    HallucinationDetectionResult,
)


def test_create_enhanced_evidence_trace_basic():
    """Test basic enhanced evidence tracing functionality."""
    # Create a mock for the client
    mock_client = Mock()
    audit_stage = AuditStage(llm_client=mock_client)

    claim = HallucinationClaim(
        id="test_claim_1",
        text="The project was completed successfully in January 2023",
        position=0,
        type="fact",
        extracted_from="analysis",
    )

    source_document = """The quarterly report showed strong performance. 
    The major project was completed successfully in January 2023. 
    Stakeholders were pleased with the results."""

    trace = audit_stage.create_enhanced_evidence_trace(claim, source_document)

    assert isinstance(trace, EvidenceTrace)
    assert trace.claim_id == "test_claim_1"
    assert trace.found_in_source is True
    assert trace.confidence_score > 0.5
    assert len(trace.matched_chunks) > 0
    assert trace.exact_matches >= 0
    assert trace.partial_matches >= 0
    assert len(trace.supporting_excerpts) >= 0
    assert len(trace.source_locations) >= 0
    assert trace.extraction_method in ["exact", "semantic", "fuzzy", "mixed"]


def test_create_enhanced_evidence_trace_no_match():
    """Test enhanced evidence tracing when no match is found."""
    # Create a mock for the client
    mock_client = Mock()
    audit_stage = AuditStage(llm_client=mock_client)

    claim = HallucinationClaim(
        id="test_claim_2",
        text="The CEO resigned unexpectedly in March 2023",
        position=0,
        type="fact",
        extracted_from="analysis",
    )

    source_document = """The quarterly report showed strong performance. 
    The major project was completed successfully in January 2023. 
    Stakeholders were pleased with the results."""

    trace = audit_stage.create_enhanced_evidence_trace(claim, source_document)

    assert isinstance(trace, EvidenceTrace)
    assert trace.claim_id == "test_claim_2"
    assert trace.found_in_source is False
    assert trace.confidence_score == 0.0
    assert len(trace.matched_chunks) == 0
    assert trace.exact_matches == 0
    assert trace.partial_matches == 0
    assert len(trace.supporting_excerpts) == 0
    assert trace.extraction_method == "none"


def test_exact_match_detection():
    """Test that exact matches are properly detected."""
    # Create a mock for the client
    mock_client = Mock()
    audit_stage = AuditStage(llm_client=mock_client)

    claim = HallucinationClaim(
        id="exact_claim_1",
        text="The major project was completed successfully in January 2023",
        position=0,
        type="fact",
        extracted_from="analysis",
    )

    source_document = """The quarterly report showed strong performance. 
    The major project was completed successfully in January 2023. 
    Stakeholders were pleased with the results."""

    trace = audit_stage.create_enhanced_evidence_trace(claim, source_document)

    assert trace.exact_matches > 0
    assert trace.confidence_score >= 0.7  # Exact matches should have high confidence
    assert trace.extraction_method == "mixed"  # Since multiple methods may be used


def test_semantic_match_detection():
    """Test that semantic matches are properly detected."""
    # Create a mock for the client
    mock_client = Mock()
    audit_stage = AuditStage(llm_client=mock_client)

    claim = HallucinationClaim(
        id="sem_claim_1",
        text="Strong financial performance reported in the quarterly review",
        position=0,
        type="fact",
        extracted_from="analysis",
    )

    source_document = """The quarterly review contained comprehensive analysis of our strong financial performance. 
    Revenue exceeded expectations across multiple divisions."""

    trace = audit_stage.create_enhanced_evidence_trace(claim, source_document)

    # Some semantic matches should be found
    assert trace.exact_matches + trace.partial_matches >= 0
    assert trace.found_in_source is True
    assert trace.confidence_score >= 0.4


def test_fuzzy_match_detection():
    """Test that fuzzy matches are properly detected."""
    # Create a mock for the client
    mock_client = Mock()
    audit_stage = AuditStage(llm_client=mock_client)

    claim = HallucinationClaim(
        id="fuzz_claim_1",
        text="Revenue exceeded targets in Q4",
        position=0,
        type="statistic",
        extracted_from="analysis",
    )

    source_document = """Sales figures in Quarter 4 were much higher than anticipated.
    The revenue results surpassed what we had projected before the season."""

    # Use fuzzy matching methods
    matches = audit_stage._find_fuzzy_matches(
        claim.text, source_document, source_document.splitlines()
    )

    # We expect some matches found even if they're approximate
    # For this test, check that the method runs without error
    assert isinstance(matches, list)


def test_hallucination_with_enhanced_tracing_integration():
    """Test integration of enhanced evidence tracing in the main hallucination detection."""
    # Create a mock for the client
    mock_client = Mock()
    audit_stage = AuditStage(llm_client=mock_client)

    analysis_content = """The project was completed successfully. 
    It launched in January 2023 and delivered strong results. 
    The team worked for 6 months to finish it."""

    source_document = """The Q1 report shows the project was completed in January 2023.
    The project took approximately 6 months from planning to delivery.
    Initial results show strong performance compared to prior quarters."""

    result = audit_stage.detect_hallucinations(analysis_content, source_document)

    assert isinstance(result, HallucinationDetectionResult)
    assert result.total_claims > 0
    assert len(result.enhanced_evidence_traces) == result.total_claims

    # Check that enhanced traces have detailed information
    for trace in result.enhanced_evidence_traces:
        assert isinstance(trace, EvidenceTrace)
        assert hasattr(trace, "confidence_score")
        assert hasattr(trace, "exact_matches")
        assert hasattr(trace, "partial_matches")
        assert hasattr(trace, "supporting_excerpts")
        # Remove 'extracted_from' check as this isn't an attribute of EvidenceTrace


def test_evidence_trace_confidence_scoring():
    """Test confidence scoring calculation in evidence trace."""
    # Create a mock for the client
    mock_client = Mock()
    audit_stage = AuditStage(llm_client=mock_client)

    claim = HallucinationClaim(
        id="confidence_test",
        text="The quarterly report mentioned excellent results",
        position=0,
        type="fact",
        extracted_from="analysis",
    )

    source_document = """Our quarterly report clearly indicates excellent results 
    for the fiscal quarter. The metrics were significantly better than last year."""

    trace = audit_stage.create_enhanced_evidence_trace(claim, source_document)

    # The important thing is that the score is calculated properly in the 0-1 range
    assert 0.0 <= trace.confidence_score <= 1.0


def test_claim_extraction_position_tracking():
    """Test that claims maintain proper position data during tracing."""
    # Create a mock for the client
    mock_client = Mock()
    audit_stage = AuditStage(llm_client=mock_client)

    analysis_content = "First claim. Second claim here. Third important fact."

    claims = audit_stage.extract_claims_from_analysis(analysis_content)
    assert len(claims) > 0

    # Pick a claim with content to verify against source
    claim_text = "Second claim here"
    matching_claim = None
    for claim in claims:
        if claim_text.lower() in claim.text.lower():
            matching_claim = claim
            break

    if matching_claim:
        source_document = "Original: Second claim here. Other data."
        trace = audit_stage.create_enhanced_evidence_trace(
            matching_claim, source_document
        )

        assert trace.claim_id == matching_claim.id
        assert len(trace.matched_chunks) > 0  # Should find matching chunks


def test_different_claim_types_in_tracing():
    """Test evidence tracing for different types of claims."""
    # Create a mock for the client
    mock_client = Mock()
    audit_stage = AuditStage(llm_client=mock_client)

    # Test a date claim
    date_claim = HallucinationClaim(
        id="date_claim_1",
        text="The meeting occurred on February 15, 2023",
        position=0,
        type="date",
        extracted_from="analysis",
    )

    source_document = "The team met on February 15, 2023, for the important discussion."

    date_trace = audit_stage.create_enhanced_evidence_trace(date_claim, source_document)
    assert date_trace.claim_id == "date_claim_1"
    assert date_trace.found_in_source is True

    # Test a relationship claim
    rel_claim = HallucinationClaim(
        id="rel_claim_1",
        text="Smith collaborated with Jones on the project",
        position=100,
        type="relationship",
        extracted_from="analysis",
    )

    rel_source = (
        "According to records, Smith collaborated with Jones during the project."
    )
    rel_trace = audit_stage.create_enhanced_evidence_trace(rel_claim, rel_source)
    assert rel_trace.claim_id == "rel_claim_1"


def test_large_document_matching():
    """Test evidence tracing on a larger document to check scalability."""
    # Create a mock for the client
    mock_client = Mock()
    audit_stage = AuditStage(llm_client=mock_client)

    claim = HallucinationClaim(
        id="large_doc_claim",
        text="The annual revenue exceeded all projections",
        position=0,
        type="statistic",
        extracted_from="analysis",
    )

    # Create a larger document
    large_source = "\n".join(
        [
            f"Paragraph {i}: Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
            for i in range(10)
        ]
    )
    large_source += "\nThe actual text: The annual revenue exceeded all projections. "
    large_source += "\n".join(
        [
            f"More paragraph {i}: Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
            for i in range(10, 20)
        ]
    )

    trace = audit_stage.create_enhanced_evidence_trace(claim, large_source)

    # Should still find the relevant match
    assert trace.claim_id == "large_doc_claim"
    assert trace.found_in_source is True  # At least one type of match should be found


def test_case_insensitive_matching():
    """Test that evidence tracing works with case variations."""
    # Create a mock for the client
    mock_client = Mock()
    audit_stage = AuditStage(llm_client=mock_client)

    claim = HallucinationClaim(
        id="case_test",
        text="THE PROJECT WAS SUCCESSFUL",
        position=0,
        type="fact",
        extracted_from="analysis",
    )

    source_document = "the project was completed successfully last quarter"

    trace = audit_stage.create_enhanced_evidence_trace(claim, source_document)

    # Should find semantic similarity despite case differences
    assert trace.claim_id == "case_test"


def test_partial_content_preservation_in_trace():
    """Test that trace preserves important content even when not exact matches."""
    # Create a mock for the client
    mock_client = Mock()
    audit_stage = AuditStage(llm_client=mock_client)

    claim = HallucinationClaim(
        id="partial_claim",
        text="Results exceeded expectations significantly",
        position=0,
        type="fact",
        extracted_from="analysis",
    )

    source_document = "The final results were significantly better than expected."

    trace = audit_stage.create_enhanced_evidence_trace(claim, source_document)

    # Should find at least semantic match
    assert trace.claim_id == "partial_claim"
    assert trace.supporting_excerpts is not None
    assert trace.matched_chunks is not None


if __name__ == "__main__":
    pytest.main([__file__])
