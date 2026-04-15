"""
Tests for hallucination detection functionality in the audit stage.

Tests the complete hallucination detection pipeline:
- Claim extraction from analysis text
- LLM-based verification against original document
- Confidence scoring (high/medium/low)
- Evidence tracing
- Overall quality scoring
"""

from unittest.mock import Mock, patch
import tempfile
import os

import pytest

from src.longtext_pipeline.pipeline.audit import (
    AuditStage,
    HallucinationDetectionResult,
)
from src.longtext_pipeline.models import Manifest, FinalAnalysis


def test_extract_claims_from_analysis():
    """Test that claims are correctly extracted from analysis text."""
    # Mock an LLM client for the audit stage to avoid needing API key
    mock_client = Mock()
    audit_stage = AuditStage(llm_client=mock_client)

    analysis_text = """The project launched successfully in January 2023. 
    Key stakeholders included Smith Corp and Johnson Ltd. The budget exceeded expectations. 
    Additionally, customer satisfaction rates improved significantly."""

    claims = audit_stage.extract_claims_from_analysis(analysis_text)

    assert len(claims) > 0
    # Check that meaningful sentences were extracted
    assert any("launched successfully" in claim.lower() for claim in claims)
    assert any("january 2023" in claim.lower() for claim in claims)
    assert any("stakeholders" in claim.lower() for claim in claims)


def test_hallucination_detection_result_creation():
    """Test creation of HallucinationDetectionResult object."""
    result = HallucinationDetectionResult(
        claim="The product sold 1000 units",
        is_hallucinated=True,
        confidence="medium",
        evidence=[
            {
                "source": "file.txt",
                "location": "line 10",
                "quote": "sales were positive",
            }
        ],
        explanation="Claim not supported by source",
    )

    assert result.claim == "The product sold 1000 units"
    assert result.is_hallucinated is True
    assert result.confidence == "medium"
    assert result.explanation == "Claim not supported by source"
    assert len(result.evidence) == 1


@patch(
    "src.longtext_pipeline.llm.openai_compatible.OpenAICompatibleClient.complete_json"
)
def test_check_claim_validity_supported(mock_complete_json):
    """Test that supported claims are correctly identified."""
    # Mock the LLM response to indicate the claim is supported
    mock_response = {
        "supported": True,
        "explanation": "The claim is directly stated in the source",
        "evidence_location": "paragraph 3",
        "confidence": "high",
        "quote": "As stated in the document, the project was completed successfully",
    }
    mock_complete_json.return_value = mock_response

    # Create a mock client for actual testing
    mock_client = Mock()
    mock_client.complete_json.return_value = mock_response

    # Create audit instance with mocked client
    audit_stage = AuditStage(llm_client=mock_client)

    # Create a mock manifest
    manifest = Manifest(
        session_id="test",
        input_path="test.txt",
        input_hash="hash",
        stages={},
        created_at=None,
        updated_at=None,
        status="test",
    )

    # Create a temporary input file
    with tempfile.NamedTemporaryFile(
        suffix=".txt", mode="w", delete=False
    ) as temp_file:
        temp_file.write("The project was completed successfully.")
        temp_file_path = temp_file.name

    manifest.input_path = temp_file_path

    try:
        # Check a claim that should be supported
        result = audit_stage.check_claim_validity(
            "The project was completed successfully", manifest, "general"
        )

        assert result.is_hallucinated is False
        assert result.confidence == "high"
        assert "directly stated" in result.explanation
        mock_client.complete_json.assert_called_once()  # Verify LLM was called
    finally:
        # Clean up
        os.unlink(temp_file_path)


@patch(
    "src.longtext_pipeline.llm.openai_compatible.OpenAICompatibleClient.complete_json"
)
def test_check_claim_validity_hallucinated(mock_complete_json):
    """Test that hallucinated claims are correctly identified."""
    # Mock the LLM response to indicate the claim is not supported
    mock_response = {
        "supported": False,
        "explanation": "The claim is not found in the source",
        "evidence_location": "not found",
        "confidence": "high",
        "quote": "",
    }
    mock_complete_json.return_value = mock_response

    # Create mock client
    mock_client = Mock()
    mock_client.complete_json.return_value = mock_response

    # Create audit instance with mocked client
    audit_stage = AuditStage(llm_client=mock_client)

    # Create a mock manifest with source content that does NOT contain this claim
    manifest = Manifest(
        session_id="test",
        input_path="test.txt",
        input_hash="hash",
        stages={},
        created_at=None,
        updated_at=None,
        status="test",
    )

    # Create a temporary input file
    with tempfile.NamedTemporaryFile(
        suffix=".txt", mode="w", delete=False
    ) as temp_file:
        temp_file.write(
            "The budget was under expectations.\nProject timeline extended."
        )
        temp_file_path = temp_file.name

    manifest.input_path = temp_file_path

    try:
        # Check a claim that should NOT be supported
        result = audit_stage.check_claim_validity(
            "Sales exceeded expectations by 200%", manifest, "general"
        )

        assert result.is_hallucinated is True
        assert result.confidence == "high"
        assert "not found" in result.explanation
        mock_client.complete_json.assert_called_once()  # Verify LLM was called
    finally:
        # Clean up
        os.unlink(temp_file_path)


@patch(
    "src.longtext_pipeline.llm.openai_compatible.OpenAICompatibleClient.complete_json"
)
def test_calculate_accuracy_score(mock_complete_json):
    """Test the accuracy scoring calculation."""
    # Create mock client for instantiation
    mock_client = Mock()

    audit_stage = AuditStage(llm_client=mock_client)

    # Create mock results - mix of supported and unsupported claims
    results = [
        HallucinationDetectionResult("Supported claim 1", False, "high", [], ""),
        HallucinationDetectionResult("Unsupported claim", True, "high", [], ""),
        HallucinationDetectionResult("Supported claim 2", False, "medium", [], ""),
        HallucinationDetectionResult("Also-supported claim", False, "high", [], ""),
    ]

    score = audit_stage.calculate_accuracy_score(results)

    # Should have 75% accuracy (3 out of 4 claims correctly made)
    # But with weighted calculation it should be slightly higher due to confidence
    expected_score = (
        (1.0 + 0.0 + 0.7 + 1.0) / 4.0
    ) * 100  # high=1.0, high=0.0, medium=0.7, high=1.0
    assert abs(score - expected_score) < 0.1


@patch(
    "src.longtext_pipeline.llm.openai_compatible.OpenAICompatibleClient.complete_json"
)
def test_calculate_consistency_score(mock_complete_json):
    """Test the consistency scoring calculation."""
    # Create mock client for instantiation
    mock_client = Mock()

    audit_stage = AuditStage(llm_client=mock_client)

    # Create mock final analysis
    final_analysis = FinalAnalysis(
        status="completed",
        stages=[],
        final_result="On one hand, the project was successful. On the other hand, it experienced problems.",
        metadata={},
    )

    manifest = Manifest(
        session_id="test",
        input_path="test.txt",
        input_hash="hash",
        stages={},
        created_at=None,
        updated_at=None,
        status="test",
    )

    consistency_score = audit_stage.calculate_consistency_score(
        final_analysis, manifest
    )

    # Should be lower due to the contradiction words "successful" and "However"
    assert consistency_score < 100
    assert consistency_score > 10  # But should still have minimum score


@patch(
    "src.longtext_pipeline.llm.openai_compatible.OpenAICompatibleClient.complete_json"
)
def test_calculate_coverage_score(mock_complete_json):
    """Test the coverage scoring calculation."""
    # Create mock client for instantiation
    mock_client = Mock()

    audit_stage = AuditStage(llm_client=mock_client)

    # Create mock final analysis and manifest
    final_analysis = FinalAnalysis(
        status="completed",
        stages=[],
        final_result="Final analysis with some content",
        metadata={},
    )

    manifest = Manifest(
        session_id="test",
        input_path="test.txt",
        input_hash="hash",
        stages={},
        created_at=None,
        updated_at=None,
        status="test",
        total_parts=5,
    )

    coverage_score = audit_stage.calculate_coverage_score(final_analysis, manifest)

    # Simple baseline implementation
    assert isinstance(coverage_score, float)
    assert 0 <= coverage_score <= 100


@patch(
    "src.longtext_pipeline.llm.openai_compatible.OpenAICompatibleClient.complete_json"
)
def test_overall_score_calculation(mock_complete_json):
    """Test the overall score calculation."""
    # Create mock client for instantiation
    mock_client = Mock()

    audit_stage = AuditStage(llm_client=mock_client)

    # Score calculation uses 60% accuracy, 25% consistency, 15% coverage
    overall = audit_stage.calculate_overall_score(80, 90, 70)  # 80*.6 + 90*.25 + 70*.15

    expected = (80 * 0.60) + (90 * 0.25) + (70 * 0.15)
    assert overall == expected


def test_get_quality_description():
    """Test conversion of numeric score to description."""
    # Create audit stage without LLM client for methods that don't need it
    mock_client = Mock()
    audit_stage = AuditStage(llm_client=mock_client)

    assert "Excellent" in audit_stage.get_quality_description(95)
    assert "Good" in audit_stage.get_quality_description(80)
    assert "Moderate" in audit_stage.get_quality_description(65)
    assert "Low" in audit_stage.get_quality_description(45)
    assert "Poor" in audit_stage.get_quality_description(25)


@patch("src.longtext_pipeline.pipeline.audit.AuditStage.check_claim_validity")
def test_perform_complete_audit(mock_check_claim_validity):
    """Test the complete audit functionality end-to-end."""
    # Create mock client
    mock_client = Mock()
    audit_stage = AuditStage(llm_client=mock_client)

    # Mock the claim validation to return consistent results
    def mock_validation(claim, manifest, mode):
        # Simple rule: even-length claims are hallucinated
        is_hallucinated = len(claim) % 2 == 0
        return HallucinationDetectionResult(
            claim=claim,
            is_hallucinated=is_hallucinated,
            confidence="high" if not is_hallucinated else "low",
            evidence=[{"source": "test.txt", "location": "N/A", "quote": claim}],
            explanation=f"Claim {'not' if is_hallucinated else ''} supported",
        )

    mock_check_claim_validity.side_effect = mock_validation

    # Create mock analysis with multiple claims
    final_analysis = FinalAnalysis(
        status="completed",
        stages=[],
        final_result="Claim 1. Another slightly longer claim 2. Third claim here.",
        metadata={},
    )

    manifest = Manifest(
        session_id="test",
        input_path="test.txt",
        input_hash="hash",
        stages={},
        created_at=None,
        updated_at=None,
        status="test",
    )

    # Perform audit
    results = audit_stage.perform_complete_audit(
        final_analysis, manifest, "general", {}
    )

    # Check that we get results
    assert "hallucination_results" in results
    assert "overall_score" in results
    assert results["total_claims"] > 0
    assert results["detected_hallucinations"] >= 0


if __name__ == "__main__":
    pytest.main([__file__])
