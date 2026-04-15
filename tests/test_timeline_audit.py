#!/usr/bin/env python
"""
Test script to verify timeline verification functionality in audit stage.
"""

import tempfile
import os
from unittest.mock import Mock
from src.longtext_pipeline.pipeline.audit import AuditStage
from src.longtext_pipeline.models import FinalAnalysis, Manifest, StageInfo


def test_timeline_verification():
    """Test the timeline verification functionality"""
    print("Testing Timeline Verification in Audit Stage...")

    # Create mock client to avoid needing actual API key
    mock_client = Mock()
    audit_stage = AuditStage(llm_client=mock_client)

    # Simulate having the detailed audit report return a basic response
    def mock_complete(prompt):
        return "No specific issues found in this analysis."

    def mock_complete_json(prompt):
        return {
            "total_claims_reviewed": 3,
            "potential_hallucinations_identified": 1,
            "overall_confidence_in_source_alignment": "medium",
            "potential_hallucinations": [],
            "validated_claims_sample": [],
            "quality_assessment": {
                "factual_accuracy_rating": "acceptable",
                "confidence_score": 70,
                "key_strengths": ["Well-structured"],
                "major_weaknesses": [],
            },
        }

    mock_client.complete = mock_complete
    mock_client.complete_json = mock_complete_json

    # Create a test input with timeline elements
    test_content = """
    John Smith was born on March 15, 1990 in New York City.
    He started his company in January 2015.
    His first product launched in July 2016.
    John passed away on December 10, 2020.
    The company was sold in March 2019.
    """

    # Create a temporary input file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as temp_file:
        temp_file.write(test_content)
        temp_filename = temp_file.name

    # Create mock manifest pointing to the temporary file
    import datetime

    now = datetime.datetime.now()
    stages = {
        "ingest": StageInfo(name="ingest", status="successful"),
        "summarize": StageInfo(name="summarize", status="successful"),
        "stage": StageInfo(name="stage", status="successful"),
        "final": StageInfo(name="final", status="successful"),
        "audit": StageInfo(name="audit", status="not_started"),
    }
    manifest = Manifest(
        session_id="test_session",
        input_path=temp_filename,
        input_hash="test_hash",
        stages=stages,
        created_at=now,
        updated_at=now,
        status="running",
        total_parts=1,
        total_stages=1,
        estimated_tokens=500,
    )

    # Create mock final analysis object
    final_analysis = FinalAnalysis(
        status="completed", stages=[], final_result=test_content, metadata={}
    )

    # Create a config
    config = {
        "model": {
            "provider": "openai",
            "name": "gpt-4",
            "temperature": 0.7,
        },
        "stages": {},
        "input_path": temp_filename,
    }

    try:
        # Run the audit stage - this should include timeline verification
        results = audit_stage.run(
            analysis_objects=final_analysis,
            config=config,
            manifest=manifest,
            mode="general",
        )

        print(f"Status: {results['status']}")
        print(f"Timeline Score: {results['timeline_verification']['timeline_score']}")
        print(
            f"Detected anomalies: {results['timeline_verification']['timeline_anomalies']}"
        )

        if "detected_timeline_anomalies" in results:
            print(
                f"Number of timeline anomalies: {len(results['detected_timeline_anomalies'])}"
            )
            for anomaly in results["detected_timeline_anomalies"][:2]:  # Print first 2
                print(
                    f"  - {anomaly['description']} (Confidence: {anomaly['confidence']})"
                )
        else:
            print("No timeline anomalies detected")

        print("\nTimeline verification test completed successfully!")

        # Check for temporal entities
        if "source_temporal_entities" in results:
            print(
                f"Detected {len(results['source_temporal_entities'])} temporal entities:"
            )
            for entity in results["source_temporal_entities"][:3]:  # Print first 3
                print(
                    f"  - {entity['entity']}: {entity['timestamp_str']} ({entity['event_type']})"
                )

    except Exception as e:
        print(f"Error running test: {e}")
        import traceback

        traceback.print_exc()
        raise

    finally:
        # Clean up temporary file
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)


if __name__ == "__main__":
    test_timeline_verification()
    print("\n✅ Timeline verification functionality test PASSED")
