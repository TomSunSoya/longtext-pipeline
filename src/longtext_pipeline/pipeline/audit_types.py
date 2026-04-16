"""Shared audit data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HallucinationClaim:
    """Represents a detected claim from analysis content."""

    id: str
    text: str
    position: int
    type: str
    extracted_from: str

    def __str__(self) -> str:
        """Return the claim text for legacy string-like callers."""
        return self.text

    def lower(self) -> str:
        """Provide string-like lowercase access for older tests/callers."""
        return self.text.lower()


@dataclass
class EvidenceTrace:
    """Represents detailed evidence trace from claims back to source document."""

    claim_id: str
    found_in_source: bool
    matched_chunks: list[dict[str, Any]]
    confidence_score: float
    exact_matches: int
    partial_matches: int
    supporting_excerpts: list[str]
    source_locations: list[tuple[int, int]]
    extraction_method: str


@dataclass
class HallucinationEvidence:
    """Represents evidence found or not found in source document."""

    claim_id: str
    found_in_source: bool
    source_excerpt: str
    source_position: tuple[int, int]
    similarity_score: float


@dataclass
class TimelineEvent:
    """Represents a detected timeline event from analysis content."""

    id: str
    text: str
    entity: str
    event_type: str
    timestamp_str: str
    timestamp_value: str | None
    position: int
    extracted_from: str


@dataclass
class TimelineAnomaly:
    """Represents timeline inconsistency or anomaly."""

    id: str
    type: str
    description: str
    timestamp_a: str
    timestamp_b: str
    event_a: str
    event_b: str
    confidence: float
    explanation: str


@dataclass
class TimelineVerificationResult:
    """Results from timeline verification process."""

    total_events: int
    verified_events: int
    timeline_anomalies: int
    chronological_issues: int
    conflicting_timestamps: int
    timeline_score: int
    detected_anomalies: list[TimelineAnomaly]
    temporal_entities: list[TimelineEvent]
    quality_assessment: str


class HallucinationDetectionResult:
    """Results from hallucination detection process.

    This class supports both the current aggregate-style constructor:

    `HallucinationDetectionResult(total_claims=..., verified_claims=..., ...)`

    and the older per-claim constructor used by legacy tests:

    `HallucinationDetectionResult(claim, is_hallucinated, confidence, evidence, explanation)`
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.total_claims = 0
        self.verified_claims = 0
        self.hallucinated_claims = 0
        self.confidence_score = 0
        self.detected_hallucinations: list[dict[str, Any]] = []
        self.quality_assessment = ""
        self.evidence_trace: list[HallucinationEvidence] = []
        self.enhanced_evidence_traces: list[EvidenceTrace] = []

        self.claim: str | None = None
        self.is_hallucinated: bool | None = None
        self.confidence: str | None = None
        self.evidence: list[Any] = []
        self.explanation: str = ""

        legacy_keys = {
            "claim",
            "is_hallucinated",
            "confidence",
            "evidence",
            "explanation",
        }
        aggregate_keys = {
            "total_claims",
            "verified_claims",
            "hallucinated_claims",
            "confidence_score",
            "detected_hallucinations",
            "quality_assessment",
            "evidence_trace",
            "enhanced_evidence_traces",
        }

        if args and isinstance(args[0], str):
            claim, is_hallucinated, confidence, evidence, explanation = (
                list(args[:5]) + [None] * 5
            )[:5]
            self._init_legacy(
                claim=claim,
                is_hallucinated=bool(is_hallucinated),
                confidence=confidence or "medium",
                evidence=evidence or [],
                explanation=explanation or "",
            )
            return

        if legacy_keys.intersection(kwargs):
            self._init_legacy(
                claim=kwargs.get("claim"),
                is_hallucinated=kwargs.get("is_hallucinated", False),
                confidence=kwargs.get("confidence", "medium"),
                evidence=kwargs.get("evidence", []),
                explanation=kwargs.get("explanation", ""),
            )
            return

        unknown_keys = set(kwargs) - aggregate_keys
        if unknown_keys:
            unknown = ", ".join(sorted(unknown_keys))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}")

        self.total_claims = int(kwargs.get("total_claims", 0))
        self.verified_claims = int(kwargs.get("verified_claims", 0))
        self.hallucinated_claims = int(kwargs.get("hallucinated_claims", 0))
        self.confidence_score = int(kwargs.get("confidence_score", 0))
        self.detected_hallucinations = list(kwargs.get("detected_hallucinations", []))
        self.quality_assessment = str(kwargs.get("quality_assessment", ""))
        self.evidence_trace = list(kwargs.get("evidence_trace", []))
        self.enhanced_evidence_traces = list(kwargs.get("enhanced_evidence_traces", []))

    def _init_legacy(
        self,
        *,
        claim: str | None,
        is_hallucinated: bool,
        confidence: str,
        evidence: list[Any],
        explanation: str,
    ) -> None:
        """Initialize the legacy single-claim result shape."""
        self.claim = claim or ""
        self.is_hallucinated = is_hallucinated
        self.confidence = confidence
        self.evidence = list(evidence)
        self.explanation = explanation

        self.total_claims = 1
        self.verified_claims = 0 if is_hallucinated else 1
        self.hallucinated_claims = 1 if is_hallucinated else 0
        self.confidence_score = {
            "high": 100,
            "medium": 70,
            "low": 40,
        }.get(confidence.lower(), 50)
        self.detected_hallucinations = (
            [
                {
                    "claim": self.claim,
                    "confidence": self.confidence,
                    "explanation": self.explanation,
                }
            ]
            if is_hallucinated
            else []
        )
        self.quality_assessment = confidence


@dataclass
class QualityMetric:
    """Represents a single quality metric score with details."""

    name: str
    score: float
    weight: float
    raw_value: float | None = None
    description: str = ""
    confidence: float = 0.0

    @property
    def Confidence(self) -> float:
        """Backward-compatible alias for older callers."""
        return self.confidence

    @Confidence.setter
    def Confidence(self, value: float) -> None:
        self.confidence = value


@dataclass
class QualityScore:
    """Represents the overall quality score with breakdown."""

    composite_score: float
    metrics: dict[str, QualityMetric]
    confidence_score: float
    quality_assessment: str
    metrics_summary: str
    timestamp: str = ""


@dataclass
class QualityScoringConfig:
    """Configuration for quality scoring computation."""

    metric_weights: dict[str, float] = field(
        default_factory=lambda: {
            "coverage": 0.30,
            "consistency": 0.25,
            "specificity": 0.25,
            "clarity": 0.20,
        }
    )
    min_metric_threshold: float = 40.0
    assessment_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "excellent": 85.0,
            "good": 70.0,
            "fair": 50.0,
            "poor": 0.0,
        }
    )
    enable_confidence_scoring: bool = True
    detail_level: str = "standard"
