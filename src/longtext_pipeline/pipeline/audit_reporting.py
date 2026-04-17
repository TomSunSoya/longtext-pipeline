"""Helpers for audit report prompt construction."""

from __future__ import annotations

from pathlib import Path

from ..utils.io import read_file
from ..utils.token_budget import TokenBudgetManager
from .audit_types import HallucinationDetectionResult


def load_audit_prompt(mode: str, prompt_dir: Path) -> str:
    """Load the preferred audit prompt template for the given mode."""
    preferred = prompt_dir / "hallucination_detection.txt"
    if preferred.exists():
        return read_file(str(preferred))

    fallback = prompt_dir / f"audit_{mode}.txt"
    return read_file(str(fallback))


def build_budgeted_audit_prompt(
    prompt_template: str,
    source_document: str,
    analysis_content: str,
    hallucination_result: HallucinationDetectionResult,
    context_window: int,
    max_output_tokens: int = 2000,
) -> str:
    """Build a compact, token-budgeted prompt for detailed audit reporting."""
    budget_manager = TokenBudgetManager(max_output_tokens=max_output_tokens)
    source_focus = _build_source_focus(source_document, hallucination_result)
    analysis_focus = _compress_text(analysis_content, target_chars=6000)
    suspicious_claims = _format_suspicious_claims(hallucination_result)

    full_prompt = (
        f"{prompt_template}\n\n"
        "AUTOMATED AUDIT SUMMARY:\n"
        f"- Total claims reviewed: {hallucination_result.total_claims}\n"
        f"- Verified claims: {hallucination_result.verified_claims}\n"
        f"- Potential hallucinations: {hallucination_result.hallucinated_claims}\n"
        f"- Confidence score: {hallucination_result.confidence_score}/100\n"
        f"- Quality assessment: {hallucination_result.quality_assessment}\n\n"
        "POTENTIAL ISSUES TO REVIEW:\n"
        f"{suspicious_claims}\n\n"
        "ANALYSIS EXCERPT TO AUDIT:\n"
        f"{analysis_focus}\n\n"
        "SOURCE EVIDENCE SNAPSHOT:\n"
        f"{source_focus}\n\n"
        "Instructions:\n"
        "- Ground every conclusion in the provided analysis excerpt and source evidence snapshot.\n"
        "- If the supplied evidence is insufficient, say that explicitly.\n"
        "- Prioritize concrete hallucination risks, unsupported claims, and evidence gaps.\n"
    )

    processed_prompt, _ = budget_manager.process_prompt_with_budget(
        prompt=full_prompt,
        system_prompt=None,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
    )
    return processed_prompt


def _format_suspicious_claims(
    hallucination_result: HallucinationDetectionResult,
) -> str:
    """Format the highest-signal suspicious claims for prompt inclusion."""
    items: list[str] = []
    for index, item in enumerate(
        hallucination_result.detected_hallucinations[:8], start=1
    ):
        claim_text = item.get("claim") or item.get("text") or "Unknown claim"
        explanation = item.get("explanation") or "No explanation provided."
        confidence = item.get("confidence") or item.get("confidence_score") or "unknown"
        items.append(
            f"{index}. Claim: {claim_text}\n"
            f"   Confidence: {confidence}\n"
            f"   Explanation: {explanation}"
        )

    if not items:
        return (
            "No specific hallucination candidates were flagged by the automated scan."
        )

    return "\n".join(items)


def _build_source_focus(
    source_document: str,
    hallucination_result: HallucinationDetectionResult,
) -> str:
    """Build a compact evidence-focused source snapshot."""
    excerpts: list[str] = []
    seen: set[str] = set()

    for trace in hallucination_result.enhanced_evidence_traces[:8]:
        for excerpt in trace.supporting_excerpts[:2]:
            normalized = excerpt.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                excerpts.append(normalized)

    if excerpts:
        joined = "\n\n".join(f"- {excerpt}" for excerpt in excerpts[:10])
        return _compress_text(joined, target_chars=7000)

    return _compress_text(source_document, target_chars=7000)


def _compress_text(text: str, target_chars: int) -> str:
    """Keep the beginning and end of a long text with a gap marker in the middle."""
    normalized = text.strip()
    if len(normalized) <= target_chars:
        return normalized

    head_chars = int(target_chars * 0.65)
    tail_chars = target_chars - head_chars
    head = normalized[:head_chars].rstrip()
    tail = normalized[-tail_chars:].lstrip()
    return f"{head}\n\n[... truncated for audit budget ...]\n\n{tail}"
