"""
Audit stage implementation for longtext pipeline with hallucination detection.

This module provides a comprehensive AuditStage implementation for the v2 pipeline.
Full audit functionality includes hallucination checking, timeline verification,
entity consistency, and quality scoring based on source document comparison.
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from ..llm.openai_compatible import OpenAICompatibleClient
from ..manifest import ManifestManager
from ..models import FinalAnalysis, Manifest
from ..utils.io import read_file
from ..utils.token_estimator import estimate_tokens
from .audit_reporting import build_budgeted_audit_prompt, load_audit_prompt
from .audit_types import (
    EvidenceTrace,
    HallucinationClaim,
    HallucinationDetectionResult,
    HallucinationEvidence,
    QualityMetric,
    QualityScore,
    QualityScoringConfig,
    TimelineAnomaly,
    TimelineEvent,
    TimelineVerificationResult,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class AuditClient(Protocol):
    """Protocol for audit clients (both LLM-backed and offline heuristics)."""

    context_window: int

    def complete(self, prompt: str) -> str: ...

    def complete_json(self, prompt: str) -> dict[str, Any]: ...


class _OfflineAuditClient:
    """Minimal offline client used when no API-backed client can be created."""

    context_window: int = 32000

    def complete(self, prompt: str) -> str:
        return (
            "Detailed audit report unavailable because no LLM client is configured. "
            "Offline audit heuristics were used instead."
        )

    def complete_json(self, prompt: str) -> dict[str, Any]:
        raise RuntimeError("No LLM client available for JSON audit verification.")


class AuditStage:
    """Comprehensive audit stage for hallucination detection and validity checking."""

    def __init__(
        self,
        manifest_manager: Optional[ManifestManager] = None,
        llm_client: Optional[AuditClient] = None,
    ):
        """Initialize the audit stage.

        Args:
            manifest_manager: Optional existing manifest manager
            llm_client: Optional LLM client (if not provided, one will be created)
        """
        self.manifest_manager = manifest_manager or ManifestManager()
        if llm_client is not None:
            self.client = llm_client
        else:
            try:
                self.client = OpenAICompatibleClient()
            except Exception as exc:
                logger.info(
                    "AuditStage running without configured LLM client; falling back to offline heuristics: %s",
                    exc,
                )
                self.client = _OfflineAuditClient()

    def extract_claims_from_analysis(
        self, analysis_content: str
    ) -> List[HallucinationClaim]:
        """Extract factual claims from analysis text for verification.

        Args:
            analysis_content: The text to extract claims from

        Returns:
            List of extracted claims with metadata
        """
        # Use regex or semantic methods to identify claims
        # Look for declarative statements that could be fact-checked

        # Simple sentence detection to start with
        sentences = re.split(r"[.!?]+", analysis_content)

        claims = []
        pos = 0  # Track approximate position in text

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip short fragments
                pos += len(sentence) + 1
                continue

            # Identify likely claims (positive declarative statements)
            # Avoid questions, imperatives, etc.
            if re.match(r"^\s*[A-Z][^.!?]*$", sentence):
                claim = HallucinationClaim(
                    id=f"claim_{i}",
                    text=sentence.strip(),
                    position=pos,
                    type=self.classify_claim_type(sentence.strip()),
                    extracted_from="analysis",
                )
                claims.append(claim)

            pos += len(sentence) + 1

        return claims

    def classify_claim_type(self, text: str) -> str:
        """Classify a claim into semantic categories."""
        text_lower = text.lower()

        if any(
            word in text_lower
            for word in [
                "relationship",
                "works with",
                "conflict",
                "collaboration",
                "interact",
            ]
        ):
            return "relationship"
        elif any(char.isdigit() for char in text):
            # Look for date/time patterns using more flexible matching
            # Check for patterns like "3 PM" where number and am/pm are separate words
            lower_words = text_lower.split()
            for i in range(len(lower_words)):
                if lower_words[i].isdigit() and i + 1 < len(lower_words):
                    if lower_words[i + 1].strip(".!?,") in ["am", "pm", "a.m.", "p.m."]:
                        return "date"  # Time pattern found

            # Also check for the original single-word date/time patterns
            # Replace punctuations that might interfere
            cleaned_for_word_split = re.sub(r"[,.();:&]", " ", text).strip()
            words_with_digits = [
                w for w in cleaned_for_word_split.split() if any(c.isdigit() for c in w)
            ]
            date_patterns = [
                w
                for w in words_with_digits
                if ":" in w  # time: 15:30, 3:00
                or re.search(r"\d+:\d+", w)  # time patterns like 3:00
                or re.search(
                    r"\d+[ap]m\b", w, re.IGNORECASE
                )  # single word time: 3pm, 3am
                or re.search(r"\d{4}-\d{2}-\d{2}", w)  # 2023-05-15
                or re.search(r"\d{2}/\d{2}/\d{4}", w)  # 15/05/2023
                or re.search(r"\d{2}-[A-Za-z]{3}-\d{4}", w)  # 15-Jan-2023
                or re.search(r"-\d{2}-\d{4}", w)  # pattern involving year
                or re.search(r"\d{4}", w)
            ]  # years like 2023
            if date_patterns:
                return "date"
            else:
                return "statistic"
        elif any(
            word in text_lower for word in ["said", "stated", "reported", "mentioned"]
        ):
            return "quote_attribute"
        else:
            return "fact"

    def extract_temporal_entities(
        self, text: str, extract_position: int = 0
    ) -> List[TimelineEvent]:
        """Extract temporal entities (dates, times, durations) from text with associated entities."""
        temporal_pattern = r"""
            (?P<date_time>
                \b\d{4}-\d{2}-\d{2}\b |                # YYYY-MM-DD
                \b\d{2}/\d{2}/\d{4}\b |                # MM/DD/YYYY
                \b\d{2}-[A-Za-z]{3}-\d{4}\b |          # DD-MMM-YYYY
                \b\d{4}\b |                             # just years
                \b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b |  # Month DD, YYYY
                \b\d{1,2}/\d{1,2}/\d{2,4}\b |          # MM/DD/YY or MM/DD/YYYY  
                \b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b |  # HH:MM with optional seconds and AM/PM
                \b\d{1,2}[.-]\d{1,2}[.-]\d{2,4}\b     # DD.MM.YYYY or DD-MM-YYYY
            )
            |(?P<time_ref>
                \b(?:early|late)\s+(?:morning|afternoon|evening)\b |
                \byesterday\b|\btoday\b|\btomorrow\b |
                this\s+(?:week|month|year) |
                last\s+(?:week|month|year) |
                next\s+(?:week|month|year)
            )
        """

        # Combine temporal expressions (events) with entity names for context
        temporal_entities: List[TimelineEvent] = []
        sentences = re.split(r"[.!?]+\s+", text)
        current_pos = extract_position

        # First extract explicit dates and times
        for i, sentence in enumerate(sentences):
            sentence_stripped = sentence.strip()
            if not sentence_stripped:
                current_pos += 1
                continue

            # Find timestamps in the sentence
            date_matches = re.finditer(
                temporal_pattern, sentence, re.VERBOSE | re.IGNORECASE
            )

            for match in date_matches:
                # Find the nearest entity (likely noun/proper name) near the timestamp
                # by looking for capitalized words before and after
                sentence_part = sentence

                # Simple heuristic: find the nearest capitalized word/entity mentioned
                # before this match to associate with the date
                context_before = sentence_part[: match.start()].strip()
                words_before = re.findall(r"\b[A-Z][a-zA-Z]*\b", context_before)[
                    -3:
                ]  # Last 3 capitalized words
                context_entity = (
                    max(words_before, key=len) if words_before else "unknown"
                )

                if not context_entity or len(context_entity) < 2:
                    # If none found in preceding context, try the sentence globally for proper nouns
                    all_cap_words = re.findall(r"\b[A-Z][a-zA-Z.]*\b", sentence_part)
                    if (
                        all_cap_words and len(all_cap_words) > 1
                    ):  # More than just the start of the sentence
                        # Exclude common words at the beginning
                        if all_cap_words[0] in [
                            "The",
                            "A",
                            "An",
                            "On",
                            "In",
                            "At",
                            "By",
                        ]:
                            all_cap_words = all_cap_words[1:]
                        if all_cap_words:
                            context_entity = all_cap_words[0]

                # Create normalized timestamp if possible
                raw_datetime = match.group(0).strip()
                normalized_time = self.normalize_datetime_string(raw_datetime)

                temp_event = TimelineEvent(
                    id=f"temporal_event_{len(temporal_entities)}",
                    text=sentence_stripped,
                    entity=context_entity if context_entity else "unspecified_entity",
                    event_type="timeline_event",
                    timestamp_str=raw_datetime,
                    timestamp_value=normalized_time,
                    position=current_pos + match.start(),
                    extracted_from="source_document",
                )

                temporal_entities.append(temp_event)

            # Move position for next sentence
            current_pos += len(sentence) + 1

        # Sort events by apparent occurrence order for potential sequence analysis
        temporal_entities.sort(key=lambda x: x.position)

        # Attempt to categorize events further based on context in the full text
        self.categorize_timeline_events(temporal_entities, text)

        return temporal_entities

    def normalize_datetime_string(self, datetime_str: str) -> Optional[str]:
        """Normalize various datetime formats to a common format."""
        dt = datetime_str.strip()

        # Try to parse known date/time formats
        import datetime as dt_module

        # Handle month-day-year patterns
        month_patterns = [
            (r"([A-Za-z]{3,})\s+(\d{1,2}),?\s+(\d{4})", "%B %d, %Y"),  # Jan 15, 2023
            (r"([A-Za-z]{3,})\s+(\d{1,2})\s+(\d{4})", "%B %d %Y"),  # Jan 15 2023
        ]

        for pattern, fmt in month_patterns:
            match = re.search(pattern, dt, re.IGNORECASE)
            if match:
                mon, day, yr = match.groups()
                try:
                    import calendar

                    month_nums = [
                        month[:3].lower() for month in calendar.month_abbr[1:]
                    ]
                    mon_lower = mon[:3].lower()
                    month_num_idx = None
                    for idx, abbr in enumerate(month_nums):
                        if mon_lower == abbr or mon_lower.title() == abbr.title():
                            month_num_idx = idx + 1
                            break

                    if month_num_idx:
                        parsed_date = dt_module.datetime(
                            int(yr), month_num_idx, int(day)
                        )
                        return parsed_date.isoformat()
                except ValueError:
                    continue

        # Handle ISO-like formats (YYYY-MM-DD)
        iso_pattern = r"^(\d{4})-(\d{2})-(\d{2})$"
        match_iso = re.match(iso_pattern, dt)
        if match_iso:
            try:
                year, month, day = match_iso.groups()
                parsed_date = dt_module.datetime(int(year), int(month), int(day))
                return parsed_date.isoformat()
            except (ValueError, AttributeError):
                pass

        # Handle just years
        year_pattern = r"^(\d{4})$"
        match_year = re.match(year_pattern, dt)
        if match_year:
            try:
                parsed_year = dt_module.datetime(int(dt), 1, 1)
                return parsed_year.isoformat()
            except ValueError:
                pass

        # Other formats will return as is for now - might expand this later
        return dt  # Return original string if normalization not supported

    def categorize_timeline_events(self, events: List[TimelineEvent], full_text: str):
        """Categorize timeline events into specific types by looking for context clues in the full text."""
        for event in events:
            # Get context around the event timestamp to better understand what type of event it is
            start_idx = max(0, event.position - 100)
            end_idx = min(len(full_text), event.position + len(event.text) + 100)
            context = full_text[start_idx:end_idx]

            # Look for contextual clues to classify the event
            lowercase_context = context.lower()

            if any(
                phrase in lowercase_context
                for phrase in [
                    "born",
                    "birth",
                    "born on",
                    "was born",
                    "birth date",
                    "first appeared",
                ]
            ):
                event.event_type = "birth"
            elif any(
                phrase in lowercase_context
                for phrase in [
                    "died",
                    "death",
                    "died on",
                    "passed away",
                    "deceased",
                    "death date",
                ]
            ):
                event.event_type = "death"
            elif any(
                phrase in lowercase_context
                for phrase in [
                    "meeting",
                    "conference",
                    "gathering",
                    "event occurred",
                    "happened",
                ]
            ):
                event.event_type = "event"
            elif any(
                phrase in lowercase_context
                for phrase in [
                    "started",
                    "began",
                    "initiated",
                    "established",
                    "commenced",
                ]
            ):
                event.event_type = "start"
            elif any(
                phrase in lowercase_context
                for phrase in [
                    "ended",
                    "concluded",
                    "completed",
                    "finished",
                    "terminated",
                ]
            ):
                event.event_type = "end"
            elif any(
                phrase in lowercase_context
                for phrase in [
                    "founded",
                    "established",
                    "created",
                    "set up",
                    "originated",
                ]
            ):
                event.event_type = "formation"
            elif any(
                phrase in lowercase_context
                for phrase in ["married", "wedding", "union", "wed", "marriage"]
            ):
                event.event_type = "marriage"
            else:
                event.event_type = "timeline_event"  # Default type

    def find_evidence_in_source(
        self, claim: HallucinationClaim, source_document: str
    ) -> HallucinationEvidence:
        """Attempt to verify a claim against the source document.

        Args:
            claim: The claim to verify
            source_document: The original source document text

        Returns:
            Evidence indicating whether the claim is supported by source
        """
        claim_text = claim.text.lower().strip()
        source_lower = source_document.lower()

        # Look for direct matches using loose matching
        found_positions = []

        # Try multiple matching strategies:
        # 1. Direct phrase match with small variations
        # 2. Word overlap
        # 3. Semantic similarity indicators

        # Exact substring search for key phrases
        for phrase in [
            claim_text,
            claim_text.replace("the", "").strip(),
            claim_text.replace("a", "").strip(),
            claim_text[: min(len(claim_text), 30)],
        ]:
            if len(phrase) >= 5 and phrase in source_lower:
                start_pos = source_lower.find(phrase)
                end_pos = start_pos + len(phrase)

                similarity_score = len(
                    [word for word in phrase.split() if word.strip()]
                ) / len(
                    set(
                        claim_text.split()
                        + source_document[start_pos:end_pos].lower().split()
                    )
                )

                found_positions.append(
                    (start_pos, end_pos, min(similarity_score * 1.2, 1.0))
                )

        if found_positions:
            # Get the best match
            best_match = max(found_positions, key=lambda x: x[2])
            start_pos, end_pos, score = best_match

            return HallucinationEvidence(
                claim_id=claim.id,
                found_in_source=True,
                source_excerpt=source_document[max(0, start_pos - 100) : end_pos + 100],
                source_position=(start_pos, end_pos),
                similarity_score=score,
            )

        # If no direct match found, compute word overlaps
        claim_words = set(re.findall(r"\w+", claim_text))
        doc_words = set(re.findall(r"\w+", source_lower))

        overlap = len(claim_words.intersection(doc_words))
        total_words = len(claim_words.union(doc_words))

        if overlap > 0:
            similarity_score = overlap / total_words
            # Find the approximate location with the closest word patterns
            best_pos = 0
            best_score: float = 0.0
            window_size = min(
                int(len(source_document) / 20), 500
            )  # Divide source into sections to find best match

            for i in range(0, len(source_document), window_size):
                window = source_document[i : i + window_size].lower()
                win_words = set(re.findall(r"\w+", window))
                win_overlap = len(claim_words.intersection(win_words))
                win_similarity = win_overlap / len(claim_words) if claim_words else 0

                if win_similarity > best_score:
                    best_score = win_similarity
                    best_pos = i

            if best_score > 0.1:  # Threshold for partial match consideration
                return HallucinationEvidence(
                    claim_id=claim.id,
                    found_in_source=True,
                    source_excerpt=source_document[best_pos : best_pos + 200],
                    source_position=(best_pos, best_pos + 200),
                    similarity_score=min(
                        best_score, 0.8
                    ),  # Lower confidence for indirect matches
                )

        # No convincing evidence found
        return HallucinationEvidence(
            claim_id=claim.id,
            found_in_source=False,
            source_excerpt="",
            source_position=(-1, -1),
            similarity_score=0.0,
        )

    def create_enhanced_evidence_trace(
        self, claim: HallucinationClaim, source_document: str
    ) -> EvidenceTrace:
        """Create enhanced evidence trace to map claims back to specific lines in source document.

        Args:
            claim: The claim to trace back to source
            source_document: The original source document text

        Returns:
            EvidenceTrace with detailed mapping and confidence scoring
        """
        claim_text = claim.text.strip()
        source_lines = source_document.splitlines()

        exact_matches = 0
        partial_matches = 0
        matched_chunks = []
        supporting_excerpts = []
        source_locations = []

        # Define search methods
        methods = {
            "exact": self._find_exact_matches,
            "semantic": self._find_semantic_matches,
            "fuzzy": self._find_fuzzy_matches,
        }

        # Search with different methods and collect matches
        for method_name, method_func in methods.items():
            matches = method_func(claim_text, source_document, source_lines)

            for match in matches:
                chunk_info = {
                    "method": method_name,
                    "start_line": match["start_line"],
                    "end_line": match["end_line"],
                    "start_pos": match["start_pos"],
                    "end_pos": match["end_pos"],
                    "excerpt": match["excerpt"],
                    "confidence": match["confidence"],
                }

                matched_chunks.append(chunk_info)

                if method_name == "exact":
                    exact_matches += 1
                else:
                    partial_matches += 1

                supporting_excerpts.append(match["excerpt"])
                source_locations.append((match["start_pos"], match["end_pos"]))

        # Calculate overall confidence score
        if matched_chunks:
            avg_confidence = sum(
                [chunk["confidence"] for chunk in matched_chunks]
            ) / len(matched_chunks)
            confidence_score = avg_confidence
            found_in_source = True
        else:
            confidence_score = 0.0
            found_in_source = False

        return EvidenceTrace(
            claim_id=claim.id,
            found_in_source=found_in_source,
            matched_chunks=matched_chunks,
            confidence_score=confidence_score,
            exact_matches=exact_matches,
            partial_matches=partial_matches,
            supporting_excerpts=supporting_excerpts,
            source_locations=source_locations,
            extraction_method="mixed" if matched_chunks else "none",
        )

    def _find_exact_matches(
        self, claim: str, source_doc: str, source_lines: List[str]
    ) -> List[Dict]:
        """Find exact phrase matches in the source document."""
        matches: List[Dict] = []

        # Convert both to lowercase for case-insensitive matching
        claim_lower = claim.lower()
        doc_lower = source_doc.lower()

        start_pos = 0
        while True:
            found_pos = doc_lower.find(claim_lower, start_pos)
            if found_pos == -1:
                break

            # Find line numbers and excerpt
            doc_prefix = source_doc[:found_pos]
            start_line = doc_prefix.count("\n")
            end_pos = found_pos + len(claim)

            # Get more context around the match
            context_start = max(0, found_pos - 50)
            context_end = min(len(source_doc), end_pos + 50)
            excerpt = source_doc[context_start:context_end]

            matches.append(
                {
                    "start_line": start_line,
                    "end_line": source_doc[:end_pos].count("\n"),
                    "start_pos": found_pos,
                    "end_pos": end_pos,
                    "excerpt": excerpt,
                    "confidence": 1.0,
                }
            )

            start_pos = found_pos + 1  # Move forward to avoid infinite loop

        return matches

    def _find_semantic_matches(
        self, claim: str, source_doc: str, source_lines: List[str]
    ) -> List[Dict]:
        """Find semantically related content in the source document."""
        matches: List[Dict] = []

        # Find paragraphs that share significant vocabulary with the claim
        claim_words = set(re.findall(r"\b\w+\b", claim.lower()))
        if not claim_words:
            return matches

        paragraphs = re.split(r"\n\s*\n", source_doc)  # Split by paragraph

        for para_idx, para in enumerate(paragraphs):
            para_words = set(re.findall(r"\b\w+\b", para.lower()))
            if not para_words:
                continue

            overlap = len(claim_words.intersection(para_words))
            total_unique = len(claim_words.union(para_words))

            if total_unique > 0:
                overlap_ratio = overlap / total_unique
                if overlap_ratio > 0.3:  # At least 30% overlap to consider relevant
                    # Find position of this paragraph in the source document
                    para_pos = source_doc.find(para.strip())

                    matches.append(
                        {
                            "start_line": source_doc[:para_pos].count("\n"),
                            "end_line": source_doc[: para_pos + len(para)].count("\n"),
                            "start_pos": para_pos,
                            "end_pos": para_pos + len(para),
                            "excerpt": para[:200] + ("..." if len(para) > 200 else ""),
                            "confidence": min(
                                overlap_ratio * 1.5, 0.9
                            ),  # Scale and cap confidence
                        }
                    )

        return matches

    def _find_fuzzy_matches(
        self, claim: str, source_doc: str, source_lines: List[str]
    ) -> List[Dict]:
        """Find approximate matches using fuzzy string matching logic."""
        matches: List[Dict] = []

        # Break down source into sentences to enable sentence-level matching
        sentences = re.split(r"[.!?]+", source_doc)
        claim_lower = claim.lower()

        for sent_idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue

            sentence_lower = sentence.lower()

            # Calculate similarity based on overlapping n-grams
            similarity = self._calculate_sentence_similarity(
                claim_lower, sentence_lower
            )

            if similarity > 0.4:  # Consider as a potential match
                pos = source_doc.lower().find(sentence.lower(), 0)

                if pos != -1:
                    context_start = max(0, pos - 30)
                    context_end = min(len(source_doc), pos + len(sentence) + 30)

                    matches.append(
                        {
                            "start_line": source_doc[:pos].count("\n"),
                            "end_line": source_doc[: pos + len(sentence)].count("\n"),
                            "start_pos": pos,
                            "end_pos": pos + len(sentence),
                            "excerpt": source_doc[context_start:context_end],
                            "confidence": similarity
                            * 0.9,  # Slightly reduce for fuzzy matching
                        }
                    )

        return matches

    def _calculate_sentence_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity between two strings based on word overlap."""
        if not s1 or not s2:
            return 0.0

        # Split into words and get overlap
        words1 = set(re.findall(r"\b\w+\b", s1))
        words2 = set(re.findall(r"\b\w+\b", s2))

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        if union == 0:
            return 0.0

        # Use Jaccard similarity (intersection over union)
        return intersection / union

    def verify_timeline_sequences(
        self, source_text_events: List[TimelineEvent], analysis_content: str
    ) -> List[TimelineAnomaly]:
        """Verify timeline sequences and detect chronological inconsistencies."""
        analysis_events = self.extract_temporal_entities(analysis_content)

        # Combine both lists to compare source vs analysis
        all_events = source_text_events + analysis_events
        anomalies: List[TimelineAnomaly] = []

        # Compare each pair of events to see if their temporal relationships differ between source and analysis
        # For this implementation, we'll look for simple chronological issues
        timeline_pairs = []
        for i in range(len(all_events)):
            for j in range(i + 1, len(all_events)):
                event_a, event_b = all_events[i], all_events[j]

                # Check if they refer to the same entity/context
                if (
                    event_a.entity == event_b.entity
                    or event_a.entity in event_b.entity
                    or event_b.entity in event_a.entity
                ):
                    # Extract and try to normalize timestamps
                    timestamp_a = self.parse_and_normalize_datetime(
                        event_a.timestamp_str
                    )
                    timestamp_b = self.parse_and_normalize_datetime(
                        event_b.timestamp_str
                    )

                    if timestamp_a and timestamp_b:
                        timeline_pairs.append(
                            (event_a, event_b, timestamp_a, timestamp_b)
                        )

        # Now check for temporal anomalies
        for event_a, event_b, timestamp_a, timestamp_b in timeline_pairs:
            # Check if analysis flips chronological order compared to source
            if event_a in analysis_events and event_b in analysis_events:
                # Both events from analysis - check for impossible chronology
                source_order = None
                # Check if both events also exist in source, in original order
                source_a_ev = next(
                    (
                        ev
                        for ev in source_text_events
                        if ev.entity == event_a.entity
                        and abs(ev.position - event_a.position) < 500
                    ),
                    None,
                )
                source_b_ev = next(
                    (
                        ev
                        for ev in source_text_events
                        if ev.entity == event_b.entity
                        and abs(ev.position - event_b.position) < 500
                    ),
                    None,
                )

                if (
                    source_a_ev
                    and source_b_ev
                    and source_a_ev.position < source_b_ev.position
                ):
                    source_order = "a_before_b"
                elif (
                    source_a_ev
                    and source_b_ev
                    and source_b_ev.position < source_a_ev.position
                ):
                    source_order = "b_before_a"

                # Check if time values match the positional order
                if isinstance(timestamp_a, datetime) and isinstance(
                    timestamp_b, datetime
                ):
                    if source_a_ev and source_b_ev and source_order:
                        # Check if analysis contradicts source temporal relationship
                        analysis_time_order = (
                            "a_before_b" if timestamp_a <= timestamp_b else "b_before_a"
                        )

                        if source_order != analysis_time_order:
                            # Timeline contradiction detected
                            anomalies.append(
                                TimelineAnomaly(
                                    id=f"chrono_anomaly_{len(anomalies)}",
                                    type="chronological_contradiction",
                                    description=f"Analysis shows {event_a.entity} event happening {analysis_time_order.replace('_', ' ')} when source shows original order",
                                    timestamp_a=event_a.timestamp_str,
                                    timestamp_b=event_b.timestamp_str,
                                    event_a=event_a.text,
                                    event_b=event_b.text,
                                    confidence=0.8,
                                    explanation=f"Temporal contradiction: source has {event_a.entity} event before {event_b.entity}, but analysis places them in reverse chronological order",
                                )
                            )

        return anomalies

    def parse_and_normalize_datetime(self, datetime_str: str) -> Optional[datetime]:
        """Enhanced datetime parsing and normalization."""
        import datetime as dt_module

        dt_str = datetime_str.strip()

        # Handle various formats
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%m-%d-%Y",
            "%Y%m%d",
            "%Y",
        ]

        # Month name formats
        month_year_formats = [
            r"([A-Za-z]{3,})\s+(\d{1,2}),?\s+(\d{4})",  # Jan 15, 2023
            r"(\d{1,2})\s+([A-Za-z]{3,})\s+(\d{4})",  # 15 Jan 2023
            r"([A-Za-z]{3,})\s+(\d{1,2})\s+(\d{4})",  # Jan 15 2023
        ]

        # Try standard date formats first:
        for fmt in formats:
            try:
                parsed_dt = dt_module.datetime.strptime(dt_str, fmt)
                return parsed_dt
            except ValueError:
                continue

        # Then try pattern matching for formats with month names
        for pattern in month_year_formats:
            match = re.search(pattern, dt_str, re.IGNORECASE)
            if match:
                try:
                    if "([A-Za-z]{3,})\\s+(\\d{1,2}),?\\s+(\\d{4})" in pattern or any(
                        p.startswith("(?:[A-Za-z]") for p in [pattern]
                    ):
                        # Pattern is month day year
                        mon, day, yr = match.groups()
                    elif "^(\\d{1,2})\\s+([A-Za-z]{3,})\\s+(\\d{4})$" in pattern:
                        # Pattern is day month year
                        day, mon, yr = match.groups()
                    else:
                        # Default to month-day-year pattern
                        mon, day, yr = match.groups()

                    month_num = self._get_month_number(mon)
                    if month_num:
                        parsed_dt = dt_module.datetime(int(yr), month_num, int(day))
                        return parsed_dt
                except ValueError:
                    continue

        # Just year format
        year_match = re.match(r"^(\d{4})$", dt_str)
        if year_match:
            return dt_module.datetime(int(year_match.group(1)), 1, 1)

        return None

    def _get_month_number(self, month_str: str) -> Optional[int]:
        """Get numeric month from month string."""
        month_names = [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ]
        month_abbrs = [
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
        ]

        month_str_clean = month_str.strip().lower()

        # Check for full month names
        try:
            return month_names.index(month_str_clean) + 1
        except ValueError:
            pass

        # Check for abbreviations (first 3 letters)
        try:
            abbr_3 = month_str_clean[:3]
            if len(abbr_3) == 3:
                return month_abbrs.index(abbr_3) + 1
        except ValueError:
            pass

        return None

    def detect_timeline_anomalies(
        self, source_document: str, analysis_content: str
    ) -> TimelineVerificationResult:
        """Detect timeline anomalies and verify chronological consistency."""
        # Extract temporal entities from both source and analysis
        source_temporal_entities = self.extract_temporal_entities(source_document)
        analysis_temporal_entities = self.extract_temporal_entities(
            analysis_content, len(source_document)
        )

        # Find timeline sequences that could be anomalous
        anomalies = self.verify_timeline_sequences(
            source_temporal_entities, analysis_content
        )

        # Look for additional temporal patterns like impossible age sequences
        # or events that should have certain durations between them
        chronological_issues = 0
        conflicting_timestamps = 0

        # Check for obvious impossibilities
        for event_pair in zip(source_temporal_entities, source_temporal_entities[1:]):
            if event_pair[0].entity == event_pair[1].entity:
                ts1 = self.parse_and_normalize_datetime(event_pair[0].timestamp_str)
                ts2 = self.parse_and_normalize_datetime(event_pair[1].timestamp_str)

                if ts1 and ts2:
                    # Check if the later entity is chronologically out of order
                    if event_pair[0].position < event_pair[1].position and ts2 < ts1:
                        anomalies.append(
                            TimelineAnomaly(
                                id=f"impossible_seq_{len(anomalies)}",
                                type="impossible_sequence",
                                description=f"Chronological issue with {event_pair[0].entity}: time {ts2} before {ts1} despite appearing after in text",
                                timestamp_a=str(ts1.date() if ts1 else "unknown"),
                                timestamp_b=str(ts2.date() if ts2 else "unknown"),
                                event_a=event_pair[0].text,
                                event_b=event_pair[1].text,
                                confidence=0.9,
                                explanation="Event mentions are temporally reversed based on timestamps",
                            )
                        )
                        chronological_issues += 1

        # Calculate timeline score
        total_events = len(source_temporal_entities) + len(analysis_temporal_entities)
        verified_events = len(source_temporal_entities) - len(anomalies)
        timeline_anomalies = len(anomalies)

        if total_events > 0:
            timeline_score = int(
                ((total_events - timeline_anomalies) / total_events) * 100
            )
            quality_assessment = (
                "high"
                if timeline_score >= 90
                else "medium"
                if timeline_score >= 70
                else "low"
            )
        else:
            timeline_score = 100
            quality_assessment = "high"

        return TimelineVerificationResult(
            total_events=total_events,
            verified_events=verified_events,
            timeline_anomalies=timeline_anomalies,
            chronological_issues=chronological_issues,
            conflicting_timestamps=conflicting_timestamps,
            timeline_score=timeline_score,
            detected_anomalies=anomalies,
            temporal_entities=source_temporal_entities + analysis_temporal_entities,
            quality_assessment=quality_assessment,
        )

    def detect_hallucinations(
        self, analysis_content: str, source_document: str
    ) -> HallucinationDetectionResult:
        """Detect hallucinations by verifying claims against source document.

        Args:
            analysis_content: The analysis text to check for hallucinations
            source_document: The original source document text

        Returns:
            Results of the hallucination detection process
        """
        claims = self.extract_claims_from_analysis(analysis_content)

        verified_claims = 0
        hallucinated_claims = 0
        detected_hallucinations = []
        evidence_trace = []
        enhanced_evidence_traces = []

        for claim in claims:
            # Legacy evidence detection
            evidence = self.find_evidence_in_source(claim, source_document)
            evidence_trace.append(evidence)

            # Enhanced evidence tracing
            enhanced_trace = self.create_enhanced_evidence_trace(claim, source_document)
            enhanced_evidence_traces.append(enhanced_trace)

            if evidence.found_in_source:
                verified_claims += 1
            else:
                hallucinated_claims += 1
                # Get detailed evidence info for the hallucination
                detailed_info = {
                    "claim_id": claim.id,
                    "claim_text": claim.text,
                    "position": claim.position,
                    "type": claim.type,
                    "confidence": evidence.similarity_score,
                    "explanation": f"Claim '{claim.text}' not supported in source document",
                }

                # Add enhanced evidence details
                if enhanced_trace.found_in_source:
                    detailed_info["enhanced_support_found"] = True
                    detailed_info["exact_matches"] = enhanced_trace.exact_matches
                    detailed_info["partial_matches"] = enhanced_trace.partial_matches
                    detailed_info["average_confidence"] = (
                        enhanced_trace.confidence_score
                    )
                else:
                    detailed_info["enhanced_support_found"] = False

                detected_hallucinations.append(detailed_info)

        total_claims = len(claims)

        # Calculate overall confidence score and quality assessment
        if total_claims > 0:
            verification_rate = verified_claims / total_claims
            confidence_score = int(verification_rate * 100)

            if verification_rate >= 0.8:
                quality_assessment = "high"
            elif verification_rate >= 0.5:
                quality_assessment = "medium"
            else:
                quality_assessment = "low"
        else:
            confidence_score = 100  # No claims means no hallucinations
            quality_assessment = "high"

        return HallucinationDetectionResult(
            total_claims=total_claims,
            verified_claims=verified_claims,
            hallucinated_claims=hallucinated_claims,
            confidence_score=confidence_score,
            detected_hallucinations=detected_hallucinations,
            quality_assessment=quality_assessment,
            evidence_trace=evidence_trace,
            enhanced_evidence_traces=enhanced_evidence_traces,
        )

    def generate_detailed_audit_report(
        self,
        source_document: str,
        analysis_content: str,
        hallucination_result: HallucinationDetectionResult,
        mode: str = "general",
    ) -> str:
        """Generate a detailed audit report using LLM for comprehensive analysis.

        Args:
            source_document: Original source document text
            analysis_content: Analysis to audit
            hallucination_result: Previous hallucination detection results
            mode: Analysis mode ('general' or 'relationship')

        Returns:
            Detailed audit report from LLM
        """
        try:
            prompt_dir = Path(__file__).parent.parent / "prompts"
            prompt_template = load_audit_prompt(mode, prompt_dir)
            context_window = getattr(self.client, "context_window", 32000)
            full_prompt = build_budgeted_audit_prompt(
                prompt_template=prompt_template,
                source_document=source_document,
                analysis_content=analysis_content,
                hallucination_result=hallucination_result,
                context_window=context_window,
            )
            response: str = self.client.complete(full_prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating detailed audit report: {e}")
            return f"Detailed audit report could not be generated: {e}"

    # ============================================================================
    # Quality Scoring Methods
    # ============================================================================

    def compute_quality_score(
        self,
        analysis_content: str,
        source_document: str,
        hallucination_result: HallucinationDetectionResult,
        timeline_result: TimelineVerificationResult,
        config: Optional[QualityScoringConfig] = None,
    ) -> QualityScore:
        """Compute comprehensive quality score from multiple metrics.

        Args:
            analysis_content: The analysis text to score
            source_document: Original source document for reference
            hallucination_result:Hallucination detection results
            timeline_result: Timeline verification results
            config: Optional quality scoring configuration

        Returns:
            QualityScore with composite score and metric breakdown
        """
        if config is None:
            config = QualityScoringConfig()

        # Compute individual metrics
        metrics = self._compute_all_metrics(
            analysis_content, source_document, hallucination_result, timeline_result
        )

        # Apply weights and compute composite score
        composite_score = self._compute_composite_score(metrics, config)

        # Determine quality assessment
        quality_assessment = self._determine_quality_assessment(composite_score, config)

        # Compute confidence score
        confidence_score = self._compute_confidence_score(
            metrics, hallucination_result, timeline_result, config
        )

        # Generate metrics summary
        metrics_summary = self._generate_metrics_summary(metrics, composite_score)

        return QualityScore(
            composite_score=composite_score,
            metrics=metrics,
            confidence_score=confidence_score,
            quality_assessment=quality_assessment,
            metrics_summary=metrics_summary,
            timestamp=datetime.now().isoformat(),
        )

    def _compute_all_metrics(
        self,
        analysis_content: str,
        source_document: str,
        hallucination_result: HallucinationDetectionResult,
        timeline_result: TimelineVerificationResult,
    ) -> Dict[str, QualityMetric]:
        """Compute all quality metrics.

        Args:
            analysis_content: The analysis text
            source_document: Source document for reference
            hallucination_result:Hallucination detection results
            timeline_result: Timeline verification results

        Returns:
            Dict mapping metric names to QualityMetric objects
        """
        # Coverage: % of source information that appears in analysis
        coverage = self._compute_coverage_metric(analysis_content, source_document)

        # Consistency: Cross-reference consistency score
        consistency = self._compute_consistency_metric(
            analysis_content, hallucination_result
        )

        # Specificity: Granular detail level
        specificity = self._compute_specificity_metric(analysis_content)

        # Clarity: Non-ambiguous claim clarity
        clarity = self._compute_clarity_metric(analysis_content)

        return {
            "coverage": coverage,
            "consistency": consistency,
            "specificity": specificity,
            "clarity": clarity,
        }

    def _compute_coverage_metric(
        self, analysis_content: str, source_document: str
    ) -> QualityMetric:
        """Compute coverage metric - how much source information is covered.

        Coverage = (analysis word overlap with source) / (analysis word count)
        """
        analysis_words = set(self._extract_words(analysis_content))
        source_words = set(self._extract_words(source_document))

        if not analysis_words:
            return QualityMetric(
                name="coverage",
                score=0.0,
                weight=0.0,
                raw_value=0.0,
                description="Coverage of source information in analysis",
                confidence=0.5,
            )

        overlap = len(analysis_words.intersection(source_words))
        coverage_ratio = overlap / len(analysis_words)

        # Normalize to 0-100 scale
        score = min(coverage_ratio * 100, 100.0)

        return QualityMetric(
            name="coverage",
            score=score,
            weight=0.30,
            raw_value=coverage_ratio,
            description=f"Source coverage: {coverage_ratio * 100:.1f}% of analysis words appear in source",
            confidence=0.85 if coverage_ratio > 0.5 else 0.65,
        )

    def _compute_consistency_metric(
        self, analysis_content: str, hallucination_result: HallucinationDetectionResult
    ) -> QualityMetric:
        """Compute consistencymetric - cross-reference and claim verification.

        Based on hallucination detection results and claim verification rate.
        """
        total_claims = hallucination_result.total_claims
        if total_claims == 0:
            return QualityMetric(
                name="consistency",
                score=100.0,
                weight=0.25,
                raw_value=1.0,
                description="No claims found to verify",
                confidence=0.5,
            )

        verified_rate = hallucination_result.verified_claims / total_claims
        consistency_score = verified_rate * 100

        return QualityMetric(
            name="consistency",
            score=consistency_score,
            weight=0.25,
            raw_value=verified_rate,
            description=f"Verified claims: {verified_rate * 100:.1f}% ({hallucination_result.verified_claims}/{total_claims})",
            confidence=0.9 if verified_rate > 0.8 else 0.7,
        )

    def _compute_specificity_metric(self, analysis_content: str) -> QualityMetric:
        """Compute specificity metric - granularity and detail level.

        Higher score = more specific details, less vague generalizations.
        """
        words = self._extract_words(analysis_content)

        if len(words) < 10:
            return QualityMetric(
                name="specificity",
                score=0.0,
                weight=0.25,
                raw_value=0.0,
                description="Insufficient content for specificity analysis",
                confidence=0.4,
            )

        # Count specific indicators
        # Number of sentences
        sentences = re.split(r"[.!?]+", analysis_content)
        sentences = [s for s in sentences if len(s.strip()) > 10]

        # Words with more than 5 characters (more specific)
        long_words = [w for w in words if len(w) > 5]
        long_word_ratio = len(long_words) / len(words) if words else 0

        # Number of unique entities (potential specificity indicator)
        # Find capitalized words that might be named entities
        sentences_text = re.split(r"[.!?]+", analysis_content)
        entities = set()
        for sentence in sentences_text:
            # Find capitalized words (potential named entities)
            caps = re.findall(r"\b[A-Z][a-z]+\b", sentence)
            # Filter out very common words
            common_words = {
                "The",
                "This",
                "That",
                "With",
                "From",
                "For",
                "There",
                "Here",
            }
            # Convert to set for subtraction, then back to list for update
            filtered_caps = [c for c in caps if c not in common_words]
            entities.update(filtered_caps)

        entity_ratio = len(entities) / max(len(sentences), 1)

        # Specificity formula: combination of factors
        specificity_score = (
            min(long_word_ratio * 100, 60)
            + min(entity_ratio * 10, 20)
            + min(len(sentences) * 2, 20)
        )

        return QualityMetric(
            name="specificity",
            score=specificity_score,
            weight=0.25,
            raw_value=long_word_ratio,
            description=f"Specificity indicators: {len(entities)} entities, {len(sentences)} sentences",
            confidence=0.75,
        )

    def _compute_clarity_metric(self, analysis_content: str) -> QualityMetric:
        """Compute clarity metric - non-ambiguous claim clarity.

        Based on sentence structure, word choice, and ambiguity indicators.
        """
        sentences = re.split(r"[.!?]+", analysis_content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if not sentences:
            return QualityMetric(
                name="clarity",
                score=0.0,
                weight=0.20,
                raw_value=0.0,
                description="No sentences to analyze",
                confidence=0.3,
            )

        clarity_indicators = 0
        total_checks = len(sentences) * 3  # 3 checks per sentence

        for sentence in sentences:
            # Check 1: Sentence length (not too long or too short)
            words = sentence.split()
            if 5 <= len(words) <= 35:
                clarity_indicators += 1

            # Check 2: No excessive modals/uncertainty indicators
            uncertainty_words = {
                "maybe",
                "perhaps",
                "possibly",
                "probably",
                "might",
                "could",
                "seems",
                "appears",
            }
            sentence_lower = sentence.lower()
            has_uncertainty = any(w in sentence_lower for w in uncertainty_words)
            if not has_uncertainty:
                clarity_indicators += 1

            # Check 3: Contains clear subject-verb structure
            # Simple heuristic: has a verb-like word
            verbs = {"is", "are", "was", "were", "be", "being", "been"}
            has_verb = any(v in sentence_lower for v in verbs)
            if has_verb:
                clarity_indicators += 1

        clarity_ratio = clarity_indicators / total_checks if total_checks > 0 else 0
        clarity_score = clarity_ratio * 100

        return QualityMetric(
            name="clarity",
            score=clarity_score,
            weight=0.20,
            raw_value=clarity_ratio,
            description=f"Clarity indicators: {clarity_indicators}/{total_checks} checks passed",
            confidence=0.7,
        )

    def _compute_composite_score(
        self, metrics: Dict[str, QualityMetric], config: QualityScoringConfig
    ) -> float:
        """Compute weighted composite quality score.

        Args:
            metrics: Dict of computed QualityMetric objects
            config: Quality scoring configuration with weights

        Returns:
            Composite score (0-100)
        """
        # Apply metric weights
        total_weighted_score = 0.0
        total_weight = 0.0

        for metric_name, metric in metrics.items():
            weight = config.metric_weights.get(metric_name, 0.25)
            total_weighted_score += metric.score * weight
            total_weight += weight

        # Normalize if weights don't sum to 1
        if total_weight > 0:
            composite_score = total_weighted_score / total_weight
        else:
            composite_score = 0.0

        return min(max(composite_score, 0.0), 100.0)

    def _determine_quality_assessment(
        self, composite_score: float, config: QualityScoringConfig
    ) -> str:
        """Determine overall quality assessment level.

        Args:
            composite_score: Computed composite score
            config: Quality scoring configuration with thresholds

        Returns:
            Assessment level: 'excellent', 'good', 'fair', 'poor'
        """
        thresholds = config.assessment_thresholds

        if composite_score >= thresholds["excellent"]:
            return "excellent"
        elif composite_score >= thresholds["good"]:
            return "good"
        elif composite_score >= thresholds["fair"]:
            return "fair"
        else:
            return "poor"

    def _compute_confidence_score(
        self,
        metrics: Dict[str, QualityMetric],
        hallucination_result: HallucinationDetectionResult,
        timeline_result: TimelineVerificationResult,
        config: QualityScoringConfig,
    ) -> float:
        """Compute confidence in the quality assessment itself.

        Higher confidence when we have more data and fewer uncertainties.

        Args:
            metrics: Computed quality metrics
            hallucination_result:Hallucination detection results
            timeline_result: Timeline verification results
            config: Quality scoring configuration

        Returns:
            Confidence score (0.0-1.0)
        """
        if not config.enable_confidence_scoring:
            return 0.5  # Default moderate confidence

        # Base confidence from metric confidence scores
        metric_confidence_scores = [m.confidence for m in metrics.values()]
        avg_metric_confidence = (
            sum(metric_confidence_scores) / len(metric_confidence_scores)
            if metric_confidence_scores
            else 0.0
        )

        # Additional confidence from verification data
        verification_confidence = 0.0
        if hallucination_result.total_claims > 0:
            # Higher confidence with more verified claims
            verification_rate = (
                hallucination_result.verified_claims / hallucination_result.total_claims
            )
            verification_confidence = 0.5 + (verification_rate * 0.3)

        # Timeline verification adds confidence
        timeline_confidence = 0.0
        if timeline_result.total_events > 0:
            verified_ratio = (
                timeline_result.verified_events / timeline_result.total_events
            )
            timeline_confidence = verified_ratio * 0.2

        # Weighted combination
        confidence_score = (
            avg_metric_confidence * 0.5
            + verification_confidence * 0.3
            + timeline_confidence * 0.2
        )

        return min(max(confidence_score, 0.0), 1.0)

    def _generate_metrics_summary(
        self, metrics: Dict[str, QualityMetric], composite_score: float
    ) -> str:
        """Generate human-readable summary of all metrics.

        Args:
            metrics: Dict of computed quality metrics
            composite_score: Computed composite score

        Returns:
            Formatted summary string
        """
        summary = []
        summary.append(f"Overall Quality Score: {composite_score:.1f}/100")
        summary.append("")

        for name, metric in sorted(metrics.items()):
            score_str = f"{metric.score:.1f}" if metric.score >= 0 else "N/A"
            confidence_str = f"{metric.confidence:.2f}"
            summary.append(
                f"- {name.capitalize()}: {score_str}/100 (Confidence: {confidence_str})"
            )
            if metric.description:
                summary.append(f"  {metric.description}")

        return "\n".join(summary)

    def _extract_words(self, text: str) -> List[str]:
        """Extract lowercase words from text, filtering numbers."""
        # Extract all words
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        # Filter out common stop words for analysis
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "used",
        }
        return [w for w in words if w not in stop_words and len(w) > 2]

    def _build_missing_analysis_result(
        self, mode: str, error_message: str
    ) -> dict[str, Any]:
        """Return a deterministic failure payload when audit prerequisites are missing."""
        timestamp = datetime.now().isoformat()
        return {
            "status": "failed",
            "stage": "audit",
            "mode": mode,
            "timestamp": timestamp,
            "message": error_message,
            "checked_items": 0,
            "issues_found": 0,
            "confidence_score": None,
            "audited_files": [],
            "recommendations": ["Run the final analysis stage before invoking audit."],
            "errors": [error_message],
            "hallucination_detection": {
                "total_claims": 0,
                "verified_claims": 0,
                "hallucinated_claims": 0,
                "confidence_score": 0,
                "quality_assessment": "unavailable",
            },
            "timeline_verification": {
                "total_events": 0,
                "verified_events": 0,
                "timeline_anomalies": 0,
                "chronological_issues": 0,
                "conflicting_timestamps": 0,
                "timeline_score": 0,
                "quality_assessment": "unavailable",
            },
            "detected_hallucinations": [],
            "detected_timeline_anomalies": [],
            "source_temporal_entities": [],
            "evidence_trace": [],
            "enhanced_evidence_trace": [],
            "source_document_available": False,
            "detailed_audit_report": "",
            "quality_scoring": {
                "composite_score": 0.0,
                "confidence_score": 0.0,
                "quality_assessment": "unavailable",
                "metrics": {},
                "metrics_summary": error_message,
                "timestamp": timestamp,
            },
        }

    def _create_claim(self, claim_text: str) -> HallucinationClaim:
        """Create a lightweight claim object for legacy compatibility helpers."""
        return HallucinationClaim(
            id="legacy_claim",
            text=claim_text,
            position=0,
            type=self.classify_claim_type(claim_text),
            extracted_from="analysis",
        )

    def _confidence_weight(self, confidence: str | None) -> float:
        """Map qualitative confidence labels to numeric weights."""
        if not confidence:
            return 0.5

        return {
            "high": 1.0,
            "medium": 0.7,
            "low": 0.4,
        }.get(str(confidence).lower(), 0.5)

    def check_claim_validity(
        self, claim: str, manifest: Manifest, mode: str = "general"
    ) -> HallucinationDetectionResult:
        """Legacy single-claim verification API used by older tests."""
        source_document = ""
        try:
            source_document = read_file(manifest.input_path)
        except Exception as exc:
            logger.warning(
                "Unable to read source document for claim validation: %s", exc
            )

        prompt = (
            f"Mode: {mode}\n"
            "Determine whether the following claim is supported by the source.\n"
            f"Claim: {claim}\n\n"
            f"Source:\n{source_document}"
        )

        if hasattr(self.client, "complete_json"):
            try:
                response: dict[str, Any] = self.client.complete_json(prompt)
                supported = bool(response.get("supported"))
                return HallucinationDetectionResult(
                    claim=claim,
                    is_hallucinated=not supported,
                    confidence=response.get("confidence", "medium"),
                    evidence=[
                        {
                            "source": manifest.input_path,
                            "location": response.get("evidence_location", "unknown"),
                            "quote": response.get("quote", ""),
                        }
                    ],
                    explanation=response.get(
                        "explanation",
                        "Claim support could not be determined from source.",
                    ),
                )
            except Exception as exc:
                logger.info(
                    "Falling back to heuristic claim validation for audit: %s", exc
                )

        evidence = self.find_evidence_in_source(
            self._create_claim(claim), source_document
        )
        supported = evidence.found_in_source and evidence.similarity_score >= 0.2
        similarity = evidence.similarity_score
        confidence = (
            "high" if similarity >= 0.75 else "medium" if similarity >= 0.4 else "low"
        )
        explanation = (
            "The claim is supported by the source document."
            if supported
            else "The claim is not found in the source document."
        )

        return HallucinationDetectionResult(
            claim=claim,
            is_hallucinated=not supported,
            confidence=confidence,
            evidence=[
                {
                    "source": manifest.input_path,
                    "location": evidence.source_position,
                    "quote": evidence.source_excerpt,
                }
            ],
            explanation=explanation,
        )

    def calculate_accuracy_score(
        self, results: list[HallucinationDetectionResult]
    ) -> float:
        """Legacy quality helper based on per-claim verification results."""
        if not results:
            return 100.0

        weighted_scores = []
        for result in results:
            if getattr(result, "is_hallucinated", False):
                weighted_scores.append(0.0)
            else:
                weighted_scores.append(
                    self._confidence_weight(getattr(result, "confidence", None))
                )

        return sum(weighted_scores) / len(weighted_scores) * 100.0

    def calculate_consistency_score(
        self, final_analysis: FinalAnalysis, manifest: Manifest
    ) -> float:
        """Legacy consistency heuristic used by compatibility tests."""
        analysis_lower = final_analysis.final_result.lower()
        contradiction_markers = [
            "however",
            "but",
            "on the other hand",
            "although",
            "despite",
            "yet",
        ]
        contradictions = sum(
            analysis_lower.count(marker) for marker in contradiction_markers
        )

        score = 100.0 - min(contradictions * 15.0, 75.0)
        return max(score, 15.0)

    def calculate_coverage_score(
        self, final_analysis: FinalAnalysis, manifest: Manifest
    ) -> float:
        """Legacy coverage heuristic for older audit tests."""
        content_tokens = max(estimate_tokens(final_analysis.final_result), 1)
        total_parts = max(manifest.total_parts or 1, 1)
        target_tokens = total_parts * 40
        ratio = min(content_tokens / target_tokens, 1.0)
        return float(max(min(ratio * 100.0, 100.0), 0.0))

    def calculate_overall_score(
        self, accuracy_score: float, consistency_score: float, coverage_score: float
    ) -> float:
        """Legacy weighted overall score."""
        return accuracy_score * 0.60 + consistency_score * 0.25 + coverage_score * 0.15

    def get_quality_description(self, score: float) -> str:
        """Map a numeric score to a human-readable legacy quality band."""
        if score >= 90:
            return "Excellent quality"
        if score >= 75:
            return "Good quality"
        if score >= 60:
            return "Moderate quality"
        if score >= 40:
            return "Low quality"
        return "Poor quality"

    def perform_complete_audit(
        self,
        final_analysis: FinalAnalysis,
        manifest: Manifest,
        mode: str = "general",
        config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Legacy end-to-end audit API retained for compatibility tests."""
        claims = self.extract_claims_from_analysis(final_analysis.final_result)
        hallucination_results = [
            self.check_claim_validity(claim.text, manifest, mode) for claim in claims
        ]
        accuracy_score = self.calculate_accuracy_score(hallucination_results)
        consistency_score = self.calculate_consistency_score(final_analysis, manifest)
        coverage_score = self.calculate_coverage_score(final_analysis, manifest)
        overall_score = self.calculate_overall_score(
            accuracy_score, consistency_score, coverage_score
        )

        return {
            "hallucination_results": hallucination_results,
            "accuracy_score": accuracy_score,
            "consistency_score": consistency_score,
            "coverage_score": coverage_score,
            "overall_score": overall_score,
            "quality_description": self.get_quality_description(overall_score),
            "total_claims": len(hallucination_results),
            "detected_hallucinations": sum(
                1
                for result in hallucination_results
                if getattr(result, "is_hallucinated", False)
            ),
        }

    def run(
        self,
        analysis_objects: Optional[FinalAnalysis],
        config: dict[str, Any],
        manifest: Manifest,
        mode: str = "general",
    ) -> dict[str, Any]:
        """Run the comprehensive audit stage.
        Args:
            analysis_objects: FinalAnalysis object from previous stage
            config: Configuration dictionary
            manifest: Manifest object to update with audit status
            mode: Analysis mode ('general' or 'relationship')
        Returns:
            Dictionary with audit results and status
        """
        logger.info(f"Starting audit stage using mode: {mode}")

        if mode not in ("general", "relationship"):
            raise ValueError(
                f"Invalid mode: '{mode}'. Must be 'general' or 'relationship'."
            )

        if analysis_objects is None:
            missing_analysis_error = (
                "Audit requires final analysis output but none was provided."
            )
            logger.error(missing_analysis_error)
            self.manifest_manager.update_stage(
                manifest,
                "audit",
                "failed",
                error=missing_analysis_error,
            )
            return self._build_missing_analysis_result(mode, missing_analysis_error)

        # Read the original source document from the manifest
        source_document = ""
        try:
            if hasattr(manifest, "input_path"):
                source_document = read_file(manifest.input_path)
            elif config.get("input_path"):
                source_document = read_file(config["input_path"])
        except Exception as e:
            logger.error(f"Could not read source document: {e}")

        if not source_document:
            logger.error("Source document unavailable for audit")

        final_analysis_content = analysis_objects.final_result

        # Perform hallucination detection
        hallucination_result = self.detect_hallucinations(
            final_analysis_content, source_document or ""
        )

        # Perform timeline verification
        timeline_result = self.detect_timeline_anomalies(
            source_document or "", final_analysis_content
        )

        # Generate detailed audit report with LLM
        detailed_report = self.generate_detailed_audit_report(
            source_document or "", final_analysis_content, hallucination_result, mode
        )

        # Compute quality score
        quality_score = self.compute_quality_score(
            final_analysis_content,
            source_document or "",
            hallucination_result,
            timeline_result,
        )

        # Determine audit status based on combined results
        total_issues = (
            hallucination_result.hallucinated_claims
            + timeline_result.timeline_anomalies
        )

        if total_issues == 0:
            stage_status = "successful"
        elif total_issues <= 3:
            stage_status = "successful_with_warnings"
        else:
            stage_status = "failed"

        # Update manifest
        audit_stats = {
            "total_claims": hallucination_result.total_claims,
            "verified_claims": hallucination_result.verified_claims,
            "hallucinated_claims": hallucination_result.hallucinated_claims,
            "confidence_score": hallucination_result.confidence_score,
            "quality_assessment": hallucination_result.quality_assessment,
            "timeline_anomalies": timeline_result.timeline_anomalies,
            "timeline_score": timeline_result.timeline_score,
            "source_document_available": bool(source_document),
            "evidence_trace_count": len(hallucination_result.evidence_trace),
            "enhanced_evidence_trace_count": len(
                hallucination_result.enhanced_evidence_traces
            ),
            "quality_scoring": {
                "composite_score": quality_score.composite_score,
                "confidence_score": quality_score.confidence_score,
                "quality_assessment": quality_score.quality_assessment,
                "metrics": {
                    name: {
                        "score": metric.score,
                        "weight": metric.weight,
                        "raw_value": metric.raw_value,
                        "description": metric.description,
                        "confidence": metric.confidence,
                    }
                    for name, metric in quality_score.metrics.items()
                },
            },
        }

        self.manifest_manager.update_stage(
            manifest, "audit", stage_status, stats=audit_stats
        )

        # Prepare results
        audit_results = {
            "status": stage_status,
            "stage": "audit",
            "mode": mode,
            "timestamp": datetime.now().isoformat(),
            "hallucination_detection": {
                "total_claims": hallucination_result.total_claims,
                "verified_claims": hallucination_result.verified_claims,
                "hallucinated_claims": hallucination_result.hallucinated_claims,
                "confidence_score": hallucination_result.confidence_score,
                "quality_assessment": hallucination_result.quality_assessment,
            },
            "timeline_verification": {
                "total_events": timeline_result.total_events,
                "verified_events": timeline_result.verified_events,
                "timeline_anomalies": timeline_result.timeline_anomalies,
                "chronological_issues": timeline_result.chronological_issues,
                "conflicting_timestamps": timeline_result.conflicting_timestamps,
                "timeline_score": timeline_result.timeline_score,
                "quality_assessment": timeline_result.quality_assessment,
            },
            "detected_hallucinations": hallucination_result.detected_hallucinations,
            "detected_timeline_anomalies": [
                {
                    "id": anomaly.id,
                    "type": anomaly.type,
                    "description": anomaly.description,
                    "confidence": anomaly.confidence,
                    "explanation": anomaly.explanation,
                    "timestamp_a": anomaly.timestamp_a,
                    "timestamp_b": anomaly.timestamp_b,
                    "event_a": anomaly.event_a,
                    "event_b": anomaly.event_b,
                }
                for anomaly in timeline_result.detected_anomalies
            ],
            "source_temporal_entities": [
                {
                    "id": entity.id,
                    "text": entity.text,
                    "entity": entity.entity,
                    "event_type": entity.event_type,
                    "timestamp_str": entity.timestamp_str,
                    "timestamp_value": entity.timestamp_value,
                    "position": entity.position,
                    "extracted_from": entity.extracted_from,
                }
                for entity in timeline_result.temporal_entities
            ],
            "evidence_trace": [
                {
                    "claim_id": ev.claim_id,
                    "found_in_source": ev.found_in_source,
                    "similarity_score": ev.similarity_score,
                    "source_position": ev.source_position,
                }
                for ev in hallucination_result.evidence_trace
            ],
            "enhanced_evidence_trace": [
                {
                    "claim_id": trace.claim_id,
                    "found_in_source": trace.found_in_source,
                    "confidence_score": trace.confidence_score,
                    "exact_matches": trace.exact_matches,
                    "partial_matches": trace.partial_matches,
                    "extraction_method": trace.extraction_method,
                    "total_match_chunks": len(trace.matched_chunks),
                    "supporting_excerpts_preview": [
                        excerpt[:100] + "..." if len(excerpt) > 100 else excerpt
                        for excerpt in trace.supporting_excerpts[:3]
                    ],  # First 3 excerpts, truncated
                }
                for trace in hallucination_result.enhanced_evidence_traces
            ],
            "source_document_available": bool(source_document),
            "detailed_audit_report": detailed_report,
            "quality_scoring": {
                "composite_score": quality_score.composite_score,
                "confidence_score": quality_score.confidence_score,
                "quality_assessment": quality_score.quality_assessment,
                "metrics": {
                    name: {
                        "score": metric.score,
                        "weight": metric.weight,
                        "raw_value": metric.raw_value,
                        "description": metric.description,
                        "confidence": metric.confidence,
                    }
                    for name, metric in quality_score.metrics.items()
                },
                "metrics_summary": quality_score.metrics_summary,
                "timestamp": quality_score.timestamp,
            },
        }

        # Log important metrics
        if hallucination_result.hallucinated_claims > 0:
            msg = f"Audit detected {hallucination_result.hallucinated_claims} potential hallucinations. Quality assessment: {hallucination_result.quality_assessment}"
            logger.warning(msg)
        else:
            msg = f"No hallucinations detected. Confidence score: {hallucination_result.confidence_score}"
            logger.info(msg)

        if timeline_result.timeline_anomalies > 0:
            detail_msg = f"Audit detected {timeline_result.timeline_anomalies} timeline anomalies. Quality assessment: {timeline_result.quality_assessment}"
            logger.warning(detail_msg)
        else:
            detail_msg = f"No timeline anomalies detected. Timeline score: {timeline_result.timeline_score}"
            logger.info(detail_msg)

        # Log quality scoring summary
        logger.info(
            f"Quality Score: {quality_score.composite_score:.1f}/100 "
            f"({quality_score.quality_assessment}) - "
            f"Confidence: {quality_score.confidence_score:.2f}"
        )

        return audit_results
