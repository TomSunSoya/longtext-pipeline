"""
Audit stage skeleton for longtext pipeline (v2 placeholder).

This module provides a placeholder AuditStage implementation for the MVP.
Full audit functionality (hallucination checking, timeline verification,
entity consistency) is deferred to v2.

The audit stage is marked as 'skipped' by default in the manifest,
and this placeholder simply issues a warning and returns a placeholder
result to maintain the 5-stage pipeline pattern.
"""

import logging
from typing import Any, Dict

from ..models import Manifest
from ..manifest import ManifestManager


logger = logging.getLogger(__name__)


class AuditStage:
    """Placeholder audit stage for MVP.

    The audit stage is deferred to v2. This implementation:
    1. Issues a warning that audit is experimental in MVP
    2. Returns a placeholder result with status='skipped'
    3. Updates the manifest to track the skipped status
    4. Does NOT perform any actual audit checks

    In v2, this will be expanded to include:
    - Hallucination detection
    - Timeline alignment verification
    - Entity consistency checking
    - Quality scoring and reporting
    """

    def __init__(self, manifest_manager: ManifestManager = None):
        """Initialize the audit stage.

        Args:
            manifest_manager: Optional existing manifest manager
        """
        self.manifest_manager = manifest_manager or ManifestManager()

    def run(
        self,
        analysis_objects: Any,
        config: Dict,
        manifest: Manifest,
        mode: str = "general",
    ) -> dict:
        """Run the audit stage (placeholder).

        Args:
            analysis_objects: Analysis objects from previous stage (deferred for v2)
            config: Configuration dictionary (currently unused in placeholder)
            manifest: Manifest object to update with audit status
            mode: Analysis mode for audit (currently unused in placeholder)

        Returns:
            dict with status='skipped' and placeholder data

        Note:
            This is a MVP placeholder. No actual audit logic is performed.
            Full audit functionality is deferred to v2.
        """
        # Issue warning that audit is experimental
        logger.warning("Audit functionality is experimental in MVP")
        logger.warning("Skipping audit; full implementation is deferred to v2")
        logger.warning("Analysis quality checks were not performed")

        # Update manifest to indicate skipped status
        self.manifest_manager.update_stage(
            manifest,
            "audit",
            "skipped",
            error="Audit functionality deferred to v2 - placeholder only",
        )

        # Return placeholder result
        placeholder_result = {
            "status": "skipped",
            "stage": "audit",
            "mode": mode,
            "message": "Audit functionality is experimental in MVP - no checks performed",
            "checked_items": 0,
            "issues_found": 0,
            "confidence_score": None,
            "audited_files": [],
            "recommendations": [],
        }

        return placeholder_result
