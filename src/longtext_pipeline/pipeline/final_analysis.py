"""
Final analysis stage implementation for the longtext pipeline.

This module provides the FinalAnalysisStage class for generating comprehensive
final analysis from all stage summaries using LLM-based analysis.
Supports both General and Relationship modes via different prompt templates.
Unlike other stages, implements error handling that captures partial results
when LLM fails (single-operation failure mode rather than Continue-with-Partial).

Supports multi-perspective analysis with parallel specialist agents (Phase 4).
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..errors import LLMError
from ..llm.factory import get_llm_client
from ..manifest import ManifestManager
from ..models import FinalAnalysis, Manifest, StageSummary
from ..utils.io import read_file, write_file
from ..utils.token_estimator import estimate_tokens


# Specialist agent definitions: type key -> (agent_type for config, prompt suffix)
_SPECIALIST_DEFINITIONS = {
    "topic_analyst": (
        "Analyze this as a Topic/Trend Analyst. Focus on identifying major themes, "
        "trends, patterns, recurring topics, and thematic connections across all stages. "
        "Extract key concepts and their evolution throughout the document."
    ),
    "entity_analyst": (
        "Analyze this as an Entity Relationship Analyst. Focus on identifying entities "
        "(people, places, organizations, concepts), their attributes, and relationships "
        "between them. Map key connections and dependencies across stages."
    ),
    "sentiment_analyst": (
        "Analyze this as a Sentiment Analyst. Focus on detecting emotions, mood, tone, "
        "and sentiment dynamics. Identify positive/negative sentiments, attitude shifts, "
        "and emotional undercurrents across stages."
    ),
    "timeline_analyst": (
        "Analyze this as a Timeline Analyst. Focus on extracting chronological information, "
        "sequence of events, timeline relationships, and temporal progression. Create a "
        "coherent chronology from the distributed information across stages."
    ),
}

_SPECIALIST_OUTPUT_FORMAT = (
    "Return a Markdown report with exactly these H2 sections:\n"
    "## Executive Summary\n"
    "## Key Findings\n"
    "## Cross-Stage Evidence\n"
    "## Risks And Unknowns\n\n"
    "Ground every claim in the Combined Context. If evidence is missing or weak, "
    "say so explicitly instead of inferring."
)

logger = logging.getLogger(__name__)


def _format_analyst_label(analyst_type: str) -> str:
    """Convert an analyst type key into a human-readable label."""
    return analyst_type.replace("_", " ").title()


class FinalAnalysisStage:
    """Generates comprehensive final analysis from all stage summaries.

    The final analysis stage synthesizes information across all stage summaries
    to produce a holistic analysis. It supports both General and Relationship
    modes (experimental) via different prompt templates. Unlike earlier stages
    that process items independently, this is a single-operation stage where
    failure results in partial result capture rather than Continue-with-Partial.
    """

    def __init__(self, manifest_manager: Optional[ManifestManager] = None):
        """Initialize the final analysis stage.

        Args:
            manifest_manager: Optional existing manifest manager
        """
        self.manifest_manager = manifest_manager or ManifestManager()

    def _load_prompt_template(self, mode: str) -> str:
        """Load prompt template from file based on mode."""
        if mode == "relationship":
            prompt_file = "final_relationship.txt"
        else:
            prompt_file = "final_general.txt"

        prompts_dir = Path(__file__).parent.parent / "prompts"
        prompt_path = prompts_dir / prompt_file

        try:
            return read_file(str(prompt_path))
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Prompt template not found: {prompt_path}. "
                f"Ensure {prompt_file} exists in the prompts directory."
            )

    def _validate_stage_summaries(self, stage_summaries: List[StageSummary]) -> None:
        """Validate that stage summaries exist and are non-empty."""
        if not stage_summaries:
            raise ValueError(
                "Cannot perform final analysis: no stage summaries provided. "
                "Ensure stage synthesis stage completed successfully."
            )

    def _build_combined_context(self, stage_summaries: List[StageSummary]) -> str:
        """Combine all stage summaries into single context string."""
        combined = ""
        for i, stage_summary in enumerate(stage_summaries):
            combined += f"\n\n--- Stage {i} (Index {stage_summary.stage_index}) ---\n"
            combined += stage_summary.synthesis
        return combined

    def _get_selected_specialists(self, config: Dict) -> List[str]:
        """Return the ordered list of specialist agents selected for this run."""
        all_specialists = list(_SPECIALIST_DEFINITIONS.keys())
        requested_count = config.get("pipeline", {}).get(
            "specialist_count", len(all_specialists)
        )

        if isinstance(requested_count, bool) or not isinstance(requested_count, int):
            raise ValueError("pipeline.specialist_count must be an integer.")

        if requested_count < 1 or requested_count > len(all_specialists):
            raise ValueError(
                f"pipeline.specialist_count must be between 1 and {len(all_specialists)}."
            )

        return all_specialists[:requested_count]

    def _get_specialist_concurrency_limit(
        self, config: Dict, selected_specialists: List[str]
    ) -> int:
        """Return the max concurrency for specialist execution."""
        requested_limit = config.get("pipeline", {}).get(
            "max_workers", len(selected_specialists)
        )

        if requested_limit is None:
            return len(selected_specialists)

        if isinstance(requested_limit, bool) or not isinstance(requested_limit, int):
            raise ValueError("pipeline.max_workers must be an integer.")

        if requested_limit < 1:
            raise ValueError("pipeline.max_workers must be at least 1.")

        return min(requested_limit, len(selected_specialists))

    def _build_specialist_prompt(
        self,
        prompt_template: str,
        prompt_suffix: str,
        stage_summaries: List[StageSummary],
    ) -> str:
        """Build the prompt for a specialist analysis run."""
        combined_context = self._build_combined_context(stage_summaries)
        return (
            prompt_template
            + f"\n\nSpecialist Role:\n{prompt_suffix}\n\n"
            + f"Output Format:\n{_SPECIALIST_OUTPUT_FORMAT}\n\n"
            + "Combined Context:\n"
            + combined_context
        )

    async def _generate_specialist_analysis(
        self,
        analyst_type: str,
        stage_summaries: List[StageSummary],
        prompt_template: str,
        client,
        model: str,
    ) -> Dict[str, Any]:
        """Generate a specialist analysis using LLM asynchronously.

        Args:
            analyst_type: Key into _SPECIALIST_DEFINITIONS
            stage_summaries: List of StageSummary objects to synthesize
            prompt_template: Loaded prompt template
            client: LLM client instance
            model: Model name for metadata

        Returns:
            Dictionary with analysis results (always returns, never raises)
        """
        prompt_suffix = _SPECIALIST_DEFINITIONS[analyst_type]
        try:
            full_prompt = self._build_specialist_prompt(
                prompt_template, prompt_suffix, stage_summaries
            )
            response = await client.acomplete(full_prompt)
            return {
                "analyst_type": analyst_type,
                "model_used": model,
                "analysis": response,
                "timestamp": datetime.now().isoformat(),
                "status": "completed",
            }
        except Exception as e:
            logger.exception("[%s] Specialist analysis failed", analyst_type)
            return {
                "analyst_type": analyst_type,
                "model_used": model,
                "analysis": f"[Error: {e}]",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
            }

    async def _aggregate_with_meta_agent(
        self,
        specialist_results: List[Dict[str, Any]],
        client,
        model: str,
    ) -> str:
        """Aggregate specialist results using a meta-agent LLM call.

        Args:
            specialist_results: List of completed specialist analysis dicts
            client: LLM client instance for the meta-agent
            model: Model name

        Returns:
            Synthesized analysis text
        """
        specialist_context = ""
        unavailable_perspectives = []
        for result in specialist_results:
            analyst_type = result.get("analyst_type", "analyst")
            status = result.get("status", "unknown")
            analysis = result.get("analysis", "")
            if status == "completed":
                section_title = _format_analyst_label(analyst_type)
                specialist_context += f"### {section_title} Perspective\n{analysis}\n\n"
            else:
                unavailable_reason = result.get("error") or analysis or status
                unavailable_perspectives.append(
                    f"- {_format_analyst_label(analyst_type)}: unavailable ({unavailable_reason})"
                )

        missing_context = ""
        if unavailable_perspectives:
            missing_context = (
                "Requested perspectives with no usable output:\n"
                + "\n".join(unavailable_perspectives)
                + "\n\nTreat these as missing evidence. Do not invent findings that "
                "depend on these missing perspectives.\n\n"
            )

        meta_prompt = (
            "Synthesize the following specialist analyses into a cohesive final analysis. "
            "The goal is to present a multi-faceted view that incorporates thematic insights, "
            "entities and their relationships, sentiment patterns, and temporal progression. "
            "If some requested perspectives are unavailable, explicitly acknowledge those "
            "gaps and stay within the available evidence.\n\n"
            f"{missing_context}"
            "Available specialist analyses:\n\n"
            f"{specialist_context}\n"
            "Create an integrated holistic report that:\n"
            "1. Weaves insights from all perspectives together naturally\n"
            "2. Identifies connections and contradictions between perspectives\n"
            "3. Maintains professional analytical tone\n"
            "4. Focuses on actionable insights\n"
            "5. Is organized in clearly separated sections with cross-references"
        )

        try:
            response = await client.acomplete(meta_prompt)
            return f"# Multi-Perspective Integrated Analysis\n\n{response}"
        except Exception as e:
            logger.warning(
                "[MetaAgent] LLM aggregation failed (%s), falling back to concatenation",
                e,
                exc_info=True,
            )
            # Fallback: concatenate specialist results without LLM
            fallback = "# Multi-Perspective Integrated Analysis\n\n"
            if unavailable_perspectives:
                fallback += "## Missing Perspectives\n"
                fallback += "\n".join(unavailable_perspectives)
                fallback += "\n\n"
            fallback += specialist_context
            return fallback

    async def _run_multi_perspective(
        self,
        stage_summaries: List[StageSummary],
        config: Dict,
        manifest: Manifest,
        mode: str,
    ) -> FinalAnalysis:
        """Run multi-perspective parallel analysis with specialist agents."""
        prompt_template = self._load_prompt_template(mode)
        selected_specialists = self._get_selected_specialists(config)
        specialist_concurrency = self._get_specialist_concurrency_limit(
            config, selected_specialists
        )
        semaphore = asyncio.Semaphore(specialist_concurrency)

        async def run_specialist(
            analyst_type: str, client, model: str
        ) -> Dict[str, Any]:
            async with semaphore:
                return await self._generate_specialist_analysis(
                    analyst_type, stage_summaries, prompt_template, client, model
                )

        # Create one client per specialist (may route to different models)
        tasks = []
        for analyst_type in selected_specialists:
            try:
                client = get_llm_client(config, agent_type=analyst_type)
            except Exception:
                client = get_llm_client(config, agent_type="analyst")
            model = getattr(client, "model", "unknown")
            tasks.append(run_specialist(analyst_type, client, model))

        logger.info(
            "[FinalAnalysisStage] Running %d specialist agents in parallel (max concurrency=%d)",
            len(tasks),
            specialist_concurrency,
        )
        results = await asyncio.gather(*tasks)

        successful = [r for r in results if r.get("status") == "completed"]
        logger.info(
            "[FinalAnalysisStage] %d/%d specialists completed",
            len(successful),
            len(results),
        )

        success_threshold = min(3, len(selected_specialists))
        if len(successful) < success_threshold:
            logger.warning(
                f"[FinalAnalysisStage] <{success_threshold} specialists succeeded, "
                "falling back to single-pass"
            )
            return await self._run_single_pass(stage_summaries, config, manifest, mode)

        # Aggregate via meta-agent
        meta_client = get_llm_client(config, agent_type="analyst")
        meta_model = getattr(meta_client, "model", "unknown")
        final_result = await self._aggregate_with_meta_agent(
            results, meta_client, meta_model
        )

        timestamp = datetime.now()
        return FinalAnalysis(
            status="completed",
            stages=stage_summaries,
            final_result=final_result,
            metadata={
                "generated_at": timestamp.isoformat(),
                "stage_count": len(stage_summaries),
                "stage_indices": [s.stage_index for s in stage_summaries],
                "model": meta_model,
                "multi_perspective_analysis": True,
                "specialist_analyses_performed": [
                    r["analyst_type"] for r in successful
                ],
                "selected_specialists": selected_specialists,
                "specialist_counts": {
                    "completed": len(successful),
                    "failed": len(results) - len(successful),
                    "total_requested": len(results),
                },
                "specialist_max_concurrency": specialist_concurrency,
                "specialist_success_threshold": success_threshold,
                "estimated_tokens": estimate_tokens(final_result),
                "specialist_full_results": results,
            },
        )

    async def _run_single_pass(
        self,
        stage_summaries: List[StageSummary],
        config: Dict,
        manifest: Manifest,
        mode: str,
    ) -> FinalAnalysis:
        """Run single-pass analysis using one LLM call.

        Raises LLMError on LLM failure so the caller (run()) can handle it.
        """
        prompt_template = self._load_prompt_template(mode)
        client = get_llm_client(config, agent_type="analyst")
        model = getattr(client, "model", "unknown")

        logger.info(
            "[FinalAnalysisStage] Generating final synthesis from %d stage summaries...",
            len(stage_summaries),
        )

        combined_context = self._build_combined_context(stage_summaries)
        full_prompt = prompt_template + combined_context

        # Let LLMError propagate so run() can handle it properly
        response = await client.acomplete(full_prompt)

        timestamp = datetime.now()
        return FinalAnalysis(
            status="completed",
            stages=stage_summaries,
            final_result=response,
            metadata={
                "generated_at": timestamp.isoformat(),
                "stage_count": len(stage_summaries),
                "stage_indices": [s.stage_index for s in stage_summaries],
                "model": model,
                "estimated_tokens": estimate_tokens(response),
                "mode": mode,
                "multi_perspective_analysis": False,
            },
        )

    def _save_final_analysis(
        self,
        final_analysis: FinalAnalysis,
        manifest: Manifest,
        output_dir: Path,
        mode: str,
    ) -> tuple:
        """Save final analysis to MD and JSON files."""
        timestamp = final_analysis.metadata.get(
            "generated_at", datetime.now().isoformat()
        )
        model = final_analysis.metadata.get("model", "unknown")
        stage_count = final_analysis.metadata.get(
            "stage_count", len(final_analysis.stages)
        )

        input_path = Path(manifest.input_path)
        input_filename = input_path.name
        estimated_tokens = manifest.estimated_tokens or "N/A"

        created_at = manifest.created_at
        now = datetime.now()
        processing_seconds = (now - created_at).total_seconds()

        md_content = f"""# Final Analysis for {input_filename}

**Generated:** {timestamp}
**Source File:** [{input_filename}]({input_path.name})
**Analysis Scope:** {manifest.total_parts or "N/A"} parts, {stage_count} stages analyzed
**Processing Time:** {processing_seconds:.1f}s
**Tokens Estimated:** {estimated_tokens}
**Models Used:** {model}
**Mode:** {mode}

{final_analysis.final_result}

---

_Final analysis generated by {model} ({timestamp})_

**Confidence Level:** [ ] High [ ] Medium [ ] Low
**Review Needed:** [ ] Yes [ ] No [ ] Critical Elements Only
"""

        md_path = output_dir / "final_analysis.md"
        write_file(str(md_path), md_content)

        json_data = {
            "status": final_analysis.status,
            "generated_at": timestamp,
            "input_file": str(input_path),
            "analysis_scope": {
                "total_parts": manifest.total_parts,
                "total_stages": stage_count,
            },
            "processing_time_seconds": processing_seconds,
            "estimated_tokens": manifest.estimated_tokens,
            "model": model,
            "mode": mode,
            "final_result": final_analysis.final_result,
            "metadata": {
                k: v
                for k, v in final_analysis.metadata.items()
                if k != "specialist_full_results"  # avoid dumping huge nested results
            },
        }

        json_path = output_dir / "final_analysis.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        return str(md_path), str(json_path)

    async def run(
        self,
        stage_summaries: List[StageSummary],
        config: Dict,
        manifest: Manifest,
        mode: str = "general",
        multi_perspective: bool = False,
    ) -> FinalAnalysis:
        """Run the final analysis stage on all stage summaries.

        Args:
            stage_summaries: List of StageSummary objects from stage synthesis
            config: Configuration dictionary with LLM settings
            manifest: Manifest object to update
            mode: Analysis mode ('general' or 'relationship')
            multi_perspective: Whether to use multi-perspective parallel analysis

        Returns:
            FinalAnalysis object with comprehensive results

        Raises:
            ValueError: If mode is invalid or stage summaries empty
            LLMError: If LLM communication fails
        """
        if mode not in ("general", "relationship"):
            raise ValueError(
                f"Invalid mode: '{mode}'. Must be 'general' or 'relationship'."
            )

        if mode == "relationship":
            logger.info(
                "[FinalAnalysisStage] EXPERIMENTAL MODE: Relationship-focused analysis enabled"
            )
        if multi_perspective:
            logger.info(
                "[FinalAnalysisStage] MULTI-PERSPECTIVE MODE: Using parallel specialist agents"
            )

        self._validate_stage_summaries(stage_summaries)

        self.manifest_manager.update_stage(manifest, "final", "running")
        self.manifest_manager.save_manifest(manifest)

        output_dir = Path(manifest.input_path).parent / ".longtext"
        output_dir.mkdir(exist_ok=True)

        try:
            if multi_perspective:
                final_analysis = await self._run_multi_perspective(
                    stage_summaries, config, manifest, mode
                )
            else:
                final_analysis = await self._run_single_pass(
                    stage_summaries, config, manifest, mode
                )

            final_analysis.metadata["mode"] = mode

            md_path, json_path = self._save_final_analysis(
                final_analysis, manifest, output_dir, mode
            )

            logger.info("[FinalAnalysisStage] Final analysis saved: %s", md_path)
            logger.info("[FinalAnalysisStage] JSON backup saved: %s", json_path)

            self.manifest_manager.update_stage(
                manifest,
                "final",
                "successful",
                output_file=md_path,
                stats={
                    "completed_at": datetime.now().isoformat(),
                    "output_md": md_path,
                    "output_json": json_path,
                    "multi_perspective": multi_perspective,
                },
            )
            manifest.status = "completed"
            manifest.updated_at = datetime.now()
            self.manifest_manager.save_manifest(manifest)

            return final_analysis

        except LLMError as e:
            logger.error(
                "[FinalAnalysisStage] LLM error during final analysis: %s",
                e,
                exc_info=True,
            )

            timestamp = datetime.now()
            json_path = output_dir / "final_analysis_partial.json"
            json_data = {
                "status": "failed",
                "error": str(e),
                "error_type": type(e).__name__,
                "generated_at": timestamp.isoformat(),
                "stage_count": len(stage_summaries),
                "mode": mode,
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            self.manifest_manager.update_stage(
                manifest,
                "final",
                "failed",
                error=str(e),
                stats={"partial_json_saved": str(json_path)},
            )
            manifest.status = "failed"
            manifest.updated_at = datetime.now()
            self.manifest_manager.save_manifest(manifest)

            raise

        except Exception as e:
            logger.exception("[FinalAnalysisStage] Unexpected error: %s", e)

            self.manifest_manager.update_stage(
                manifest,
                "final",
                "failed",
                error=f"Unexpected error: {e}",
            )
            manifest.status = "failed"
            manifest.updated_at = datetime.now()
            self.manifest_manager.save_manifest(manifest)

            raise
