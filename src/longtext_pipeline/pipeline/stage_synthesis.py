"""
Stage synthesis stage implementation for the longtext pipeline.

This module provides the StageSynthesisStage class for aggregating groups
of summaries into higher-level stage summaries using LLM-based analysis.
Implements stage-level synthesis with support for both General and Relationship
modes via different prompt templates. Uses configurable group_size from config
for grouping summaries. Implements Continue-with-Partial error handling to
maximize throughput despite individual group failures.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..errors import LLMError, StageFailedError
from ..grouper import SummaryGrouper
from ..llm.factory import get_llm_client
from ..manifest import ManifestManager
from ..models import Manifest, StageSummary, Summary
from ..utils.io import read_file, write_file
from ..utils.token_estimator import estimate_tokens


logger = logging.getLogger(__name__)


class StageSynthesisStage:
    """Aggregates groups of summaries into higher-level stage summaries.
    
    The stage synthesis stage processes groups of part summaries, combining
    them into cohesive stage-level syntheses. It supports both General and
    Relationship modes (experimental) via different prompt templates. Implements
    Continue-with-Partial error handling to maximize throughput despite
    individual group failures.
    """
    
    def __init__(self, manifest_manager: Optional[ManifestManager] = None):
        """Initialize the stage synthesis stage.
        
        Args:
            manifest_manager: Optional existing manifest manager
        """
        self.manifest_manager = manifest_manager or ManifestManager()
    
    def _load_prompt_template(self, mode: str) -> str:
        """Load prompt template from file based on mode.
        
        Args:
            mode: Either 'general' or 'relationship'
            
        Returns:
            Prompt template string
            
        Raises:
            FileNotFoundError: If prompt template file not found
        """
        # Determine prompt file based on mode
        if mode == "relationship":
            prompt_file = "stage_relationship.txt"
        else:
            prompt_file = "stage_general.txt"
        
        # Build path to prompts directory
        prompts_dir = Path(__file__).parent.parent / "prompts"
        prompt_path = prompts_dir / prompt_file
        
        # Load prompt template
        try:
            return read_file(str(prompt_path))
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Prompt template not found: {prompt_path}. "
                f"Ensure {prompt_file} exists in the prompts directory."
            )
    
    async def _synthesize_group(
        self,
        group: List[Summary],
        group_index: int,
        prompt_template: str,
        client,
        model: str
    ) -> StageSummary:
        """Synthesize a single group of summaries.
        
        Args:
            group: List of Summary objects to synthesize
            group_index: Index of this group (0-based)
            prompt_template: Loaded prompt template
            client: LLM client instance
            model: Model name for metadata
            
        Returns:
            StageSummary object with synthesized content and metadata
            
        Raises:
            LLMError: If LLM call fails
        """
        # Build combined summaries text
        summaries_text = ""
        for i, summary in enumerate(group):
            summaries_text += f"\n\n--- Summary {i} (Part {summary.part_index}) ---\n"
            summaries_text += summary.content
        
        # Build full prompt by appending summaries content
        full_prompt = prompt_template + summaries_text
        
        # Call LLM to generate synthesis (async)
        response = await client.acomplete(full_prompt)
        
        # Parse response into StageSummary object
        timestamp = datetime.now()
        stage_summary = StageSummary(
            stage_index=group_index,
            summaries=group,
            synthesis=response,
            metadata={
                "generated_at": timestamp.isoformat(),
                "summary_count": len(group),
                "summary_indices": [s.part_index for s in group],
                "model": model,
                "estimated_tokens": estimate_tokens(response)
            }
        )
        
        return stage_summary
    
    def _save_stage_summary(self, stage_summary: StageSummary, output_dir: Path) -> str:
        """Save stage summary to file and return path.
        
        Args:
            stage_summary: StageSummary object to save
            output_dir: Directory to save stage summary in
            
        Returns:
            Path to saved stage summary file
        """
        stage_filename = f"stage_{stage_summary.stage_index:02d}.md"
        stage_path = output_dir / stage_filename
        
        # Format summary file references
        summary_refs = []
        for summary in stage_summary.summaries:
            summary_file = f"summary_{summary.part_index:02d}.md"
            summary_refs.append(f"[{summary_file}](./{summary_file})")
        
        # Format stage content with frontmatter
        timestamp = stage_summary.metadata.get("generated_at", datetime.now().isoformat())
        model = stage_summary.metadata.get("model", "unknown")
        
        stage_content = f"""# Stage Summary {stage_summary.stage_index:02d}

**Generated:** {timestamp}  
**Combined From:** {', '.join(summary_refs)}  
**Summary Count:** {len(stage_summary.summaries)}  
**Tokens:** {stage_summary.metadata.get('estimated_tokens', 'N/A')}  

{stage_summary.synthesis}

---

_Stage synthesized by {model} ({timestamp})_
"""
        
        write_file(str(stage_path), stage_content)
        return str(stage_path)
    
    async def run(
        self,
        summaries: List[Summary],
        config: Dict,
        manifest: Manifest,
        mode: str = "general",
    ) -> List[StageSummary]:
        """Async run the stage synthesis stage on all summaries.

        Args:
            summaries: List of Summary objects from summarize stage
            config: Configuration dictionary with LLM settings and group_size
            manifest: Manifest object to update
            mode: Analysis mode ('general' or 'relationship')

        Returns:
            List of StageSummary objects for successfully processed groups

        Raises:
            StageFailedError: If all groups fail (but may contain partial results)
        """
        if mode not in ("general", "relationship"):
            raise ValueError(
                f"Invalid mode: '{mode}'. Must be 'general' or 'relationship'."
            )

        if mode == "relationship":
            logger.warning("Relationship-focused stage synthesis mode is experimental")

        self.manifest_manager.update_stage(manifest, 'stage', 'running')

        prompt_template = self._load_prompt_template(mode)

        group_size = config.get('stages', {}).get('stage', {}).get('group_size', 5)
        logger.info("Stage synthesis using group_size=%s", group_size)

        # Create LLM client using agent-specific model config for stage_synthesizer
        client = get_llm_client(config, agent_type='stage_synthesizer')
        model = getattr(client, "model", "unknown")

        output_dir = Path(manifest.input_path).parent / ".longtext"
        output_dir.mkdir(exist_ok=True)

        max_workers = config.get('pipeline', {}).get('max_workers', 4)
        semaphore = asyncio.Semaphore(max_workers)
        # Create grouper with configurable group_size
        grouper = SummaryGrouper(group_size=group_size)
        
        # Group summaries into stages
        groups = grouper.group_summaries(summaries)
        logger.info(
            "Stage synthesis created %s groups from %s summaries",
            len(groups),
            len(summaries),
        )
        
        # Handle empty summaries list - nothing to synthesize
        if not groups:
            logger.info("No summaries to synthesize")
            stats = {
                "stages_completed": 0,
                "stages_total": 0,
                "errors": [],
                "saved_stages": []
            }
            self.manifest_manager.update_stage(
                manifest,
                'stage',
                'successful',
                stats=stats
            )
            manifest.status = 'staging'
            manifest.updated_at = datetime.now()
            self.manifest_manager.save_manifest(manifest)
            return []
        
        # Track results and errors
        stage_summaries: List[StageSummary] = []
        saved_stage_paths: List[str] = []
        errors: List[Dict] = []
        
        # Create process_group closure for use in both modes
        process_group = self._make_process_group_processor(
            prompt_template=prompt_template,
            client=client,
            model=model,
            mode=mode,
            output_dir=output_dir,
            semaphore=semaphore
        )
        
        # Schedule all groups for concurrent execution within semaphore limit
        tasks = [process_group(group_index, group) for group_index, group in enumerate(groups)]
        results = await asyncio.gather(*tasks)
        
        # Process results maintaining order from original groups list
        for result in results:
            stage_summary, stage_path, error = result
            if error:
                # Error occurred
                errors.append(error)
            else:
                # Success case, add to appropriate lists
                if stage_summary:
                    stage_summaries.append(stage_summary)
                    saved_stage_paths.append(stage_path)
        
        # Calculate statistics
        total_groups = len(groups)
        successful = len(stage_summaries)
        failed = len(errors)
        
        # Prepare manifest statistics
        stats = {
            "stages_completed": successful,
            "stages_total": total_groups,
            "errors": errors,
            "saved_stages": saved_stage_paths
        }
        
        # Determine stage status
        if successful == 0 and failed == total_groups:
            # All groups failed
            stage_status = "failed"
            error_message = f"All {total_groups} groups failed to synthesize"
            self.manifest_manager.update_stage(
                manifest,
                'stage',
                stage_status,
                error=error_message,
                stats=stats
            )
            raise StageFailedError(
                stage_name="stage",
                errors=[Exception(e["error"]) for e in errors],
                partial_result=stage_summaries
            )
        elif failed > 0:
            # Partial success
            stage_status = "successful"  # Stage succeeded overall with partials
            logger.warning(
                "Stage synthesis partial success: %s/%s successful, %s failed",
                successful,
                total_groups,
                failed,
            )
            self.manifest_manager.update_stage(
                manifest,
                'stage',
                stage_status,
                stats=stats
            )
        else:
            # All succeeded
            stage_status = "successful"
            self.manifest_manager.update_stage(
                manifest,
                'stage',
                stage_status,
                stats=stats
            )
        
        # Update manifest status
        manifest.status = 'staging'
        manifest.updated_at = datetime.now()
        self.manifest_manager.save_manifest(manifest)
        
        return stage_summaries

    def _make_process_group_processor(
        self,
        prompt_template: str,
        client,
        model: str,
        mode: str,
        output_dir: Path,
        semaphore: asyncio.Semaphore
    ):
        """Create a closure for processing groups with captured context."""
        async def process_group(group_index: int, group: List[Summary]):
            async with semaphore:
                try:
                    logger.debug(
                        "Processing group %s containing %s summaries",
                        group_index,
                        len(group),
                    )
                    stage_summary = await self._synthesize_group(
                        group=group,
                        group_index=group_index,
                        prompt_template=prompt_template,
                        client=client,
                        model=model
                    )
                    stage_summary.metadata['mode'] = mode
                    stage_path = self._save_stage_summary(stage_summary, output_dir)
                    logger.info("Completed group %s: %s", group_index, stage_path)
                    return stage_summary, stage_path, None
                except LLMError as e:
                    error_info = {
                        "index": group_index,
                        "summary_files": [f"summary_{s.part_index:02d}.md" for s in group],
                        "error": str(e)
                    }
                    logger.warning("Failed group %s: %s", group_index, e)
                    return None, "", error_info
                except Exception as e:
                    error_info = {
                        "index": group_index,
                        "summary_files": [f"summary_{s.part_index:02d}.md" for s in group],
                        "error": f"Unexpected error: {str(e)}"
                    }
                    logger.exception("Unexpected error on group %s", group_index)
                    return None, "", error_info

        return process_group
