"""
Final analysis stage implementation for the longtext pipeline.

This module provides the FinalAnalysisStage class for generating comprehensive
final analysis from all stage summaries using LLM-based cross-stage synthesis.
Supports both General and Relationship modes via different prompt templates.
Unlike other stages, implements error handling that captures partial results
when LLM fails (single-operation failure mode rather than Continue-with-Partial).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..errors import LLMError
from ..llm.factory import get_llm_client
from ..manifest import ManifestManager
from ..models import FinalAnalysis, Manifest, StageSummary
from ..utils.io import read_file, write_file


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
            prompt_file = "final_relationship.txt"
        else:
            prompt_file = "final_general.txt"
        
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
    
    def _validate_stage_summaries(self, stage_summaries: List[StageSummary]) -> None:
        """Validate that stage summaries exist and are non-empty.
        
        Args:
            stage_summaries: List of StageSummary objects to validate
            
        Raises:
            ValueError: If stage summaries list is empty
        """
        if not stage_summaries:
            raise ValueError(
                "Cannot perform final analysis: no stage summaries provided. "
                "Ensure stage synthesis stage completed successfully."
            )
    
    def _build_combined_context(self, stage_summaries: List[StageSummary]) -> str:
        """Combine all stage summaries into single context string.
        
        Args:
            stage_summaries: List of StageSummary objects to combine
            
        Returns:
            Combined text of all stage summaries with separators
        """
        combined = ""
        for i, stage_summary in enumerate(stage_summaries):
            combined += f"\n\n--- Stage {i} (Index {stage_summary.stage_index}) ---\n"
            combined += stage_summary.synthesis
        
        return combined
    
    def _generate_final_analysis(
        self,
        stage_summaries: List[StageSummary],
        prompt_template: str,
        client,
        model: str,
        manifest: Manifest
    ) -> FinalAnalysis:
        """Generate final analysis using LLM.
        
        Args:
            stage_summaries: List of StageSummary objects to synthesize
            prompt_template: Loaded prompt template
            client: LLM client instance
            model: Model name for metadata
            manifest: Manifest for metadata extraction
            
        Returns:
            FinalAnalysis object with comprehensive results
            
        Raises:
            LLMError: If LLM call fails
        """
        # Build combined context from all stage summaries
        combined_context = self._build_combined_context(stage_summaries)
        
        # Build full prompt by appending combined context
        full_prompt = prompt_template + combined_context
        
        # Call LLM to generate final analysis
        response = client.complete(full_prompt)
        
        # Create FinalAnalysis object
        timestamp = datetime.now()
        final_analysis = FinalAnalysis(
            status="completed",
            stages=stage_summaries,
            final_result=response,
            metadata={
                "generated_at": timestamp.isoformat(),
                "stage_count": len(stage_summaries),
                "stage_indices": [s.stage_index for s in stage_summaries],
                "model": model,
                "estimated_tokens": len(response.split()) // 1.3  # Rough estimate
            }
        )
        
        return final_analysis
    
    def _save_final_analysis(
        self,
        final_analysis: FinalAnalysis,
        manifest: Manifest,
        output_dir: Path,
        mode: str
    ) -> tuple:
        """Save final analysis to MD and JSON files.
        
        Args:
            final_analysis: FinalAnalysis object to save
            manifest: Manifest for metadata
            output_dir: Directory to save files in
            mode: Analysis mode for metadata
            
        Returns:
            Tuple of (md_path, json_path) strings
        """
        timestamp = final_analysis.metadata.get("generated_at", datetime.now().isoformat())
        model = final_analysis.metadata.get("model", "unknown")
        stage_count = final_analysis.metadata.get("stage_count", len(final_analysis.stages))
        
        # Extract manifest metadata
        input_path = Path(manifest.input_path)
        input_filename = input_path.name
        estimated_tokens = manifest.estimated_tokens or "N/A"
        
        # Calculate processing time
        created_at = manifest.created_at
        now = datetime.now()
        processing_seconds = (now - created_at).total_seconds()
        
        # Format MD content
        md_content = f"""# Final Analysis for {input_filename}

**Generated:** {timestamp}
**Source File:** [{input_filename}]({input_path.absolute()})
**Analysis Scope:** {manifest.total_parts or 'N/A'} parts, {stage_count} stages analyzed
**Processing Time:** {processing_seconds:.1f}s
**Tokens Analyzed:** {estimated_tokens}
**Models Used:** {model}
**Mode:** {mode}

{final_analysis.final_result}

---

_Final analysis generated by {model} ({timestamp})_

**Confidence Level:** [ ] High [ ] Medium [ ] Low
**Review Needed:** [ ] Yes [ ] No [ ] Critical Elements Only
"""
        
        # Save MD file
        md_path = output_dir / "final_analysis.md"
        write_file(str(md_path), md_content)
        
        # Prepare JSON backup
        json_data = {
            "status": final_analysis.status,
            "generated_at": timestamp,
            "input_file": str(input_path),
            "analysis_scope": {
                "total_parts": manifest.total_parts,
                "total_stages": stage_count
            },
            "processing_time_seconds": processing_seconds,
            "estimated_tokens": manifest.estimated_tokens,
            "model": model,
            "mode": mode,
            "final_result": final_analysis.final_result,
            "metadata": final_analysis.metadata
        }
        
        # Save JSON backup
        json_path = output_dir / "final_analysis.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        return str(md_path), str(json_path)
    
    def run(
        self,
        stage_summaries: List[StageSummary],
        config: Dict,
        manifest: Manifest,
        mode: str = "general"
    ) -> FinalAnalysis:
        """Run the final analysis stage on all stage summaries.
        
        Args:
            stage_summaries: List of StageSummary objects from stage synthesis
            config: Configuration dictionary with LLM settings
            manifest: Manifest object to update
            mode: Analysis mode ('general' or 'relationship')
            
        Returns:
            FinalAnalysis object with comprehensive results
            
        Raises:
            ValueError: If stage summaries list is empty
            FileNotFoundError: If prompt template not found
            LLMError: If LLM communication fails (returns partial result)
        """
        # Validate mode parameter
        if mode not in ("general", "relationship"):
            raise ValueError(
                f"Invalid mode: '{mode}'. Must be 'general' or 'relationship'."
            )
        
        # Log experimental mode warning
        if mode == "relationship":
            print("[FinalAnalysisStage] EXPERIMENTAL MODE: Relationship-focused analysis enabled")
        
        # Validate stage summaries exist
        self._validate_stage_summaries(stage_summaries)
        
        # Update manifest to indicate running final stage
        self.manifest_manager.update_stage(manifest, 'final', 'running')
        self.manifest_manager.save_manifest(manifest)
        
        # Load prompt template
        prompt_template = self._load_prompt_template(mode)
        
        # Create LLM client
        client = get_llm_client(config)
        model = config.get('model', 'unknown')
        
        # Determine output directory
        output_dir = Path(manifest.input_path).parent / ".longtext"
        output_dir.mkdir(exist_ok=True)
        
        try:
            print(f"[FinalAnalysisStage] Generating final analysis from {len(stage_summaries)} stage summaries...")
            
            # Generate final analysis
            final_analysis = self._generate_final_analysis(
                stage_summaries=stage_summaries,
                prompt_template=prompt_template,
                client=client,
                model=model,
                manifest=manifest
            )
            
            # Add mode to metadata
            final_analysis.metadata['mode'] = mode
            
            # Save final analysis to MD and JSON
            md_path, json_path = self._save_final_analysis(
                final_analysis=final_analysis,
                manifest=manifest,
                output_dir=output_dir,
                mode=mode
            )
            
            print(f"[FinalAnalysisStage] Final analysis saved: {md_path}")
            print(f"[FinalAnalysisStage] JSON backup saved: {json_path}")
            
            # Update manifest on success
            self.manifest_manager.update_stage(
                manifest,
                'final',
                'successful',
                output_file=md_path,
                stats={
                    "completed_at": datetime.now().isoformat(),
                    "output_md": md_path,
                    "output_json": json_path
                }
            )
            manifest.status = 'completed'
            manifest.updated_at = datetime.now()
            self.manifest_manager.save_manifest(manifest)
            
            return final_analysis
            
        except LLMError as e:
            # Handle LLM error - capture partial result if available
            print(f"[FinalAnalysisStage] LLM error during final analysis: {e}")
            
            # Create partial result with error details
            timestamp = datetime.now()
            partial_analysis = FinalAnalysis(
                status="failed",
                stages=stage_summaries,
                final_result=f"[LLM Error: {str(e)}]",
                metadata={
                    "generated_at": timestamp.isoformat(),
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "partial": True,
                    "stage_count": len(stage_summaries),
                    "model": model,
                    "mode": mode
                }
            )
            
            # Save partial result as JSON for debugging
            json_path = output_dir / "final_analysis_partial.json"
            json_data = {
                "status": "failed",
                "error": str(e),
                "error_type": type(e).__name__,
                "generated_at": timestamp.isoformat(),
                "stage_count": len(stage_summaries),
                "model": model,
                "mode": mode,
                "stages_processed": [
                    {
                        "stage_index": s.stage_index,
                        "summary_count": len(s.summaries),
                        "summary_indices": s.metadata.get('summary_indices', [])
                    }
                    for s in stage_summaries
                ]
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            # Update manifest on failure
            self.manifest_manager.update_stage(
                manifest,
                'final',
                'failed',
                error=str(e),
                stats={
                    "error_details": str(e),
                    "partial_json_saved": str(json_path)
                }
            )
            manifest.status = 'failed'
            manifest.updated_at = datetime.now()
            self.manifest_manager.save_manifest(manifest)
            
            # Re-raise to propagate error up
            raise
            
        except Exception as e:
            # Handle unexpected errors
            print(f"[FinalAnalysisStage] Unexpected error: {e}")
            
            # Update manifest
            self.manifest_manager.update_stage(
                manifest,
                'final',
                'failed',
                error=f"Unexpected error: {str(e)}"
            )
            manifest.status = 'failed'
            manifest.updated_at = datetime.now()
            self.manifest_manager.save_manifest(manifest)
            
            raise
