"""Core data models for the longtext pipeline.

This module defines pure data containers (dataclasses) for all pipeline operations.
No business logic is implemented in these classes - they are strictly for data
representation and serialization.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class StageInfo:
    """Represents information about a single pipeline stage.
    
    Attributes:
        name: Stage name (e.g., 'ingest', 'summarize', 'stage', 'final', 'audit')
        status: Current status ('not_started', 'running', 'successful', 'failed')
        input_file: Path to input file for this stage
        output_file: Path to output file for this stage
        timestamp: When this stage was processed
        error: Error message if stage failed
    """
    name: str
    status: str
    input_file: Optional[str] = None
    output_file: Optional[str] = None
    timestamp: Optional[datetime] = None
    error: Optional[str] = None


@dataclass
class Part:
    """Represents a single text part after ingestion.
    
    Attributes:
        index: 0-based index of this part
        content: Raw text content of this chunk
        token_count: Approximate token count
        metadata: Additional metadata about this part
    """
    index: int
    content: str
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Summary:
    """Represents a summary of a single part.
    
    Attributes:
        part_index: Index of the part this summarizes (0-based)
        content: Summary content
        metadata: Additional metadata about the summary
    """
    part_index: int
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageSummary:
    """Represents a synthesized stage summary from multiple part summaries.
    
    Attributes:
        stage_index: 0-based index of this stage
        summaries: List of Summary objects included in this stage
        synthesis: Overall synthesis content for this stage
        metadata: Additional metadata about the stage summary
    """
    stage_index: int
    summaries: List[Summary]
    synthesis: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FinalAnalysis:
    """Represents the final analysis result.
    
    Attributes:
        status: Overall pipeline status ('completed', 'partial_success', 'failed')
        stages: List of StageSummary objects processed
        final_result: Final synthesis content
        metadata: Additional metadata about the final analysis
    """
    status: str
    stages: List[StageSummary]
    final_result: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Manifest:
    """Main manifest object tracking the entire pipeline execution state.
    
    This is the core checkpoint structure that enables resumable execution.
    
    Attributes:
        session_id: Unique ID for this pipeline run
        input_path: Path to original input file
        input_hash: SHA-256 hash of input file content
        stages: Dict mapping stage names to StageInfo objects
        created_at: When session began
        updated_at: When manifest was last updated
        status: Overall pipeline status
        total_parts: Total number of parts created after ingestion
        total_stages: Total number of stages after grouping 
        estimated_tokens: Rough token count of input
    """
    session_id: str
    input_path: str
    input_hash: str
    stages: Dict[str, StageInfo]
    created_at: datetime
    updated_at: datetime
    status: str
    total_parts: Optional[int] = None
    total_stages: Optional[int] = None
    estimated_tokens: Optional[int] = None
