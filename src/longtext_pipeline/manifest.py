"""
Manifest system for checkpoint tracking and resume capability.

Provides atomic write operations and hash validation for stale checkpoint detection.
"""

import json
import random
import string
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .models import Manifest, StageInfo
from .utils.io import ensure_dir, write_file
from .utils.hashing import hash_content


class ManifestManager:
    """
    Manages pipeline state through manifest files.

    Supports atomic writes to prevent corruption and hash validation
    to detect stale checkpoints.
    """

    MANIFEST_FILENAME = "manifest.json"
    OUTPUT_DIR = ".longtext"

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize manifest manager.

        Args:
            base_dir: Base output directory. Defaults to current directory.
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self._lock = threading.Lock()

    def _get_manifest_path(self, input_path: str) -> Path:
        """
        Get the manifest file path for a given input file.

        The manifest is stored in .longtext/ directory within the input file's directory.

        Args:
            input_path: Path to the input file

        Returns:
            Path to manifest.json file
        """
        input_file = Path(input_path)
        manifest_dir = input_file.parent / self.OUTPUT_DIR
        return manifest_dir / self.MANIFEST_FILENAME

    def _generate_session_id(self) -> str:
        """
        Generate a unique session ID.

        Format: YYYYMMDD_HHMMSS_[random_suffix]

        Returns:
            Unique session identifier
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = "".join(
            random.choices(string.ascii_lowercase + string.digits, k=6)
        )
        return f"{timestamp}_{random_suffix}"

    def _ensure_output_structure(self, manifest_path: Path) -> None:
        """
        Create the output directory structure.

        Creates .longtext/ directory and .metadata/ subdirectory.

        Args:
            manifest_path: Path to manifest file
        """
        manifest_dir = manifest_path.parent
        ensure_dir(manifest_dir)

        # Create .metadata/ subdirectory
        metadata_dir = manifest_dir / ".metadata"
        ensure_dir(metadata_dir)

    def _convert_to_dict(self, manifest: Manifest) -> Dict[str, Any]:
        """
        Convert Manifest dataclass to JSON-serializable dict.

        Args:
            manifest: Manifest object to convert

        Returns:
            Dictionary ready for JSON serialization
        """
        stages_dict = {}
        for stage_name, stage_info in manifest.stages.items():
            stages_dict[stage_name] = {
                "status": stage_info.status,
                "input_file": stage_info.input_file,
                "output_file": stage_info.output_file,
                "timestamp": stage_info.timestamp.isoformat()
                if stage_info.timestamp
                else None,
                "error": stage_info.error,
                "stats": stage_info.stats,
            }

        return {
            "session_id": manifest.session_id,
            "input_path": manifest.input_path,
            "input_hash": manifest.input_hash,
            "created_at": manifest.created_at.isoformat(),
            "updated_at": manifest.updated_at.isoformat(),
            "status": manifest.status,
            "stages": stages_dict,
            "total_parts": manifest.total_parts,
            "total_stages": manifest.total_stages,
            "estimated_tokens": manifest.estimated_tokens,
        }

    def _convert_from_dict(self, data: Dict[str, Any]) -> Manifest:
        """
        Convert dict to Manifest dataclass.

        Args:
            data: Dictionary from JSON

        Returns:
            Manifest object
        """
        stages = {}
        for stage_name, stage_data in data.get("stages", {}).items():
            stages[stage_name] = StageInfo(
                name=stage_name,
                status=stage_data.get("status", "not_started"),
                input_file=stage_data.get("input_file"),
                output_file=stage_data.get("output_file"),
                timestamp=datetime.fromisoformat(stage_data["timestamp"])
                if stage_data.get("timestamp")
                else None,
                error=stage_data.get("error"),
                stats=stage_data.get("stats"),
            )

        return Manifest(
            session_id=data["session_id"],
            input_path=data["input_path"],
            input_hash=data["input_hash"],
            stages=stages,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            status=data["status"],
            total_parts=data.get("total_parts"),
            total_stages=data.get("total_stages"),
            estimated_tokens=data.get("estimated_tokens"),
        )

    def create_manifest(
        self, input_path: str, content_hash: Optional[str] = None
    ) -> Manifest:
        """
        Create a new manifest for a pipeline session.

        Args:
            input_path: Path to input file
            content_hash: Optional pre-computed hash of input content.
                         If None, will be computed from file.

        Returns:
            New Manifest object
        """
        input_path = str(Path(input_path).resolve())

        if content_hash is None:
            from .utils.io import read_file

            content = read_file(input_path)
            content_hash = hash_content(content)

        session_id = self._generate_session_id()
        now = datetime.now()

        # Initialize all stages with not_started status
        stages = {
            "ingest": StageInfo(name="ingest", status="not_started"),
            "summarize": StageInfo(name="summarize", status="not_started"),
            "stage": StageInfo(name="stage", status="not_started"),
            "final": StageInfo(name="final", status="not_started"),
            "audit": StageInfo(name="audit", status="not_started"),
        }

        manifest = Manifest(
            session_id=session_id,
            input_path=input_path,
            input_hash=content_hash,
            stages=stages,
            created_at=now,
            updated_at=now,
            status="not_started",
        )

        return manifest

    def load_manifest(self, input_path: str) -> Optional[Manifest]:
        """
        Load manifest from .longtext/manifest.json for a given input file.

        Thread-safe for concurrent reads alongside async writes.

        Args:
            input_path: Path to input file

        Returns:
            Manifest object if exists, None otherwise
        """
        manifest_path = self._get_manifest_path(input_path)

        if not manifest_path.exists():
            return None

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return self._convert_from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ManifestError(f"Failed to load manifest from {manifest_path}: {e}")

    def save_manifest(self, manifest: Manifest) -> None:
        """
        Save manifest atomically to .longtext/manifest.json.

        Uses atomic write pattern:
        1. Write to temp file first
        2. Sync to disk
        3. Rename to target path

        Thread-safe with threading.Lock for concurrent operations.

        Args:
            manifest: Manifest object to save

        Raises:
            ManifestError: If save operation fails
        """
        manifest_path = self._get_manifest_path(manifest.input_path)

        # Ensure output directory structure
        self._ensure_output_structure(manifest_path)

        # Convert to JSON
        data = self._convert_to_dict(manifest)
        json_content = json.dumps(data, indent=2, ensure_ascii=False)

        with self._lock:
            try:
                # Atomic write using the utilities module
                write_file(str(manifest_path), json_content)
            except Exception as e:
                raise ManifestError(f"Failed to save manifest to {manifest_path}: {e}")

    def update_stage(
        self,
        manifest: Manifest,
        stage_name: str,
        status: str,
        output_file: Optional[str] = None,
        error: Optional[str] = None,
        stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update stage progress in manifest.

        Thread-safe for concurrent reads alongside async writes.

        Args:
            manifest: Manifest to update
            stage_name: Name of stage to update
            status: New status ('not_started', 'running', 'successful',
                'successful_with_warnings', 'failed', 'skipped')
            output_file: Optional path to output file
            error: Optional error message if stage failed
            stats: Optional stage-specific statistics
        """
        if stage_name not in manifest.stages:
            manifest.stages[stage_name] = StageInfo(name=stage_name, status=status)

        stage = manifest.stages[stage_name]
        stage.status = status
        stage.output_file = output_file
        stage.error = error
        stage.stats = stats if stats is not None else stage.stats
        stage.timestamp = datetime.now()
        manifest.updated_at = datetime.now()
        if status not in {"successful", "successful_with_warnings", "skipped"}:
            manifest.status = status

    def is_stage_complete(self, manifest: Manifest, stage_name: str) -> bool:
        """
        Check if a stage has been completed successfully.

        Args:
            manifest: Manifest to check
            stage_name: Name of stage to check

        Returns:
            True if stage status is a completed success state
        """
        if stage_name not in manifest.stages:
            return False

        return manifest.stages[stage_name].status in {
            "successful",
            "successful_with_warnings",
        }

    def should_resume(self, manifest: Manifest, input_hash: str) -> bool:
        """
        Validate if checkpoint is fresh based on input hash.

        Thread-safe for concurrent reads alongside async writes.

        Args:
            manifest: Loaded manifest to check
            input_hash: Current hash of input file

        Returns:
            True if hashes match (checkpoint is fresh), False if stale
        """
        return manifest.input_hash == input_hash

    def is_pipeline_complete(self, manifest: Manifest) -> bool:
        """
        Check if the entire pipeline has completed successfully.

        Args:
            manifest: Manifest to check

        Returns:
            True if final stage is successful
        """
        return self.is_stage_complete(manifest, "final")

    def get_completed_stages(self, manifest: Manifest) -> list:
        """
        Get list of stages that have completed successfully.

        Args:
            manifest: Manifest to check

        Returns:
            List of completed stage names
        """
        return [
            name
            for name, stage in manifest.stages.items()
            if stage.status in {"successful", "successful_with_warnings"}
        ]

    def create_from_existing(
        self, existing_manifest: Manifest, input_hash: str
    ) -> Optional[Manifest]:
        """
        Create a resumed manifest from existing one with hash validation.

        Args:
            existing_manifest: Previously saved manifest
            input_hash: Current hash of input file

        Returns:
            Manifest if hash matches, None if stale (caller should create fresh)
        """
        if not self.should_resume(existing_manifest, input_hash):
            return None
        return existing_manifest


class ManifestError(Exception):
    """Raised when manifest operation fails."""

    pass
