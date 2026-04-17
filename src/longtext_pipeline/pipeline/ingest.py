"""
Ingest stage implementation for the longtext pipeline.

This module provides the IngestStage class for reading input files,
cleaning text, and splitting content into manageable parts for downstream
processing. Implements the input processing pipeline with PDF and DOCX support.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from ..errors import InputError
from ..manifest import ManifestManager
from ..models import Manifest, Part
from ..splitter import TextSplitter
from ..utils.io import read_file
from ..utils.text_clean import clean_text
from ..utils.token_estimator import estimate_tokens
from .pdf_extraction import PDFTextExtractor
from .docx_extraction import DOCXTextExtractor


logger = logging.getLogger(__name__)


def _safe_display_relative_path(input_path: str, base_dir: Path) -> str:
    """Return a stable relative path for metadata, even across Windows path aliases."""
    try:
        return Path(input_path).resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        try:
            return Path(
                os.path.relpath(
                    str(Path(input_path).resolve()), str(base_dir.resolve())
                )
            ).as_posix()
        except ValueError:
            return Path(input_path).name


def get_content_type(ext: str) -> str:
    """Determine content type based on file extension - indicates the original file type that generated this text."""
    content_types = {
        ".txt": "text/plain.source",  # Plain text coming from original plain text
        ".md": "text/markdown.source",  # Markdown coming from original markdown
        ".pdf": "text/plain.extracted_from_pdf",  # Text extracted from PDF
        ".docx": "text/plain.extracted_from_docx",  # Text extracted from DOCX
    }
    return content_types.get(ext, "text/plain.unknown_source")


class IngestStage:
    """Reads input, performs preprocessing, splits content into parts.

    The ingest stage handles file reading, text cleaning, and content
    splitting according to the pipeline's checkpoint and manifest system.
    It's responsible for creating the part_* files and updating the manifest
    with processing state.
    """

    def __init__(self, manifest_manager: Optional[ManifestManager] = None):
        """Initialize the ingest stage.

        Args:
            manifest_manager: Optional existing manifest manager
        """
        self.manifest_manager = manifest_manager or ManifestManager()
        self.splitter = TextSplitter(chunk_size=1000, overlap=100)

    def run(self, input_path: str, config: Dict, manifest: Manifest) -> List[Part]:
        """Run the ingest stage on input file.

        Args:
            input_path: Path to input file (txt/md/pdf/docx)
            config: Configuration dictionary
            manifest: Manifest object to update

        Returns:
            List of Part objects representing split text parts

        Raises:
            InputError: If input is empty or invalid
        """
        # Check if input file exists
        input_path = str(Path(input_path).resolve())
        if not Path(input_path).exists():
            raise InputError(f"Input file does not exist: {input_path}")

        # Verify input file extension (now supporting txt/md/pdf/docx)
        input_ext = Path(input_path).suffix.lower()
        if input_ext not in [".txt", ".md", ".pdf", ".docx"]:
            raise InputError(
                f"Only .txt, .md, .pdf, and .docx files supported, got: {input_ext}"
            )

        # Update manifest to indicate started ingest stage
        self.manifest_manager.update_stage(
            manifest, "ingest", "running", output_file=input_path
        )

        # 1. Read input file (txt/md/pdf/docx)
        if input_ext == ".pdf":
            # Handle PDF files with pdfplumber
            try:
                pdf_extractor = PDFTextExtractor()
                raw_content = pdf_extractor.extract_and_preprocess_pdf(
                    input_path, config=config
                )
            except ImportError as e:
                raise InputError(
                    f"PDF support requires additional dependencies: {e}"
                ) from e
        elif input_ext == ".docx":
            # Handle DOCX files with python-docx
            try:
                docx_extractor = DOCXTextExtractor()
                raw_content = docx_extractor.extract_and_preprocess_docx(
                    input_path, config=config
                )
            except ImportError as e:
                raise InputError(
                    f"DOCX support requires additional dependencies: {e}"
                ) from e
        else:
            # Handle text-based files as before
            raw_content = read_file(input_path)

        # Handle empty input - raise InputError early
        if not raw_content.strip():
            error_msg = "Input file is empty"
            self.manifest_manager.update_stage(
                manifest, "ingest", "failed", error=error_msg
            )
            raise InputError(error_msg)

        # 2. Clean text (using text_clean.clean_text)
        cleaned_content = clean_text(raw_content)

        # Get token count estimate and log for awareness
        estimated_token_count = estimate_tokens(cleaned_content)
        logger.info("Estimated token count for input: %s", f"{estimated_token_count:,}")

        # 3. Split into parts (using splitter.split_text)
        # Accept either full pipeline config or a flat ingest config.
        ingest_config = config.get("stages", {}).get("ingest", config)
        chunk_size = ingest_config.get("chunk_size", 1000)
        overlap_size = ingest_config.get("overlap")
        if overlap_size is None:
            overlap_rate = ingest_config.get("overlap_rate")
            if overlap_rate is not None:
                overlap_size = int(chunk_size * overlap_rate)
            else:
                overlap_size = 100

        # Get parts from the splitter
        try:
            parts = self.splitter.split_text(
                content=cleaned_content, chunk_size=chunk_size, overlap=overlap_size
            )
        except InputError as e:
            # If the splitter raises an error, update manifest and re-raise
            self.manifest_manager.update_stage(
                manifest, "ingest", "failed", error=str(e)
            )
            raise e

        # Handle tiny input: single part marked as "skip_summary" if too small
        if len(parts) == 1:
            if estimated_token_count < 100:  # Very tiny input
                # Add metadata to mark this part as needing special handling
                parts[0].metadata["skip_summary"] = True
                parts[0].metadata["reason"] = "Tiny input - skip summarization"
                logger.info(
                    "Tiny input detected (<100 tokens); marking for skip_summary: %s tokens",
                    estimated_token_count,
                )

        # 4. Determine output directory from config or use default .longtext/
        output_dir_config = config.get("output", {}).get("dir")
        if output_dir_config:
            # Use configured output directory
            parts_dir = Path(output_dir_config) / ".longtext"
        else:
            # Default: adjacent .longtext/ directory
            parts_dir = Path(input_path).parent / ".longtext"
        parts_dir.mkdir(parents=True, exist_ok=True)

        saved_parts_paths = []
        for part in parts:
            # Create the text portion of the part file
            part_relative_path = _safe_display_relative_path(
                input_path, parts_dir.parent
            )
            part_filename = f"part_{part.index:02d}.txt"
            part_path = parts_dir / part_filename

            # Format the part file content following the Part Files Format in DATA_MODEL.md
            metadata_lines = [
                f"INPUT_PATH: {part_relative_path}",
                f"PART_INDEX: {part.index}",
                f"TOKEN_COUNT: {part.token_count}",
                f"CHUNK_SIZE: {len(part.content)}",
                f"CONTENT_TYPE: {get_content_type(input_ext)}",
                "METADATA_END: ---END---",
                "",  # Empty line before content
                part.content,
            ]
            part_file_content = "\n".join(metadata_lines)

            # Write the part file with atomic write operations
            from ..utils.io import write_file

            write_file(part_path, part_file_content)
            saved_parts_paths.append(str(part_path))

        # Update the manifest with successful ingest stage completion
        self.manifest_manager.update_stage(
            manifest,
            "ingest",
            "successful",
            output_file=str(parts_dir),  # Root directory for parts
            stats={
                "parts_created": len(parts),
                "estimated_tokens": estimated_token_count,
                "saved_parts": saved_parts_paths,
                "output_dir_used": str(parts_dir),
            },
        )

        # Update the overall manifest status to reflect completion of ingest stage
        manifest.status = "ingesting"  # Partially completed
        from datetime import datetime

        manifest.updated_at = datetime.now()  # Refresh timestamp

        # Update manifest properties with the processing results
        manifest.total_parts = len(parts)
        manifest.estimated_tokens = estimated_token_count

        # Save the updated manifest
        self.manifest_manager.save_manifest(manifest)

        # 6. Return parts for next stage
        return parts
