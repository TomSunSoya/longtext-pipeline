"""
PDF text extraction module using pdfplumber for the longtext pipeline.

This module provides PDF text extraction functionality with pdfplumber as
the primary extraction engine. It handles various PDF types including text-based
PDFs, multi-column layouts, and basic table structures while providing
fallback mechanisms for complex document formats.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Optional
from ..errors import InputError
from ..config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# Import OCR fallback module - wrap in conditional import
try:
    from .ocr_fallback import OCREngine
except ImportError:
    OCREngine = None

# Conditional import since pdfplumber is not yet in requirements
try:
    import pdfplumber
except ImportError:
    pdfplumber = None


class PDFTextExtractor:
    """Handles PDF text extraction with pdfplumber and fallback mechanisms."""

    def __init__(self):
        """Initialize the PDF text extractor.

        If pdfplumber is not available, raise an error.
        """
        if pdfplumber is None:
            raise ImportError(
                "pdfplumber is required for PDF extraction. "
                "Install with: pip install pdfplumber"
            )

    def extract_text_from_pdf(
        self,
        pdf_path: str,
        extraction_mode: str = "basic",
        config: Optional[Dict] = None,
        **kwargs,
    ) -> str:
        """Extract text from PDF file using pdfplumber initially, with OCR fallback if needed.

        Args:
            pdf_path: Path to the PDF file to extract text from
            extraction_mode: Strategy for text extraction (currently supports "basic",
                            but may support other modes in the future)
            config: Configuration dictionary that may include OCR settings
            **kwargs: Additional extraction parameters

        Returns:
            Extracted text content as string

        Raises:
            InputError: If PDF extraction fails or results in empty content
        """
        if config is None:
            config = {}

        pdf_file_path = Path(pdf_path)

        if not pdf_file_path.exists():
            raise InputError(f"PDF file does not exist: {pdf_path}")

        if not pdf_file_path.is_file():
            raise InputError(f"Path is not a file: {pdf_path}")

        text_content = ""

        try:
            # Suppress pdfminer warnings that are often noisy
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=".*Unknown.*"
                )  # Filter various pdfminer warnings
                warnings.filterwarnings(
                    "ignore", message=".*stream.*"
                )  # Filter PDF stream warnings

                with pdfplumber.open(pdf_file_path) as pdf:
                    # Try basic text extraction first - check if it might be encrypted/password protected
                    if len(pdf.pages) == 0:
                        raise InputError(
                            f"Cannot extract text from encrypted or invalid PDF: {pdf_path}"
                        )

                    # Process pages
                    for page_num, page in enumerate(pdf.pages, 1):
                        try:
                            page_text = page.extract_text()

                            if page_text:
                                # Add page marker for better traceability
                                if extraction_mode in ["basic", "formatted"]:
                                    text_content += f"\n--- PAGE {page_num} ---\n"
                                    text_content += page_text
                                    text_content += "\n-----------\n"
                                else:
                                    text_content += page_text + "\n"

                            # Process tables on the page (optional)
                            if kwargs.get("include_tables", False):
                                tables = page.extract_tables()
                                for table_idx, table in enumerate(tables):
                                    if table:
                                        text_content += f"\nTABLE {table_idx + 1} ON PAGE {page_num}:\n"
                                        # Convert table to text, handling None values
                                        for row in table:
                                            filtered_row = []
                                            for cell in row:
                                                if cell is None:
                                                    filtered_row.append("")
                                                else:
                                                    filtered_row.append(str(cell))
                                            text_content += (
                                                "| " + " | ".join(filtered_row) + " |\n"
                                            )
                                        text_content += "\n"

                        except Exception as e:
                            logger.warning(
                                f"Failed to process page {page_num} in {pdf_path}: {e}"
                            )
                            # Try alternative extraction methods for the problematic page
                            try:
                                # Try extracting text with layout consideration
                                other_text = page.extract_text(
                                    layout=True,
                                    y_tolerance=kwargs.get("y_tolerance", 3),
                                )
                                if other_text:
                                    text_content += (
                                        f"PAGE {page_num} TEXT: {other_text}\n"
                                    )
                            except Exception:
                                logger.warning(
                                    f"Alternative extraction failed for page {page_num}: {e}"
                                )
                                # Attempt most basic extraction that might work when others fail
                                try:
                                    # Some PDFs might have text in different layers
                                    chars = (
                                        page.chars if hasattr(page, "chars") else None
                                    )
                                    if chars:
                                        page_text_basic = "".join(
                                            [c["text"] for c in chars]
                                        )
                                        if (
                                            page_text_basic.replace("\n", "")
                                            .replace(" ", "")
                                            .strip()
                                        ):
                                            text_content += f"PAGE {page_num} BASIC TEXT: {page_text_basic}\n"
                                except Exception:
                                    logger.error(
                                        f"All text extraction methods failed for page {page_num}"
                                    )
                                    continue

        except PermissionError:
            raise InputError(f"No permission to read PDF file: {pdf_path}")
        except Exception as e:
            # Check for known error messages that indicate common issues
            # PDF encryption related errors usually contain certain keywords
            error_msg = str(e).lower()
            if "password" in error_msg or "encrypt" in error_msg:
                raise InputError(
                    f"Encrypted/password-protected PDF cannot be processed: {pdf_path}"
                )
            elif "permission" in error_msg:
                raise InputError(f"No permission to read PDF file: {pdf_path}")
            elif (
                "corrupt" in error_msg
                or "invalid" in error_msg
                or "not a pdf" in error_msg
            ):
                raise InputError(f"Invalid or corrupted PDF file: {pdf_path}")
            elif isinstance(e, UnicodeDecodeError):
                raise InputError(f"Unicode decode error in PDF file: {e}")
            else:
                # Check if it looks like a pdf plumber specific error (they are often PDFTypes)
                error_type_str = type(e).__name__
                if "PDF" in error_type_str or "pdf" in error_type_str.lower():
                    raise InputError(f"PDF processing error in {pdf_path}: {e}")
                else:
                    raise InputError(f"Failed to process PDF file: {e}")

        if not text_content.strip():
            # PDF extraction resulted in empty content - check for OCR fallback
            logger.info(
                f"Basic PDF extraction failed for {pdf_path}, checking for OCR fallback..."
            )

            ocr_enabled = config.get("ocr", {}).get(
                "enabled", DEFAULT_CONFIG["ocr"]["enabled"]
            )
            threshold_token_ratio = config.get("ocr", {}).get(
                "threshold_token_ratio", DEFAULT_CONFIG["ocr"]["threshold_token_ratio"]
            )

            # Determine if OCR should be attempted based on content quality
            alphanum_ratio = 0.0
            text_stripped = text_content.strip() if text_content else ""

            if text_stripped:
                alphanum_count = sum(c.isalnum() for c in text_stripped)
                alphanum_ratio = alphanum_count / len(text_stripped)

            if ocr_enabled and alphanum_ratio < threshold_token_ratio:
                # Try OCR fallback
                logger.info(
                    f"Attempting OCR fallback for {pdf_path} with token ratio threshold: {threshold_token_ratio}"
                )
                if OCREngine:
                    try:
                        ocr_config = config.get("ocr", {})
                        ocr_engine = OCREngine(config=ocr_config)
                        ocr_text = ocr_engine.extract_text_from_pdf(
                            pdf_path=pdf_path,
                            extraction_mode=extraction_mode,
                            threshold_token_ratio=threshold_token_ratio,
                        )

                        if ocr_text and ocr_text.strip():
                            logger.info(f"OCR fallback successful for {pdf_path}")
                            return ocr_text
                        else:
                            logger.warning(
                                f"OCR fallback also failed to extract meaningful text from {pdf_path}"
                            )
                    except Exception as e:
                        logger.warning(f"OCR fallback failed for {pdf_path}: {e}")
                else:
                    logger.warning(
                        "OCR fallback requested but OCREngine is not available"
                    )

            # If OCR fallback was either not attempted or failed, raise error as before
            raise InputError(
                f"PDF extraction resulted in empty content for file: {pdf_path}"
            )

        # Before returning the content, evaluate if the text quality is low and OCR should be used
        alphanum_count = sum(c.isalnum() for c in text_content)
        alphanum_ratio = alphanum_count / len(text_content) if text_content else 0
        ocr_enabled = config.get("ocr", {}).get(
            "enabled", DEFAULT_CONFIG["ocr"]["enabled"]
        )
        threshold_token_ratio = config.get("ocr", {}).get(
            "threshold_token_ratio", DEFAULT_CONFIG["ocr"]["threshold_token_ratio"]
        )

        if ocr_enabled and alphanum_ratio < threshold_token_ratio:
            logger.info(
                f"Low text quality detected (ratio: {alphanum_ratio:.3f}), attempting OCR fallback"
            )
            if OCREngine:
                try:
                    ocr_config = config.get("ocr", {})
                    ocr_engine = OCREngine(config=ocr_config)
                    ocr_text = ocr_engine.extract_text_from_pdf(
                        pdf_path=pdf_path,
                        extraction_mode=extraction_mode,
                        threshold_token_ratio=threshold_token_ratio,
                    )

                    # If OCR provided better results, use it
                    ocr_alphanum_count = sum(c.isalnum() for c in ocr_text)
                    ocr_alphanum_ratio = (
                        ocr_alphanum_count / len(ocr_text) if ocr_text else 0
                    )

                    if ocr_text and ocr_alphanum_ratio > alphanum_ratio:
                        logger.info(
                            f"OCR fallback improved text quality from {alphanum_ratio:.3f} to {ocr_alphanum_ratio:.3f}"
                        )
                        return ocr_text
                    else:
                        logger.info(
                            "Basic pdfplumber extraction has better quality than OCR fallback"
                        )
                        text_content = text_content  # Use original content but still log the result
                except Exception as e:
                    logger.warning(f"OCR fallback encountered an error: {e}")
            else:
                logger.warning(
                    "OCR fallback was requested but OCREngine is not available"
                )

        logger.info(
            f"Successfully extracted {len(text_content)} characters from {pdf_path}"
        )
        return text_content

    def extract_and_preprocess_pdf(
        self,
        pdf_path: str,
        extraction_mode: str = "basic",
        preprocess_options: Optional[Dict] = None,
        config: Optional[Dict] = None,
    ) -> str:
        """Extract text and apply standard preprocessing like text files."""

        if preprocess_options is None:
            preprocess_options = {}
        if config is None:
            config = {}

        # Extract text from PDF with config
        raw_text = self.extract_text_from_pdf(
            pdf_path, extraction_mode, config=config, **preprocess_options
        )

        # Apply basic preprocessing similar to text files
        # Remove excessive whitespace but preserve paragraph structure
        processed_text = self._normalize_whitespace(raw_text)

        return processed_text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize white space while preserving paragraph structure."""
        import re

        # Replace newlines greater than 3 in a row with 2 newlines
        # This preserves paragraph breaks while removing excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Clean up tab/space indented content, normalize indentation
        lines = text.split("\n")
        normalized_lines = []

        for line in lines:
            stripped_line = line.strip()
            if stripped_line:
                # Preserve original indentation that seems meaningful
                leading_spaces = len(line) - len(line.lstrip())
                if leading_spaces == 0:
                    normalized_lines.append(stripped_line)
                else:
                    # Normalize but keep some indentation for document structure
                    normalized_indented = " " * min(leading_spaces, 4) + stripped_line
                    normalized_lines.append(normalized_indented)
            else:
                normalized_lines.append(stripped_line)  # Keep blank lines

        return "\n".join(normalized_lines)


def detect_pdf_format(file_path: str) -> Dict[str, bool]:
    """Detect characteristics of the PDF to inform extraction strategy.

    Args:
        file_path: Path to the PDF file

    Returns:
        Dictionary indicating detected characteristics
    """
    if pdfplumber is None:
        raise ImportError(
            "pdfplumber is required for PDF format detection. "
            "Install with: pip install pdfplumber"
        )

    # Convert to a consistent return type (all booleans for simplicity)
    # We can derive from page_count > 0 to return True when there are pages
    features = {
        "encrypted": False,
        "has_password": False,
        "page_count_available": False,  # This returns whether there are pages
        "has_tables": False,
        "has_images": False,
        "multi_column": False,
    }

    try:
        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            features["page_count_available"] = page_count > 0

            for page in pdf.pages[:3]:  # Only check first 3 pages for efficiency
                try:
                    # Check for multiple content streams
                    chars = page.chars if hasattr(page, "chars") and page.chars else []

                    if chars:
                        # Check for multiple columns by analyzing character positioning
                        page_width = page.width
                        # Identify characters on the left and right halves
                        left_chars = [c for c in chars if c["x0"] < page_width / 2]
                        right_chars = [c for c in chars if c["x0"] >= page_width / 2]

                        if len(left_chars) > 0 and len(right_chars) > 0:
                            features["multi_column"] = True

                    # Check for tables
                    tables = page.extract_tables()
                    if tables and len(tables) > 0:
                        features["has_tables"] = True

                    # Check for images
                    images = page.images
                    if images and len(images) > 0:
                        features["has_images"] = True

                except Exception:
                    # If we can't analyze a page, continue to next
                    continue
    except Exception as e:
        if "password" in str(e).lower() or "encrypted" in str(e).lower():
            features["encrypted"] = True
            features["has_password"] = True

    return features
