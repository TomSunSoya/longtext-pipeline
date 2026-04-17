"""
OCR fallback module using PaddleOCR AI Studio API and local OCR with pytesseract.

This module provides OCR functionality for complex PDFs or image-based documents
where pdfplumber cannot extract adequate text. It uses the PaddleOCR AI Studio API
as primary option and falls back to local OCR with pytesseract.
"""

import base64
import logging
import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional
import httpx
from ..errors import InputError

# Set up logging
logger = logging.getLogger(__name__)

# Conditional imports for fallback OCR
try:
    import pytesseract  # type: ignore[import-untyped]
    from PIL import Image  # type: ignore[import-untyped]
except ImportError:
    pytesseract = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]

try:
    import pdf2image  # type: ignore[import-untyped]
except ImportError:
    pdf2image = None  # type: ignore[assignment]


class OCRAPIError(Exception):
    """Specific exception for OCR API errors."""

    pass


class OCRAPIClient:
    """OCR API client for PaddleOCR AI Studio API service."""

    def __init__(self, api_token: Optional[str] = None, api_url: Optional[str] = None):
        """Initialize OCR API client.

        Args:
            api_token: PaddleOCR AI Studio API token, defaults to environment variable
            api_url: PaddleOCR AI Studio API URL, defaults to environment variable
        """
        self.api_token = api_token or os.getenv("PADDLE_OCR_API_TOKEN")
        if not self.api_token:
            raise ValueError(
                "PADDLE_OCR_API_TOKEN must be set as environment variable or passed as argument"
            )
        self.api_url = (
            api_url
            or os.getenv("PADDLE_OCR_API_URL")
            or "https://kbierdt4sav0zbee.aistudio-app.com/layout-parsing"
        )

    def convert_pdf_to_base64(self, pdf_path: str) -> str:
        """Convert PDF file to base64 encoded string."""
        pdf_file_path = Path(pdf_path)
        if not pdf_file_path.exists():
            raise InputError(f"PDF file does not exist: {pdf_path}")

        try:
            with open(pdf_file_path, "rb") as pdf_file:
                pdf_data = pdf_file.read()
                base64_encoded = base64.b64encode(pdf_data).decode("utf-8")
                return base64_encoded
        except Exception as e:
            raise InputError(f"Failed to encode PDF to base64: {e}")

    def submit_to_api(
        self, base64_pdf: str, mode: Literal["relationship", "general"] = "general"
    ) -> Dict:
        """Submit base64-encoded PDF to OCR API for text extraction.

        Args:
            base64_pdf: Base64-encoded PDF content
            mode: Processing mode retained for compatibility with callers.
                The current Paddle sync endpoint does not use this value.

        Returns:
            Dictionary with parsed response from OCR API
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"token {self.api_token}",
        }

        payload = {
            "file": base64_pdf,
            "fileType": 0,
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useChartRecognition": False,
        }

        try:
            response = httpx.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=60.0,  # 60-second timeout for OCR operations
            )

            response.raise_for_status()  # Raises exception for bad status codes
            result = response.json()
            logger.info(f"OCR API responded with status: {response.status_code}")
            return result  # type: ignore[no-any-return]

        except httpx.TimeoutException:
            error_msg = "OCR API request timed out after 60 seconds"
            logger.error(error_msg)
            raise OCRAPIError(error_msg)
        except httpx.HTTPStatusError as e:
            error_msg = f"OCR API request failed with status {e.response.status_code}: {e.response.text}"
            logger.error(error_msg)
            raise OCRAPIError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during OCR API request: {str(e)}"
            logger.error(error_msg)
            raise OCRAPIError(error_msg)

    @staticmethod
    def extract_markdown_text(result: Dict[str, Any]) -> str:
        """Extract combined markdown text from Paddle layout parsing response."""
        result_root = result.get("result", result)

        if isinstance(result_root, str):
            return result_root

        if not isinstance(result_root, dict):
            return ""

        layout_results = result_root.get("layoutParsingResults", [])
        if isinstance(layout_results, list):
            extracted_chunks: list[str] = []
            for layout_result in layout_results:
                if not isinstance(layout_result, dict):
                    continue

                markdown = layout_result.get("markdown", {})
                if not isinstance(markdown, dict):
                    continue

                text = markdown.get("text", "")
                if isinstance(text, str) and text.strip():
                    extracted_chunks.append(text.strip())

            if extracted_chunks:
                return "\n\n".join(extracted_chunks)

        for fallback_key in ("text", "content"):
            fallback_value = result_root.get(fallback_key, "")
            if isinstance(fallback_value, str):
                return fallback_value

        return ""


class OCRLocalFallback:
    """Local OCR fallback using pytesseract for when API fails."""

    @staticmethod
    def _validate_dependencies():
        """Check if local OCR dependencies are available."""
        missing_deps = []
        if pytesseract is None:
            missing_deps.append("pytesseract")
        if Image is None:
            missing_deps.append("Pillow")
        if pdf2image is None:
            missing_deps.append("pdf2image")

        if missing_deps:
            missing_str = ", ".join(missing_deps)
            raise ImportError(
                f"Missing local OCR dependencies: {missing_str}. "
                "Install with: pip install pytesseract Pillow pdf2image"
            )

    def extract_from_images(self, image_paths: list) -> str:
        """Extract text from a list of image paths."""
        self._validate_dependencies()

        extracted_text = []
        for img_path in image_paths:
            try:
                # Open the image and perform OCR
                img = Image.open(img_path)
                text = pytesseract.image_to_string(img)
                extracted_text.append(text)
            except Exception as e:
                logger.warning(f"Failed to perform OCR on {img_path}: {e}")
                continue

        return "\n".join(extracted_text)

    def extract_from_pdf(self, pdf_path: str) -> str:
        """Convert PDF pages to images and extract text using local OCR."""
        self._validate_dependencies()

        pdf_file_path = Path(pdf_path)
        if not pdf_file_path.exists():
            raise InputError(f"PDF file does not exist: {pdf_path}")

        try:
            # Convert PDF to a list of PIL Images
            pages = pdf2image.convert_from_path(
                pdf_path,
                dpi=200,
                thread_count=4,
                poppler_path=os.getenv("POPPLER_PATH"),  # type: ignore[arg-type]
            )

            # Apply OCR on each page image
            extracted_pages = []
            for i, page in enumerate(pages):
                try:
                    text = pytesseract.image_to_string(page)
                    extracted_pages.append(
                        f"\n--- PAGE {i + 1} ---\n{text}\n-----------\n"
                    )
                except Exception as e:
                    logger.warning(f"Failed to perform OCR on page {i + 1}: {e}")
                    continue

            return "".join(extracted_pages)

        except Exception as e:
            logger.error(f"Failed to convert PDF to images for OCR: {e}")
            raise InputError(f"PDF to image conversion failed: {e}")


class OCREngine:
    """Main OCR engine that coordinates API and local fallback."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the OCR engine.

        Args:
            config: Configuration dictionary for OCR options
        """
        raw_config = config or {}
        normalized_config = raw_config.get("ocr", raw_config)
        self.ocr_config = (
            normalized_config if isinstance(normalized_config, dict) else {}
        )
        self.api_failures_before_fallback = self.ocr_config.get(
            "api_failures_before_fallback", 1
        )
        self.use_local_fallback = self.ocr_config.get("use_local_fallback", True)
        self.current_api_failure_count = 0

    def extract_text_from_pdf(
        self,
        pdf_path: str,
        extraction_mode: str = "general",
        threshold_token_ratio: float = 0.05,
    ) -> str:
        """
        Extract text from PDF using OCR. Tries API first, falls back to local if needed.

        Args:
            pdf_path: Path to the PDF file
            extraction_mode: Processing mode ('general' or 'relationship')
            threshold_token_ratio: Minimum ratio of tokens to character length considered acceptable for text extraction

        Returns:
            Extracted text content as string
        """

        # Before trying OCR, let's verify if basic pdfplumber extraction was inadequate
        # This allows us to determine if OCR is really needed based on token density
        initial_raw_text = self._evaluate_initial_text_density(pdf_path)
        if self._is_acceptable_text_extraction(initial_raw_text, threshold_token_ratio):
            logger.info(
                "Initial PDF text extraction quality is sufficient, skipping OCR fallback."
            )
            return initial_raw_text

        logger.info(
            f"PDF requires OCR as initial extraction is insufficient. Starting OCR with mode '{extraction_mode}'..."
        )

        try:
            # Try API OCR first
            if self.current_api_failure_count < self.api_failures_before_fallback:
                try:
                    api_token = self.ocr_config.get("paddle_api_token") or os.getenv(
                        "PADDLE_OCR_API_TOKEN"
                    )
                    if api_token:
                        api_url = self.ocr_config.get("paddle_api_url")
                        ocr_client = OCRAPIClient(
                            api_token=api_token,
                            api_url=api_url if isinstance(api_url, str) else None,
                        )

                        # Convert PDF to base64
                        base64_pdf = ocr_client.convert_pdf_to_base64(pdf_path)

                        # Send to API
                        result = ocr_client.submit_to_api(
                            base64_pdf=base64_pdf,
                            mode=extraction_mode,  # type: ignore[arg-type]
                        )

                        extracted_text = ocr_client.extract_markdown_text(result)

                        # Validate that API OCR produced meaningful text
                        if self._is_acceptable_ocr_result(extracted_text):
                            return extracted_text  # type: ignore[no-any-return]
                        else:
                            logger.warning(
                                "API OCR result is not adequate, proceeding to local fallback"
                            )

                except OCRAPIError as e:
                    self.current_api_failure_count += 1
                    logger.warning(
                        f"API OCR failed ({self.current_api_failure_count}/{self.api_failures_before_fallback}): {e}"
                    )

                    # If we have not exceeded failure threshold, try again on next attempt
                    if (
                        self.current_api_failure_count
                        >= self.api_failures_before_fallback
                        and self.use_local_fallback
                    ):
                        logger.info(
                            "Switching to local OCR fallback due to API failures"
                        )

            # Use local OCR if API failed or disabled
            if self.use_local_fallback:
                fallback_engine = OCRLocalFallback()
                local_result = fallback_engine.extract_from_pdf(pdf_path)

                if self._is_acceptable_ocr_result(local_result):
                    logger.info("Local OCR fallback completed successfully")
                    return local_result
                else:
                    logger.warning(
                        "Local OCR fallback did not produce adequate text extraction"
                    )
                    raise InputError(
                        f"Both OCR attempts failed to extract meaningful text from PDF: {pdf_path}"
                    )
            else:
                raise InputError(
                    f"OCR API failed and local fallback is disabled: {pdf_path}"
                )

        except Exception as e:
            error_msg = f"OCR processing failed for {pdf_path}: {e}"
            logger.error(error_msg)
            raise InputError(error_msg)

    def _evaluate_initial_text_density(self, pdf_path: str) -> str:
        """Extract text using basic pdfplumber to evaluate if OCR is needed."""
        try:
            import pdfplumber

            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            logger.debug(
                f"Error during initial density evaluation with pdfplumber: {e}"
            )
            return ""  # Return empty string to indicate text extraction isn't reliable

    def _is_acceptable_text_extraction(
        self, text: str, threshold_token_ratio: float = 0.05
    ) -> bool:
        """Determine if extracted text from pdfplumber is adequate without OCR.

        Args:
            text: Text to evaluate
            threshold_token_ratio: Minimum ratio of alphanumeric tokens to total characters
        """
        if not text or not text.strip():
            return False

        # Simple heuristic: count alphanumeric characters vs total
        text_stripped = text.strip()
        if len(text_stripped) == 0:
            return False

        alphanum_count = sum(c.isalnum() for c in text_stripped)
        ratio = alphanum_count / len(text_stripped)

        return ratio >= threshold_token_ratio

    def _is_acceptable_ocr_result(self, text: str) -> bool:
        """Validate that OCR result contains meaningful text content."""
        if not text or not text.strip():
            return False

        # Additional validation for OCR quality
        text_stripped = text.strip()
        if len(text_stripped) < 5:  # Too little content
            return False

        # Check if it's mostly garbage
        alphanum_count = sum(c.isalnum() for c in text_stripped)
        ratio = alphanum_count / len(text_stripped)

        if ratio < 0.15:  # Less than 15% alphanumeric chars - probably OCR errors
            return False

        return True
