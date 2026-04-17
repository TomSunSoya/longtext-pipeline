"""
Tests for OCR fallback functionality in the longtext pipeline.
"""

import os
import tempfile
from unittest.mock import Mock, patch
import pytest
import httpx
import longtext_pipeline.pipeline.ocr_fallback as ocr_fallback_module
from longtext_pipeline.pipeline.ocr_fallback import (
    OCRAPIClient,
    OCRLocalFallback,
    OCREngine,
    OCRAPIError,
)
from longtext_pipeline.errors import InputError


class TestOCRAPIClient:
    """Test OCR API Client functionality."""

    @patch.dict(os.environ, {"PADDLE_OCR_API_TOKEN": "test-token"})
    def test_init_with_env_token(self):
        """Test initializing OCRAPIClient with token from environment."""
        client = OCRAPIClient()
        assert client.api_token == "test-token"
        assert (
            client.api_url == "https://kbierdt4sav0zbee.aistudio-app.com/layout-parsing"
        )

    def test_init_with_provided_token(self):
        """Test initializing OCRAPIClient with provided API token."""
        client = OCRAPIClient(api_token="provided-token")
        assert client.api_token == "provided-token"

    def test_init_without_token_raises_error(self):
        """Test that initialization without token raises ValueError."""
        with patch.dict(os.environ, {"PADDLE_OCR_API_TOKEN": ""}, clear=True):
            with pytest.raises(ValueError, match="PADDLE_OCR_API_TOKEN must be"):
                OCRAPIClient()

    def test_convert_pdf_to_base64_success(self):
        """Test PDF to base64 conversion."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(b"fake pdf content")
            tmp.flush()
            temp_path = tmp.name

        try:
            client = OCRAPIClient(api_token="test-token")
            base64_result = client.convert_pdf_to_base64(temp_path)
            # Ensure it's a valid base64 string
            import base64

            decoded = base64.b64decode(base64_result.encode("utf-8"))
            assert decoded == b"fake pdf content"
        finally:
            os.remove(temp_path)

    def test_convert_pdf_nonexistent_file(self):
        """Test PDF to base64 conversion with nonexistent file."""
        client = OCRAPIClient(api_token="test-token")
        with pytest.raises(InputError, match="PDF file does not exist"):
            client.convert_pdf_to_base64("/nonexistent/file.pdf")

    @patch("httpx.post")
    def test_submit_to_api_success(self, mock_post):
        """Test successful API submission."""
        # Mock successful response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"result": "extracted text"}
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        client = OCRAPIClient(api_token="test-token")

        result = client.submit_to_api("base64encoded", "general")

        assert result == {"result": "extracted text"}
        mock_post.assert_called_once_with(
            "https://kbierdt4sav0zbee.aistudio-app.com/layout-parsing",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer test-token",
            },
            json={
                "pdf_content": "base64encoded",
                "mode": "general",
                "return_format": "markdown",
            },
            timeout=60.0,
        )

    @patch("httpx.post")
    def test_submit_to_api_timeout(self, mock_post):
        """Test API timeout handling."""
        mock_post.side_effect = httpx.TimeoutException("Request timed out")

        client = OCRAPIClient(api_token="test-token")

        with pytest.raises(OCRAPIError, match="OCR API request timed out"):
            client.submit_to_api("base64encoded", "general")

    @patch("httpx.post")
    def test_submit_to_api_status_error(self, mock_post):
        """Test API error status handling."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response
        mock_post.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Error", request=Mock(), response=mock_response
        )

        client = OCRAPIClient(api_token="test-token")

        with pytest.raises(OCRAPIError, match="OCR API request failed"):
            client.submit_to_api("base64encoded", "general")


class TestOCRLocalFallback:
    """Test local OCR fallback functionality."""

    def test_validate_dependencies_missing_pytesseract(self):
        """Test that missing pytesseract raises ImportError."""
        with patch.object(ocr_fallback_module, "pytesseract", None):
            with patch.object(ocr_fallback_module, "Image", None):
                with patch.object(ocr_fallback_module, "pdf2image", Mock()):
                    fallback = OCRLocalFallback()
                    with pytest.raises(ImportError, match="pytesseract, Pillow"):
                        fallback._validate_dependencies()


class TestOCREngine:
    """Test OCR engine that orchestrates API and local fallback."""

    def test_extract_with_empty_pre_extracted_text(self):
        """Test that OCR proceeds even when initial evaluation returns empty text."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(b"fake pdf content")
            tmp.flush()
            temp_path = tmp.name

        try:
            config = {
                "ocr": {
                    "enabled": True,
                    "paddle_api_token": "test-token",
                    "use_local_fallback": False,
                }
            }

            # Mock the API client
            with patch(
                "longtext_pipeline.pipeline.ocr_fallback.OCRAPIClient"
            ) as mock_api_client_class:
                mock_api_client = Mock()
                mock_api_client.convert_pdf_to_base64.return_value = "base64_encoded"
                mock_api_client.submit_to_api.return_value = {
                    "result": "extracted OCR text!"
                }
                mock_api_client_class.return_value = mock_api_client

                engine = OCREngine(config)
                # Use threshold_token_ratio to force OCR fallback behavior
                result = engine.extract_text_from_pdf(
                    temp_path, threshold_token_ratio=0.01
                )
                assert "extracted OCR text" in result
        finally:
            os.remove(temp_path)
