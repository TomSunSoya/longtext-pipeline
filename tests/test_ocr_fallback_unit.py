"""Additional OCR fallback unit tests for branch coverage."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import src.longtext_pipeline.pipeline.ocr_fallback as ocr_module
from src.longtext_pipeline.errors import InputError


class _DummyPDF:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_local_fallback_extract_from_images_skips_failures(monkeypatch):
    monkeypatch.setattr(
        ocr_module,
        "Image",
        SimpleNamespace(open=lambda path: object()),
    )
    calls = {"count": 0}

    def image_to_string(_img):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("ocr failed")
        return "recognized text"

    monkeypatch.setattr(
        ocr_module,
        "pytesseract",
        SimpleNamespace(image_to_string=image_to_string),
    )
    monkeypatch.setattr(ocr_module, "pdf2image", object())

    result = ocr_module.OCRLocalFallback().extract_from_images(["a.png", "b.png"])

    assert result == "recognized text"


def test_local_fallback_extract_from_pdf_success(tmp_path, monkeypatch):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(
        ocr_module,
        "pdf2image",
        SimpleNamespace(convert_from_path=lambda *args, **kwargs: [object(), object()]),
    )
    monkeypatch.setattr(
        ocr_module,
        "pytesseract",
        SimpleNamespace(image_to_string=lambda page: "page text"),
    )
    monkeypatch.setattr(ocr_module, "Image", object())

    result = ocr_module.OCRLocalFallback().extract_from_pdf(str(pdf_path))

    assert "--- PAGE 1 ---" in result
    assert "--- PAGE 2 ---" in result


def test_ocr_engine_uses_local_fallback_after_api_failure(tmp_path, monkeypatch):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    engine = ocr_module.OCREngine(
        {
            "ocr": {
                "paddle_api_token": "token",
                "api_failures_before_fallback": 1,
                "use_local_fallback": True,
            }
        }
    )
    monkeypatch.setattr(engine, "_evaluate_initial_text_density", lambda path: "")
    monkeypatch.setattr(
        engine, "_is_acceptable_text_extraction", lambda text, ratio: False
    )
    monkeypatch.setattr(
        ocr_module,
        "OCRAPIClient",
        lambda api_token=None: SimpleNamespace(
            convert_pdf_to_base64=lambda path: "b64",
            submit_to_api=lambda **kwargs: (_ for _ in ()).throw(
                ocr_module.OCRAPIError("api unavailable")
            ),
        ),
    )
    monkeypatch.setattr(
        ocr_module,
        "OCRLocalFallback",
        lambda: SimpleNamespace(extract_from_pdf=lambda path: "Local OCR result"),
    )

    result = engine.extract_text_from_pdf(str(pdf_path))

    assert result == "Local OCR result"
    assert engine.current_api_failure_count == 1


def test_ocr_engine_raises_when_local_fallback_disabled(tmp_path, monkeypatch):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    engine = ocr_module.OCREngine(
        {
            "ocr": {
                "paddle_api_token": "token",
                "api_failures_before_fallback": 1,
                "use_local_fallback": False,
            }
        }
    )
    monkeypatch.setattr(engine, "_evaluate_initial_text_density", lambda path: "")
    monkeypatch.setattr(
        engine, "_is_acceptable_text_extraction", lambda text, ratio: False
    )
    monkeypatch.setattr(
        ocr_module,
        "OCRAPIClient",
        lambda api_token=None: SimpleNamespace(
            convert_pdf_to_base64=lambda path: "b64",
            submit_to_api=lambda **kwargs: (_ for _ in ()).throw(
                ocr_module.OCRAPIError("api unavailable")
            ),
        ),
    )

    with pytest.raises(InputError, match="OCR processing failed"):
        engine.extract_text_from_pdf(str(pdf_path))


def test_evaluate_initial_text_density_reads_pdfplumber(monkeypatch):
    page_one = SimpleNamespace(extract_text=lambda: "first page")
    page_two = SimpleNamespace(extract_text=lambda: "second page")
    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda path: _DummyPDF([page_one, page_two])),
    )

    engine = ocr_module.OCREngine()
    result = engine._evaluate_initial_text_density("sample.pdf")

    assert "first page" in result
    assert "second page" in result


def test_ocr_quality_heuristics():
    engine = ocr_module.OCREngine()

    assert engine._is_acceptable_text_extraction("Readable text 123", 0.1) is True
    assert engine._is_acceptable_text_extraction("!!!", 0.5) is False
    assert engine._is_acceptable_ocr_result("Good OCR text") is True
    assert engine._is_acceptable_ocr_result("??") is False
