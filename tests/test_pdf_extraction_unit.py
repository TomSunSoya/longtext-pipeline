"""Focused unit tests for PDF extraction helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.longtext_pipeline.pipeline.pdf_extraction as pdf_module
from src.longtext_pipeline.errors import InputError


class _DummyPDF:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_pdf_extractor_requires_pdfplumber(monkeypatch):
    monkeypatch.setattr(pdf_module, "pdfplumber", None)

    with pytest.raises(ImportError, match="pdfplumber is required"):
        pdf_module.PDFTextExtractor()


def test_extract_text_from_pdf_handles_basic_text_and_tables(tmp_path, monkeypatch):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    page = SimpleNamespace(
        extract_text=lambda *args, **kwargs: "Hello PDF world",
        extract_tables=lambda: [[["A", None], ["B", "C"]]],
        images=[],
        chars=[],
        width=100,
    )
    fake_pdfplumber = SimpleNamespace(open=lambda path: _DummyPDF([page]))
    monkeypatch.setattr(pdf_module, "pdfplumber", fake_pdfplumber)

    extractor = pdf_module.PDFTextExtractor()
    result = extractor.extract_text_from_pdf(
        str(pdf_path), config={"ocr": {"enabled": False}}, include_tables=True
    )

    assert "--- PAGE 1 ---" in result
    assert "Hello PDF world" in result
    assert "| A |  |" in result
    assert "| B | C |" in result


def test_extract_text_from_pdf_uses_layout_fallback(tmp_path, monkeypatch):
    pdf_path = tmp_path / "fallback.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    def extract_text(*args, **kwargs):
        if kwargs.get("layout"):
            return "Recovered layout text"
        raise RuntimeError("primary extraction failed")

    page = SimpleNamespace(
        extract_text=extract_text,
        extract_tables=lambda: [],
        images=[],
        chars=[],
        width=100,
    )
    monkeypatch.setattr(
        pdf_module,
        "pdfplumber",
        SimpleNamespace(open=lambda path: _DummyPDF([page])),
    )

    extractor = pdf_module.PDFTextExtractor()
    result = extractor.extract_text_from_pdf(
        str(pdf_path), config={"ocr": {"enabled": False}}
    )

    assert "PAGE 1 TEXT: Recovered layout text" in result


def test_extract_text_from_pdf_uses_ocr_when_empty(tmp_path, monkeypatch):
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    page = SimpleNamespace(
        extract_text=lambda *args, **kwargs: "",
        extract_tables=lambda: [],
        images=[],
        chars=[],
        width=100,
    )
    monkeypatch.setattr(
        pdf_module,
        "pdfplumber",
        SimpleNamespace(open=lambda path: _DummyPDF([page])),
    )

    mock_engine = SimpleNamespace(
        extract_text_from_pdf=lambda **kwargs: "OCR recovered text"
    )
    monkeypatch.setattr(pdf_module, "OCREngine", lambda config=None: mock_engine)

    extractor = pdf_module.PDFTextExtractor()
    result = extractor.extract_text_from_pdf(
        str(pdf_path),
        config={"ocr": {"enabled": True, "threshold_token_ratio": 0.3}},
    )

    assert result == "OCR recovered text"


def test_extract_text_from_pdf_prefers_better_ocr_result(tmp_path, monkeypatch):
    pdf_path = tmp_path / "noisy.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    noisy_text = "!!! ???"
    page = SimpleNamespace(
        extract_text=lambda *args, **kwargs: noisy_text,
        extract_tables=lambda: [],
        images=[],
        chars=[],
        width=100,
    )
    monkeypatch.setattr(
        pdf_module,
        "pdfplumber",
        SimpleNamespace(open=lambda path: _DummyPDF([page])),
    )
    monkeypatch.setattr(
        pdf_module,
        "OCREngine",
        lambda config=None: SimpleNamespace(
            extract_text_from_pdf=lambda **kwargs: "Useful OCR output"
        ),
    )

    extractor = pdf_module.PDFTextExtractor()
    result = extractor.extract_text_from_pdf(
        str(pdf_path),
        config={"ocr": {"enabled": True, "threshold_token_ratio": 0.5}},
    )

    assert result == "Useful OCR output"


def test_extract_and_preprocess_pdf_normalizes_whitespace(monkeypatch):
    monkeypatch.setattr(
        pdf_module, "pdfplumber", SimpleNamespace(open=lambda path: _DummyPDF([]))
    )
    extractor = pdf_module.PDFTextExtractor()
    monkeypatch.setattr(
        extractor,
        "extract_text_from_pdf",
        lambda *args, **kwargs: "Line 1\n\n\n    Line 2\n\tLine 3",
    )

    result = extractor.extract_and_preprocess_pdf("fake.pdf")

    assert result == "Line 1\n\n    Line 2\n Line 3"


def test_detect_pdf_format_reports_features(monkeypatch):
    page = SimpleNamespace(
        chars=[{"x0": 10}, {"x0": 80}],
        width=100,
        extract_tables=lambda: [[["cell"]]],
        images=[{"id": "img-1"}],
    )
    monkeypatch.setattr(
        pdf_module,
        "pdfplumber",
        SimpleNamespace(open=lambda path: _DummyPDF([page])),
    )

    features = pdf_module.detect_pdf_format("sample.pdf")

    assert features["page_count_available"] is True
    assert features["multi_column"] is True
    assert features["has_tables"] is True
    assert features["has_images"] is True


def test_detect_pdf_format_marks_encrypted(monkeypatch):
    def raise_password_error(path):
        raise RuntimeError("password required")

    monkeypatch.setattr(
        pdf_module,
        "pdfplumber",
        SimpleNamespace(open=raise_password_error),
    )

    features = pdf_module.detect_pdf_format("encrypted.pdf")

    assert features["encrypted"] is True
    assert features["has_password"] is True


def test_extract_text_from_pdf_maps_invalid_errors(tmp_path, monkeypatch):
    pdf_path = tmp_path / "broken.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    def raise_invalid(path):
        raise RuntimeError("invalid pdf structure")

    monkeypatch.setattr(
        pdf_module,
        "pdfplumber",
        SimpleNamespace(open=raise_invalid),
    )

    extractor = pdf_module.PDFTextExtractor()

    with pytest.raises(InputError, match="Invalid or corrupted PDF file"):
        extractor.extract_text_from_pdf(str(pdf_path))
