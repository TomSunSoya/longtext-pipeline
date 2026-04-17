"""Focused unit tests for DOCX extraction helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.longtext_pipeline.pipeline.docx_extraction as docx_module
from src.longtext_pipeline.errors import InputError


def _make_run(
    text: str,
    *,
    bold: bool = False,
    italic: bool = False,
    underline: bool = False,
    has_image: bool = False,
):
    element = SimpleNamespace(graphicData=True) if has_image else SimpleNamespace()
    return SimpleNamespace(
        text=text,
        bold=bold,
        italic=italic,
        underline=underline,
        _element=element,
    )


def _make_paragraph(text: str, *, style_name: str = "Normal", runs=None):
    return SimpleNamespace(
        text=text,
        style=SimpleNamespace(name=style_name),
        runs=runs if runs is not None else [_make_run(text)],
    )


def _make_table(rows):
    return SimpleNamespace(
        rows=[
            SimpleNamespace(cells=[SimpleNamespace(text=cell) for cell in row])
            for row in rows
        ]
    )


def test_docx_extractor_requires_python_docx(monkeypatch):
    monkeypatch.setattr(docx_module, "DOCX_AVAILABLE", False)

    with pytest.raises(ImportError, match="python-docx is required"):
        docx_module.DOCXTextExtractor()


def test_extract_text_from_docx_validates_missing_file(monkeypatch):
    monkeypatch.setattr(docx_module, "DOCX_AVAILABLE", True)

    extractor = docx_module.DOCXTextExtractor()
    with pytest.raises(InputError, match="DOCX file does not exist"):
        extractor.extract_text_from_docx("missing.docx")


def test_extract_text_from_docx_handles_formatted_content(tmp_path, monkeypatch):
    docx_path = tmp_path / "sample.docx"
    docx_path.write_bytes(b"DOCX")

    fake_doc = SimpleNamespace(
        paragraphs=[
            _make_paragraph("Heading Title", style_name="Heading 1"),
            _make_paragraph(
                "Formatted paragraph",
                runs=[
                    _make_run("Bold", bold=True),
                    _make_run("Italic", italic=True),
                    _make_run("Under", underline=True),
                ],
            ),
        ],
        tables=[_make_table([["Col1", "Col2"], ["A", "B"]])],
    )
    monkeypatch.setattr(docx_module, "DOCX_AVAILABLE", True)
    monkeypatch.setattr(
        docx_module,
        "docx",
        SimpleNamespace(Document=lambda path: fake_doc),
    )

    extractor = docx_module.DOCXTextExtractor()
    result = extractor.extract_text_from_docx(
        str(docx_path), extraction_mode="formatted"
    )

    assert "# Heading Title" in result
    assert "**Bold**" in result
    assert "*Italic*" in result
    assert "_Under_" in result
    assert "TABLE:" in result


def test_extract_and_preprocess_docx_normalizes_whitespace(monkeypatch):
    monkeypatch.setattr(docx_module, "DOCX_AVAILABLE", True)
    extractor = docx_module.DOCXTextExtractor()
    monkeypatch.setattr(
        extractor,
        "extract_text_from_docx",
        lambda *args, **kwargs: "Line 1\n\n\n  Line 2  \n\n\nLine 3",
    )

    result = extractor.extract_and_preprocess_docx("fake.docx")

    assert result == "Line 1\n\nLine 2\n\nLine 3"


def test_detect_docx_format_reports_features(monkeypatch):
    fake_doc = SimpleNamespace(
        paragraphs=[
            _make_paragraph("Heading One", style_name="Heading 1"),
            _make_paragraph("Body", runs=[_make_run("Body", bold=True)]),
        ],
        tables=[_make_table([["a"]])],
    )
    monkeypatch.setattr(docx_module, "DOCX_AVAILABLE", True)
    monkeypatch.setattr(
        docx_module,
        "docx",
        SimpleNamespace(Document=lambda path: fake_doc),
    )

    features = docx_module.detect_docx_format("sample.docx")

    assert features["valid_docx"] is True
    assert features["has_tables"] is True
    assert features["has_headings"] is True
    assert features["has_formatted_text"] is True
    assert features["page_count_estimate"] >= 1


def test_detect_docx_format_handles_invalid_archive(monkeypatch):
    monkeypatch.setattr(docx_module, "DOCX_AVAILABLE", True)
    monkeypatch.setattr(
        docx_module,
        "docx",
        SimpleNamespace(
            Document=lambda path: (_ for _ in ()).throw(RuntimeError("not a zip file"))
        ),
    )

    features = docx_module.detect_docx_format("bad.docx")

    assert features["valid_docx"] is False


def test_get_document_structure_counts_headings_tables_images(monkeypatch):
    fake_doc = SimpleNamespace(
        paragraphs=[
            _make_paragraph("Heading One", style_name="Heading 1"),
            _make_paragraph("Image paragraph", runs=[_make_run("img", has_image=True)]),
        ],
        tables=[_make_table([["a", "b"]])],
        sections=[1, 2],
    )
    monkeypatch.setattr(docx_module, "DOCX_AVAILABLE", True)
    monkeypatch.setattr(
        docx_module,
        "docx",
        SimpleNamespace(Document=lambda path: fake_doc),
    )

    structure = docx_module.get_document_structure("sample.docx")

    assert structure == {
        "paragraphs": 2,
        "tables": 1,
        "sections": 2,
        "images": 1,
        "headings": 1,
    }


def test_extract_text_from_docx_maps_invalid_docx_errors(tmp_path, monkeypatch):
    docx_path = tmp_path / "broken.docx"
    docx_path.write_bytes(b"DOCX")

    monkeypatch.setattr(docx_module, "DOCX_AVAILABLE", True)
    monkeypatch.setattr(
        docx_module,
        "docx",
        SimpleNamespace(
            Document=lambda path: (_ for _ in ()).throw(RuntimeError("corrupt archive"))
        ),
    )

    extractor = docx_module.DOCXTextExtractor()

    with pytest.raises(InputError, match="Invalid or corrupted DOCX file"):
        extractor.extract_text_from_docx(str(docx_path))
