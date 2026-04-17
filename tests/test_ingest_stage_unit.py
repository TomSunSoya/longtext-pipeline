from pathlib import Path

import pytest

from src.longtext_pipeline.errors import InputError
from src.longtext_pipeline.manifest import ManifestManager
from src.longtext_pipeline.pipeline.ingest import IngestStage


def _make_stage_and_manifest(input_file: Path) -> tuple[IngestStage, object]:
    manager = ManifestManager()
    stage = IngestStage(manifest_manager=manager)
    manifest = manager.create_manifest(str(input_file), "a" * 64)
    return stage, manifest


def test_pdf_dependency_import_error_is_chained(tmp_path, monkeypatch):
    input_file = tmp_path / "input.pdf"
    input_file.write_text("pdf", encoding="utf-8")
    stage, manifest = _make_stage_and_manifest(input_file)

    class _MissingPDFExtractor:
        def __init__(self):
            raise ImportError("pdfplumber is required")

    monkeypatch.setattr(
        "src.longtext_pipeline.pipeline.ingest.PDFTextExtractor",
        _MissingPDFExtractor,
    )

    with pytest.raises(
        InputError, match="PDF support requires additional dependencies"
    ) as exc:
        stage.run(str(input_file), {}, manifest)

    assert isinstance(exc.value.__cause__, ImportError)


def test_docx_dependency_import_error_is_chained(tmp_path, monkeypatch):
    input_file = tmp_path / "input.docx"
    input_file.write_text("docx", encoding="utf-8")
    stage, manifest = _make_stage_and_manifest(input_file)

    class _MissingDOCXExtractor:
        def __init__(self):
            raise ImportError("python-docx is required")

    monkeypatch.setattr(
        "src.longtext_pipeline.pipeline.ingest.DOCXTextExtractor",
        _MissingDOCXExtractor,
    )

    with pytest.raises(
        InputError, match="DOCX support requires additional dependencies"
    ) as exc:
        stage.run(str(input_file), {}, manifest)

    assert isinstance(exc.value.__cause__, ImportError)
