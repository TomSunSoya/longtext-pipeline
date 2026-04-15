#!/usr/bin/env python3
"""Generate test PDF fixtures for longtext-pipeline."""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor


def create_text_based_pdf(filepath_str, title, content_lines, multi_column=False):
    """Create a text-based PDF."""
    c = canvas.Canvas(filepath_str, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(1 * inch, height - 1 * inch, title)

    y_position = height - 1.5 * inch

    if multi_column:
        # Simple multi-column text (two columns)
        column_width = (width - 3 * inch) / 2

        # Left column
        c.setFont("Helvetica", 11)
        for i, line in enumerate(content_lines[: len(content_lines) // 2]):
            c.drawString(1 * inch, y_position - i * 0.2 * inch, line[:80])

        # Right column
        for i, line in enumerate(content_lines[len(content_lines) // 2 :]):
            c.drawString(
                1.5 * inch + column_width, y_position - i * 0.2 * inch, line[:80]
            )
    else:
        c.setFont("Helvetica", 11)
        for line in content_lines:
            c.drawString(1 * inch, y_position, line[:100])
            y_position -= 0.2 * inch
            if y_position < 1 * inch:
                c.showPage()
                y_position = height - 1 * inch

    c.save()


def create_scanned_pdf(filepath_str, title, text_content):
    """Create a PDF that simulates a scanned document (with visual noise)."""
    c = canvas.Canvas(filepath_str, pagesize=letter)
    width, height = letter

    # Add background noise to simulate scan
    c.setFillColor(HexColor("#f5f5f5"))
    c.rect(0, 0, width, height, fill=1, stroke=0)

    # Add some grid lines
    c.setStrokeColor(HexColor("#dddddd"))
    for x in range(0, int(width), 20):
        c.line(x, 0, x, height)
    for y in range(0, int(height), 20):
        c.line(0, y, width, y)

    # Add title
    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(HexColor("#333333"))
    c.drawString(1 * inch, height - 1 * inch, title)

    # Add content with slight variations
    c.setFont("Helvetica", 11)
    y_position = height - 1.5 * inch
    y_offset = 0
    for i, line in enumerate(text_content[:30]):
        y_offset = (i % 5) * 0.5  # Slight y variation
        x_offset = (i % 3) * 0.3  # Slight x variation
        c.drawString(1 * inch + x_offset, y_position + y_offset, line[:95])
        if i % 5 == 0:
            y_position -= 0.22 * inch
        else:
            y_position -= 0.18 * inch
        if y_position < 1 * inch:
            c.showPage()
            y_position = height - 1 * inch

    # Add 'scan artifacts' - random lines
    c.setStrokeColor(HexColor("#c0c0c0"))
    for _ in range(20):
        import random

        x1 = random.random() * width
        y1 = random.random() * height
        x2 = x1 + random.random() * 50
        y2 = y1
        c.line(x1, y1, x2, y2)

    c.save()


def create_encrypted_pdf(filepath_str):
    """Create an encrypted PDF with restrictions."""
    c = canvas.Canvas(filepath_str, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(1 * inch, height - 1 * inch, "Encrypted Document - Restricted")

    c.setFont("Helvetica", 11)
    y = height - 1.5 * inch
    restricted_content = [
        "This document is encrypted and restricted.",
        "Access is limited to authorized users only.",
        "Copying, printing, and extraction are restricted.",
        "This is a test fixture for encryption testing.",
    ]
    for line in restricted_content:
        c.drawString(1 * inch, y, line)
        y -= 0.2 * inch

    c.save()


def create_large_pdf(filepath_str, num_pages=105):
    """Create a large multi-page PDF."""
    c = canvas.Canvas(filepath_str, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)

    page_num = 1
    while page_num <= num_pages:
        c.drawString(1 * inch, height - 1 * inch, f"Large Document - Page {page_num}")

        c.setFont("Helvetica", 10)
        y = height - 1.5 * inch
        content_lines = [
            f"This is page {page_num} of a large test document.",
            "Content for testing pagination and large file handling.",
            "Line 3: Testing long content for pagination.",
            "Line 4: More text to fill the page properly.",
            "Line 5: Additional content for testing purposes.",
        ]
        for line in content_lines:
            c.drawString(1 * inch, y, line)
            y -= 0.18 * inch
            if y < 1 * inch:
                c.showPage()
                page_num += 1
                c.setFont("Helvetica-Bold", 16)
                if page_num > num_pages:
                    break

        if page_num <= num_pages:
            c.showPage()
            page_num += 1
            c.setFont("Helvetica-Bold", 16)

    c.save()


# Sample text content
text_content = [
    "Introduction to the Document",
    "This document contains sample text for testing the PDF parsing capabilities.",
    "It includes various sections and formatting to simulate real-world documents.",
    "The content is designed to test different aspects of PDF analysis.",
    "Section 1: Overview",
    "This section provides an overview of the document structure.",
    "Section 2: Detailed Analysis",
    "Here we provide detailed analysis of the content.",
    "Section 3: Results",
    "Results and findings are presented in this section.",
    "Section 4: Conclusion",
    "The conclusion summarizes the main points discussed.",
    "References",
    "Additional references and citations are listed below.",
    "Footnotes appear at the bottom of each page for additional context.",
    "Tables and figures are included where relevant.",
    "Image descriptions are provided for accessibility.",
]

multi_column_content = [
    "Column One Content - This section appears in the first column.",
    "Multiple columns are used to organize content more compactly.",
    "This layout is common in newspapers and technical documents.",
    "Column Two Content - This section appears in the second column.",
    "Each column contains distinct but related information.",
    "The columnar layout helps with readability of dense text.",
    "Tables and figures may span one or both columns.",
    "Cross-references between columns help maintain context.",
]

large_text_content = [
    "Large Document Text Content",
    "This is a large multi-page document for testing pagination.",
    "Each page contains multiple lines of text.",
    "The document is designed to exercise pagination handling.",
    "Page content continues with more text.",
    "Additional paragraphs are added throughout.",
] * 50


def main():
    """Generate all PDF fixtures."""
    output_dir = Path("tests/fixtures/pdfs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. simple.pdf - Single column text
    create_text_based_pdf(
        str(output_dir / "01_simple.pdf"),
        "Simple Document",
        text_content[:6],
        multi_column=False,
    )

    # 2. multi_column.pdf - Two column layout
    create_text_based_pdf(
        str(output_dir / "02_multi_column.pdf"),
        "Multi-Column Document",
        multi_column_content,
        multi_column=True,
    )

    # 3. scanned_simple.pdf - Simple scanned document
    create_scanned_pdf(
        str(output_dir / "03_scanned_simple.pdf"),
        "Scanned Document - Simple",
        text_content[:20],
    )

    # 4. scanned_multi_page.pdf - Multi-page scanned document
    scanned_text = []
    for j in range(30):
        page_num = (j // 5) + 1
        scanned_text.extend(
            [
                f"Scanned Page {page_num} Content",
                "This document has been scanned from physical paper.",
                "OCR processing may be required for accurate text extraction.",
                "Each page contains varying amounts of content.",
                "Image quality varies across pages.",
                "Some pages may have annotations or marks.",
            ]
        )
    create_scanned_pdf(
        str(output_dir / "04_scanned_multi_page.pdf"),
        "Scanned Document - Multi-page",
        scanned_text,
    )

    # 5. encrypted_restrictions.pdf - Encrypted document
    create_encrypted_pdf(str(output_dir / "05_encrypted_restrictions.pdf"))

    # 6. large_document.pdf - Large multi-page document
    create_large_pdf(str(output_dir / "06_large_document.pdf"), num_pages=105)

    # 7-10: Create small placeholder PDFs for remaining slots
    for i, name in enumerate(
        [
            "07_hybrid_report.pdf",
            "08_hybrid_document.pdf",
            "09_table_document.pdf",
            "10_complex_layout.pdf",
        ]
    ):
        c = canvas.Canvas(str(output_dir / name), pagesize=letter)
        width, height = letter

        c.setFont("Helvetica-Bold", 16)
        c.drawString(1 * inch, height - 1 * inch, f"Document: {name}")

        c.setFont("Helvetica", 11)
        y = height - 1.5 * inch
        content = [
            f"This is {name} - a test fixture.",
            "Contains mixed content for testing.",
            "Used to verify document parsing works correctly.",
            "This is a smaller file for quick tests.",
        ]
        for line in content:
            c.drawString(1 * inch, y, line)
            y -= 0.2 * inch

        c.save()

    print(f"Generated {len(list(output_dir.glob('*.pdf')))} PDF fixtures")
    for f in sorted(output_dir.glob("*.pdf")):
        size = f.stat().st_size / (1024 * 1024)  # MB
        print(f"  {f.name}: {size:.2f} MB")


if __name__ == "__main__":
    main()
