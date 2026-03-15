from __future__ import annotations

import pytest


def test_extract_from_bytes_rejects_invalid_image_bytes():
    from src.ocr.extract_lab_text import MedicalOCR

    ocr = MedicalOCR.__new__(MedicalOCR)
    # Ensure the failure happens before OCR engine execution.
    with pytest.raises(RuntimeError, match="Unsupported or corrupted image upload"):
        ocr.extract_from_bytes(b"not-an-image", filename="report.png")


def test_pdf_bytes_to_images_returns_rgb_images():
    from src.ocr.extract_lab_text import MedicalOCR

    try:
        import fitz
    except Exception:
        pytest.skip("PyMuPDF (fitz) not installed")

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hemoglobin: 9.5 g/dL\nWBC: 15000 cells/uL")
    pdf_bytes = doc.tobytes()
    doc.close()

    ocr = MedicalOCR.__new__(MedicalOCR)
    images = ocr._pdf_bytes_to_images(pdf_bytes)
    assert images and len(images) >= 1
    assert all(getattr(img, "mode", "") == "RGB" for img in images)

