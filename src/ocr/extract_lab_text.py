"""
ocr/extract_lab_text.py
-----------------------
OCR pipeline for extracting structured data from lab reports (PDF / image).

Supported engines: EasyOCR, Tesseract
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from PIL import Image, UnidentifiedImageError

from src.labs.interpretation import (
    REFERENCE_RANGES,
    interpret_metric,
    normalize_unit,
    parse_numeric_value,
)

try:
    import easyocr

    _EASYOCR_AVAILABLE = True
except ImportError:
    _EASYOCR_AVAILABLE = False

try:
    import pytesseract

    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False

try:
    import fitz

    _FITZ_AVAILABLE = True
except ImportError:
    _FITZ_AVAILABLE = False


@dataclass
class LabMetric:
    """A single extracted lab metric."""

    name: str
    value: float
    unit: str
    raw_text: str
    in_normal_range: Optional[bool] = None
    reference_range: Optional[str] = None
    status: Optional[str] = None  # "Low" | "Normal" | "High"
    interpretation: Optional[str] = None


@dataclass
class LabReport:
    """Full structured output of an OCR extraction run."""

    raw_text: str
    metrics: list[LabMetric] = field(default_factory=list)
    patient_info: dict = field(default_factory=dict)
    source_file: str = ""
    ocr_engine: str = ""
    confidence: float = 0.0


_LAB_VALUE_PATTERN = re.compile(
    r"(?P<name>[A-Za-z][A-Za-z0-9 \-/_()]+?)"
    r"\s*[:\-=]\s*"
    # Supports "17000", "17,000", "8.2", ".8"
    r"(?P<value>[-+]?(?:\d+(?:,\d{3})*(?:[.,]\d+)?|\.\d+))"
    r"\s*(?P<unit>[A-Za-z/%u][A-Za-z0-9/%u.]*)?",
    re.IGNORECASE,
)

_PATIENT_ID_PATTERN = re.compile(
    r"(?:patient\s*(?:id|name|#)?|name|mr\.?|dob|date of birth)\s*[:\-]?\s*([A-Za-z0-9 ,/\-]+)",
    re.IGNORECASE,
)

_VALID_UNIT_PATTERN = re.compile(
    r"^(?:"
    r"g/dl|mg/dl|mmol/l|meq/l|miu/l|iu/l|u/l|ng/ml|pg/ml|"
    r"g/l|mg/l|mm/hr|cells/[a-z0-9%u]+|million/[a-z0-9%u]+|"
    r"x10\^?\d+/[a-z0-9%u]+|%|fl|pg|dl|ul|ml|l"
    r")$",
    re.IGNORECASE,
)

_BLOCKED_METRIC_TOKENS = {
    "localhost",
    "http",
    "https",
    "dashboard",
    "export",
    "print",
    "vision",
    "analysis",
    "review",
    "json",
}


class MedicalOCR:
    """
    Multi-engine OCR extractor for medical lab reports.
    """

    def __init__(
        self,
        engine: str = "easyocr",
        languages: list[str] | None = None,
        gpu: bool = False,
        confidence_threshold: float = 0.5,
    ):
        self.engine = engine.lower()
        self.languages = languages or ["en"]
        self.gpu = gpu
        self.confidence_threshold = confidence_threshold
        self._reader = None
        logger.info("MedicalOCR initialised | engine={}", self.engine)

    def extract(self, source: str | Path | np.ndarray | Image.Image) -> LabReport:
        source_name = str(source) if isinstance(source, (str, Path)) else "<array>"
        logger.info("Extracting from: {}", source_name)

        images = self._load_input(source)
        all_text_blocks: list[tuple[str, float]] = []

        for img in images:
            all_text_blocks.extend(self._run_ocr(img))

        confident = [(txt, conf) for txt, conf in all_text_blocks if conf >= self.confidence_threshold]
        raw_text = "\n".join(txt for txt, _ in confident)
        avg_conf = float(np.mean([c for _, c in confident])) if confident else 0.0

        report = LabReport(
            raw_text=raw_text,
            source_file=source_name,
            ocr_engine=self.engine,
            confidence=avg_conf,
        )
        report.metrics = self._parse_metrics(raw_text)
        report.patient_info = self._extract_patient_info(raw_text)

        logger.success(
            "Extraction complete | metrics={} | avg_confidence={:.2f}",
            len(report.metrics),
            avg_conf,
        )
        return report

    def extract_from_bytes(self, file_bytes: bytes, filename: str = "") -> LabReport:
        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf":
            images = self._pdf_bytes_to_images(file_bytes)
        else:
            try:
                images = [Image.open(io.BytesIO(file_bytes)).convert("RGB")]
            except UnidentifiedImageError as exc:
                raise RuntimeError(
                    "Unsupported or corrupted image upload. Provide a valid PDF, PNG, JPG, or JPEG."
                ) from exc

        all_text_blocks: list[tuple[str, float]] = []
        for img in images:
            all_text_blocks.extend(self._run_ocr(img))

        confident = [(txt, conf) for txt, conf in all_text_blocks if conf >= self.confidence_threshold]
        raw_text = "\n".join(txt for txt, _ in confident)
        avg_conf = float(np.mean([c for _, c in confident])) if confident else 0.0

        report = LabReport(
            raw_text=raw_text,
            source_file=filename,
            ocr_engine=self.engine,
            confidence=avg_conf,
        )
        report.metrics = self._parse_metrics(raw_text)
        report.patient_info = self._extract_patient_info(raw_text)
        return report

    def _load_input(self, source: str | Path | np.ndarray | Image.Image) -> list[Image.Image]:
        if isinstance(source, np.ndarray):
            return [Image.fromarray(source).convert("RGB")]
        if isinstance(source, Image.Image):
            return [source.convert("RGB")]

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Source not found: {path}")

        if path.suffix.lower() == ".pdf":
            return self._pdf_to_images(path)
        return [Image.open(path).convert("RGB")]

    def _pdf_to_images(self, pdf_path: Path) -> list[Image.Image]:
        if not _FITZ_AVAILABLE:
            raise RuntimeError("PyMuPDF (fitz) required for PDF support. pip install PyMuPDF")
        doc = fitz.open(str(pdf_path))
        try:
            images = []
            for page in doc:
                images.append(self._pixmap_to_rgb(page.get_pixmap(dpi=200, alpha=False)))
            return images
        finally:
            doc.close()

    def _pdf_bytes_to_images(self, data: bytes) -> list[Image.Image]:
        if not _FITZ_AVAILABLE:
            raise RuntimeError("PyMuPDF required for PDF support.")
        doc = fitz.open(stream=data, filetype="pdf")
        try:
            images = []
            for page in doc:
                images.append(self._pixmap_to_rgb(page.get_pixmap(dpi=200, alpha=False)))
            return images
        finally:
            doc.close()

    def _pixmap_to_rgb(self, pix) -> Image.Image:
        """
        Convert a PyMuPDF Pixmap into a PIL RGB image.

        PyMuPDF pixmaps can be grayscale (n=1) or RGB/RGBA depending on content.
        The rest of the OCR pipeline expects RGB PIL images.
        """
        width, height = int(pix.width), int(pix.height)
        channels = int(getattr(pix, "n", 3))
        if channels == 1:
            return Image.frombytes("L", (width, height), pix.samples).convert("RGB")
        if channels >= 4:
            return Image.frombytes("RGBA", (width, height), pix.samples).convert("RGB")
        return Image.frombytes("RGB", (width, height), pix.samples)

    def _run_ocr(self, image: Image.Image) -> list[tuple[str, float]]:
        if self.engine == "easyocr":
            return self._run_easyocr(image)
        if self.engine == "tesseract":
            return self._run_tesseract(image)
        raise ValueError(f"Unknown OCR engine: {self.engine}")

    def _run_easyocr(self, image: Image.Image) -> list[tuple[str, float]]:
        if not _EASYOCR_AVAILABLE:
            raise RuntimeError("easyocr not installed. pip install easyocr")
        if self._reader is None:
            logger.info("Loading EasyOCR reader ...")
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)

        results = self._reader.readtext(np.array(image))
        return [(text, conf) for _, text, conf in results]

    def _run_tesseract(self, image: Image.Image) -> list[tuple[str, float]]:
        if not _TESSERACT_AVAILABLE:
            raise RuntimeError("pytesseract not installed. pip install pytesseract")
        data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
            config="--oem 3 --psm 6",
        )
        results = []
        for index, text in enumerate(data["text"]):
            text = text.strip()
            conf = int(data["conf"][index])
            if text and conf > 0:
                results.append((text, conf / 100.0))
        return results

    def _parse_metrics(self, text: str) -> list[LabMetric]:
        """Extract likely lab metrics from raw OCR text."""
        metrics: list[LabMetric] = []
        seen_names: set[str] = set()

        for match in _LAB_VALUE_PATTERN.finditer(text):
            name = match.group("name").strip().lower()
            raw_value = match.group("value")
            unit = normalize_unit((match.group("unit") or "").strip())

            if name in seen_names:
                continue

            value = parse_numeric_value(raw_value)
            if value is None:
                continue

            ref = self._find_reference(name)
            if not self._is_metric_candidate(name=name, value=value, unit=unit, ref=ref):
                continue

            seen_names.add(name)
            metric = LabMetric(
                name=name,
                value=value,
                unit=unit,
                raw_text=match.group(0),
            )

            if ref:
                metric.reference_range = f"{ref['min']} - {ref['max']} {ref['unit']}"
                interpreted = interpret_metric(
                    name=name,
                    value=value,
                    unit=unit,
                    reference_low=ref.get("min"),
                    reference_high=ref.get("max"),
                    reference_unit=ref.get("unit", ""),
                )
                metric.status = interpreted.status
                metric.interpretation = interpreted.interpretation or None
                metric.in_normal_range = interpreted.status == "Normal"
            else:
                interpreted = interpret_metric(name=name, value=value, unit=unit)
                metric.status = interpreted.status
                metric.interpretation = interpreted.interpretation or None
                metric.in_normal_range = interpreted.status == "Normal"

            metrics.append(metric)

        return metrics

    def _find_reference(self, name: str) -> dict | None:
        name_lower = name.lower()
        for key, ref in REFERENCE_RANGES.items():
            if key in name_lower or name_lower in key:
                return ref
        return None

    def _is_metric_candidate(self, name: str, value: float, unit: str, ref: dict | None) -> bool:
        normalized_name = " ".join(name.lower().split())
        normalized_unit = unit.lower().replace(" ", "")

        if any(token in normalized_name for token in _BLOCKED_METRIC_TOKENS):
            return False
        if "://" in normalized_name or "/" in normalized_name or "." in normalized_name:
            return False
        if ref is not None:
            return True
        if not unit:
            return False
        if not _VALID_UNIT_PATTERN.match(normalized_unit):
            return False
        return value >= 0

    def _extract_patient_info(self, text: str) -> dict:
        info: dict = {}
        for match in _PATIENT_ID_PATTERN.finditer(text):
            raw = match.group(1).strip()
            if raw:
                label = match.group(0).split(":")[0].strip().lower()
                info[label] = raw
        return info


def extract_lab_report(
    source: str | Path,
    engine: str = "easyocr",
    gpu: bool = False,
) -> LabReport:
    ocr = MedicalOCR(engine=engine, gpu=gpu)
    return ocr.extract(source)


def format_report_summary(report: LabReport) -> str:
    lines = [
        "=== Lab Report Summary ===",
        f"Source : {report.source_file}",
        f"Engine : {report.ocr_engine}",
        f"Avg OCR confidence: {report.confidence:.1%}",
        "",
        "--- Extracted Metrics ---",
    ]
    for metric in report.metrics:
        flag = ""
        if metric.in_normal_range is False:
            flag = " WARNING OUT OF RANGE"
        elif metric.in_normal_range is True:
            flag = " normal"
        lines.append(f"  {metric.name.title()}: {metric.value} {metric.unit}{flag}")
        if metric.reference_range:
            lines.append(f"    Reference: {metric.reference_range}")
    if report.patient_info:
        lines += ["", "--- Patient Info ---"]
        for key, value in report.patient_info.items():
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)
