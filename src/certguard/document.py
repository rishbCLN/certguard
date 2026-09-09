from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from certguard.models import ExtractionResult

URL_PATTERN = re.compile(r"https?://[^\s<>\]\[\"']+", re.IGNORECASE)
GENERIC_ID_PATTERN = re.compile(
    r"(?:certificate|credential|verification)(?:\s+(?:id|code|number|no))?\s*[:#-]\s*"
    r"([A-Za-z0-9][A-Za-z0-9._-]{5,63})",
    re.IGNORECASE,
)


def load_document(path: Path, dpi: int = 200) -> tuple[np.ndarray, int]:
    if path.suffix.casefold() == ".pdf":
        try:
            import fitz
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise RuntimeError("PDF support requires PyMuPDF") from exc
        try:
            with fitz.open(path) as document:
                if not document.page_count:
                    raise ValueError("PDF has no pages")
                page = document[0]
                pixmap = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
                image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                    pixmap.height, pixmap.width, pixmap.n
                )
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), document.page_count
        except ValueError:
            raise
        except (fitz.FileDataError, OSError, RuntimeError) as exc:
            raise ValueError(f"Unsupported or unreadable PDF: {path}") from exc
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unsupported or unreadable document: {path}")
    return image, 1


def extract_document(image: np.ndarray, page_count: int = 1) -> ExtractionResult:
    errors: list[str] = []
    text = ""
    confidence: float | None = None
    try:
        import pytesseract

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data = pytesseract.image_to_data(
            Image.fromarray(rgb), output_type=pytesseract.Output.DICT, config="--psm 6"
        )
        words = [word.strip() for word in data["text"] if word.strip()]
        text = " ".join(words)
        values = [float(value) for value in data["conf"] if float(value) >= 0]
        confidence = round(sum(values) / len(values) / 100, 3) if values else None
    except Exception as exc:
        errors.append(f"OCR unavailable: {type(exc).__name__}: {exc}")

    qr_values: list[str] = []
    try:
        detector = cv2.QRCodeDetector()
        found, decoded, _points, _ = detector.detectAndDecodeMulti(image)
        if found:
            qr_values.extend(value.strip() for value in decoded if value.strip())
        else:
            value, _points, _ = detector.detectAndDecode(image)
            if value.strip():
                qr_values.append(value.strip())
    except cv2.error as exc:
        errors.append(f"QR decode failed: {exc}")

    combined = "\n".join([text, *qr_values])
    urls = list(dict.fromkeys(match.rstrip(".,);}") for match in URL_PATTERN.findall(combined)))
    certificate_ids = list(dict.fromkeys(GENERIC_ID_PATTERN.findall(combined)))
    return ExtractionResult(
        text=text,
        certificate_ids=certificate_ids,
        urls=urls,
        qr_values=qr_values,
        ocr_confidence=confidence,
        page_count=page_count,
        errors=errors,
    )
