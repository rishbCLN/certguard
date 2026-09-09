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
STANDALONE_CODE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9/])([A-Za-z0-9][A-Za-z0-9._-]{7,63})(?![A-Za-z0-9._-])",
)
UUID_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(?![A-Za-z0-9])",
    re.IGNORECASE,
)
NUMERIC_ID_PATTERN = re.compile(r"(?<![0-9])([0-9]{8,20})(?![0-9])")

URL_REPLACEMENTS = (
    (" ", ""),
    ("\n", ""),
    ("\r", ""),
    ("..", "."),
    ("coursera org", "coursera.org"),
    ("hackerrank com", "hackerrank.com"),
    ("edx org", "edx.org"),
    ("nptel ac in", "nptel.ac.in"),
    ("unstop com", "unstop.com"),
    ("devfolio co", "devfolio.co"),
    ("httpS", "https"),
    ("httpps", "https"),
)

MAX_SOURCE_BYTES = 50 * 1024 * 1024
MAX_PDF_PAGES = 20
MAX_RENDER_DPI = 400
MAX_IMAGE_PIXELS = 30_000_000
OCR_TIMEOUT_SECONDS = 60


class DocumentTooLargeError(ValueError):
    """Raised when an untrusted document exceeds configured resource limits."""


def load_document(path: Path, dpi: int = 200) -> tuple[np.ndarray, int]:
    if dpi > MAX_RENDER_DPI:
        raise DocumentTooLargeError(f"Render DPI {dpi} exceeds the maximum of {MAX_RENDER_DPI}")
    if path.is_file() and path.stat().st_size > MAX_SOURCE_BYTES:
        raise DocumentTooLargeError(
            f"Document exceeds the maximum supported size of {MAX_SOURCE_BYTES} bytes"
        )
    if path.suffix.casefold() == ".pdf":
        try:
            import fitz
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise RuntimeError("PDF support requires PyMuPDF") from exc
        try:
            with fitz.open(path) as document:
                if not document.page_count:
                    raise ValueError("PDF has no pages")
                if document.page_count > MAX_PDF_PAGES:
                    raise DocumentTooLargeError(
                        f"PDF has {document.page_count} pages; "
                        f"the maximum supported is {MAX_PDF_PAGES}"
                    )
                page = document[0]
                scale = dpi / 72
                width = page.rect.width * scale
                height = page.rect.height * scale
                if width * height > MAX_IMAGE_PIXELS:
                    raise DocumentTooLargeError(
                        "Rendered page dimensions exceed the maximum supported pixel budget"
                    )
                pixmap = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
                pixels = pixmap.width * pixmap.height
                if pixels > MAX_IMAGE_PIXELS:
                    raise DocumentTooLargeError(
                        "Rendered page dimensions exceed the maximum supported pixel budget"
                    )
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
    if image.shape[0] * image.shape[1] > MAX_IMAGE_PIXELS:
        raise DocumentTooLargeError(
            "Image dimensions exceed the maximum supported pixel budget"
        )
    return image, 1


def extract_document(image: np.ndarray, page_count: int = 1) -> ExtractionResult:
    errors: list[str] = []
    text = ""
    confidence: float | None = None
    try:
        import pytesseract

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data = pytesseract.image_to_data(
            Image.fromarray(rgb),
            output_type=pytesseract.Output.DICT,
            config="--psm 6",
            timeout=OCR_TIMEOUT_SECONDS,
        )
        words = [word.strip() for word in data["text"] if word.strip()]
        text = " ".join(words)
        values = [float(value) for value in data["conf"] if float(value) >= 0]
        confidence = round(sum(values) / len(values) / 100, 3) if values else None
    except Exception as exc:
        errors.append(f"OCR unavailable: {type(exc).__name__}: {exc}")

    qr_values: list[str] = []
    try:
        qr_values.extend(_decode_qr(image))
    except cv2.error as exc:
        errors.append(f"QR decode failed: {exc}")

    combined = "\n".join([text, *qr_values])
    urls = _extract_urls(combined)
    certificate_ids = _extract_certificate_ids(combined, qr_values)
    return ExtractionResult(
        text=text,
        certificate_ids=certificate_ids,
        urls=urls,
        qr_values=qr_values,
        ocr_confidence=confidence,
        page_count=page_count,
        errors=errors,
    )


def _decode_qr(image: np.ndarray) -> list[str]:
    values: list[str] = []
    detector = cv2.QRCodeDetector()
    found, decoded, _points, _ = detector.detectAndDecodeMulti(image)
    if found:
        values.extend(value.strip() for value in decoded if value.strip())
    else:
        value, _points, _ = detector.detectAndDecode(image)
        if value.strip():
            values.append(value.strip())
    if values:
        return values
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for preprocess in (
        lambda g: cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        lambda g: cv2.adaptiveThreshold(
            g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10
        ),
        lambda g: cv2.GaussianBlur(g, (3, 3), 0),
        lambda g: cv2.convertScaleAbs(g, alpha=1.5, beta=0),
    ):
        processed = preprocess(gray)
        found, decoded, _points, _ = detector.detectAndDecodeMulti(processed)
        if found:
            values.extend(value.strip() for value in decoded if value.strip())
        else:
            value, _points, _ = detector.detectAndDecode(processed)
            if value.strip():
                values.append(value.strip())
        if values:
            break
    return values


def _extract_urls(text: str) -> list[str]:
    raw_urls = URL_PATTERN.findall(text)
    cleaned: list[str] = []
    for url in raw_urls:
        url = url.rstrip(".,);}")
        for old, new in URL_REPLACEMENTS:
            url = url.replace(old, new)
        if url.startswith("http://"):
            url = "https://" + url[len("http://") :]
        cleaned.append(url)
    return list(dict.fromkeys(cleaned))


def _extract_certificate_ids(text: str, qr_values: list[str]) -> list[str]:
    ids: list[str] = []
    ids.extend(UUID_PATTERN.findall(text))
    ids.extend(GENERIC_ID_PATTERN.findall(text))
    for match in STANDALONE_CODE_PATTERN.finditer(text):
        candidate = match.group(1)
        if not _looks_like_word(candidate):
            ids.append(candidate)
    ids.extend(NUMERIC_ID_PATTERN.findall(text))
    for value in qr_values:
        if "/" in value:
            segment = value.rstrip("/").rsplit("/", 1)[-1].rstrip(".")
            if segment:
                ids.append(segment)
    return list(dict.fromkeys(ids))


def _looks_like_word(value: str) -> bool:
    letters = sum(1 for char in value if char.isalpha())
    digits = sum(1 for char in value if char.isdigit())
    if letters == 0:
        return False
    if digits == 0:
        return True
    return letters / (letters + digits) > 0.7
