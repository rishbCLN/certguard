from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from certguard.models import ExtractionResult, PageExtraction

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

FIELD_PATTERNS = {
    "recipient": (
        re.compile(
            r"(?:presented|awarded|issued|granted)\s+to\s*[:\-]?\s*([^\n]{2,100})",
            re.IGNORECASE,
        ),
        re.compile(r"(?:recipient|student|candidate|name)\s*[:\-]\s*([^\n]{2,100})", re.IGNORECASE),
    ),
    "credential_title": (
        re.compile(
            r"(?:credential|course|program|qualification|certificate)\s*"
            r"(?:title|name)?\s*[:\-]\s*([^\n]{2,160})",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:successfully\s+completed|completion\s+of)\s*[:\-]?\s*([^\n]{2,160})",
            re.IGNORECASE,
        ),
    ),
    "issuer": (
        re.compile(
            r"(?:issued|awarded|provided)\s+by\s*[:\-]?\s*([^\n]{2,120})",
            re.IGNORECASE,
        ),
        re.compile(r"(?:issuer|organization)\s*[:\-]\s*([^\n]{2,120})", re.IGNORECASE),
    ),
    "issue_date": (
        re.compile(
            r"(?:issue(?:d)?\s+(?:date|on)|date\s+of\s+issue)\s*[:\-]?\s*([^\n]{2,60})",
            re.IGNORECASE,
        ),
    ),
    "expiration_date": (
        re.compile(
            r"(?:expir(?:y|ation)\s+date|expires?|valid\s+until)\s*[:\-]?\s*([^\n]{2,60})",
            re.IGNORECASE,
        ),
    ),
}

URL_REPLACEMENTS = (
    (" ", ""),
    ("\n", ""),
    ("\r", ""),
    ("..", "."),
    ("coursera org", "coursera.org"),
    ("coderank com", "coderank.com"),
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


@dataclass(slots=True)
class LoadedDocument:
    images: list[np.ndarray]
    native_texts: list[str]
    page_count: int


def load_document(path: Path, dpi: int = 200) -> tuple[np.ndarray, int]:
    document = load_document_pages(path, dpi=dpi, render_all_pages=False)
    return document.images[0], document.page_count


def load_document_pages(
    path: Path, dpi: int = 200, *, render_all_pages: bool = True
) -> LoadedDocument:
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
                images: list[np.ndarray] = []
                native_texts: list[str] = []
                page_indexes = range(document.page_count) if render_all_pages else range(1)
                for page_index in page_indexes:
                    page = document[page_index]
                    scale = dpi / 72
                    width = page.rect.width * scale
                    height = page.rect.height * scale
                    if width * height > MAX_IMAGE_PIXELS:
                        raise DocumentTooLargeError(
                            "Rendered page dimensions exceed the maximum supported pixel budget"
                        )
                    pixmap = page.get_pixmap(
                        matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False
                    )
                    pixels = pixmap.width * pixmap.height
                    if pixels > MAX_IMAGE_PIXELS:
                        raise DocumentTooLargeError(
                            "Rendered page dimensions exceed the maximum supported pixel budget"
                        )
                    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                        pixmap.height, pixmap.width, pixmap.n
                    )
                    images.append(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    get_text = getattr(page, "get_text", None)
                    native_texts.append(get_text("text").strip() if get_text else "")
                return LoadedDocument(images, native_texts, document.page_count)
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
    return LoadedDocument([image], [""], 1)


def extract_document(
    image: np.ndarray,
    page_count: int = 1,
    *,
    native_text: str = "",
    page_number: int = 1,
) -> ExtractionResult:
    errors: list[str] = []
    ocr_text = ""
    confidence: float | None = None
    if not native_text.strip():
        try:
            import pytesseract

            ocr_text, confidence = _run_ocr(image, pytesseract)
            if confidence is not None and confidence < 0.55:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                prepared = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )[1]
                retry_text, retry_confidence = _run_ocr(prepared, pytesseract)
                if retry_confidence is not None and retry_confidence > confidence:
                    ocr_text, confidence = retry_text, retry_confidence
        except Exception as exc:
            errors.append(f"OCR unavailable: {type(exc).__name__}: {exc}")

    qr_values: list[str] = []
    try:
        qr_values.extend(_decode_qr(image))
    except cv2.error as exc:
        errors.append(f"QR decode failed: {exc}")

    normalized_native = _normalize_text_layout(native_text)
    text_parts = [value for value in (normalized_native, ocr_text) if value]
    text = "\n".join(dict.fromkeys(text_parts))
    combined = "\n".join([text, *qr_values])
    urls = _extract_urls(combined)
    certificate_ids = _extract_certificate_ids(combined, qr_values)
    structured_fields = _extract_structured_fields(text, certificate_ids)
    return ExtractionResult(
        text=text,
        structured_fields=structured_fields,
        formatted_text=_format_structured_fields(structured_fields),
        certificate_ids=certificate_ids,
        urls=urls,
        qr_values=qr_values,
        ocr_confidence=confidence,
        page_count=page_count,
        pages=[
            PageExtraction(
                page_number=page_number,
                text=text,
                text_sources=[
                    source
                    for source, value in (
                        ("native-pdf", normalized_native),
                        ("tesseract", ocr_text),
                    )
                    if value
                ],
                ocr_confidence=confidence,
                errors=list(errors),
            )
        ],
        errors=errors,
    )


def extract_loaded_document(document: LoadedDocument) -> ExtractionResult:
    page_results = [
        extract_document(
            image,
            document.page_count,
            native_text=document.native_texts[index],
            page_number=index + 1,
        )
        for index, image in enumerate(document.images)
    ]
    text = "\n".join(result.text for result in page_results if result.text)
    qr_values = [value for result in page_results for value in result.qr_values]
    combined = "\n".join([text, *qr_values])
    confidences = [
        result.ocr_confidence
        for result in page_results
        if result.ocr_confidence is not None
    ]
    certificate_ids = _extract_certificate_ids(combined, qr_values)
    structured_fields = _extract_structured_fields(text, certificate_ids)
    return ExtractionResult(
        text=text,
        structured_fields=structured_fields,
        formatted_text=_format_structured_fields(structured_fields),
        certificate_ids=certificate_ids,
        urls=_extract_urls(combined),
        qr_values=list(dict.fromkeys(qr_values)),
        ocr_confidence=(round(sum(confidences) / len(confidences), 3) if confidences else None),
        page_count=document.page_count,
        pages=[page for result in page_results for page in result.pages],
        errors=[error for result in page_results for error in result.errors],
    )


def _run_ocr(image: np.ndarray, pytesseract) -> tuple[str, float | None]:  # noqa: ANN001
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if image.ndim == 2 else cv2.cvtColor(
        image, cv2.COLOR_BGR2RGB
    )
    data = pytesseract.image_to_data(
        Image.fromarray(rgb),
        output_type=pytesseract.Output.DICT,
        config="--psm 6",
        lang="eng",
        timeout=OCR_TIMEOUT_SECONDS,
    )
    words = [word.strip() for word in data["text"] if word.strip()]
    values = [float(value) for value in data["conf"] if float(value) >= 0]
    confidence = round(sum(values) / len(values) / 100, 3) if values else None
    return _text_from_ocr_data(data, words), confidence


def _text_from_ocr_data(data: dict, words: list[str]) -> str:
    line_columns = ("page_num", "block_num", "par_num", "line_num")
    has_line_data = all(
        column in data and len(data[column]) == len(data["text"])
        for column in line_columns
    )
    if not has_line_data:
        return " ".join(words)

    lines: list[list[str]] = []
    previous_key: tuple[object, ...] | None = None
    for index, raw_word in enumerate(data["text"]):
        word = raw_word.strip()
        if not word:
            continue
        key = tuple(data[column][index] for column in line_columns)
        if key != previous_key:
            lines.append([])
            previous_key = key
        lines[-1].append(word)
    return "\n".join(" ".join(line) for line in lines)


def _normalize_text_layout(text: str) -> str:
    return "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())


def _extract_structured_fields(text: str, certificate_ids: list[str]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for name, patterns in FIELD_PATTERNS.items():
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                value = _clean_field_value(match.group(1))
                if value:
                    fields[name] = value
                    break
    if certificate_ids:
        fields["certificate_id"] = certificate_ids[0]
    return fields


def _clean_field_value(value: str) -> str:
    value = " ".join(value.split()).strip(" .,:;|-_")
    return value if any(char.isalnum() for char in value) else ""


def _format_structured_fields(fields: dict[str, str]) -> str:
    labels = {
        "recipient": "Recipient",
        "credential_title": "Credential",
        "issuer": "Issuer",
        "issue_date": "Issue date",
        "expiration_date": "Expiration date",
        "certificate_id": "Certificate ID",
    }
    return "\n".join(f"{labels[name]}: {fields[name]}" for name in labels if name in fields)


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
