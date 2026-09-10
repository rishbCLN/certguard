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
    evidence: list[PageEvidence] | None = None


@dataclass(slots=True)
class TextSpan:
    text: str
    box: tuple[float, float, float, float]
    source: str
    confidence: float | None = None
    baseline: float | None = None
    font_name: str | None = None
    font_size: float | None = None


@dataclass(slots=True)
class QRObservation:
    value: str
    polygon: tuple[tuple[float, float], ...]


@dataclass(slots=True)
class PageEvidence:
    page_number: int
    width_px: int
    height_px: int
    width_pt: float | None = None
    height_pt: float | None = None
    text_spans: list[TextSpan] | None = None
    qr_observations: list[QRObservation] | None = None
    vector_drawing_count: int = 0
    embedded_image_dpi: list[float] | None = None

    def __post_init__(self) -> None:
        if self.text_spans is None:
            self.text_spans = []
        if self.qr_observations is None:
            self.qr_observations = []
        if self.embedded_image_dpi is None:
            self.embedded_image_dpi = []


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
                evidence: list[PageEvidence] = []
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
                    spans = _pdf_text_spans(page)
                    drawings = getattr(page, "get_drawings", lambda: [])()
                    evidence.append(
                        PageEvidence(
                            page_number=page_index + 1,
                            width_px=pixmap.width,
                            height_px=pixmap.height,
                            width_pt=float(page.rect.width),
                            height_pt=float(page.rect.height),
                            text_spans=spans,
                            vector_drawing_count=len(drawings),
                            embedded_image_dpi=_pdf_image_dpi(page),
                        )
                    )
                return LoadedDocument(images, native_texts, document.page_count, evidence)
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
    return LoadedDocument(
        [image],
        [""],
        1,
        [PageEvidence(1, image.shape[1], image.shape[0])],
    )


def extract_document(
    image: np.ndarray,
    page_count: int = 1,
    *,
    native_text: str = "",
    page_number: int = 1,
    page_evidence: PageEvidence | None = None,
) -> ExtractionResult:
    errors: list[str] = []
    ocr_text = ""
    confidence: float | None = None
    ocr_spans: list[TextSpan] = []
    if not native_text.strip():
        try:
            import pytesseract

            ocr_text, confidence, ocr_spans = _run_ocr(image, pytesseract)
            if confidence is not None and confidence < 0.55:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                prepared = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )[1]
                retry_text, retry_confidence, retry_spans = _run_ocr(prepared, pytesseract)
                if retry_confidence is not None and retry_confidence > confidence:
                    ocr_text, confidence, ocr_spans = retry_text, retry_confidence, retry_spans
        except Exception as exc:
            errors.append(f"OCR unavailable: {type(exc).__name__}: {exc}")

    qr_values: list[str] = []
    qr_observations: list[QRObservation] = []
    try:
        qr_observations = _decode_qr_observations(image)
        qr_values.extend(item.value for item in qr_observations)
    except cv2.error as exc:
        errors.append(f"QR decode failed: {exc}")

    normalized_native = _normalize_text_layout(native_text)
    text_parts = [value for value in (normalized_native, ocr_text) if value]
    text = "\n".join(dict.fromkeys(text_parts))
    combined = "\n".join([text, *qr_values])
    urls = _extract_urls(combined)
    certificate_ids = _extract_certificate_ids(combined, qr_values)
    structured_fields = _extract_structured_fields(text, certificate_ids)
    if page_evidence is not None:
        page_evidence.text_spans = _merge_text_spans(page_evidence.text_spans or [], ocr_spans)
        page_evidence.qr_observations = qr_observations
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
    if document.evidence is None:
        document.evidence = [
            PageEvidence(index + 1, image.shape[1], image.shape[0])
            for index, image in enumerate(document.images)
        ]
    page_results = [
        extract_document(
            image,
            document.page_count,
            native_text=document.native_texts[index],
            page_number=index + 1,
            page_evidence=document.evidence[index],
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


def _run_ocr(
    image: np.ndarray, pytesseract
) -> tuple[str, float | None, list[TextSpan]]:  # noqa: ANN001
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
    return _text_from_ocr_data(data, words), confidence, _ocr_text_spans(data, image.shape)


def _ocr_text_spans(data: dict, shape: tuple[int, ...]) -> list[TextSpan]:
    required = ("text", "left", "top", "width", "height")
    if not all(key in data for key in required):
        return []
    height_px, width_px = shape[:2]
    spans: list[TextSpan] = []
    for index, raw_text in enumerate(data["text"]):
        text = str(raw_text).strip()
        if not text:
            continue
        left = float(data["left"][index]) / width_px
        top = float(data["top"][index]) / height_px
        width = float(data["width"][index]) / width_px
        height = float(data["height"][index]) / height_px
        raw_confidence = float(data.get("conf", [-1] * len(data["text"]))[index])
        spans.append(
            TextSpan(
                text=text,
                box=(left, top, width, height),
                source="tesseract",
                confidence=(raw_confidence / 100 if raw_confidence >= 0 else None),
                baseline=top + height,
            )
        )
    return spans


def _pdf_text_spans(page) -> list[TextSpan]:  # noqa: ANN001
    get_text = getattr(page, "get_text", None)
    if get_text is None:
        return []
    try:
        payload = get_text("dict")
    except (TypeError, ValueError):
        return []
    width = float(page.rect.width)
    height = float(page.rect.height)
    spans: list[TextSpan] = []
    for block in payload.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = str(span.get("text", "")).strip()
                box = span.get("bbox")
                if not text or not isinstance(box, list | tuple) or len(box) != 4:
                    continue
                x0, y0, x1, y1 = (float(value) for value in box)
                spans.append(
                    TextSpan(
                        text=text,
                        box=(x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height),
                        source="native-pdf",
                        baseline=y1 / height,
                        font_name=str(span.get("font")) if span.get("font") else None,
                        font_size=float(span["size"]) if span.get("size") else None,
                    )
                )
    return spans


def _pdf_image_dpi(page) -> list[float]:  # noqa: ANN001
    values: list[float] = []
    get_images = getattr(page, "get_images", None)
    get_rects = getattr(page, "get_image_rects", None)
    if get_images is None or get_rects is None:
        return values
    for image in get_images(full=True):
        if len(image) < 4:
            continue
        xref, width_px, height_px = image[0], image[2], image[3]
        for rect in get_rects(xref):
            if rect.width > 0 and rect.height > 0:
                dpi_x = float(width_px) / (float(rect.width) / 72)
                dpi_y = float(height_px) / (float(rect.height) / 72)
                values.append(round((dpi_x + dpi_y) / 2, 3))
    return values


def _merge_text_spans(native: list[TextSpan], ocr: list[TextSpan]) -> list[TextSpan]:
    merged = list(native)
    for candidate in ocr:
        if not any(
            existing.text.casefold() == candidate.text.casefold()
            and _box_overlap(existing.box, candidate.box) > 0.7
            for existing in native
        ):
            merged.append(candidate)
    return merged


def _box_overlap(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    intersection = max(0.0, min(lx + lw, rx + rw) - max(lx, rx)) * max(
        0.0, min(ly + lh, ry + rh) - max(ly, ry)
    )
    return intersection / max(min(lw * lh, rw * rh), 1e-9)


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
    return [item.value for item in _decode_qr_observations(image)]


def _decode_qr_observations(image: np.ndarray) -> list[QRObservation]:
    observations: list[QRObservation] = []
    detector = cv2.QRCodeDetector()
    found, decoded, points, _ = detector.detectAndDecodeMulti(image)
    if found:
        observations.extend(_qr_observations(decoded, points, image.shape))
    else:
        value, points, _ = detector.detectAndDecode(image)
        if value.strip():
            observations.extend(_qr_observations((value,), points, image.shape))
    if observations:
        return observations
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
        found, decoded, points, _ = detector.detectAndDecodeMulti(processed)
        if found:
            observations.extend(_qr_observations(decoded, points, image.shape))
        else:
            value, points, _ = detector.detectAndDecode(processed)
            if value.strip():
                observations.extend(_qr_observations((value,), points, image.shape))
        if observations:
            break
    return observations


def _qr_observations(decoded, points, shape: tuple[int, ...]) -> list[QRObservation]:  # noqa: ANN001
    height, width = shape[:2]
    point_sets = np.asarray(points).reshape(-1, 4, 2) if points is not None else []
    observations: list[QRObservation] = []
    for index, raw_value in enumerate(decoded):
        value = str(raw_value).strip()
        if not value:
            continue
        polygon: tuple[tuple[float, float], ...] = ()
        if index < len(point_sets):
            polygon = tuple(
                (float(point[0]) / width, float(point[1]) / height)
                for point in point_sets[index]
            )
        observations.append(QRObservation(value, polygon))
    return observations


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
