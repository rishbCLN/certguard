import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from certguard.document import (
    DocumentTooLargeError,
    LoadedDocument,
    extract_document,
    extract_loaded_document,
    load_document,
)


def test_loads_raster_image(tmp_path) -> None:
    path = tmp_path / "certificate.png"
    expected = np.full((2, 3, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(path), expected)

    image, page_count = load_document(path)

    assert np.array_equal(image, expected)
    assert page_count == 1


def test_rejects_unreadable_raster_image(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unsupported or unreadable document"):
        load_document(tmp_path / "missing.png")


def test_rejects_oversized_source_file(tmp_path) -> None:
    path = tmp_path / "certificate.png"
    path.write_bytes(b"\x00" * 64)
    oversized = DocumentTooLargeError.__bases__[0]  # sanity: subclass of ValueError
    assert issubclass(oversized, ValueError)

    import certguard.document as document_module

    original = document_module.MAX_SOURCE_BYTES
    document_module.MAX_SOURCE_BYTES = 32
    try:
        with pytest.raises(DocumentTooLargeError, match="maximum supported size"):
            load_document(path)
    finally:
        document_module.MAX_SOURCE_BYTES = original


def test_rejects_excessive_pdf_page_count(monkeypatch, tmp_path) -> None:
    import fitz

    class Document:
        page_count = 21

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(fitz, "open", lambda _path: Document())

    with pytest.raises(DocumentTooLargeError, match="pages"):
        load_document(tmp_path / "huge.pdf")


def test_rejects_rendered_page_over_pixel_budget(monkeypatch, tmp_path) -> None:
    import fitz

    class Rect:
        width = 60000.0
        height = 60000.0

    class Page:
        rect = Rect()

    class Document:
        page_count = 1

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def __getitem__(self, index):
            return Page()

    monkeypatch.setattr(fitz, "open", lambda _path: Document())

    with pytest.raises(DocumentTooLargeError, match="pixel budget"):
        load_document(tmp_path / "certificate.pdf", dpi=200)


def test_rejects_raster_image_over_pixel_budget(tmp_path) -> None:
    path = tmp_path / "certificate.png"
    assert cv2.imwrite(str(path), np.zeros((4, 4, 3), dtype=np.uint8))

    import certguard.document as document_module

    original = document_module.MAX_IMAGE_PIXELS
    document_module.MAX_IMAGE_PIXELS = 4
    try:
        with pytest.raises(DocumentTooLargeError, match="pixel budget"):
            load_document(path)
    finally:
        document_module.MAX_IMAGE_PIXELS = original


def test_loads_first_pdf_page(monkeypatch, tmp_path) -> None:
    import fitz

    class Pixmap:
        samples = bytes([10, 20, 30])
        height = 1
        width = 1
        n = 3

    class Page:
        class rect:
            width = 1.0
            height = 1.0

        def get_pixmap(self, *, matrix, alpha):
            assert matrix is not None
            assert alpha is False
            return Pixmap()

    class Document:
        page_count = 2

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def __getitem__(self, index):
            assert index == 0
            return Page()

    monkeypatch.setattr(fitz, "open", lambda _path: Document())

    image, page_count = load_document(tmp_path / "certificate.pdf", dpi=144)

    assert image.tolist() == [[[30, 20, 10]]]
    assert page_count == 2


def test_rejects_empty_pdf(monkeypatch, tmp_path) -> None:
    import fitz

    class Document:
        page_count = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(fitz, "open", lambda _path: Document())

    with pytest.raises(ValueError, match="PDF has no pages"):
        load_document(tmp_path / "empty.pdf")


def test_normalizes_corrupted_pdf_error(monkeypatch, tmp_path) -> None:
    import fitz

    def fail_to_open(_path):
        raise fitz.FileDataError("damaged xref")

    monkeypatch.setattr(fitz, "open", fail_to_open)

    with pytest.raises(ValueError, match="Unsupported or unreadable PDF") as error:
        load_document(tmp_path / "corrupt.pdf")

    assert isinstance(error.value.__cause__, fitz.FileDataError)


def test_qr_url_is_extracted() -> None:
    url = "https://www.coderank.com/certificates/abc123def"
    encoder = cv2.QRCodeEncoder_create()
    qr = encoder.encode(url)
    qr = cv2.resize(qr, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
    image = cv2.copyMakeBorder(qr, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    result = extract_document(image)

    assert url in result.qr_values
    assert url in result.urls


def test_ocr_text_confidence_and_single_qr_are_extracted(monkeypatch) -> None:
    pytesseract = SimpleNamespace(
        Output=SimpleNamespace(DICT="dict"),
        image_to_data=lambda *_args, **_kwargs: {
            "text": ["Certificate", "ID:", "ABC123", ""],
            "conf": ["80", "-1", "100"],
        },
    )

    class Detector:
        def detectAndDecodeMulti(self, _image):
            return False, (), None, None

        def detectAndDecode(self, _image):
            return "https://verify.example/c/ABC123.", None, None

    monkeypatch.setitem(sys.modules, "pytesseract", pytesseract)
    monkeypatch.setattr(cv2, "QRCodeDetector", Detector)

    result = extract_document(np.zeros((2, 2, 3), dtype=np.uint8), page_count=3)

    assert result.text == "Certificate ID: ABC123"
    assert result.certificate_ids == ["ABC123"]
    assert result.urls == ["https://verify.example/c/ABC123"]
    assert result.ocr_confidence == 0.9
    assert result.page_count == 3
    assert result.errors == []


def test_failed_ocr_and_qr_extraction_are_reported(monkeypatch) -> None:
    def fail_ocr(*_args, **_kwargs):
        raise RuntimeError("engine unavailable")

    class Detector:
        def detectAndDecodeMulti(self, _image):
            raise cv2.error("invalid image")

    pytesseract = SimpleNamespace(
        Output=SimpleNamespace(DICT="dict"), image_to_data=fail_ocr
    )
    monkeypatch.setitem(sys.modules, "pytesseract", pytesseract)
    monkeypatch.setattr(cv2, "QRCodeDetector", Detector)

    result = extract_document(np.zeros((1, 1, 3), dtype=np.uint8))

    assert result.text == ""
    assert result.ocr_confidence is None
    assert result.qr_values == []
    assert result.errors[0] == "OCR unavailable: RuntimeError: engine unavailable"
    assert result.errors[1].startswith("QR decode failed:")


def test_multiple_qr_values_are_deduplicated_in_urls(monkeypatch) -> None:
    class Detector:
        def detectAndDecodeMulti(self, _image):
            decoded = (
                " https://verify.example/ABC123 ",
                "",
                "https://verify.example/ABC123",
            )
            return True, decoded, None, None

    pytesseract = SimpleNamespace(
        Output=SimpleNamespace(DICT="dict"),
        image_to_data=lambda *_args, **_kwargs: {"text": [], "conf": []},
    )
    monkeypatch.setitem(sys.modules, "pytesseract", pytesseract)
    monkeypatch.setattr(cv2, "QRCodeDetector", Detector)

    result = extract_document(np.zeros((1, 1, 3), dtype=np.uint8))

    assert result.qr_values == ["https://verify.example/ABC123", "https://verify.example/ABC123"]
    assert result.urls == ["https://verify.example/ABC123"]


def test_loaded_document_combines_text_from_every_page(monkeypatch) -> None:
    pytesseract = SimpleNamespace(
        Output=SimpleNamespace(DICT="dict"),
        image_to_data=lambda *_args, **_kwargs: {
            "text": ["Certificate", "ID:", "ABC12345"],
            "conf": ["80", "90", "100"],
        },
    )

    class Detector:
        def detectAndDecodeMulti(self, _image):
            return False, (), None, None

        def detectAndDecode(self, _image):
            return "", None, None

    monkeypatch.setitem(sys.modules, "pytesseract", pytesseract)
    monkeypatch.setattr(cv2, "QRCodeDetector", Detector)
    document = LoadedDocument(
        images=[
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.zeros((2, 2, 3), dtype=np.uint8),
        ],
        native_texts=["Native page one", ""],
        page_count=2,
    )

    result = extract_loaded_document(document)

    assert result.page_count == 2
    assert result.certificate_ids == ["ABC12345"]
    assert [page.page_number for page in result.pages] == [1, 2]
    assert result.pages[0].text_sources == ["native-pdf"]
    assert "Native page one" in result.text
    assert "Certificate ID: ABC12345" in result.text


def test_low_confidence_ocr_uses_better_preprocessed_result(monkeypatch) -> None:
    responses = iter(
        [
            {"text": ["Certiflcate"], "conf": ["40"]},
            {"text": ["Certificate"], "conf": ["95"]},
        ]
    )
    pytesseract = SimpleNamespace(
        Output=SimpleNamespace(DICT="dict"),
        image_to_data=lambda *_args, **_kwargs: next(responses),
    )

    class Detector:
        def detectAndDecodeMulti(self, _image):
            return False, (), None, None

        def detectAndDecode(self, _image):
            return "", None, None

    monkeypatch.setitem(sys.modules, "pytesseract", pytesseract)
    monkeypatch.setattr(cv2, "QRCodeDetector", Detector)

    result = extract_document(np.zeros((2, 2, 3), dtype=np.uint8))

    assert result.text == "Certificate"
    assert result.ocr_confidence == 0.95


def test_ocr_preserves_lines_and_extracts_structured_certificate_view(monkeypatch) -> None:
    words = [
        "Certificate", "of", "Completion", "Presented", "to", "Alice", "Example",
        "Course:", "Python", "Basics", "Issued", "by:", "Example", "Learning",
        "Issue", "Date:", "2026-09-10", "Certificate", "ID:", "ABC12345",
    ]
    line_numbers = [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6]
    pytesseract = SimpleNamespace(
        Output=SimpleNamespace(DICT="dict"),
        image_to_data=lambda *_args, **_kwargs: {
            "text": words,
            "conf": ["95"] * len(words),
            "page_num": [1] * len(words),
            "block_num": [1] * len(words),
            "par_num": [1] * len(words),
            "line_num": line_numbers,
        },
    )

    class Detector:
        def detectAndDecodeMulti(self, _image):
            return False, (), None, None

        def detectAndDecode(self, _image):
            return "", None, None

    monkeypatch.setitem(sys.modules, "pytesseract", pytesseract)
    monkeypatch.setattr(cv2, "QRCodeDetector", Detector)

    result = extract_document(np.zeros((2, 2, 3), dtype=np.uint8))

    assert "Certificate of Completion\nPresented to Alice Example" in result.text
    assert result.structured_fields == {
        "recipient": "Alice Example",
        "credential_title": "Python Basics",
        "issuer": "Example Learning",
        "issue_date": "2026-09-10",
        "certificate_id": "ABC12345",
    }
    assert result.formatted_text == (
        "Recipient: Alice Example\n"
        "Credential: Python Basics\n"
        "Issuer: Example Learning\n"
        "Issue date: 2026-09-10\n"
        "Certificate ID: ABC12345"
    )


def test_native_pdf_text_keeps_line_layout_for_structured_extraction(monkeypatch) -> None:
    class Detector:
        def detectAndDecodeMulti(self, _image):
            return False, (), None, None

        def detectAndDecode(self, _image):
            return "", None, None

    monkeypatch.setattr(cv2, "QRCodeDetector", Detector)

    result = extract_document(
        np.zeros((2, 2, 3), dtype=np.uint8),
        native_text="Recipient:  Bob Example\nCourse: Data Engineering\nCertificate ID: ZXCV1234",
    )

    assert result.text.splitlines() == [
        "Recipient: Bob Example",
        "Course: Data Engineering",
        "Certificate ID: ZXCV1234",
    ]
    assert result.structured_fields["recipient"] == "Bob Example"
    assert result.structured_fields["credential_title"] == "Data Engineering"
    assert result.structured_fields["certificate_id"] == "ZXCV1234"
