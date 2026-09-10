import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from certguard.document import (
    DocumentTooLargeError,
    LoadedDocument,
    PageEvidence,
    TextSpan,
    extract_document,
    extract_loaded_document,
    load_document,
    load_document_pages,
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


def test_rejects_raster_dimensions_before_opencv_decode(monkeypatch, tmp_path) -> None:
    path = tmp_path / "certificate.png"
    assert cv2.imwrite(str(path), np.zeros((4, 4, 3), dtype=np.uint8))
    import certguard.document as document_module

    monkeypatch.setattr(document_module, "MAX_IMAGE_PIXELS", 4)
    decode_called = False

    def decode(*_args, **_kwargs):
        nonlocal decode_called
        decode_called = True

    monkeypatch.setattr(cv2, "imread", decode)

    with pytest.raises(DocumentTooLargeError, match="pixel budget"):
        load_document(path)

    assert not decode_called


def test_rejects_pdf_aggregate_render_budget_before_render(monkeypatch, tmp_path) -> None:
    import fitz

    import certguard.document as document_module

    class Rect:
        width = 72.0
        height = 72.0

    class Page:
        rect = Rect()

        def get_pixmap(self, **_kwargs):
            raise AssertionError("aggregate limit must run before rendering")

    class Document:
        page_count = 2

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def __getitem__(self, _index):
            return Page()

    monkeypatch.setattr(fitz, "open", lambda _path: Document())
    monkeypatch.setattr(document_module, "MAX_TOTAL_RENDERED_PIXELS", 1)

    with pytest.raises(DocumentTooLargeError, match="aggregate"):
        document_module.load_document_pages(tmp_path / "certificate.pdf")


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

    pytesseract = SimpleNamespace(Output=SimpleNamespace(DICT="dict"), image_to_data=fail_ocr)
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
    assert result.pages[0].text_sources == ["native-pdf", "tesseract"]
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
        "Certificate",
        "of",
        "Completion",
        "Presented",
        "to",
        "Alice",
        "Example",
        "Course:",
        "Python",
        "Basics",
        "Issued",
        "by:",
        "Example",
        "Learning",
        "Issue",
        "Date:",
        "2026-09-10",
        "Certificate",
        "ID:",
        "ABC12345",
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


def test_ocr_geometry_is_preserved_as_normalized_spans(monkeypatch) -> None:
    pytesseract = SimpleNamespace(
        Output=SimpleNamespace(DICT="dict"),
        image_to_data=lambda *_args, **_kwargs: {
            "text": ["Alice"],
            "conf": ["90"],
            "left": [20],
            "top": [10],
            "width": [40],
            "height": [20],
        },
    )

    class Detector:
        def detectAndDecodeMulti(self, _image):
            return False, (), None, None

        def detectAndDecode(self, _image):
            return "", None, None

    monkeypatch.setitem(sys.modules, "pytesseract", pytesseract)
    monkeypatch.setattr(cv2, "QRCodeDetector", Detector)
    evidence = PageEvidence(1, 200, 100)
    extract_document(
        np.zeros((100, 200, 3), dtype=np.uint8),
        page_evidence=evidence,
    )

    assert len(evidence.text_spans) == 1
    assert evidence.text_spans[0].box == pytest.approx((0.1, 0.1, 0.2, 0.2))
    assert evidence.text_spans[0].confidence == 0.9


def test_sparse_native_text_runs_ocr_and_deduplicates_overlapping_spans(monkeypatch) -> None:
    pytesseract = SimpleNamespace(
        Output=SimpleNamespace(DICT="dict"),
        image_to_data=lambda *_args, **_kwargs: {
            "text": ["Certificate", "Raster", "Seal"],
            "conf": ["95", "91", "90"],
            "left": [10, 100, 150],
            "top": [10, 50, 50],
            "width": [80, 45, 30],
            "height": [20, 20, 20],
            "page_num": [1, 1, 1],
            "block_num": [1, 2, 2],
            "par_num": [1, 1, 1],
            "line_num": [1, 1, 1],
        },
    )

    class Detector:
        def detectAndDecodeMulti(self, _image):
            return False, (), None, None

        def detectAndDecode(self, _image):
            return "", None, None

    monkeypatch.setitem(sys.modules, "pytesseract", pytesseract)
    monkeypatch.setattr(cv2, "QRCodeDetector", Detector)
    evidence = PageEvidence(
        1,
        200,
        100,
        text_spans=[TextSpan("Certificate", (0.05, 0.1, 0.4, 0.2), "native-pdf")],
    )

    result = extract_document(
        np.zeros((100, 200, 3), dtype=np.uint8),
        native_text="Certificate",
        page_evidence=evidence,
    )

    assert result.text == "Certificate\nRaster Seal"
    assert result.pages[0].text_sources == ["native-pdf", "tesseract"]
    assert [span.text for span in evidence.text_spans] == ["Certificate", "Raster", "Seal"]
    assert evidence.text_spans[1].line_box == pytest.approx((0.5, 0.5, 0.4, 0.2))
    assert evidence.text_spans[1].baseline == pytest.approx(0.7)


def test_native_pdf_spans_preserve_boxes_fonts_and_line_baseline() -> None:
    import certguard.document as document_module

    page = SimpleNamespace(
        rect=SimpleNamespace(width=200.0, height=100.0),
        get_text=lambda _format: {
            "blocks": [
                {
                    "lines": [
                        {
                            "bbox": [20, 10, 120, 30],
                            "spans": [
                                {
                                    "text": "Recipient",
                                    "bbox": [20, 10, 80, 30],
                                    "origin": [20, 27],
                                    "font": "ExampleSans",
                                    "size": 14,
                                }
                            ],
                        }
                    ]
                }
            ]
        },
    )

    spans = document_module._pdf_text_spans(page)

    assert spans[0].box == pytest.approx((0.1, 0.1, 0.3, 0.2))
    assert spans[0].line_box == pytest.approx((0.1, 0.1, 0.5, 0.2))
    assert spans[0].baseline == pytest.approx(0.27)
    assert (spans[0].font_name, spans[0].font_size) == ("ExampleSans", 14.0)


def test_qr_polygon_is_normalized_and_retained(monkeypatch) -> None:
    class Detector:
        def detectAndDecodeMulti(self, _image):
            points = np.array([[[20, 10], [60, 10], [60, 30], [20, 30]]], dtype=np.float32)
            return True, ("https://verify.example/id",), points, None

    monkeypatch.setattr(cv2, "QRCodeDetector", Detector)
    evidence = PageEvidence(1, 200, 100)

    extract_document(
        np.zeros((100, 200, 3), dtype=np.uint8),
        native_text="Enough native words to avoid invoking optical character recognition",
        page_evidence=evidence,
    )

    assert np.asarray(evidence.qr_observations[0].polygon) == pytest.approx(
        np.array(((0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)))
    )
    assert evidence.qr_observations[0].coordinate_system == "normalized-page"


def test_pdf_evidence_contains_raster_placements_vectors_and_bounded_metadata(
    monkeypatch, tmp_path
) -> None:
    import fitz

    class Pixmap:
        samples = bytes([255, 255, 255] * 8)
        height = 2
        width = 4
        n = 3

    class Page:
        rect = fitz.Rect(0, 0, 200, 100)

        def get_pixmap(self, **_kwargs):
            return Pixmap()

        def get_text(self, format_name):
            return "Certificate text" if format_name == "text" else {"blocks": []}

        def get_images(self, *, full):
            assert full
            return [(7, 0, 600, 300)]

        def get_image_rects(self, xref):
            assert xref == 7
            return [fitz.Rect(20, 10, 164, 82)]

        def get_drawings(self):
            return [
                {"rect": fitz.Rect(0, 0, 100, 50)},
                {"rect": fitz.Rect(50, 25, 150, 75)},
            ]

    class Document:
        page_count = 1
        metadata = {"producer": "p" * 300, "author": "private", "creationDate": "today"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def __getitem__(self, _index):
            return Page()

    monkeypatch.setattr(fitz, "open", lambda _path: Document())

    loaded = load_document_pages(tmp_path / "evidence.pdf", dpi=72)
    evidence = loaded.evidence[0]

    assert evidence.embedded_rasters[0].box == pytest.approx((0.1, 0.1, 0.72, 0.72))
    assert evidence.embedded_rasters[0].effective_dpi_x == pytest.approx(300)
    assert evidence.embedded_rasters[0].effective_dpi_y == pytest.approx(300)
    assert len(evidence.vector_drawings) == 2
    assert evidence.vector_coverage == pytest.approx(0.4375)
    assert evidence.pdf_metadata == {"producer": "p" * 256, "creationDate": "today"}
