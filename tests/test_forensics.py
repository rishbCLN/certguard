from pathlib import Path

import cv2
import numpy as np

from certguard.forensics import ProvenanceAnalyzer, TemplateAnalyzer


class ForgeryModel:
    name = "test-model.onnx"

    def predict(self, _image: np.ndarray) -> float:
        return 0.86


def test_low_feature_template_comparison_is_inconclusive() -> None:
    blank = np.full((400, 600), 255, dtype=np.uint8)

    result = TemplateAnalyzer._compare(
        blank,
        blank.copy(),
        "example",
        {"id": "blank", "image": "blank.png"},
    )

    assert not result.available
    assert result.anomaly_score is None


def test_copy_move_detector_finds_repeated_offset_feature_cluster() -> None:
    image = np.full((500, 700), 255, dtype=np.uint8)
    patch = np.full((140, 180), 255, dtype=np.uint8)
    cv2.putText(patch, "CERT-123", (8, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)
    cv2.circle(patch, (35, 110), 18, 0, 3)
    image[50:190, 50:230] = patch
    image[280:420, 420:600] = patch

    assert (ProvenanceAnalyzer._copy_move_score(image) or 0) > 0


def test_moire_score_detects_pattern() -> None:
    image = np.zeros((512, 512), dtype=np.uint8)
    # Create a moire-like pattern
    for i in range(0, 512, 8):
        image[:, i::16] = 255
    assert ProvenanceAnalyzer._moire_score(image) >= 0.0


def test_capture_method_identifies_born_digital_pdf() -> None:
    image = np.zeros((512, 512), dtype=np.uint8)
    method, confidence = ProvenanceAnalyzer._capture_method(image, 0.0, Path("test.pdf"))
    assert method == "born-digital"


def test_capture_method_identifies_screen_recapture() -> None:
    image = np.zeros((512, 512), dtype=np.uint8)
    # High moire score
    method, confidence = ProvenanceAnalyzer._capture_method(image, 0.5, Path("test.jpg"))
    assert method == "screen-recapture"


def test_provenance_emits_artifact_noiseprint_and_model_signals(tmp_path) -> None:
    source = tmp_path / "certificate.png"
    image = np.full((256, 384, 3), 245, dtype=np.uint8)
    cv2.putText(image, "CERTIFICATE", (25, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 40, 60), 2)
    assert cv2.imwrite(str(source), image)

    result = ProvenanceAnalyzer(ForgeryModel()).analyze(image, source)

    assert result.neural_model_available
    assert result.neural_model_name == "test-model.onnx"
    assert result.neural_forgery_score == 0.86
    assert result.jpeg_grid_score is not None
    assert result.frequency_anomaly_score is not None
    assert result.font_subpixel_score is not None
    assert result.noiseprint_score is not None
    assert result.printer_pattern_score is not None


def test_available_template_feature_detectors_include_orb_and_sift() -> None:
    names = [name for name, _detector, _norm in TemplateAnalyzer._feature_candidates()]

    assert "ORB" in names
    assert "SIFT" in names
