import hashlib
import json
from importlib.resources import files

import cv2
import numpy as np
import pytest
import yaml

from certguard.document import LoadedDocument, PageEvidence, TextSpan
from certguard.models import (
    ContentResult,
    ExtractionResult,
    ProvenanceResult,
    SSDDResult,
    TemplateResult,
    VerificationResult,
    VerificationStatus,
)
from certguard.scoring import calculate_risk
from certguard.ssdd import (
    GrammarProfileError,
    GrammarStore,
    QRBinding,
    build_grammar_profile,
    color_diff_ciede2000,
    compute_aa_variance,
    detect_text_qr_binding,
    haar_texture_correlation,
    load_profile,
    run_ssdd,
)


def template_image(path) -> None:
    image = np.full((600, 900, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (880, 580), (30, 30, 30), 4)
    cv2.putText(image, "CERTIFICATE", (250, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    cv2.circle(image, (120, 100), 50, (180, 60, 20), -1)
    assert cv2.imwrite(str(path), image)


def builder_manifest(path, *, active=True, scoring=False) -> None:
    data = {
        "display_name": "Synthetic certificate",
        "version": "1.0",
        "active": active,
        "page_size_mm": [297, 210],
        "minimum_alignment": 0.2,
        "minimum_evaluable_fraction": 0.5,
        "regions": {
            "title": {"kind": "text", "box": [0.2, 0.05, 0.6, 0.2]},
            "logo": {"kind": "visual", "box": [0.03, 0.03, 0.2, 0.25]},
        },
        "grammar": [
            {
                "id": "title-present",
                "type": "region_presence",
                "first": "title",
                "value": True,
                "unit": "boolean",
                "tolerance": 0,
                "required": True,
            }
        ],
        "calibration": {
            "scoring_enabled": scoring,
            "benchmark_id": "synthetic-v1" if scoring else None,
        },
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_text_qr_binding_success_and_failure() -> None:
    binding = QRBinding(
        source_field="recipient",
        claim_key="student",
        trusted_hosts=("nptel.ac.in",),
        normalizers=("url-decode", "strip", "casefold"),
    )
    fields = {"recipient": "RAJ KUMAR"}

    assert detect_text_qr_binding(
        fields,
        ["https://nptel.ac.in/cert?student=RAJ%20KUMAR"],
        binding,
    ) == "pass"
    assert detect_text_qr_binding(
        fields,
        ["https://nptel.ac.in/cert?student=JOHN"],
        binding,
    ) == "violation"
    assert detect_text_qr_binding(
        fields,
        ["https://fake.example/cert?student=RAJ%20KUMAR"],
        binding,
    ) == "unavailable"


def test_color_distance_and_rendering_helpers() -> None:
    blue = np.array([138, 68, 22], dtype=np.uint8)
    red = np.array([0, 0, 255], dtype=np.uint8)
    assert color_diff_ciede2000(blue, blue) == pytest.approx(0)
    assert color_diff_ciede2000(blue, red) > 15

    text = np.full((80, 240, 3), 255, dtype=np.uint8)
    cv2.putText(text, "TEXT", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    assert compute_aa_variance(text) is not None
    assert compute_aa_variance(np.full_like(text, 255)) is None
    assert haar_texture_correlation(text, text.copy()) == pytest.approx(1)


def test_profile_validation_rejects_bad_regions_and_duplicate_rules(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "issuer_id": "nptel",
                "variant_id": "bad",
                "active": False,
                "regions": {"title": {"box": [0.9, 0.1, 0.2, 0.2]}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(GrammarProfileError, match="normalized"):
        load_profile(path)


def test_builder_copies_template_signs_profile_and_store_verifies(tmp_path) -> None:
    template = tmp_path / "source.png"
    manifest = tmp_path / "manifest.yaml"
    root = tmp_path / "grammar"
    output = root / "nptel.yaml"
    key = tmp_path / "hmac.key"
    key.write_bytes(b"x" * 32)
    template_image(template)
    builder_manifest(manifest)

    build_grammar_profile(
        "nptel",
        "nptel-v1",
        template,
        manifest,
        output,
        key_file=key,
    )

    profile = GrammarStore(root, key_file=key).profiles[0]
    assert profile.active
    assert profile.template_path is not None
    assert profile.template_path.parent.name == "templates"
    assert profile.template_sha256 == hashlib.sha256(template.read_bytes()).hexdigest()
    signatures = json.loads((root / ".signatures.json").read_text(encoding="utf-8"))
    assert signatures["nptel.yaml"]["sha256"] == profile.sha256


def test_changed_signed_profile_fails_closed(tmp_path) -> None:
    template = tmp_path / "source.png"
    manifest = tmp_path / "manifest.yaml"
    root = tmp_path / "grammar"
    output = root / "nptel.yaml"
    key = tmp_path / "hmac.key"
    key.write_bytes(b"x" * 32)
    template_image(template)
    builder_manifest(manifest)
    build_grammar_profile("nptel", "v1", template, manifest, output, key_file=key)
    output.write_text(output.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(GrammarProfileError, match="checksum"):
        GrammarStore(root, key_file=key)


def test_inactive_packaged_profiles_are_loadable_and_neutral() -> None:
    root = files("certguard.data").joinpath("grammar")
    profile_names = {
        "nptel.example.yaml",
        "swayam.example.yaml",
        "coursera.example.yaml",
    }

    loaded = [load_profile(root.joinpath(name)) for name in profile_names]

    assert all(not profile.active for profile in loaded)
    assert {profile.issuer_id for profile in loaded} == {"nptel", "coursera"}


def test_ssdd_scoring_is_calibration_gated_and_capped() -> None:
    verification = VerificationResult(
        VerificationStatus.FAILED_LOOKUP,
        "issuer",
        "Issuer",
        "failed",
    )
    ssdd = SSDDResult(
        status="completed",
        delta=0.9,
        scoring_enabled=True,
        risk_points=10,
    )
    score, _, contributions = calculate_risk(
        verification,
        TemplateResult(available=True, anomaly_score=1.0),
        ProvenanceResult(neural_model_available=True, neural_forgery_score=1.0),
        ContentResult(available=True, anomaly_score=1.0),
        ssdd,
    )

    assert score == 100
    assert contributions[-1].calculation == "calibrated-threshold-points"


def test_ssdd_missing_required_evidence_caps_coverage() -> None:
    _, coverage, _ = calculate_risk(
        VerificationResult(VerificationStatus.VERIFIED, "issuer", "Issuer", "ok"),
        TemplateResult(available=True, anomaly_score=0),
        ProvenanceResult(),
        ContentResult(available=True, anomaly_score=0),
        SSDDResult(status="insufficient-evidence", delta=None),
    )

    assert coverage <= 0.72


def test_run_ssdd_returns_neutral_when_no_profile() -> None:
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    document = LoadedDocument(
        [image],
        [""],
        1,
        [PageEvidence(1, 200, 100)],
    )

    result = run_ssdd(GrammarStore(), "nptel", document, ExtractionResult()).result

    assert result.status == "not-configured"
    assert result.delta == 0
    assert result.violations == []


def test_run_ssdd_with_missing_required_span_is_insufficient(tmp_path) -> None:
    template = tmp_path / "source.png"
    manifest = tmp_path / "manifest.yaml"
    root = tmp_path / "grammar"
    output = root / "nptel.yaml"
    key = tmp_path / "hmac.key"
    key.write_bytes(b"x" * 32)
    template_image(template)
    builder_manifest(manifest)
    build_grammar_profile("nptel", "v1", template, manifest, output, key_file=key)
    image = cv2.imread(str(template), cv2.IMREAD_COLOR)
    document = LoadedDocument(
        [image],
        [""],
        1,
        [PageEvidence(1, image.shape[1], image.shape[0], text_spans=[])],
    )

    result = run_ssdd(
        GrammarStore(root, key_file=key),
        "nptel",
        document,
        ExtractionResult(),
    ).result

    assert result.status == "insufficient-evidence"
    assert result.delta is None
    assert "grammar-title-present-mismatch" in result.unavailable


def test_run_ssdd_evaluates_required_structural_rule(tmp_path) -> None:
    template = tmp_path / "source.png"
    manifest = tmp_path / "manifest.yaml"
    root = tmp_path / "grammar"
    output = root / "nptel.yaml"
    key = tmp_path / "hmac.key"
    key.write_bytes(b"x" * 32)
    template_image(template)
    builder_manifest(manifest)
    build_grammar_profile("nptel", "v1", template, manifest, output, key_file=key)
    image = cv2.imread(str(template), cv2.IMREAD_COLOR)
    evidence = PageEvidence(
        1,
        image.shape[1],
        image.shape[0],
        text_spans=[
            TextSpan(
                "CERTIFICATE",
                (0.28, 0.10, 0.35, 0.08),
                "tesseract",
                confidence=0.99,
            )
        ],
    )
    document = LoadedDocument([image], ["CERTIFICATE"], 1, [evidence])

    result = run_ssdd(
        GrammarStore(root, key_file=key),
        "nptel",
        document,
        ExtractionResult(text="CERTIFICATE"),
    ).result

    assert result.status == "completed"
    assert result.delta == 0
    assert result.issuer_grammar_match == 1
