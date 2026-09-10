import hashlib
import json
from dataclasses import replace
from importlib.resources import files
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from certguard import ssdd
from certguard.document import LoadedDocument, PageEvidence, TextSpan
from certguard.forensics import AlignmentContext
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
    CalibrationProfile,
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
            "dataset_hash": "a" * 64 if scoring else None,
            "dataset_version": "synthetic-1" if scoring else None,
            "sample_count": 100 if scoring else None,
            "false_positive_rate": 0.01 if scoring else None,
            "false_negative_rate": 0.05 if scoring else None,
            "review_threshold": 0.5 if scoring else None,
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

    assert (
        detect_text_qr_binding(
            fields,
            ["https://nptel.ac.in/cert?student=RAJ%20KUMAR"],
            binding,
        )
        == "pass"
    )
    assert (
        detect_text_qr_binding(
            fields,
            ["https://nptel.ac.in/cert?student=JOHN"],
            binding,
        )
        == "violation"
    )
    assert (
        detect_text_qr_binding(
            fields,
            [
                "https://nptel.ac.in/cert?student=RAJ%20KUMAR",
                "https://nptel.ac.in/cert?student=JOHN",
            ],
            binding,
        )
        == "violation"
    )


def test_text_qr_binding_supports_path_claims_and_rejects_unsafe_authority() -> None:
    binding = QRBinding(
        source_field="recipient",
        claim_key="student",
        trusted_hosts=("nptel.ac.in",),
        claim_source="path",
    )
    fields = {"recipient": "Raj Kumar"}

    assert (
        detect_text_qr_binding(
            fields,
            ["https://nptel.ac.in/cert/student/Raj%20Kumar"],
            binding,
        )
        == "pass"
    )
    assert (
        detect_text_qr_binding(
            fields,
            ["https://user:secret@nptel.ac.in/cert/student/Raj%20Kumar"],
            binding,
        )
        == "unavailable"
    )
    assert (
        detect_text_qr_binding(
            fields,
            ["https://nptel.ac.in:444/cert/student/Raj%20Kumar"],
            binding,
        )
        == "unavailable"
    )


@pytest.mark.parametrize(
    "calibration",
    [
        {"scoring_enabled": True, "benchmark_id": "bad", "high_points": -1},
        {"scoring_enabled": True, "benchmark_id": "bad", "high_points": 11},
        {"scoring_enabled": True, "benchmark_id": "bad", "medium_delta": 0.9, "high_delta": 0.5},
        {"scoring_enabled": True, "benchmark_id": "bad", "high_delta": float("nan")},
    ],
)
def test_profile_rejects_unsafe_calibration(tmp_path, calibration) -> None:
    path = tmp_path / "bad-calibration.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "issuer_id": "nptel",
                "variant_id": "bad",
                "active": False,
                "calibration": calibration,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(GrammarProfileError, match="Calibration"):
        load_profile(path)


@pytest.mark.parametrize(
    ("rule", "message"),
    [
        (
            {
                "id": "wrong-unit",
                "type": "region_presence",
                "first": "title",
                "unit": "mm",
                "value": True,
            },
            "does not support unit",
        ),
        (
            {
                "id": "wrong-value",
                "type": "region_presence",
                "first": "title",
                "unit": "boolean",
                "value": 1,
            },
            "value must be boolean",
        ),
        (
            {
                "id": "nan-tolerance",
                "type": "region_presence",
                "first": "title",
                "unit": "boolean",
                "value": True,
                "tolerance": float("nan"),
            },
            "must be finite",
        ),
    ],
)
def test_profile_rejects_rule_specific_schema_errors(tmp_path, rule, message) -> None:
    path = tmp_path / "bad-rule.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "issuer_id": "nptel",
                "variant_id": "bad",
                "active": False,
                "regions": {"title": {"box": [0.1, 0.1, 0.2, 0.2]}},
                "grammar": [rule],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(GrammarProfileError, match=message):
        load_profile(path)


def test_profile_requires_complete_calibration_metadata_for_scoring(tmp_path) -> None:
    path = tmp_path / "uncalibrated.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "issuer_id": "nptel",
                "variant_id": "uncalibrated",
                "active": False,
                "calibration": {"scoring_enabled": True, "benchmark_id": "only-one-field"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(GrammarProfileError, match="complete calibration metadata"):
        load_profile(path)


def test_builder_rejects_path_components_before_writing(tmp_path) -> None:
    template = tmp_path / "source.png"
    manifest = tmp_path / "manifest.yaml"
    output = tmp_path / "grammar" / "profile.yaml"
    key = tmp_path / "hmac.key"
    key.write_bytes(b"x" * 32)
    template_image(template)
    builder_manifest(manifest)

    with pytest.raises(GrammarProfileError, match="variant"):
        build_grammar_profile(
            "nptel",
            "../../outside",
            template,
            manifest,
            output,
            key_file=key,
        )

    assert not output.parent.exists()


def test_malformed_signature_manifest_has_domain_error(tmp_path) -> None:
    (tmp_path / ".signatures.json").write_text("{broken", encoding="utf-8")

    with pytest.raises(GrammarProfileError, match="signature manifest"):
        GrammarStore(tmp_path)


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


def test_builder_derives_canonical_geometry_and_refuses_overwrite(tmp_path) -> None:
    template = tmp_path / "source.png"
    manifest = tmp_path / "manifest.yaml"
    root = tmp_path / "grammar"
    output = root / "nptel.yaml"
    key = tmp_path / "hmac.key"
    key.write_bytes(b"x" * 32)
    template_image(template)
    data = {
        "active": False,
        "regions": {
            "title": {"box": [0.2, 0.1, 0.4, 0.1], "baseline": 0.18},
            "recipient": {"box": [0.25, 0.3, 0.5, 0.1], "baseline": 0.38},
        },
        "grammar": [
            {
                "id": "title-recipient-gap",
                "type": "vertical_gap",
                "first": "title",
                "second": "recipient",
                "unit": "normalized",
                "tolerance": 0.01,
            },
            {
                "id": "baseline-spacing",
                "type": "baseline_offset",
                "first": "title",
                "second": "recipient",
                "unit": "normalized",
                "tolerance": 0.01,
            },
        ],
    }
    manifest.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    build_grammar_profile("nptel", "v1", template, manifest, output, key_file=key)

    profile = load_profile(output)
    assert profile.rules[0].value == pytest.approx(0.1)
    assert profile.rules[1].value == pytest.approx(-0.2)
    with pytest.raises(GrammarProfileError, match="replace existing profile"):
        build_grammar_profile("nptel", "v1", template, manifest, output, key_file=key)


def test_builder_derives_physical_horizontal_geometry(tmp_path) -> None:
    template = tmp_path / "source.png"
    manifest = tmp_path / "manifest.yaml"
    output = tmp_path / "grammar" / "nptel.yaml"
    key = tmp_path / "hmac.key"
    key.write_bytes(b"x" * 32)
    template_image(template)
    manifest.write_text(
        yaml.safe_dump(
            {
                "active": False,
                "page_size_mm": [297, 210],
                "regions": {
                    "left": {"box": [0.1, 0.1, 0.1, 0.1]},
                    "right": {"box": [0.6, 0.1, 0.1, 0.1]},
                },
                "grammar": [
                    {
                        "id": "horizontal",
                        "type": "horizontal_alignment",
                        "first": "left",
                        "second": "right",
                        "unit": "mm",
                        "tolerance": 1,
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    build_grammar_profile("nptel", "v1", template, manifest, output, key_file=key)

    assert load_profile(output).rules[0].value == pytest.approx(-148.5)


def test_builder_validation_failure_leaves_no_output_artifacts(tmp_path) -> None:
    template = tmp_path / "source.png"
    manifest = tmp_path / "manifest.yaml"
    output = tmp_path / "grammar" / "nptel.yaml"
    key = tmp_path / "hmac.key"
    key.write_bytes(b"x" * 32)
    template_image(template)
    manifest.write_text(
        yaml.safe_dump(
            {
                "active": True,
                "regions": {"title": {"box": [0.1, 0.1, 0.2, 0.2]}},
                "grammar": [
                    {
                        "id": "bad",
                        "type": "region_presence",
                        "first": "title",
                        "unit": "boolean",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(GrammarProfileError, match="explicit tolerance"):
        build_grammar_profile("nptel", "v1", template, manifest, output, key_file=key)

    assert not output.exists()
    assert not output.parent.exists()


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
    for name in profile_names:
        raw = yaml.safe_load(root.joinpath(name).read_text(encoding="utf-8"))
        assert raw["annotation_requirements"]["regions"]
        assert raw["annotation_requirements"]["planned_rules"]
        assert raw["regions"] == {}
        assert raw["grammar"] == []


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
        SSDDResult(status="insufficient-evidence", delta=None, applicable_profile=True),
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

    assert result.status == "completed"
    assert result.delta == 1
    assert "grammar-title-present-mismatch" in result.violations


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


def profile_for_rules(tmp_path, rules, *, page_size_mm=None):
    path = tmp_path / "rules.yaml"
    template = {"page_size_mm": page_size_mm} if page_size_mm else {}
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "issuer_id": "nptel",
                "variant_id": "rules",
                "active": False,
                "template": template,
                "regions": {
                    "first": {"box": [0.0, 0.0, 0.5, 0.5]},
                    "second": {"box": [0.5, 0.5, 0.5, 0.5]},
                },
                "grammar": rules,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return load_profile(path)


def test_physical_geometry_uses_axes_and_is_unavailable_without_dimensions(tmp_path) -> None:
    rules = [
        {
            "id": "horizontal",
            "type": "horizontal_alignment",
            "first": "first",
            "second": "second",
            "value": -148.5,
            "unit": "mm",
            "tolerance": 0.001,
        },
        {
            "id": "distance",
            "type": "distance",
            "first": "first",
            "second": "second",
            "value": 0,
            "unit": "mm",
            "tolerance": 0,
        },
    ]
    profile = profile_for_rules(tmp_path, rules, page_size_mm=[297, 210])
    observed = {
        "first": ssdd.ObservedRegion((0.0, 0.0, 0.2, 0.2)),
        "second": ssdd.ObservedRegion((0.5, 0.0, 0.2, 0.2)),
    }
    evidence = PageEvidence(1, 1000, 1000)

    assert ssdd._evaluate_rule(profile.rules[0], observed, profile, evidence).state == "pass"
    assert ssdd._evaluate_rule(profile.rules[1], observed, profile, evidence).state == "violation"

    unknown = replace(profile, page_width_mm=None, page_height_mm=None)
    assert ssdd._evaluate_rule(profile.rules[0], observed, unknown, evidence).state == "unavailable"


def test_baseline_and_expected_false_region_presence(tmp_path) -> None:
    profile = profile_for_rules(
        tmp_path,
        [
            {
                "id": "baseline",
                "type": "baseline_offset",
                "first": "first",
                "second": "second",
                "value": -0.2,
                "unit": "normalized",
                "tolerance": 0,
            },
            {
                "id": "optional-region-absent",
                "type": "region_presence",
                "first": "first",
                "value": False,
                "unit": "boolean",
                "tolerance": 0,
            },
        ],
    )
    evidence = PageEvidence(1, 100, 100)
    observed = {
        "first": ssdd.ObservedRegion((0.1, 0.1, 0.2, 0.4), baseline=0.2),
        "second": ssdd.ObservedRegion((0.1, 0.4, 0.2, 0.1), baseline=0.4),
    }

    assert ssdd._evaluate_rule(profile.rules[0], observed, profile, evidence).state == "pass"
    assert ssdd._evaluate_rule(profile.rules[1], {"first": None}, profile, evidence).state == "pass"
    assert ssdd._evaluate_rule(profile.rules[1], observed, profile, evidence).state == "violation"


def test_span_homography_uses_all_corners_and_transforms_baseline() -> None:
    image = np.zeros((100, 100), dtype=np.uint8)
    homography = np.array([[1.0, 0.5, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 1.0]])
    alignment = AlignmentContext("test", 1.0, homography, image, 8, 8, 0.0)
    evidence = PageEvidence(1, 100, 100)
    span = TextSpan("text", (0.1, 0.1, 0.2, 0.2), "test", baseline=0.25)

    [mapped] = ssdd._aligned_spans([span], alignment, evidence)

    assert mapped.box == pytest.approx((0.15, 0.12, 0.3, 0.24))
    assert mapped.baseline == pytest.approx(0.29)


def test_span_homography_rejects_out_of_bounds_mapping() -> None:
    image = np.zeros((100, 100), dtype=np.uint8)
    homography = np.array([[1.0, 0.0, 95.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    alignment = AlignmentContext("test", 1.0, homography, image, 8, 8, 0.0)

    assert (
        ssdd._aligned_spans(
            [TextSpan("text", (0.1, 0.1, 0.2, 0.2), "test")],
            alignment,
            PageEvidence(1, 100, 100),
        )
        == []
    )


def test_region_aware_raster_and_vector_helpers_support_richer_evidence() -> None:
    evidence = SimpleNamespace(
        embedded_image_placements=[
            {"box": [0.0, 0.0, 0.4, 0.4], "effective_dpi": 300},
            {"box": [0.6, 0.6, 0.2, 0.2], "effective_dpi": 72},
        ],
        vector_regions=[{"box": [0.0, 0.0, 0.5, 0.5]}],
    )

    assert ssdd._region_raster_dpi(evidence, (0.0, 0.0, 0.5, 0.5)) == [300]
    assert ssdd._vector_region_coverage(evidence, (0.0, 0.0, 1.0, 1.0)) == pytest.approx(0.25)


def test_weighted_grammar_match_affects_result(tmp_path, monkeypatch) -> None:
    template = tmp_path / "source.png"
    manifest = tmp_path / "manifest.yaml"
    root = tmp_path / "grammar"
    output = root / "nptel.yaml"
    key = tmp_path / "hmac.key"
    key.write_bytes(b"x" * 32)
    template_image(template)
    builder_manifest(manifest)
    build_grammar_profile("nptel", "v1", template, manifest, output, key_file=key)
    profile = GrammarStore(root, key_file=key).profiles[0]
    image = cv2.imread(str(template), cv2.IMREAD_COLOR)
    alignment = AlignmentContext("test", 1.0, np.eye(3), image, 8, 8, 0.0)
    monkeypatch.setattr(
        ssdd,
        "_select_profile",
        lambda profiles, document: ssdd.SelectedProfile(profile, 0, alignment, image),
    )
    monkeypatch.setattr(
        ssdd,
        "_run_checks",
        lambda selected, evidence, extraction: [
            ssdd.SSDDCheck("grammar-heavy-mismatch", "pass", True, 3),
            ssdd.SSDDCheck("grammar-light-mismatch", "violation", True, 1),
        ],
    )
    document = LoadedDocument([image], [""], 1, [PageEvidence(1, 900, 600)])

    result = run_ssdd(
        GrammarStore(root, key_file=key), "nptel", document, ExtractionResult()
    ).result

    assert result.issuer_grammar_match == 0.75
    assert result.delta == 0.45


def test_scoring_profile_accepts_complete_calibration(tmp_path) -> None:
    template = tmp_path / "source.png"
    manifest = tmp_path / "manifest.yaml"
    output = tmp_path / "grammar" / "nptel.yaml"
    key = tmp_path / "hmac.key"
    key.write_bytes(b"x" * 32)
    template_image(template)
    builder_manifest(manifest, scoring=True)

    build_grammar_profile("nptel", "v1", template, manifest, output, key_file=key)

    assert load_profile(output).calibration == CalibrationProfile(
        scoring_enabled=True,
        benchmark_id="synthetic-v1",
        dataset_hash="a" * 64,
        dataset_version="synthetic-1",
        sample_count=100,
        false_positive_rate=0.01,
        false_negative_rate=0.05,
        review_threshold=0.5,
    )
