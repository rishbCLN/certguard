from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np
import yaml

from certguard.document import LoadedDocument, PageEvidence, TextSpan
from certguard.forensics import AlignmentContext, TemplateAnalyzer
from certguard.models import ExtractionResult, SSDDProfileRef, SSDDResult

DETECTOR_VERSION = "ssdd-v2.1"
SUPPORTED_RULES = {
    "vertical_gap",
    "horizontal_alignment",
    "distance",
    "baseline_offset",
    "relative_size",
    "region_presence",
    "embedded_raster_effective_dpi",
    "vector_presence",
    "vector_region_coverage",
}
SUPPORTED_UNITS = {"normalized", "pt", "mm", "dpi", "ratio", "boolean"}
SUPPORTED_NORMALIZERS = {"strip", "casefold", "upper", "spaces-to-underscore", "url-decode"}
SAFE_PROFILE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
RULE_UNITS = {
    "vertical_gap": {"normalized", "pt", "mm"},
    "horizontal_alignment": {"normalized", "pt", "mm"},
    "distance": {"normalized", "pt", "mm"},
    "baseline_offset": {"normalized", "pt", "mm"},
    "relative_size": {"ratio"},
    "region_presence": {"boolean"},
    "embedded_raster_effective_dpi": {"dpi"},
    "vector_presence": {"boolean"},
    "vector_region_coverage": {"ratio"},
}
RULES_REQUIRING_SECOND = {
    "vertical_gap",
    "horizontal_alignment",
    "distance",
    "baseline_offset",
    "relative_size",
}
KNOWN_ISSUER_IDS = {"coursera", "nptel"}
CALIBRATION_METADATA_FIELDS = (
    "benchmark_id",
    "dataset_hash",
    "dataset_version",
    "sample_count",
    "false_positive_rate",
    "false_negative_rate",
    "review_threshold",
)


class GrammarProfileError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RegionProfile:
    region_id: str
    box: tuple[float, float, float, float]
    kind: str = "text"
    selector: str | None = None


@dataclass(frozen=True, slots=True)
class GrammarRule:
    rule_id: str
    rule_type: str
    first: str
    second: str | None
    value: float | str | bool
    unit: str
    tolerance: float
    required: bool = False
    weight: float = 1.0
    applicability: tuple[tuple[str, bool], ...] = ()


@dataclass(frozen=True, slots=True)
class QRBinding:
    source_field: str
    claim_key: str
    trusted_hosts: tuple[str, ...]
    normalizers: tuple[str, ...] = ("strip", "casefold")
    required: bool = False
    binding_id: str = "claim"
    claim_source: str = "query"


@dataclass(frozen=True, slots=True)
class RenderingProfile:
    color_regions: tuple[str, str] | None = None
    color_tolerance: float = 15.0
    antialiasing_regions: tuple[str, str] | None = None
    antialiasing_ratio: float = 2.5
    texture_regions: tuple[str, str] | None = None
    texture_min_correlation: float = 0.3
    color_required: bool = False
    antialiasing_required: bool = False
    texture_required: bool = False


@dataclass(frozen=True, slots=True)
class CalibrationProfile:
    scoring_enabled: bool = False
    medium_delta: float = 0.5
    high_delta: float = 0.8
    medium_points: float = 7.0
    high_points: float = 10.0
    benchmark_id: str | None = None
    dataset_hash: str | None = None
    dataset_version: str | None = None
    sample_count: int | None = None
    false_positive_rate: float | None = None
    false_negative_rate: float | None = None
    review_threshold: float | None = None


@dataclass(frozen=True, slots=True)
class GrammarProfile:
    path: Path
    schema_version: int
    issuer_id: str
    variant_id: str
    display_name: str
    version: str
    active: bool
    template_path: Path | None
    template_sha256: str | None
    page_width_mm: float | None
    page_height_mm: float | None
    minimum_alignment: float
    minimum_evaluable_fraction: float
    regions: dict[str, RegionProfile]
    rules: tuple[GrammarRule, ...]
    qr_bindings: tuple[QRBinding, ...]
    rendering: RenderingProfile
    calibration: CalibrationProfile
    sha256: str


@dataclass(slots=True)
class SSDDCheck:
    code: str
    state: str
    grammar: bool = False
    weight: float = 1.0
    deviation: float | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ObservedRegion:
    box: tuple[float, float, float, float]
    baseline: float | None = None


@dataclass(slots=True)
class SSDDRun:
    result: SSDDResult
    profile: GrammarProfile | None = None


@dataclass(slots=True)
class SelectedProfile:
    profile: GrammarProfile
    page_index: int
    alignment: AlignmentContext
    template: np.ndarray


class GrammarStore:
    def __init__(
        self,
        root: Path | None = None,
        *,
        key_file: Path | None = None,
        require_signatures: bool = True,
    ) -> None:
        self.root = root.resolve() if root else None
        self.key_file = key_file
        self.require_signatures = require_signatures
        self.profiles: list[GrammarProfile] = []
        if self.root:
            self.profiles = self._load_all()

    @classmethod
    def from_environment(cls, root: Path | None) -> GrammarStore:
        key = os.environ.get("CERTGUARD_GRAMMAR_HMAC_KEY_FILE")
        return cls(root, key_file=Path(key) if key else None)

    def active_for(self, issuer_id: str | None) -> list[GrammarProfile]:
        if issuer_id is None:
            return []
        return [
            profile
            for profile in self.profiles
            if profile.active and profile.issuer_id == issuer_id
        ]

    def fingerprint_payload(self) -> list[dict[str, object]]:
        return [
            {
                "issuer_id": profile.issuer_id,
                "variant_id": profile.variant_id,
                "version": profile.version,
                "sha256": profile.sha256,
                "template_sha256": profile.template_sha256,
                "scoring_enabled": profile.calibration.scoring_enabled,
                "calibration": {
                    "benchmark_id": profile.calibration.benchmark_id,
                    "dataset_hash": profile.calibration.dataset_hash,
                    "dataset_version": profile.calibration.dataset_version,
                    "sample_count": profile.calibration.sample_count,
                    "false_positive_rate": profile.calibration.false_positive_rate,
                    "false_negative_rate": profile.calibration.false_negative_rate,
                    "review_threshold": profile.calibration.review_threshold,
                    "medium_delta": profile.calibration.medium_delta,
                    "high_delta": profile.calibration.high_delta,
                    "medium_points": profile.calibration.medium_points,
                    "high_points": profile.calibration.high_points,
                },
            }
            for profile in sorted(self.profiles, key=lambda item: (item.issuer_id, item.variant_id))
            if profile.active
        ]

    def _load_all(self) -> list[GrammarProfile]:
        if not self.root or not self.root.is_dir():
            raise GrammarProfileError(f"Grammar root does not exist: {self.root}")
        signatures = _load_signature_manifest(self.root)
        profiles: list[GrammarProfile] = []
        for path in sorted(self.root.glob("*.yaml")):
            if path.name.startswith("_"):
                continue
            profile = load_profile(path)
            if profile.active and self.require_signatures:
                _verify_profile_signature(profile, signatures, self.root, self.key_file)
                _verify_template(profile)
            profiles.append(profile)
        return profiles


def load_profile(path: Path) -> GrammarProfile:
    raw_bytes = path.read_bytes()
    try:
        data = yaml.safe_load(raw_bytes) or {}
    except yaml.YAMLError as exc:
        raise GrammarProfileError(f"Invalid grammar YAML {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise GrammarProfileError("Grammar profile must be a mapping")
    if data.get("schema_version") != 1:
        raise GrammarProfileError("Unsupported grammar schema_version")
    issuer_id = _required_string(data, "issuer_id")
    variant_id = _required_string(data, "variant_id")
    if issuer_id not in KNOWN_ISSUER_IDS:
        raise GrammarProfileError(f"Unknown issuer_id: {issuer_id}")
    if not SAFE_PROFILE_ID.fullmatch(variant_id):
        raise GrammarProfileError("variant_id must be filename-safe")
    template = data.get("template") or {}
    if not isinstance(template, dict):
        raise GrammarProfileError("template must be a mapping")
    template_path = _safe_relative_path(path.parent, template.get("path"))
    dimensions = template.get("page_size_mm") or []
    if dimensions and (not isinstance(dimensions, list) or len(dimensions) != 2):
        raise GrammarProfileError("template.page_size_mm must contain width and height")
    if dimensions:
        dimensions = [
            _finite_number(value, "template.page_size_mm", minimum=0, exclusive_minimum=True)
            for value in dimensions
        ]
    minimum_alignment = _finite_number(
        template.get("minimum_alignment", 0.35),
        "template.minimum_alignment",
        minimum=0,
        maximum=1,
    )

    regions: dict[str, RegionProfile] = {}
    raw_regions = data.get("regions") or {}
    if not isinstance(raw_regions, dict):
        raise GrammarProfileError("regions must be a mapping")
    for region_id, raw in raw_regions.items():
        if not isinstance(raw, dict):
            raise GrammarProfileError(f"Region {region_id} must be a mapping")
        box = _box(raw.get("box"), f"Region {region_id}")
        kind = str(raw.get("kind", "text"))
        if kind not in {"text", "qr", "visual"}:
            raise GrammarProfileError(f"Region {region_id} has unsupported kind: {kind}")
        selector = raw.get("selector")
        if selector is not None and (not isinstance(selector, str) or not selector):
            raise GrammarProfileError(f"Region {region_id} selector must be a non-empty string")
        if selector:
            try:
                re.compile(selector)
            except re.error as exc:
                raise GrammarProfileError(f"Region {region_id} selector is invalid: {exc}") from exc
        regions[str(region_id)] = RegionProfile(str(region_id), box, kind, selector)

    rules: list[GrammarRule] = []
    seen: set[str] = set()
    raw_rules = data.get("grammar") or []
    if not isinstance(raw_rules, list):
        raise GrammarProfileError("grammar must be a list")
    for raw in raw_rules:
        if not isinstance(raw, dict):
            raise GrammarProfileError("Each grammar rule must be a mapping")
        rule_id = _required_string(raw, "id")
        if rule_id in seen:
            raise GrammarProfileError(f"Duplicate grammar rule: {rule_id}")
        seen.add(rule_id)
        rule_type = _required_string(raw, "type")
        unit = str(raw.get("unit", "normalized"))
        if rule_type not in SUPPORTED_RULES or unit not in SUPPORTED_UNITS:
            raise GrammarProfileError(f"Unsupported grammar rule or unit: {rule_type}/{unit}")
        if unit not in RULE_UNITS[rule_type]:
            raise GrammarProfileError(f"Rule {rule_id} does not support unit {unit}")
        first = _required_string(raw, "first")
        second = str(raw["second"]) if raw.get("second") is not None else None
        if first not in regions or (second is not None and second not in regions):
            raise GrammarProfileError(f"Rule {rule_id} references an unknown region")
        if rule_type in RULES_REQUIRING_SECOND and second is None:
            raise GrammarProfileError(f"Rule {rule_id} requires second")
        if rule_type not in RULES_REQUIRING_SECOND and second is not None:
            raise GrammarProfileError(f"Rule {rule_id} does not support second")
        tolerance = _finite_number(raw.get("tolerance", 0), f"Rule {rule_id} tolerance", minimum=0)
        value = raw.get("value", True)
        if rule_type in {"region_presence", "vector_presence"}:
            if not isinstance(value, bool):
                raise GrammarProfileError(f"Rule {rule_id} value must be boolean")
            if tolerance != 0:
                raise GrammarProfileError(f"Rule {rule_id} boolean tolerance must be zero")
        else:
            value = _finite_number(value, f"Rule {rule_id} value")
            if rule_type in {"relative_size", "embedded_raster_effective_dpi"} and value <= 0:
                raise GrammarProfileError(f"Rule {rule_id} value must be positive")
            if rule_type == "vector_region_coverage" and not 0 <= value <= 1:
                raise GrammarProfileError(f"Rule {rule_id} coverage must be within [0, 1]")
        weight = _finite_number(
            raw.get("weight", 1), f"Rule {rule_id} weight", minimum=0, exclusive_minimum=True
        )
        applicability = _applicability(raw.get("applicability"), rule_id)
        rules.append(
            GrammarRule(
                rule_id,
                rule_type,
                first,
                second,
                value,
                unit,
                tolerance,
                _boolean(raw.get("required", False), f"Rule {rule_id} required"),
                weight,
                applicability,
            )
        )

    bindings: list[QRBinding] = []
    raw_bindings = data.get("qr_bindings") or []
    if not isinstance(raw_bindings, list):
        raise GrammarProfileError("qr_bindings must be a list")
    binding_ids: set[str] = set()
    for index, raw in enumerate(raw_bindings):
        if not isinstance(raw, dict):
            raise GrammarProfileError("Each QR binding must be a mapping")
        normalizers = tuple(raw.get("normalizers", ["strip", "casefold"]))
        if any(item not in SUPPORTED_NORMALIZERS for item in normalizers):
            raise GrammarProfileError("QR binding contains unsupported normalization")
        raw_hosts = raw.get("trusted_hosts", [])
        if not isinstance(raw_hosts, list):
            raise GrammarProfileError("QR binding trusted_hosts must be a list")
        hosts = tuple(_trusted_host(host) for host in raw_hosts)
        if not hosts:
            raise GrammarProfileError("QR binding requires trusted_hosts")
        binding_id = str(raw.get("id") or f"claim-{index + 1}")
        if not SAFE_PROFILE_ID.fullmatch(binding_id) or binding_id in binding_ids:
            raise GrammarProfileError("QR binding IDs must be unique and filename-safe")
        claim_source = str(raw.get("claim_source", "query"))
        if claim_source not in {"query", "path"}:
            raise GrammarProfileError("QR binding claim_source must be query or path")
        binding_ids.add(binding_id)
        bindings.append(
            QRBinding(
                _required_string(raw, "source_field"),
                _required_string(raw, "claim_key"),
                hosts,
                normalizers,
                _boolean(raw.get("required", False), f"QR binding {binding_id} required"),
                binding_id,
                claim_source,
            )
        )

    rendering_raw = data.get("rendering") or {}
    if not isinstance(rendering_raw, dict):
        raise GrammarProfileError("rendering must be a mapping")
    rendering = RenderingProfile(
        color_regions=_pair(rendering_raw.get("color_regions"), regions),
        color_tolerance=_finite_number(
            rendering_raw.get("color_tolerance", 15),
            "rendering.color_tolerance",
            minimum=0,
        ),
        antialiasing_regions=_pair(rendering_raw.get("antialiasing_regions"), regions),
        antialiasing_ratio=_finite_number(
            rendering_raw.get("antialiasing_ratio", 2.5),
            "rendering.antialiasing_ratio",
            minimum=1,
        ),
        texture_regions=_pair(rendering_raw.get("texture_regions"), regions),
        texture_min_correlation=_finite_number(
            rendering_raw.get("texture_min_correlation", 0.3),
            "rendering.texture_min_correlation",
            minimum=-1,
            maximum=1,
        ),
        color_required=_boolean(
            rendering_raw.get("color_required", False), "rendering.color_required"
        ),
        antialiasing_required=_boolean(
            rendering_raw.get("antialiasing_required", False),
            "rendering.antialiasing_required",
        ),
        texture_required=_boolean(
            rendering_raw.get("texture_required", False), "rendering.texture_required"
        ),
    )
    calibration_raw = data.get("calibration") or {}
    if not isinstance(calibration_raw, dict):
        raise GrammarProfileError("calibration must be a mapping")
    calibration = CalibrationProfile(
        scoring_enabled=_boolean(
            calibration_raw.get("scoring_enabled", False), "Calibration scoring_enabled"
        ),
        medium_delta=_finite_number(
            calibration_raw.get("medium_delta", 0.5), "Calibration medium_delta"
        ),
        high_delta=_finite_number(calibration_raw.get("high_delta", 0.8), "Calibration high_delta"),
        medium_points=_finite_number(
            calibration_raw.get("medium_points", 7), "Calibration medium_points"
        ),
        high_points=_finite_number(
            calibration_raw.get("high_points", 10), "Calibration high_points"
        ),
        benchmark_id=_optional_string(calibration_raw.get("benchmark_id")),
        dataset_hash=_optional_string(calibration_raw.get("dataset_hash")),
        dataset_version=_optional_string(calibration_raw.get("dataset_version")),
        sample_count=_optional_integer(calibration_raw.get("sample_count"), "sample_count"),
        false_positive_rate=_optional_number(calibration_raw.get("false_positive_rate")),
        false_negative_rate=_optional_number(calibration_raw.get("false_negative_rate")),
        review_threshold=_optional_number(calibration_raw.get("review_threshold")),
    )
    _validate_calibration(calibration, calibration_raw)
    fraction = _finite_number(
        data.get("minimum_evaluable_fraction", 0.75),
        "minimum_evaluable_fraction",
        minimum=0,
        maximum=1,
        exclusive_minimum=True,
    )
    return GrammarProfile(
        path=path.resolve(),
        schema_version=1,
        issuer_id=issuer_id,
        variant_id=variant_id,
        display_name=str(data.get("display_name", variant_id)),
        version=str(data.get("version", "1.0")),
        active=_boolean(data.get("active", False), "active"),
        template_path=template_path,
        template_sha256=str(template.get("sha256")) if template.get("sha256") else None,
        page_width_mm=float(dimensions[0]) if dimensions else None,
        page_height_mm=float(dimensions[1]) if dimensions else None,
        minimum_alignment=minimum_alignment,
        minimum_evaluable_fraction=fraction,
        regions=regions,
        rules=tuple(rules),
        qr_bindings=tuple(bindings),
        rendering=rendering,
        calibration=calibration,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def run_ssdd(
    store: GrammarStore,
    issuer_id: str | None,
    document: LoadedDocument,
    extraction: ExtractionResult,
) -> SSDDRun:
    profiles = store.active_for(issuer_id)
    if not profiles:
        return SSDDRun(SSDDResult())
    selected = _select_profile(profiles, document)
    if selected is None:
        profile = profiles[0]
        return SSDDRun(
            _insufficient_result(profile, ["template-alignment-unavailable"], len(profile.rules)),
            profile,
        )
    evidence = (document.evidence or [])[selected.page_index]
    checks = _run_checks(selected, evidence, extraction)
    required_unavailable = [
        check.code
        for check in checks
        if check.state in {"unavailable", "error"} and _is_required(check.code, selected.profile)
    ]
    evaluated = [check for check in checks if check.state in {"pass", "violation"}]
    possible = len(checks)
    fraction = len(evaluated) / possible if possible else 0
    unavailable = sorted(check.code for check in checks if check.state in {"unavailable", "error"})
    if required_unavailable or fraction < selected.profile.minimum_evaluable_fraction:
        return SSDDRun(
            _insufficient_result(selected.profile, unavailable, possible, len(evaluated)),
            selected.profile,
        )
    violations = sorted({check.code for check in evaluated if check.state == "violation"})
    grammar_checks = [check for check in evaluated if check.grammar]
    grammar_weight = sum(check.weight for check in grammar_checks)
    grammar_match = (
        sum(check.weight for check in grammar_checks if check.state == "pass") / grammar_weight
        if grammar_weight
        else 1.0
    )
    delta = (len(violations) / len(evaluated)) * 0.8 + (1 - grammar_match) * 0.2
    points = _risk_points(selected.profile.calibration, delta)
    result = SSDDResult(
        status="completed",
        delta=round(delta, 3),
        violations=violations,
        issuer_grammar_match=round(grammar_match, 3),
        model_sha256=selected.profile.sha256,
        profile=_profile_ref(selected.profile),
        checks_possible=possible,
        checks_evaluated=len(evaluated),
        unavailable=unavailable,
        scoring_enabled=selected.profile.calibration.scoring_enabled,
        risk_points=points,
        applicable_profile=True,
        required_binding_violations=sorted(
            check.code
            for check in checks
            if check.state == "violation" and _is_required(check.code, selected.profile)
        ),
    )
    return SSDDRun(result, selected.profile)


def detect_text_qr_binding(
    structured_fields: dict[str, str], qr_values: list[str], binding: QRBinding
) -> str:
    source = structured_fields.get(binding.source_field)
    if not source:
        return "unavailable"
    observed_claims: list[str] = []
    for raw_value in qr_values:
        parsed = urlparse(raw_value)
        try:
            port = parsed.port
        except ValueError:
            continue
        if (
            parsed.scheme.casefold() != "https"
            or parsed.username is not None
            or parsed.password is not None
            or port not in {None, 443}
            or (parsed.hostname or "").casefold() not in binding.trusted_hosts
        ):
            continue
        if binding.claim_source == "query":
            claims = parse_qs(parsed.query, keep_blank_values=True)
            values = claims.get(binding.claim_key)
        else:
            values = _path_claims(parsed.path, binding.claim_key)
        if not values:
            continue
        observed_claims.extend(_normalize(value, binding.normalizers) for value in values)
    if not observed_claims:
        return "unavailable"
    expected = _normalize(source, binding.normalizers)
    return "pass" if all(value == expected for value in observed_claims) else "violation"


def color_diff_ciede2000(left_bgr: np.ndarray, right_bgr: np.ndarray) -> float:
    left = cv2.cvtColor(np.uint8([[left_bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    right = cv2.cvtColor(np.uint8([[right_bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    left = np.array([left[0] * 100 / 255, left[1] - 128, left[2] - 128])
    right = np.array([right[0] * 100 / 255, right[1] - 128, right[2] - 128])
    return _delta_e_2000(left, right)


def compute_aa_variance(region: np.ndarray) -> float | None:
    if region.size == 0:
        return None
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region
    if np.count_nonzero(cv2.Canny(gray, 80, 160)) < 20:
        return None
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return float(np.var(cv2.Laplacian(normalized, cv2.CV_64F)))


def haar_texture_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if left.ndim == 3 else left
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if right.ndim == 3 else right
    size = (
        max(8, min(left_gray.shape[1], right_gray.shape[1])),
        max(8, min(left_gray.shape[0], right_gray.shape[0])),
    )
    left_gray = cv2.resize(left_gray, size).astype(float)
    right_gray = cv2.resize(right_gray, size).astype(float)
    left_detail = _haar_detail(left_gray)
    right_detail = _haar_detail(right_gray)
    if left_detail.std() < 1e-6 or right_detail.std() < 1e-6:
        return None
    return float(np.clip(np.corrcoef(left_detail.ravel(), right_detail.ravel())[0, 1], -1, 1))


def build_grammar_profile(
    issuer_id: str,
    variant_id: str,
    template_path: Path,
    manifest_path: Path,
    output_path: Path,
    *,
    key_file: Path,
) -> Path:
    for label, value in (("issuer", issuer_id), ("variant", variant_id)):
        if not SAFE_PROFILE_ID.fullmatch(value):
            raise GrammarProfileError(
                f"{label} must contain only letters, numbers, dots, underscores, and hyphens"
            )
    if issuer_id not in KNOWN_ISSUER_IDS:
        raise GrammarProfileError(f"Unknown issuer_id: {issuer_id}")
    if output_path.exists():
        raise GrammarProfileError(f"Refusing to replace existing profile: {output_path}")
    if not template_path.is_file():
        raise GrammarProfileError(f"Template does not exist: {template_path}")
    key = _read_key(key_file)
    try:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise GrammarProfileError(f"Invalid builder manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise GrammarProfileError("Builder manifest must be a mapping")
    regions = manifest.get("regions", {})
    if not isinstance(regions, dict):
        raise GrammarProfileError("Builder manifest regions must be a mapping")
    grammar = _derive_canonical_grammar(
        manifest.get("grammar", []), regions, manifest.get("page_size_mm")
    )
    template_hash = _sha256_file(template_path)
    template_name = (
        f"{issuer_id}-{variant_id}-{template_hash[:12]}{template_path.suffix.casefold()}"
    )
    packaged_template = output_path.parent / "templates" / template_name
    if packaged_template.exists() and _sha256_file(packaged_template) != template_hash:
        raise GrammarProfileError(f"Refusing to replace different template: {packaged_template}")
    data = {
        "schema_version": 1,
        "issuer_id": issuer_id,
        "variant_id": variant_id,
        "display_name": manifest.get("display_name", variant_id),
        "version": str(manifest.get("version", "1.0")),
        "active": bool(manifest.get("active", True)),
        "template": {
            "path": os.path.relpath(
                packaged_template.resolve(), output_path.parent.resolve()
            ).replace("\\", "/"),
            "sha256": template_hash,
            "page_size_mm": manifest.get("page_size_mm"),
            "minimum_alignment": manifest.get("minimum_alignment", 0.35),
        },
        "minimum_evaluable_fraction": manifest.get("minimum_evaluable_fraction", 0.75),
        "regions": regions,
        "grammar": grammar,
        "qr_bindings": manifest.get("qr_bindings", []),
        "rendering": manifest.get("rendering", {}),
        "calibration": manifest.get("calibration", {"scoring_enabled": False}),
    }
    payload = yaml.safe_dump(data, sort_keys=False).encode("utf-8")
    _validate_built_profile(data, template_path, payload)
    signature_path = output_path.parent / ".signatures.json"
    signatures = _load_signature_manifest(output_path.parent)
    relative = output_path.name
    if relative in signatures:
        raise GrammarProfileError(f"Refusing to replace existing signature: {relative}")
    profile_hash = hashlib.sha256(payload).hexdigest()
    signatures[relative] = {
        "sha256": profile_hash,
        "hmac_sha256": hmac.new(key, payload, hashlib.sha256).hexdigest(),
    }
    signature_payload = json.dumps(signatures, indent=2, sort_keys=True).encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    packaged_template.parent.mkdir(parents=True, exist_ok=True)
    copied_template = False
    try:
        if not packaged_template.exists():
            _atomic_copy(template_path, packaged_template)
            copied_template = True
        _atomic_write(output_path, payload)
        _atomic_write(signature_path, signature_payload)
        profile = load_profile(output_path)
        _verify_template(profile)
    except Exception:
        output_path.unlink(missing_ok=True)
        if copied_template:
            packaged_template.unlink(missing_ok=True)
        raise
    return output_path


def _select_profile(
    profiles: list[GrammarProfile], document: LoadedDocument
) -> SelectedProfile | None:
    best: SelectedProfile | None = None
    for profile in profiles:
        if profile.template_path is None:
            continue
        template = cv2.imread(str(profile.template_path), cv2.IMREAD_GRAYSCALE)
        if template is None:
            continue
        for page_index, image in enumerate(document.images):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            alignment = TemplateAnalyzer.align_images(gray, template)
            if alignment is None or alignment.score < profile.minimum_alignment:
                continue
            candidate = SelectedProfile(profile, page_index, alignment, template)
            if best is None or alignment.score > best.alignment.score:
                best = candidate
    return best


def _run_checks(
    selected: SelectedProfile, evidence: PageEvidence, extraction: ExtractionResult
) -> list[SSDDCheck]:
    profile = selected.profile
    aligned_spans = _aligned_spans(evidence.text_spans or [], selected.alignment, evidence)
    aligned_qr = _aligned_qr_boxes(evidence.qr_observations or [], selected.alignment, evidence)
    observed = {
        name: _observe_region(
            region,
            aligned_spans,
            aligned_qr,
            selected.alignment.aligned_image,
        )
        for name, region in profile.regions.items()
    }
    checks: list[SSDDCheck] = []
    for rule in profile.rules:
        if not _rule_applies(rule, profile, evidence):
            continue
        try:
            checks.append(_evaluate_rule(rule, observed, profile, evidence))
        except (AttributeError, TypeError, ValueError, cv2.error) as exc:
            checks.append(
                SSDDCheck(
                    f"grammar-{rule.rule_id}-mismatch",
                    "error",
                    True,
                    rule.weight,
                    error=type(exc).__name__,
                )
            )
    for binding in profile.qr_bindings:
        state = detect_text_qr_binding(
            extraction.structured_fields,
            [item.value for item in evidence.qr_observations or []],
            binding,
        )
        checks.append(SSDDCheck(f"text-qr-binding-{binding.binding_id}-failure", state))
    image = selected.alignment.aligned_image
    rendering = profile.rendering
    if rendering.color_regions:
        first, second = (
            _crop_region(image, profile.regions[name].box) for name in rendering.color_regions
        )
        colors = (_foreground_color(first), _foreground_color(second))
        state = (
            "unavailable"
            if any(value is None for value in colors)
            else (
                "pass"
                if color_diff_ciede2000(colors[0], colors[1]) <= rendering.color_tolerance
                else "violation"
            )
        )
        checks.append(SSDDCheck("logo-font-color-mismatch", state))
    if rendering.antialiasing_regions:
        values = [
            compute_aa_variance(_crop_region(image, profile.regions[name].box))
            for name in rendering.antialiasing_regions
        ]
        if any(value is None or value <= 0 for value in values):
            state = "unavailable"
        else:
            state = (
                "pass" if max(values) / min(values) <= rendering.antialiasing_ratio else "violation"
            )
        checks.append(SSDDCheck("subpixel-kernel-mismatch", state))
    if rendering.texture_regions:
        regions = [
            _crop_region(image, profile.regions[name].box) for name in rendering.texture_regions
        ]
        correlation = haar_texture_correlation(*regions)
        state = (
            "unavailable"
            if correlation is None
            else ("pass" if correlation >= rendering.texture_min_correlation else "violation")
        )
        checks.append(SSDDCheck("ink-bleed-decorrelation", state))
    return checks


def _evaluate_rule(
    rule: GrammarRule,
    observed: dict[str, ObservedRegion | None],
    profile: GrammarProfile,
    evidence: PageEvidence,
) -> SSDDCheck:
    first = observed.get(rule.first)
    second = observed.get(rule.second) if rule.second else None
    code = f"grammar-{rule.rule_id}-mismatch"
    if rule.rule_type == "region_presence":
        actual = first is not None
        return SSDDCheck(
            code,
            "pass" if actual == rule.value else "violation",
            True,
            rule.weight,
            float(actual != rule.value),
        )
    if rule.rule_type == "embedded_raster_effective_dpi":
        values = _region_raster_dpi(evidence, profile.regions[rule.first].box)
        if not values:
            return SSDDCheck(code, "unavailable", True, rule.weight)
        return _numeric_check(
            code, float(np.median(values)), float(rule.value), rule.tolerance, rule.weight
        )
    if rule.rule_type in {"vector_presence", "vector_region_coverage"}:
        coverage = _vector_region_coverage(evidence, profile.regions[rule.first].box)
        if coverage is None:
            return SSDDCheck(code, "unavailable", True, rule.weight)
        if rule.rule_type == "vector_presence":
            actual = coverage > 0
            return SSDDCheck(
                code,
                "pass" if actual == rule.value else "violation",
                True,
                rule.weight,
                float(actual != rule.value),
            )
        return _numeric_check(code, coverage, float(rule.value), rule.tolerance, rule.weight)
    if first is None or (rule.second and second is None):
        return SSDDCheck(code, "unavailable", True, rule.weight)
    value = _measure_rule(rule.rule_type, first, second, rule.unit, profile, evidence)
    if value is None:
        return SSDDCheck(code, "unavailable", True, rule.weight)
    return _numeric_check(code, value, float(rule.value), rule.tolerance, rule.weight)


def _relationship(
    rule_type: str,
    first: ObservedRegion,
    second: ObservedRegion | None,
) -> float | None:
    if second is None:
        return None
    ax, ay, aw, ah = first.box
    bx, by, bw, bh = second.box
    if rule_type == "vertical_gap":
        return by - (ay + ah)
    if rule_type == "horizontal_alignment":
        return (ax + aw / 2) - (bx + bw / 2)
    if rule_type == "distance":
        return math.hypot((ax + aw / 2) - (bx + bw / 2), (ay + ah / 2) - (by + bh / 2))
    if rule_type == "baseline_offset":
        if first.baseline is None or second.baseline is None:
            return None
        return first.baseline - second.baseline
    if rule_type == "relative_size":
        return (aw * ah) / max(bw * bh, 1e-9)
    return None


def _measure_rule(
    rule_type: str,
    first: ObservedRegion,
    second: ObservedRegion | None,
    unit: str,
    profile: GrammarProfile,
    evidence: PageEvidence,
) -> float | None:
    value = _relationship(rule_type, first, second)
    if value is None or unit in {"normalized", "ratio"}:
        return value
    dimensions = _physical_dimensions_mm(profile, evidence)
    if dimensions is None or second is None:
        return None
    width_mm, height_mm = dimensions
    ax, ay, aw, ah = first.box
    bx, by, bw, bh = second.box
    if rule_type == "horizontal_alignment":
        millimeters = ((ax + aw / 2) - (bx + bw / 2)) * width_mm
    elif rule_type == "distance":
        dx = ((ax + aw / 2) - (bx + bw / 2)) * width_mm
        dy = ((ay + ah / 2) - (by + bh / 2)) * height_mm
        millimeters = math.hypot(dx, dy)
    else:
        millimeters = value * height_mm
    return millimeters / 0.3527777778 if unit == "pt" else millimeters


def _aligned_spans(
    spans: list[TextSpan], alignment: AlignmentContext, evidence: PageEvidence
) -> list[TextSpan]:
    output: list[TextSpan] = []
    width = alignment.aligned_image.shape[1]
    height = alignment.aligned_image.shape[0]
    for span in spans:
        x, y, w, h = span.box
        corners = np.float32(
            [
                [
                    [x * evidence.width_px, y * evidence.height_px],
                    [(x + w) * evidence.width_px, y * evidence.height_px],
                    [(x + w) * evidence.width_px, (y + h) * evidence.height_px],
                    [x * evidence.width_px, (y + h) * evidence.height_px],
                ]
            ]
        )
        transformed = _transform_points(corners, alignment, evidence)
        if transformed is None:
            continue
        x0, y0 = transformed.min(axis=0)
        x1, y1 = transformed.max(axis=0)
        baseline = None
        if span.baseline is not None and math.isfinite(span.baseline):
            baseline_points = np.float32(
                [
                    [
                        [x * evidence.width_px, span.baseline * evidence.height_px],
                        [(x + w) * evidence.width_px, span.baseline * evidence.height_px],
                    ]
                ]
            )
            transformed_baseline = _transform_points(baseline_points, alignment, evidence)
            if transformed_baseline is not None:
                baseline = float(np.mean(transformed_baseline[:, 1]) / height)
        output.append(
            TextSpan(
                span.text,
                (
                    x0 / width,
                    y0 / height,
                    (x1 - x0) / width,
                    (y1 - y0) / height,
                ),
                span.source,
                span.confidence,
                baseline,
                span.font_name,
                span.font_size,
            )
        )
    return output


def _transform_points(
    points: np.ndarray, alignment: AlignmentContext, evidence: PageEvidence
) -> np.ndarray | None:
    if evidence.width_px <= 0 or evidence.height_px <= 0:
        return None
    homography = np.asarray(alignment.homography, dtype=np.float64)
    if homography.shape != (3, 3) or not np.isfinite(homography).all():
        return None
    try:
        transformed = cv2.perspectiveTransform(points, homography)[0]
    except cv2.error:
        return None
    height, width = alignment.aligned_image.shape[:2]
    return _validate_mapped_points(transformed, width, height)


def _validate_mapped_points(points: np.ndarray, width: int, height: int) -> np.ndarray | None:
    points = np.asarray(points, dtype=float)
    if (
        width <= 0
        or height <= 0
        or points.ndim != 2
        or points.shape[1:] != (2,)
        or not np.isfinite(points).all()
    ):
        return None
    epsilon = 1e-3
    if (
        np.any(points[:, 0] < -epsilon)
        or np.any(points[:, 1] < -epsilon)
        or np.any(points[:, 0] > width + epsilon)
        or np.any(points[:, 1] > height + epsilon)
    ):
        return None
    x_range = float(np.ptp(points[:, 0]))
    y_range = float(np.ptp(points[:, 1]))
    if len(points) >= 4 and (x_range <= epsilon or y_range <= epsilon):
        return None
    return points


def _observe_region(
    region: RegionProfile,
    spans: list[TextSpan],
    qr_boxes: list[tuple[float, float, float, float]],
    image: np.ndarray,
) -> ObservedRegion | None:
    if region.kind == "qr":
        box = next((box for box in qr_boxes if _center_in(box, region.box)), None)
        return ObservedRegion(box) if box else None
    if region.kind == "visual":
        box = _observe_visual_region(image, region.box)
        return ObservedRegion(box) if box else None
    matches = [span for span in spans if _center_in(span.box, region.box)]
    if region.selector:
        pattern = re.compile(region.selector, re.IGNORECASE)
        matches = [span for span in matches if pattern.search(span.text)]
    if not matches:
        return None
    x0 = min(span.box[0] for span in matches)
    y0 = min(span.box[1] for span in matches)
    x1 = max(span.box[0] + span.box[2] for span in matches)
    y1 = max(span.box[1] + span.box[3] for span in matches)
    baselines = [span.baseline for span in matches if span.baseline is not None]
    baseline = float(np.median(baselines)) if baselines else None
    return ObservedRegion((x0, y0, x1 - x0, y1 - y0), baseline)


def _aligned_qr_boxes(
    observations,
    alignment: AlignmentContext,
    evidence: PageEvidence,
) -> list[tuple[float, float, float, float]]:
    output: list[tuple[float, float, float, float]] = []
    width = alignment.aligned_image.shape[1]
    height = alignment.aligned_image.shape[0]
    for observation in observations:
        if len(observation.polygon) < 4:
            continue
        points = np.float32(
            [[[x * evidence.width_px, y * evidence.height_px] for x, y in observation.polygon]]
        )
        transformed = cv2.perspectiveTransform(points, alignment.homography)[0]
        mapped = _validate_mapped_points(transformed, width, height)
        if mapped is None:
            continue
        x0, y0 = mapped.min(axis=0)
        x1, y1 = mapped.max(axis=0)
        output.append((x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height))
    return output


def _observe_visual_region(
    image: np.ndarray, configured: tuple[float, float, float, float]
) -> tuple[float, float, float, float] | None:
    crop = _crop_region(image, configured)
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    foreground = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= 4]
    if not contours:
        return None
    local_x, local_y, local_width, local_height = cv2.boundingRect(np.vstack(contours))
    x, y, width, height = configured
    return (
        x + local_x / crop.shape[1] * width,
        y + local_y / crop.shape[0] * height,
        local_width / crop.shape[1] * width,
        local_height / crop.shape[0] * height,
    )


def _center_in(
    box: tuple[float, float, float, float], region: tuple[float, float, float, float]
) -> bool:
    x, y, w, h = box
    rx, ry, rw, rh = region
    return rx <= x + w / 2 <= rx + rw and ry <= y + h / 2 <= ry + rh


def _crop_region(image: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    x, y, width, height = box
    image_height, image_width = image.shape[:2]
    return image[
        max(0, int(y * image_height)) : min(image_height, int((y + height) * image_height)),
        max(0, int(x * image_width)) : min(image_width, int((x + width) * image_width)),
    ]


def _foreground_color(region: np.ndarray) -> np.ndarray | None:
    if region.size == 0:
        return None
    pixels = region.reshape(-1, 3)
    luminance = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).reshape(-1)
    threshold = np.percentile(luminance, 35)
    foreground = pixels[luminance <= threshold]
    return np.median(foreground, axis=0).astype(np.uint8) if foreground.size else None


def _haar_detail(image: np.ndarray) -> np.ndarray:
    image = image[: image.shape[0] // 2 * 2, : image.shape[1] // 2 * 2]
    return (image[0::2, 0::2] - image[1::2, 0::2] + image[0::2, 1::2] - image[1::2, 1::2]) / 2


def _delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> float:
    l1, a1, b1 = lab1
    l2, a2, b2 = lab2
    c1 = math.hypot(a1, b1)
    c2 = math.hypot(a2, b2)
    c_bar = (c1 + c2) / 2
    g = 0.5 * (1 - math.sqrt(c_bar**7 / (c_bar**7 + 25**7)))
    ap1, ap2 = (1 + g) * a1, (1 + g) * a2
    cp1, cp2 = math.hypot(ap1, b1), math.hypot(ap2, b2)
    hp1 = math.degrees(math.atan2(b1, ap1)) % 360
    hp2 = math.degrees(math.atan2(b2, ap2)) % 360
    dl = l2 - l1
    dc = cp2 - cp1
    dh_angle = hp2 - hp1
    if cp1 * cp2 == 0:
        dh_angle = 0
    elif dh_angle > 180:
        dh_angle -= 360
    elif dh_angle < -180:
        dh_angle += 360
    dh = 2 * math.sqrt(cp1 * cp2) * math.sin(math.radians(dh_angle / 2))
    l_bar = (l1 + l2) / 2
    c_bar_p = (cp1 + cp2) / 2
    if cp1 * cp2 == 0:
        h_bar = hp1 + hp2
    elif abs(hp1 - hp2) <= 180:
        h_bar = (hp1 + hp2) / 2
    elif hp1 + hp2 < 360:
        h_bar = (hp1 + hp2 + 360) / 2
    else:
        h_bar = (hp1 + hp2 - 360) / 2
    t = (
        1
        - 0.17 * math.cos(math.radians(h_bar - 30))
        + 0.24 * math.cos(math.radians(2 * h_bar))
        + 0.32 * math.cos(math.radians(3 * h_bar + 6))
        - 0.20 * math.cos(math.radians(4 * h_bar - 63))
    )
    sl = 1 + 0.015 * (l_bar - 50) ** 2 / math.sqrt(20 + (l_bar - 50) ** 2)
    sc = 1 + 0.045 * c_bar_p
    sh = 1 + 0.015 * c_bar_p * t
    rt = (
        -2
        * math.sqrt(c_bar_p**7 / (c_bar_p**7 + 25**7))
        * math.sin(math.radians(60 * math.exp(-(((h_bar - 275) / 25) ** 2))))
    )
    return math.sqrt((dl / sl) ** 2 + (dc / sc) ** 2 + (dh / sh) ** 2 + rt * (dc / sc) * (dh / sh))


def _insufficient_result(
    profile: GrammarProfile,
    unavailable: list[str],
    possible: int,
    evaluated: int = 0,
) -> SSDDResult:
    return SSDDResult(
        status="insufficient-evidence",
        delta=None,
        issuer_grammar_match=None,
        model_sha256=profile.sha256,
        profile=_profile_ref(profile),
        checks_possible=possible,
        checks_evaluated=evaluated,
        unavailable=sorted(set(unavailable)),
        scoring_enabled=profile.calibration.scoring_enabled,
        applicable_profile=True,
    )


def _profile_ref(profile: GrammarProfile) -> SSDDProfileRef:
    return SSDDProfileRef(profile.issuer_id, profile.variant_id, profile.version)


def _risk_points(calibration: CalibrationProfile, delta: float) -> float:
    if not calibration.scoring_enabled:
        return 0.0
    if delta >= calibration.high_delta:
        return calibration.high_points
    if delta >= calibration.medium_delta:
        return calibration.medium_points
    return 0.0


def _numeric_check(
    code: str, actual: float, expected: float, tolerance: float, weight: float
) -> SSDDCheck:
    difference = abs(actual - expected)
    scale = tolerance if tolerance > 0 else max(abs(expected), 1.0)
    deviation = difference / scale
    return SSDDCheck(
        code,
        _compare_numeric(actual, expected, tolerance),
        True,
        weight,
        float(deviation),
    )


def _physical_dimensions_mm(
    profile: GrammarProfile, evidence: PageEvidence
) -> tuple[float, float] | None:
    width_pt = getattr(evidence, "width_pt", None)
    height_pt = getattr(evidence, "height_pt", None)
    if (
        width_pt is not None
        and height_pt is not None
        and math.isfinite(width_pt)
        and math.isfinite(height_pt)
        and width_pt > 0
        and height_pt > 0
    ):
        return width_pt * 0.3527777778, height_pt * 0.3527777778
    if profile.page_width_mm is not None and profile.page_height_mm is not None:
        return profile.page_width_mm, profile.page_height_mm
    return None


def _rule_applies(rule: GrammarRule, profile: GrammarProfile, evidence: PageEvidence) -> bool:
    for key, expected in rule.applicability:
        if key == "physical_dimensions_known":
            actual = _physical_dimensions_mm(profile, evidence) is not None
        elif key == "pdf":
            actual = getattr(evidence, "width_pt", None) is not None
        elif key == "scan":
            actual = getattr(evidence, "width_pt", None) is None
        elif key == "has_embedded_raster":
            actual = bool(_all_raster_dpi(evidence))
        elif key == "has_vector":
            coverage = _vector_region_coverage(evidence, (0.0, 0.0, 1.0, 1.0))
            actual = coverage is not None and coverage > 0
        else:  # pragma: no cover - load_profile rejects unknown keys
            return False
        if actual != expected:
            return False
    return True


def _region_raster_dpi(
    evidence: PageEvidence, region: tuple[float, float, float, float]
) -> list[float]:
    placements = _first_evidence_attr(
        evidence,
        "embedded_image_placements",
        "embedded_rasters",
        "raster_regions",
    )
    if placements is not None:
        values: list[float] = []
        for item in placements:
            box = _evidence_box(item)
            dpi = _evidence_value(item, "effective_dpi", "dpi")
            if box is not None and dpi is not None and _boxes_intersect(box, region):
                values.append(dpi)
        return values
    return _all_raster_dpi(evidence) if region == (0.0, 0.0, 1.0, 1.0) else []


def _all_raster_dpi(evidence: PageEvidence) -> list[float]:
    values = getattr(evidence, "embedded_image_dpi", None) or []
    return [float(value) for value in values if _is_finite_number(value) and float(value) > 0]


def _vector_region_coverage(
    evidence: PageEvidence, region: tuple[float, float, float, float]
) -> float | None:
    vectors = _first_evidence_attr(
        evidence,
        "vector_regions",
        "vector_drawings",
        "vector_boxes",
    )
    if vectors is not None:
        region_area = region[2] * region[3]
        if region_area <= 0:
            return None
        area = 0.0
        for item in vectors:
            box = _evidence_box(item)
            coverage = _evidence_value(item, "coverage")
            if box is not None:
                area += _intersection_area(box, region)
            elif coverage is not None:
                area += coverage * region_area
        return min(1.0, max(0.0, area / region_area))
    count = getattr(evidence, "vector_drawing_count", None)
    if count is None:
        return None
    if count == 0:
        return 0.0
    if region == (0.0, 0.0, 1.0, 1.0):
        return 1.0
    return None


def _first_evidence_attr(evidence: PageEvidence, *names: str):
    for name in names:
        if hasattr(evidence, name):
            return getattr(evidence, name)
    return None


def _evidence_box(item: object) -> tuple[float, float, float, float] | None:
    value = item.get("box") if isinstance(item, dict) else getattr(item, "box", None)
    if isinstance(value, list | tuple) and len(value) == 4:
        box = tuple(float(part) for part in value)
        if all(math.isfinite(part) for part in box) and box[2] >= 0 and box[3] >= 0:
            return box  # type: ignore[return-value]
    return None


def _evidence_value(item: object, *names: str) -> float | None:
    for name in names:
        value = item.get(name) if isinstance(item, dict) else getattr(item, name, None)
        if _is_finite_number(value):
            return float(value)
    return None


def _boxes_intersect(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> bool:
    return _intersection_area(left, right) > 0


def _intersection_area(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    return max(0.0, min(lx + lw, rx + rw) - max(lx, rx)) * max(
        0.0, min(ly + lh, ry + rh) - max(ly, ry)
    )


def _is_required(code: str, profile: GrammarProfile) -> bool:
    if code.startswith("text-qr-binding-") and code.endswith("-failure"):
        binding_id = code[len("text-qr-binding-") : -len("-failure")]
        return any(
            binding.binding_id == binding_id and binding.required for binding in profile.qr_bindings
        )
    if code.startswith("grammar-") and code.endswith("-mismatch"):
        rule_id = code[len("grammar-") : -len("-mismatch")]
        return any(rule.rule_id == rule_id and rule.required for rule in profile.rules)
    if code == "logo-font-color-mismatch":
        return profile.rendering.color_required
    if code == "subpixel-kernel-mismatch":
        return profile.rendering.antialiasing_required
    if code == "ink-bleed-decorrelation":
        return profile.rendering.texture_required
    return False


def _compare_numeric(actual: float, expected: float, tolerance: float) -> str:
    return "pass" if abs(actual - expected) <= tolerance else "violation"


def _normalize(value: str, operations: tuple[str, ...]) -> str:
    for operation in operations:
        if operation == "strip":
            value = value.strip()
        elif operation == "casefold":
            value = value.casefold()
        elif operation == "upper":
            value = value.upper()
        elif operation == "spaces-to-underscore":
            value = re.sub(r"\s+", "_", value)
        elif operation == "url-decode":
            value = unquote(value)
    return value


def _path_claims(path: str, claim_key: str) -> list[str]:
    segments = [unquote(segment) for segment in path.split("/") if segment]
    return [
        segments[index + 1] for index, segment in enumerate(segments[:-1]) if segment == claim_key
    ]


def _applicability(value: object, rule_id: str) -> tuple[tuple[str, bool], ...]:
    if value in (None, {}):
        return ()
    if not isinstance(value, dict):
        raise GrammarProfileError(f"Rule {rule_id} applicability must be a mapping")
    supported = {
        "physical_dimensions_known",
        "pdf",
        "scan",
        "has_embedded_raster",
        "has_vector",
    }
    if any(key not in supported for key in value):
        raise GrammarProfileError(f"Rule {rule_id} applicability contains an unsupported condition")
    if any(not isinstance(expected, bool) for expected in value.values()):
        raise GrammarProfileError(f"Rule {rule_id} applicability values must be boolean")
    return tuple(sorted((str(key), expected) for key, expected in value.items()))


def _trusted_host(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise GrammarProfileError("QR binding trusted hosts must be non-empty strings")
    host = value.casefold().rstrip(".")
    parsed = urlparse(f"https://{host}")
    try:
        port = parsed.port
    except ValueError as exc:
        raise GrammarProfileError("QR binding trusted hosts must be exact hostnames") from exc
    if parsed.hostname != host or port is not None or parsed.username is not None:
        raise GrammarProfileError("QR binding trusted hosts must be exact hostnames")
    return host


def _validate_calibration(calibration: CalibrationProfile, raw: dict[str, object]) -> None:
    values = (
        calibration.medium_delta,
        calibration.high_delta,
        calibration.medium_points,
        calibration.high_points,
    )
    if not all(math.isfinite(value) for value in values):
        raise GrammarProfileError("Calibration values must be finite")
    if not 0 <= calibration.medium_delta <= calibration.high_delta <= 1:
        raise GrammarProfileError("Calibration deltas must satisfy 0 <= medium <= high <= 1")
    if not 0 <= calibration.medium_points <= calibration.high_points <= 10:
        raise GrammarProfileError("Calibration points must satisfy 0 <= medium <= high <= 10")
    for label, value in (
        ("false_positive_rate", calibration.false_positive_rate),
        ("false_negative_rate", calibration.false_negative_rate),
        ("review_threshold", calibration.review_threshold),
    ):
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            raise GrammarProfileError(f"Calibration {label} must be within [0, 1]")
    if calibration.sample_count is not None and calibration.sample_count <= 0:
        raise GrammarProfileError("Calibration sample_count must be positive")
    if calibration.scoring_enabled:
        missing = [field for field in CALIBRATION_METADATA_FIELDS if raw.get(field) in (None, "")]
        if missing:
            raise GrammarProfileError(
                "Scoring requires complete calibration metadata: " + ", ".join(missing)
            )


def _derive_canonical_grammar(
    raw_rules: object, regions: dict[str, object], page_size_mm: object = None
) -> list[dict[str, object]]:
    if raw_rules in (None, []):
        return []
    if not isinstance(raw_rules, list):
        raise GrammarProfileError("Builder manifest grammar must be a list")
    boxes: dict[str, tuple[float, float, float, float]] = {}
    baselines: dict[str, float] = {}
    for region_id, raw_region in regions.items():
        if not isinstance(raw_region, dict):
            raise GrammarProfileError(f"Region {region_id} must be a mapping")
        boxes[str(region_id)] = _box(raw_region.get("box"), f"Region {region_id}")
        if raw_region.get("baseline") is not None:
            baselines[str(region_id)] = _finite_number(
                raw_region["baseline"],
                f"Region {region_id} baseline",
                minimum=0,
                maximum=1,
            )
    output: list[dict[str, object]] = []
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, dict):
            raise GrammarProfileError("Each builder grammar rule must be a mapping")
        rule = dict(raw_rule)
        rule_id = _required_string(rule, "id")
        rule_type = _required_string(rule, "type")
        if rule_type not in SUPPORTED_RULES:
            raise GrammarProfileError(f"Unsupported grammar rule: {rule_type}")
        if "tolerance" not in rule:
            raise GrammarProfileError(f"Builder rule {rule_id} requires an explicit tolerance")
        if "value" not in rule:
            first_id = _required_string(rule, "first")
            if first_id not in boxes:
                raise GrammarProfileError(f"Rule {rule_id} references an unknown region")
            if rule_type in {"region_presence", "vector_presence"}:
                rule["value"] = True
            elif rule_type in RULES_REQUIRING_SECOND:
                second_id = _required_string(rule, "second")
                if second_id not in boxes:
                    raise GrammarProfileError(f"Rule {rule_id} references an unknown region")
                first = ObservedRegion(boxes[first_id], baselines.get(first_id))
                second = ObservedRegion(boxes[second_id], baselines.get(second_id))
                derived = _relationship(rule_type, first, second)
                if derived is None:
                    raise GrammarProfileError(
                        f"Builder rule {rule_id} cannot derive a canonical value from annotations"
                    )
                unit = str(rule.get("unit", "normalized"))
                if unit in {"normalized", "ratio"}:
                    rule["value"] = float(derived)
                elif unit in {"mm", "pt"}:
                    dimensions = _manifest_page_dimensions(page_size_mm)
                    if dimensions is None:
                        raise GrammarProfileError(
                            f"Builder rule {rule_id} needs page_size_mm for physical units"
                        )
                    width_mm, height_mm = dimensions
                    ax, ay, aw, ah = first.box
                    bx, by, bw, bh = second.box
                    if rule_type == "horizontal_alignment":
                        value = ((ax + aw / 2) - (bx + bw / 2)) * width_mm
                    elif rule_type == "distance":
                        dx = ((ax + aw / 2) - (bx + bw / 2)) * width_mm
                        dy = ((ay + ah / 2) - (by + bh / 2)) * height_mm
                        value = math.hypot(dx, dy)
                    else:
                        value = derived * height_mm
                    rule["value"] = value / 0.3527777778 if unit == "pt" else value
                else:
                    raise GrammarProfileError(f"Builder rule {rule_id} has an invalid unit")
            else:
                raise GrammarProfileError(
                    f"Builder rule {rule_id} needs an explicit evidence-derived value"
                )
        output.append(rule)
    return output


def _validate_built_profile(data: dict[str, object], template_path: Path, payload: bytes) -> None:
    with tempfile.TemporaryDirectory(prefix="certguard-grammar-") as temporary:
        root = Path(temporary)
        profile_path = root / "profile.yaml"
        validation_data = dict(data)
        validation_template = dict(validation_data["template"])  # type: ignore[arg-type]
        validation_template["path"] = template_path.name
        validation_data["template"] = validation_template
        profile_path.write_bytes(yaml.safe_dump(validation_data, sort_keys=False).encode("utf-8"))
        shutil.copy2(template_path, root / template_path.name)
        profile = load_profile(profile_path)
        _verify_template(profile)
    if not payload:
        raise GrammarProfileError("Builder produced an empty profile")


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest_page_dimensions(value: object) -> tuple[float, float] | None:
    if value in (None, []):
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise GrammarProfileError("Builder page_size_mm must contain width and height")
    return (
        _finite_number(value[0], "Builder page width", minimum=0, exclusive_minimum=True),
        _finite_number(value[1], "Builder page height", minimum=0, exclusive_minimum=True),
    )


def _load_signature_manifest(root: Path) -> dict[str, dict[str, str]]:
    path = root / ".signatures.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise GrammarProfileError(f"Invalid grammar signature manifest {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise GrammarProfileError("Grammar signature manifest must be a mapping")
    return data


def _verify_profile_signature(
    profile: GrammarProfile,
    signatures: dict[str, dict[str, str]],
    root: Path,
    key_file: Path | None,
) -> None:
    relative = profile.path.relative_to(root).as_posix()
    signature = signatures.get(relative)
    if not signature or key_file is None:
        raise GrammarProfileError(f"Active grammar profile is unsigned: {relative}")
    if not hmac.compare_digest(str(signature.get("sha256", "")), profile.sha256):
        raise GrammarProfileError(f"Grammar checksum mismatch: {relative}")
    expected = hmac.new(_read_key(key_file), profile.path.read_bytes(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(str(signature.get("hmac_sha256", "")), expected):
        raise GrammarProfileError(f"Grammar HMAC mismatch: {relative}")


def _verify_template(profile: GrammarProfile) -> None:
    if profile.template_path is None or profile.template_sha256 is None:
        raise GrammarProfileError(f"Active profile {profile.variant_id} requires a template hash")
    actual = _sha256_file(profile.template_path)
    if not hmac.compare_digest(actual, profile.template_sha256):
        raise GrammarProfileError(f"Template checksum mismatch: {profile.variant_id}")


def _read_key(path: Path) -> bytes:
    value = path.read_bytes().strip()
    if len(value) < 32:
        raise GrammarProfileError("Grammar HMAC key must contain at least 32 bytes")
    return value


def _safe_relative_path(root: Path, value: object) -> Path | None:
    if value in (None, ""):
        return None
    raw = Path(str(value))
    if raw.is_absolute():
        raise GrammarProfileError("Template path must be relative to the profile")
    resolved = (root / raw).resolve()
    if resolved == root.resolve() or root.resolve() not in resolved.parents:
        raise GrammarProfileError("Template path escapes the grammar directory")
    return resolved


def _box(value: object, label: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise GrammarProfileError(f"{label} box must have four values")
    box = tuple(float(item) for item in value)
    x, y, width, height = box
    if (
        not all(math.isfinite(item) for item in box)
        or min(box) < 0
        or width <= 0
        or height <= 0
        or x + width > 1
        or y + height > 1
    ):
        raise GrammarProfileError(f"{label} box must be normalized within the page")
    return box  # type: ignore[return-value]


def _pair(value: object, regions: dict[str, RegionProfile]) -> tuple[str, str] | None:
    if value in (None, []):
        return None
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(str(item) not in regions for item in value)
    ):
        raise GrammarProfileError("Rendering region pairs must reference two configured regions")
    return str(value[0]), str(value[1])


def _required_string(data: dict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise GrammarProfileError(f"{key} must be a non-empty string")
    return value.strip()


def _finite_number(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> float:
    if isinstance(value, bool):
        raise GrammarProfileError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise GrammarProfileError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise GrammarProfileError(f"{label} must be finite")
    if minimum is not None and (number <= minimum if exclusive_minimum else number < minimum):
        comparator = "greater than" if exclusive_minimum else "at least"
        raise GrammarProfileError(f"{label} must be {comparator} {minimum}")
    if maximum is not None and number > maximum:
        raise GrammarProfileError(f"{label} must be at most {maximum}")
    return number


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise GrammarProfileError("Calibration string metadata must be non-empty strings")
    return value.strip()


def _optional_number(value: object) -> float | None:
    return None if value is None else _finite_number(value, "Calibration metadata")


def _optional_integer(value: object, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise GrammarProfileError(f"Calibration {label} must be an integer")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise GrammarProfileError(f"{label} must be boolean")
    return value


def _is_finite_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
