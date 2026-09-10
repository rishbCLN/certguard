from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import shutil
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


@dataclass(frozen=True, slots=True)
class QRBinding:
    source_field: str
    claim_key: str
    trusted_hosts: tuple[str, ...]
    normalizers: tuple[str, ...] = ("strip", "casefold")
    required: bool = False


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
        return [profile for profile in self.profiles if profile.active and profile.issuer_id == issuer_id]

    def fingerprint_payload(self) -> list[dict[str, object]]:
        return [
            {
                "issuer_id": profile.issuer_id,
                "variant_id": profile.variant_id,
                "version": profile.version,
                "sha256": profile.sha256,
                "template_sha256": profile.template_sha256,
                "scoring_enabled": profile.calibration.scoring_enabled,
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
    data = yaml.safe_load(raw_bytes) or {}
    if not isinstance(data, dict):
        raise GrammarProfileError("Grammar profile must be a mapping")
    if data.get("schema_version") != 1:
        raise GrammarProfileError("Unsupported grammar schema_version")
    issuer_id = _required_string(data, "issuer_id")
    variant_id = _required_string(data, "variant_id")
    template = data.get("template") or {}
    if not isinstance(template, dict):
        raise GrammarProfileError("template must be a mapping")
    template_path = _safe_relative_path(path.parent, template.get("path"))
    dimensions = template.get("page_size_mm") or []
    if dimensions and (not isinstance(dimensions, list) or len(dimensions) != 2):
        raise GrammarProfileError("template.page_size_mm must contain width and height")

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
        regions[str(region_id)] = RegionProfile(
            str(region_id), box, kind, raw.get("selector")
        )

    rules: list[GrammarRule] = []
    seen: set[str] = set()
    for raw in data.get("grammar") or []:
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
        first = _required_string(raw, "first")
        second = str(raw["second"]) if raw.get("second") is not None else None
        if first not in regions or (second is not None and second not in regions):
            raise GrammarProfileError(f"Rule {rule_id} references an unknown region")
        tolerance = float(raw.get("tolerance", 0))
        if tolerance < 0:
            raise GrammarProfileError(f"Rule {rule_id} tolerance cannot be negative")
        rules.append(
            GrammarRule(
                rule_id,
                rule_type,
                first,
                second,
                raw.get("value", True),
                unit,
                tolerance,
                bool(raw.get("required", False)),
            )
        )

    bindings: list[QRBinding] = []
    for raw in data.get("qr_bindings") or []:
        normalizers = tuple(raw.get("normalizers", ["strip", "casefold"]))
        if any(item not in SUPPORTED_NORMALIZERS for item in normalizers):
            raise GrammarProfileError("QR binding contains unsupported normalization")
        hosts = tuple(str(host).casefold().rstrip(".") for host in raw.get("trusted_hosts", []))
        if not hosts:
            raise GrammarProfileError("QR binding requires trusted_hosts")
        bindings.append(
            QRBinding(
                _required_string(raw, "source_field"),
                _required_string(raw, "claim_key"),
                hosts,
                normalizers,
                bool(raw.get("required", False)),
            )
        )

    rendering_raw = data.get("rendering") or {}
    rendering = RenderingProfile(
        color_regions=_pair(rendering_raw.get("color_regions"), regions),
        color_tolerance=float(rendering_raw.get("color_tolerance", 15)),
        antialiasing_regions=_pair(rendering_raw.get("antialiasing_regions"), regions),
        antialiasing_ratio=float(rendering_raw.get("antialiasing_ratio", 2.5)),
        texture_regions=_pair(rendering_raw.get("texture_regions"), regions),
        texture_min_correlation=float(rendering_raw.get("texture_min_correlation", 0.3)),
        color_required=bool(rendering_raw.get("color_required", False)),
        antialiasing_required=bool(rendering_raw.get("antialiasing_required", False)),
        texture_required=bool(rendering_raw.get("texture_required", False)),
    )
    calibration_raw = data.get("calibration") or {}
    calibration = CalibrationProfile(
        scoring_enabled=bool(calibration_raw.get("scoring_enabled", False)),
        medium_delta=float(calibration_raw.get("medium_delta", 0.5)),
        high_delta=float(calibration_raw.get("high_delta", 0.8)),
        medium_points=float(calibration_raw.get("medium_points", 7)),
        high_points=float(calibration_raw.get("high_points", 10)),
        benchmark_id=calibration_raw.get("benchmark_id"),
    )
    if calibration.scoring_enabled and not calibration.benchmark_id:
        raise GrammarProfileError("Scoring requires calibration.benchmark_id")
    fraction = float(data.get("minimum_evaluable_fraction", 0.75))
    if not 0 < fraction <= 1:
        raise GrammarProfileError("minimum_evaluable_fraction must be within (0, 1]")
    return GrammarProfile(
        path=path.resolve(),
        schema_version=1,
        issuer_id=issuer_id,
        variant_id=variant_id,
        display_name=str(data.get("display_name", variant_id)),
        version=str(data.get("version", "1.0")),
        active=bool(data.get("active", False)),
        template_path=template_path,
        template_sha256=str(template.get("sha256")) if template.get("sha256") else None,
        page_width_mm=float(dimensions[0]) if dimensions else None,
        page_height_mm=float(dimensions[1]) if dimensions else None,
        minimum_alignment=float(template.get("minimum_alignment", 0.35)),
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
        if check.state == "unavailable" and _is_required(check.code, selected.profile)
    ]
    evaluated = [check for check in checks if check.state in {"pass", "violation"}]
    possible = len(checks)
    fraction = len(evaluated) / possible if possible else 0
    unavailable = sorted(check.code for check in checks if check.state == "unavailable")
    if required_unavailable or fraction < selected.profile.minimum_evaluable_fraction:
        return SSDDRun(_insufficient_result(selected.profile, unavailable, possible, len(evaluated)), selected.profile)
    violations = sorted({check.code for check in evaluated if check.state == "violation"})
    grammar_checks = [check for check in evaluated if check.grammar]
    grammar_match = (
        sum(check.state == "pass" for check in grammar_checks) / len(grammar_checks)
        if grammar_checks
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
    )
    return SSDDRun(result, selected.profile)


def detect_text_qr_binding(
    structured_fields: dict[str, str], qr_values: list[str], binding: QRBinding
) -> str:
    source = structured_fields.get(binding.source_field)
    if not source:
        return "unavailable"
    for raw_value in qr_values:
        parsed = urlparse(raw_value)
        if parsed.scheme.casefold() != "https" or (parsed.hostname or "").casefold() not in binding.trusted_hosts:
            continue
        claims = parse_qs(parsed.query, keep_blank_values=True)
        values = claims.get(binding.claim_key)
        if not values:
            continue
        return "pass" if _normalize(source, binding.normalizers) == _normalize(values[0], binding.normalizers) else "violation"
    return "unavailable"


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
    size = (max(8, min(left_gray.shape[1], right_gray.shape[1])), max(8, min(left_gray.shape[0], right_gray.shape[0])))
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
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if not isinstance(manifest, dict):
        raise GrammarProfileError("Builder manifest must be a mapping")
    template_hash = _sha256_file(template_path)
    template_name = f"{issuer_id}-{variant_id}-{template_hash[:12]}{template_path.suffix.casefold()}"
    packaged_template = output_path.parent / "templates" / template_name
    packaged_template.parent.mkdir(parents=True, exist_ok=True)
    if packaged_template.exists() and _sha256_file(packaged_template) != template_hash:
        raise GrammarProfileError(f"Refusing to replace different template: {packaged_template}")
    if not packaged_template.exists():
        shutil.copy2(template_path, packaged_template)
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
        "regions": manifest.get("regions", {}),
        "grammar": manifest.get("grammar", []),
        "qr_bindings": manifest.get("qr_bindings", []),
        "rendering": manifest.get("rendering", {}),
        "calibration": manifest.get("calibration", {"scoring_enabled": False}),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    profile = load_profile(output_path)
    _verify_template(profile)
    signature_path = output_path.parent / ".signatures.json"
    signatures = _load_signature_manifest(output_path.parent)
    key = _read_key(key_file)
    relative = output_path.name
    signatures[relative] = {
        "sha256": profile.sha256,
        "hmac_sha256": hmac.new(key, output_path.read_bytes(), hashlib.sha256).hexdigest(),
    }
    signature_path.write_text(json.dumps(signatures, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def _select_profile(profiles: list[GrammarProfile], document: LoadedDocument) -> SelectedProfile | None:
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
    aligned_qr = _aligned_qr_boxes(
        evidence.qr_observations or [], selected.alignment, evidence
    )
    observed = {
        name: _observe_region(
            region,
            aligned_spans,
            aligned_qr,
            selected.alignment.aligned_image,
        )
        for name, region in profile.regions.items()
    }
    checks = [_evaluate_rule(rule, observed, profile, evidence) for rule in profile.rules]
    for binding in profile.qr_bindings:
        state = detect_text_qr_binding(
            extraction.structured_fields,
            [item.value for item in evidence.qr_observations or []],
            binding,
        )
        checks.append(SSDDCheck("text-qr-binding-failure", state))
    image = selected.alignment.aligned_image
    rendering = profile.rendering
    if rendering.color_regions:
        first, second = (_crop_region(image, profile.regions[name].box) for name in rendering.color_regions)
        colors = (_foreground_color(first), _foreground_color(second))
        state = "unavailable" if any(value is None for value in colors) else (
            "pass" if color_diff_ciede2000(colors[0], colors[1]) <= rendering.color_tolerance else "violation"
        )
        checks.append(SSDDCheck("logo-font-color-mismatch", state))
    if rendering.antialiasing_regions:
        values = [compute_aa_variance(_crop_region(image, profile.regions[name].box)) for name in rendering.antialiasing_regions]
        if any(value is None or value <= 0 for value in values):
            state = "unavailable"
        else:
            state = "pass" if max(values) / min(values) <= rendering.antialiasing_ratio else "violation"
        checks.append(SSDDCheck("subpixel-kernel-mismatch", state))
    if rendering.texture_regions:
        regions = [_crop_region(image, profile.regions[name].box) for name in rendering.texture_regions]
        correlation = haar_texture_correlation(*regions)
        state = "unavailable" if correlation is None else (
            "pass" if correlation >= rendering.texture_min_correlation else "violation"
        )
        checks.append(SSDDCheck("ink-bleed-decorrelation", state))
    return checks


def _evaluate_rule(
    rule: GrammarRule,
    observed: dict[str, tuple[float, float, float, float] | None],
    profile: GrammarProfile,
    evidence: PageEvidence,
) -> SSDDCheck:
    first = observed.get(rule.first)
    second = observed.get(rule.second) if rule.second else None
    code = f"grammar-{rule.rule_id}-mismatch"
    if rule.rule_type == "embedded_raster_effective_dpi":
        values = evidence.embedded_image_dpi or []
        state = _compare_numeric(float(np.median(values)), float(rule.value), rule.tolerance) if values else "unavailable"
        return SSDDCheck(code, state, True)
    if rule.rule_type in {"vector_presence", "vector_region_coverage"}:
        if rule.rule_type == "vector_region_coverage":
            return SSDDCheck(code, "unavailable", True)
        actual = evidence.vector_drawing_count > 0
        return SSDDCheck(code, "pass" if actual == bool(rule.value) else "violation", True)
    if first is None or (rule.second and second is None):
        return SSDDCheck(code, "unavailable", True)
    if rule.rule_type == "region_presence":
        return SSDDCheck(code, "pass" if first is not None else "violation", True)
    value = _relationship(rule.rule_type, first, second)
    if value is None:
        return SSDDCheck(code, "unavailable", True)
    if isinstance(rule.value, str):
        expected = 0.0 if rule.value == "center" else None
        if expected is None:
            return SSDDCheck(code, "unavailable", True)
    else:
        expected = float(rule.value)
    value = _convert_unit(value, rule.unit, profile)
    return SSDDCheck(code, _compare_numeric(value, expected, rule.tolerance), True)


def _relationship(
    rule_type: str,
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float] | None,
) -> float | None:
    if second is None:
        return None
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    if rule_type == "vertical_gap":
        return by - (ay + ah)
    if rule_type == "horizontal_alignment":
        return (ax + aw / 2) - (bx + bw / 2)
    if rule_type == "distance":
        return math.hypot((ax + aw / 2) - (bx + bw / 2), (ay + ah / 2) - (by + bh / 2))
    if rule_type == "baseline_offset":
        return (ay + ah) - (by + bh)
    if rule_type == "relative_size":
        return (aw * ah) / max(bw * bh, 1e-9)
    return None


def _convert_unit(value: float, unit: str, profile: GrammarProfile) -> float:
    if unit in {"normalized", "ratio"}:
        return value
    if profile.page_height_mm is None:
        return value
    millimeters = value * profile.page_height_mm
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
            [[[x * evidence.width_px, y * evidence.height_px], [(x + w) * evidence.width_px, (y + h) * evidence.height_px]]]
        )
        transformed = cv2.perspectiveTransform(corners, alignment.homography)[0]
        x0, y0 = transformed[0]
        x1, y1 = transformed[1]
        output.append(
            TextSpan(
                span.text,
                (min(x0, x1) / width, min(y0, y1) / height, abs(x1 - x0) / width, abs(y1 - y0) / height),
                span.source,
                span.confidence,
                max(y0, y1) / height,
                span.font_name,
                span.font_size,
            )
        )
    return output


def _observe_region(
    region: RegionProfile,
    spans: list[TextSpan],
    qr_boxes: list[tuple[float, float, float, float]],
    image: np.ndarray,
) -> tuple[float, float, float, float] | None:
    if region.kind == "qr":
        return next((box for box in qr_boxes if _center_in(box, region.box)), None)
    if region.kind == "visual":
        return _observe_visual_region(image, region.box)
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
    return x0, y0, x1 - x0, y1 - y0


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
            [[
                [x * evidence.width_px, y * evidence.height_px]
                for x, y in observation.polygon
            ]]
        )
        transformed = cv2.perspectiveTransform(points, alignment.homography)[0]
        x, y, box_width, box_height = cv2.boundingRect(transformed)
        output.append((x / width, y / height, box_width / width, box_height / height))
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


def _center_in(box: tuple[float, float, float, float], region: tuple[float, float, float, float]) -> bool:
    x, y, w, h = box
    rx, ry, rw, rh = region
    return rx <= x + w / 2 <= rx + rw and ry <= y + h / 2 <= ry + rh


def _crop_region(image: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    x, y, width, height = box
    image_height, image_width = image.shape[:2]
    return image[
        max(0, int(y * image_height)):min(image_height, int((y + height) * image_height)),
        max(0, int(x * image_width)):min(image_width, int((x + width) * image_width)),
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
    t = 1 - 0.17 * math.cos(math.radians(h_bar - 30)) + 0.24 * math.cos(math.radians(2 * h_bar)) + 0.32 * math.cos(math.radians(3 * h_bar + 6)) - 0.20 * math.cos(math.radians(4 * h_bar - 63))
    sl = 1 + 0.015 * (l_bar - 50) ** 2 / math.sqrt(20 + (l_bar - 50) ** 2)
    sc = 1 + 0.045 * c_bar_p
    sh = 1 + 0.015 * c_bar_p * t
    rt = -2 * math.sqrt(c_bar_p**7 / (c_bar_p**7 + 25**7)) * math.sin(math.radians(60 * math.exp(-((h_bar - 275) / 25) ** 2)))
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


def _is_required(code: str, profile: GrammarProfile) -> bool:
    if code == "text-qr-binding-failure":
        return any(binding.required for binding in profile.qr_bindings)
    if code.startswith("grammar-") and code.endswith("-mismatch"):
        rule_id = code[len("grammar-"):-len("-mismatch")]
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


def _load_signature_manifest(root: Path) -> dict[str, dict[str, str]]:
    path = root / ".signatures.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


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
    if min(box) < 0 or width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
        raise GrammarProfileError(f"{label} box must be normalized within the page")
    return box  # type: ignore[return-value]


def _pair(value: object, regions: dict[str, RegionProfile]) -> tuple[str, str] | None:
    if value in (None, []):
        return None
    if not isinstance(value, list) or len(value) != 2 or any(str(item) not in regions for item in value):
        raise GrammarProfileError("Rendering region pairs must reference two configured regions")
    return str(value[0]), str(value[1])


def _required_string(data: dict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise GrammarProfileError(f"{key} must be a non-empty string")
    return value.strip()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
