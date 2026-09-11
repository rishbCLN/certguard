from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageChops
from skimage.metrics import structural_similarity

from certguard.models import ProvenanceResult, TemplateResult
from certguard.registry import IssuerDefinition


class ForgeryModel(Protocol):
    name: str

    def predict(self, image: np.ndarray) -> float: ...


@dataclass(slots=True)
class AlignmentContext:
    detector: str
    score: float
    homography: np.ndarray
    aligned_image: np.ndarray
    inlier_count: int
    match_count: int
    reprojection_error: float
    source_dimensions: tuple[int, int] | None = None
    template_dimensions: tuple[int, int] | None = None
    inlier_ratio: float | None = None
    detector_metadata: dict[str, float | int | str] | None = None
    coordinate_system: str = "source-pixels-to-template-pixels"


class OnnxForgeryModel:
    def __init__(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        self.name = model_path.name
        self.fingerprint = hashlib.sha256(model_path.read_bytes()).hexdigest()
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("ONNX inference requires the optional onnxruntime package") from exc
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0]

    def predict(self, image: np.ndarray) -> float:
        shape = self._input.shape
        height = shape[2] if len(shape) == 4 and isinstance(shape[2], int) else 224
        width = shape[3] if len(shape) == 4 and isinstance(shape[3], int) else 224
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.transpose(rgb, (2, 0, 1))[None, ...]
        output = np.asarray(self._session.run(None, {self._input.name: tensor})[0]).squeeze()
        values = np.ravel(output).astype(float)
        if values.size == 1:
            score = float(values[0])
            if score < 0 or score > 1:
                score = 1 / (1 + np.exp(-score))
        elif values.size == 2:
            shifted = values - values.max()
            score = float(np.exp(shifted)[1] / np.exp(shifted).sum())
        else:
            raise ValueError("Forgery model must produce one score or two class logits")
        return float(np.clip(score, 0, 1))


def _normalized_ssim(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    right = cv2.resize(right, (left.shape[1], left.shape[0]))
    return float(np.clip(structural_similarity(left, right, data_range=255), 0, 1))


def _crop(image: np.ndarray, region: list[float] | tuple[float, ...]) -> np.ndarray:
    height, width = image.shape[:2]
    x, y, w, h = region
    return image[
        max(0, int(y * height)) : min(height, int((y + h) * height)),
        max(0, int(x * width)) : min(width, int((x + w) * width)),
    ]


class TemplateAnalyzer:
    def __init__(self, template_root: Path | None = None) -> None:
        self.template_root = template_root

    def analyze(self, image: np.ndarray, issuer: IssuerDefinition | None) -> TemplateResult:
        if issuer is None or not issuer.templates or self.template_root is None:
            return TemplateResult(available=False, issuer_id=issuer.issuer_id if issuer else None)

        best: TemplateResult | None = None
        gray = _as_gray(image)
        for definition in issuer.templates:
            root = self.template_root.resolve()
            relative = str(definition["image"])
            path = root / relative
            expected_digest = definition.get("sha256")
            if expected_digest is not None and (path.is_symlink() or path.parent.is_symlink()):
                continue
            path = path.resolve()
            if path == root or root not in path.parents:
                continue
            if expected_digest is not None:
                if (
                    not isinstance(expected_digest, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", expected_digest)
                    or relative != f"artifacts/{expected_digest}.png"
                ):
                    continue
                try:
                    artifact_bytes = path.read_bytes()
                except OSError:
                    continue
                if hashlib.sha256(artifact_bytes).hexdigest() != expected_digest:
                    continue
                template = cv2.imdecode(
                    np.frombuffer(artifact_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
                )
            else:
                template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if template is None:
                continue
            candidate = self._compare(gray, template, issuer.issuer_id, definition)
            if best is None or (candidate.alignment_score or 0) > (best.alignment_score or 0):
                best = candidate
        return best or TemplateResult(
            available=False,
            issuer_id=issuer.issuer_id,
            explanation="Configured template images were unavailable.",
        )

    @staticmethod
    def _compare(
        image: np.ndarray,
        template: np.ndarray,
        issuer_id: str,
        definition: dict[str, object],
    ) -> TemplateResult:
        candidates = TemplateAnalyzer._feature_candidates()
        best_alignment: tuple[str, float, np.ndarray] | None = None
        match_scores: dict[str, float] = {}
        for name, detector, norm in candidates:
            aligned = TemplateAnalyzer._align(image, template, detector, norm)
            if aligned is None:
                continue
            alignment, transformed = aligned
            match_scores[name] = round(alignment, 3)
            if best_alignment is None or alignment > best_alignment[1]:
                best_alignment = (name, alignment, transformed)
        if best_alignment is None:
            return TemplateResult(
                available=False,
                issuer_id=issuer_id,
                template_id=str(definition.get("id", definition["image"])),
                alignment_score=0.0,
                feature_match_scores=match_scores,
                explanation="The upload could not be aligned with any available feature detector.",
            )
        detector_name, alignment, aligned = best_alignment
        regions = definition.get("regions", {})
        regions = regions if isinstance(regions, dict) else {}
        logo = (
            _normalized_ssim(_crop(template, regions["logo"]), _crop(aligned, regions["logo"]))
            if "logo" in regions
            else None
        )
        font = (
            _normalized_ssim(_crop(template, regions["text"]), _crop(aligned, regions["text"]))
            if "text" in regions
            else None
        )
        template_edges = cv2.Canny(template, 80, 160)
        aligned_edges = cv2.Canny(aligned, 80, 160)
        layout = _normalized_ssim(template_edges, aligned_edges)
        scores = [value for value in (alignment, logo, font, layout) if value is not None]
        anomaly = 1 - float(np.mean(scores))
        checks = ["feature alignment", "layout edge SSIM"]
        if logo is not None:
            checks.append("logo-region SSIM")
        if font is not None:
            checks.append("text-shape-region SSIM")
        return TemplateResult(
            available=True,
            issuer_id=issuer_id,
            template_id=str(definition.get("id", definition["image"])),
            alignment_score=round(alignment, 3),
            feature_detector=detector_name,
            feature_match_scores=match_scores,
            logo_similarity=round(logo, 3) if logo is not None else None,
            font_shape_similarity=round(font, 3) if font is not None else None,
            layout_similarity=round(layout, 3),
            anomaly_score=round(anomaly, 3),
            explanation=f"The upload was compared using {', '.join(checks)}.",
        )

    @staticmethod
    def _feature_candidates() -> list[tuple[str, object, int]]:
        candidates: list[tuple[str, object, int]] = [
            ("ORB", cv2.ORB_create(nfeatures=3000), cv2.NORM_HAMMING)
        ]
        if hasattr(cv2, "SIFT_create"):
            candidates.append(("SIFT", cv2.SIFT_create(nfeatures=3000), cv2.NORM_L2))
        xfeatures = getattr(cv2, "xfeatures2d", None)
        if xfeatures is not None and hasattr(xfeatures, "SURF_create"):
            try:
                candidates.append(("SURF", xfeatures.SURF_create(400), cv2.NORM_L2))
            except cv2.error:
                pass
        return candidates

    @staticmethod
    def _align(
        image: np.ndarray, template: np.ndarray, detector: object, norm: int
    ) -> tuple[float, np.ndarray] | None:
        result = TemplateAnalyzer._align_context(image, template, "feature", detector, norm)
        return (result.score, result.aligned_image) if result else None

    @staticmethod
    def align_images(image: np.ndarray, template: np.ndarray) -> AlignmentContext | None:
        best: AlignmentContext | None = None
        for name, detector, norm in TemplateAnalyzer._feature_candidates():
            result = TemplateAnalyzer._align_context(image, template, name, detector, norm)
            if result is not None and (best is None or result.score > best.score):
                best = result
        return best

    @staticmethod
    def _align_context(
        image: np.ndarray,
        template: np.ndarray,
        name: str,
        detector: object,
        norm: int,
    ) -> AlignmentContext | None:
        image_gray = _as_gray(image)
        template_gray = _as_gray(template)
        keypoints_image, descriptors_image = detector.detectAndCompute(image_gray, None)
        keypoints_template, descriptors_template = detector.detectAndCompute(template_gray, None)
        if descriptors_image is None or descriptors_template is None:
            return None
        matches = cv2.BFMatcher(norm).knnMatch(descriptors_image, descriptors_template, k=2)
        good = [
            pair[0]
            for pair in matches
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
        ]
        if len(good) < 8:
            return None
        source = np.float32([keypoints_image[item.queryIdx].pt for item in good]).reshape(-1, 1, 2)
        target = np.float32([keypoints_template[item.trainIdx].pt for item in good]).reshape(
            -1, 1, 2
        )
        homography, mask = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
        if homography is None or mask is None:
            return None
        aligned = cv2.warpPerspective(image, homography, (template.shape[1], template.shape[0]))
        alignment = float(mask.ravel().mean())
        projected = cv2.perspectiveTransform(source, homography)
        inliers = mask.ravel().astype(bool)
        errors = np.linalg.norm(projected.reshape(-1, 2) - target.reshape(-1, 2), axis=1)
        reprojection_error = float(errors[inliers].mean()) if inliers.any() else float("inf")
        return AlignmentContext(
            detector=name,
            score=alignment,
            homography=homography,
            aligned_image=aligned,
            inlier_count=int(inliers.sum()),
            match_count=len(good),
            reprojection_error=reprojection_error,
            source_dimensions=(image.shape[1], image.shape[0]),
            template_dimensions=(template.shape[1], template.shape[0]),
            inlier_ratio=alignment,
            detector_metadata={
                "detector": name,
                "source_keypoints": len(keypoints_image),
                "template_keypoints": len(keypoints_template),
                "candidate_matches": len(matches),
                "ratio_test": 0.75,
                "ransac_threshold_px": 5.0,
            },
        )


def _as_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError("Alignment images must be grayscale, BGR, or BGRA")


class ProvenanceAnalyzer:
    def __init__(self, forgery_model: ForgeryModel | None = None) -> None:
        self.forgery_model = forgery_model

    def analyze(self, image: np.ndarray, source_path: Path) -> ProvenanceResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        moire = self._moire_score(gray)
        method, confidence = self._capture_method(gray, moire, source_path)
        ela = self._ela_score(source_path)
        jpeg_grid = self._jpeg_grid_score(gray)
        frequency = self._frequency_anomaly_score(gray)
        font_subpixel = self._font_subpixel_score(image)
        copy_move = self._copy_move_score(gray)
        noiseprint = self._noiseprint_score(gray)
        printer_pattern = self._printer_pattern_score(gray)
        neural_score = None
        neural_error = None
        if self.forgery_model is not None:
            try:
                neural_score = self.forgery_model.predict(image)
            except Exception as exc:
                neural_error = f"{type(exc).__name__}: {exc}"
        flags = self._metadata_flags(source_path)
        edit_scores = [
            score
            for score in (ela, jpeg_grid, frequency, font_subpixel, copy_move)
            if score is not None
        ]
        digital_anomaly = float(np.mean(edit_scores)) if edit_scores else 0.0
        explanation = (
            f"Capture characteristics are most consistent with {method}. "
            "This describes provenance and is not, by itself, evidence of invalidity."
        )
        return ProvenanceResult(
            capture_method=method,
            capture_confidence=round(confidence, 3),
            moire_score=round(moire, 3),
            ela_score=round(ela, 3) if ela is not None else None,
            jpeg_grid_score=round(jpeg_grid, 3),
            frequency_anomaly_score=round(frequency, 3),
            font_subpixel_score=round(font_subpixel, 3),
            copy_move_score=round(copy_move, 3) if copy_move is not None else None,
            noiseprint_score=round(noiseprint, 3),
            printer_pattern_score=round(printer_pattern, 3),
            neural_model_available=self.forgery_model is not None,
            neural_model_name=(self.forgery_model.name if self.forgery_model else None),
            neural_forgery_score=(round(neural_score, 3) if neural_score is not None else None),
            neural_error=neural_error,
            metadata_flags=flags,
            digital_edit_anomaly=round(digital_anomaly, 3),
            explanation=explanation,
        )

    @staticmethod
    def _jpeg_grid_score(gray: np.ndarray) -> float:
        image = gray.astype(np.float32)
        vertical_boundaries = np.arange(8, image.shape[1], 8)
        horizontal_boundaries = np.arange(8, image.shape[0], 8)
        vertical = (
            np.abs(image[:, vertical_boundaries] - image[:, vertical_boundaries - 1]).mean()
            if vertical_boundaries.size
            else 0
        )
        horizontal = (
            np.abs(image[horizontal_boundaries, :] - image[horizontal_boundaries - 1, :]).mean()
            if horizontal_boundaries.size
            else 0
        )
        baseline_v = np.abs(np.diff(image, axis=1)).mean() + 1e-6 if image.shape[1] > 1 else 1
        baseline_h = np.abs(np.diff(image, axis=0)).mean() + 1e-6 if image.shape[0] > 1 else 1
        ratio = ((vertical / baseline_v) + (horizontal / baseline_h)) / 2
        return float(np.clip(abs(ratio - 1) / 2, 0, 1))

    @staticmethod
    def _frequency_anomaly_score(gray: np.ndarray) -> float:
        resized = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32)
        spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(resized))))
        height, width = spectrum.shape
        yy, xx = np.ogrid[:height, :width]
        radius = np.sqrt((xx - width / 2) ** 2 + (yy - height / 2) ** 2)
        high = spectrum[(radius > 90) & (radius < 230)]
        if not high.size:
            return 0.0
        median = np.median(high)
        mad = np.median(np.abs(high - median)) + 1e-6
        spikes = np.mean(high > median + 6 * mad)
        return float(np.clip(spikes * 40, 0, 1))

    @staticmethod
    def _font_subpixel_score(image: np.ndarray) -> float:
        channels = [channel.astype(np.float32) for channel in cv2.split(image)]
        edge = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 80, 160) > 0
        if edge.sum() < 50:
            return 0.0
        disagreement = (
            np.abs(channels[0] - channels[1])
            + np.abs(channels[1] - channels[2])
            + np.abs(channels[0] - channels[2])
        ) / 3
        return float(np.clip(np.percentile(disagreement[edge], 90) / 64, 0, 1))

    @staticmethod
    def _noiseprint_score(gray: np.ndarray) -> float:
        source = gray.astype(np.float32) / 255
        residual = source - cv2.GaussianBlur(source, (0, 0), 1.2)
        local_energy = cv2.GaussianBlur(residual * residual, (0, 0), 8)
        mean = float(local_energy.mean())
        if mean <= 1e-8:
            return 0.0
        coefficient_of_variation = float(local_energy.std() / mean)
        return float(np.clip((coefficient_of_variation - 0.5) / 2.5, 0, 1))

    @staticmethod
    def _printer_pattern_score(gray: np.ndarray) -> float:
        residual = gray.astype(np.float32) - cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2)
        spectrum = np.abs(np.fft.fftshift(np.fft.fft2(residual)))
        center_y, center_x = np.array(spectrum.shape) // 2
        spectrum[center_y - 12 : center_y + 13, center_x - 12 : center_x + 13] = 0
        median = float(np.median(spectrum))
        mad = float(np.median(np.abs(spectrum - median))) + 1e-6
        peaks = np.mean(spectrum > median + 10 * mad)
        return float(np.clip(peaks * 100, 0, 1))

    @staticmethod
    def _moire_score(gray: np.ndarray) -> float:
        resized = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
        spectrum = np.abs(np.fft.fftshift(np.fft.fft2(resized)))
        height, width = spectrum.shape
        yy, xx = np.ogrid[:height, :width]
        radius = np.sqrt((xx - width / 2) ** 2 + (yy - height / 2) ** 2)
        annulus = spectrum[(radius > 50) & (radius < 220)]
        if not annulus.size:
            return 0.0
        threshold = np.median(annulus) + 8 * np.median(np.abs(annulus - np.median(annulus)))
        return float(np.clip(np.mean(annulus > threshold) * 20, 0, 1))

    @staticmethod
    def _capture_method(gray: np.ndarray, moire: float, path: Path) -> tuple[str, float]:
        if path.suffix.casefold() == ".pdf":
            return "born-digital", 0.8
        laplacian = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if moire > 0.35:
            return "screen-recapture", min(0.95, 0.55 + moire / 2)
        if laplacian < 80:
            return "print-recapture", min(0.8, 0.55 + (80 - laplacian) / 200)
        return "born-digital", 0.55

    @staticmethod
    def _ela_score(path: Path) -> float | None:
        if path.suffix.casefold() not in {".jpg", ".jpeg"}:
            return None
        try:
            original = Image.open(path).convert("RGB")
            buffer = io.BytesIO()
            original.save(buffer, "JPEG", quality=90)
            buffer.seek(0)
            difference = ImageChops.difference(original, Image.open(buffer).convert("RGB"))
            values = np.asarray(difference, dtype=np.float32)
            channel_max = values.max(axis=2)
            return float(np.clip(np.percentile(channel_max, 99) / 64, 0, 1))
        except OSError:
            return None

    @staticmethod
    def _copy_move_score(gray: np.ndarray) -> float | None:
        detector = cv2.ORB_create(nfeatures=1500)
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < 20:
            return None
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(descriptors, descriptors, k=3)
        displacements: list[tuple[int, int]] = []
        for query_index, neighbors in enumerate(matches):
            for match in neighbors:
                if match.trainIdx == query_index or match.distance > 24:
                    continue
                first = np.array(keypoints[query_index].pt)
                second = np.array(keypoints[match.trainIdx].pt)
                delta = second - first
                if np.linalg.norm(delta) > 50:
                    displacements.append((round(delta[0] / 10), round(delta[1] / 10)))
                    break
        if len(displacements) < 4:
            return 0.0
        largest_cluster = max(displacements.count(value) for value in set(displacements))
        return float(np.clip((largest_cluster - 3) / 12, 0, 1))

    @staticmethod
    def _metadata_flags(path: Path) -> list[str]:
        if path.suffix.casefold() == ".pdf":
            return []
        try:
            with Image.open(path) as image:
                exif = {
                    ExifTags.TAGS.get(key, str(key)): value
                    for key, value in image.getexif().items()
                }
            flags = []
            software = str(exif.get("Software", "")).casefold()
            if any(name in software for name in ("photoshop", "gimp", "affinity")):
                flags.append("Metadata names image-editing software; this is weak evidence only.")
            if (
                "DateTimeOriginal" in exif
                and "DateTime" in exif
                and exif["DateTimeOriginal"] > exif["DateTime"]
            ):
                flags.append("Metadata timestamps are inconsistent.")
            return flags
        except (OSError, TypeError, ValueError):
            return ["Metadata could not be read."]
