from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageChops
from skimage.metrics import structural_similarity

from certguard.models import ProvenanceResult, TemplateResult
from certguard.registry import IssuerDefinition


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
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for definition in issuer.templates:
            root = self.template_root.resolve()
            path = (root / str(definition["image"])).resolve()
            if path == root or root not in path.parents:
                continue
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
        detector = cv2.ORB_create(nfeatures=3000)
        keypoints_image, descriptors_image = detector.detectAndCompute(image, None)
        keypoints_template, descriptors_template = detector.detectAndCompute(template, None)
        if descriptors_image is None or descriptors_template is None:
            return TemplateResult(
                available=False,
                issuer_id=issuer_id,
                template_id=str(definition.get("id", definition["image"])),
                explanation="A template was found, but there were too few visual features to align it.",
            )
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(descriptors_image, descriptors_template, k=2)
        good = [
            pair[0]
            for pair in matches
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
        ]
        if len(good) < 8:
            return TemplateResult(
                available=False,
                issuer_id=issuer_id,
                template_id=str(definition.get("id", definition["image"])),
                alignment_score=0.0,
                explanation="The upload could not be reliably aligned to the closest issuer template.",
            )
        source = np.float32([keypoints_image[item.queryIdx].pt for item in good]).reshape(-1, 1, 2)
        target = np.float32([keypoints_template[item.trainIdx].pt for item in good]).reshape(-1, 1, 2)
        homography, mask = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
        if homography is None or mask is None:
            return TemplateResult(
                available=False,
                issuer_id=issuer_id,
                explanation="Template alignment was inconclusive and was not scored.",
            )
        aligned = cv2.warpPerspective(image, homography, (template.shape[1], template.shape[0]))
        alignment = float(mask.ravel().mean())
        regions = definition.get("regions", {})
        regions = regions if isinstance(regions, dict) else {}
        logo = _normalized_ssim(_crop(template, regions["logo"]), _crop(aligned, regions["logo"])) \
            if "logo" in regions else None
        font = _normalized_ssim(_crop(template, regions["text"]), _crop(aligned, regions["text"])) \
            if "text" in regions else None
        template_edges = cv2.Canny(template, 80, 160)
        aligned_edges = cv2.Canny(aligned, 80, 160)
        layout = _normalized_ssim(template_edges, aligned_edges)
        scores = [value for value in (alignment, logo, font, layout) if value is not None]
        anomaly = 1 - float(np.mean(scores))
        return TemplateResult(
            available=True,
            issuer_id=issuer_id,
            template_id=str(definition.get("id", definition["image"])),
            alignment_score=round(alignment, 3),
            logo_similarity=round(logo, 3) if logo is not None else None,
            font_shape_similarity=round(font, 3) if font is not None else None,
            layout_similarity=round(layout, 3),
            anomaly_score=round(anomaly, 3),
            explanation="The upload was aligned to a configured reference and compared by logo, text shape, and layout.",
        )


class ProvenanceAnalyzer:
    def analyze(self, image: np.ndarray, source_path: Path) -> ProvenanceResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        moire = self._moire_score(gray)
        method, confidence = self._capture_method(gray, moire, source_path)
        ela = None
        copy_move = None
        if method == "born-digital":
            ela = self._ela_score(source_path)
            copy_move = self._copy_move_score(gray)
        flags = self._metadata_flags(source_path)
        edit_scores = [score for score in (ela, copy_move) if score is not None]
        digital_anomaly = float(np.mean(edit_scores)) if edit_scores else 0.0
        explanation = (
            f"Capture characteristics are most consistent with {method}. "
            "This describes provenance and is not, by itself, evidence of fraud."
        )
        return ProvenanceResult(
            capture_method=method,
            capture_confidence=round(confidence, 3),
            moire_score=round(moire, 3),
            ela_score=round(ela, 3) if ela is not None else None,
            copy_move_score=round(copy_move, 3) if copy_move is not None else None,
            metadata_flags=flags,
            digital_edit_anomaly=round(digital_anomaly, 3),
            explanation=explanation,
        )

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
                exif = {ExifTags.TAGS.get(key, str(key)): value for key, value in image.getexif().items()}
            flags = []
            software = str(exif.get("Software", "")).casefold()
            if any(name in software for name in ("photoshop", "gimp", "affinity")):
                flags.append("Metadata names image-editing software; this is weak evidence only.")
            if "DateTimeOriginal" in exif and "DateTime" in exif and exif["DateTimeOriginal"] > exif["DateTime"]:
                flags.append("Metadata timestamps are inconsistent.")
            return flags
        except (OSError, TypeError, ValueError):
            return ["Metadata could not be read."]
