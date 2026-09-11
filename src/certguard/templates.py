from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np

from certguard.document import MAX_SOURCE_BYTES, load_document_pages
from certguard.forensics import TemplateAnalyzer
from certguard.registry import IssuerRegistry

CATALOG_VERSION = 1
CATALOG_FILENAME = "catalog.json"
ARTIFACT_DIRECTORY = "artifacts"
MAX_TEMPLATE_BYTES = MAX_SOURCE_BYTES
DISPLAY_NAME_PATTERN = re.compile(r"^[^\x00-\x1f\x7f]{1,120}$")
ISSUER_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
TEMPLATE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
MIN_DIMENSION = 64
MAX_DIMENSION = 30_000_000
MIN_KEYPOINTS = 12
MIN_EDGE_DENSITY = 0.002
MIN_GRAY_STDDEV = 4.0
FEATURE_NAMES = frozenset({"ORB", "SIFT", "SURF"})


class TemplateCatalogError(ValueError):
    """Raised when a catalog operation cannot be completed safely."""


@dataclass(frozen=True, slots=True)
class TemplateEntry:
    id: str
    issuer_id: str
    name: str
    artifact: str
    sha256: str
    width: int
    height: int
    source_type: str
    created_at: str
    features: dict[str, dict[str, int | bool]]
    edge_density: float
    grayscale_stddev: float

    def to_dict(self, *, integrity: str | None = None) -> dict[str, object]:
        result = asdict(self)
        if integrity is not None:
            result["integrity"] = integrity
        return result


class TemplateCatalog:
    _thread_locks: dict[Path, threading.RLock] = {}
    _thread_locks_guard = threading.Lock()

    def __init__(self, root: Path, registry: IssuerRegistry | None = None) -> None:
        if not str(root).strip():
            raise TemplateCatalogError("Catalog root must not be empty")
        self.root = root.expanduser().resolve()
        self.registry = registry or IssuerRegistry.default()
        self.catalog_path = self.root / CATALOG_FILENAME
        self.artifact_root = self.root / ARTIFACT_DIRECTORY
        with self._thread_locks_guard:
            self._thread_lock = self._thread_locks.setdefault(self.root, threading.RLock())

    def add(
        self,
        issuer_id: str,
        name: str,
        source: Path,
        *,
        confirm_anonymized: bool = False,
    ) -> dict[str, object]:
        self._require_confirmation(confirm_anonymized)
        source = source.expanduser().resolve()
        if not source.is_file():
            raise TemplateCatalogError("Template upload is not a readable file")
        if source.stat().st_size > MAX_TEMPLATE_BYTES:
            raise TemplateCatalogError(
                f"Template upload exceeds the {MAX_TEMPLATE_BYTES}-byte limit"
            )
        suffix = source.suffix.casefold()
        if suffix not in {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
            raise TemplateCatalogError("Template must be a supported image or PDF")
        issuer_id, name = self._validate_identity(issuer_id, name)

        document = load_document_pages(source)
        if document.page_count != 1:
            raise TemplateCatalogError("Template PDFs must contain exactly one page")
        image = document.images[0]
        metadata = self._analyze_image(image)
        encoded, png = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not encoded:
            raise TemplateCatalogError("Template could not be normalized to PNG")
        artifact_bytes = png.tobytes()
        digest = hashlib.sha256(artifact_bytes).hexdigest()
        artifact = f"{ARTIFACT_DIRECTORY}/{digest}.png"
        entry = TemplateEntry(
            id=secrets.token_urlsafe(18),
            issuer_id=issuer_id,
            name=name,
            artifact=artifact,
            sha256=digest,
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            source_type="pdf" if suffix == ".pdf" else "image",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            features=metadata["features"],
            edge_density=metadata["edge_density"],
            grayscale_stddev=metadata["grayscale_stddev"],
        )

        with self._locked():
            catalog = self._read_catalog()
            entries = self._entries(catalog)
            if any(item.name.casefold() == name.casefold() for item in entries):
                raise TemplateCatalogError(f"Template name already exists: {name}")
            if any(item.id == entry.id for item in entries):  # pragma: no cover - random collision
                raise TemplateCatalogError("Generated template ID already exists")
            artifact_path = self._artifact_path(artifact)
            artifact_created = False
            try:
                self.artifact_root.mkdir(parents=True, exist_ok=True)
                if self.artifact_root.is_symlink():
                    raise TemplateCatalogError("Managed artifact directory must not be a symlink")
                if artifact_path.is_symlink():
                    raise TemplateCatalogError("Managed template artifact must not be a symlink")
                if artifact_path.exists():
                    if self._sha256(artifact_path) != digest:
                        raise TemplateCatalogError(
                            "Existing managed artifact failed integrity check"
                        )
                else:
                    self._atomic_write_bytes(artifact_path, artifact_bytes, create_only=True)
                    artifact_created = True
                catalog["templates"] = [*[asdict(item) for item in entries], asdict(entry)]
                self._write_catalog(catalog)
            except Exception:
                if artifact_created:
                    try:
                        if not self.artifact_root.is_symlink():
                            artifact_path.unlink(missing_ok=True)
                            self._flush_directory(artifact_path.parent)
                    except OSError:
                        pass
                raise
        return entry.to_dict(integrity="ok")

    def add_stream(
        self,
        issuer_id: str,
        name: str,
        stream: BinaryIO,
        filename: str,
        size: int,
        *,
        confirm_anonymized: bool = False,
    ) -> dict[str, object]:
        self._require_confirmation(confirm_anonymized)
        if size < 1:
            raise TemplateCatalogError("Template upload is empty")
        if size > MAX_TEMPLATE_BYTES:
            raise TemplateCatalogError(
                f"Template upload exceeds the {MAX_TEMPLATE_BYTES}-byte limit"
            )
        suffix = Path(filename).suffix.casefold()
        if not suffix:
            raise TemplateCatalogError("Template filename must include an image or PDF extension")
        self.root.mkdir(parents=True, exist_ok=True)
        fd, raw_path = tempfile.mkstemp(prefix=".upload-", suffix=suffix, dir=self.root)
        temporary = Path(raw_path)
        try:
            written = 0
            with os.fdopen(fd, "wb") as output:
                while chunk := stream.read(min(1024 * 1024, MAX_TEMPLATE_BYTES + 1 - written)):
                    written += len(chunk)
                    if written > MAX_TEMPLATE_BYTES:
                        raise TemplateCatalogError(
                            f"Template upload exceeds the {MAX_TEMPLATE_BYTES}-byte limit"
                        )
                    output.write(chunk)
            if written != size:
                raise TemplateCatalogError("Template upload was incomplete")
            return self.add(
                issuer_id,
                name,
                temporary,
                confirm_anonymized=confirm_anonymized,
            )
        finally:
            temporary.unlink(missing_ok=True)

    def list(self) -> list[dict[str, object]]:
        with self._locked():
            entries = self._entries(self._read_catalog())
            return [entry.to_dict(integrity=self._integrity(entry)) for entry in entries]

    def remove(self, template_id: str) -> dict[str, object]:
        if not isinstance(template_id, str) or not TEMPLATE_ID_PATTERN.fullmatch(template_id):
            raise TemplateCatalogError("Invalid template ID")
        with self._locked():
            catalog = self._read_catalog()
            entries = self._entries(catalog)
            removed = next((entry for entry in entries if entry.id == template_id), None)
            if removed is None:
                raise TemplateCatalogError("Template ID was not found")
            remaining = [entry for entry in entries if entry.id != template_id]
            catalog["templates"] = [asdict(entry) for entry in remaining]
            self._write_catalog(catalog)
            cleanup = "not-needed"
            if not any(entry.artifact == removed.artifact for entry in remaining):
                try:
                    if self.artifact_root.is_symlink():
                        raise TemplateCatalogError("Catalog contains an unsafe artifact directory")
                    artifact_path = self._artifact_path(removed.artifact)
                    was_symlink = artifact_path.is_symlink()
                    artifact_path.unlink(missing_ok=True)
                    self._flush_directory(artifact_path.parent)
                    cleanup = "deleted"
                    if was_symlink:
                        cleanup = "deleted-symlink"
                except TemplateCatalogError:
                    cleanup = "skipped-unsafe-path"
                except OSError as exc:
                    cleanup = f"failed: {type(exc).__name__}"
            result = removed.to_dict()
            result["artifact_cleanup"] = cleanup
            return result

    def apply_to_registry(self, registry: IssuerRegistry | None = None) -> IssuerRegistry:
        base = registry or self.registry
        grouped: dict[str, list[dict[str, object]]] = {}
        for raw in self.list():
            if raw["integrity"] != "ok":
                continue
            grouped.setdefault(str(raw["issuer_id"]), []).append(
                {"id": raw["id"], "image": raw["artifact"], "sha256": raw["sha256"]}
            )
        issuers = {
            issuer_id: replace(
                issuer,
                templates=tuple([*issuer.templates, *grouped.get(issuer_id, [])]),
            )
            for issuer_id, issuer in base.issuers.items()
        }
        return IssuerRegistry(issuers)

    def _validate_identity(self, issuer_id: str, name: str) -> tuple[str, str]:
        if not isinstance(issuer_id, str) or not isinstance(name, str):
            raise TemplateCatalogError("Issuer ID and template name must be strings")
        issuer_id = issuer_id.strip()
        name = " ".join(name.split())
        if not ISSUER_ID_PATTERN.fullmatch(issuer_id):
            raise TemplateCatalogError("Issuer ID is invalid")
        if self.registry.get(issuer_id) is None:
            raise TemplateCatalogError(f"Unknown issuer ID: {issuer_id}")
        if not DISPLAY_NAME_PATTERN.fullmatch(name) or not any(char.isalnum() for char in name):
            raise TemplateCatalogError("Template name must be a safe nonempty display name")
        return issuer_id, name

    @staticmethod
    def _require_confirmation(confirm_anonymized: bool) -> None:
        if confirm_anonymized is not True:
            raise TemplateCatalogError(
                "Explicit confirmation is required that the template is anonymized, "
                "rights-cleared, and contains no personal recipient data"
            )

    @staticmethod
    def _analyze_image(image: np.ndarray) -> dict[str, object]:
        height, width = image.shape[:2]
        if min(height, width) < MIN_DIMENSION:
            raise TemplateCatalogError("Template dimensions are too small for reliable alignment")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        stddev = float(np.std(gray))
        edges = cv2.Canny(gray, 80, 160)
        edge_density = float(np.count_nonzero(edges) / edges.size)
        features: dict[str, dict[str, int | bool]] = {}
        best_keypoints = 0
        for name, detector, _norm in TemplateAnalyzer._feature_candidates():
            keypoints, descriptors = detector.detectAndCompute(gray, None)
            keypoint_count = len(keypoints or [])
            descriptor_count = int(descriptors.shape[0]) if descriptors is not None else 0
            features[name] = {
                "available": True,
                "keypoints": keypoint_count,
                "descriptors": descriptor_count,
            }
            best_keypoints = max(best_keypoints, keypoint_count)
        for name in ("ORB", "SIFT", "SURF"):
            features.setdefault(name, {"available": False, "keypoints": 0, "descriptors": 0})
        if (
            stddev < MIN_GRAY_STDDEV
            or edge_density < MIN_EDGE_DENSITY
            or best_keypoints < MIN_KEYPOINTS
        ):
            raise TemplateCatalogError(
                "Template has too little visual information for feature-based alignment"
            )
        return {
            "features": features,
            "edge_density": round(edge_density, 6),
            "grayscale_stddev": round(stddev, 3),
        }

    def _integrity(self, entry: TemplateEntry) -> str:
        try:
            path = self._artifact_path(entry.artifact)
        except TemplateCatalogError:
            return "unsafe-path"
        if self.artifact_root.is_symlink():
            return "symlink"
        if path.is_symlink():
            return "symlink"
        if not path.is_file():
            return "missing"
        return "ok" if self._sha256(path) == entry.sha256 else "corrupt"

    def _artifact_path(self, relative: str) -> Path:
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise TemplateCatalogError("Catalog contains an unsafe artifact path")
        match = re.fullmatch(r"artifacts/([0-9a-f]{64})\.png", relative)
        if match is None:
            raise TemplateCatalogError("Catalog contains an unsafe artifact path")
        return self.root / ARTIFACT_DIRECTORY / f"{match.group(1)}.png"

    def _read_catalog(self) -> dict[str, object]:
        if not self.catalog_path.exists():
            return {"schema_version": CATALOG_VERSION, "templates": []}
        try:
            data = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TemplateCatalogError("Template catalog is unreadable or invalid") from exc
        if not isinstance(data, dict) or data.get("schema_version") != CATALOG_VERSION:
            raise TemplateCatalogError("Unsupported template catalog schema version")
        if not isinstance(data.get("templates"), list):
            raise TemplateCatalogError("Template catalog entries must be a list")
        return data

    def _entries(self, catalog: dict[str, object]) -> list[TemplateEntry]:
        raw_entries = catalog["templates"]
        if not isinstance(raw_entries, list):
            raise TemplateCatalogError("Template catalog entries must be a list")
        entries = [self._validate_entry(raw) for raw in raw_entries]
        ids = [entry.id for entry in entries]
        names = [entry.name.casefold() for entry in entries]
        if len(ids) != len(set(ids)) or len(names) != len(set(names)):
            raise TemplateCatalogError("Template catalog contains duplicate IDs or names")
        return entries

    def _validate_entry(self, raw: object) -> TemplateEntry:
        fields = set(TemplateEntry.__dataclass_fields__)
        if not isinstance(raw, dict) or set(raw) != fields:
            raise TemplateCatalogError("Template catalog contains an invalid entry")
        identifier = raw["id"]
        issuer_id = raw["issuer_id"]
        name = raw["name"]
        artifact = raw["artifact"]
        digest = raw["sha256"]
        width = raw["width"]
        height = raw["height"]
        source_type = raw["source_type"]
        created_at = raw["created_at"]
        features = raw["features"]
        edge_density = raw["edge_density"]
        grayscale_stddev = raw["grayscale_stddev"]
        if not isinstance(identifier, str) or not TEMPLATE_ID_PATTERN.fullmatch(identifier):
            raise TemplateCatalogError("Template catalog contains an invalid template ID")
        self._validate_identity_value(issuer_id, name)
        if not isinstance(digest, str) or not DIGEST_PATTERN.fullmatch(digest):
            raise TemplateCatalogError("Template catalog contains an invalid SHA-256 digest")
        if artifact != f"{ARTIFACT_DIRECTORY}/{digest}.png":
            raise TemplateCatalogError("Template artifact path does not match its SHA-256 digest")
        self._artifact_path(artifact)
        for label, value in (("width", width), ("height", height)):
            if type(value) is not int or not MIN_DIMENSION <= value <= MAX_DIMENSION:
                raise TemplateCatalogError(f"Template catalog contains an invalid {label}")
        if width * height > MAX_DIMENSION:
            raise TemplateCatalogError("Template catalog dimensions exceed the pixel limit")
        if source_type not in {"image", "pdf"}:
            raise TemplateCatalogError("Template catalog contains an invalid source type")
        self._validate_timestamp(created_at)
        self._validate_features(features)
        if not self._finite_number(edge_density) or not 0 <= edge_density <= 1:
            raise TemplateCatalogError("Template catalog contains invalid edge density")
        if not self._finite_number(grayscale_stddev) or not 0 <= grayscale_stddev <= 127.5:
            raise TemplateCatalogError(
                "Template catalog contains invalid grayscale standard deviation"
            )
        return TemplateEntry(**raw)

    def _validate_identity_value(self, issuer_id: object, name: object) -> None:
        if not isinstance(issuer_id, str) or not ISSUER_ID_PATTERN.fullmatch(issuer_id):
            raise TemplateCatalogError("Template catalog contains an invalid issuer ID")
        if self.registry.get(issuer_id) is None:
            raise TemplateCatalogError(f"Template catalog contains unknown issuer ID: {issuer_id}")
        if (
            not isinstance(name, str)
            or name != " ".join(name.split())
            or not DISPLAY_NAME_PATTERN.fullmatch(name)
            or not any(char.isalnum() for char in name)
        ):
            raise TemplateCatalogError("Template catalog contains an invalid display name")

    @staticmethod
    def _validate_timestamp(value: object) -> None:
        if not isinstance(value, str) or not value.endswith("Z"):
            raise TemplateCatalogError("Template catalog contains an invalid timestamp")
        try:
            timestamp = datetime.fromisoformat(value[:-1] + "+00:00")
        except ValueError as exc:
            raise TemplateCatalogError("Template catalog contains an invalid timestamp") from exc
        if (
            timestamp.tzinfo is None
            or timestamp < datetime(2000, 1, 1, tzinfo=UTC)
            or timestamp > datetime.now(UTC) + timedelta(minutes=5)
        ):
            raise TemplateCatalogError("Template catalog contains an unreasonable timestamp")

    @staticmethod
    def _validate_features(features: object) -> None:
        if not isinstance(features, dict) or set(features) != FEATURE_NAMES:
            raise TemplateCatalogError("Template catalog contains an invalid feature mapping")
        for name, values in features.items():
            if not isinstance(name, str) or not isinstance(values, dict):
                raise TemplateCatalogError("Template catalog contains an invalid feature mapping")
            if set(values) != {"available", "keypoints", "descriptors"}:
                raise TemplateCatalogError("Template catalog contains invalid feature metrics")
            if type(values["available"]) is not bool:
                raise TemplateCatalogError("Template catalog contains invalid feature availability")
            for metric in ("keypoints", "descriptors"):
                if type(values[metric]) is not int or not 0 <= values[metric] <= 10_000_000:
                    raise TemplateCatalogError("Template catalog contains invalid feature counts")
            if not values["available"] and (values["keypoints"] or values["descriptors"]):
                raise TemplateCatalogError("Unavailable features must have zero counts")

    @staticmethod
    def _finite_number(value: object) -> bool:
        return type(value) in {int, float} and math.isfinite(value)

    def _write_catalog(self, catalog: dict[str, object]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(catalog, indent=2, sort_keys=True) + "\n").encode()
        self._atomic_write_bytes(self.catalog_path, payload)

    @staticmethod
    def _atomic_write_bytes(path: Path, payload: bytes, *, create_only: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if create_only:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            with os.fdopen(os.open(path, flags, 0o600), "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            TemplateCatalog._flush_directory(path.parent)
            return
        fd, raw_temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(raw_temporary)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            TemplateCatalog._flush_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _flush_directory(path: Path) -> None:
        if os.name == "nt":
            return
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        fd = os.open(path, flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / ".catalog.lock"
        with self._thread_lock:
            with lock_path.open("a+b") as lock:
                self._lock_file(lock)
                try:
                    yield
                finally:
                    self._unlock_file(lock)

    @staticmethod
    def _lock_file(stream: BinaryIO) -> None:
        if os.name == "nt":
            import msvcrt

            stream.seek(0)
            if stream.read(1) == b"":
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:  # pragma: no cover - exercised on non-Windows CI
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)

    @staticmethod
    def _unlock_file(stream: BinaryIO) -> None:
        if os.name == "nt":
            import msvcrt

            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:  # pragma: no cover - exercised on non-Windows CI
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
