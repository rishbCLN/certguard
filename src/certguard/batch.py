from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from certguard.models import AnalysisReport
from certguard.pipeline import CertGuardPipeline

SUPPORTED_DOCUMENT_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(slots=True)
class ManifestEntry:
    filename: str
    student_id: str | None = None
    expected_recipient: str | None = None
    expected_credential_title: str | None = None


@dataclass(slots=True)
class BatchItem:
    source: str
    status: str
    student_id: str | None = None
    report: AnalysisReport | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "status": self.status,
            "student_id": self.student_id,
            "report": self.report.to_dict() if self.report is not None else None,
            "error": self.error,
        }


@dataclass(slots=True)
class BatchReport:
    created_at: str
    total: int
    completed: int
    failed: int
    review_recommended: int
    verification_statuses: dict[str, int] = field(default_factory=dict)
    unmatched_manifest_entries: list[str] = field(default_factory=list)
    items: list[BatchItem] = field(default_factory=list)
    report_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "review_recommended": self.review_recommended,
            "verification_statuses": self.verification_statuses,
            "unmatched_manifest_entries": self.unmatched_manifest_entries,
            "items": [item.to_dict() for item in self.items],
            "report_version": self.report_version,
        }


class BatchProcessor:
    def __init__(self, pipeline: CertGuardPipeline) -> None:
        self.pipeline = pipeline

    def analyze(
        self,
        sources: Iterable[Path],
        *,
        recursive: bool = False,
        expected_credential_title: str | None = None,
        manifest: dict[str, ManifestEntry] | None = None,
    ) -> BatchReport:
        documents = discover_documents(sources, recursive=recursive)
        if not documents:
            raise ValueError("No supported certificate documents were found")
        if manifest:
            duplicate_names = sorted(
                name
                for name, count in Counter(
                    document.name.casefold() for document in documents
                ).items()
                if count > 1
            )
            if duplicate_names:
                raise ValueError(
                    "Manifest matching is ambiguous for duplicate filenames: "
                    + ", ".join(duplicate_names)
                )

        items: list[BatchItem] = []
        verification_statuses: dict[str, int] = {}
        review_count = 0
        matched_manifest_keys: set[str] = set()
        for document in documents:
            manifest_key = document.name.casefold()
            entry = manifest.get(manifest_key) if manifest else None
            if entry is not None:
                matched_manifest_keys.add(manifest_key)
            try:
                report = self.pipeline.analyze(
                    document,
                    submission_id=entry.student_id if entry else None,
                    expected_recipient=entry.expected_recipient if entry else None,
                    expected_credential_title=(
                        entry.expected_credential_title
                        if entry and entry.expected_credential_title
                        else expected_credential_title
                    ),
                )
            except Exception as exc:
                items.append(
                    BatchItem(
                        source=str(document),
                        status="failed",
                        student_id=entry.student_id if entry else None,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue

            status = str(report.verification.status)
            verification_statuses[status] = verification_statuses.get(status, 0) + 1
            review_count += int(report.review_recommended)
            items.append(
                BatchItem(
                    source=str(document),
                    status="completed",
                    student_id=entry.student_id if entry else None,
                    report=report,
                )
            )

        completed = sum(item.status == "completed" for item in items)
        return BatchReport(
            created_at=datetime.now(UTC).isoformat(),
            total=len(items),
            completed=completed,
            failed=len(items) - completed,
            review_recommended=review_count,
            verification_statuses=verification_statuses,
            unmatched_manifest_entries=sorted(
                entry.filename
                for key, entry in (manifest or {}).items()
                if key not in matched_manifest_keys
            ),
            items=items,
        )


def discover_documents(sources: Iterable[Path], *, recursive: bool = False) -> list[Path]:
    documents: dict[Path, None] = {}
    for source in sources:
        source = source.resolve()
        if source.is_file():
            if source.suffix.casefold() not in SUPPORTED_DOCUMENT_EXTENSIONS:
                raise ValueError(f"Unsupported document type: {source}")
            documents[source] = None
            continue
        if source.is_dir():
            candidates = source.rglob("*") if recursive else source.glob("*")
            for candidate in candidates:
                if (
                    candidate.is_file()
                    and candidate.suffix.casefold() in SUPPORTED_DOCUMENT_EXTENSIONS
                ):
                    documents[candidate.resolve()] = None
            continue
        raise FileNotFoundError(source)
    return sorted(documents, key=lambda path: str(path).casefold())


def load_manifest(path: Path) -> dict[str, ManifestEntry]:
    entries: dict[str, ManifestEntry] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = {name.strip().casefold(): name for name in reader.fieldnames or []}
        if "filename" not in columns:
            raise ValueError("Manifest must include a 'filename' column")
        if not ({"expected_recipient", "expected_credential_title"} & columns.keys()):
            raise ValueError(
                "Manifest must include an 'expected_recipient' or "
                "'expected_credential_title' column"
            )
        for row in reader:
            values = {key: (row.get(column) or "").strip() for key, column in columns.items()}
            filename = values["filename"]
            if not filename:
                raise ValueError("Manifest row is missing a filename")
            key = Path(filename).name.casefold()
            if key in entries:
                raise ValueError(f"Duplicate manifest entry for: {filename}")
            entries[key] = ManifestEntry(
                filename=filename,
                student_id=values.get("student_id") or None,
                expected_recipient=values.get("expected_recipient") or None,
                expected_credential_title=values.get("expected_credential_title") or None,
            )
    return entries
