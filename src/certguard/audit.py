from __future__ import annotations

import json
import threading
from pathlib import Path

from certguard.models import AnalysisReport


class AuditWriteError(OSError):
    """Raised when an audit record cannot be persisted."""


class JsonlAuditSink:
    """Append-only local audit sink; production deployments can replace this adapter."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def append(self, report: AnalysisReport) -> None:
        record = _redact_sensitive_fields(report.to_dict())
        extraction = record.get("extraction")
        if isinstance(extraction, dict):
            extraction["text"] = "[redacted from audit log]"
            extraction["formatted_text"] = "[redacted from audit log]"
            extraction["structured_fields"] = {}
            extraction["certificate_ids"] = []
            extraction["urls"] = []
            extraction["qr_values"] = []
            extraction["pages"] = []
        search = record.get("search")
        if isinstance(search, dict):
            search["query"] = None
            search["results"] = []
            search["accepted_urls"] = []
        verification = record.get("verification")
        if isinstance(verification, dict):
            verification["authoritative_claims"] = {}
            verification["claim_comparisons"] = {}
            verification["attempts"] = [
                attempt for attempt in verification.get("attempts", []) if isinstance(attempt, dict)
            ]
        serialized = json.dumps(record, separators=(",", ":"), ensure_ascii=True)
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(serialized + "\n")
        except OSError as exc:
            raise AuditWriteError(f"Failed to append audit log {self.path}: {exc}") from exc


_REDACTED = "[redacted from audit log]"
_SENSITIVE_SCALAR_KEYS = {
    "certificate_id",
    "claim",
    "claim_value",
    "comparison",
    "credential_title",
    "expected_value",
    "url",
    "final_url",
    "lookup_url",
    "observed_value",
    "query",
    "qr_claim",
    "qr_payload",
    "qr_value",
    "recipient",
    "student_id",
    "text",
    "value",
    "formatted_text",
    "error",
}
_SENSITIVE_LIST_KEYS = {
    "accepted_urls",
    "certificate_ids",
    "errors",
    "pages",
    "qr_claims",
    "qr_payloads",
    "qr_values",
    "results",
    "urls",
}
_SENSITIVE_MAPPING_KEYS = {
    "authoritative_claims",
    "claims",
    "claim_comparisons",
    "comparisons",
    "structured_fields",
}


def _redact_sensitive_fields(value, key: str | None = None):  # noqa: ANN001, ANN202
    """Remove private values recursively, including copies nested in check evidence."""
    normalized_key = key.casefold() if isinstance(key, str) else None
    if normalized_key in _SENSITIVE_SCALAR_KEYS or (
        normalized_key is not None
        and (
            normalized_key.endswith("_error")
            or normalized_key.endswith("_text")
            or normalized_key.endswith("_url")
        )
    ):
        return None if key == "query" else _REDACTED
    if normalized_key in _SENSITIVE_LIST_KEYS:
        return []
    if normalized_key in _SENSITIVE_MAPPING_KEYS:
        return {}
    if isinstance(value, dict):
        return {
            item_key: _redact_sensitive_fields(item, item_key) for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive_fields(item) for item in value]
    return value
