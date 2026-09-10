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
        record = report.to_dict()
        extraction = record.get("extraction")
        if isinstance(extraction, dict):
            extraction["text"] = "[redacted from audit log]"
            extraction["formatted_text"] = "[redacted from audit log]"
            extraction["structured_fields"] = {}
            extraction["pages"] = []
        search = record.get("search")
        if isinstance(search, dict):
            search["query"] = None
            search["results"] = []
        serialized = json.dumps(record, separators=(",", ":"), ensure_ascii=True)
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(serialized + "\n")
        except OSError as exc:
            raise AuditWriteError(f"Failed to append audit log {self.path}: {exc}") from exc
