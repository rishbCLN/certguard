import json
import re
from pathlib import Path

import pytest

from certguard.audit import AuditWriteError, JsonlAuditSink


class Report:
    def __init__(self, submission_id: str) -> None:
        self.submission_id = submission_id

    def to_dict(self) -> dict[str, str]:
        return {"submission_id": self.submission_id}


def test_appends_compact_json_records(tmp_path) -> None:
    path = tmp_path / "logs" / "audit.jsonl"
    sink = JsonlAuditSink(path)

    sink.append(Report("first"))
    sink.append(Report("second"))

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records == [{"submission_id": "first"}, {"submission_id": "second"}]


def test_disk_io_error_has_audit_path_context(monkeypatch, tmp_path) -> None:
    path = tmp_path / "audit.jsonl"
    sink = JsonlAuditSink(path)

    def deny_write(self, *args, **kwargs):
        raise PermissionError("read-only filesystem")

    monkeypatch.setattr(Path, "open", deny_write)

    with pytest.raises(AuditWriteError, match=re.escape(str(path))) as error:
        sink.append(Report("blocked"))

    assert isinstance(error.value.__cause__, PermissionError)
