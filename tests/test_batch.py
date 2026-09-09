from pathlib import Path

import pytest

from certguard.batch import BatchProcessor, discover_documents
from certguard.models import VerificationStatus


class Report:
    def __init__(self, status: VerificationStatus, review_recommended: bool) -> None:
        self.verification = type("Verification", (), {"status": status})()
        self.review_recommended = review_recommended

    def to_dict(self) -> dict[str, object]:
        return {
            "verification": {"status": self.verification.status},
            "review_recommended": self.review_recommended,
        }


class Pipeline:
    def analyze(self, source: Path, **_kwargs) -> Report:
        if source.name == "broken.pdf":
            raise ValueError("unreadable")
        return Report(
            VerificationStatus.VERIFIED
            if source.name == "verified.pdf"
            else VerificationStatus.NO_CODE_PRESENT,
            review_recommended=source.name != "verified.pdf",
        )


def test_discovers_supported_documents_in_stable_order(tmp_path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "b.PNG").touch()
    (tmp_path / "a.pdf").touch()
    (tmp_path / "notes.txt").touch()
    (nested / "c.jpg").touch()

    direct = discover_documents([tmp_path])
    recursive = discover_documents([tmp_path], recursive=True)

    assert [path.name for path in direct] == ["a.pdf", "b.PNG"]
    assert [path.name for path in recursive] == ["a.pdf", "b.PNG", "c.jpg"]


def test_rejects_an_explicit_unsupported_file(tmp_path) -> None:
    source = tmp_path / "roster.csv"
    source.touch()

    with pytest.raises(ValueError, match="Unsupported document type"):
        discover_documents([source])


def test_batch_continues_after_an_individual_document_failure(tmp_path) -> None:
    sources = [tmp_path / name for name in ("verified.pdf", "review.png", "broken.pdf")]
    for source in sources:
        source.touch()

    report = BatchProcessor(Pipeline()).analyze(sources)
    rendered = report.to_dict()

    assert report.total == 3
    assert report.completed == 2
    assert report.failed == 1
    assert report.review_recommended == 1
    assert report.verification_statuses == {"no-code-present": 1, "verified": 1}
    assert [item["status"] for item in rendered["items"]] == [
        "failed",
        "completed",
        "completed",
    ]
    assert rendered["items"][0]["error"] == "ValueError: unreadable"


def test_empty_directory_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="No supported certificate documents"):
        BatchProcessor(Pipeline()).analyze([tmp_path])
