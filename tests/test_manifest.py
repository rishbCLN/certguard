import csv

import pytest

from certguard.batch import BatchProcessor, load_manifest
from certguard.models import VerificationStatus


class RecordingPipeline:
    def __init__(self) -> None:
        self.calls = []

    def analyze(self, source, **kwargs):
        self.calls.append((source.name, kwargs))
        report = type(
            "Report",
            (),
            {
                "verification": type("V", (), {"status": VerificationStatus.VERIFIED})(),
                "review_recommended": False,
                "to_dict": lambda _self: {},
            },
        )()
        return report


def write_manifest(
    path, rows, header=("filename", "student_id", "expected_recipient", "expected_credential_title")
):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def test_manifest_binds_claims_per_document(tmp_path) -> None:
    (tmp_path / "rahul.pdf").touch()
    (tmp_path / "priya.png").touch()
    write_manifest(
        tmp_path / "class.csv",
        [
            ["rahul.pdf", "CS21001", "Rahul Kumar", "Python Basics"],
            ["priya.png", "CS21002", "Priya Sharma", ""],
        ],
    )
    pipeline = RecordingPipeline()

    report = BatchProcessor(pipeline).analyze(
        [tmp_path],
        expected_credential_title="Default Course",
        manifest=load_manifest(tmp_path / "class.csv"),
    )

    assert pipeline.calls == [
        (
            "priya.png",
            {
                "submission_id": "CS21002",
                "expected_recipient": "Priya Sharma",
                "expected_credential_title": "Default Course",
            },
        ),
        (
            "rahul.pdf",
            {
                "submission_id": "CS21001",
                "expected_recipient": "Rahul Kumar",
                "expected_credential_title": "Python Basics",
            },
        ),
    ]
    assert report.unmatched_manifest_entries == []
    assert [item.student_id for item in report.items] == ["CS21002", "CS21001"]


def test_manifest_match_is_case_insensitive(tmp_path) -> None:
    (tmp_path / "rahul.pdf").touch()
    write_manifest(tmp_path / "class.csv", [["RAHUL.PDF", "CS21001", "Rahul Kumar", "X"]])

    report = BatchProcessor(RecordingPipeline()).analyze(
        [tmp_path],
        manifest=load_manifest(tmp_path / "class.csv"),
    )

    assert report.unmatched_manifest_entries == []


def test_manifest_entries_without_documents_are_reported(tmp_path) -> None:
    (tmp_path / "rahul.pdf").touch()
    write_manifest(
        tmp_path / "class.csv",
        [
            ["rahul.pdf", "CS21001", "Rahul Kumar", "X"],
            ["ghost.pdf", "CS29999", "Ghost Student", "X"],
        ],
    )

    report = BatchProcessor(RecordingPipeline()).analyze(
        [tmp_path],
        manifest=load_manifest(tmp_path / "class.csv"),
    )

    assert report.unmatched_manifest_entries == ["ghost.pdf"]


def test_manifest_rejects_duplicate_filenames(tmp_path) -> None:
    manifest = write_manifest(
        tmp_path / "class.csv",
        [
            ["rahul.pdf", "CS21001", "A", "X"],
            ["rahul.pdf", "CS21002", "B", "X"],
        ],
    )

    with pytest.raises(ValueError, match="Duplicate manifest entry"):
        load_manifest(manifest)


def test_manifest_requires_filename_column(tmp_path) -> None:
    manifest = write_manifest(
        tmp_path / "class.csv",
        [["CS21001", "Rahul Kumar"]],
        header=("student_id", "expected_recipient"),
    )

    with pytest.raises(ValueError, match="'filename' column"):
        load_manifest(manifest)


def test_manifest_requires_a_claim_column(tmp_path) -> None:
    manifest = write_manifest(
        tmp_path / "class.csv",
        [["rahul.pdf", "CS21001"]],
        header=("filename", "student_id"),
    )

    with pytest.raises(ValueError, match="expected_recipient"):
        load_manifest(manifest)


def test_manifest_row_without_filename_is_rejected(tmp_path) -> None:
    manifest = write_manifest(tmp_path / "class.csv", [["", "CS21001", "A", "X"]])

    with pytest.raises(ValueError, match="missing a filename"):
        load_manifest(manifest)


def test_manifest_rejects_duplicate_basenames_in_recursive_batch(tmp_path) -> None:
    first = tmp_path / "alice"
    second = tmp_path / "bob"
    first.mkdir()
    second.mkdir()
    (first / "certificate.pdf").touch()
    (second / "certificate.pdf").touch()
    manifest = write_manifest(
        tmp_path / "class.csv",
        [["certificate.pdf", "CS21001", "Alice", "X"]],
    )

    with pytest.raises(ValueError, match="ambiguous"):
        BatchProcessor(RecordingPipeline()).analyze(
            [tmp_path],
            recursive=True,
            manifest=load_manifest(manifest),
        )
