from pathlib import Path

import pytest

from certguard.batch import BatchProcessor, discover_documents
from certguard.models import (
    AnalysisReport,
    ContentResult,
    ExtractionResult,
    ProvenanceResult,
    SearchEvidence,
    SSDDProfileRef,
    SSDDResult,
    TemplateResult,
    VerificationResult,
    VerificationStatus,
)


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


def analysis_report(source: Path) -> AnalysisReport:
    return AnalysisReport(
        submission_id=source.stem,
        source_sha256="a" * 64,
        created_at="2026-09-10T19:13:56+00:00",
        risk_score=7.0,
        evidence_coverage=0.9,
        review_recommended=False,
        review_reasons=[],
        decision="human-review-triage-only",
        authenticity_assessment="inconclusive",
        ai_origin_assessment="not-assessed",
        ruleset_fingerprint="b" * 64,
        extraction=ExtractionResult(),
        search=SearchEvidence(),
        verification=VerificationResult(
            status=VerificationStatus.VERIFIED,
            issuer_id="nptel",
            issuer_name="NPTEL",
            explanation="Verified.",
        ),
        template=TemplateResult(available=False),
        provenance=ProvenanceResult(),
        content=ContentResult(),
        ssdd=SSDDResult(
            status="completed",
            delta=0.5,
            violations=["text-qr-binding-recipient-failure"],
            issuer_grammar_match=0.75,
            model_sha256="c" * 64,
            profile=SSDDProfileRef("nptel", "swayam-v1", "1.0"),
            checks_possible=4,
            checks_evaluated=3,
            unavailable=["signature-region-absent"],
            applicable_profile=True,
        ),
        contributions=[],
        checks=[],
    )


class AnalysisPipeline:
    def __init__(self, failures: set[str] | None = None) -> None:
        self.failures = failures or set()

    def analyze(self, source: Path, **_kwargs) -> AnalysisReport:
        if source.name in self.failures:
            raise ValueError(f"unreadable {source.name}")
        return analysis_report(source)


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


def test_batch_serializes_analysis_report_with_ssdd_fields(tmp_path) -> None:
    source = tmp_path / "certificate.pdf"
    source.touch()

    rendered = BatchProcessor(AnalysisPipeline()).analyze([source]).to_dict()

    assert rendered["report_version"] == "1.0"
    assert rendered["completed"] == 1
    assert rendered["failed"] == 0
    report = rendered["items"][0]["report"]
    assert report["report_version"] == "2.1"
    assert report["ssdd"] == {
        "status": "completed",
        "delta": 0.5,
        "violations": ["text-qr-binding-recipient-failure"],
        "issuer_grammar_match": 0.75,
        "detected_by": "ssdd-v2.1",
        "model_sha256": "c" * 64,
        "profile": {
            "issuer_id": "nptel",
            "variant_id": "swayam-v1",
            "version": "1.0",
        },
        "checks_possible": 4,
        "checks_evaluated": 3,
        "unavailable": ["signature-region-absent"],
        "scoring_enabled": False,
        "risk_points": 0.0,
        "applicable_profile": True,
        "required_binding_violations": [],
    }


@pytest.mark.parametrize(
    ("failures", "expected_statuses", "completed", "failed"),
    [
        (
            {"b.pdf"},
            ["completed", "failed", "completed"],
            2,
            1,
        ),
        (
            {"a.pdf", "b.pdf", "c.pdf"},
            ["failed", "failed", "failed"],
            0,
            3,
        ),
    ],
)
def test_analysis_batch_failure_results_are_deterministic(
    tmp_path, failures, expected_statuses, completed, failed
) -> None:
    sources = [tmp_path / name for name in ("c.pdf", "a.pdf", "b.pdf")]
    for source in sources:
        source.touch()

    rendered = BatchProcessor(AnalysisPipeline(failures)).analyze(sources).to_dict()

    assert [Path(item["source"]).name for item in rendered["items"]] == [
        "a.pdf",
        "b.pdf",
        "c.pdf",
    ]
    assert [item["status"] for item in rendered["items"]] == expected_statuses
    assert rendered["completed"] == completed
    assert rendered["failed"] == failed
    assert rendered["verification_statuses"] == ({"verified": completed} if completed else {})
    assert all(
        item["report"] is not None if item["status"] == "completed" else item["report"] is None
        for item in rendered["items"]
    )
    assert all(
        item["error"] is None
        if item["status"] == "completed"
        else item["error"] == f"ValueError: unreadable {Path(item['source']).name}"
        for item in rendered["items"]
    )
