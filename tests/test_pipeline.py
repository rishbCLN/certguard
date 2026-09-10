import cv2
import numpy as np
import pytest

from certguard.models import (
    ExtractionResult,
    ProvenanceResult,
    SearchResult,
    SSDDResult,
    TemplateResult,
    VerificationResult,
    VerificationStatus,
)
from certguard.pipeline import CertGuardPipeline, _review_reasons
from certguard.registry import IssuerRegistry
from certguard.verification import LookupResponse


class ClaimLookupClient:
    def get(self, url: str, allowed_hosts: set[str], timeout: float) -> LookupResponse:
        body = "Credential valid Recipient: Alice Example Credential: Python Basics"
        return LookupResponse(200, url, body)


class OfficialSearchClient:
    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        return [SearchResult("Official record", "https://verify.example.org/c/ABC123")]


def claim_registry() -> IssuerRegistry:
    return IssuerRegistry.from_data(
        {
            "issuers": [
                {
                    "id": "example",
                    "display_name": "Example",
                    "aliases": ["Example"],
                    "verification_url_patterns": [
                        "^https://verify\\.example\\.org/c/[A-Za-z0-9]+$"
                    ],
                    "allowed_hosts": ["verify.example.org"],
                    "certificate_id_patterns": ["/c/([A-Za-z0-9]+)"],
                    "endpoints": [
                        {
                            "url_template": "https://verify.example.org/c/{certificate_id}",
                            "allowed_hosts": ["verify.example.org"],
                            "success_markers": ["Credential valid"],
                            "failure_markers": ["Credential not found"],
                            "recipient_patterns": ["Recipient: ([A-Za-z ]+) Credential:"],
                            "credential_patterns": ["Credential: ([A-Za-z ]+)$"],
                        }
                    ],
                }
            ]
        }
    )


def write_qr(path, value: str) -> None:
    qr = cv2.QRCodeEncoder_create().encode(value)
    qr = cv2.resize(qr, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
    qr = cv2.copyMakeBorder(qr, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    cv2.imwrite(str(path), qr)


def test_pipeline_binds_record_to_trusted_submission_claims(tmp_path) -> None:
    source = tmp_path / "certificate.png"
    write_qr(source, "https://verify.example.org/c/ABC123")
    pipeline = CertGuardPipeline(registry=claim_registry())
    pipeline.verification.client = ClaimLookupClient()

    report = pipeline.analyze(
        source,
        submission_id="submission-1",
        expected_recipient="Alice Example",
        expected_credential_title="Python Basics",
    )

    assert report.verification.status == "verified"
    assert report.authenticity_assessment == "issuer-record-consistent"
    assert not report.review_recommended
    assert report.evidence_coverage > 0.7
    assert len(report.ruleset_fingerprint) == 64


def test_pipeline_routes_mismatched_record_claims_to_review(tmp_path) -> None:
    source = tmp_path / "certificate.png"
    write_qr(source, "https://verify.example.org/c/ABC123")
    pipeline = CertGuardPipeline(registry=claim_registry())
    pipeline.verification.client = ClaimLookupClient()

    report = pipeline.analyze(
        source,
        expected_recipient="Mallory Example",
        expected_credential_title="Python Basics",
    )

    assert report.verification.status == "claims-mismatch"
    assert report.authenticity_assessment == "issuer-record-mismatch"
    assert report.risk_score == 70
    assert report.review_recommended


def test_pipeline_uses_search_discovered_official_record(tmp_path, monkeypatch) -> None:
    source = tmp_path / "certificate.png"
    assert cv2.imwrite(str(source), 255 * np.ones((20, 20, 3), dtype=np.uint8))
    pipeline = CertGuardPipeline(
        registry=claim_registry(),
        search_client=OfficialSearchClient(),
        search_enabled=True,
    )
    pipeline.verification.client = ClaimLookupClient()

    monkeypatch.setattr(
        "certguard.pipeline.extract_loaded_document",
        lambda _document: ExtractionResult(
            text="Example Certificate ID: ABC123",
            certificate_ids=["ABC123"],
        ),
    )

    report = pipeline.analyze(
        source,
        expected_recipient="Alice Example",
        expected_credential_title="Python Basics",
    )

    assert report.search.accepted_urls == ["https://verify.example.org/c/ABC123"]
    assert report.verification.status == VerificationStatus.VERIFIED
    assert report.extraction.text == "Example Certificate ID: ABC123"
    assert report.report_version == "2.1"


def test_review_threshold_is_honored() -> None:
    verification = VerificationResult(
        status=VerificationStatus.VERIFIED,
        issuer_id="example",
        issuer_name="Example",
        explanation="",
    )
    template = TemplateResult(available=False)
    default = _review_reasons(
        verification, template, ProvenanceResult(), SSDDResult(), 0.9, 60.0, 55.0
    )
    raised = _review_reasons(
        verification, template, ProvenanceResult(), SSDDResult(), 0.9, 60.0, 65.0
    )

    assert any("exceeds the review threshold" in reason for reason in default)
    assert not any("exceeds the review threshold" in reason for reason in raised)


def test_invalid_review_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="review_threshold"):
        CertGuardPipeline(review_threshold=0.0)


def test_unused_provenance_parameter_is_accepted() -> None:
    assert _review_reasons(
        VerificationResult(
            status=VerificationStatus.VERIFIED, issuer_id="x", issuer_name="x", explanation=""
        ),
        TemplateResult(available=False),
        ProvenanceResult(),
        SSDDResult(),
        0.9,
        0.0,
        55.0,
    ) == []


def test_high_neural_forgery_signal_routes_to_review() -> None:
    reasons = _review_reasons(
        VerificationResult(
            status=VerificationStatus.VERIFIED,
            issuer_id="x",
            issuer_name="x",
            explanation="",
        ),
        TemplateResult(available=False),
        ProvenanceResult(neural_model_available=True, neural_forgery_score=0.9),
        SSDDResult(),
        0.9,
        5.0,
        55.0,
    )

    assert reasons == ["The configured forgery model returned a high-risk signal."]
