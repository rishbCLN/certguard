import cv2

from certguard.pipeline import CertGuardPipeline
from certguard.registry import IssuerRegistry
from certguard.verification import LookupResponse


class ClaimLookupClient:
    def get(self, url: str, allowed_hosts: set[str], timeout: float) -> LookupResponse:
        body = "Credential valid Recipient: Alice Example Credential: Python Basics"
        return LookupResponse(200, url, body)


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


def test_pipeline_routes_stolen_record_claims_to_review(tmp_path) -> None:
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
