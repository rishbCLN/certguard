import json
import re
from pathlib import Path

import pytest

from certguard.audit import AuditWriteError, JsonlAuditSink
from certguard.models import (
    AnalysisReport,
    AuditCheck,
    CheckState,
    ContentResult,
    ExtractionResult,
    PageExtraction,
    ProvenanceResult,
    SearchEvidence,
    SearchResult,
    SSDDProfileRef,
    SSDDResult,
    TemplateResult,
    VerificationResult,
    VerificationStatus,
)


class Report:
    def __init__(self, submission_id: str) -> None:
        self.submission_id = submission_id

    def to_dict(self) -> dict[str, str]:
        return {"submission_id": self.submission_id}


class PrivateDataReport:
    def to_dict(self) -> dict[str, object]:
        return {
            "extraction": {
                "text": "Alice Private",
                "formatted_text": "Recipient: Alice Private",
                "structured_fields": {"recipient": "Alice Private"},
                "certificate_ids": ["ABC123"],
                "urls": ["https://private.example/ABC123"],
                "qr_values": ["https://private.example/ABC123"],
                "pages": [{"text": "Alice Private"}],
            },
            "search": {
                "query": '"ABC123"',
                "results": [{"description": "Alice Private"}],
                "accepted_urls": ["https://private.example/ABC123"],
            },
            "verification": {
                "authoritative_claims": {"recipient": "Alice Private"},
                "claim_comparisons": {"recipient": "match"},
                "attempts": [
                    {
                        "url": "https://private.example/ABC123",
                        "final_url": "https://private.example/ABC123?recipient=Alice",
                        "outcome": "verified",
                    }
                ],
            },
            "checks": [
                {
                    "evidence": {
                        "attempts": [
                            {
                                "url": "https://private.example/ABC123",
                                "final_url": "https://private.example/ABC123",
                            }
                        ]
                    }
                }
            ],
        }


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


def test_private_extraction_and_search_content_is_redacted(tmp_path) -> None:
    path = tmp_path / "audit.jsonl"

    JsonlAuditSink(path).append(PrivateDataReport())

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["extraction"]["text"] == "[redacted from audit log]"
    assert record["extraction"]["formatted_text"] == "[redacted from audit log]"
    assert record["extraction"]["structured_fields"] == {}
    assert record["extraction"]["certificate_ids"] == []
    assert record["extraction"]["urls"] == []
    assert record["extraction"]["qr_values"] == []
    assert record["extraction"]["pages"] == []
    assert record["search"]["query"] is None
    assert record["search"]["results"] == []
    assert record["search"]["accepted_urls"] == []
    assert record["verification"]["authoritative_claims"] == {}
    assert record["verification"]["claim_comparisons"] == {}
    assert record["verification"]["attempts"] == [
        {
            "url": "[redacted from audit log]",
            "final_url": "[redacted from audit log]",
            "outcome": "verified",
        }
    ]
    nested_attempt = record["checks"][0]["evidence"]["attempts"][0]
    assert nested_attempt == {
        "url": "[redacted from audit log]",
        "final_url": "[redacted from audit log]",
    }


def test_real_analysis_report_audit_preserves_codes_but_not_source_evidence(tmp_path) -> None:
    secrets = {
        "Alice Private",
        "Advanced Privacy",
        "CERT-PRIVATE-123",
        "https://private.example/verify/CERT-PRIVATE-123?recipient=Alice",
        "authoritative-secret",
        "comparison-secret",
        "sensitive parser failure",
    }
    report = AnalysisReport(
        submission_id="submission-42",
        source_sha256="a" * 64,
        created_at="2026-09-10T19:13:56+00:00",
        risk_score=7.0,
        evidence_coverage=0.72,
        review_recommended=True,
        review_reasons=["SSDD required evidence was unavailable."],
        decision="human-review-triage-only",
        authenticity_assessment="inconclusive",
        ai_origin_assessment="not-assessed",
        ruleset_fingerprint="b" * 64,
        extraction=ExtractionResult(
            text="Alice Private completed Advanced Privacy",
            formatted_text="Recipient: Alice Private",
            structured_fields={
                "recipient": "Alice Private",
                "credential_title": "Advanced Privacy",
                "certificate_id": "CERT-PRIVATE-123",
            },
            certificate_ids=["CERT-PRIVATE-123"],
            urls=["https://private.example/verify/CERT-PRIVATE-123"],
            qr_values=["https://private.example/verify/CERT-PRIVATE-123?recipient=Alice"],
            page_count=1,
            pages=[
                PageExtraction(
                    page_number=1,
                    text="Alice Private",
                    errors=["sensitive parser failure"],
                )
            ],
            errors=["sensitive parser failure"],
        ),
        search=SearchEvidence(
            enabled=True,
            issuer_id="issuer-safe",
            query='"CERT-PRIVATE-123"',
            results=[
                SearchResult(
                    title="Alice Private credential",
                    url="https://private.example/verify/CERT-PRIVATE-123",
                    description="Advanced Privacy",
                    accepted=True,
                )
            ],
            accepted_urls=["https://private.example/verify/CERT-PRIVATE-123"],
            error="sensitive parser failure",
        ),
        verification=VerificationResult(
            status=VerificationStatus.VERIFIED,
            issuer_id="issuer-safe",
            issuer_name="Safe Issuer",
            explanation="Trusted lookup completed.",
            attempts=[
                {
                    "url": "https://private.example/verify/CERT-PRIVATE-123",
                    "final_url": "https://private.example/verify/CERT-PRIVATE-123?recipient=Alice",
                    "status_code": 200,
                    "outcome": "verified",
                    "error": "sensitive parser failure",
                }
            ],
            authoritative_claims={"recipient": "authoritative-secret"},
            claim_comparisons={"recipient": "comparison-secret"},
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
            profile=SSDDProfileRef("issuer-safe", "variant-safe", "1.0"),
            checks_possible=4,
            checks_evaluated=3,
            unavailable=["signature-region-absent"],
            applicable_profile=True,
        ),
        contributions=[],
        checks=[
            AuditCheck(
                name="semantic_structural_dissonance",
                state=CheckState.COMPLETED,
                summary="SSDD completed with status completed.",
                evidence={
                    "status": "completed",
                    "checks_possible": 4,
                    "checks_evaluated": 3,
                    "violations": ["text-qr-binding-recipient-failure"],
                    "qr_claim": "Alice Private",
                    "certificate_id": "CERT-PRIVATE-123",
                    "lookup_url": "https://private.example/verify/CERT-PRIVATE-123",
                    "comparison": "comparison-secret",
                    "error": "sensitive parser failure",
                },
            )
        ],
    )
    path = tmp_path / "audit.jsonl"

    JsonlAuditSink(path).append(report)

    serialized = path.read_text(encoding="utf-8")
    record = json.loads(serialized)
    assert all(secret not in serialized for secret in secrets)
    assert record["extraction"]["errors"] == []
    assert record["search"]["error"] == "[redacted from audit log]"
    assert record["verification"]["attempts"] == [
        {
            "url": "[redacted from audit log]",
            "final_url": "[redacted from audit log]",
            "status_code": 200,
            "outcome": "verified",
            "error": "[redacted from audit log]",
        }
    ]
    assert record["verification"]["status"] == "verified"
    assert record["ssdd"] == {
        "status": "completed",
        "delta": 0.5,
        "violations": ["text-qr-binding-recipient-failure"],
        "issuer_grammar_match": 0.75,
        "detected_by": "ssdd-v2.1",
        "model_sha256": "c" * 64,
        "profile": {
            "issuer_id": "issuer-safe",
            "variant_id": "variant-safe",
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
    evidence = record["checks"][0]["evidence"]
    assert evidence["status"] == "completed"
    assert evidence["checks_possible"] == 4
    assert evidence["checks_evaluated"] == 3
    assert evidence["violations"] == ["text-qr-binding-recipient-failure"]
    assert evidence["qr_claim"] == "[redacted from audit log]"
    assert evidence["certificate_id"] == "[redacted from audit log]"
    assert evidence["lookup_url"] == "[redacted from audit log]"
    assert evidence["comparison"] == "[redacted from audit log]"
    assert evidence["error"] == "[redacted from audit log]"
