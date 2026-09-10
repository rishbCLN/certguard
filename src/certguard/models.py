from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class VerificationStatus(StrEnum):
    VERIFIED = "verified"
    RECORD_FOUND = "record-found"
    CLAIMS_MISMATCH = "claims-mismatch"
    FAILED_LOOKUP = "failed-lookup"
    LOOKUP_INCONCLUSIVE = "lookup-inconclusive"
    LOOKUP_UNAVAILABLE = "lookup-unavailable"
    AMBIGUOUS_ISSUER = "ambiguous-issuer"
    UNRECOGNIZED_ISSUER = "unrecognized-issuer"
    NO_CODE_PRESENT = "no-code-present"


class CheckState(StrEnum):
    COMPLETED = "completed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass(slots=True)
class AuditCheck:
    name: str
    state: CheckState
    summary: str
    evidence: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    duration_ms: int = 0


@dataclass(slots=True)
class PageExtraction:
    page_number: int
    text: str = ""
    text_sources: list[str] = field(default_factory=list)
    ocr_confidence: float | None = None
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ExtractionResult:
    text: str = ""
    structured_fields: dict[str, str] = field(default_factory=dict)
    formatted_text: str = ""
    certificate_ids: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    qr_values: list[str] = field(default_factory=list)
    ocr_confidence: float | None = None
    page_count: int = 1
    pages: list[PageExtraction] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SearchResult:
    title: str
    url: str
    description: str = ""
    accepted: bool = False


@dataclass(slots=True)
class SearchEvidence:
    enabled: bool = False
    issuer_id: str | None = None
    query: str | None = None
    results: list[SearchResult] = field(default_factory=list)
    accepted_urls: list[str] = field(default_factory=list)
    explanation: str = "Online search was not enabled."
    error: str | None = None


@dataclass(slots=True)
class SubmissionClaims:
    recipient: str | None = None
    credential_title: str | None = None


@dataclass(slots=True)
class VerificationResult:
    status: VerificationStatus
    issuer_id: str | None
    issuer_name: str | None
    explanation: str
    attempts: list[dict[str, Any]] = field(default_factory=list)
    authoritative_claims: dict[str, str] = field(default_factory=dict)
    claim_comparisons: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class TemplateResult:
    available: bool
    issuer_id: str | None = None
    template_id: str | None = None
    alignment_score: float | None = None
    logo_similarity: float | None = None
    font_shape_similarity: float | None = None
    layout_similarity: float | None = None
    anomaly_score: float | None = None
    explanation: str = "No reference template was available."


@dataclass(slots=True)
class ProvenanceResult:
    capture_method: str = "uncertain"
    capture_confidence: float = 0.0
    moire_score: float | None = None
    ela_score: float | None = None
    copy_move_score: float | None = None
    metadata_flags: list[str] = field(default_factory=list)
    digital_edit_anomaly: float = 0.0
    explanation: str = "Capture method could not be classified."


@dataclass(slots=True)
class ContentResult:
    available: bool = False
    phrase_match_score: float | None = None
    anomaly_score: float | None = None
    explanation: str = "No issuer language profile was available."


@dataclass(slots=True)
class RiskContribution:
    signal: str
    raw_risk: float
    weight: float
    points: float
    explanation: str


@dataclass(slots=True)
class AnalysisReport:
    submission_id: str
    source_sha256: str
    created_at: str
    risk_score: float
    evidence_coverage: float
    review_recommended: bool
    review_reasons: list[str]
    decision: str
    authenticity_assessment: str
    ai_origin_assessment: str
    ruleset_fingerprint: str
    extraction: ExtractionResult
    search: SearchEvidence
    verification: VerificationResult
    template: TemplateResult
    provenance: ProvenanceResult
    content: ContentResult
    contributions: list[RiskContribution]
    checks: list[AuditCheck]
    report_version: str = "1.2"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> None:
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
