from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from certguard.audit import JsonlAuditSink
from certguard.content import analyze_content
from certguard.document import extract_loaded_document, load_document_pages
from certguard.forensics import ProvenanceAnalyzer, TemplateAnalyzer
from certguard.models import (
    AnalysisReport,
    AuditCheck,
    CheckState,
    ContentResult,
    ExtractionResult,
    ProvenanceResult,
    SearchEvidence,
    SubmissionClaims,
    TemplateResult,
    VerificationResult,
    VerificationStatus,
)
from certguard.registry import IssuerRegistry
from certguard.scoring import calculate_risk
from certguard.search import SearchClient, discover_official_pages
from certguard.verification import VerificationService


class CertGuardPipeline:
    def __init__(
        self,
        registry: IssuerRegistry | None = None,
        template_root: Path | None = None,
        audit_sink: JsonlAuditSink | None = None,
        network_enabled: bool = True,
        search_client: SearchClient | None = None,
        search_enabled: bool = False,
        review_threshold: float = 55.0,
    ) -> None:
        if not 0 < review_threshold <= 100:
            raise ValueError("review_threshold must be within (0, 100]")
        self.registry = registry or IssuerRegistry.default()
        self.verification = VerificationService(
            self.registry, network_enabled=network_enabled
        )
        self.templates = TemplateAnalyzer(template_root)
        self.provenance = ProvenanceAnalyzer()
        self.search_client = search_client
        self.search_enabled = search_enabled and network_enabled
        self.audit_sink = audit_sink
        self.review_threshold = review_threshold

    def analyze(
        self,
        source_path: Path,
        submission_id: str | None = None,
        expected_recipient: str | None = None,
        expected_credential_title: str | None = None,
    ) -> AnalysisReport:
        source_path = source_path.resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        submission_id = submission_id or str(uuid.uuid4())
        source_hash = _sha256(source_path)
        checks: list[AuditCheck] = []

        started = time.perf_counter()
        document = load_document_pages(source_path)
        image = document.images[0]
        checks.append(
            AuditCheck(
                name="document_load",
                state=CheckState.COMPLETED,
                summary="Document decoded for analysis.",
                evidence={
                    "page_count": document.page_count,
                    "rendered_page_count": len(document.images),
                    "width": image.shape[1],
                    "height": image.shape[0],
                },
                duration_ms=_elapsed_ms(started),
            )
        )

        started = time.perf_counter()
        extraction = extract_loaded_document(document)
        extraction_state = CheckState.COMPLETED if not extraction.errors else CheckState.ERROR
        checks.append(
            AuditCheck(
                name="ocr_and_qr_extraction",
                state=extraction_state,
                summary=(
                    "Text and machine-readable codes were extracted."
                    if not extraction.errors
                    else "Extraction completed with unavailable or failed components."
                ),
                evidence={
                    "ocr_confidence": extraction.ocr_confidence,
                    "certificate_id_count": len(extraction.certificate_ids),
                    "structured_field_count": len(extraction.structured_fields),
                    "url_count": len(extraction.urls),
                    "qr_count": len(extraction.qr_values),
                    "text_page_count": sum(bool(page.text) for page in extraction.pages),
                    "text_sources": sorted(
                        {source for page in extraction.pages for source in page.text_sources}
                    ),
                    "errors": extraction.errors,
                },
                duration_ms=_elapsed_ms(started),
            )
        )

        search = self._search(extraction, checks)
        extraction.urls = list(dict.fromkeys([*extraction.urls, *search.accepted_urls]))
        claims = SubmissionClaims(expected_recipient, expected_credential_title)
        verification = self._verify(extraction, checks, claims)
        issuer = self.registry.get(verification.issuer_id)

        started = time.perf_counter()
        try:
            template = self.templates.analyze(image, issuer)
            template_state = CheckState.COMPLETED if template.available else CheckState.SKIPPED
            checks.append(
                AuditCheck(
                    name="template_layout_forensics",
                    state=template_state,
                    summary=template.explanation,
                    evidence={
                        "template_id": template.template_id,
                        "alignment_score": template.alignment_score,
                        "logo_similarity": template.logo_similarity,
                        "font_shape_similarity": template.font_shape_similarity,
                        "layout_similarity": template.layout_similarity,
                    },
                    duration_ms=_elapsed_ms(started),
                )
            )
        except Exception as exc:
            template = TemplateResult(
                available=False,
                issuer_id=verification.issuer_id,
                explanation="Template analysis failed and was excluded from risk scoring.",
            )
            checks.append(_error_check("template_layout_forensics", exc, started))

        started = time.perf_counter()
        try:
            provenance = self.provenance.analyze(image, source_path)
            checks.append(
                AuditCheck(
                    name="recapture_and_provenance",
                    state=CheckState.COMPLETED,
                    summary=provenance.explanation,
                    evidence={
                        "capture_method": provenance.capture_method,
                        "capture_confidence": provenance.capture_confidence,
                        "moire_score": provenance.moire_score,
                        "ela_score": provenance.ela_score,
                        "copy_move_score": provenance.copy_move_score,
                        "metadata_flags": provenance.metadata_flags,
                    },
                    duration_ms=_elapsed_ms(started),
                )
            )
        except Exception as exc:
            provenance = ProvenanceResult(explanation="Provenance analysis failed.")
            checks.append(_error_check("recapture_and_provenance", exc, started))

        started = time.perf_counter()
        content = analyze_content(extraction, issuer)
        checks.append(
            AuditCheck(
                name="content_plausibility",
                state=CheckState.COMPLETED if content.available else CheckState.SKIPPED,
                summary=content.explanation,
                evidence={"phrase_match_score": content.phrase_match_score},
                duration_ms=_elapsed_ms(started),
            )
        )

        risk_score, coverage, contributions = calculate_risk(
            verification, template, provenance, content
        )
        review_reasons = _review_reasons(
            verification, template, coverage, risk_score, self.review_threshold
        )
        report = AnalysisReport(
            submission_id=submission_id,
            source_sha256=source_hash,
            created_at=datetime.now(UTC).isoformat(),
            risk_score=risk_score,
            evidence_coverage=coverage,
            review_recommended=bool(review_reasons),
            review_reasons=review_reasons,
            decision="human-review-triage-only",
            authenticity_assessment=_authenticity_assessment(verification, template),
            ai_origin_assessment=(
                "not-determined: visual style alone cannot reliably establish whether a certificate "
                "was AI-generated; CertGuard checks issuer records and document consistency instead"
            ),
            ruleset_fingerprint=_ruleset_fingerprint(self.registry),
            extraction=extraction,
            search=search,
            verification=verification,
            template=template,
            provenance=provenance,
            content=content,
            contributions=contributions,
            checks=checks,
        )
        if self.audit_sink:
            self.audit_sink.append(report)
        return report

    def _search(
        self, extraction: ExtractionResult, checks: list[AuditCheck]
    ) -> SearchEvidence:
        started = time.perf_counter()
        result = discover_official_pages(
            extraction,
            self.registry,
            self.search_client,
            enabled=self.search_enabled,
        )
        checks.append(
            AuditCheck(
                name="official_web_search",
                state=(
                    CheckState.ERROR
                    if result.error and result.error != "search-client-unavailable"
                    else CheckState.COMPLETED
                    if result.enabled
                    else CheckState.SKIPPED
                ),
                summary=result.explanation,
                evidence={
                    "issuer_id": result.issuer_id,
                    "result_count": len(result.results),
                    "accepted_url_count": len(result.accepted_urls),
                    "error": result.error,
                },
                duration_ms=_elapsed_ms(started),
            )
        )
        return result

    def _verify(
        self,
        extraction: ExtractionResult,
        checks: list[AuditCheck],
        submission_claims: SubmissionClaims,
    ) -> VerificationResult:
        started = time.perf_counter()
        try:
            result = self.verification.verify(extraction, submission_claims)
            checks.append(
                AuditCheck(
                    name="issuer_verification_lookup",
                    state=(
                        CheckState.SKIPPED
                        if result.status == VerificationStatus.LOOKUP_UNAVAILABLE
                        and any(
                            attempt.get("outcome") == "network-disabled"
                            for attempt in result.attempts
                        )
                        else CheckState.COMPLETED
                    ),
                    summary=result.explanation,
                    evidence={
                        "status": result.status,
                        "issuer_id": result.issuer_id,
                        "attempts": result.attempts,
                    },
                    duration_ms=_elapsed_ms(started),
                )
            )
            return result
        except Exception as exc:
            checks.append(_error_check("issuer_verification_lookup", exc, started))
            return VerificationResult(
                status=VerificationStatus.LOOKUP_UNAVAILABLE,
                issuer_id=None,
                issuer_name=None,
                explanation="Verification lookup failed operationally; a reviewer should retry it.",
            )


def _elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _error_check(name: str, error: Exception, started: float) -> AuditCheck:
    return AuditCheck(
        name=name,
        state=CheckState.ERROR,
        summary="The check failed and was not silently converted into adverse evidence.",
        evidence={"error": f"{type(error).__name__}: {error}"},
        duration_ms=_elapsed_ms(started),
    )


def _review_reasons(
    verification: VerificationResult,
    template: TemplateResult,
    coverage: float,
    risk_score: float,
    review_threshold: float,
) -> list[str]:
    reasons: list[str] = []
    if verification.status in {
        VerificationStatus.CLAIMS_MISMATCH,
        VerificationStatus.FAILED_LOOKUP,
    }:
        reasons.append("Issuer verification returned adverse evidence.")
    elif verification.status != VerificationStatus.VERIFIED:
        reasons.append("The certificate was not fully bound to a matching issuer record.")
    if template.available and template.anomaly_score is not None and template.anomaly_score >= 0.45:
        reasons.append("The layout differs substantially from a configured issuer reference.")
    if coverage < 0.75:
        reasons.append("Evidence coverage is limited; manual verification is required.")
    if risk_score >= review_threshold and not reasons:
        reasons.append("The combined adverse evidence exceeds the review threshold.")
    return reasons


def _authenticity_assessment(
    verification: VerificationResult, template: TemplateResult
) -> str:
    if verification.status == VerificationStatus.CLAIMS_MISMATCH:
        return "issuer-record-mismatch"
    if verification.status == VerificationStatus.FAILED_LOOKUP:
        return "issuer-record-not-confirmed"
    if verification.status == VerificationStatus.VERIFIED:
        if template.available and (template.anomaly_score or 0) < 0.35:
            return "strong-consistency-evidence"
        return "issuer-record-consistent"
    return "inconclusive"


def _ruleset_fingerprint(registry: IssuerRegistry) -> str:
    payload = []
    for issuer in sorted(registry.issuers.values(), key=lambda item: item.issuer_id):
        payload.append(
            {
                "id": issuer.issuer_id,
                "urls": issuer.verification_url_patterns,
                "official_domains": issuer.official_domains,
                "ids": issuer.certificate_id_patterns,
                "endpoints": [endpoint.url_template for endpoint in issuer.endpoints],
                "templates": issuer.templates,
            }
        )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
