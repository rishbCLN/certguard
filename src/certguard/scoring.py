from __future__ import annotations

from certguard.models import (
    ContentResult,
    ProvenanceResult,
    RiskContribution,
    TemplateResult,
    VerificationResult,
    VerificationStatus,
)

VERIFICATION_RISK = {
    VerificationStatus.VERIFIED: 0.02,
    VerificationStatus.CLAIMS_MISMATCH: 1.0,
    VerificationStatus.FAILED_LOOKUP: 0.95,
}

SCORABLE_WEIGHT = 0.93


def calculate_risk(
    verification: VerificationResult,
    template: TemplateResult,
    provenance: ProvenanceResult,
    content: ContentResult,
) -> tuple[float, float, list[RiskContribution]]:
    signals: list[tuple[str, float, float, str]] = []
    if verification.status in VERIFICATION_RISK:
        signals.append(
            (
                "verification_lookup",
                VERIFICATION_RISK[verification.status],
                0.70,
                verification.explanation,
            )
        )
    if template.available and template.anomaly_score is not None:
        signals.append(
            ("template_layout", template.anomaly_score, 0.20, template.explanation)
        )
    if content.available and content.anomaly_score is not None:
        signals.append(("content_plausibility", content.anomaly_score, 0.03, content.explanation))

    contributions = [
        RiskContribution(
            signal=name,
            raw_risk=round(raw_risk, 3),
            weight=weight,
            points=round(raw_risk * weight * 100, 1),
            explanation=explanation,
        )
        for name, raw_risk, weight, explanation in signals
    ]
    coverage = round(sum(weight for _, _, weight, _ in signals) / SCORABLE_WEIGHT, 3)
    return round(sum(item.points for item in contributions), 1), coverage, contributions
