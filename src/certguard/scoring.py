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

BASE_SCORABLE_WEIGHT = 0.93
FORENSIC_MODEL_WEIGHT = 0.07


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
    if provenance.neural_forgery_score is not None:
        artifact_scores = [
            score
            for score in (
                provenance.jpeg_grid_score,
                provenance.frequency_anomaly_score,
                provenance.font_subpixel_score,
                provenance.digital_edit_anomaly,
            )
            if score is not None
        ]
        artifact_score = sum(artifact_scores) / len(artifact_scores) if artifact_scores else 0.0
        forensic_score = 0.7 * provenance.neural_forgery_score + 0.3 * artifact_score
        signals.append(
            (
                "forgery_model_consensus",
                forensic_score,
                FORENSIC_MODEL_WEIGHT,
                "A configured neural model was combined with image-artifact signals; "
                "this is triage evidence, not proof of origin.",
            )
        )

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
    model_expected = provenance.neural_model_available
    scorable_weight = BASE_SCORABLE_WEIGHT + (FORENSIC_MODEL_WEIGHT if model_expected else 0)
    coverage = round(sum(weight for _, _, weight, _ in signals) / scorable_weight, 3)
    return round(sum(item.points for item in contributions), 1), coverage, contributions
