from certguard.models import (
    ContentResult,
    ProvenanceResult,
    TemplateResult,
    VerificationResult,
    VerificationStatus,
)
from certguard.scoring import calculate_risk


def verification(status: VerificationStatus) -> VerificationResult:
    return VerificationResult(status, "issuer", "Issuer", "Lookup result")


def test_verified_lookup_dominates_missing_optional_signals() -> None:
    score, coverage, contributions = calculate_risk(
        verification(VerificationStatus.VERIFIED),
        TemplateResult(available=False),
        ProvenanceResult(capture_method="print-recapture"),
        ContentResult(available=False),
    )

    assert score == 1.4
    assert coverage == 0.753
    assert [item.signal for item in contributions] == ["verification_lookup"]


def test_provenance_capture_method_is_not_scored_as_fraud() -> None:
    digital_score, _, _ = calculate_risk(
        verification(VerificationStatus.VERIFIED),
        TemplateResult(available=False),
        ProvenanceResult(capture_method="born-digital"),
        ContentResult(available=False),
    )
    recapture_score, _, _ = calculate_risk(
        verification(VerificationStatus.VERIFIED),
        TemplateResult(available=False),
        ProvenanceResult(capture_method="screen-recapture", moire_score=0.9),
        ContentResult(available=False),
    )

    assert recapture_score == digital_score


def test_failed_lookup_scores_higher_than_content_wording() -> None:
    score, coverage, contributions = calculate_risk(
        verification(VerificationStatus.FAILED_LOOKUP),
        TemplateResult(available=True, anomaly_score=0.0),
        ProvenanceResult(),
        ContentResult(available=True, anomaly_score=1.0),
    )

    assert score > 60
    assert contributions[0].points > contributions[-1].points
    assert coverage == 1.0


def test_operational_lookup_failure_is_not_scored_as_fraud() -> None:
    score, coverage, contributions = calculate_risk(
        verification(VerificationStatus.LOOKUP_UNAVAILABLE),
        TemplateResult(available=False),
        ProvenanceResult(),
        ContentResult(available=False),
    )

    assert score == 0
    assert coverage == 0
    assert contributions == []


def test_missing_optional_signal_does_not_inflate_other_weights() -> None:
    score, _, contributions = calculate_risk(
        verification(VerificationStatus.FAILED_LOOKUP),
        TemplateResult(available=False),
        ProvenanceResult(),
        ContentResult(available=False),
    )

    assert score == 66.5
    assert contributions[0].weight == 0.7
