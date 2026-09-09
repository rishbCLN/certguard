from __future__ import annotations

from certguard.models import ContentResult, ExtractionResult
from certguard.registry import IssuerDefinition


def analyze_content(
    extraction: ExtractionResult, issuer: IssuerDefinition | None
) -> ContentResult:
    if (
        issuer is None
        or not issuer.language_phrases
        or not extraction.text
        or extraction.ocr_confidence is None
        or extraction.ocr_confidence < 0.55
    ):
        return ContentResult()
    text = " ".join(extraction.text.casefold().split())
    matches = sum(phrase.casefold() in text for phrase in issuer.language_phrases)
    score = matches / len(issuer.language_phrases)
    return ContentResult(
        available=True,
        phrase_match_score=round(score, 3),
        anomaly_score=round(1 - score, 3),
        explanation=(
            f"Matched {matches} of {len(issuer.language_phrases)} configured issuer-language phrases. "
            "Wording is treated only as supplementary evidence."
        ),
    )
