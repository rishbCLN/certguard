from __future__ import annotations

import re
import ssl
from dataclasses import dataclass
from html import unescape
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener

from certguard.models import (
    ExtractionResult,
    SubmissionClaims,
    VerificationResult,
    VerificationStatus,
)
from certguard.registry import IssuerDefinition, IssuerRegistry, VerificationEndpoint


class UnsafeRedirectError(RuntimeError):
    pass


class AllowlistedRedirectHandler(HTTPRedirectHandler):
    def __init__(self, allowed_hosts: set[str]) -> None:
        super().__init__()
        self.allowed_hosts = {host.casefold() for host in allowed_hosts}

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, ANN201
        parsed = urlparse(newurl)
        if (
            parsed.scheme.casefold() != "https"
            or not _host_allowed(parsed.hostname, self.allowed_hosts)
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in (None, 443)
        ):
            raise UnsafeRedirectError("Verification service redirected outside its allowlist")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _host_allowed(hostname: str | None, allowed_hosts: set[str]) -> bool:
    if not hostname:
        return False
    hostname = hostname.casefold().rstrip(".")
    return hostname in {host.casefold().rstrip(".") for host in allowed_hosts}


@dataclass(slots=True)
class LookupResponse:
    status_code: int
    final_url: str
    body: str


class LookupClient:
    def get(self, url: str, allowed_hosts: set[str], timeout: float) -> LookupResponse:
        parsed = urlparse(url)
        if (
            parsed.scheme.casefold() != "https"
            or not _host_allowed(parsed.hostname, allowed_hosts)
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in (None, 443)
        ):
            raise ValueError("Verification URL is not an allowlisted HTTPS URL")
        opener = build_opener(
            HTTPSHandler(context=ssl.create_default_context()),
            AllowlistedRedirectHandler(allowed_hosts),
        )
        request = Request(url, headers={"User-Agent": "CertGuard/0.1 verification-triage"})
        with opener.open(request, timeout=timeout) as response:
            body = response.read(1_000_000).decode("utf-8", errors="replace")
            return LookupResponse(response.status, response.url, body)


def _read_http_error_body(exc: HTTPError) -> str:
    try:
        return exc.read(1_000_000).decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return ""


class VerificationService:
    def __init__(
        self,
        registry: IssuerRegistry,
        client: LookupClient | None = None,
        timeout: float = 8.0,
        network_enabled: bool = True,
    ) -> None:
        self.registry = registry
        self.client = client or LookupClient()
        self.timeout = timeout
        self.network_enabled = network_enabled

    def verify(
        self,
        extraction: ExtractionResult,
        submission_claims: SubmissionClaims | None = None,
    ) -> VerificationResult:
        values = extraction.urls + extraction.qr_values
        url_issuers = self.registry.identify("", values)
        issuers = url_issuers or self.registry.identify(extraction.text, values)
        if not issuers:
            return VerificationResult(
                status=VerificationStatus.UNRECOGNIZED_ISSUER,
                issuer_id=None,
                issuer_name=None,
                explanation="The issuer did not match the configured registry; no authenticity conclusion was made.",
            )

        if len(issuers) > 1:
            return VerificationResult(
                status=VerificationStatus.AMBIGUOUS_ISSUER,
                issuer_id=None,
                issuer_name=None,
                explanation="Conflicting issuer evidence was found; no automated issuer lookup was selected.",
            )

        issuer = issuers[0]
        urls = self._matching_urls(issuer, extraction.urls + extraction.qr_values)
        ids = self._extract_ids(issuer, extraction, urls)
        candidates: list[tuple[str, VerificationEndpoint]] = []
        for url in urls:
            endpoint = self._endpoint_for_url(issuer, url)
            if endpoint is not None:
                candidates.append((url, endpoint))
        for certificate_id in ids:
            candidates.extend((endpoint.build_url(certificate_id), endpoint) for endpoint in issuer.endpoints)

        unique_candidates = list(dict.fromkeys((url, endpoint) for url, endpoint in candidates))
        if not unique_candidates:
            status = (
                VerificationStatus.LOOKUP_INCONCLUSIVE
                if urls and not issuer.endpoints
                else VerificationStatus.NO_CODE_PRESENT
            )
            return VerificationResult(
                status=status,
                issuer_id=issuer.issuer_id,
                issuer_name=issuer.display_name,
                explanation=(
                    "A recognized issuer URL was found, but no configured authoritative response parser is available."
                    if status == VerificationStatus.LOOKUP_INCONCLUSIVE
                    else "The issuer was recognized, but no usable verification URL or certificate code was found."
                ),
            )
        if not self.network_enabled:
            return VerificationResult(
                status=VerificationStatus.LOOKUP_UNAVAILABLE,
                issuer_id=issuer.issuer_id,
                issuer_name=issuer.display_name,
                explanation="Verification candidates were found, but network lookup was disabled.",
                attempts=[{"url": url, "outcome": "network-disabled"} for url, _ in unique_candidates],
            )

        attempts: list[dict[str, object]] = []
        saw_authoritative_negative = False
        saw_operational_error = False
        for url, endpoint in unique_candidates[:5]:
            allowed_hosts = set(endpoint.allowed_hosts)
            try:
                response = self.client.get(url, allowed_hosts, self.timeout)
                success_markers = endpoint.success_markers
                failure_markers = endpoint.failure_markers
                body_folded = response.body.casefold()
                failed = any(marker.casefold() in body_folded for marker in failure_markers)
                marker_found = bool(success_markers) and all(
                    marker.casefold() in body_folded for marker in success_markers
                )
                verified = 200 <= response.status_code < 300 and marker_found and not failed
                saw_authoritative_negative = saw_authoritative_negative or failed
                attempts.append(
                    {
                        "url": url,
                        "final_url": response.final_url,
                        "status_code": response.status_code,
                        "outcome": "verified" if verified else "not-confirmed",
                    }
                )
                if verified:
                    authoritative = self._extract_authoritative_claims(response.body, endpoint)
                    comparisons = self._compare_claims(submission_claims, authoritative)
                    return self._claim_bound_result(
                        issuer, attempts, authoritative, comparisons, submission_claims
                    )
            except HTTPError as exc:
                body = _read_http_error_body(exc)
                body_folded = body.casefold()
                authoritative_negative = bool(endpoint.failure_markers) and all(
                    marker.casefold() in body_folded for marker in endpoint.failure_markers
                )
                if authoritative_negative and 400 <= exc.code < 500:
                    saw_authoritative_negative = True
                    attempts.append(
                        {
                            "url": url,
                            "status_code": exc.code,
                            "outcome": "authoritative-negative",
                        }
                    )
                else:
                    saw_operational_error = True
                    attempts.append(
                        {"url": url, "status_code": exc.code, "outcome": "http-error"}
                    )
            except (URLError, TimeoutError, ValueError, UnsafeRedirectError, OSError) as exc:
                saw_operational_error = True
                attempts.append(
                    {"url": url, "outcome": "lookup-error", "error": f"{type(exc).__name__}: {exc}"}
                )

        return VerificationResult(
            status=(
                VerificationStatus.FAILED_LOOKUP
                if saw_authoritative_negative
                else VerificationStatus.LOOKUP_UNAVAILABLE
                if saw_operational_error
                else VerificationStatus.LOOKUP_INCONCLUSIVE
            ),
            issuer_id=issuer.issuer_id,
            issuer_name=issuer.display_name,
            explanation=(
                "The issuer service returned an explicit not-found or invalid response."
                if saw_authoritative_negative
                else "The issuer lookup could not produce a conclusive authenticity result."
            ),
            attempts=attempts,
        )

    @staticmethod
    def _claim_bound_result(
        issuer: IssuerDefinition,
        attempts: list[dict[str, object]],
        authoritative: dict[str, str],
        comparisons: dict[str, str],
        submission_claims: SubmissionClaims | None,
    ) -> VerificationResult:
        base = {
            "issuer_id": issuer.issuer_id,
            "issuer_name": issuer.display_name,
            "attempts": attempts,
            "authoritative_claims": authoritative,
            "claim_comparisons": comparisons,
        }
        if "mismatch" in comparisons.values():
            return VerificationResult(
                status=VerificationStatus.CLAIMS_MISMATCH,
                explanation=(
                    "The issuer record exists, but its identity or credential claims conflict "
                    "with the submitted certificate."
                ),
                **base,
            )
        supplied = VerificationService._supplied_claims(submission_claims)
        if any(field not in comparisons for field in supplied):
            # A trusted claim was supplied but the record did not expose it, so
            # the certificate is not fully bound; never return VERIFIED here.
            return VerificationResult(
                status=VerificationStatus.LOOKUP_INCONCLUSIVE,
                explanation=(
                    "The issuer record was found, but it did not expose every claim supplied "
                    "from the trusted submission record; the certificate is not fully bound "
                    "to the record."
                ),
                **base,
            )
        if comparisons and all(value == "match" for value in comparisons.values()):
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                explanation=(
                    "The issuer record was found and its available claims matched the "
                    "submitted certificate."
                ),
                **base,
            )
        return VerificationResult(
            status=VerificationStatus.RECORD_FOUND,
            explanation=(
                "The issuer record was found, but there was not enough structured claim data "
                "to bind it to the submitted certificate."
            ),
            **base,
        )

    @staticmethod
    def _endpoint_for_url(
        issuer: IssuerDefinition, url: str
    ) -> VerificationEndpoint | None:
        hostname = urlparse(url).hostname
        for endpoint in issuer.endpoints:
            if _host_allowed(hostname, set(endpoint.allowed_hosts)):
                return endpoint
        return None

    @staticmethod
    def _matching_urls(issuer: IssuerDefinition, values: list[str]) -> list[str]:
        return list(
            dict.fromkeys(
                value
                for value in values
                if urlparse(value).scheme.casefold() == "https"
                and any(re.fullmatch(pattern, value, re.IGNORECASE) for pattern in issuer.verification_url_patterns)
            )
        )

    @staticmethod
    def _extract_ids(
        issuer: IssuerDefinition, extraction: ExtractionResult, urls: list[str]
    ) -> list[str]:
        ids: list[str] = []
        source = "\n".join([*urls, *extraction.qr_values, extraction.text])
        for pattern in issuer.certificate_id_patterns:
            ids.extend(match.group(1) for match in re.finditer(pattern, source, re.IGNORECASE))
        ids.extend(extraction.certificate_ids)
        return list(dict.fromkeys(ids))[:5]

    @staticmethod
    def _extract_authoritative_claims(
        body: str, endpoint: VerificationEndpoint
    ) -> dict[str, str]:
        text = unescape(body)
        claims: dict[str, str] = {}
        for field, patterns in (
            ("recipient", endpoint.recipient_patterns),
            ("credential_title", endpoint.credential_patterns),
        ):
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    claims[field] = VerificationService._normalize_claim(match.group(1))
                    break
        return claims

    @staticmethod
    def _compare_claims(
        submitted: SubmissionClaims | None, authoritative: dict[str, str]
    ) -> dict[str, str]:
        if submitted is None:
            return {}
        comparisons: dict[str, str] = {}
        for field in ("recipient", "credential_title"):
            submitted_value = getattr(submitted, field)
            authoritative_value = authoritative.get(field)
            if not submitted_value or not authoritative_value:
                continue
            left = VerificationService._normalize_claim(submitted_value)
            comparisons[field] = "match" if left == authoritative_value else "mismatch"
        return comparisons

    @staticmethod
    def _supplied_claims(submitted: SubmissionClaims | None) -> list[str]:
        if submitted is None:
            return []
        return [
            field
            for field in ("recipient", "credential_title")
            if getattr(submitted, field)
        ]

    @staticmethod
    def _normalize_claim(value: str) -> str:
        return " ".join(re.sub(r"[^\w\s]", " ", value.casefold()).split())
