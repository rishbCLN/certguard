from __future__ import annotations

import json
import re
import ssl
from typing import Protocol
from urllib.parse import urlencode, urlparse
from urllib.request import HTTPSHandler, Request, build_opener

from certguard.models import ExtractionResult, SearchEvidence, SearchResult
from certguard.registry import IssuerDefinition, IssuerRegistry

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


class SearchClient(Protocol):
    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]: ...


class BraveSearchClient:
    def __init__(self, api_key: str, timeout: float = 8.0) -> None:
        if not api_key.strip():
            raise ValueError("A non-empty Brave Search API key is required")
        self.api_key = api_key
        self.timeout = timeout

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        params = urlencode({"q": query, "count": min(max(max_results, 1), 20)})
        request = Request(
            f"{BRAVE_SEARCH_URL}?{params}",
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key,
                "User-Agent": "CertGuard/0.1 verification-discovery",
            },
        )
        opener = build_opener(HTTPSHandler(context=ssl.create_default_context()))
        with opener.open(request, timeout=self.timeout) as response:
            payload = json.loads(response.read(1_000_000).decode("utf-8"))
        raw_results = payload.get("web", {}).get("results", [])
        if not isinstance(raw_results, list):
            raise ValueError("Brave Search returned an invalid result payload")
        return [
            SearchResult(
                title=str(item.get("title", "")),
                url=str(item.get("url", "")),
                description=str(item.get("description", "")),
            )
            for item in raw_results[:max_results]
            if isinstance(item, dict) and item.get("url")
        ]


def discover_official_pages(
    extraction: ExtractionResult,
    registry: IssuerRegistry,
    client: SearchClient | None,
    *,
    enabled: bool,
) -> SearchEvidence:
    if not enabled:
        return SearchEvidence()
    if client is None:
        return SearchEvidence(
            enabled=True,
            explanation="Online search was enabled, but no search client was configured.",
            error="search-client-unavailable",
        )

    issuers = registry.identify(extraction.text, extraction.urls + extraction.qr_values)
    if len(issuers) != 1:
        return SearchEvidence(
            enabled=True,
            explanation=(
                "Online search was skipped because the certificate issuer was not uniquely recognized."
            ),
        )
    issuer = issuers[0]
    certificate_ids = _issuer_certificate_ids(issuer, extraction)
    if not certificate_ids:
        return SearchEvidence(
            enabled=True,
            issuer_id=issuer.issuer_id,
            explanation="Online search was skipped because no certificate identifier was extracted.",
        )

    query = _build_query(issuer, certificate_ids[0])
    try:
        results = client.search(query, max_results=5)
    except Exception as exc:
        return SearchEvidence(
            enabled=True,
            issuer_id=issuer.issuer_id,
            query=query,
            explanation="The search provider was unavailable; no adverse conclusion was made.",
            error=f"{type(exc).__name__}: {exc}",
        )

    accepted_urls: list[str] = []
    for result in results:
        result.accepted = _is_official_verification_url(result.url, issuer)
        if result.accepted:
            accepted_urls.append(result.url)
    accepted_urls = list(dict.fromkeys(accepted_urls))
    return SearchEvidence(
        enabled=True,
        issuer_id=issuer.issuer_id,
        query=query,
        results=results,
        accepted_urls=accepted_urls,
        explanation=(
            "Official verification pages were discovered and passed to issuer verification."
            if accepted_urls
            else "Search completed, but no result matched an approved issuer verification URL."
        ),
    )


def _issuer_certificate_ids(
    issuer: IssuerDefinition, extraction: ExtractionResult
) -> list[str]:
    source = "\n".join([extraction.text, *extraction.urls, *extraction.qr_values])
    ids = [
        match.group(1)
        for pattern in issuer.certificate_id_patterns
        for match in re.finditer(pattern, source, re.IGNORECASE)
    ]
    ids.extend(extraction.certificate_ids)
    return list(dict.fromkeys(ids))[:5]


def _build_query(issuer: IssuerDefinition, certificate_id: str) -> str:
    domains = issuer.official_domains or issuer.allowed_hosts
    site_filter = " OR ".join(f"site:{domain}" for domain in domains)
    return f'"{issuer.display_name}" "{certificate_id}" certificate ({site_filter})'


def _is_official_verification_url(url: str, issuer: IssuerDefinition) -> bool:
    parsed = urlparse(url)
    if (
        parsed.scheme.casefold() != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
        or not _domain_allowed(parsed.hostname, issuer.official_domains or issuer.allowed_hosts)
    ):
        return False
    return any(
        re.fullmatch(pattern, url, re.IGNORECASE)
        for pattern in issuer.verification_url_patterns
    )


def _domain_allowed(hostname: str | None, domains: tuple[str, ...]) -> bool:
    if not hostname:
        return False
    hostname = hostname.casefold().rstrip(".")
    return any(
        hostname == domain.casefold().rstrip(".")
        or hostname.endswith(f".{domain.casefold().rstrip('.')}")
        for domain in domains
    )
