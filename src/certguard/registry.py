from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True, slots=True)
class VerificationEndpoint:
    url_template: str
    allowed_hosts: tuple[str, ...]
    success_markers: tuple[str, ...] = ()
    failure_markers: tuple[str, ...] = ()
    recipient_patterns: tuple[str, ...] = ()
    credential_patterns: tuple[str, ...] = ()

    def build_url(self, certificate_id: str) -> str:
        from urllib.parse import quote

        return self.url_template.format(certificate_id=quote(certificate_id, safe="-._~"))


@dataclass(frozen=True, slots=True)
class IssuerDefinition:
    issuer_id: str
    display_name: str
    aliases: tuple[str, ...]
    verification_url_patterns: tuple[str, ...]
    allowed_hosts: tuple[str, ...]
    official_domains: tuple[str, ...] = ()
    endpoints: tuple[VerificationEndpoint, ...] = ()
    certificate_id_patterns: tuple[str, ...] = ()
    language_phrases: tuple[str, ...] = ()
    templates: tuple[dict[str, object], ...] = ()


@dataclass(slots=True)
class IssuerRegistry:
    issuers: dict[str, IssuerDefinition] = field(default_factory=dict)

    @classmethod
    def default(cls) -> IssuerRegistry:
        resource = files("certguard.data").joinpath("issuers.json")
        return cls.from_data(json.loads(resource.read_text(encoding="utf-8")))

    @classmethod
    def from_file(cls, path: Path) -> IssuerRegistry:
        return cls.from_data(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def from_data(cls, data: dict[str, object]) -> IssuerRegistry:
        definitions: dict[str, IssuerDefinition] = {}
        raw_issuers = data.get("issuers", [])
        if not isinstance(raw_issuers, list):
            raise ValueError("Registry 'issuers' must be a list")
        for raw in raw_issuers:
            if not isinstance(raw, dict):
                raise ValueError("Each issuer must be an object")
            issuer_id = str(raw["id"])
            endpoints = tuple(
                VerificationEndpoint(
                    url_template=str(endpoint["url_template"]),
                    allowed_hosts=tuple(endpoint.get("allowed_hosts", [])),
                    success_markers=tuple(endpoint.get("success_markers", [])),
                    failure_markers=tuple(endpoint.get("failure_markers", [])),
                    recipient_patterns=tuple(endpoint.get("recipient_patterns", [])),
                    credential_patterns=tuple(endpoint.get("credential_patterns", [])),
                )
                for endpoint in raw.get("endpoints", [])
            )
            definition = IssuerDefinition(
                issuer_id=issuer_id,
                display_name=str(raw["display_name"]),
                aliases=tuple(raw.get("aliases", [])),
                verification_url_patterns=tuple(raw.get("verification_url_patterns", [])),
                allowed_hosts=tuple(raw.get("allowed_hosts", [])),
                official_domains=tuple(raw.get("official_domains", raw.get("allowed_hosts", []))),
                endpoints=endpoints,
                certificate_id_patterns=tuple(raw.get("certificate_id_patterns", [])),
                language_phrases=tuple(raw.get("language_phrases", [])),
                templates=tuple(raw.get("templates", [])),
            )
            cls._validate(definition)
            if issuer_id in definitions:
                raise ValueError(f"Duplicate issuer id: {issuer_id}")
            definitions[issuer_id] = definition
        return cls(definitions)

    @staticmethod
    def _validate(issuer: IssuerDefinition) -> None:
        for pattern in (*issuer.verification_url_patterns, *issuer.certificate_id_patterns):
            re.compile(pattern)
        for endpoint in issuer.endpoints:
            parsed = urlparse(endpoint.url_template)
            if parsed.scheme != "https" or not parsed.hostname:
                raise ValueError(f"Endpoint for {issuer.issuer_id} must use HTTPS")
            if parsed.hostname.lower() not in {host.lower() for host in endpoint.allowed_hosts}:
                raise ValueError(f"Endpoint host for {issuer.issuer_id} is not allowlisted")
            if not endpoint.success_markers:
                raise ValueError(f"Endpoint for {issuer.issuer_id} needs positive response markers")
            for pattern in (*endpoint.recipient_patterns, *endpoint.credential_patterns):
                if re.compile(pattern).groups != 1:
                    raise ValueError(
                        f"Claim pattern for {issuer.issuer_id} must have one capture group"
                    )
        for domain in issuer.official_domains:
            parsed = urlparse(f"https://{domain}")
            if not parsed.hostname or parsed.hostname != domain.casefold().rstrip("."):
                raise ValueError(f"Official domain for {issuer.issuer_id} is invalid")

    def identify(self, text: str, urls: list[str]) -> list[IssuerDefinition]:
        haystack = text.casefold()
        matches: list[IssuerDefinition] = []
        for issuer in self.issuers.values():
            alias_match = any(alias.casefold() in haystack for alias in issuer.aliases)
            url_match = any(
                re.search(pattern, url, flags=re.IGNORECASE)
                for pattern in issuer.verification_url_patterns
                for url in urls
            )
            if alias_match or url_match:
                matches.append(issuer)
        return matches

    def get(self, issuer_id: str | None) -> IssuerDefinition | None:
        return self.issuers.get(issuer_id) if issuer_id else None
