import io
from urllib.error import HTTPError, URLError

from certguard.models import ExtractionResult, SubmissionClaims, VerificationStatus
from certguard.registry import IssuerRegistry
from certguard.verification import LookupResponse, VerificationService, _host_allowed


class StubClient:
    def __init__(self, body: str, status_code: int = 200) -> None:
        self.body = body
        self.status_code = status_code

    def get(self, url: str, allowed_hosts: set[str], timeout: float) -> LookupResponse:
        assert url.startswith("https://")
        assert allowed_hosts
        assert timeout > 0
        return LookupResponse(self.status_code, url, self.body)


class FailingClient:
    def get(self, url: str, allowed_hosts: set[str], timeout: float) -> LookupResponse:
        raise URLError("issuer unavailable")


class HttpErrorClient:
    def __init__(self, code: int, body: str) -> None:
        self.code = code
        self.body = body

    def get(self, url: str, allowed_hosts: set[str], timeout: float) -> LookupResponse:
        raise HTTPError(url, self.code, "error", None, io.BytesIO(self.body.encode()))


class RecordingClient(StubClient):
    def __init__(self, body: str) -> None:
        super().__init__(body)
        self.urls: list[str] = []

    def get(self, url: str, allowed_hosts: set[str], timeout: float) -> LookupResponse:
        self.urls.append(url)
        return super().get(url, allowed_hosts, timeout)


def registry_with_claims() -> IssuerRegistry:
    return IssuerRegistry.from_data(
        {
            "issuers": [
                {
                    "id": "example",
                    "display_name": "Example",
                    "aliases": ["Example"],
                    "verification_url_patterns": [
                        "^https://verify\\.example\\.org/c/[A-Za-z0-9]+$"
                    ],
                    "allowed_hosts": ["verify.example.org"],
                    "certificate_id_patterns": ["/c/([A-Za-z0-9]+)"],
                    "endpoints": [
                        {
                            "url_template": "https://verify.example.org/c/{certificate_id}",
                            "allowed_hosts": ["verify.example.org"],
                            "success_markers": ["Credential valid"],
                            "failure_markers": ["Credential not found"],
                            "recipient_patterns": ["Recipient: ([A-Za-z ]+) Credential:"],
                            "credential_patterns": ["Credential: ([A-Za-z ]+)$"],
                        }
                    ],
                }
            ]
        }
    )


def extraction_for_example() -> ExtractionResult:
    return ExtractionResult(
        text="Example",
        urls=["https://verify.example.org/c/ABC123"],
    )


def test_record_without_claim_binding_is_not_fully_verified() -> None:
    service = VerificationService(
        IssuerRegistry.default(), client=StubClient("CodeRank Certificate")
    )
    extraction = ExtractionResult(
        text="CodeRank certificate",
        urls=["https://www.coderank.com/certificates/abc123def"],
    )

    result = service.verify(extraction)

    assert result.status == VerificationStatus.RECORD_FOUND
    assert result.issuer_id == "coderank"


def test_recognized_issuer_without_code_is_not_failed_lookup() -> None:
    service = VerificationService(IssuerRegistry.default(), network_enabled=False)

    result = service.verify(ExtractionResult(text="Awarded by Coursera"))

    assert result.status == VerificationStatus.NO_CODE_PRESENT


def test_unrecognized_issuer_is_reported_separately() -> None:
    service = VerificationService(IssuerRegistry.default(), network_enabled=False)

    result = service.verify(ExtractionResult(text="Example Training Company"))

    assert result.status == VerificationStatus.UNRECOGNIZED_ISSUER


def test_generic_success_page_does_not_verify() -> None:
    service = VerificationService(IssuerRegistry.default(), client=StubClient("CodeRank"))

    result = service.verify(
        ExtractionResult(
            text="CodeRank certificate",
            urls=["https://www.coderank.com/certificates/abc123def"],
        )
    )

    assert result.status == VerificationStatus.LOOKUP_INCONCLUSIVE


def test_authoritative_claim_mismatch_is_adverse() -> None:
    body = "Credential valid Recipient: Alice Example Credential: Python Basics"
    service = VerificationService(registry_with_claims(), client=StubClient(body))

    result = service.verify(
        ExtractionResult(
            text="Example",
            urls=["https://verify.example.org/c/ABC123"],
        ),
        SubmissionClaims(recipient="Mallory Example", credential_title="Python Basics"),
    )

    assert result.status == VerificationStatus.CLAIMS_MISMATCH
    assert result.claim_comparisons["recipient"] == "mismatch"


def test_authoritative_claim_match_verifies() -> None:
    body = "Credential valid Recipient: Alice Example Credential: Python Basics"
    service = VerificationService(registry_with_claims(), client=StubClient(body))

    result = service.verify(
        ExtractionResult(
            text="Example",
            urls=["https://verify.example.org/c/ABC123"],
        ),
        SubmissionClaims(recipient="Alice Example", credential_title="Python Basics"),
    )

    assert result.status == VerificationStatus.VERIFIED


def test_partially_exposed_claims_do_not_verify() -> None:
    # The page exposes only the recipient, so the supplied trusted credential
    # title is never checked; the record must not become fully verified.
    body = "Credential valid Recipient: Alice Example Credential:"
    service = VerificationService(registry_with_claims(), client=StubClient(body))

    result = service.verify(
        extraction_for_example(),
        SubmissionClaims(recipient="Alice Example", credential_title="Python Basics"),
    )

    assert result.status == VerificationStatus.LOOKUP_INCONCLUSIVE
    assert result.claim_comparisons == {"recipient": "match"}


def test_http_error_with_failure_markers_is_authoritative_negative() -> None:
    service = VerificationService(
        registry_with_claims(),
        client=HttpErrorClient(404, "<html>Credential not found</html>"),
    )

    result = service.verify(extraction_for_example())

    assert result.status == VerificationStatus.FAILED_LOOKUP
    assert result.attempts[0]["outcome"] == "authoritative-negative"


def test_http_server_error_remains_operationally_unavailable() -> None:
    service = VerificationService(
        registry_with_claims(),
        client=HttpErrorClient(503, "temporarily unavailable"),
    )

    result = service.verify(extraction_for_example())

    assert result.status == VerificationStatus.LOOKUP_UNAVAILABLE


def test_http_not_found_without_failure_markers_is_inconclusive() -> None:
    service = VerificationService(
        registry_with_claims(),
        client=HttpErrorClient(404, "some unknown error page"),
    )

    result = service.verify(extraction_for_example())

    assert result.status == VerificationStatus.LOOKUP_UNAVAILABLE


def test_network_failure_is_non_adverse() -> None:
    service = VerificationService(IssuerRegistry.default(), client=FailingClient())

    result = service.verify(
        ExtractionResult(
            text="CodeRank",
            urls=["https://www.coderank.com/certificates/abc123def"],
        )
    )

    assert result.status == VerificationStatus.LOOKUP_UNAVAILABLE


def test_host_allowlist_does_not_implicitly_allow_subdomains() -> None:
    assert _host_allowed("verify.example.org", {"verify.example.org"})
    assert not _host_allowed("untrusted.verify.example.org", {"verify.example.org"})


def test_discovered_official_url_is_fetched_before_rebuilt_endpoint() -> None:
    client = RecordingClient("Credential valid Recipient: Alice Example Credential: Python Basics")
    service = VerificationService(registry_with_claims(), client=client)

    result = service.verify(
        ExtractionResult(
            text="Example",
            urls=["https://verify.example.org/c/ABC123"],
        ),
        SubmissionClaims(recipient="Alice Example", credential_title="Python Basics"),
    )

    assert result.status == VerificationStatus.VERIFIED
    assert client.urls == ["https://verify.example.org/c/ABC123"]
