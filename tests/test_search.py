from certguard.models import ExtractionResult, SearchResult
from certguard.registry import IssuerRegistry
from certguard.search import discover_official_pages


class StubSearchClient:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.query = ""

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        self.query = query
        return self.results[:max_results]


def test_search_uses_id_and_official_domain_without_recipient_name() -> None:
    client = StubSearchClient(
        [
            SearchResult(
                title="Verify certificate",
                url="https://www.coderank.com/certificates/abc123def",
            )
        ]
    )
    extraction = ExtractionResult(
        text="CodeRank awarded to Alice Private Certificate ID: abc123def",
        certificate_ids=["abc123def"],
    )

    evidence = discover_official_pages(extraction, IssuerRegistry.default(), client, enabled=True)

    assert '"abc123def"' in client.query
    assert "site:coderank.com" in client.query
    assert "Alice Private" not in client.query
    assert evidence.accepted_urls == ["https://www.coderank.com/certificates/abc123def"]


def test_search_rejects_lookalike_and_non_verification_pages() -> None:
    client = StubSearchClient(
        [
            SearchResult(
                title="Unapproved",
                url="https://coderank.com.untrusted.example/certificates/abc123def",
            ),
            SearchResult(title="Home", url="https://www.coderank.com/"),
        ]
    )
    extraction = ExtractionResult(
        text="CodeRank Certificate ID: abc123def",
        certificate_ids=["abc123def"],
    )

    evidence = discover_official_pages(extraction, IssuerRegistry.default(), client, enabled=True)

    assert evidence.accepted_urls == []
    assert not any(result.accepted for result in evidence.results)


def test_search_failure_remains_non_adverse() -> None:
    class FailingSearchClient:
        def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
            raise TimeoutError("search unavailable")

    evidence = discover_official_pages(
        ExtractionResult(
            text="CodeRank Certificate ID: abc123def",
            certificate_ids=["abc123def"],
        ),
        IssuerRegistry.default(),
        FailingSearchClient(),
        enabled=True,
    )

    assert evidence.accepted_urls == []
    assert evidence.error == "TimeoutError: search unavailable"
