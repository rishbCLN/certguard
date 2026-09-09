from certguard.registry import IssuerRegistry


def test_default_registry_identifies_issuer_from_verification_url() -> None:
    registry = IssuerRegistry.default()

    matches = registry.identify(
        "Certificate awarded to Student",
        ["https://www.hackerrank.com/certificates/abc123def"],
    )

    assert [issuer.issuer_id for issuer in matches] == ["hackerrank"]


def test_registry_rejects_non_https_endpoint() -> None:
    data = {
        "issuers": [
            {
                "id": "unsafe",
                "display_name": "Unsafe",
                "endpoints": [
                    {
                        "url_template": "http://example.com/{certificate_id}",
                        "allowed_hosts": ["example.com"],
                    }
                ],
            }
        ]
    }

    try:
        IssuerRegistry.from_data(data)
    except ValueError as error:
        assert "HTTPS" in str(error)
    else:
        raise AssertionError("Expected an unsafe endpoint to be rejected")
