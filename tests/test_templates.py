import hashlib
import json
import os
import threading
from http.client import HTTPConnection
from pathlib import Path

import cv2
import fitz
import numpy as np
import pytest

import certguard.cli as cli
from certguard.forensics import TemplateAnalyzer
from certguard.registry import IssuerRegistry
from certguard.template_server import MAX_REQUEST_BYTES, TemplateHTTPServer
from certguard.templates import TemplateCatalog, TemplateCatalogError


def reference_image(path: Path) -> Path:
    image = np.full((480, 720, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (18, 18), (701, 461), (20, 70, 110), 8)
    cv2.circle(image, (110, 105), 55, (30, 120, 180), -1)
    cv2.putText(
        image,
        "CERTIFICATE OF COMPLETION",
        (170, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    for index in range(6):
        y = 180 + index * 36
        cv2.line(image, (130, y), (590 - index * 17, y), (40, 40, 40), 2)
    rng = np.random.default_rng(42)
    for x, y in rng.integers([40, 130], [680, 440], size=(80, 2)):
        cv2.circle(image, (int(x), int(y)), 2, (50, 80, 120), -1)
    assert cv2.imwrite(str(path), image)
    return path


def pdf_from_image(path: Path, image_path: Path, pages: int = 1) -> Path:
    document = fitz.open()
    for _ in range(pages):
        page = document.new_page(width=720, height=480)
        page.insert_image(page.rect, filename=str(image_path))
    document.save(path)
    document.close()
    return path


@pytest.fixture
def registry() -> IssuerRegistry:
    return IssuerRegistry.default()


def test_image_add_list_registry_integration_and_remove(tmp_path, registry) -> None:
    source = reference_image(tmp_path / "reference.jpg")
    catalog = TemplateCatalog(tmp_path / "catalog", registry)

    added = catalog.add("coursera", "2026 Landscape", source, confirm_anonymized=True)
    entries = catalog.list()

    assert entries == [added]
    assert added["integrity"] == "ok"
    assert added["artifact"].startswith("artifacts/")
    assert added["artifact"].endswith(".png")
    assert added["source_type"] == "image"
    assert added["features"]["ORB"]["keypoints"] >= 12
    assert "text" not in json.dumps(added).casefold()
    artifact = catalog.root / str(added["artifact"])
    assert artifact.read_bytes().startswith(b"\x89PNG")
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == added["sha256"]

    integrated = catalog.apply_to_registry(registry)
    definition = integrated.get("coursera")
    assert definition is not None
    assert definition.templates == (
        {"id": added["id"], "image": added["artifact"], "sha256": added["sha256"]},
    )
    image = cv2.imread(str(artifact))
    result = TemplateAnalyzer(catalog.root).analyze(image, definition)
    assert result.available
    assert result.template_id == added["id"]
    assert "logo" not in result.explanation
    assert "text" not in result.explanation

    removed = catalog.remove(str(added["id"]))
    assert removed["artifact_cleanup"] == "deleted"
    assert catalog.list() == []
    assert not artifact.exists()


def test_add_requires_explicit_anonymized_confirmation(tmp_path, registry) -> None:
    source = reference_image(tmp_path / "reference.png")
    catalog = TemplateCatalog(tmp_path / "catalog", registry)

    with pytest.raises(TemplateCatalogError, match="Explicit confirmation"):
        catalog.add("coursera", "Reference", source)
    with pytest.raises(TemplateCatalogError, match="Explicit confirmation"):
        catalog.add_stream(
            "coursera",
            "Reference",
            source.open("rb"),
            source.name,
            source.stat().st_size,
        )

    assert not catalog.catalog_path.exists()


def test_single_page_pdf_is_accepted_and_multipage_is_rejected(tmp_path, registry) -> None:
    image = reference_image(tmp_path / "reference.png")
    catalog = TemplateCatalog(tmp_path / "catalog", registry)
    single = pdf_from_image(tmp_path / "single.pdf", image)
    multiple = pdf_from_image(tmp_path / "multiple.pdf", image, pages=2)

    assert catalog.add("edx", "Single PDF", single, confirm_anonymized=True)["source_type"] == "pdf"
    with pytest.raises(TemplateCatalogError, match="exactly one page"):
        catalog.add("nptel", "Multiple PDF", multiple, confirm_anonymized=True)
    assert [entry["name"] for entry in catalog.list()] == ["Single PDF"]


@pytest.mark.parametrize("name", ["", "   ", "\x00unsafe", "---"])
def test_unsafe_or_empty_names_are_rejected(tmp_path, registry, name) -> None:
    source = reference_image(tmp_path / "reference.png")
    with pytest.raises(TemplateCatalogError, match="safe nonempty"):
        TemplateCatalog(tmp_path / "catalog", registry).add(
            "coursera", name, source, confirm_anonymized=True
        )


def test_unknown_issuer_duplicate_name_and_low_feature_image_are_rejected(
    tmp_path, registry
) -> None:
    source = reference_image(tmp_path / "reference.png")
    blank = tmp_path / "blank.png"
    assert cv2.imwrite(str(blank), np.full((400, 600, 3), 255, dtype=np.uint8))
    catalog = TemplateCatalog(tmp_path / "catalog", registry)

    with pytest.raises(TemplateCatalogError, match="Unknown issuer"):
        catalog.add("unknown", "Reference", source, confirm_anonymized=True)
    catalog.add("coursera", "Reference", source, confirm_anonymized=True)
    with pytest.raises(TemplateCatalogError, match="already exists"):
        catalog.add("edx", "reference", source, confirm_anonymized=True)
    with pytest.raises(TemplateCatalogError, match="too little visual information"):
        catalog.add("edx", "Blank", blank, confirm_anonymized=True)


def test_list_flags_missing_and_corrupt_artifacts(tmp_path, registry) -> None:
    source = reference_image(tmp_path / "reference.png")
    catalog = TemplateCatalog(tmp_path / "catalog", registry)
    first = catalog.add("coursera", "First", source, confirm_anonymized=True)
    second_source = reference_image(tmp_path / "second.png")
    image = cv2.imread(str(second_source))
    cv2.line(image, (0, 0), (719, 479), (0, 0, 255), 4)
    assert cv2.imwrite(str(second_source), image)
    second = catalog.add("edx", "Second", second_source, confirm_anonymized=True)

    (catalog.root / str(first["artifact"])).unlink()
    (catalog.root / str(second["artifact"])).write_bytes(b"not a png")

    assert [entry["integrity"] for entry in catalog.list()] == ["missing", "corrupt"]
    integrated = catalog.apply_to_registry(registry)
    assert integrated.get("coursera").templates == ()
    assert integrated.get("edx").templates == ()


def test_mutated_managed_artifact_is_not_analyzed_and_changes_no_fingerprint(
    tmp_path, registry, monkeypatch
) -> None:
    source = reference_image(tmp_path / "reference.png")
    catalog = TemplateCatalog(tmp_path / "catalog", registry)
    added = catalog.add("coursera", "Reference", source, confirm_anonymized=True)
    integrated = catalog.apply_to_registry(registry)
    definition = integrated.get("coursera")
    artifact = catalog.root / str(added["artifact"])
    image = cv2.imread(str(artifact))
    analyzed = []
    monkeypatch.setattr(
        TemplateAnalyzer,
        "_compare",
        staticmethod(lambda *args: analyzed.append(args) or None),
    )

    artifact.write_bytes(b"altered")
    result = TemplateAnalyzer(catalog.root).analyze(image, definition)

    assert not result.available
    assert analyzed == []
    assert catalog.list()[0]["integrity"] == "corrupt"


def test_managed_template_digest_changes_ruleset_fingerprint(tmp_path, registry) -> None:
    from certguard.pipeline import _ruleset_fingerprint

    source = reference_image(tmp_path / "reference.png")
    catalog = TemplateCatalog(tmp_path / "catalog", registry)
    catalog.add("coursera", "Reference", source, confirm_anonymized=True)
    integrated = catalog.apply_to_registry(registry)
    issuer = integrated.issuers["coursera"]
    changed_definition = dict(issuer.templates[0])
    changed_definition["sha256"] = "f" * 64
    integrated.issuers["coursera"] = type(issuer)(
        **{
            field: (changed_definition,) if field == "templates" else getattr(issuer, field)
            for field in issuer.__dataclass_fields__
        }
    )

    original = catalog.apply_to_registry(registry)
    assert _ruleset_fingerprint(original) != _ruleset_fingerprint(integrated)


def test_add_rolls_back_new_artifact_when_catalog_write_fails(
    tmp_path, registry, monkeypatch
) -> None:
    source = reference_image(tmp_path / "reference.png")
    catalog = TemplateCatalog(tmp_path / "catalog", registry)
    monkeypatch.setattr(
        catalog, "_write_catalog", lambda _catalog: (_ for _ in ()).throw(OSError())
    )

    with pytest.raises(OSError):
        catalog.add("coursera", "Reference", source, confirm_anonymized=True)

    assert not list(catalog.artifact_root.glob("*.png"))
    assert not catalog.catalog_path.exists()


def test_remove_preserves_shared_artifact_until_last_reference(tmp_path, registry) -> None:
    source = reference_image(tmp_path / "reference.png")
    catalog = TemplateCatalog(tmp_path / "catalog", registry)
    first = catalog.add("coursera", "First", source, confirm_anonymized=True)
    second = catalog.add("edx", "Second", source, confirm_anonymized=True)
    artifact = catalog.root / str(first["artifact"])
    assert first["artifact"] == second["artifact"]

    assert catalog.remove(str(first["id"]))["artifact_cleanup"] == "not-needed"
    assert artifact.exists()
    assert catalog.remove(str(second["id"]))["artifact_cleanup"] == "deleted"
    assert not artifact.exists()


def test_catalog_rejects_traversal_artifact_without_deleting_outside_file(
    tmp_path, registry
) -> None:
    root = tmp_path / "catalog"
    root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    entry = {
        "id": "opaque-id",
        "issuer_id": "coursera",
        "name": "Unsafe",
        "artifact": "../outside.png",
        "sha256": hashlib.sha256(b"outside").hexdigest(),
        "width": 100,
        "height": 100,
        "source_type": "image",
        "created_at": "2026-01-01T00:00:00Z",
        "features": {},
        "edge_density": 0.1,
        "grayscale_stddev": 10.0,
    }
    (root / "catalog.json").write_text(
        json.dumps({"schema_version": 1, "templates": [entry]}), encoding="utf-8"
    )
    catalog = TemplateCatalog(root, registry)

    with pytest.raises(TemplateCatalogError, match="artifact path"):
        catalog.list()
    with pytest.raises(TemplateCatalogError, match="artifact path"):
        catalog.remove("opaque-id")
    assert outside.read_bytes() == b"outside"


def test_catalog_rejects_malformed_entry_schema(tmp_path, registry) -> None:
    root = tmp_path / "catalog"
    root.mkdir()
    digest = "a" * 64
    valid = {
        "id": "opaque-id",
        "issuer_id": "coursera",
        "name": "Reference",
        "artifact": f"artifacts/{digest}.png",
        "sha256": digest,
        "width": 720,
        "height": 480,
        "source_type": "image",
        "created_at": "2026-01-01T00:00:00Z",
        "features": {
            name: {
                "available": name != "SURF",
                "keypoints": 1 if name != "SURF" else 0,
                "descriptors": 1 if name != "SURF" else 0,
            }
            for name in ("ORB", "SIFT", "SURF")
        },
        "edge_density": 0.1,
        "grayscale_stddev": 10.0,
    }
    invalid_entries = [
        {**valid, "id": "../unsafe"},
        {**valid, "issuer_id": "missing"},
        {**valid, "width": "720"},
        {**valid, "source_type": "svg"},
        {**valid, "created_at": "1900-01-01T00:00:00Z"},
        {**valid, "sha256": "A" * 64},
        {**valid, "artifact": f"artifacts/{'b' * 64}.png"},
        {**valid, "edge_density": float("nan")},
        {**valid, "features": {"ORB": {"available": True, "keypoints": -1, "descriptors": 1}}},
    ]
    for invalid in invalid_entries:
        (root / "catalog.json").write_text(
            json.dumps({"schema_version": 1, "templates": [invalid]}), encoding="utf-8"
        )
        with pytest.raises(TemplateCatalogError):
            TemplateCatalog(root, registry).list()


def test_managed_artifact_symlink_is_not_listed_analyzed_or_followed_on_remove(
    tmp_path, registry
) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlinks are unavailable")
    source = reference_image(tmp_path / "reference.png")
    catalog = TemplateCatalog(tmp_path / "catalog", registry)
    added = catalog.add("coursera", "Reference", source, confirm_anonymized=True)
    artifact = catalog.root / str(added["artifact"])
    outside = tmp_path / "outside.png"
    outside.write_bytes(artifact.read_bytes())
    artifact.unlink()
    try:
        artifact.symlink_to(outside)
    except OSError:
        pytest.skip("creating symlinks is not permitted")

    assert catalog.list()[0]["integrity"] == "symlink"
    definition = registry.get("coursera")
    definition = type(definition)(
        **{
            field: (
                ({"id": added["id"], "image": added["artifact"], "sha256": added["sha256"]},)
                if field == "templates"
                else getattr(definition, field)
            )
            for field in definition.__dataclass_fields__
        }
    )
    assert (
        not TemplateAnalyzer(catalog.root).analyze(cv2.imread(str(outside)), definition).available
    )
    assert catalog.remove(str(added["id"]))["artifact_cleanup"] == "deleted-symlink"
    assert outside.exists()


def test_atomic_create_only_never_overwrites_existing_artifact(tmp_path) -> None:
    path = tmp_path / "artifact.png"
    path.write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        TemplateCatalog._atomic_write_bytes(path, b"replacement", create_only=True)

    assert path.read_bytes() == b"existing"


def test_html_entry_values_are_escaped() -> None:
    from certguard.template_server import TemplateRequestHandler

    rendered = TemplateRequestHandler._entry_card(
        {
            "id": '"><script>id</script>',
            "issuer_id": "<issuer>",
            "name": "<name>",
            "width": "<width>",
            "height": "<height>",
            "features": {"ORB": {"keypoints": "<count>"}},
            "integrity": "<integrity>",
        },
        "token",
    )

    assert "<script>" not in rendered
    for value in ("issuer", "name", "width", "height", "count", "integrity"):
        assert f"&lt;{value}&gt;" in rendered


def multipart(fields: dict[str, str], upload: tuple[str, bytes] | None = None) -> tuple[bytes, str]:
    boundary = "certguard-test-boundary"
    body = bytearray()
    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(value.encode())
        body.extend(b"\r\n")
    if upload:
        filename, payload = upload
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode()
        )
        body.extend(b"Content-Type: image/png\r\n\r\n")
        body.extend(payload)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def duplicate_multipart(parts: list[tuple[str, str, bytes]]) -> tuple[bytes, str]:
    boundary = "certguard-duplicate-boundary"
    body = bytearray()
    for name, filename, payload in parts:
        body.extend(f"--{boundary}\r\n".encode())
        disposition = f'Content-Disposition: form-data; name="{name}"'
        if filename:
            disposition += f'; filename="{filename}"'
        body.extend(f"{disposition}\r\n\r\n".encode())
        body.extend(payload)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    return bytes(body), f"multipart/form-data; boundary={boundary}"


@pytest.fixture
def template_server(tmp_path, registry):
    catalog = TemplateCatalog(tmp_path / "catalog", registry)
    server = TemplateHTTPServer(("127.0.0.1", 0), catalog, registry)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, catalog
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()


def request(server, method: str, path: str, body: bytes | None = None, headers=None):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    payload = response.read()
    result = response.status, dict(response.getheaders()), payload
    connection.close()
    return result


def test_web_ui_add_remove_csrf_origin_and_security_headers(template_server, tmp_path) -> None:
    server, catalog = template_server
    status, headers, page = request(server, "GET", "/")
    cookie = headers["Set-Cookie"].split(";", 1)[0]
    token = cookie.split("=", 1)[1]
    assert status == 200
    assert b"Add Template" in page
    assert b"Certificate template" in page
    assert headers["X-Frame-Options"] == "DENY"
    assert "default-src 'none'" in headers["Content-Security-Policy"]

    source = reference_image(tmp_path / "web.png")
    body, content_type = multipart(
        {
            "csrf": token,
            "issuer": "coursera",
            "name": "Web Reference",
            "confirm_anonymized": "yes",
        },
        ("web.png", source.read_bytes()),
    )
    host = f"127.0.0.1:{server.server_port}"
    status, _, _ = request(
        server,
        "POST",
        "/add",
        body,
        {
            "Content-Type": content_type,
            "Cookie": cookie,
            "Origin": f"http://{host}",
            "Host": host,
        },
    )
    assert status == 303
    entry = catalog.list()[0]

    remove_body, remove_type = multipart({"csrf": token, "id": str(entry["id"])})
    status, _, _ = request(
        server,
        "POST",
        "/remove",
        remove_body,
        {"Content-Type": remove_type, "Cookie": cookie, "Host": host},
    )
    assert status == 303
    assert catalog.list() == []

    bad_body, bad_type = multipart({"csrf": "wrong", "id": "../outside"})
    status, _, response = request(
        server,
        "POST",
        "/remove",
        bad_body,
        {"Content-Type": bad_type, "Cookie": cookie, "Host": host},
    )
    assert status == 400
    assert b"CSRF validation failed" in response

    status, _, _ = request(
        server,
        "POST",
        "/remove",
        remove_body,
        {
            "Content-Type": remove_type,
            "Cookie": cookie,
            "Origin": "http://evil.example",
            "Host": host,
        },
    )
    assert status == 403


def test_web_ui_rejects_invalid_host_and_oversized_request(template_server) -> None:
    server, _ = template_server
    status, _, _ = request(server, "GET", "/", headers={"Host": "evil.example"})
    assert status == 404
    status, _, response = request(
        server,
        "POST",
        "/add",
        b"x",
        {
            "Content-Type": "multipart/form-data; boundary=x",
            "Content-Length": str(MAX_REQUEST_BYTES + 1),
            "Host": f"127.0.0.1:{server.server_port}",
        },
    )
    assert status == 400
    assert b"request limit" in response


def test_web_ui_requires_anonymized_confirmation(template_server, tmp_path) -> None:
    server, catalog = template_server
    _, headers, _ = request(server, "GET", "/")
    cookie = headers["Set-Cookie"].split(";", 1)[0]
    token = cookie.split("=", 1)[1]
    source = reference_image(tmp_path / "unconfirmed.png")
    body, content_type = multipart(
        {"csrf": token, "issuer": "coursera", "name": "Unconfirmed"},
        (source.name, source.read_bytes()),
    )

    status, _, response = request(
        server,
        "POST",
        "/add",
        body,
        {
            "Content-Type": content_type,
            "Cookie": cookie,
            "Host": f"localhost:{server.server_port}",
        },
    )

    assert status == 400
    assert b"Explicit confirmation" in response
    assert catalog.list() == []


@pytest.mark.parametrize(
    "duplicate_parts",
    [
        [("csrf", "", b"token"), ("csrf", "", b"token")],
        [("file", "one.png", b"one"), ("file", "two.png", b"two")],
    ],
)
def test_web_ui_rejects_duplicate_security_or_file_fields(template_server, duplicate_parts) -> None:
    server, _ = template_server
    body, content_type = duplicate_multipart(duplicate_parts)
    status, _, response = request(
        server,
        "POST",
        "/add",
        body,
        {
            "Content-Type": content_type,
            "Host": f"localhost:{server.server_port}",
        },
    )

    assert status == 400
    assert b"Duplicate" in response


def test_template_cli_dispatch_and_analysis_option(tmp_path, registry, monkeypatch, capsys) -> None:
    source = reference_image(tmp_path / "reference.png")
    catalog_root = tmp_path / "catalog"
    assert (
        cli.templates_main(
            [
                "add",
                "--catalog",
                str(catalog_root),
                "--issuer",
                "coursera",
                "--name",
                "CLI Reference",
                "--file",
                str(source),
                "--confirm-anonymized",
            ]
        )
        == 0
    )
    added = json.loads(capsys.readouterr().out)
    assert cli.templates_main(["list", "--catalog", str(catalog_root)]) == 0
    assert json.loads(capsys.readouterr().out)[0]["id"] == added["id"]
    assert cli.templates_main(["remove", "--catalog", str(catalog_root), "--id", added["id"]]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == added["id"]

    class Pipeline:
        kwargs = None

        def __init__(self, **kwargs):
            Pipeline.kwargs = kwargs

        def analyze(self, _source, **_kwargs):
            return type("Result", (), {"to_dict": lambda self: {"ok": True}})()

    monkeypatch.setattr(cli, "CertGuardPipeline", Pipeline)
    monkeypatch.setattr(
        "sys.argv",
        ["certguard", str(source), "--template-catalog", str(catalog_root)],
    )
    assert cli.main() == 0
    assert Pipeline.kwargs["template_catalog"] == catalog_root


def test_remote_template_server_requires_explicit_override(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["certguard", "templates", "serve", "--catalog", "x", "--host", "0.0.0.0"],
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_template_cli_add_requires_confirmation(tmp_path) -> None:
    source = reference_image(tmp_path / "reference.png")

    with pytest.raises(SystemExit) as error:
        cli.templates_main(
            [
                "add",
                "--catalog",
                str(tmp_path / "catalog"),
                "--issuer",
                "coursera",
                "--name",
                "Reference",
                "--file",
                str(source),
            ]
        )

    assert error.value.code == 2


def test_template_server_constructor_rejects_non_loopback(tmp_path, registry) -> None:
    catalog = TemplateCatalog(tmp_path / "catalog", registry)

    with pytest.raises(ValueError, match="loopback"):
        TemplateHTTPServer(("0.0.0.0", 0), catalog, registry)
