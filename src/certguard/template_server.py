# Embedded HTML and CSS remain local to this dependency-light server.
# ruff: noqa: E501

from __future__ import annotations

import html
import io
import ipaddress
import secrets
import socket
from email.parser import BytesParser
from email.policy import default
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from certguard.registry import IssuerRegistry
from certguard.templates import MAX_TEMPLATE_BYTES, TemplateCatalog, TemplateCatalogError

MAX_REQUEST_BYTES = MAX_TEMPLATE_BYTES + 1024 * 1024


class TemplateHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        catalog: TemplateCatalog,
        registry: IssuerRegistry,
    ) -> None:
        if not _is_loopback_host(address[0]):
            raise ValueError("Template server must bind to a loopback address or localhost")
        resolved_host = _resolve_loopback_host(address[0])
        if ":" in resolved_host:
            self.address_family = socket.AF_INET6
        super().__init__((resolved_host, address[1]), TemplateRequestHandler)
        self.catalog = catalog
        self.registry = registry
        self.csrf_token = secrets.token_urlsafe(32)


class TemplateRequestHandler(BaseHTTPRequestHandler):
    server: TemplateHTTPServer

    def do_GET(self) -> None:  # noqa: N802
        if not self._request_is_local() or urlsplit(self.path).path != "/":
            self._error(HTTPStatus.NOT_FOUND, "Page not found")
            return
        self._render_page()

    def do_POST(self) -> None:  # noqa: N802
        if not self._request_is_local():
            self._error(HTTPStatus.FORBIDDEN, "Request origin is not allowed")
            return
        path = urlsplit(self.path).path
        try:
            fields, upload = self._multipart()
            self._validate_csrf(fields.get("csrf", ""))
            if path == "/add":
                if upload is None:
                    raise TemplateCatalogError("A certificate template file is required")
                filename, payload = upload
                self.server.catalog.add_stream(
                    fields.get("issuer", ""),
                    fields.get("name", ""),
                    io.BytesIO(payload),
                    filename,
                    len(payload),
                    confirm_anonymized=fields.get("confirm_anonymized") == "yes",
                )
                self._redirect("/?message=added")
                return
            if path == "/remove":
                self.server.catalog.remove(fields.get("id", ""))
                self._redirect("/?message=removed")
                return
            self._error(HTTPStatus.NOT_FOUND, "Page not found")
        except TemplateCatalogError as exc:
            self._render_page(str(exc), status=HTTPStatus.BAD_REQUEST)
        except (OSError, ValueError):
            self._error(HTTPStatus.BAD_REQUEST, "Invalid request")

    def _multipart(self) -> tuple[dict[str, str], tuple[str, bytes] | None]:
        content_type = self.headers.get("Content-Type", "")
        if not content_type.casefold().startswith("multipart/form-data;"):
            raise TemplateCatalogError("Form submission must use multipart encoding")
        try:
            content_length = int(self.headers.get("Content-Length", ""))
        except ValueError as exc:
            raise TemplateCatalogError("A valid Content-Length header is required") from exc
        if content_length < 1 or content_length > MAX_REQUEST_BYTES:
            raise TemplateCatalogError(
                f"Request exceeds the {MAX_REQUEST_BYTES}-byte request limit"
            )
        body = self.rfile.read(content_length)
        if len(body) != content_length:
            raise TemplateCatalogError("Request body was incomplete")
        message = BytesParser(policy=default).parsebytes(
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body
        )
        if not message.is_multipart():
            raise TemplateCatalogError("Malformed multipart request")
        fields: dict[str, str] = {}
        upload: tuple[str, bytes] | None = None
        parts = list(message.iter_parts())
        if len(parts) > 12:
            raise TemplateCatalogError("Multipart request contains too many fields")
        for part in parts:
            name = part.get_param("name", header="content-disposition")
            filename = part.get_filename()
            payload = part.get_payload(decode=True) or b""
            if filename is not None and name == "file":
                if upload is not None:
                    raise TemplateCatalogError("Duplicate template file field")
                if len(payload) > MAX_TEMPLATE_BYTES:
                    raise TemplateCatalogError(
                        f"Template upload exceeds the {MAX_TEMPLATE_BYTES}-byte limit"
                    )
                upload = (Path(filename).name, payload)
            elif name in {"csrf", "issuer", "name", "id", "confirm_anonymized"}:
                if name in fields:
                    raise TemplateCatalogError(f"Duplicate form field: {name}")
                if len(payload) > 1024:
                    raise TemplateCatalogError("Form field is too long")
                fields[str(name)] = payload.decode("utf-8", errors="strict")
        return fields, upload

    def _validate_csrf(self, submitted: str) -> None:
        cookies = SimpleCookie(self.headers.get("Cookie", ""))
        cookie = cookies.get("certguard_csrf")
        if (
            cookie is None
            or not secrets.compare_digest(cookie.value, self.server.csrf_token)
            or not secrets.compare_digest(submitted, self.server.csrf_token)
        ):
            raise TemplateCatalogError("CSRF validation failed; reload the page and try again")

    def _request_is_local(self) -> bool:
        host = self.headers.get("Host", "").strip().casefold()
        if not _valid_loopback_authority(host, self.server.server_port):
            return False
        origin = self.headers.get("Origin")
        if not origin:
            return True
        parsed = urlsplit(origin)
        return parsed.scheme == "http" and parsed.netloc.casefold() == host

    def _render_page(self, error: str | None = None, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        try:
            entries = self.server.catalog.list()
        except TemplateCatalogError:
            entries = []
            error = "The catalog could not be read."
            status = HTTPStatus.INTERNAL_SERVER_ERROR
        issuers = sorted(
            self.server.registry.issuers.values(), key=lambda issuer: issuer.display_name.casefold()
        )
        token = html.escape(self.server.csrf_token, quote=True)
        issuer_options = "".join(
            f'<option value="{html.escape(item.issuer_id, quote=True)}">'
            f"{html.escape(item.display_name)}</option>"
            for item in issuers
        )
        rows = "".join(self._entry_card(entry, token) for entry in entries)
        notice = f'<div class="notice error">{html.escape(error)}</div>' if error else ""
        if not rows:
            rows = '<p class="empty">No managed templates yet. Add a known reference above.</p>'
        page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CertGuard Template Catalog</title><style>
:root{{--ink:#17221d;--muted:#607068;--paper:#f4f0e7;--card:#fffdf8;--accent:#176b53;--line:#d7d0c2;--danger:#9e3429}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 system-ui,sans-serif}}
main{{width:min(1080px,92vw);margin:48px auto 72px}} h1{{font:700 clamp(2rem,5vw,4rem)/1.05 Georgia,serif;margin:.15em 0}}
.eyebrow{{color:var(--accent);font-weight:750;letter-spacing:.14em;text-transform:uppercase}} .intro{{max-width:760px;color:var(--muted)}}
.panel{{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:clamp(20px,4vw,38px);box-shadow:0 18px 55px #24372b12;margin:32px 0}}
.steps{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}} label{{display:block;font-weight:700}} label span{{display:block;color:var(--accent);font-size:.76rem;letter-spacing:.1em;text-transform:uppercase;margin-bottom:7px}}
input,select{{width:100%;border:1px solid #b7b0a3;border-radius:9px;padding:12px;background:white;color:var(--ink);font:inherit}} button{{border:0;border-radius:999px;padding:12px 23px;background:var(--accent);color:white;font-weight:750;cursor:pointer}}
.action{{margin-top:22px;display:flex;align-items:center;gap:18px}} .action small{{color:var(--muted)}} .cards{{display:grid;gap:12px}} .item{{display:grid;grid-template-columns:1fr auto;gap:14px;align-items:center;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 20px}}
.item h3{{margin:0 0 3px}} .meta{{color:var(--muted);font-size:.9rem}} .bad{{color:var(--danger);font-weight:700}} .remove{{background:transparent;color:var(--danger);border:1px solid #d9aaa5}} .notice{{padding:13px 16px;border-radius:9px;margin:18px 0}} .error{{background:#fae5e2;color:#7b281f}} .empty{{color:var(--muted);font-style:italic}}
@media(max-width:760px){{main{{margin-top:26px}}.steps{{grid-template-columns:1fr}}.item{{grid-template-columns:1fr}}.remove{{width:100%}}}}
</style></head><body><main><div class="eyebrow">Local reference workspace</div><h1>Template catalog</h1>
<p class="intro">Store a rights-cleared certificate reference as a normalized PNG. References improve visual comparison, but do not prove issuer authenticity.</p>{notice}
<section class="panel"><h2>Add Template</h2><form method="post" action="/add" enctype="multipart/form-data">
<input type="hidden" name="csrf" value="{token}"><div class="steps">
<label><span>1. Issuer</span><select name="issuer" required>{issuer_options}</select></label>
<label><span>2. Template name</span><input name="name" maxlength="120" placeholder="2026 landscape" required></label>
<label><span>3. Certificate template</span><input type="file" name="file" accept="image/*,.pdf,application/pdf" required></label></div>
<label class="confirm"><input type="checkbox" name="confirm_anonymized" value="yes" required> I confirm this template is anonymized, rights-cleared, and contains no personal recipient data.</label>
<div class="action"><button type="submit">Add Template</button><small>Single-page PDF or supported image, up to {MAX_TEMPLATE_BYTES // (1024 * 1024)} MB.</small></div></form></section>
<section><h2>Managed templates</h2><div class="cards">{rows}</div></section></main></body></html>"""
        payload = page.encode("utf-8")
        self.send_response(status)
        self._security_headers()
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header(
            "Set-Cookie",
            f"certguard_csrf={self.server.csrf_token}; Path=/; HttpOnly; SameSite=Strict",
        )
        self.end_headers()
        self.wfile.write(payload)

    @staticmethod
    def _entry_card(entry: dict[str, object], token: str) -> str:
        name = html.escape(str(entry["name"]), quote=True)
        issuer = html.escape(str(entry["issuer_id"]), quote=True)
        integrity = html.escape(str(entry["integrity"]), quote=True)
        integrity_class = "bad" if integrity != "ok" else ""
        identifier = html.escape(str(entry["id"]), quote=True)
        features = entry.get("features", {})
        orb = features.get("ORB", {}) if isinstance(features, dict) else {}
        width = html.escape(str(entry["width"]), quote=True)
        height = html.escape(str(entry["height"]), quote=True)
        keypoints = html.escape(str(orb.get("keypoints", 0)), quote=True)
        return f"""<article class="item"><div><h3>{name}</h3><div class="meta">Issuer: {issuer} · {width}×{height} · ORB keypoints: {keypoints} · <span class="{integrity_class}">integrity: {integrity}</span></div></div>
<form method="post" action="/remove" enctype="multipart/form-data"><input type="hidden" name="csrf" value="{token}"><input type="hidden" name="id" value="{identifier}"><button class="remove" type="submit">Remove</button></form></article>"""

    def _redirect(self, location: str) -> None:
        self.send_response(HTTPStatus.SEE_OTHER)
        self._security_headers()
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _error(self, status: HTTPStatus, message: str) -> None:
        payload = message.encode("utf-8")
        self.send_response(status)
        self._security_headers()
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _security_headers(self) -> None:
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'",
        )
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")

    def log_message(self, format: str, *args: object) -> None:
        super().log_message(format, *args)


def serve_templates(
    catalog_root: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    registry: IssuerRegistry | None = None,
) -> None:
    registry = registry or IssuerRegistry.default()
    catalog = TemplateCatalog(catalog_root, registry)
    server = TemplateHTTPServer((host, port), catalog, registry)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _is_loopback_host(host: str) -> bool:
    if host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _resolve_loopback_host(host: str) -> str:
    if host.casefold() != "localhost":
        return host
    addresses = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    for family, _type, _proto, _canonname, address in addresses:
        candidate = address[0]
        if (
            family in {socket.AF_INET, socket.AF_INET6}
            and ipaddress.ip_address(candidate).is_loopback
        ):
            return candidate
    raise ValueError("localhost did not resolve to a loopback address")


def _valid_loopback_authority(authority: str, port: int) -> bool:
    parsed = urlsplit(f"//{authority}")
    try:
        host = parsed.hostname
        requested_port = parsed.port
    except ValueError:
        return False
    return (
        bool(host)
        and requested_port == port
        and not parsed.username
        and not parsed.password
        and _is_loopback_host(str(host))
    )
