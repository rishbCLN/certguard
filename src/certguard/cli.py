from __future__ import annotations

import argparse
import ipaddress
import json
import os
import sys
import time
from pathlib import Path

from certguard.audit import JsonlAuditSink
from certguard.batch import BatchProcessor, load_manifest
from certguard.forensics import OnnxForgeryModel
from certguard.pipeline import CertGuardPipeline
from certguard.registry import IssuerRegistry
from certguard.search import BraveSearchClient
from certguard.ssdd import GrammarProfileError, build_grammar_profile
from certguard.template_server import serve_templates
from certguard.templates import TemplateCatalog, TemplateCatalogError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="certguard", description="Triage a submitted certificate for human review."
    )
    parser.add_argument(
        "documents",
        type=Path,
        nargs="+",
        help="Certificate image/PDF or directory; multiple values create a batch report",
    )
    parser.add_argument("--registry", type=Path, help="Custom issuer registry JSON")
    parser.add_argument("--grammar-root", type=Path, help="Signed issuer grammar profile directory")
    parser.add_argument("--templates", type=Path, help="Reference template directory")
    parser.add_argument(
        "--template-catalog",
        type=Path,
        help="Managed template catalog root (supplies definitions and reference images)",
    )
    parser.add_argument(
        "--forgery-model",
        type=Path,
        help="Trained ONNX binary classifier producing genuine/forgery logits or a forgery score",
    )
    parser.add_argument("--output", type=Path, help="Write the report as JSON")
    parser.add_argument("--audit-log", type=Path, help="Append the report to a JSONL audit log")
    parser.add_argument("--submission-id", help="Stable ID from the host submission system")
    parser.add_argument(
        "--expected-recipient",
        help="Recipient name from the academic submission record for issuer-record binding",
    )
    parser.add_argument(
        "--expected-credential-title",
        help="Expected course, assessment, or event title for issuer-record binding",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Extract candidates but do not contact issuer services",
    )
    parser.add_argument(
        "--search",
        choices=("brave",),
        help="Discover official verification pages with a text search API",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Include supported documents in nested input directories",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "CSV binding filenames to student_id, expected_recipient, and expected_credential_title"
        ),
    )
    return parser


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "build-grammar":
        return build_grammar_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        return benchmark_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "templates":
        return templates_main(sys.argv[2:])
    parser = build_parser()
    args = parser.parse_args()
    batch_requested = len(args.documents) > 1 or any(path.is_dir() for path in args.documents)
    if batch_requested and args.submission_id:
        parser.error("--submission-id can only be used with one document")
    if batch_requested and args.expected_recipient:
        parser.error("--expected-recipient can only be used with one document")
    if args.recursive and not any(path.is_dir() for path in args.documents):
        parser.error("--recursive requires at least one directory")
    if args.manifest and not batch_requested:
        parser.error("--manifest can only be used with batch documents")
    if args.offline and args.search:
        parser.error("--search cannot be used with --offline")
    if args.templates and args.template_catalog:
        parser.error("--templates cannot be combined with --template-catalog")

    search_client = None
    if args.search == "brave":
        api_key = os.environ.get("BRAVE_SEARCH_API_KEY", "")
        if not api_key:
            parser.error("--search brave requires BRAVE_SEARCH_API_KEY")
        search_client = BraveSearchClient(api_key)

    registry = (
        IssuerRegistry.from_file(args.registry) if args.registry else IssuerRegistry.default()
    )
    audit_sink = JsonlAuditSink(args.audit_log) if args.audit_log else None
    try:
        forgery_model = OnnxForgeryModel(args.forgery_model) if args.forgery_model else None
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    try:
        pipeline_options = dict(
            registry=registry,
            template_root=args.templates,
            audit_sink=audit_sink,
            network_enabled=not args.offline,
            search_client=search_client,
            search_enabled=bool(args.search),
            forgery_model=forgery_model,
            grammar_root=args.grammar_root,
        )
        if args.template_catalog is not None:
            pipeline_options["template_catalog"] = args.template_catalog
        pipeline = CertGuardPipeline(**pipeline_options)
    except (GrammarProfileError, TemplateCatalogError, OSError, ValueError) as exc:
        parser.error(str(exc))
    if batch_requested:
        manifest = load_manifest(args.manifest) if args.manifest else None
        result = BatchProcessor(pipeline).analyze(
            args.documents,
            recursive=args.recursive,
            expected_credential_title=args.expected_credential_title,
            manifest=manifest,
        )
    else:
        result = pipeline.analyze(
            args.documents[0],
            submission_id=args.submission_id,
            expected_recipient=args.expected_recipient,
            expected_credential_title=args.expected_credential_title,
        )
    rendered = json.dumps(result.to_dict(), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered)
    return 1 if batch_requested and getattr(result, "failed", 0) else 0


def build_grammar_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="certguard build-grammar")
    parser.add_argument("--issuer", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    key_path = os.environ.get("CERTGUARD_GRAMMAR_HMAC_KEY_FILE")
    if not key_path:
        parser.error("build-grammar requires CERTGUARD_GRAMMAR_HMAC_KEY_FILE")
    try:
        build_grammar_profile(
            args.issuer,
            args.variant,
            args.template,
            args.manifest,
            args.output,
            key_file=Path(key_path),
        )
    except (GrammarProfileError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(args.output)
    return 0


def benchmark_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="certguard benchmark")
    parser.add_argument("document", type=Path)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--grammar-root", type=Path)
    args = parser.parse_args(argv)
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")

    try:
        pipeline = CertGuardPipeline(grammar_root=args.grammar_root, network_enabled=False)
    except GrammarProfileError as exc:
        parser.error(str(exc))
    durations: list[float] = []
    ssdd_durations: list[int] = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        report = pipeline.analyze(args.document)
        durations.append((time.perf_counter() - started) * 1000)
        ssdd_check = next(
            check for check in report.checks if check.name == "semantic_structural_dissonance"
        )
        ssdd_durations.append(ssdd_check.duration_ms)

    output = {
        "document": str(args.document.resolve()),
        "iterations": args.iterations,
        "end_to_end_ms": _latency_summary(durations),
        "ssdd_ms": _latency_summary(ssdd_durations),
        "threshold_enforced": False,
    }
    print(json.dumps(output, indent=2))
    return 0


def templates_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="certguard templates")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add = subparsers.add_parser("add", help="Normalize and add a template")
    add.add_argument("--catalog", type=Path, required=True)
    add.add_argument("--registry", type=Path)
    add.add_argument("--issuer", required=True)
    add.add_argument("--name", required=True)
    add.add_argument("--file", type=Path, required=True)
    add.add_argument(
        "--confirm-anonymized",
        action="store_true",
        required=True,
        help=(
            "Confirm the template is anonymized, rights-cleared, and contains no personal "
            "recipient data"
        ),
    )
    list_parser = subparsers.add_parser("list", help="List templates and integrity status")
    list_parser.add_argument("--catalog", type=Path, required=True)
    list_parser.add_argument("--registry", type=Path)
    remove = subparsers.add_parser("remove", help="Remove a template by opaque ID")
    remove.add_argument("--catalog", type=Path, required=True)
    remove.add_argument("--registry", type=Path)
    remove.add_argument("--id", required=True)
    serve = subparsers.add_parser("serve", help="Run the local template web UI")
    serve.add_argument("--catalog", type=Path, required=True)
    serve.add_argument("--registry", type=Path)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    registry = (
        IssuerRegistry.from_file(args.registry) if args.registry else IssuerRegistry.default()
    )
    catalog = TemplateCatalog(args.catalog, registry)
    try:
        if args.command == "add":
            output: object = catalog.add(
                args.issuer,
                args.name,
                args.file,
                confirm_anonymized=args.confirm_anonymized,
            )
        elif args.command == "list":
            output = catalog.list()
        elif args.command == "remove":
            output = catalog.remove(args.id)
        else:
            if not 0 <= args.port <= 65535:
                parser.error("--port must be between 0 and 65535")
            if not _is_loopback_host(args.host):
                parser.error("--host must be a loopback address or localhost")
            print(f"CertGuard template UI listening on http://{args.host}:{args.port}")
            serve_templates(args.catalog, host=args.host, port=args.port, registry=registry)
            return 0
    except (OSError, TemplateCatalogError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(output, indent=2))
    return 0


def _is_loopback_host(host: str) -> bool:
    if host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _latency_summary(values: list[float | int]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    percentile_index = min(len(ordered) - 1, max(0, round(0.95 * len(ordered) - 1)))
    return {
        "min": round(ordered[0], 3),
        "median": round(ordered[len(ordered) // 2], 3),
        "p95": round(ordered[percentile_index], 3),
        "max": round(ordered[-1], 3),
    }


if __name__ == "__main__":
    raise SystemExit(main())
