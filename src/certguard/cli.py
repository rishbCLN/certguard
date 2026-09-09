from __future__ import annotations

import argparse
import json
from pathlib import Path

from certguard.audit import JsonlAuditSink
from certguard.batch import BatchProcessor, load_manifest
from certguard.pipeline import CertGuardPipeline
from certguard.registry import IssuerRegistry


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
    parser.add_argument("--templates", type=Path, help="Reference template directory")
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
        "--offline", action="store_true", help="Extract candidates but do not contact issuer services"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Include supported documents in nested input directories",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="CSV binding filenames to student_id, expected_recipient, and expected_credential_title",
    )
    return parser


def main() -> int:
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

    registry = IssuerRegistry.from_file(args.registry) if args.registry else IssuerRegistry.default()
    audit_sink = JsonlAuditSink(args.audit_log) if args.audit_log else None
    pipeline = CertGuardPipeline(
        registry=registry,
        template_root=args.templates,
        audit_sink=audit_sink,
        network_enabled=not args.offline,
    )
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
