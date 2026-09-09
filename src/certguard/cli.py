from __future__ import annotations

import argparse
import json
from pathlib import Path

from certguard.audit import JsonlAuditSink
from certguard.pipeline import CertGuardPipeline
from certguard.registry import IssuerRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="certguard", description="Triage a submitted certificate for human review."
    )
    parser.add_argument("document", type=Path, help="Certificate image or PDF")
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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    registry = IssuerRegistry.from_file(args.registry) if args.registry else IssuerRegistry.default()
    audit_sink = JsonlAuditSink(args.audit_log) if args.audit_log else None
    pipeline = CertGuardPipeline(
        registry=registry,
        template_root=args.templates,
        audit_sink=audit_sink,
        network_enabled=not args.offline,
    )
    report = pipeline.analyze(
        args.document,
        submission_id=args.submission_id,
        expected_recipient=args.expected_recipient,
        expected_credential_title=args.expected_credential_title,
    )
    rendered = json.dumps(report.to_dict(), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
