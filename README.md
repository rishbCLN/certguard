# CertGuard

CertGuard is a Python scaffold for human-in-the-loop certificate-fraud triage. It extracts text, certificate codes, and QR links; checks allowlisted issuer verification services; compares uploads to issuer templates when references exist; records provenance and edit indicators; and emits one explainable risk score plus a complete per-check audit trail.

CertGuard does not claim to detect "AI-generated" certificates from visual style. That classification is not reliable enough for academic decisions. It distinguishes stronger authenticity evidence instead: an issuer record bound to the submitted claims, explicit issuer rejection, claim mismatch, configured-template inconsistency, and inconclusive or unavailable evidence. Generic generated designs should fail to gain a positive result unless their code and identity claims match an authoritative issuer record.

It never emits an academic-credit decision. `review_recommended` is a queueing hint, and `decision` is always `human-review-triage-only`.

## Pipeline

1. **Verification lookup (70% fixed weight):** OCR and QR extraction identify issuer URLs and codes. Only HTTPS endpoints and redirects on each issuer's exact host allowlist are contacted. A bare HTTP 2xx response is never enough. Outcomes distinguish a claim-bound `verified` record, an unbound `record-found`, `claims-mismatch`, explicit `failed-lookup`, operationally unavailable/inconclusive lookup, unknown issuer, and absent code.
2. **Template/layout forensics (20% base weight):** ORB keypoints and RANSAC homography align a document with configured genuine references. SSIM compares logo and text regions, while edge SSIM compares layout. This check is skipped rather than penalized when no reference exists.
3. **Recapture/provenance:** Frequency peaks and sharpness provide a deliberately conservative capture-method estimate. Screen or print recapture does not add risk by itself. ELA, copy-move, and EXIF observations are reviewer evidence and are not scored until calibrated on representative data.
4. **Content plausibility (3% base weight):** OCR wording is compared with issuer phrases and kept supplementary.

Weights are fixed rather than renormalized. Missing OCR, templates, metadata, network access, or unsupported forensic checks do not silently become adverse evidence. `evidence_coverage` and `review_reasons` expose what still requires manual work.

## Setup

Python 3.11+ and the Tesseract executable are required for OCR. QR decoding and the rest of the image pipeline work without Tesseract.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
```

Analyze a submission offline first:

```powershell
certguard certificate.pdf --offline --output reports\submission.json --audit-log audit\checks.jsonl
```

Enable live issuer lookup by omitting `--offline`. Public verification pages change over time, so their patterns and success/failure markers require monitored maintenance before production use.

Analyze several documents or all supported files in a directory as one fault-tolerant batch:

```powershell
certguard submissions --recursive --offline --output reports\batch.json
certguard alice.pdf bob.png --expected-credential-title "Python Basics" --output reports\batch.json
```

The batch report includes completed/failed counts, the number requiring review, verification-status totals, and each document's complete analysis report. A failed or unreadable document is recorded without stopping the remaining analyses. `--submission-id` and `--expected-recipient` remain single-document options because batch identity binding requires a roster or manifest.

Bind each document to its student with a CSV manifest so batch results become claim-bound:

```powershell
certguard submissions --manifest class.csv --output reports\batch.json
```

```csv
filename,student_id,expected_recipient,expected_credential_title
rahul.pdf,CS21001,Rahul Kumar,Python Basics
priya.png,CS21002,Priya Sharma,Python Basics
```

Filename matching is case-insensitive and uses the file name only. Manifest rows override the batch-level `--expected-credential-title`, each report item carries the student ID, and manifest entries without a matching document are listed under `unmatched_manifest_entries`.

For a full issuer-record match, pass claims from the trusted academic submission record rather than trusting OCR alone:

```powershell
certguard certificate.pdf --expected-recipient "Student Name" --expected-credential-title "Python Basics"
```

An issuer page without parsable claims is reported as `record-found`, not `verified`. A copied credential belonging to a different recipient or course becomes `claims-mismatch` when the issuer adapter exposes those fields.

## Adding An Issuer

Copy `src/certguard/data/issuers.json`, add an entry, and pass it with `--registry`. Keep endpoints on HTTPS and list every permitted host. URL patterns are full-match-oriented regular expressions; endpoint templates interpolate only `{certificate_id}`. A minimal entry is:

```json
{
  "id": "example",
  "display_name": "Example Learning",
  "aliases": ["Example Learning"],
  "verification_url_patterns": ["^https://verify\\.example\\.org/c/[A-Za-z0-9_-]+$"],
  "allowed_hosts": ["verify.example.org"],
  "endpoints": [{
    "url_template": "https://verify.example.org/c/{certificate_id}",
    "allowed_hosts": ["verify.example.org"],
    "success_markers": ["Credential issued", "Credential valid"],
    "failure_markers": ["Not found"]
  }],
  "certificate_id_patterns": ["/c/([A-Za-z0-9_-]+)"],
  "language_phrases": ["has successfully completed"]
}
```

## Adding Templates

Template support is registry-driven. Add `templates` to an issuer and provide `--templates <root>`:

```json
"templates": [{
  "id": "2026-landscape",
  "image": "example/2026-landscape.png",
  "regions": {
    "logo": [0.05, 0.05, 0.25, 0.18],
    "text": [0.15, 0.30, 0.70, 0.45]
  }
}]
```

Region coordinates are normalized `[x, y, width, height]`. Reference images must be rights-cleared, access-controlled known-genuine samples. Version registry/template changes in deployment so an appeal can be replayed against the exact rules used originally.

## Audit And Fairness

Every report contains a SHA-256 source fingerprint, ruleset fingerprint, evidence coverage, timestamp, report version, check status, duration, evidence, signal contribution, and plain-language explanation. The JSONL adapter is append-only at process level; production systems should replace it with immutable storage, access controls, retention policy, encryption, and authenticated reviewer actions.

Operational safeguards required before deployment:

- Calibrate score thresholds on representative, consented samples and compare error rates across document language, issuer, scan quality, device, and accessibility workflows.
- Never treat recapture, missing EXIF, low OCR confidence, an unknown issuer, or service downtime as proof of fraud.
- Show the source evidence and failed/omitted checks to reviewers and appellants; allow issuer confirmation or replacement documents.
- Cache only what policy permits, redact sensitive OCR from logs, and use request throttling consistent with issuer terms.
- Add issuer-specific response parsers or documented APIs where available. HTML marker matching here is a scaffold, not a production trust anchor.

## Layout

```text
src/certguard/
  document.py       OCR, PDF rendering, QR and candidate extraction
  registry.py       extensible issuer and endpoint definitions
  verification.py   allowlisted network lookup
  forensics.py      alignment, SSIM, recapture, ELA, copy-move, EXIF
  scoring.py        transparent weighted risk contributions
  pipeline.py       orchestration and per-check audit trail
  batch.py          multi-document discovery, processing, and summary reports
  audit.py          replaceable JSONL audit sink
  cli.py            command-line entry point
```
