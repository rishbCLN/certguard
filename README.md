# CertGuard

CertGuard is a Python scaffold for human-in-the-loop certificate verification triage. It extracts English text from every image/PDF page, certificate codes, and QR links; checks allowlisted issuer verification services; optionally discovers official verification pages with Brave Search; compares uploads to issuer templates when references exist; records provenance and edit indicators; and emits one explainable risk score plus a complete per-check audit trail.

CertGuard does not claim to detect "AI-generated" certificates from visual style. That classification is not reliable enough for academic decisions. It distinguishes stronger authenticity evidence instead: an issuer record bound to the submitted claims, explicit issuer rejection, claim mismatch, configured-template inconsistency, and inconclusive or unavailable evidence. Generic generated designs should fail to gain a positive result unless their code and identity claims match an authoritative issuer record.

It never emits an academic-credit decision. `review_recommended` is a queueing hint, and `decision` is always `human-review-triage-only`.

## Pipeline

1. **Verification lookup (70% fixed weight):** Native PDF text, English OCR, and QR extraction identify issuer URLs and codes. Optional Brave text search uses only the issuer and certificate ID to discover registry-approved official verification URLs. Only HTTPS endpoints and redirects on each issuer's exact host allowlist are contacted. Search snippets and a bare HTTP 2xx response are never enough. Outcomes distinguish a claim-bound `verified` record, an unbound `record-found`, `claims-mismatch`, explicit `failed-lookup`, operationally unavailable/inconclusive lookup, unknown issuer, and absent code.
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

Enable a signed issuer grammar directory for semantic-structural checks:

```powershell
$env:CERTGUARD_GRAMMAR_HMAC_KEY_FILE = "C:\secure\certguard-grammar.key"
certguard certificate.pdf --grammar-root issuer\grammar --offline
```

Grammar profiles are issuer- and layout-specific. Active profiles require an HMAC-SHA256
signature and a SHA-256-pinned template; invalid profiles fail closed and route applicable
documents to review. The packaged NPTEL, SWAYAM, and Coursera files are intentionally inactive
placeholders and do not change scores.

Build a signed profile after adding an approved template and annotated manifest:

```powershell
$env:CERTGUARD_GRAMMAR_HMAC_KEY_FILE = "C:\secure\certguard-grammar.key"
certguard build-grammar --issuer nptel --variant nptel-v1 `
  --template approved\nptel-v1.png --manifest approved\nptel-v1.manifest.yaml `
  --output issuer\grammar\nptel-v1.yaml
```

The builder copies the immutable template into the grammar directory, records its hash, validates
the profile, and updates `.signatures.json`. Profiles default to shadow-mode evidence; numeric SSDD
points require signed calibration metadata with `scoring_enabled: true`.

Measure local CPU latency without enforcing a hardware-independent threshold:

```powershell
certguard benchmark certificate.pdf --iterations 10 --grammar-root issuer\grammar
```

The benchmark reports SSDD-check and end-to-end minimum, median, p95, and maximum durations. Use
representative deployment hardware and documents before defining a latency acceptance gate.

Enable live issuer lookup by omitting `--offline`. Public verification pages change over time, so their patterns and success/failure markers require monitored maintenance before production use.

Optionally enable Brave text-search discovery. Certificate images, full OCR text, and recipient names are not sent to Brave; the query contains the recognized issuer, certificate ID, and configured official domains. Search results are evidence discovery only and cannot verify a certificate unless the resulting approved official page is fetched and its claims match.

```powershell
$env:BRAVE_SEARCH_API_KEY = "your-api-key"
certguard certificate.pdf --search brave --expected-recipient "Student Name" --expected-credential-title "Python Basics"
```

`--offline` disables both search and issuer-page lookup. API keys are read only from the environment and are not included in reports or audit checks.

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

The preferred workflow uses an external, user-selected managed catalog. Start the local web UI,
then open the displayed loopback URL:

```powershell
certguard templates serve --catalog C:\certguard-data\templates
```

The UI presents the complete click flow: select an issuer, enter a template display name, choose
a blank or anonymized official certificate image or single-page PDF, explicitly confirm that it is
anonymized, rights-cleared, and contains no personal recipient data, and click **Add Template**. The
server defaults to `127.0.0.1:8765` and rejects every non-loopback bind address; it is a local-only
management interface.

The equivalent automation commands emit JSON:

```powershell
certguard templates add --catalog C:\certguard-data\templates `
  --issuer coursera --name "2026 landscape" --file approved\coursera-2026.pdf `
  --confirm-anonymized
certguard templates list --catalog C:\certguard-data\templates
certguard templates remove --catalog C:\certguard-data\templates --id OPAQUE_ID_FROM_LIST
```

Uploads use the normal document size and pixel limits. Multi-page PDFs and blank, low-information,
or featureless references are rejected. Accepted files are normalized to PNG and stored beneath the
catalog root with dimensions, SHA-256, detector availability/keypoint counts, and edge/layout
metrics. The normalized image pixels are stored locally and may themselves contain visible personal
data; CertGuard does not redact or prove that pixels are anonymous. Only blank/anonymized official
references that are rights-cleared and contain no personal recipient data may be uploaded. Metadata,
OCR text, and OCR-derived values are not extracted or persisted by catalog management. Listing
validates artifact presence, symlink safety, and SHA-256 integrity. The normalized reference image
remains the source of truth so runtime `TemplateAnalyzer` descriptors stay compatible with the
installed OpenCV version.

Use the catalog directly during normal analysis; it supplies both issuer template definitions and
the reference-image root:

```powershell
certguard certificate.pdf --template-catalog C:\certguard-data\templates --offline
```

`--template-catalog` cannot be combined with legacy `--templates`. A custom issuer registry can be
used consistently with both catalog management and analysis via `--registry`.

Adding a file establishes only a local known reference. It does not prove that the issuer created
the reference or that any compared certificate is authentic. Catalog access and reference images
must be access-controlled, rights-cleared, reviewed, and versioned according to local policy.
Managed template definitions include the normalized image SHA-256 in the report ruleset fingerprint.
Analysis verifies that digest immediately before decoding the captured bytes; missing, changed, or
symlinked artifacts are unavailable and are not analyzed. Legacy registry definitions without a
`sha256` retain their prior path-based behavior.

Legacy registry-driven template definitions remain available. Add `templates` to an issuer and
provide `--templates <root>`:

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

Region coordinates are normalized `[x, y, width, height]`. Managed catalogs deliberately do not
invent logo or text regions; only explicitly reviewed legacy definitions use these optional regions.
Reference images must be rights-cleared, access-controlled known-genuine samples. Version
registry/template changes in deployment so an appeal can be replayed against the exact rules used
originally.

## Audit And Fairness

Every report contains a SHA-256 source fingerprint, ruleset fingerprint, extracted page text and its source, evidence coverage, timestamp, report version, check status, duration, evidence, signal contribution, and plain-language explanation. The JSONL audit adapter redacts raw extracted text, page text, search queries, and search snippets; it is append-only at process level. Production systems should replace it with immutable storage, access controls, retention policy, encryption, and authenticated reviewer actions.

Operational safeguards required before deployment:

- Calibrate score thresholds on representative, consented samples and compare error rates across document language, issuer, scan quality, device, and accessibility workflows.
- Never treat recapture, missing EXIF, low OCR confidence, an unknown issuer, or service downtime as proof of invalidity.
- Show the source evidence and failed/omitted checks to reviewers and appellants; allow issuer confirmation or replacement documents.
- Cache only what policy permits, redact private OCR from logs, and use request throttling consistent with issuer terms.
- Add issuer-specific response parsers or documented APIs where available. HTML marker matching here is a scaffold, not a production trust anchor.

## Layout

```text
src/certguard/
  document.py       Multi-page OCR, native PDF text, QR and candidate extraction
  search.py         Brave text search and official-result filtering
  registry.py       extensible issuer and endpoint definitions
  verification.py   allowlisted network lookup
  forensics.py      alignment, SSIM, recapture, ELA, copy-move, EXIF
  scoring.py        transparent weighted risk contributions
  pipeline.py       orchestration and per-check audit trail
  templates.py      external managed template catalog and registry integration
  template_server.py dependency-light local template web UI
  batch.py          multi-document discovery, processing, and summary reports
  audit.py          replaceable JSONL audit sink
  cli.py            command-line entry point
```
