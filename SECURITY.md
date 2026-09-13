# Security policy

This policy explains how to report suspected vulnerabilities in Measured RAG.
Architecture, deployment, telemetry, privacy, and product limitations are
documented separately and are linked below.

## Supported versions

Security support applies on a best-effort basis to the latest published release.
Older releases, development snapshots, forks, and modified deployments do not
receive guaranteed security support. Reproduce a report against the latest
published release when practical, and identify any relevant commit or local
modification.

## Report a vulnerability privately

Email `filipebraiman@gmail.com` with the subject **Measured RAG security
report**. Do not open a public GitHub issue for an undisclosed vulnerability.

Include enough information to evaluate the report safely:

- affected version or commit;
- affected component;
- reproduction steps and prerequisites;
- security impact and the boundary crossed; and
- a suggested mitigation, when available.

Sanitize screenshots, logs, documents, and payloads before attaching them. Do
not include secrets or confidential third-party data unless strictly necessary.
If you find a leaked credential, identify its location without reproducing the
credential more than needed to locate and revoke it.

Reports are acknowledged, investigated, and remediated on a best-effort basis.
When a vulnerability is confirmed, the maintainer may coordinate disclosure
with the reporter and affected third parties. No response, remediation, or
disclosure deadline is guaranteed.

## Scope

Appropriate reports include concrete, reproducible vulnerabilities involving:

- unauthorized data disclosure;
- secret or credential exposure;
- unsafe file processing, path traversal, or unintended filesystem access;
- dependency or container vulnerabilities with a demonstrated impact on
  Measured RAG;
- authentication or authorization failures if those controls are introduced;
- cross-session access or isolation failures;
- injection that crosses an actual security boundary;
- hosted telemetry unexpectedly containing document or conversation content
  excluded by the documented hosted schema; or
- denial-of-service weaknesses that can be demonstrated safely and locally.

The following are generally not security vulnerabilities by themselves:

- incorrect or hallucinated model answers;
- low retrieval quality;
- prompt injection that influences only the uploader's own session and does not
  cross a security boundary;
- expected external-provider variability;
- publicly documented demonstration limits; or
- missing enterprise features already disclosed in
  [Limitations](docs/limitations.md).

A report in one of these categories is still relevant if it demonstrates a
separate security-boundary failure.

## Safe testing

Prefer local reproduction with non-sensitive test documents. Do not perform
destructive testing against `demo.measured-rag.com`. Do not conduct denial-of-service
or quota-exhaustion testing, automated high-volume traffic, social engineering,
credential attacks, or persistence attempts, and do not attempt to access another
user’s data.
Stop testing and report privately if unintended data access occurs.

Respect applicable law and the terms of third-party platforms and providers. Do
not include private infrastructure identifiers, account details, secret names,
or non-public defensive thresholds in a report unless they are necessary to
identify the affected resource privately.

## Privacy and third parties

The application's logging boundary is documented in
[Telemetry and diagnostics](docs/telemetry.md). Storage, session, and
sensitive-document limitations are documented in
[Limitations](docs/limitations.md), while the hosted operational boundary is in
[Deployment](docs/deployment.md).

A vulnerability in Google Cloud, Groq, Hugging Face, or another independent
dependency or provider may also need to be reported through that vendor's
security process. Measured RAG cannot make commitments about a third party's
response.
