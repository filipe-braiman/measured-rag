# Deployment and hosted operation

Measured RAG is available at [https://demo.measured-rag.com](https://demo.measured-rag.com), a constrained public demonstration of a production-style document RAG system. It runs as a containerized Gradio application on Google Cloud Run. The hosted demo runs the repository's core application workflow with additional guardrails for shared public use. It is a portfolio demonstration, not an enterprise service or a commitment to production-grade availability, confidentiality, or support.

Cloud Run runs the application container, supplies the listening port, and supports the stable custom domain independently of individual revisions. That makes it a suitable host for the demonstration while keeping the application image and its runtime contract explicit.

## Public deployment

The custom domain is the stable user-facing and documentation address. It represents the service rather than a specific revision, so visitors do not need to follow revision URLs when the service is updated.

The public demo is intentionally constrained. It is useful for trying document ingestion, retrieval, source inspection, and grounded-answer diagnostics, but it is a shared service with finite compute and provider capacity. It should not be used as a repository for durable knowledge bases, as a confidential-document service, or as evidence of a particular availability or latency commitment.

## Runtime architecture

The application image targets the verified Python 3.11 Linux/amd64 contract. It packages the Gradio interface and the production application dependencies, but not the separate benchmark evaluation dependencies. The documented clean-install workflow keeps local application, container application, and evaluation environments isolated because their validated dependency contracts conflict. [Development](development.md) and [Benchmarking](benchmarking.md) own the corresponding setup and evaluation guidance.

Embedding and reranking execute locally on CPU within the application container. Answer generation, history-aware query rewriting for questions that pass document and scope checks, and runtime relevance and groundedness checkers use the configured external provider. The image therefore packages the local model assets needed for its retrieval path, but the application is not fully offline: provider calls remain necessary for those external inference roles.

During image construction, the embedding and reranker assets are prefetched. Runtime Hugging Face model lookup is offline, so an absent packaged asset fails instead of triggering a visitor-time model download. The image runs its application process without root privileges and includes LibreOffice support for the existing legacy-DOC ingestion path. Writable temporary storage is container-local, and the build context excludes local secrets and generated artifacts.

The provider credential, `GROQ_API_KEY`, should be supplied through managed secret configuration and must not be built into the image. `DEPLOYMENT_MODE=cloud` enables hosted behavior, while Cloud Run supplies `PORT` for the application listener. Additional cache and offline settings belong to the image's container contract rather than to public application configuration. See [Configuration](configuration.md#local-and-hosted-application-settings) for the documented application settings and [Architecture](architecture.md) for the component boundaries.

## Hosted-mode behavior

Hosted mode applies shared-demo guardrails that differ from local use. Uploads may be PDF, DOC, or DOCX, with a maximum of three documents, 10 MiB per file, and 20 MiB accumulated upload size. Questions are limited to 2,000 characters, and each session can admit five questions. These limits are capacity guardrails for a shared demo; they are not local-mode limits or security isolation.

Expensive indexing and answer-generation work is deliberately bounded and serialized within each application process to protect shared demo capacity. External-provider quotas are another operational constraint and may temporarily affect generation or checker availability. The user-facing flow, including single-document replacement, multi-document behavior, sources, diagnostics, and resets, is documented in the [user guide](user-guide.md#hosted-demo-behavior).

## Sessions, uploads, and privacy

Upload processing and indexing occur inside the running application container. Knowledge bases and conversations are session-scoped application state. Container replacement, instance recycling, or scale-to-zero removes in-memory state; scaling out creates independent instances with their own in-memory state and model copies. These lifecycle properties do not establish durable isolation, immediate deletion, or a retention guarantee.

Reset behavior helps a visitor manage the visible application state, but it is not a guaranteed deletion contract for temporary storage, platform systems, or external services. Do not upload confidential, regulated, or otherwise sensitive documents to the shared demo.

Local execution can produce content-bearing JSONL telemetry for inspection and evaluation. Hosted mode emits a reduced operational event stream that excludes document text, questions, answers, filenames, source previews, and conversation identifiers. This application policy does not control Cloud Run platform logs, dependency logs, or provider-side handling; their access and retention are separate concerns. The fuller privacy and operational boundary is in [Limitations](limitations.md#privacy-logging-and-the-hosted-demo).
The [telemetry reference](telemetry.md#local-and-hosted-telemetry) defines the
application event contracts and their observability boundary.

## Operational posture

The recommended operational posture uses Secret Manager, a dedicated least-privilege service identity, and no secrets in source, images, build arguments, or logs. A pinned container image is built from the repository's production dependency contract, stored in a managed image registry, and deployed to Cloud Run using an immutable version tag or digest. Cloud Build and Artifact Registry are suitable supporting services for that lifecycle, without making this guide a service-administration procedure.

Resource allocation, request handling, scaling bounds, and other infrastructure settings are deliberately managed outside the repository. They must be selected from measured startup, memory, latency, concurrency, quota, and cost behavior. When configured to scale to zero, the service can experience cold starts. Scaling and container replacement also mean that the service does not provide durable application state.

Updates use immutable images, candidate validation, and a known-good revision retained until the stable custom domain is verified. The high-level controls for the demo combine Cloud Run scaling bounds, application upload and question limits, bounded expensive work, provider quotas, logging controls, and billing monitoring. Together they reduce exposure; they do not guarantee prevention of abuse or unexpected cost.

Published deployments should be built from an immutable public release commit
or tag so that the documented source corresponds to the running application.
Benchmark Git identifiers record evaluation-workspace provenance independently
of deployed release identity.

## Stable access and verification

Use [https://demo.measured-rag.com](https://demo.measured-rag.com) when linking to or visiting the demo. Before a candidate update is treated as stable, public-facing smoke checks cover:

- Interface loading and static assets.
- A non-sensitive upload and indexing flow.
- A representative question and answer.
- Sources and diagnostics.
- Reset behavior.
- Custom-domain TLS.

These checks validate the hosted experience at a point in time; they do not convert the demo into a guarantee of service availability, document confidentiality, or model correctness.

## Related documentation

- [Architecture](architecture.md) explains the application and benchmark boundaries.
- [User guide](user-guide.md) covers local and hosted interaction flows.
- [Configuration](configuration.md) is the canonical reference for controls, model roles, and hosted guardrails.
- [Development](development.md) covers local setup, tests, and contribution-oriented workflows.
- [Limitations](limitations.md) details validated scope, privacy boundaries, and operating tradeoffs.
