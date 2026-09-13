# Telemetry and diagnostics

Measured RAG uses application-level structured telemetry to inspect retrieval and generation behavior, compare configurations, review diagnostics, and analyze latency. It can also support benchmark provenance and failure analysis. It is not a complete production observability platform.

This page covers runtime telemetry only. It is distinct from the user-visible **Indexed Chunks**, **Retrieved Sources**, and **Debug Last Answer** inspection surfaces; from retrieval-relevance and groundedness diagnostics; and from QASPER scoring, semantic judging, and generated benchmark artifacts. Relevance and groundedness are operational signals, not measures of answer accuracy.

## Event lifecycle

[`chat()`](../chat/service.py) is the sole production call site for `log_evaluation_entry()`. It calls the logger after it has generated an answer, run retrieval-relevance and groundedness checks, and appended the diagnostic suffix. The value passed to the logger is that post-suffix answer, so local `generation.answer` includes the suffix. A normally returned answer is therefore recorded even when either diagnostic is unavailable or partially available.

The logger is not an all-path audit trail. Empty questions, a missing knowledge base, or an empty document scope return before the logging call. A failed or unusable generation response also prevents the call from being reached. Diagnostic futures, however, are resolved into `unknown` values on provider, malformed-result, or validation failures, allowing the completed answer path to retain the unavailable diagnostic state.

## Local and hosted telemetry

`DEPLOYMENT_MODE` selects the serializer. Invalid modes are rejected during logger import.

| Concern | Local mode | Hosted mode |
| --- | --- | --- |
| Destination | UTF-8 JSON Lines appended to repository-relative `answer_logs/rag_eval_log.jsonl`, resolved from the logger's repository root | One JSON event written to standard output |
| Content-bearing fields | Questions, answers, document names and metadata, retrieved-text previews, diagnostic explanations, and identifiers | Excludes document text, questions, answers, filenames, source previews, diagnostic explanations, and identifiers |
| Identifiers | Conversation, query, document, and chunk identifiers are recorded | No conversation, query, document, or chunk identifiers are serialized |
| Intended use | Development and evaluation inspection | Constrained operational monitoring |
| Retention control | The repository creates the local directory and appends records; it does not implement rotation or retention | The application controls only its stdout event shape, not downstream platform retention |

Local writes use append mode, one JSON object per line; creating the logger in local mode creates the `answer_logs` directory. A local serialization or file-write exception is printed as an evaluation-logging error and does not re-raise through the answer path. In hosted mode, the serializer independently constructs an allowlisted event rather than filtering a detailed record. If that construction or emission fails, it attempts a generic `{"event": "rag_telemetry_error"}` event and never falls back to content-bearing local logging. A broken standard output is also non-fatal to the request.

## Field reference

The local record has these top-level fields: `timestamp`, `conversation_id`, `query_id`, `kb_mode`, `active_doc_names`, `query`, `retrieval`, `generation`, and `llm_metrics`. The following groups describe the serialized contract without reproducing a complete log entry.

| Group | Exact fields and meaning |
| --- | --- |
| Request context | `timestamp`, `conversation_id`, `query_id`, `kb_mode`, and `active_doc_names`; `query.original` and `query.rewritten` hold the original and rewritten questions. |
| Retrieval settings and counts | `retrieval.route`, `fusion_alpha`, `retrieval_top_n`, `rerank_candidate_n`, `rerank_top_k`, `fusion_threshold`, `num_fusion_survivors`, `num_reranker_candidates`, and `num_final_sources`. |
| Retrieval evidence | `raw_retrieval` and `final_sources` are arrays. Their entries include rank, document/chunk identifiers, filename, page, type metadata, distance, similarity, fusion score, and `content_preview`; final sources also include `rerank_score`. `rerank_margin` is present when at least two reranker scores exist. |
| Diagnostics | `retrieval_relevance`, `relevance_score`, `relevance_explanation`, and `retrieval_eval_latency`; `generation.groundedness_label`, `generation.groundedness_explanation`, and `generation.groundedness_eval_latency`. |
| Answer and provider use | `generation.answer`; `llm_metrics.total_tokens` and `llm_metrics.latency_seconds`. |
| Retrieval timings | `retrieval.latency.total_seconds`, `dense_seconds`, `bm25_seconds`, `fusion_seconds`, and `rerank_seconds`. |

The hosted `rag_request_completed` event contains only `event`, `timestamp`, `kb_mode`, `retrieval_route`, `fusion_alpha`, `retrieval_top_n`, `rerank_candidate_n`, `rerank_top_k`, `fusion_threshold`, `num_fusion_survivors`, `num_reranker_candidates`, `num_final_sources`, `retrieval_relevance`, `relevance_score`, `groundedness_label`, `retrieval_latency`, `dense_latency`, `bm25_latency`, `fusion_latency`, `rerank_latency`, `retrieval_eval_latency`, `groundedness_eval_latency`, `llm_latency`, and `total_tokens`. Numeric values are accepted only when finite and labels are restricted to their known categories; invalid labels become `unknown`. The hosted event does not contain the local record's nested objects.

## Interpreting timings and diagnostics

`retrieval.latency.total_seconds` spans dense retrieval, BM25 retrieval, fusion, reranking, and intervening retrieval work. The component timings are measured around those individual calls, so they need not sum exactly to the total. `llm_metrics.latency_seconds` measures the answer-generation provider call, and `total_tokens` is present only when that response reports total usage; neither covers query rewriting, retrieval-relevance checking, or groundedness checking. Runtime telemetry exposes these component timings but does not record a complete end-to-end request duration. The benchmark harness's per-case `elapsed_seconds` is a separate benchmark measurement.

The two post-generation checks run concurrently. `retrieval_eval_latency` and `groundedness_eval_latency` measure their respective checker work and can overlap, so neither should be added to the other as elapsed request time. Their model-generated labels and explanations support review, not reference-answer correctness. Fusion and reranker scores rank available candidates; they are not calibrated probabilities.

`unknown` means the relevant checker was unavailable, failed, or returned an invalid result. It is neither a successful nor negative label. With no final context, the checker implementations instead return `low` retrieval relevance and `not_grounded` without provider checking; those labels are different from diagnostic unavailability.

## Benchmark relationship

Benchmark answer generation calls the production `chat()` service in the application environment, so successful benchmark answer paths can produce the same local telemetry. The harness records the starting byte offset of the global JSONL file, then selects only later records whose `conversation_id` belongs to that run. Before writing the run's `application_logs.jsonl`, it deep-copies each match and only processes diagnostic explanations: `unknown` labels receive normalized failure causes, while other explanations have known secret, provider-URL, email, and path patterns replaced. It does not redact the remaining content-bearing fields, so the resulting file is not automatically suitable for public distribution; checker availability is accounted for separately.

Application telemetry is not QASPER Answer F1, semantic judging, or a benchmark manifest/score-file contract. Those artifacts have their own provenance and coverage rules; see [Benchmarking](benchmarking.md#pipeline-stages) and [Benchmark results](benchmark-results.md). The canonical publication workflow retained application logs privately for analysis. Its curated public bundle deliberately excludes `application_logs.jsonl` because retrieved-content previews contained substantial extracted research-paper text; see the [publication bundle notes](../benchmarking/publication/qasper-publication-v1-28p/README.md).

## Privacy and safe handling

Treat local telemetry as sensitive. It can contain original and rewritten questions, generated answers, filenames and document metadata, document/chunk identifiers, retrieved-text previews, and diagnostic explanations, as well as conversation and query identifiers. `.gitignore` exclusion is not a retention or deletion policy. Review and sanitize records before sharing them.

Do not upload confidential, regulated, personally identifiable, or otherwise sensitive documents to the shared demo. The reduced hosted application event does not control Cloud Run platform logs, dependency logs, or provider-side handling and retention. The [limitations guide](limitations.md#privacy-logging-and-the-hosted-demo) owns the broader privacy boundary; the curated publication bundle illustrates a safer public-artifact boundary rather than a guarantee of anonymization.

## Operational limitations

The current implementation is not OpenTelemetry instrumentation, distributed tracing, a centralized metrics backend, a dashboard or alerting system, a durable audit-log service, an automatic log-rotation system, a repository-enforced retention policy, or a security/compliance monitoring product. It records completed application answer paths according to the two serializers above; it does not establish universal request coverage or external-platform observability.

## Related documentation

- [Architecture](architecture.md#8-telemetry-and-inspection) places telemetry and inspection in the request path.
- [Configuration](configuration.md#telemetry-configuration) owns mode selection and related settings.
- [Deployment](deployment.md#sessions-uploads-and-privacy) describes hosted logging boundaries and operational posture.
- [Benchmarking](benchmarking.md) owns evaluation workflow and artifact production.
- [Limitations](limitations.md#privacy-logging-and-the-hosted-demo) explains privacy and non-guarantees.
- [Development](development.md) covers local setup and offline test commands.
