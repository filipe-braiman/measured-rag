# Benchmark results

This page summarizes the approved Measured RAG v1.0 publication evaluation.
The [curated public copy of the generated publication report](../benchmarking/publication/qasper-publication-v1-28p/evaluation_report.md)
preserves the approved results and provenance; [Benchmarking](benchmarking.md)
documents the methodology and execution workflow.

## Headline results

| Measure | Result |
| --- | --- |
| Evaluation subset | 28 papers · 129 questions |
| Answer-generation coverage | 129/129 (100.00%) |
| QASPER normalized-token Answer F1 | 23.78% |
| Mean semantic correctness | 71.32% |
| Semantic pass rate | 84/129 (65.12%) at a 0.70 threshold |
| Semantic-evaluation coverage | 129/129 (100.00%) |
| Median end-to-end answer latency | 7.216 s across 129 cases |

No composite score is calculated. These are results for the recorded condition,
not a general guarantee for other documents, configurations, models, or future
versions.

See the [complete curated public publication report](../benchmarking/publication/qasper-publication-v1-28p/evaluation_report.md)
for the full metric breakdown, diagnostics, latency distributions, and
provenance.

## Evaluation condition

| Condition | Recorded value |
| --- | --- |
| Dataset | Custom 28-paper, 129-question QASPER-derived subset from the QASPER validation source |
| Document condition | Single document; one paper indexed at a time |
| Conversations | Independent questions with empty history and a distinct conversation ID per question |
| Retrieval mode | Auto |
| Indexing | 600-character chunks with 50-character overlap |
| Retrieval and reranking | Top-N 30; fusion threshold 0.15; 30 rerank candidates; final Top-K 10 |
| Generator | `openai/gpt-oss-120b` |
| Embedding / reranker | `BAAI/bge-small-en-v1.5` / `cross-encoder/ms-marco-MiniLM-L6-v2` |
| Semantic judge | `qwen/qwen3.8-27b` |
| Human references judged | 229 independent reference judgments across 129 cases |
| Evaluator | `qasper-reference-geval-v1.6`; semantic threshold 0.70 |

## What the metrics mean

**QASPER normalized-token Answer F1** measures normalized lexical token overlap
against the best-matching valid QASPER annotation. It does not establish
semantic equivalence or groundedness.

**Mean semantic correctness** is a reference-based LLM-judge score, not
accuracy. Each valid human annotation is judged independently with its supplied
gold evidence, and the maximum reference-level score is retained for the case.
The **semantic pass rate** is the share of cases meeting the recorded 0.70
threshold. LLM judging remains nondeterministic even with fixed inputs and
recorded settings.

**Retrieval relevance** and **groundedness** are runtime diagnostics for the
retrieved context and answer. They are not reference-answer correctness metrics
and are reported separately below.

## Results interpretation

The recorded run completed answer generation and semantic evaluation for every
selected question, with no generation or unresolved judge errors. Its semantic
results are materially higher than its lexical-overlap result. This is
consistent with a known limitation of token F1: explanatory or more complete
answers can overlap only partly with short or highly extractive references. It
does not show that every low-F1 answer is semantically correct.

The case-level record also shows remaining weaknesses. Precise factual details,
numeric results, baseline comparisons, and short boolean answers can still fail
semantic evaluation. Retrieval accounted for most recorded end-to-end latency,
with reranking accounting for most of the retrieval time (median end-to-end:
7.216 s; retrieval: 5.266 s; reranking: 5.158 s).

## Auxiliary diagnostics

These runtime-diagnostic distributions were complete for the recorded run.
`unknown` means checker output was unavailable or failed; it is not a positive
or negative label. No `unknown` results were recorded here.

| Retrieval relevance | Count | Share |
| --- | ---: | ---: |
| High | 102 | 79.07% |
| Medium | 10 | 7.75% |
| Low | 17 | 13.18% |
| Unknown | 0 | 0.00% |

| Groundedness | Count | Share |
| --- | ---: | ---: |
| Grounded | 106 | 82.17% |
| Partially grounded | 7 | 5.43% |
| Not grounded | 16 | 12.40% |
| Unknown | 0 | 0.00% |

## Scope and limitations

- This is a custom QASPER-derived subset, not the full official split or an
  official QASPER leaderboard submission. Direct comparison with published
  QASPER baselines is not justified unless conditions are demonstrably
  equivalent.
- Evidence F1 is not reported because retrieved chunks are not mapped to the
  official QASPER evidence units.
- The evaluation primarily covers English-language research documents.
- External generation and judge outputs are not guaranteed to be identical on
  rerun.
- The results characterize this recorded benchmark condition only; they do not
  establish behavior for every document, domain, configuration, or future model
  version.

## Reproducibility and evidence

| Provenance | Recorded value |
| --- | --- |
| Run ID | `qasper-publication-v1-28p` |
| Profile | `qasper-28p-publication-v1` |
| Judge ID | `qwen3.8-27b-publication-v1` |
| Completion timestamp | `2026-09-09T19:25:21.296882+00:00` |
| Dataset SHA-256 | `ca391c629a520ad9aaba61c5373f8fa54729e10bae7001f451bfc730d1b24853` |
| Configuration SHA-256 | `e611b6f1d6606a6aed01d8624d9437380ab87a865af5a7ddf80db1c5a90e5ebb` |
| Recorded evaluation-workspace Git identifier / dirty state | `45eac4a1bcef060ad24e409b9b69d38aca8d1e5b` / clean |
| Evaluator version | `qasper-reference-geval-v1.6` |

The ignored canonical local run contains the immutable generated report. The
[curated public report](../benchmarking/publication/qasper-publication-v1-28p/evaluation_report.md)
is a reviewed derivative that preserves its approved metrics and recorded
provenance while adding public-distribution disclosures and relative links. Raw
application logs are excluded because they contain document-text previews; their
aggregate coverage remains reported. Public JSONL copies remove gold answers,
evidence excerpts, duplicated reference text, semantic-judge reasons, and all
109 substantive generated answers whose quoted evidence reproduced source text.
Twenty formulaic abstentions remain; scores, statuses, identifiers, and timings
are preserved. The
[redaction manifest](../benchmarking/publication/qasper-publication-v1-28p/redaction_manifest.json)
records exact fields, match lengths, affected identifiers, and before/after
hashes. Those JSONL files support inspection and provenance but cannot serve as
drop-in inputs for reference-based rescoring. The
[telemetry reference](telemetry.md#benchmark-relationship)
explains the runtime-log contract and why those records require safe handling.
The complete canonical run remains locally preserved and unchanged.

For the methodology and reproducible staged workflow, see
[Benchmarking](benchmarking.md). For system boundaries and benchmark flow, see
[Architecture](architecture.md). For production retrieval settings and model
roles, see [Configuration](configuration.md).
