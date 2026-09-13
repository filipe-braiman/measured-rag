# Limitations and operating boundaries

Measured RAG is a production-style reference implementation and evaluated
portfolio system. It makes a document RAG workflow inspectable and configurable;
it is not a claim of universal reliability or a turnkey enterprise document
platform. These boundaries describe the current v1.0 implementation and the
recorded publication evaluation.

## Language scope

v1.0 is optimized and evaluated primarily for English-language documents. The
application does not detect or reject another language, so other-language input
may be indexed and queried. Equivalent retrieval or reranking quality is not
established: the configured embedding model is `BAAI/bge-small-en-v1.5` and the
reranker is `cross-encoder/ms-marco-MiniLM-L6-v2`, while the recorded evaluation
uses an English-language QASPER-derived research-document subset. Treat
non-English results as unvalidated experimentation rather than supported
performance.

## Document parsing and indexing

The interface accepts PDF, DOC, and DOCX uploads. PDFs are loaded as extracted
page text, while Word documents are loaded as elements and grouped around
detected titles or headers. Format acceptance is not a guarantee that every
part of a document will be extracted well. Scanned or image-only PDFs have no
OCR fallback and can yield no usable text. Images, figure contents, and visual
relationships are not interpreted; only extracted text near them can become
available to retrieval.

Complex page layouts, tables, formulas, headers, footers, multi-column reading
order, and Word conversion can affect extracted text and section boundaries.
Legacy DOC handling depends on the installed word-processing stack; the Linux
container includes LibreOffice Writer for that conversion path, but this is not
a guarantee of faithful document rendering. Inspect **Indexed Chunks** after an
upload, especially for layout-heavy material.

Chunks use character-based windows. Changing chunk size or overlap can improve
or degrade retrieval depending on the document and question, and boundaries can
split a semantic unit. The controls support experimentation; they do not select
optimal settings automatically. In single-document mode, removing the current
upload resets the knowledge base, conversation, and inspection outputs; a new
document can then be uploaded and indexed. Multi-document mode appends
non-duplicate inputs, while removing an existing multi-document uploader entry
resets the collection. The [Configuration](configuration.md#indexing-controls)
and [user guide](user-guide.md#reset-and-replacement-behavior) own the exact
settings and reset behavior.

The public cloud mode limits a knowledge base to three documents, 10 MiB per
file, and 20 MiB of accumulated uploaded files. Those are shared-demo
guardrails, not general local limits.

## Retrieval, reranking, and scope

Every retrieval mode runs both dense retrieval and BM25, then applies weighted
fusion and cross-encoder reranking. **Semantic**, **Keyword**, and **Balanced**
are weighted hybrid routes, not pure retrievers. **Auto** selects one of these
routes from simple rewritten-query signals; it is a fixed heuristic, not a
learned confidence model or a guarantee that it selected the best route.

Dense retrieval can miss relevant wording; BM25 can miss semantic paraphrases;
and weighted fusion can change candidate ordering as settings change. The fusion
threshold, Retrieval Top-N, reranker candidate limit, and Final Rerank Top-K
trade recall, context size, and latency. Reranker scores and thresholds are
ranking controls, not calibrated probabilities. Broader candidate sets also add
work and may change score normalization. In the recorded publication condition,
retrieval accounted for most median answer latency and reranking accounted for
most of that retrieval time; this relationship will vary with documents and
settings.

Query rewriting uses recent conversation history to resolve references. It can
therefore alter the retrieval query, and a previous topic can affect a later
question until the conversation is cleared. In multi-document mode, **Only
selected documents** restricts retrieval to the selected indexed entries; at
least one document must be selected before a question can be processed.
Retrieved sources are ranked evidence, not proof that the most relevant passage
was found. See [retrieval controls](configuration.md#retrieval-controls) and
the [multi-document workflow](user-guide.md#multi-document-workflow).

## Answer generation and evidence

The generator (`openai/gpt-oss-120b`) is prompted with the final retrieved
context and is instructed to abstain when that context is insufficient.
Abstention reduces unsupported answering, but it cannot prove that retrieval
found all relevant text or that the answer is correct. Answers can omit,
distort, or infer details, and a plausible-sounding answer can still be wrong
or incomplete.

Displayed sources show the passages supplied to the pipeline. They are useful
for inspection but are not formal scholarly citations or independent citation
verification. For consequential use, inspect the retrieved passages and the
original document rather than relying on the answer or its source list alone.
Generation, rewriting, and checker calls use an external provider; responses
can vary across runs and provider availability or quota can affect behavior.

## Relevance and groundedness diagnostics

**Retrieval relevance** checks whether final context appears useful for the
rewritten question. **Groundedness** checks whether the generated answer appears
supported by that context. Neither is reference-answer correctness. Both are
model-generated operational diagnostics, not manually calibrated accuracy
measures or probability estimates, and they do not repair an answer.

Checker calls can fail or return malformed output. In that situation the
application reports `unknown`, which means unavailable checking rather than a
positive or negative judgment. Favorable labels can disagree with external
evaluation and do not replace human review. The published run records these
diagnostics separately from QASPER normalized-token Answer F1 and semantic
correctness; see [Benchmark results](benchmark-results.md#auxiliary-diagnostics).

## Privacy, logging, and the hosted demo

Questions, recent conversation excerpts, retrieved passages, and generated
answers are assembled into provider requests as required for rewriting,
generation, relevance checking, and groundedness checking. Local use is
therefore not a fully offline workflow. Provider-side handling and retention
depend on external provider policies, not a promise made by this repository.

Local telemetry writes content-bearing JSONL records under `answer_logs/`.
Those records can include queries, rewritten queries, answers, retrieved-text
previews, source metadata, diagnostic explanations, and timing/token fields.
The curated publication bundle deliberately excludes application logs because
their retrieval previews contained substantial independently extracted document
text.

In cloud mode, telemetry emits a constrained operational event to standard
output rather than the local content-bearing JSONL format. It records
categorical labels and numeric timing/configuration fields, not document or
conversation text. This narrower application telemetry does not establish the
contents or retention of platform, dependency, or provider logs.

See [Telemetry and diagnostics](telemetry.md) for the complete application data
contract, event-coverage boundary, and safe-handling guidance.

The shared hosted demonstration has no confidentiality, account-based
isolation, durable knowledge-base persistence, or guaranteed-deletion contract
in this implementation. Do not upload confidential, regulated, personally
identifiable, or otherwise sensitive documents to it. Cloud mode also enforces
a 2,000 character question limit, five admitted questions per session, a queue
size of 20, and single-concurrency admission and heavy-work processing. An admitted
nonempty question can consume the session allowance before downstream document
or scope validation succeeds. Local execution removes these cloud-specific
budgets, but not resource or provider limits.

## Deployment and operational scope

Local desktop execution, the Linux/amd64 container, and the benchmark evaluation
environment use distinct dependency contracts. In particular, the standalone
evaluation environment is intentionally isolated from the application stack
because its DeepEval and Chroma dependencies require incompatible PostHog major
versions. This is an operational reproducibility boundary, not a combined
installation recipe.

The container prefetches CPU embedding and reranking assets and runs with
offline Hugging Face lookup. CPU inference, model loading, document size,
candidate breadth, provider calls, memory pressure, queueing, and cold starts
can all affect response time. The implementation does not provide enterprise
authentication, durable multi-user storage, audit governance, service-level
objectives, or autoscaling guarantees. See [Deployment](deployment.md) for the
hosted architecture and operational posture, [Development](development.md) for
local setup, and [Configuration](configuration.md) for the supported runtime
contract.

## Evaluation scope

The approved publication evaluation uses a custom 28-paper, 129-question
QASPER-derived subset under a single-document condition, with one paper indexed
at a time and independent empty-history questions. Its recorded run completed
generation and semantic evaluation for all selected cases. QASPER
normalized-token Answer F1 measures lexical overlap; semantic correctness
judging measures reference-based answer correctness. They measure different
properties and are not combined.

This is not an official full-split QASPER leaderboard result, and its results
apply only to the recorded configuration, models, and dataset. The semantic
judge is model-based and nondeterministic. Evidence F1 is not reported because
retrieved chunks were not mapped to official QASPER evidence units. Read the
[curated benchmark results](benchmark-results.md), [benchmarking guide](benchmarking.md),
and [curated public publication report](../benchmarking/publication/qasper-publication-v1-28p/evaluation_report.md)
for the complete condition, metrics, and provenance.

The curated public JSONL artifacts omit reference/evidence text and
source-overlapping substantive generated answers. They support inspection and
provenance but are not complete inputs for rerunning reference-based scoring;
the local reconstruction workflow is documented in [Benchmarking](benchmarking.md).

## Appropriate and inappropriate use

| Appropriate v1.0 uses | Inappropriate reliance |
| --- | --- |
| Learning, portfolio review, and inspection of an end-to-end RAG system | Unreviewed medical, legal, financial, safety-critical, or other high-stakes decisions |
| Prototyping on non-sensitive English documents | Confidential documents in the shared public demo |
| Comparing retrieval settings and inspecting ranked evidence | Treating checker labels as correctness guarantees or displayed sources as formal citation verification |
| Demonstrating evaluation and telemetry workflows | Assuming the recorded benchmark generalizes to arbitrary domains or languages |

## Potential future directions

Potential directions suggested by these boundaries include multilingual retrieval
evaluation, OCR and layout-aware parsing, stronger table and figure handling,
human-calibrated diagnostics, broader multi-document evaluation, authentication
and durable deployment controls, and further retrieval/configuration
experiments. These are directions for future evaluation and design work, not
committed features or release dates.
