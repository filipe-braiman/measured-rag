# Benchmarking

Measured RAG includes an evaluation workflow that runs the production document
RAG path on a fixed QASPER-derived subset, records provenance and operational
diagnostics, and evaluates answers through two separate correctness methods.
This guide covers how to reproduce, inspect, resume, and interpret that
workflow. The system flow is shown in the
[benchmark workflow diagram](architecture.md#benchmark-architecture).

## Scope and evaluation philosophy

The benchmark indexes one paper at a time and asks independent questions with
empty conversation history. It uses the production ingestion and
[`chat()`](../chat/service.py) services; it does not import `app.py` or automate
the Gradio interface.

The following records answer different questions and must remain separate:

| Record or measure | What it evaluates | What it does not establish |
| --- | --- | --- |
| Production retrieval and generation | The configured end-to-end RAG path | Reference-answer correctness by itself |
| Retrieval relevance | Whether the runtime checker considers selected context useful | Reference-answer correctness |
| Groundedness | Whether the runtime checker considers the answer supported by selected context | Reference-answer correctness |
| QASPER normalized-token Answer F1 | Lexical overlap with QASPER annotations | Semantic equivalence or groundedness |
| Semantic correctness | Reference-based LLM judging against each valid annotation | Deterministic correctness or an official leaderboard result |
| Generated report | Validated presentation of recorded artifacts | A replacement for source artifacts |

Checker labels are operational diagnostics, not reference-answer correctness
metrics. Semantic correctness is not accuracy. Token F1 and semantic
correctness measure different properties and are never combined into a composite
score. The custom subset is not an official complete-split QASPER leaderboard
submission. Evidence F1 is not reported because retrieved chunks are not mapped
to QASPER evidence units.

## Pipeline overview

Each benchmark stage is available as a separate command-line module. When
`run_pipeline` is used, it launches those stages as separate subprocesses using
the same Python interpreter; subprocess separation does not isolate Python
dependencies. It records each stage command, status, timing, return code,
validated artifact paths, and failures in the pipeline manifest. A failed or
interrupted stage leaves a recoverable record; later pending stages are marked
not run.

```text
Validated profile and fixed dataset
  -> production answer generation
  -> deterministic QASPER Answer F1
  -> semantic correctness judging
  -> validated Markdown report
```

Generation calls external inference services and loads the local embedding and
reranking models. Semantic correctness judging calls its configured Groq judge.
After the required artifacts exist, deterministic Answer F1 and report
generation do not call a provider or load application models. The report stage
validates its inputs before rendering.

## Evaluation environment

Measured RAG uses two isolated CPython 3.11 environments. The application
environment from [Development](development.md#create-the-application-environment)
runs production answer generation. A separate evaluation environment runs
dataset construction, deterministic scoring, semantic judging, and report
generation. The split is required because the validated Chroma and DeepEval
stacks require incompatible major versions of PostHog.

Create the evaluation environment from the repository root.

Windows PowerShell:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r benchmarking/requirements-eval.txt
```

Linux and macOS:

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r benchmarking/requirements-eval.txt
```

Conda alternative:

```bash
conda create -n measured-rag-eval python=3.11
conda activate measured-rag-eval
python -m pip install --upgrade pip
python -m pip install -r benchmarking/requirements-eval.txt
```

`GROQ_API_KEY` is required for provider-backed semantic judging. Dataset
construction additionally makes Hugging Face and arXiv network requests. Do not
combine this requirement file with `requirements.txt`, bypass dependency
resolution, or ignore resolver failures.

## Dataset construction

The active publication condition is a **28-paper, 129-question QASPER-derived
evaluation subset**. It is built from the QASPER validation parquet at revision
`d57bf25d29d3f089384b087ab24fb9bd077a9e09`, not from the complete QASPER
validation or test split.

[`build_eval_data.py`](../benchmarking/build_eval_data.py) loads that validation
source, retains papers with at least three questions and nonempty section names,
then applies a fixed shuffle seed of `1001` and selects 28 papers. It writes the
question/annotation records to the profile's `qa_data_path` and downloads each
selected paper from arXiv to `pdf_directory`, matching the PDF name to the paper
ID after removing an `arXiv:` prefix. The builder preserves the source QASPER
answer bundles; answer normalization occurs in evaluation, not dataset creation.

Run the builder only when intentionally reconstructing the dataset material. It
makes network requests and can overwrite the selected QA JSON while downloading
missing PDFs. Download availability does not grant redistribution rights for the
papers.

```bash
python -m benchmarking.build_eval_data
```

The command deterministically recreates the selected QASPER question data and
downloads the 28 source papers into the profile's local PDF directory. Research
paper PDFs and the content-bearing generated question file are local benchmark
inputs and are excluded from Git; they are not part of the public source
distribution. The tracked
[`qasper_selection_seed1001_size28.json`](../benchmarking/data/qasper_selection_seed1001_size28.json)
records the selection parameters, arXiv identifiers, and canonical download
URLs. Obtain each paper from its original source and respect its own terms;
QASPER attribution does not grant permission to redistribute the papers.

Run the builder from the repository root in the evaluation environment.

## Evaluation configuration

[`evaluation_settings.py`](../benchmarking/evaluation_settings.py) is the
editable pre-run profile. `load_evaluation_config()` validates its
`EvaluationConfig` before generation, scoring, or orchestration uses it. A
recorded run embeds the resolved configuration and its digest, so changing this
file affects a new run only; it does not retroactively change an existing run.

Benchmark values are separate from interactive UI defaults. See
[Configuration](configuration.md) for shared production retrieval semantics and
runtime model settings.

| Profile field | Purpose |
| --- | --- |
| `profile_name` | Human-readable identity for the pre-run profile |
| `qa_data_path`, `pdf_directory` | Repository-relative selected data and paper locations |
| `retrieval_mode` | Production retrieval route for generation |
| `chunk_size`, `chunk_overlap` | Per-paper indexing configuration |
| `retrieval_top_n`, `fusion_threshold` | Candidate breadth and fusion filtering |
| `rerank_candidate_count`, `rerank_top_k` | Reranker input cap and final context cap |
| `judge_model`, `judge_threshold`, `judge_reasoning_effort` | Semantic judge identity and pass rule |
| `judge_max_retries` and judge pacing/backoff fields | Judge retry and request pacing policy |
| `generation_case_interval_seconds` | Minimum start-to-start generation-case interval |

The accepted benchmark and runtime modes are `auto`, `semantic`, `keyword`, and
`balanced`. The validated profile value passes unchanged into production
generation. Semantic and Keyword are weighted hybrid routes: both dense and
BM25 retrieval still execute. They are not dense-only or BM25-only modes.

The active publication profile records `auto`; its recorded manifest, rather
than mutable settings, is the authority for that run. Resume validates profile
and dataset provenance before it reuses work.

## Run identity and provenance

A profile name, run ID, and judge ID identify different things:

| Identifier or record | Meaning |
| --- | --- |
| Profile name | Editable pre-run configuration identity |
| Generation run ID | Directory identity for one answer-generation condition |
| Judge ID | Directory identity for a semantic-judge run under that generation run |
| Dataset digest | SHA-256 of the selected QA data file |
| Configuration digest | SHA-256 of the resolved profile |
| Git identifier and dirty state | Evaluation-workspace provenance captured when available; it is not necessarily a public source revision |
| Evaluator and pipeline versions | Versioned evaluation and orchestration contracts |

Run IDs and judge IDs must start with an alphanumeric character and otherwise
contain only letters, digits, dots, underscores, or hyphens. Use a new run ID
for a materially different generation condition and a new judge ID for a
materially different judge condition. Do not reuse a directory to overwrite an
unrelated evaluation.

Generation manifests also record model IDs, request settings, selected-case
limits, application-log start offset, case-selection provenance, and counts.
Judge manifests record the judge request, evaluator version, evidence policy,
and attempt accounting. Pipeline manifests and summaries join those stages
without replacing their source records.

## Canonical isolated stage workflow

The two environments exchange files under `benchmarking/runs/<run-id>/`; no
installed package state is passed between them.

First activate the application environment and generate answers through the
production RAG path:

```bash
python -m benchmarking.generate_answers --run-id <run-id>
```

Then activate the evaluation environment and run the downstream stages against
the same repository-contained artifacts:

```bash
python -m benchmarking.evaluate_qasper_f1 --run-id <run-id>
python -m benchmarking.evaluate_deepeval --run-id <run-id> --judge-id <judge-id>
python -m benchmarking.generate_report --run-id <run-id> --judge-id <judge-id>
```

Angle-bracket identifiers are placeholders; replace them with valid run and
judge IDs. Generation and judging call Groq. Answer F1 and report generation
are provider-free after their required artifacts exist.

## Orchestrator alternative

`run_pipeline` launches every stage with the same `sys.executable`; its
subprocesses do not isolate Python dependencies. The published application and
evaluation contracts require incompatible PostHog major versions, so they cannot
form one resolver-clean environment.

The orchestrator is retained as a legacy convenience for pre-existing
environments in which the complete workflow has been independently
runtime-validated. Such environments are not reproducible from the two
published requirement files and can contain dependency-metadata conflicts. The
isolated stage sequence above is the supported clean-install workflow.

### Quick smoke run

First inspect a plan without writing artifacts or making provider calls:

```bash
python -m benchmarking.run_pipeline --run-id smoke-rag-001 --max-papers 1 --max-questions 2 --dry-run
```

`--dry-run` validates arguments, resolves the profile, and prints sanitized
planned commands. It does not execute stages or write a run directory.

A minimal live orchestrator smoke run requires a separately validated legacy
environment. It creates a new run directory, loads local retrieval models, calls
generation/checker providers, calls the semantic judge, and writes artifacts:

```bash
python -m benchmarking.run_pipeline --run-id smoke-rag-001 --max-papers 1 --max-questions 2
```

`--max-papers` takes the first papers in saved dataset order; `--max-questions`
takes questions in that selected paper order and stops at the total limit. A
smoke result is a diagnostic subset, not a replacement for the complete
benchmark.

### Complete orchestrator invocation in a legacy environment

For a new complete evaluation, use a new placeholder run ID:

```bash
python -m benchmarking.run_pipeline --run-id full-rag-20260911
```

This command can load local models, call runtime providers during generation,
call the semantic judge, and create or update artifacts under
`benchmarking/runs/full-rag-20260911/`.

The orchestrator supports stage control:

```bash
python -m benchmarking.run_pipeline --run-id existing-run --skip-generation
python -m benchmarking.run_pipeline --run-id interrupted-run --resume-generation --skip-qasper --skip-deepeval --skip-report
```

The first form validates and reuses an existing complete generation run, then
runs the remaining non-skipped stages. The second form resumes compatible
generation artifacts through the orchestrator and deliberately leaves
downstream stages for a later invocation. `--skip-generation` cannot be combined
with generation limits or `--resume-generation`; skipped stages are explicitly
recorded.

## Pipeline stages

### Answer generation

[`generate_answers.py`](../benchmarking/generate_answers.py) loads the validated
profile and QASPER records, indexes each configured PDF in a fresh
single-document knowledge base, and calls the production
[`chat()`](../chat/service.py) service. Every question receives empty history and
a fresh conversation ID. The configured chunking, retrieval mode, breadth,
fusion threshold, and reranker limits are passed to that production path.

Each answer record retains the raw UI-returned answer, a cleaned answer, status,
error, elapsed time, question, annotation bundle, and conversation ID. The
cleaner removes only a complete terminal diagnostics suffix: it supports the
current Markdown **Answer diagnostics** block and retained historical terminal
formats. It does not remove isolated words that resemble diagnostic labels.
An empty answer after stripping becomes a recorded generation error.

The harness records errors rather than dropping cases. It copies only matched
application log entries after the manifest's starting byte offset and only for
the run's conversation IDs. Before writing them to the run directory, it
normalizes failure causes and redacts known sensitive patterns only in diagnostic
explanations; the remaining records are still content-bearing. That preserves
checker availability/counts without copying unrelated global-log activity. See
the [telemetry benchmark contract](telemetry.md#benchmark-relationship) for the
runtime record and safe-handling boundary.

### Deterministic QASPER Answer F1

[`evaluate_qasper_f1.py`](../benchmarking/evaluate_qasper_f1.py) is offline once
generation artifacts exist. It validates the generation manifest and current
profile provenance, selects the latest successful attempt per case (or the last
attempt when no success exists), and writes deterministic scores and metrics.

**QASPER normalized-token Answer F1** lowercases text, removes ASCII
punctuation and articles (`a`, `an`, `the`), normalizes whitespace, and computes
token-overlap F1. It converts annotations to extractive, abstractive, boolean,
or unanswerable references, canonicalizes the standard abstention to
`Unanswerable`, and retains the maximum F1 across available human references.
Recorded generation errors score zero; a missing selected answer record is an
integrity error that stops evaluation.

Short gold references can give a correct longer explanation low lexical overlap.
The implementation is adapted to this project's append-only artifacts, so it
does not establish direct comparability with the official QASPER leaderboard.

```bash
python -m benchmarking.evaluate_qasper_f1 --run-id existing-run
```

This command writes `qasper_scores.jsonl` and `qasper_metrics.json`; it makes no
provider calls but does mutate those derived artifacts.

### Semantic correctness evaluation

[`evaluate_deepeval.py`](../benchmarking/evaluate_deepeval.py) uses DeepEval
G-Eval with the configured Groq judge. The publication record identifies the
semantic evaluator as `qasper-reference-geval-v1.6`; its exact model, DeepEval
version, threshold, request settings, and provider SDK version are persisted in
the judge artifacts.

For every successful generated answer, the evaluator makes one independent
provider request for each valid QASPER annotation/reference. Each request
receives that annotation's answer and annotation-specific gold evidence:
nonempty `highlighted_evidence` first, then `evidence`, otherwise an explicit
absence message. The maximum semantic score is retained; the first annotation
wins ties. QASPER cross-reference placeholders such as `BIBREF<n>` and
`TABREF<n>` are non-semantic annotation markup. The v1.6 policy does not treat
an absent detail in an abbreviated reference as fabrication by itself.

Scores are normalized from the fixed 0--10 rubric to 0--1. A score passes when
it is at least the recorded threshold. A generation error receives semantic
score zero without a judge call. A judge error remains unavailable rather than
being converted to zero, reducing semantic-evaluation coverage and making the
full-run semantic correctness field null. LLM judging remains nondeterministic
even with fixed inputs and settings.

```bash
python -m benchmarking.evaluate_deepeval --run-id existing-run --judge-id judge-v1
```

This command calls the judge provider and writes
`deepeval/judge-v1/scores.jsonl`, `metrics.json`, and `judge_manifest.json`.

### Report generation

[`generate_report.py`](../benchmarking/generate_report.py) validates generation,
Answer F1, semantic, application-log, provenance, and pipeline records before
rendering Markdown. Given fixed valid inputs, the pipeline freezes its completion
timestamp before report rendering so a regenerated report is byte-identical.
It writes `evaluation_report.md` inside the run directory.

```bash
python -m benchmarking.generate_report --run-id existing-run --judge-id judge-v1
```

The command is provider-free but writes the report. It can exclude intentionally
skipped deterministic or semantic stages using `--skip-qasper` or
`--skip-deepeval`; those switches exclude stale files as well. Do not manually
rewrite generated reports or historical benchmark artifacts. Headline results
belong in the separate benchmark-results documentation.

For the selected publication run, the retained canonical run's generated report
is the immutable canonical record. The distributable report is a reviewed
derivative: public-distribution disclosures and relative links may differ, and
every difference must be recorded in the publication-bundle README or redaction
manifest. This does not authorize editing the canonical generated report or its
source artifacts. Because public JSONL copies omit reference/evidence text and
substantive generated answers, they support inspection and provenance but are
not drop-in inputs for rerunning reference-based scoring. Recreate the licensed
local dataset and papers with the documented builder before executing the
canonical staged workflow.

## Resuming interrupted work

Use resume only when identifiers, selected cases, settings, dataset digest, and
recorded provenance describe the same evaluation.

Resume generation in the application environment:

```bash
python -m benchmarking.generate_answers --run-id interrupted-run --resume
```

Resume semantic judging in the evaluation environment:

```bash
python -m benchmarking.evaluate_deepeval --run-id interrupted-run --judge-id judge-v1 --resume
```

In a previously runtime-validated legacy orchestrator environment only:

```bash
python -m benchmarking.run_pipeline --run-id interrupted-run --resume-generation --resume-judge
```

Generation resumes its append-only answer history, skipping successfully
completed cases and retaining failed attempts for accounting. Judge resume
selects the latest attempt per evaluator resume key and retries unresolved judge
errors; successful judgments and generation-error records are terminal. The
pipeline validates existing generation artifacts before reuse with
`--skip-generation`, and it can recover a stale temporary pipeline manifest only
after it verifies provenance, stage artifacts, and forward progress.

Choose resume flags according to the interrupted stage:

- `--resume-generation` when generation has compatible resumable artifacts.
- `--skip-generation --resume-judge` when generation is complete and the
  matching judge run is interrupted.
- Both resume flags when both stages have compatible resumable artifacts.
- Do not use `--resume-judge` when no matching judge run exists.

If configuration, dataset digest, limits, case selection, run ID, judge ID,
evaluator version, or recorded request settings do not match, resume stops
rather than merging incompatible work. Use a new run or judge ID when the
condition genuinely changes. Do not delete or hand-edit JSON/JSONL as routine
recovery; retain attempts and use the validated resume path after transient
provider failures or quota recovery.

## Subset and failure-focused evaluation

Simple smoke subsets use `--max-papers` and `--max-questions` as shown above.
The generation limits are applied in saved dataset order. Semantic evaluation
also has `--max-questions`, which limits judge cases independently of generation.

Failure-focused regeneration selects only latest source-judge results strictly
below that source's recorded threshold. It requires both the source run and
source judge IDs, preserves source dataset/threshold/evaluator provenance, and
orders selected cases according to the saved dataset. In the application
environment, generate the selected answers with a new run ID:

```bash
python -m benchmarking.generate_answers --run-id recovery-run --select-below-threshold-from-run source-run --selection-judge-id source-judge
```

Run the downstream Answer F1, semantic-judging, and report stages in the
evaluation environment with the corresponding commands in the [canonical
isolated stage workflow](#canonical-isolated-stage-workflow).

In a separately runtime-validated legacy orchestrator environment only, the
equivalent orchestration command is:

```bash
python -m benchmarking.run_pipeline --run-id recovery-run --select-below-threshold-from-run source-run --selection-judge-id source-judge
```

It cannot be combined with generation `--max-papers` or `--max-questions`, and
the new run ID must differ from the source run. A targeted subset is
selection-biased diagnostic work, never a new complete-benchmark result.
Artifacts retain the selection provenance and leave full-run semantic
correctness inapplicable.

`compare_checkers.py` is a separate controlled comparison of two production
checker models on identical RAG inputs; it is not a stage of the standard
generation/F1/semantic pipeline.

## Artifact layout

The standard pipeline creates this layout (some stage artifacts are absent only
when that stage was intentionally skipped):

```text
benchmarking/runs/<run-id>/
  answers.jsonl
  application_logs.jsonl
  generation_manifest.json
  qasper_scores.jsonl
  qasper_metrics.json
  deepeval/<judge-id>/
    scores.jsonl
    metrics.json
    judge_manifest.json
  pipeline_manifest.json
  pipeline_summary.json
  evaluation_report.md
```

| Artifact | Purpose |
| --- | --- |
| `answers.jsonl` | Append-only generation attempts, including raw/cleaned answers and errors |
| `application_logs.jsonl` | Conversation-matched, content-bearing runtime records with diagnostic-explanation sanitization |
| `generation_manifest.json` | Generation condition, resolved profile, model/request provenance, selection, and counts |
| `qasper_scores.jsonl`, `qasper_metrics.json` | Per-case deterministic scores and their aggregate |
| `deepeval/<judge-id>/scores.jsonl` | Append-only semantic reference-judgment attempts |
| `deepeval/<judge-id>/metrics.json` | Semantic coverage, aggregation, pass rate, and reliability aggregates |
| `deepeval/<judge-id>/judge_manifest.json` | Judge identity, evaluator contract, request provenance, and attempt counts |
| `pipeline_manifest.json`, `pipeline_summary.json` | Orchestrator stage state and cross-stage summary |
| `evaluation_report.md` | Generated, validated run-level report |

Atomic writes protect JSON summaries, reports, and snapshots during replacement.
On Windows or cloud-synced folders, a temporary replacement failure can be
retried by the artifact helper; it is not a reason to manually alter a
historical record.

## Coverage, failures, and diagnostics

Generation coverage counts selected cases, successful answers, errors, and
application-log matches. Deterministic Answer F1 requires one selected answer
record per case. Semantic-evaluation coverage is the share of selected cases
with non-null semantic scores; it separately accounts for generation errors,
judge errors, and successful or failed reference judgments.

`unknown` retrieval relevance or groundedness means a runtime checker was
unavailable, failed, or returned invalid output. It is neither a semantic-judge
failure nor a negative correctness label. Missing diagnostic records, unknown
checker labels, generation errors, and judge errors are distinct dimensions.
Complete answer generation does not imply complete semantic judging, and
complete semantic judging does not guarantee correct answers.

The report treats incomplete generation, absent required evaluation artifacts,
incomplete semantic coverage, unresolved judge errors, or incomplete matched
application logs as publication-blocking. Diagnostic unavailability is reported
separately; complete primary evaluation with only unavailable checker labels is
a minor diagnostic-coverage warning rather than silently relabeling those cases.

## Interpreting the metrics

Interpret recorded values with their denominators, coverage, threshold, and
provenance. Generation success is not correctness. Answer F1 is normalized
lexical overlap. Mean semantic correctness is the mean among non-null evaluated
cases; semantic pass rate is the share of those cases meeting the recorded
threshold. Full-run semantic correctness is present only for a non-targeted run
with complete semantic coverage.

Retrieval relevance and groundedness describe the final runtime context and
answer, not a comparison to QASPER annotations. Latency and transport telemetry
describe observed operational behavior, not a service guarantee. See
[Benchmark results](benchmark-results.md) for the approved publication-run
metrics and interpretation.

## Pacing and provider limits

Generation applies a configured minimum start-to-start case interval. Runtime
generation also invokes the production relevance and groundedness checkers.
Semantic judging separately applies its request interval, bounded retry backoff
with jitter, and a maximum accepted `Retry-After` duration. It records provider
attempts, retries, rate-limit responses, pacing sleep, backoff sleep, and
unresolved judge errors.

Pacing reduces bursts but cannot increase, reset, or guarantee provider quotas.
Daily-token and per-minute limits remain provider/account constraints. Resume is
appropriate after a transient failure or renewed quota, but cannot repair a
generation/checker request that was never admitted by an exhausted daily quota.

## Windows and OneDrive considerations

Run modules from the repository root and activate the environment for the stage
being run. Keep `evaluation_settings.py` paths repository-relative and quote
paths that contain spaces when invoking external tooling. PowerShell uses the
backtick for line continuation; Bash uses a backslash, so do not copy one
shell's continuation syntax into the other.

OneDrive or another synchronizer can temporarily lock files while the pipeline
atomically replaces JSON, Markdown, or manifests. Pause conflicting sync/edit
activity and retry through the documented resume command if a replacement error
occurs. Do not manually edit generated JSON or JSONL to work around a lock.

## Reproducibility boundaries

This workflow makes the evaluation inspectable through fixed dataset selection,
resolved configuration and dataset digests, model IDs, run/judge identity,
versioned evaluator and pipeline contracts, Git provenance, and preserved
artifacts. It does not promise identical external model responses, identical
semantic-judge outputs, permanent provider availability, official QASPER
leaderboard equivalence, or complete environmental reconstruction from an
unrecorded dirty worktree.

## Troubleshooting

| Symptom | Safe response |
| --- | --- |
| Dataset or PDF path error | Check the repository-relative profile paths and acquire required papers through the documented dataset-build process. |
| Existing run directory | Use `--resume` only for the same condition; otherwise choose a new run ID. |
| Resume provenance error | Treat the recorded run as immutable and start a new compatible run/judge condition rather than editing artifacts. |
| Judge directory already exists | Use its matching `--judge-id` with `--resume`, or choose a new judge ID for a changed condition. |
| Provider, timeout, or quota failure | Preserve artifacts, wait for a suitable provider state, then resume the matching stage. |
| Missing semantic coverage | Inspect judge attempts and generation errors; do not convert unavailable judgments to zero. |
| OneDrive replacement error | Pause conflicting synchronization, then retry the appropriate resume command. |
| Need a report from completed artifacts | Run `generate_report` with the matching run and judge IDs; do not hand-edit the historical report. |
