# Configuration

This is the canonical reference for Measured RAG's interactive controls, model
requests, environment settings, and application-level limits. The
[User guide](user-guide.md) explains workflows and visible results;
[Architecture](architecture.md) explains component boundaries.

## Configuration layers

| Layer | Authority | When it applies |
| --- | --- | --- |
| Interactive controls | [app.py](../app.py), [UI callbacks](../ui/controls.py) | Current browser session; indexing or the next question as specified below |
| Helper defaults | [Indexing](../ingestion/indexing.py), [chunk preparation](../ingestion/chunking.py) | Only when a caller omits arguments; not a substitute for UI defaults |
| Models and requests | [config.py](../config.py), [core/models.py](../core/models.py) | Code-defined process settings; not exposed through the UI or project-defined environment variables |
| Local/cloud policy | [Runtime](../deployment/runtime.py), [guardrails](../deployment/guardrails.py) | Selected at process startup |
| Container configuration | [Dockerfile](../Dockerfile), [model prefetch](../deployment/prefetch_models.py) | Image build and container runtime |
| Benchmark profiles | [Typed evaluation configuration](../benchmarking/evaluation_config.py) | Separate callers with explicit values; not interactive defaults |

The runtime dependency baseline pins Gradio 6.5.1 and Groq SDK 0.37.1 in
[requirements-app.txt](../requirements-app.txt) and
[constraints-app.txt](../constraints-app.txt). Framework/SDK defaults below refer
to those versions, not an unrestricted future installation.

## Indexing controls

Numeric ranges are inclusive. The following defaults describe a newly constructed
interface; changing a control does not rebuild an existing index.

| Exact UI label | Default | Minimum | Maximum | Step | Unit | Effect and timing |
| --- | --- | --- | --- | --- | --- | --- |
| Chunk Size (characters) | 700 | 200 | 2,000 | 50 | Characters | Target text window used on the next indexing operation |
| Chunk Overlap (characters) | 100 | 0 | 300 initially; dynamic after size changes | 10 | Characters | Requested overlap between adjacent windows on the next indexing operation |

On a chunk-size change, [update_chunk_overlap_limit](../ui/controls.py) sets the
overlap maximum to `floor(chunk_size / 2)` and clamps the current overlap to that
maximum. The initial maximum is 300, not 350: the dynamic rule runs on change.

Chunk size is a target window, not an exact emitted length. Whitespace boundaries,
long words, and final fragments affect length. The chunker requires positive
integer size and nonnegative integer overlap smaller than size, and can reduce
overlap at a boundary to keep advancing. PDF pages and Word section groups are
chunked separately. See [text processing](../ingestion/text_processing.py) and
[chunk preparation](../ingestion/chunking.py).

The UI upload callback passes both values explicitly. `index_files()` defaults
to size 700 and overlap 100 when omitted; `prepare_chunks_for_file()` instead
defaults to size 700 and overlap 70. That lower-level overlap is not the UI
default. Multi-document duplicate keys include content fingerprint, chunk size,
and overlap: changing either setting can create another indexed entry for the
same file. See [upload coordination](../ui/upload.py).

### Document and input controls

Non-numeric controls have no numeric minimum, maximum, step, or unit.

| Exact UI label | Default and accepted values | When it takes effect / caution |
| --- | --- | --- |
| Document Mode | Single-document; choices Single-document and Multi-document | Change immediately resets the knowledge base and conversation |
| Upload Document / Upload Documents | Empty; PDF, DOC, DOCX; one file in single mode, multiple in multi mode | Uploader changes initiate validation/indexing or reset behavior |
| Document Scope | All uploaded documents (`all`); alternative Only selected documents (`selected`) | Visible in multi mode; affects retrieval for the next question, not stored chunks |
| Selected Documents | Empty selection; choices are indexed filenames | Visible for selected scope; indexing updates clear selection; with **Only selected documents** active, an empty selection blocks the question until at least one indexed document is selected |
| Question | Empty text | Submitted through Ask Measured RAG or Enter; whitespace trimmed before admission |

Selected-document lookup maps filenames to IDs. Repeated filenames are ambiguous:
the map retains one ID for a given name. Use distinct filenames when selecting
individual documents. All-document scope uses every indexed ID. See
[indexing and scope resolution](../ingestion/indexing.py).

**Ask Measured RAG** submits; **Clear Conversation** resets the conversation.
**Indexed Chunks**, **Retrieved Sources**, and **Debug Last Answer** are read-only
outputs in initially open accordions. Opening or closing an accordion does not
disable processing.
Reset effects are owned by the
[user-guide reset table](user-guide.md#reset-and-replacement-behavior).

## Retrieval controls

All controls below affect subsequent questions without rebuilding the index.
Definitions and event inputs come from [app.py](../app.py); effective values are
used by [chat/service.py](../chat/service.py).

| Exact UI label | Default | Minimum | Maximum | Step | Unit | Purpose / interaction |
| --- | --- | --- | --- | --- | --- | --- |
| Retrieval Mode | Auto | Not applicable | Not applicable | Not applicable | Route | Auto, Semantic, Keyword, Balanced; see weights below |
| Retrieval Top-N (per retriever) | 20 | 5 | 50 | 1 | Candidates per retriever | Limits each retrieval branch before fusion; scope can reduce dense results |
| Hybrid Fusion Threshold | 0.30 | 0 | 1 | 0.01 | Normalized fused score | Retains candidates with score at least the threshold; 0 disables threshold filtering |
| Reranker Candidate Limit | 15 | 5 | 40 | 1 | Fused candidates | Caps fusion survivors sent to the cross-encoder, subject to the Top-K floor |
| Final Rerank Top-K | 5 | 1 | 10 | 1 | Passages | Caps reranked passages passed to answer generation |

There is no exposed reranker-score threshold, and the current reranker selects
by order and Top-K without a score cutoff. The fusion threshold applies before
reranking; it is not a reranker threshold.

## Retrieval modes and fusion weights

Both dense and BM25 retrieval execute in every mode. The UI values and
[routing implementation](../retrieval/routing.py) are:

| UI label | Runtime value | Dense weight, alpha | BM25 weight |
| --- | --- | --- | --- |
| Semantic | `semantic` | 0.80 | 0.20 |
| Keyword | `keyword` | 0.45 | 0.55 |
| Balanced | `balanced` | 0.65 | 0.35 |
| Auto | `auto` | Selected by the rules below | `1 - alpha` |

Auto inspects the rewritten query. Each of these adds one keyword signal:

- At most four whitespace-separated tokens.
- At least one digit anywhere in the query.
- At least one all-uppercase token with length at least two.
- Any single or double quotation mark, including an apostrophe.

A semantic signal is present when the lowercased, trimmed query is exactly one
of the following words, or starts with that word followed by a space: `what`,
`why`, `how`, `summarize`, `explain`, `describe`, `compare`, `discuss`, `tell`,
`outline`.

Two or more keyword signals with no semantic signal choose Keyword. A semantic
signal with zero keyword signals chooses Semantic. Every other case chooses
Balanced. This is a fixed heuristic, not a learned classifier or a guarantee of
the best route. Supported runtime values are `auto`, `semantic`, `keyword`, and
`balanced`.

## Parameter interactions

1. Retrieval Top-N limits each branch. Dense search takes the collection's top
   results and then filters by document scope; BM25 filters allowed documents
   before taking its top results. Relevant in-scope dense passages below the
   collection cutoff are not recovered. See [dense](../retrieval/dense.py) and
   [BM25](../retrieval/bm25.py) retrieval.
2. Fusion merges candidates by chunk ID. Its union can contain more candidates
   than a single branch, and fewer after duplicates and threshold filtering.
3. The effective reranker limit is `max(Final Rerank Top-K, Reranker Candidate
   Limit)`. Raising Top-K can therefore raise the effective limit above the value
   displayed in the candidate slider. Debug output reports effective counts.
4. Final Top-K is a cap, not a promised number of sources. Fewer surviving
   candidates produce fewer final passages; reranking cannot recover an excluded
   passage. See [reranking](../retrieval/reranking.py).
5. Broader retrieval changes the candidate population and its normalization, not
   just the number of passages. Larger candidate sets increase reranker work;
   larger final context also increases provider input. No setting is uniformly
   optimal.

For direct service callers, `chat()` coerces Top-N and Top-K to integers of at
least 1, raises the candidate limit to at least Top-K, and clamps fusion threshold
to 0 through 1. These guards are not the UI range contract; they do not enforce
every UI maximum. Query rewriting uses up to the last four history messages;
generation includes up to the last six. These are messages, not conversation
turns, and neither history window is a UI control.

## Diagnostics and score semantics

| Quantity | Meaning and active computation | Interpretation boundary |
| --- | --- | --- |
| Dense distance / displayed similarity | Chroma distance is converted to `1 / (1 + distance)` | The displayed similarity is not the normalized dense contribution to fusion |
| Normalized dense score | Min-max normalization of candidate similarities; equal similarities all receive 1 | Relative to the current dense candidate set |
| Normalized BM25 score | Min-max normalization within the selected BM25 Top-N; equal raw scores all receive 1, including all-zero scores | BM25 tokenization uses whitespace splitting; this score is not proof of a lexical match |
| Fusion score | `alpha * normalized_dense + (1 - alpha) * normalized_BM25`; missing branch contributes 0 | Ranking/filtering score, not a probability or calibrated confidence |
| Reranker score | Cross-encoder prediction for rewritten-query/structured-passage pairs; highest first | No calibrated probability claim or current reranker-score cutoff |
| Retrieval relevance label | External checker returns `high`, `medium`, or `low` based on sufficiency of selected context | Not assigned by thresholding the numeric score below |
| Retrieval relevance score | External passage judgments 0, 1, 2 are divided by 2 and rank-weighted over at most five positions | Operational context diagnostic, not reference correctness |
| Groundedness | External checker returns `grounded`, `partially_grounded`, or `not_grounded` for answer support in context | No numeric confidence or reference-answer comparison |
| `unknown` | Failed/unavailable/invalid diagnostic result; relevance numeric score is unavailable | Never silently treat it as a success or a negative judgment |

The relevance weights are `1.0, 0.8, 0.6, 0.4, 0.2`. The score is the weighted
sum divided by the sum of weights used. All returned passage judgments must be
valid, but only the first five positions contribute to this number even when
Top-K exceeds five. The overall label is separately supplied by the checker.
There are no application numeric cutoffs mapping this score to High/Medium/Low.

When retrieval returns no passages, the service assigns retrieval relevance
`low` with score `0.0` and groundedness `not_grounded` without invoking the
corresponding diagnostic checkers. Otherwise the two diagnostics run
concurrently after generation and do not revise or gate the answer. Label
definitions for readers are in the
[user guide](user-guide.md#understanding-answer-diagnostics).

For a BM25-only candidate, fusion retains a fallback distance of `1.0`, which
source/debug formatting displays as similarity `0.5`. Those values are not an
observed dense match. Implementation sources:
[fusion](../retrieval/fusion.py), [source formatting](../utils/source_formatter.py),
[relevance](../retrieval/relevance.py), and
[groundedness](../generation/groundedness.py).

## Model configuration

These IDs match the approved [architecture model table](architecture.md#local-models-and-external-services).
They are source constants in [config.py](../config.py), not environment variables
or UI choices.

| Role | Active model | Execution |
| --- | --- | --- |
| Embedding | `BAAI/bge-small-en-v1.5` | Local; normalized document and query embeddings |
| Reranking | `cross-encoder/ms-marco-MiniLM-L6-v2` | Local cross-encoder |
| Query rewriting | `openai/gpt-oss-20b` | Groq |
| Answer generation | `openai/gpt-oss-120b` | Groq |
| Retrieval relevance | `openai/gpt-oss-20b` | Groq passage judgments plus local aggregation |
| Groundedness | `openai/gpt-oss-20b` | Groq |

[core/models.py](../core/models.py) initializes and caches the local models by
model ID without a revision argument. The container prefetches these exact
snapshots and maps its offline cache references to them:

| Model | Container snapshot revision |
| --- | --- |
| `BAAI/bge-small-en-v1.5` | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | `233902d25c440f23af6f7d6e94d2946bac0bee0a` |

Source: [deployment/prefetch_models.py](../deployment/prefetch_models.py). The
image enables offline model lookup after prefetching; it still loads weights
into process memory at startup. Offline lookup does not disable Groq requests.
Local execution also sends role-specific question, history, context, and answer
content to Groq. See the [architecture boundary](architecture.md#local-models-and-external-services).

## Provider request settings

These values come from [core/models.py](../core/models.py) and the actual request
builders. They are process-level code settings, not sliders or environment knobs.

| Role | Temperature | Token budget | Reasoning parameters | Seed | Timeout / SDK retries |
| --- | --- | --- | --- | --- | --- |
| Answer generation | 0.5 | `max_completion_tokens=1536` | `reasoning_effort=low`; `include_reasoning=False` | Not supplied | Shared client defaults: 60-second timeout, 5-second connect timeout; 2 retries |
| Query rewriting | 0.0 | `max_tokens=200` | No request-level reasoning settings supplied | Not supplied | Same shared client defaults |
| Retrieval relevance | 0.0 | `max_completion_tokens=512` | `reasoning_effort=low`; `reasoning_format=hidden` | `None` (no fixed seed) | Explicit 45-second timeout; 2 retries |
| Groundedness | 0.0 | `max_completion_tokens=512` | `reasoning_effort=low`; `reasoning_format=hidden` | `None` (no fixed seed) | Explicit 45-second timeout; 2 retries |

[core/llm.py](../core/llm.py) constructs the shared client without overriding its
timeout or retries. The inherited values above are Groq SDK 0.37.1 defaults.
Two retries permit up to three transport attempts for retryable failures, not
three freshly judged answers. Timeout settings are not an end-to-end request
deadline: queuing, other stages, retries, and backoff can add elapsed time.

Both diagnostic builders send strict JSON-schema response formats and validate
the returned choice count, completion status, content, labels, and explanations.
Relevance additionally validates one integer judgment per passage. The recorded
`strict_schema_mode=json_schema_strict` setting corresponds to this fixed schema
construction; it is not a user-facing format selector. The commented historical
seed is not an active application seed.

- [Rewriting](../generation/query_rewriter.py) falls back to the original question
  on empty output or an exception. It is attempted for every query that passes
  document and scope checks, even with empty history.
- [Generation](../generation/answer_generator.py) requires one nonempty answer
  choice and rejects completion-token truncation. Provider or response-validation
  failure can fail the request; there is no application answer-repair loop.
- [Relevance](../retrieval/relevance.py) and
  [groundedness](../generation/groundedness.py) return `unknown` on failures or
  malformed output. They do not retry a judgment merely because its label is
  unfavorable.

## Local and hosted application settings

### Application-read environment variables

[config.py](../config.py) calls `load_dotenv()` and reads the first two variables;
[deployment/runtime.py](../deployment/runtime.py) reads `PORT`. An existing process
environment takes precedence over values loaded by the default dotenv call.

| Name | Required / default | Accepted values and scope | Effect / caution |
| --- | --- | --- | --- |
| `GROQ_API_KEY` | Required for provider-client initialization; no application default | Provider-issued credential; shared local/cloud | Keep secret; do not put a real value in documentation, logs, or Git |
| `DEPLOYMENT_MODE` | Optional; `local` in source, `cloud` in the image | Exactly `local` or `cloud` | Selects guardrails, launch options, queue wiring, and telemetry; invalid values raise errors |
| `PORT` | Required in cloud; no application default; ignored by the local launch helper | Integer string from 1 through 65535 | Cloud binds `0.0.0.0` at this port; normally provided by the hosting platform |

`SHARE=False` is a Python constant, not an application environment variable.
Local launch passes it explicitly; cloud launch forces sharing off. The local
helper does not set the listening address or port, leaving Gradio's defaults in
effect. No public environment variable changes the application's model IDs,
retrieval defaults, hosted-budget constants, or local evaluation-log path.

### Inherited framework and SDK settings

The following optional settings are read by the pinned dependencies, not directly
by application configuration. They are distinguished here from project-defined
knobs. Set process-level framework variables before startup; some are read when
Gradio is imported, before `config.py` loads dotenv.

| Name | Default / accepted value | Scope and effect |
| --- | --- | --- |
| `GRADIO_SERVER_NAME` | `127.0.0.1`; bindable hostname or address | Local listening address; cloud supplies its own explicit address |
| `GRADIO_SERVER_PORT` | `7860`; valid port integer | Local starting port; cloud supplies `PORT` explicitly |
| `GRADIO_NUM_PORTS` | `100`; positive integer | Number of successive local ports Gradio tries when no explicit port is supplied; not a cloud instance count |
| `GRADIO_ANALYTICS_ENABLED` | `True`; use `True` or `False` | Gradio analytics; image sets `False`; separate from application telemetry |
| `GROQ_BASE_URL` | `https://api.groq.com`; API base URL | SDK endpoint override; affects where credentials and request content are sent; the application does not set it |

Gradio's `GRADIO_SHARE` fallback does not override the explicit share argument.
Other framework facilities such as alternate root paths, SSR, and filesystem
allowlists are not established Measured RAG configuration profiles here; enabling
them can change serving behavior and requires separate validation. This is not
an exhaustive inventory of every environment variable consumed by dependencies.

### Container environment

These are the values explicitly set by the [Dockerfile](../Dockerfile), not
required additions to a local environment. Paths below are image-internal paths,
not personal host paths. Cache/temp overrides must point to appropriate readable
or writable locations; changing a path does not establish retention or deletion.

| Variable | Image value | Consumer / effect |
| --- | --- | --- |
| `HOME` | `/home/rag` | Non-root runtime user's home |
| `TMPDIR` | `/runtime/tmp` | General temporary-file location |
| `GRADIO_TEMP_DIR` | `/runtime/gradio` | Gradio file cache/upload temporary location |
| `XDG_CACHE_HOME` | `/runtime/cache` | Cache base for consumers using XDG conventions |
| `HF_HOME` | `/opt/models/huggingface` | Hugging Face cache/configuration base |
| `HF_HUB_CACHE` | `/opt/models/huggingface/hub` | Prefetch destination and Hub cache; prefetch code requires it |
| `SENTENCE_TRANSFORMERS_HOME` | `/opt/models/huggingface/hub` | Sentence Transformers model-cache location |
| `HF_HUB_OFFLINE` | `1` | Enables offline Hub lookup after model prefetch |
| `TRANSFORMERS_OFFLINE` | `1` | Requests offline Transformers operation after prefetch |
| `HF_HUB_DISABLE_TELEMETRY` | `1` | Disables Hugging Face telemetry |
| `ANONYMIZED_TELEMETRY` | `False` | Disables Chroma's anonymized telemetry setting |
| `PYTHONDONTWRITEBYTECODE` | `1` | Suppresses Python bytecode writes |
| `PYTHONUNBUFFERED` | `1` | Unbuffered Python standard output/error |
| `PIP_NO_CACHE_DIR` | `1` | Disables pip caching during installation |
| `PIP_DISABLE_PIP_VERSION_CHECK` | `1` | Disables pip version checks |

The image also sets `DEPLOYMENT_MODE=cloud` and
`GRADIO_ANALYTICS_ENABLED=False`, described above. The listed switches are
optional dependency/interpreter settings outside the image; these are the
configured image values, not an alternative local-default table. Offline switches
require cached assets and do not make the overall application offline. No
application log-directory environment override exists. See
[Deployment](deployment.md) for the public deployment architecture and hosted
operational posture.

## Hosted-demo guardrails

Values are fixed in [config.py](../config.py) and enforced in
[deployment/guardrails.py](../deployment/guardrails.py).

| Guardrail | Cloud value | Enforcement and accounting |
| --- | --- | --- |
| File formats | PDF, DOC, DOCX | Extension and file availability checked before indexing; loaders also restrict formats locally |
| Maximum file size | 10 MiB = 10,485,760 bytes | Each server-side uploaded file |
| Accumulated upload size | 20 MiB = 20,971,520 bytes | Indexed files plus distinct incoming indexing inputs; not vector-store or extracted-text size |
| Maximum documents | 3 | Accumulated knowledge base; single mode still retains only one document |
| Maximum question length | 2,000 characters | Length after trimming leading/trailing whitespace |
| Accepted submissions | 5 per session | Nonempty, length-valid submissions admitted through the session counter |

Duplicate accounting uses content plus chunk size and overlap. Matching duplicate
inputs do not add to the accumulated count/size; different chunk settings can
count the same file again. In single-document mode, removing the current upload
resets the knowledge base and conversation before another document is indexed.
Hosted upload-size accounting evaluates the subsequent document against the
fresh knowledge base rather than adding it to the removed document's size.

Admission consumes allowance before the chat service validates documents/scope
or calls providers. Missing documents, empty selection, or downstream failure can
therefore use an accepted submission. Blank/overlength input does not consume it.
Conversation, document, and mode resets never replenish the counter. See
[submission preparation](../utils/helpers.py) and [event wiring](../app.py).

These are callback/process controls. File checks occur after Gradio receives an
upload; they do not define a platform ingress-body limit. Local mode bypasses
these cloud budgets, without promising unlimited capacity or provider use.
Cloud Run instance count, resources, request concurrency, and domain operations
are deployment topics, not values inferred from these guards.

## Queue and concurrency settings

| Setting | Cloud | Local |
| --- | --- | --- |
| Explicit Gradio queue capacity | 20 waiting events | No application-set capacity |
| Explicit default event concurrency | 1 | Framework default; no application-wide override |
| Upload indexing and chat | Shared `cloud-heavy-work` group, limit 1 | No shared application heavy-work group |
| Question admission | Shared `query-admission` group, limit 1 for button and Enter | Each admission event has explicit limit 1, without a shared named group |
| Chunk-overlap adjustment | `queue=False` | `queue=False` |

Cloud chat is chained to successful admission, so rejected admission does not
invoke generation. Other event chains retain their existing Gradio behavior.
These groups serialize the specified callbacks within one process; they are not
fleet-wide quotas or locks around every UI action. Source: [app.py](../app.py).

## Telemetry configuration

`DEPLOYMENT_MODE` selects the policy in [telemetry/logger.py](../telemetry/logger.py):

- `local`: append detailed UTF-8 JSONL at repository-relative
  `answer_logs/rag_eval_log.jsonl`. Records contain questions, answers, document
  names, source previews, diagnostic explanations, and operational measurements.
- `cloud`: emit allowlisted operational JSON to stdout. It includes settings,
  counts, diagnostic labels/scores, timings, and token totals; excludes query and
  answer text, filenames, source previews, and conversation identifiers. A
  serialization failure emits a generic error event instead of falling back to
  detailed local logging.

There is no separate application switch for log path or content policy.
Dependency telemetry switches above do not disable Measured RAG's runtime
logging. This policy does not govern all dependency, provider, or platform logs.
See [Telemetry and diagnostics](telemetry.md) for serialized fields, event
destinations, interpretation, and privacy, and
[telemetry and inspection](architecture.md#8-telemetry-and-inspection) for its
place in the request path.

## Benchmark configuration boundary

Benchmark profiles use a separate typed, frozen configuration and explicit
settings. They can differ from interactive defaults. This page does not assign
benchmark values to the UI or reproduce benchmark commands, results, or judging
methodology. See [Benchmarking](benchmarking.md#evaluation-configuration) for
the complete benchmark-profile and evaluation configuration workflow.

Benchmark profiles use the same retrieval-mode vocabulary as the production
runtime: `auto`, `semantic`, `keyword`, and `balanced`.
[Generation](../benchmarking/generate_answers.py) passes the validated profile
value unchanged to the production chat service. The active
[settings](../benchmarking/evaluation_settings.py) use `auto`.

## Configuration cautions

Choose indexing settings before uploading and review the
[reset/replacement behavior](user-guide.md#reset-and-replacement-behavior) before
reindexing. Reset does not prove temporary-file deletion. Inspection can include
document content and the full prompt; review it before sharing.

Thresholds, ranking scores, and diagnostic labels do not establish reference
correctness or guaranteed groundedness. v1.0 remains optimized and evaluated for
English-language documents without blocking other languages. Avoid treating
local mode as fully offline, model caching as durable knowledge base storage, or
session state as an authentication or confidentiality guarantee.
