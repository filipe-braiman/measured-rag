# User guide

Measured RAG combines document question answering with hands-on RAG
experimentation. You can tune chunking and retrieval, ask questions across PDF
and Word documents, and inspect indexed chunks, ranked evidence, diagnostics,
and debug output to understand how configuration choices affect each result.
This guide explains the interface and workflows; see
[Configuration](configuration.md) for exact settings and limits and
[Architecture](architecture.md) for system flow.

## Before you begin

Use a PDF, DOC, or DOCX document with extractable text. PDF parsing does not
include an optical character recognition (OCR) fallback: scanned pages that
contain only images may provide no usable text. Word documents are organized
using their detected sections; PDF sources retain page attribution.

v1.0 is optimized and evaluated for English-language documents. Other languages
are not blocked, but equivalent retrieval quality is not established.

The hosted portfolio demo is configured for
[demo.measured-rag.com](https://demo.measured-rag.com); availability depends on
the current deployment. In hosted mode, the **Public Portfolio Demo** notice
explains that shared use is limited. Do not upload confidential, sensitive,
regulated, or personally identifiable documents to the public demo.

Local execution also sends questions, relevant conversation excerpts, retrieved
passages, and generated answers to external inference services as needed. It is
not a fully offline workflow. See the
[model and provider boundary](configuration.md#model-configuration).

## Interface overview

### Knowledge Base

**Document Mode** chooses replacement or append behavior. **Upload Document /
Upload Documents** accepts one file in single-document mode or a collection in
multi-document mode. **Knowledge Base Status** reports the active document mode,
indexed-document and chunk counts, per-document indexing settings, indexing
outcome and timing, duplicates, and any reported errors. **Indexing
Configuration** controls how uploaded text is split.

The **Retrieval** section contains **Retrieval Mode** and **Retrieval
Configuration**. In multi-document mode, **Document Scope** lets you search all
uploaded documents or choose entries in **Selected Documents**.

### Conversation

The **Document Conversation** panel displays the exchange. Type into **Question**
and select **Ask Measured RAG**, or press Enter. **Clear Conversation** starts a
fresh exchange while retaining the indexed documents.

### Inspection & Debugging

Three read-only outputs appear in expandable accordions:

- **Indexed Chunks** shows the indexed text with document, chunk, page, and
  section information where available.
- **Retrieved Sources** shows the final passages supplied to generation, with
  attribution and ranking scores.
- **Debug Last Answer** shows the original and rewritten questions, selected
  route, candidate counts, ranking information, diagnostic explanations, timings,
  final prompt, and available token usage.

The accordions start open. Collapsing one changes its presentation, not whether
the underlying output or diagnostics are produced.

## Basic workflow

### 1. Choose a document mode

Use **Single-document** to explore one document. Use **Multi-document** to build
a collection and optionally restrict each question to selected documents.
Choose the mode before uploading: changing it resets the knowledge base and
conversation.

### 2. Configure indexing

Begin with the default **Chunk Size (characters)** and **Chunk Overlap
(characters)**. Chunk size controls the target text window; overlap retains some
text across adjacent windows. Actual chunk lengths vary with text boundaries.

These settings take effect when indexing runs. Moving a slider does not rebuild
existing chunks. To replace the current indexing, clear the uploader, choose the
settings, and upload again; this also clears the conversation. See
[indexing controls](configuration.md#indexing-controls) before changing them.

### 3. Upload and index documents

Select your file or files. The upload change starts indexing; there is no separate
index button. Wait for **Knowledge Base Status** and **Indexed Chunks** to update
before asking a question. The ask button is temporarily unavailable while the
normal indexing chain runs.

Check the document list and chunks, not just the presence of a filename in the
uploader. **Knowledge base ready** describes indexed state; **Errors** can appear
alongside successful entries when some inputs failed.

### 4. Configure retrieval

Start with **Auto** or choose **Semantic**, **Keyword**, or **Balanced**. All modes
still use both dense semantic retrieval and BM25 keyword retrieval. Modes change
their weighting or select a route; they do not turn either retriever off.

**Retrieval Configuration** controls how many candidates are retrieved, which
survive fusion filtering, how many are reranked, and how many final passages
reach generation. Changes apply to subsequent questions without reindexing.
Exact values and interactions are in
[retrieval controls](configuration.md#retrieval-controls).

### 5. Ask a question

Ask a specific question about the indexed material. The interface clears the
input and temporarily displays **Processing…** while the request runs.

Query rewriting is automatic for questions that pass document and scope checks,
including the first question. It uses recent conversation context to resolve
references. Answer generation receives the original question and uses history
for reference resolution, not as documentary evidence. When moving to a new
document or topic, consider clearing the conversation first.

### 6. Inspect the answer and evidence

Read the answer alongside **Retrieved Sources**. The model is instructed to
answer from the selected evidence and may abstain when that evidence is
insufficient. An abstention describes the context available to the model;
retrieval may still have missed a relevant passage.

Supporting excerpts are prompt-directed model output, not independently
guaranteed quotations. Compare them with the retrieved text and original
document. Then read **Answer diagnostics** and, when useful, the explanations in
**Debug Last Answer**.

## Single-document workflow

Single-document mode keeps one indexed document. To use a different document,
remove the current file and then upload the new one. Removing the current file
resets the knowledge base, clears the conversation and inspection outputs, and
creates a new conversation ID. The subsequent upload therefore starts from a
fresh state.

## Multi-document workflow

Add documents while retaining the existing uploader entries. New inputs append
to the knowledge base. Duplicate detection compares file content together with
chunk size and overlap; a duplicate is reported under **Skipped duplicates**.
The same file indexed with different chunk settings can become another entry
and count toward hosted limits.

**All uploaded documents** searches the indexed collection. **Only selected
documents** reveals **Selected Documents** and restricts retrieval to that
selection. Select at least one entry before asking. An accepted indexing update
clears the selection, including when it only skips duplicates, so check it again
after uploads. Changing scope alone does not clear the conversation.

Use distinct filenames for distinct inputs. Selection resolves names to document
IDs, so multiple indexed entries with the same filename cannot be distinguished
reliably through the selector. **All uploaded documents** includes all indexed
entries regardless of that naming ambiguity.

Removing an existing entry from the multi-document uploader resets the entire
knowledge base, clears the conversation, and empties the uploader. It is not an
individual-document deletion operation. Upload the desired collection again
after the reset.

## Understanding answer diagnostics

**Retrieval relevance** judges whether the final retrieved context helps answer
the rewritten question. **High** means the checker considers the information
sufficient; **Medium** means useful but incomplete or indirect; **Low** means
insufficient relevant information.

**Groundedness** judges the generated answer against that context. **Grounded**
means the checker considers its important claims supported; **Partially grounded**
indicates unsupported, vague, or extrapolated content; **Not grounded** indicates
mostly unsupported or contradicted content.

These are model judgments, not correctness scores against a reference answer.
They can disagree, and favorable labels do not establish that the answer is
correct. They run after generation and do not trigger automatic answer repair.

**Unknown** means a check was unavailable, failed, or returned an invalid result.
It is neither a successful assessment nor automatically a negative result. If
there is no retrieved context at all, the application reports **Low** relevance
and **Not grounded** without calling those checkers. Numeric score semantics
belong in [Configuration](configuration.md#diagnostics-and-score-semantics).
Inspect **Debug Last Answer** for any available diagnostic details.

## Reset and replacement behavior

| Action | Indexed documents and chunks | Conversation and inspection |
| --- | --- | --- |
| Replace the single document through the UI | Previous knowledge base reset, then new document indexed | Conversation and inspection cleared; new conversation ID created during the empty-uploader reset |
| Append documents | Extended; duplicates skipped | History retained; sources/debug and document selection cleared |
| Remove a multi-document uploader entry | Entire knowledge base reset | Same reset as emptying the uploader |
| Change Document Mode | Entire knowledge base reset | History and inspection cleared; new conversation ID; scope returns to all documents |
| Clear Conversation | Retained | History, pending question, sources/debug cleared; new conversation ID; Indexed Chunks remains populated |

None of these actions replenishes the hosted query allowance. Resetting state or
deleting the vector collection is not proof that uploaded temporary files were
erased. See [state and lifecycle](architecture.md#state-and-lifecycle).

## Hosted-demo behavior

The **Public Portfolio Demo** applies file-size, accumulated-upload, document-count,
question-length, session-allowance, queue, and concurrency limits. The exact
values are owned by [hosted-demo guardrails](configuration.md#hosted-demo-guardrails)
and [queue settings](configuration.md#queue-and-concurrency-settings).

Wait for indexing to finish and verify scope before submitting. A nonempty
question that passes length validation consumes allowance when admitted, before
document/scope validation and downstream success. An accepted request can
therefore consume allowance even if it produces no answer. Blank or overlength
questions are rejected before allowance consumption.

Shared processing can make requests wait. These controls do not promise service
availability or unrestricted use. Local mode removes the cloud-specific budgets,
but remains subject to available resources and external-provider limits.

Cloud application telemetry excludes document and conversation text from its
operational records. This does not establish the contents or retention of all
dependency, provider, or platform logs. The demo provides no confidentiality,
account-based isolation, durable knowledge base, or guaranteed-deletion contract.
See the [telemetry privacy guidance](telemetry.md#privacy-and-safe-handling) for
the local and hosted application logging boundary.

## Practical usage guidance

- Begin with defaults and change one control at a time so you can interpret the
  effect in the sources and debug output.
- Try **Keyword** for identifiers or terminology-heavy questions, **Semantic**
  for conceptual questions, and **Balanced** when both matter. Keyword is still
  hybrid, not a literal phrase-search guarantee. **Auto** chooses using simple
  query signals; no mode is promised to perform better on every document.
- If relevant text is indexed but absent from the sources, inspect scope,
  candidate counts, and filtering. Increase retrieval breadth cautiously;
  returning more candidates can add work and change normalized scores.
- Treat thresholds as candidate-selection controls, not confidence probabilities.
  More passages do not automatically improve an answer.
- Try an intentionally unsupported question to observe abstention behavior, while
  remembering that a hosted submission uses the same allowance as any other.

## Common states and troubleshooting

| State | What to check or do |
| --- | --- |
| No documents indexed / Document required | Upload a supported document and confirm its chunks appear before asking. |
| Unsupported file type | Use PDF, DOC, or DOCX. Renaming another format does not convert its contents. |
| Empty or unreadable document / Errors | Check that the document opens and contains extractable text. Inspect the status; the application does not provide one universal empty-document message. |
| Scanned PDF with no usable chunks | Supply a text-bearing version. The application has no OCR fallback. |
| No documents selected | Select an indexed entry or switch to All uploaded documents. Recheck selection after indexing. |
| Skipped duplicates | The content and chunk settings match an indexed input. No additional copy is needed. |
| No evidence survives filtering | Inspect Indexed Chunks, scope, and Debug Last Answer. Consider less restrictive fusion filtering or broader retrieval; generation may still run with empty context. |
| Answer abstention | Compare the question and retrieved passages. The missing information may be absent from the document or simply not retrieved. |
| Unknown diagnostic | Inspect **Debug Last Answer** for any available diagnostic details. Do not interpret unavailable checking as correctness or failure of the answer itself. |
| Hosted allowance exhausted | Clear Conversation and document resets do not restore it. Run the application locally for further development; repeated hosted retries do not resolve exhaustion. |
| Replacement/reset surprise | Replacing the single document requires removing the current upload, which resets the knowledge base and conversation; multi-document removal clears the whole collection. Follow the reset table before rebuilding. |
| Provider or generation failure | The request can fail without a completed answer or fresh inspection output. Retry only when appropriate, considering hosted allowance. A previous source/debug view is not evidence for the failed request. |

Per-file indexing errors can leave a partial multi-document collection. After a
single-document knowledge base has been cleared, a failed new upload does not
restore the previous document.

## Current usage boundaries

Answers, generated excerpts, and diagnostic judgments require review against
the source documents. The interface does not establish OCR coverage,
multilingual optimization, persistent storage, authentication, or enterprise
tenancy. See [architectural boundaries](architecture.md#architectural-boundaries)
for the system-level scope, [Configuration](configuration.md) before changing
parameters, and [Limitations](limitations.md) for the complete validated scope,
privacy boundaries, and non-guarantees.
