# Measured RAG publication artifact bundle

This directory is the curated public artifact bundle for the Measured RAG
publication run `qasper-publication-v1-28p`, evaluated with judge ID
`qwen3.8-27b-publication-v1`. It preserves the recorded aggregate metrics,
per-question scores, run configuration, and provenance needed to inspect the
published result. The generated report in the retained canonical run is the
immutable record. This public report is a reviewed derivative with disclosure
and relative-link changes; public JSONL copies are deliberately redacted. The
canonical local run remains unchanged outside this bundle.

The original generated report SHA-256 is
`9EA8F3E045CD14151D9D4371FB286759B8EFAB96C8B3A386132E68C6B214F816`.
The reviewed public report SHA-256 is
`F396CAFB42DE1EEA2CA3793F3E7ECF70005270FF8CF66BE6827CD9187AF86D3B`.

Included artifacts:

- `evaluation_report.md`
- `generation_manifest.json`, `answers.jsonl`
- `qasper_metrics.json`, `qasper_scores.jsonl`
- `pipeline_manifest.json`, `pipeline_summary.json`
- `redaction_manifest.json`
- `deepeval/qwen3.8-27b-publication-v1/judge_manifest.json`
- `deepeval/qwen3.8-27b-publication-v1/metrics.json`
- `deepeval/qwen3.8-27b-publication-v1/scores.jsonl`

Deliberately excluded artifacts:

- `application_logs.jsonl`, because it contains substantial independently
  extracted retrieval previews
- Downloaded research-paper PDFs, which are not included and remain subject to
  their respective rights
- Dataset source files, caches, vector stores, temporary files, and unrelated
  operational run artifacts
- Gold answers, evidence and highlighted-evidence excerpts, extractive and
  free-form references, semantic-judge reasons, and duplicated reference text
  in the public JSONL copies
- All 109 substantive generated answers, because their evidence blocks contain
  contiguous source-text reproduction; 20 formulaic abstentions remain

## QASPER attribution and license

This publication bundle uses a selected 28-paper, 129-question subset derived
from [QASPER](https://allenai.org/data/qasper), which is distributed under the
[Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/).
The selection and evaluation-artifact formatting were produced for Measured RAG;
QASPER questions and non-textual evaluation facts needed to interpret the run
are retained. Gold annotation text and evidence excerpts are not included.
[`redaction_manifest.json`](redaction_manifest.json) identifies each removed
field, its count, character total, and the before/after file hashes. Downloaded
PDFs and independently extracted full-document content are also excluded.

QASPER was introduced by Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah
A. Smith, and Matt Gardner:

> Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A. Smith, and Matt
> Gardner. “A Dataset of Information-Seeking Questions and Answers Anchored in
> Research Papers.” In *Proceedings of the 2021 Conference of the North
> American Chapter of the Association for Computational Linguistics: Human
> Language Technologies*, pages 4599–4610. Association for Computational
> Linguistics, 2021.

Resources: [canonical paper record](https://aclanthology.org/2021.naacl-main.365/) ·
[QASPER project page](https://allenai.org/data/qasper) ·
[Hugging Face dataset page](https://huggingface.co/datasets/allenai/qasper) ·
[CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/)

```bibtex
@inproceedings{Dasigi2021ADO,
  title={A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers},
  author={Pradeep Dasigi and Kyle Lo and Iz Beltagy and Arman Cohan and Noah A. Smith and Matt Gardner},
  booktitle={Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  pages={4599--4610},
  publisher={Association for Computational Linguistics},
  year={2021},
  doi={10.18653/v1/2021.naacl-main.365},
  url={https://aclanthology.org/2021.naacl-main.365/}
}
```

## Public-artifact boundary

The overlap audit compared each pre-redaction `cleaned_answer` with its
corresponding locally retained PDF text and QASPER annotation/evidence strings.
It measured the longest whitespace-normalized exact-token sequence and the
longest sequence after HTML removal, Unicode NFKC normalization, casefolding,
punctuation removal, and whitespace collapse. Manual review began at eight
normalized tokens or 50 matched characters. The maximum was 38 exact tokens
and 40 normalized tokens (261 normalized characters); the longest matches were
substantive source sentences. All 109 quote-bearing answers were therefore
removed from each public JSONL copy. The remaining 20 standard abstentions had
at most two normalized tokens of source overlap.

The [redaction manifest](redaction_manifest.json) records affected identifiers,
answer hashes, match lengths, removed-field counts, and exact public/canonical
artifact hashes without reproducing the matched text. It also records the
public `qasper_metrics.json` change from a moving evaluator URL to the exact
audited upstream commit URL; no metric changed. The reviewed report also uses a
QASPER-derived subtitle and clarifies that the evaluator version was not
recorded in the canonical run. These curated JSONL files
support result inspection and provenance, but they are not drop-in inputs for
rerunning reference-based Answer F1 or semantic scoring because reference,
evidence, and substantive answer text is intentionally absent. Reconstruct the
licensed local inputs and follow the
[benchmarking workflow](../../../docs/benchmarking.md#canonical-isolated-stage-workflow)
for an independent evaluation.
