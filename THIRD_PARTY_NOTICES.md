# Third-party notices

Project SPDX license expression: AGPL-3.0-only

> Copyright (C) 2026 Filipe Braiman Carvalho

The complete project license is in [`LICENSE`](LICENSE). Third-party works keep
their own licenses. This inventory records components material to the source
release and application image; it is not a certification of every transitive
dependency.

## Document processing

### PyMuPDF and MuPDF

| Field | Recorded information |
| --- | --- |
| Component | PyMuPDF 1.26.7, bundling MuPDF 1.26.12 |
| Upstream | Artifex Software, Inc.; PyMuPDF was originally written by Jorj X. McKie |
| License | PyMuPDF is dual-licensed under GNU Affero GPL version 3 or an Artifex commercial license. Exact-release material is inconsistent about `only` versus `or-later`; this notice does not assign PyMuPDF an unsupported SPDX suffix. Bundled MuPDF source notices permit GNU AGPL version 3 or later. |
| Source | [PyMuPDF 1.26.7](https://pypi.org/project/pymupdf/1.26.7/), [PyMuPDF source release](https://github.com/pymupdf/PyMuPDF/releases/tag/1.26.7), and [MuPDF releases](https://mupdf.com/releases/history) |
| Distribution | Installed directly in the application image from `requirements-app.txt`; used by the production PDF loader. The package is not copied into this source repository. |
| Required action | Retain PyMuPDF's packaged `COPYING` and metadata. Provide the complete GNU AGPLv3 text, supplied by [`LICENSE`](LICENSE), and corresponding source for the covered combined application when applicable. Commercial licensing is an alternative upstream path, not the path selected for Measured RAG v1.0. |

## Locally packaged models

The image build downloads only the pinned revisions below. Their model cards
are retained beside the weights. The image also contains this notice and the
applicable license texts under `/app/third_party/licenses/`.

| Component | Revision | Recorded license | Exact source | Distribution and required action |
| --- | --- | --- | --- | --- |
| `BAAI/bge-small-en-v1.5` | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` | MIT; bundled upstream notice says Copyright (c) 2022 staoxiao | [Pinned model snapshot](https://huggingface.co/BAAI/bge-small-en-v1.5/tree/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a); [FlagEmbedding license at commit `fd1a2bdf`](https://github.com/FlagOpen/FlagEmbedding/blob/fd1a2bdf69488ffebe0327999d4400d8c8058a0b/LICENSE) | Downloaded during the image build and shipped in the image. Retain the model card and [`MIT-FlagEmbedding.txt`](third_party/licenses/MIT-FlagEmbedding.txt). |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | `233902d25c440f23af6f7d6e94d2946bac0bee0a` | Apache License 2.0 | [Pinned model snapshot](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/tree/233902d25c440f23af6f7d6e94d2946bac0bee0a) | Downloaded during the image build and shipped in the image. Retain the model card and [`Apache-2.0.txt`](third_party/licenses/Apache-2.0.txt). The exact pinned snapshot contains no standalone `LICENSE`, `COPYING`, or `NOTICE` file. |
| `en_core_web_sm` | 3.8.0; wheel SHA-256 `1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85` | MIT; packaged notice says Copyright 2021 ExplosionAI GmbH | [spaCy model release](https://github.com/explosion/spacy-models/releases/tag/en_core_web_sm-3.8.0) | Installed from the exact hashed wheel. Retain the MIT `LICENSE` included in the package data and distribution metadata. |

External generation and diagnostic models named in `config.py` are accessed
through the Groq provider. Their weights are not included in this repository or
application image. Provider access does not relicense Measured RAG or the local
components listed above.

## QASPER evaluator and data

### Adapted evaluator

[`benchmarking/evaluate_qasper_f1.py`](benchmarking/evaluate_qasper_f1.py)
contains modified portions adapted from AllenAI's
[QASPER evaluator at commit `e996b6c7`](https://github.com/allenai/qasper-led-baseline/blob/e996b6c7b1b5f95d9308a74e3586416c6e780df1/scripts/evaluator.py).
The upstream `qasper-led-baseline` repository is licensed under Apache License
2.0. Retain the modification/provenance notice in the source file and
[`Apache-2.0.txt`](third_party/licenses/Apache-2.0.txt). Measured RAG's project
license does not replace the Apache 2.0 terms on adapted upstream material.

### QASPER-derived materials

The curated publication artifacts under
`benchmarking/publication/qasper-publication-v1-28p/` retain the selected
QASPER questions and structured evaluation facts under the
[Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/).
The QASPER-derived public artifacts were selected and redacted for distribution.
See the [publication bundle documentation](benchmarking/publication/qasper-publication-v1-28p/README.md)
and its [redaction manifest](benchmarking/publication/qasper-publication-v1-28p/redaction_manifest.json)
for the distribution boundary, transformations, and hashes.

Required attribution:

> Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A. Smith, and Matt
> Gardner. “A Dataset of Information-Seeking Questions and Answers Anchored in
> Research Papers.” In *Proceedings of the 2021 Conference of the North
> American Chapter of the Association for Computational Linguistics: Human
> Language Technologies*, pages 4599–4610. Association for Computational
> Linguistics, 2021.

Resources: [canonical paper record](https://aclanthology.org/2021.naacl-main.365/),
[QASPER project page](https://allenai.org/data/qasper),
[Hugging Face dataset page](https://huggingface.co/datasets/allenai/qasper), and
[CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).

The content-bearing generated QASPER QA file and all 28 downloaded research
PDFs are intentionally excluded from Git and from the application image. The
repository retains only the reproducible paper-selection manifest at
[`benchmarking/data/qasper_selection_seed1001_size28.json`](benchmarking/data/qasper_selection_seed1001_size28.json).
`benchmarking/build_eval_data.py` can reacquire the inputs for local evaluation.
Each paper has independent terms: QASPER attribution and an arXiv download URL
do not grant permission to redistribute the paper. Research-paper PDFs are not
part of this distribution; redistribution depends on each paper's independent
terms.

## Project branding assets

The repository owner confirms that the Measured RAG mark, favicon, and social
cards under `assets/` are original project-owned works and use no external
stock assets or templates. They are covered by the project's `AGPL-3.0-only`
license and 2026 project copyright. The application uses text-only GitHub,
LinkedIn, and contact links; no third-party brand-icon artwork is embedded.

## Other application dependencies

Exact direct Python dependency versions are controlled by
`requirements-app.txt` and `constraints-app.txt`. Installed distributions keep
their packaged license and notice files. Material direct components include
PyTorch (BSD-3-Clause), Gradio and the Groq Python SDK (Apache-2.0), LangChain
packages (MIT), ChromaDB and rank-bm25 (Apache-2.0), Sentence Transformers and
Transformers packages (Apache-2.0), Unstructured (Apache-2.0), python-docx and
spaCy (MIT), and NumPy, Pydantic, python-dotenv, and PostHog (BSD/MIT families).
This source-oriented summary does not enumerate image-specific transitive
components, which retain their own upstream terms.

## Container base and operating-system components

The application base is the pinned amd64 image
`python:3.11.14-slim-bookworm@sha256:83f339c1be6340ae1096010fdccf6552ac932d8f410d45d206014916bdf37e48`.
The Dockerfile installs `ca-certificates`, `libmagic1`, `libgomp1`,
`libreoffice-writer`, and `fonts-dejavu-core` from the Debian snapshot dated
2026-09-01. Debian package copyright files under `/usr/share/doc/*/copyright`
and applicable common license texts must remain in the image; LibreOffice and
DejaVu carry multiple upstream notices that the project license does not
replace.

Exact resolved operating-system and transitive Python package versions are
image-specific. Installed components retain their upstream license and
copyright material; release inventories may be generated from the corresponding
image.

No generic root `NOTICE` file is created. The project AGPL does not require
one. Upstream license and `NOTICE` files are retained where their respective
licenses require them.
