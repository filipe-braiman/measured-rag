# Measured RAG — QASPER Evaluation Report

Documented end-to-end evaluation of the Measured RAG pipeline on a fixed QASPER-derived subset.

**Project repository:** [Measured RAG on GitHub](https://github.com/filipe-braiman/measured-rag)

- **Run ID:** qasper-publication-v1-28p
- **Overall status:** complete
- **Benchmark condition:** single document
- **Selected papers:** 28
- **Selected questions:** 129
- **Completion timestamp:** 2026-09-09T19:25:21.296882+00:00
- **Configuration profile:** qasper-28p-publication-v1
- **Configuration digest:** e611b6f1d660
- **Dataset digest:** ca391c629a52

## Executive Summary

| Measure | Result |
| --- | --- |
| Generation success | 129/129 (100.0%) |
| QASPER Answer F1 | 23.78% |
| Semantic correctness among non-null judged cases | 0.7131782945736438 (71.32%) across 129 non-null judged cases |
| Semantic pass rate | 0.6511627906976745 (65.12%; 84/129) |
| Semantic evaluation coverage | 100.00% (129/129) |
| Retrieval relevance valid-result coverage | 129/129 (100.0%) |
| Groundedness valid-result coverage | 129/129 (100.0%) |
| End-to-end median latency | 7.216 s across 129 cases |

Full-run semantic correctness: **0.7131782945736438 (71.32%)**.

No composite metric is calculated.

## Evaluation Status and Coverage

| Stage or condition | Status |
| --- | --- |
| Answer generation | 129/129 successful; 0 generation errors |
| Diagnostic checker availability | complete; 0 unknown |
| Deterministic QASPER evaluator | complete |
| Semantic judge | complete |
| Semantic selected cases | 129 |
| Successful judgments | 129 |
| Generation errors represented in semantic evaluation | 0 |
| Judge errors | 0 |
| Missing semantic cases | 0 |
| Duplicate generation attempts | 0 |
| Historical judge-error attempts | 0 |
| Unresolved judge errors | 0 |
| Application-log coverage | 129/129 (100.0%) |

Benchmark-run publication readiness: **Ready.**

## Benchmark Configuration

| Setting | Recorded value |
| --- | --- |
| QA data | benchmarking/data/qasper_qas_seed1001_size28.json |
| PDF directory | benchmarking/data/qasper_papers_seed1001_size28 |
| Retrieval mode | auto |
| Chunk size | 600 |
| Chunk overlap | 50 |
| Retrieval Top N | 30 |
| Fusion threshold | 0.15 |
| Rerank candidates | 30 |
| Rerank Top K | 10 |
| Judge threshold | 0.7 |
| Judge reasoning effort | default |
| Judge retry limit | 3 |
| Judge completion-token limit | 2048 |
| Judge request interval (seconds) | 6.0 |
| Judge retry base delay (seconds) | 2.0 |
| Judge retry maximum delay (seconds) | 60.0 |
| Judge retry jitter bound (seconds) | 0.5 |
| Judge maximum Retry-After (seconds) | 300.0 |
| Generation case interval (seconds) | 30.0 |

## Models

| Role | Model |
| --- | --- |
| Embedding | BAAI/bge-small-en-v1.5 |
| Reranker | cross-encoder/ms-marco-MiniLM-L6-v2 |
| Generator | openai/gpt-oss-120b |
| Query rewriter | openai/gpt-oss-20b |
| Relevance checker | openai/gpt-oss-20b |
| Groundedness checker | openai/gpt-oss-20b |
| Semantic judge | qwen/qwen3.8-27b |

## Answer Correctness

- QASPER Answer F1: **0.237785982516** (23.77859825%).
- Selected/evaluated cases: **129/129**.
- Generation errors: **0**; recorded generation errors score zero.
- Evidence F1: **Not computed.** Evidence F1 is not computed because retrieved chunks are not mapped to QASPER evidence format.

| Matched answer type | Mean Answer F1 |
| --- | --- |
| extractive | 0.213050 |
| abstractive | 0.196746 |
| boolean | 0.026518 |
| none | 0.800000 |

- Semantic mean: **0.7131782945736438 (71.32%) across 129 non-null judged cases**.
- Semantic pass rate: **0.6511627906976745 (65.12%; 84/129)** at threshold **0.70**.
- Semantic coverage: **100.00% (129/129)**.
- Full-run semantic correctness: **0.7131782945736438 (71.32%)**.
- Judge: **qwen/qwen3.8-27b**; evaluator version **qasper-reference-geval-v1.6**.
- Reference aggregation: **maximum over independently judged annotations; annotation-specific gold evidence is supplied; QASPER cross-reference placeholders are treated as non-semantic annotation markup; absence from an abbreviated reference is not itself evidence of fabrication**

| Rubric band | Count |
| --- | --- |
| 0-1 | 21 |
| 2-4 | 12 |
| 5-6 | 12 |
| 7-8 | 5 |
| 9-10 | 79 |

- Annotation evidence: **use `highlighted_evidence` when usable; otherwise use `evidence`; otherwise state that no gold evidence was supplied. Original evidence order is preserved.**
- Abbreviated-reference policy: **absence of a detail from the short reference is not itself evidence of fabrication.**

| Reference-judgment measure | Count |
| --- | --- |
| Total expected reference judgments | 229 |
| Successful reference judgments | 229 |
| Failed reference judgments | 0 |
| Cases where a non-first annotation won | 23 |

| Matched reference answer type | Cases |
| --- | --- |
| extractive | 73 |
| abstractive | 33 |
| boolean | 13 |
| none | 10 |

### Judge Reliability

| Reliability measure | Recorded value |
| --- | --- |
| Total provider attempts | 229 |
| Cases requiring at least one retry | 0 |
| Total retries | 0 |
| Rate-limit events | 0 |
| Rate-limit retries | 0 |
| Other transient retries | 0 |
| Schema/output retries | 0 |
| Pacing sleep | 1219.872 s |
| Retry/backoff sleep | 0.000 s |
| Telemetry-covered selected cases | 129 |
| Unresolved judge errors | 0 |
| Historical judge-error attempts | 0 |

Token F1 measures normalized lexical overlap against short references. Semantic correctness measures reference-based answer correctness and permits valid paraphrasing. These metrics measure different properties and must not be combined; semantic evaluation does not supersede deterministic F1.

## RAG Diagnostics

| Retrieval relevance | Count | Share |
| --- | --- | --- |
| high | 102 | 79.07% |
| medium | 10 | 7.75% |
| low | 17 | 13.18% |
| unknown | 0 | 0.00% |
| missing | 0 | 0.00% |

| Groundedness | Count | Share |
| --- | --- | --- |
| grounded | 106 | 82.17% |
| partially_grounded | 7 | 5.43% |
| not_grounded | 16 | 12.40% |
| unknown | 0 | 0.00% |
| missing | 0 | 0.00% |

Unknown means that a checker result was unavailable; it is not low relevance, not-grounded, or an incorrect answer. These checker labels are operational diagnostics, not manually validated accuracy metrics.

| Checker | Availability/cause | Count |
| --- | --- | --- |
| Relevance | valid | 129 |
| Relevance | unknown | 0 |
| Relevance | rate_limit_tpm | 0 |
| Relevance | rate_limit_tpd | 0 |
| Relevance | malformed_output | 0 |
| Relevance | timeout | 0 |
| Relevance | connection | 0 |
| Relevance | other_failure | 0 |
| Groundedness | valid | 129 |
| Groundedness | unknown | 0 |
| Groundedness | rate_limit_tpm | 0 |
| Groundedness | rate_limit_tpd | 0 |
| Groundedness | malformed_output | 0 |
| Groundedness | timeout | 0 |
| Groundedness | connection | 0 |
| Groundedness | other_failure | 0 |

TPM pacing reduces request bursts but does not increase or reset TPD allowance. TPD exhaustion cannot be repaired by semantic-judge resume; generation/checker diagnostics require sufficient daily quota, and concurrent use of the shared openai/gpt-oss-20b checker model can consume it.

## Latency and Usage

| Series (seconds) | n | Mean | Median | p95 | Maximum |
| --- | --- | --- | --- | --- | --- |
| End-to-end answer | 129 | 8.155 | 7.216 | 16.635 | 28.072 |
| Total retrieval | 129 | 6.327 | 5.266 | 15.015 | 25.736 |
| Dense retrieval | 129 | 0.226 | 0.128 | 0.594 | 4.522 |
| BM25 | 129 | 0.001 | 0.001 | 0.002 | 0.008 |
| Fusion | 129 | 0.000 | 0.000 | 0.001 | 0.009 |
| Reranking | 129 | 6.099 | 5.158 | 14.772 | 21.211 |
| Generator | 129 | 0.725 | 0.655 | 1.321 | 1.585 |
| Relevance checker | 129 | 0.668 | 0.570 | 1.250 | 2.523 |
| Groundedness checker | 129 | 0.451 | 0.431 | 0.611 | 1.327 |
| Semantic judge | 129 | 10.458 | 11.902 | 12.697 | 18.251 |

- Total recorded generation tokens (n=129): **300651**.
- Mean generation tokens per recorded case (n=129): **2330.63**.

p95 uses the deterministic nearest-rank definition. Checker calls run concurrently in the production chat path; checker latencies are not added together or presented as production wall-clock latency.

## Cases Requiring Attention

| Paper ID | Question ID | Question | Generation | QASPER F1 | Semantic | Matched reference type | Relevance | Groundedness | Issue |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1806.04511 | c7486d039304… | which non-english language was the had the worst results? | success | 0.0000 | 0.0000 | extractive | low | not_grounded | failed semantic threshold, not grounded, low retrieval relevance |
| 1806.04511 | f1f1dcc67b3e… | what datasets were used in evaluation? | success | 0.2286 | 0.5000 | extractive | high | grounded | failed semantic threshold |
| 1911.00841 | 3c3807f226ba… | Are the experts comparable to real-world users? | success | 0.0000 | 0.1000 | boolean | low | not_grounded | failed semantic threshold, not grounded, low retrieval relevance |
| 1911.00841 | c70bafc35e27… | Are the answers double (and not triple) annotated? | success | 0.0000 | 0.0000 | boolean | high | not_grounded | failed semantic threshold, not grounded |
| 1911.00841 | 81d607fc2061… | Who were the experts used for annotation? | success | 0.0000 | 0.0000 | abstractive | medium | not_grounded | failed semantic threshold, not grounded |
| 2001.02885 | 78536da059b8… | What were the baselines? | success | 0.0000 | 0.0000 | extractive | low | grounded | failed semantic threshold, low retrieval relevance |
| 2001.02885 | 9122de265577… | What were the previously reported results? | success | 0.1695 | 0.1000 | extractive | high | grounded | failed semantic threshold |
| 2002.04181 | 9b7655d39c7a… | Which toolkits do they use? | success | 0.0364 | 0.3000 | extractive | high | grounded | failed semantic threshold |
| 2002.04181 | cd06d775f491… | Which sentiment class is the most accurately predicted by ELS systems? | success | 0.0000 | 0.0000 | abstractive | low | not_grounded | failed semantic threshold, not grounded, low retrieval relevance |
| 2002.04181 | 1329280df5ee… | Is datasets for sentiment analysis balanced? | success | 0.0000 | 0.0000 | boolean | medium | not_grounded | failed semantic threshold, not grounded |
| 1810.04428 | 4625cfba3083… | by how much did their model improve? | success | 0.4528 | 0.6000 | abstractive | high | grounded | failed semantic threshold |
| 1912.08960 | 50e80cfa8420… | Are the images from a specific domain? | success | 0.0000 | 0.0000 | boolean | low | grounded | failed semantic threshold, low retrieval relevance |
| 1908.06267 | 2858620e0498… | Which component is the least impactful? | success | 0.0667 | 0.1000 | abstractive | medium | not_grounded | failed semantic threshold, not grounded |
| 1908.06267 | cb12c19f9d14… | What is the state-of-the-art system? | success | 0.0000 | 0.0000 | extractive | low | partially_grounded | failed semantic threshold, partially grounded, low retrieval relevance |
| 1908.06267 | 9193006f359c… | Which datasets are used? | success | 0.2388 | 0.5000 | extractive | high | partially_grounded | failed semantic threshold, partially grounded |
| 2002.12612 | d7644c674887… | What are the global network features which quantify different aspects of the sharing proc… | success | 0.0000 | 0.0000 | extractive | high | grounded | failed semantic threshold |
| 1811.00383 | a313e98994fc… | How do they match words before reordering them? | success | 0.0000 | 0.0000 | none | low | not_grounded | failed semantic threshold, not grounded, low retrieval relevance |
| 1811.00383 | 7e62a53823ab… | Which dataset(s) do they experiment with? | success | 0.3226 | 0.5000 | extractive | high | grounded | failed semantic threshold |
| 1908.08345 | 97d1ac71eed1… | What is novel about their document-level encoder? | success | 0.1905 | 0.2000 | extractive | high | grounded | failed semantic threshold |
| 1908.08345 | c17b609b0b09… | What rouge score do they achieve? | success | 0.2545 | 0.2000 | abstractive | high | grounded | failed semantic threshold |
| 1904.10503 | 729694a9fe1e… | What results do they achieve using their proposed approach? | success | 0.2273 | 0.4000 | abstractive | high | grounded | failed semantic threshold |
| 1904.10503 | 1c997c268c68… | How do they combine a deep learning model with a knowledge base? | success | 0.1684 | 0.5000 | abstractive | high | grounded | failed semantic threshold |
| 1612.03226 | f94b53db3076… | How do they calculate variance from the model outputs? | success | 0.1370 | 0.6000 | extractive | high | grounded | failed semantic threshold |
| 1612.03226 | 551457ed34ca… | Which dataset do they use? | success | 0.0000 | 0.0000 | extractive | high | not_grounded | failed semantic threshold, not grounded |
| 1610.08815 | fabf6fdcfb4c… | Which benchmark datasets are used? | success | 0.2759 | 0.5000 | extractive | high | not_grounded | failed semantic threshold, not grounded |
| 1910.05603 | ceb4bac1f6c5… | What is the performance reported for the best models in the VLSP 2018 and VLSP 2019 chall… | success | 0.0000 | 0.0000 | none | high | grounded | failed semantic threshold |
| 1910.05603 | 91bc8c0bc163… | Is the model tested against any baseline? | success | 0.0000 | 0.2000 | boolean | low | grounded | failed semantic threshold, low retrieval relevance |
| 1910.05603 | fe1dcd6ef1f8… | What is the language model combination technique used in the paper? | success | 0.0721 | 0.6000 | extractive | high | grounded | failed semantic threshold |
| 1911.07555 | 012b8a89aea2… | What is the approach of previous work? | success | 0.5135 | 0.6000 | extractive | high | grounded | failed semantic threshold |
| 1911.07555 | 307e8ab37b67… | Does the paper report the performance of a baseline model on South African languages LID? | success | 0.0000 | 0.0000 | boolean | low | grounded | failed semantic threshold, low retrieval relevance |
| 2003.08385 | 71ba1b09bb03… | What is the performance of the baseline? | success | 0.0656 | 0.4000 | abstractive | high | grounded | failed semantic threshold |
| 2003.08385 | 787c4d4628ea… | What annotations are present in dataset? | success | 0.0964 | 0.1000 | extractive | high | grounded | failed semantic threshold |
| 1909.00175 | badc9db40adb… | What state-of-the-art results are achieved? | success | 0.3500 | 0.2000 | abstractive | high | grounded | failed semantic threshold |
| 1909.00175 | 67b66fe67a3c… | What baselines do they compare with? | success | 0.2093 | 0.3000 | abstractive | high | grounded | failed semantic threshold |
| 1611.02988 | fa30a938b58f… | What was their performance on emotion detection? | success | 0.3117 | 0.2000 | abstractive | high | grounded | failed semantic threshold |
| 1611.02988 | de53af4eddbc… | Which Facebook pages did they look at? | success | 0.5176 | 0.6000 | extractive | high | grounded | failed semantic threshold |
| 1902.09666 | d015faf0f8dc… | What kinds of offensive content are explored? | success | 0.2000 | 0.5000 | abstractive | high | partially_grounded | failed semantic threshold, partially grounded |
| 1902.09666 | 521280a87c43… | How many annotators participated? | success | 0.0741 | 0.0000 | extractive | medium | grounded | failed semantic threshold |
| 1902.09666 | 5a8cc8f80509… | What is the definition of offensive language? | success | 0.2105 | 0.0000 | extractive | high | grounded | failed semantic threshold |
| 1902.09666 | 1b72aa2ec3ce… | How long is the dataset for each step of hierarchy? | success | 0.0000 | 0.0000 | abstractive | high | grounded | failed semantic threshold |
| 1805.07133 | 219af68afeae… | what japanese-vietnamese dataset do they use? | success | 0.0741 | 0.1000 | extractive | high | grounded | failed semantic threshold |
| 1809.03449 | d64383e39357… | What type of model is KAR? | success | 0.0000 | 0.4000 | extractive | high | grounded | failed semantic threshold |
| 1912.03804 | d859cc37799a… | How are prominent topics idenified in Dabiq and Rumiyah? | success | 0.1562 | 0.4000 | abstractive | high | grounded | failed semantic threshold |
| 1912.01214 | f5e6f4345433… | what are the pivot-based baselines? | success | 0.0800 | 0.5000 | extractive | high | grounded | failed semantic threshold |
| 1912.01214 | 5eda469a8a77… | what language pairs are explored? | success | 0.1972 | 0.2000 | extractive | high | partially_grounded | failed semantic threshold, partially grounded |
| 2001.02885 | 96b07373756d… | Does RoBERTa outperform BERT? | success | 1.0000 | 1.0000 | none | high | not_grounded | not grounded |
| 2001.02885 | 45be665a4504… | What is the size of bioScope corpus? | success | 1.0000 | 1.0000 | none | high | not_grounded | not grounded |
| 1811.00383 | 37861be6aecd… | On how many language pairs do they show that preordering assisting language sentences hel… | success | 0.2642 | 1.0000 | abstractive | medium | not_grounded | not grounded |
| 1910.05603 | 53f742509480… | What are the deep learning architectures used in the task? | success | 1.0000 | 1.0000 | none | low | not_grounded | not grounded, low retrieval relevance |
| 1911.07555 | 6415f38a06c2… | What are the languages represented in the DSL datasets? | success | 1.0000 | 1.0000 | none | high | not_grounded | not grounded |
| 1805.07133 | b9ea841b817b… | did they collect their own data? | success | 0.0455 | 1.0000 | boolean | high | not_grounded | not grounded |
| 1908.09246 | 0602a974a879… | What baseline approaches does this approach out-perform? | success | 0.0690 | 1.0000 | extractive | high | partially_grounded | partially grounded |
| 1908.06267 | 545e92833b0a… | Which component has the greatest impact on performance? | success | 0.4706 | 1.0000 | extractive | high | partially_grounded | partially grounded |
| 1911.07555 | 92dfacbbfa73… | Which languages are similar to each other? | success | 0.3099 | 1.0000 | extractive | high | partially_grounded | partially grounded |
| 2001.02885 | e86130c5b9ab… | What is the size of SFU Review corpus? | success | 1.0000 | 1.0000 | none | low | grounded | low retrieval relevance |
| 1707.06806 | 252a645af987… | What is the average length of the title text? | success | 1.0000 | 1.0000 | none | low | grounded | low retrieval relevance |
| 1707.06806 | ddb23a71113c… | What is the source of the news articles? | success | 0.4889 | 1.0000 | extractive | low | grounded | low retrieval relevance |
| 1612.03226 | aa7d327ef98f… | How much data samples do they start with before obtaining the initial model labels? | success | 0.1875 | 0.9000 | extractive | low | grounded | low retrieval relevance |
| 1911.07555 | c59802881506… | Is the lexicon the same for all languages? | success | 0.0741 | 1.0000 | boolean | low | grounded | low retrieval relevance |
| 1911.07555 | 0ab3df10f0b7… | What evaluation metric is used? | success | 0.1765 | 0.9000 | extractive | low | grounded | low retrieval relevance |
| 1912.03804 | 2ccc26e11df4… | Do they report results only on English data? | success | 0.0400 | 1.0000 | boolean | low | grounded | low retrieval relevance |

Low token F1 alone is not treated as factual incorrectness.

## Per-Question Results

| Paper | Question ID | Question | Generation | QASPER F1 | Semantic | Semantic pass | Relevance | Groundedness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1908.09246 | f3c204723da5… | Do they report results only on English data? | success | 1.0000 | 1.0000 | Yes | medium | grounded |
| 1908.09246 | 0602a974a879… | What baseline approaches does this approach out-perform? | success | 0.0690 | 1.0000 | Yes | high | partially_grounded |
| 1908.09246 | 56b034c30398… | What datasets are used? | success | 0.2264 | 1.0000 | Yes | high | grounded |
| 1908.09246 | 15e481e66811… | What alternative to Gibbs sampling is used? | success | 0.1364 | 0.7000 | Yes | high | grounded |
| 1908.09246 | 3d7a982c718e… | How does this model overcome the assumption that all words in a document are generated fr… | success | 0.1633 | 1.0000 | Yes | high | grounded |
| 1806.04511 | e79a5b6b6680… | which non-english language had the best performance? | success | 0.1250 | 1.0000 | Yes | medium | grounded |
| 1806.04511 | c7486d039304… | which non-english language was the had the worst results? | success | 0.0000 | 0.0000 | No | low | not_grounded |
| 1806.04511 | f1f1dcc67b3e… | what datasets were used in evaluation? | success | 0.2286 | 0.5000 | No | high | grounded |
| 1806.04511 | a103636c8d1d… | what are the baselines? | success | 0.2051 | 1.0000 | Yes | high | grounded |
| 1806.04511 | 55139fcfe04c… | how did the authors translate the reviews to other languages? | success | 0.3636 | 1.0000 | Yes | high | grounded |
| 1806.04511 | fbaf060004f1… | what dataset was used for training? | success | 0.3256 | 1.0000 | Yes | high | grounded |
| 1911.00841 | 3c3807f226ba… | Are the experts comparable to real-world users? | success | 0.0000 | 0.1000 | No | low | not_grounded |
| 1911.00841 | c70bafc35e27… | Are the answers double (and not triple) annotated? | success | 0.0000 | 0.0000 | No | high | not_grounded |
| 1911.00841 | 81d607fc2061… | Who were the experts used for annotation? | success | 0.0000 | 0.0000 | No | medium | not_grounded |
| 1911.00841 | 51fe4d44887c… | What type of neural model was used? | success | 0.1143 | 1.0000 | Yes | high | grounded |
| 1911.00841 | f0848e7a339d… | Were other baselines tested to compare with the neural baseline? | success | 0.1967 | 0.7000 | Yes | high | grounded |
| 2001.02885 | 78536da059b8… | What were the baselines? | success | 0.0000 | 0.0000 | No | low | grounded |
| 2001.02885 | 96b07373756d… | Does RoBERTa outperform BERT? | success | 1.0000 | 1.0000 | Yes | high | not_grounded |
| 2001.02885 | 511517efc96e… | Which multiple datasets did they train on during joint training? | success | 0.3846 | 1.0000 | Yes | high | grounded |
| 2001.02885 | 9122de265577… | What were the previously reported results? | success | 0.1695 | 0.1000 | No | high | grounded |
| 2001.02885 | e86130c5b9ab… | What is the size of SFU Review corpus? | success | 1.0000 | 1.0000 | Yes | low | grounded |
| 2001.02885 | 45be665a4504… | What is the size of bioScope corpus? | success | 1.0000 | 1.0000 | Yes | high | not_grounded |
| 2002.04181 | 08b57deb237f… | Who are the crowdworkers? | success | 0.3429 | 1.0000 | Yes | high | grounded |
| 2002.04181 | 9b7655d39c7a… | Which toolkits do they use? | success | 0.0364 | 0.3000 | No | high | grounded |
| 2002.04181 | cd06d775f491… | Which sentiment class is the most accurately predicted by ELS systems? | success | 0.0000 | 0.0000 | No | low | not_grounded |
| 2002.04181 | 1329280df5ee… | Is datasets for sentiment analysis balanced? | success | 0.0000 | 0.0000 | No | medium | not_grounded |
| 2002.04181 | 58c6737070ef… | What measures are used for evaluation? | success | 0.1702 | 1.0000 | Yes | high | grounded |
| 1810.04428 | 2c6b50877133… | what language does this paper focus on? | success | 0.2000 | 1.0000 | Yes | high | grounded |
| 1810.04428 | f651cd144b77… | what evaluation metrics did they use? | success | 0.1667 | 1.0000 | Yes | high | grounded |
| 1810.04428 | 4625cfba3083… | by how much did their model improve? | success | 0.4528 | 0.6000 | No | high | grounded |
| 1810.04428 | 326588b1de9b… | what state of the art methods did they compare with? | success | 0.0833 | 0.7000 | Yes | high | grounded |
| 1810.04428 | ebf0d9f9260e… | what are the sizes of both datasets? | success | 0.6061 | 1.0000 | Yes | high | grounded |
| 1707.06806 | 252a645af987… | What is the average length of the title text? | success | 1.0000 | 1.0000 | Yes | low | grounded |
| 1707.06806 | ed67359889cf… | Which pretrained word vectors did they use? | success | 0.4444 | 1.0000 | Yes | high | grounded |
| 1707.06806 | 425bd2ccfd95… | What evaluation metrics are used? | success | 0.1277 | 1.0000 | Yes | high | grounded |
| 1707.06806 | 955de9f7412b… | Which shallow approaches did they experiment with? | success | 0.1053 | 1.0000 | Yes | high | grounded |
| 1707.06806 | 3b371ea554fa… | Where do they obtain the news videos from? | success | 0.2000 | 1.0000 | Yes | high | grounded |
| 1707.06806 | ddb23a71113c… | What is the source of the news articles? | success | 0.4889 | 1.0000 | Yes | low | grounded |
| 1811.08603 | bb3267c3f0a1… | Which datasets do they use? | success | 0.1633 | 1.0000 | Yes | high | grounded |
| 1811.08603 | 114934e1a1e8… | How effective is their NCEL approach overall? | success | 0.1013 | 0.7000 | Yes | high | grounded |
| 1811.08603 | 2439b6b92d73… | How do they verify generalization ability? | success | 0.2000 | 1.0000 | Yes | high | grounded |
| 1811.08603 | b8d0e4e0e820… | Do they only use adjacent entity mentions or use more than that in some cases (next to ad… | success | 0.0896 | 0.9000 | Yes | high | grounded |
| 1912.08960 | 50e80cfa8420… | Are the images from a specific domain? | success | 0.0000 | 0.0000 | No | low | grounded |
| 1912.08960 | b1bc9ae9d40e… | Which datasets are used? | success | 0.1379 | 1.0000 | Yes | high | grounded |
| 1912.08960 | 63a1cbe66fd5… | Which existing models are evaluated? | success | 0.1818 | 1.0000 | Yes | high | grounded |
| 1912.08960 | 509af1f11bd6… | How is diversity measured? | success | 0.4828 | 1.0000 | Yes | high | grounded |
| 1908.06267 | 2858620e0498… | Which component is the least impactful? | success | 0.0667 | 0.1000 | No | medium | not_grounded |
| 1908.06267 | 545e92833b0a… | Which component has the greatest impact on performance? | success | 0.4706 | 1.0000 | Yes | high | partially_grounded |
| 1908.06267 | cb12c19f9d14… | What is the state-of-the-art system? | success | 0.0000 | 0.0000 | No | low | partially_grounded |
| 1908.06267 | 9193006f359c… | Which datasets are used? | success | 0.2388 | 0.5000 | No | high | partially_grounded |
| 1908.06267 | bc67b91dd73a… | What is the message passing framework? | success | 0.1905 | 1.0000 | Yes | high | grounded |
| 2002.12612 | dd20d93166c1… | Which two news domains are country-independent? | success | 0.1600 | 1.0000 | Yes | high | grounded |
| 2002.12612 | dc2a2c177cd5… | How is the political bias of different sources included in the model? | success | 0.3518 | 0.9000 | Yes | high | grounded |
| 2002.12612 | ae90c5567746… | What are the two large-scale datasets used? | success | 0.0000 | 1.0000 | Yes | high | grounded |
| 2002.12612 | d7644c674887… | What are the global network features which quantify different aspects of the sharing proc… | success | 0.0000 | 0.0000 | No | high | grounded |
| 1811.00383 | a313e98994fc… | How do they match words before reordering them? | success | 0.0000 | 0.0000 | No | low | not_grounded |
| 1811.00383 | 37861be6aecd… | On how many language pairs do they show that preordering assisting language sentences hel… | success | 0.2642 | 1.0000 | Yes | medium | not_grounded |
| 1811.00383 | 7e62a53823ab… | Which dataset(s) do they experiment with? | success | 0.3226 | 0.5000 | No | high | grounded |
| 1908.08345 | 97d1ac71eed1… | What is novel about their document-level encoder? | success | 0.1905 | 0.2000 | No | high | grounded |
| 1908.08345 | c17b609b0b09… | What rouge score do they achieve? | success | 0.2545 | 0.2000 | No | high | grounded |
| 1908.08345 | 53014cfb506f… | What are the datasets used for evaluation? | success | 0.3793 | 1.0000 | Yes | high | grounded |
| 1904.10503 | 5a65ad10ff95… | Which other approaches do they compare their model with? | success | 0.4889 | 1.0000 | Yes | high | grounded |
| 1904.10503 | 729694a9fe1e… | What results do they achieve using their proposed approach? | success | 0.2273 | 0.4000 | No | high | grounded |
| 1904.10503 | 1c997c268c68… | How do they combine a deep learning model with a knowledge base? | success | 0.1684 | 0.5000 | No | high | grounded |
| 1612.03226 | f94b53db3076… | How do they calculate variance from the model outputs? | success | 0.1370 | 0.6000 | No | high | grounded |
| 1612.03226 | aa7d327ef98f… | How much data samples do they start with before obtaining the initial model labels? | success | 0.1875 | 0.9000 | Yes | low | grounded |
| 1612.03226 | b8d7d055ddb9… | Which model do they use for end-to-end speech recognition? | success | 0.2222 | 1.0000 | Yes | high | grounded |
| 1612.03226 | 551457ed34ca… | Which dataset do they use? | success | 0.0000 | 0.0000 | No | high | not_grounded |
| 1610.08815 | 3a6e843c6c81… | What are the state of the art models? | success | 0.0000 | 1.0000 | Yes | high | grounded |
| 1610.08815 | fabf6fdcfb4c… | Which benchmark datasets are used? | success | 0.2759 | 0.5000 | No | high | not_grounded |
| 1610.08815 | 1beb4a590fa6… | What are the network's baseline features? | success | 0.0678 | 1.0000 | Yes | high | grounded |
| 1910.05603 | ceb4bac1f6c5… | What is the performance reported for the best models in the VLSP 2018 and VLSP 2019 chall… | success | 0.0000 | 0.0000 | No | high | grounded |
| 1910.05603 | 91bc8c0bc163… | Is the model tested against any baseline? | success | 0.0000 | 0.2000 | No | low | grounded |
| 1910.05603 | fe1dcd6ef1f8… | What is the language model combination technique used in the paper? | success | 0.0721 | 0.6000 | No | high | grounded |
| 1910.05603 | 53f742509480… | What are the deep learning architectures used in the task? | success | 1.0000 | 1.0000 | Yes | low | not_grounded |
| 1702.03856 | 8acab64ba728… | what is the domain of the corpus? | success | 0.1481 | 1.0000 | Yes | high | grounded |
| 1702.03856 | 53aa07cc4cc4… | what challenges are identified? | success | 0.1882 | 1.0000 | Yes | high | grounded |
| 1702.03856 | 72755c2d7921… | what is the size of the speech corpus? | success | 0.5000 | 1.0000 | Yes | high | grounded |
| 1911.07555 | 012b8a89aea2… | What is the approach of previous work? | success | 0.5135 | 0.6000 | No | high | grounded |
| 1911.07555 | c59802881506… | Is the lexicon the same for all languages? | success | 0.0741 | 1.0000 | Yes | low | grounded |
| 1911.07555 | ca4daafdc23f… | How do they obtain the lexicon? | success | 0.4727 | 1.0000 | Yes | medium | grounded |
| 1911.07555 | 0ab3df10f0b7… | What evaluation metric is used? | success | 0.1765 | 0.9000 | Yes | low | grounded |
| 1911.07555 | 92dfacbbfa73… | Which languages are similar to each other? | success | 0.3099 | 1.0000 | Yes | high | partially_grounded |
| 1911.07555 | c8541ff10c4e… | Which datasets are employed for South African languages LID? | success | 0.1739 | 0.9000 | Yes | high | grounded |
| 1911.07555 | 307e8ab37b67… | Does the paper report the performance of a baseline model on South African languages LID? | success | 0.0000 | 0.0000 | No | low | grounded |
| 1911.07555 | 6415f38a06c2… | What are the languages represented in the DSL datasets? | success | 1.0000 | 1.0000 | Yes | high | not_grounded |
| 1911.07555 | e5c8e9e54e77… | Does the algorithm improve on the state-of-the-art methods? | success | 0.1892 | 1.0000 | Yes | medium | grounded |
| 1611.01576 | 2f901dab6b75… | What languages pairs are used in machine translation? | success | 0.0606 | 1.0000 | Yes | high | grounded |
| 1611.01576 | b591853e9389… | What sentiment classification dataset is used? | success | 0.2500 | 1.0000 | Yes | high | grounded |
| 1611.01576 | a130306c6662… | What pooling function is used? | success | 0.0435 | 1.0000 | Yes | high | grounded |
| 2003.08385 | 35b3ce3a7499… | Does the paper report the performance of the model for each individual language? | success | 1.0000 | 1.0000 | Yes | high | grounded |
| 2003.08385 | 71ba1b09bb03… | What is the performance of the baseline? | success | 0.0656 | 0.4000 | No | high | grounded |
| 2003.08385 | 612c3675b6c5… | Did they pefrorm any cross-lingual vs single language evaluation? | success | 0.0426 | 0.9000 | Yes | high | grounded |
| 2003.08385 | bd40f33452da… | What was the performance of multilingual BERT? | success | 0.3226 | 1.0000 | Yes | high | grounded |
| 2003.08385 | 787c4d4628ea… | What annotations are present in dataset? | success | 0.0964 | 0.1000 | No | high | grounded |
| 1909.00175 | badc9db40adb… | What state-of-the-art results are achieved? | success | 0.3500 | 0.2000 | No | high | grounded |
| 1909.00175 | 67b66fe67a3c… | What baselines do they compare with? | success | 0.2093 | 0.3000 | No | high | grounded |
| 1909.00175 | f56d07f73b31… | What datasets are used in evaluation? | success | 0.6557 | 1.0000 | Yes | high | grounded |
| 1909.00175 | 38e4aaeabf06… | What is the tagging scheme employed? | success | 0.2338 | 1.0000 | Yes | high | grounded |
| 1611.02988 | fa30a938b58f… | What was their performance on emotion detection? | success | 0.3117 | 0.2000 | No | high | grounded |
| 1611.02988 | f875337f2ecd… | Which existing benchmarks did they compare to? | success | 0.3556 | 1.0000 | Yes | high | grounded |
| 1611.02988 | de53af4eddbc… | Which Facebook pages did they look at? | success | 0.5176 | 0.6000 | No | high | grounded |
| 1902.09666 | 8c852fc29bda… | What models are used in the experiment? | success | 0.3600 | 1.0000 | Yes | high | grounded |
| 1902.09666 | 682e26262abb… | What are the differences between this dataset and pre-existing ones? | success | 0.0460 | 0.9000 | Yes | high | grounded |
| 1902.09666 | 5daeb8d4d6f3… | In what language are the tweets? | success | 0.0909 | 1.0000 | Yes | high | grounded |
| 1902.09666 | 74fb77a624ea… | What is the size of the new dataset? | success | 0.3333 | 1.0000 | Yes | high | grounded |
| 1902.09666 | d015faf0f8dc… | What kinds of offensive content are explored? | success | 0.2000 | 0.5000 | No | high | partially_grounded |
| 1902.09666 | 55bd59076a49… | What is the best performing model? | success | 0.1000 | 1.0000 | Yes | high | grounded |
| 1902.09666 | 521280a87c43… | How many annotators participated? | success | 0.0741 | 0.0000 | No | medium | grounded |
| 1902.09666 | 5a8cc8f80509… | What is the definition of offensive language? | success | 0.2105 | 0.0000 | No | high | grounded |
| 1902.09666 | 290ee79b5e38… | What are the three layers of the annotation scheme? | success | 0.4194 | 1.0000 | Yes | high | grounded |
| 1902.09666 | 1b72aa2ec3ce… | How long is the dataset for each step of hierarchy? | success | 0.0000 | 0.0000 | No | high | grounded |
| 1805.07133 | 1269c5d8f61e… | what methods were used to reduce data sparsity effects? | success | 0.0625 | 0.9000 | Yes | high | grounded |
| 1805.07133 | e35a7f9513ff… | what was the baseline? | success | 0.1714 | 1.0000 | Yes | high | grounded |
| 1805.07133 | b9ea841b817b… | did they collect their own data? | success | 0.0455 | 1.0000 | Yes | high | not_grounded |
| 1805.07133 | 219af68afeae… | what japanese-vietnamese dataset do they use? | success | 0.0741 | 0.1000 | No | high | grounded |
| 1809.03449 | 8bb0011ad1d6… | Do they report results only on English datasets? | success | 0.0769 | 1.0000 | Yes | high | grounded |
| 1809.03449 | b0dbe7504731… | How do the authors examine whether a model is robust to noise or not? | success | 0.3168 | 1.0000 | Yes | high | grounded |
| 1809.03449 | d64383e39357… | What type of model is KAR? | success | 0.0000 | 0.4000 | No | high | grounded |
| 1809.03449 | 52f9cd05d831… | Do the authors hypothesize that humans' robustness to noise is due to their general knowl… | success | 0.0392 | 1.0000 | Yes | medium | grounded |
| 1912.03804 | 2ccc26e11df4… | Do they report results only on English data? | success | 0.0400 | 1.0000 | Yes | low | grounded |
| 1912.03804 | f318a2851d70… | What conclusions do the authors draw from their finding that the emotional appeal of ISIS… | success | 0.2247 | 0.9000 | Yes | high | grounded |
| 1912.03804 | 6bbbb9933aab… | How id Depechemood trained? | success | 0.4031 | 1.0000 | Yes | high | grounded |
| 1912.03804 | 2007bfb8f66e… | How are similarities and differences between the texts from violent and non-violent relig… | success | 0.1944 | 0.7000 | Yes | high | grounded |
| 1912.03804 | d859cc37799a… | How are prominent topics idenified in Dabiq and Rumiyah? | success | 0.1562 | 0.4000 | No | high | grounded |
| 1912.01214 | b6f15fb6279b… | which multilingual approaches do they compare with? | success | 0.1538 | 1.0000 | Yes | high | grounded |
| 1912.01214 | f5e6f4345433… | what are the pivot-based baselines? | success | 0.0800 | 0.5000 | No | high | grounded |
| 1912.01214 | 9a05a5f4351d… | which datasets did they experiment with? | success | 0.1176 | 1.0000 | Yes | high | grounded |
| 1912.01214 | 5eda469a8a77… | what language pairs are explored? | success | 0.1972 | 0.2000 | No | high | partially_grounded |

## Methodology and Limitations

- This is a single-document evaluation whose questions are independent and begin with empty chat history.
- QASPER Answer F1 takes the maximum normalized token F1 over valid human annotations.
- Explanatory answers can receive low token F1 against short gold spans.
- DeepEval is an LLM-as-a-judge measurement and remains nondeterministic.
- Semantic reference aggregation: maximum over independently judged annotations; annotation-specific gold evidence is supplied; QASPER cross-reference placeholders are treated as non-semantic annotation markup; absence from an abbreviated reference is not itself evidence of fabrication
- Checker labels are diagnostic and were not manually calibrated for accuracy.
- Evidence F1 is unavailable because retrieved chunks are not mapped to QASPER paragraph evidence.
- The selected corpus is a custom subset, so results are not directly comparable with the official full-split QASPER baseline.
- Results apply only to the recorded dataset, configuration, and model versions.
- No prior-score failure-subset selection was applied.
- Incomplete required generation or semantic-evaluation coverage blocks publication; incomplete auxiliary checker diagnostics are disclosed as warnings.

## Reproducibility

| Provenance | Recorded value |
| --- | --- |
| Profile name | qasper-28p-publication-v1 |
| Configuration SHA-256 | e611b6f1d6606a6aed01d8624d9437380ab87a865af5a7ddf80db1c5a90e5ebb |
| Dataset SHA-256 | ca391c629a520ad9aaba61c5373f8fa54729e10bae7001f451bfc730d1b24853 |
| Recorded evaluation-workspace Git identifier | 45eac4a1bcef060ad24e409b9b69d38aca8d1e5b |
| Git dirty | No |
| Generation version | Not recorded |
| QASPER evaluator version | Not recorded in the canonical run; the curated metrics copy links the audited upstream evaluator commit |
| Semantic evaluator version | qasper-reference-geval-v1.6 |

Source artifacts:

Application logs are retained with the canonical local run but are not included
in this public bundle because they contain substantial document-text previews
captured during retrieval. Their validated coverage and aggregate diagnostics
remain reported above.

The public `answers.jsonl`, `qasper_scores.jsonl`, and semantic `scores.jsonl`
copies omit gold answers, evidence excerpts, duplicated reference text, judge
reasons, and all 109 substantive generated answers whose evidence blocks
reproduced source text. Twenty formulaic abstentions remain. Scores, statuses,
identifiers, timing, and recorded aggregate results are preserved. See the
[redaction manifest](redaction_manifest.json) for exact fields, counts, and
before/after SHA-256 hashes and the overlap-audit method. The canonical local
run and its generated report were not modified; this report is a reviewed
public derivative. The redacted JSONL files are not complete inputs for
reference-based rescoring. See the
[benchmarking workflow](../../../docs/benchmarking.md#canonical-isolated-stage-workflow)
for reconstruction and execution.

- Generation Manifest: [`generation_manifest.json`](generation_manifest.json)
- Answers (redacted public copy): [`answers.jsonl`](answers.jsonl)
- QASPER Metrics: [`qasper_metrics.json`](qasper_metrics.json)
- QASPER Scores (redacted public copy): [`qasper_scores.jsonl`](qasper_scores.jsonl)
- Pipeline Manifest: [`pipeline_manifest.json`](pipeline_manifest.json)
- Pipeline Summary: [`pipeline_summary.json`](pipeline_summary.json)
- Judge Manifest: [`judge_manifest.json`](deepeval/qwen3.8-27b-publication-v1/judge_manifest.json)
- DeepEval Metrics: [`metrics.json`](deepeval/qwen3.8-27b-publication-v1/metrics.json)
- DeepEval Scores (redacted public copy): [`scores.jsonl`](deepeval/qwen3.8-27b-publication-v1/scores.jsonl)

---

**Measured RAG**
Developed by **Filipe Braiman Carvalho** · [GitHub](https://github.com/filipe-braiman) · [LinkedIn](https://linkedin.com/in/filipe-b-carvalho) · [Email](mailto:filipebraiman@gmail.com)
