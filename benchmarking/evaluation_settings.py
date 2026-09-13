"""Editable pre-run settings for the QASPER evaluation framework."""

from benchmarking.evaluation_config import EvaluationConfig


EVALUATION_CONFIG = EvaluationConfig(
    profile_name="qasper-28p-publication-v1",
    profile_schema_version=1,
    qa_data_path="benchmarking/data/qasper_qas_seed1001_size28.json",
    pdf_directory="benchmarking/data/qasper_papers_seed1001_size28",
    retrieval_mode="auto",
    chunk_size=600,
    chunk_overlap=50,
    retrieval_top_n=30,
    fusion_threshold=0.15,
    rerank_candidate_count=30,
    rerank_top_k=10,
    judge_model="qwen/qwen3.8-27b",
    judge_threshold=0.70,
    judge_reasoning_effort="default",
    judge_max_retries=3,
    judge_max_completion_tokens=2048,
    judge_request_interval_seconds=6.0,
    judge_retry_base_delay_seconds=2.0,
    judge_retry_max_delay_seconds=60.0,
    judge_retry_jitter_seconds=0.5,
    judge_max_retry_after_seconds=300.0,
    generation_case_interval_seconds=30.0,
)
