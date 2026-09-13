"""Controlled, paired comparison of production RAG checker models with the Deepeval Framework."""

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace

from benchmarking.artifact_io import atomic_write_json
from config import (
    EMBEDDING_MODEL_NAME,
    GENERATOR_MODEL,
    GROUNDEDNESS_CHECKER,
    RELEVANCE_CHECKER,
    RERANK_MODEL_NAME,
    REWRITER_MODEL,
)
from core.models import (
    GROUNDEDNESS_CHECKER_REQUEST_SETTINGS,
    RELEVANCE_CHECKER_REQUEST_SETTINGS,
    generation_request_settings,
    groundedness_checker_request_settings,
    relevance_checker_request_settings,
    rewriter_request_settings,
)
from benchmarking.evaluation_config import (
    PATH_BASE,
    load_evaluation_config,
    profile_provenance,
    sanitize_error,
    serialize_artifact_path,
    validate_profile_provenance,
)
from benchmarking.generate_answers import (
    load_qasper_records,
    select_cases,
    validate_kb_state,
)
from generation.groundedness import check_groundedness
from retrieval.formatting import build_structured_chunk_text
from retrieval.relevance import compute_retrieval_relevance


EVAL_DIR = Path(__file__).resolve().parent
RUNS_DIR = EVAL_DIR / "runs"
DEFAULT_MODELS = ("openai/gpt-oss-20b", "qwen/qwen3.8-27b")
COMPARISON_VERSION = "controlled-checker-comparison-v2"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
RELEVANCE_LABELS = ("high", "medium", "low")
GROUNDEDNESS_LABELS = (
    "grounded",
    "partially_grounded",
    "not_grounded",
)
PORTABLE_METADATA_FIELDS = (
    "doc_id",
    "source_name",
    "filename",
    "chunk_id",
    "page",
    "file_type",
    "doc_type",
    "section_header",
    "fusion_score",
    "fingerprint",
)


COMPARISON_CHECKER_REQUESTS = {
    "openai/gpt-oss-20b": {
        "relevance": {
            "model": "openai/gpt-oss-20b",
            **RELEVANCE_CHECKER_REQUEST_SETTINGS,
        },
        "groundedness": {
            "model": "openai/gpt-oss-20b",
            **GROUNDEDNESS_CHECKER_REQUEST_SETTINGS,
        },
    },
    "qwen/qwen3.8-27b": {
        "relevance": {
            "model": "qwen/qwen3.8-27b",
            **RELEVANCE_CHECKER_REQUEST_SETTINGS,
            "reasoning_effort": "none",
        },
        "groundedness": {
            "model": "qwen/qwen3.8-27b",
            **GROUNDEDNESS_CHECKER_REQUEST_SETTINGS,
            "reasoning_effort": "none",
        },
    },
}


def comparison_checker_request_settings(model, checker_role):
    try:
        settings = COMPARISON_CHECKER_REQUESTS[model][checker_role]
    except KeyError as exc:
        supported = ", ".join(sorted(COMPARISON_CHECKER_REQUESTS))
        raise ValueError(
            f"Unsupported comparison checker configuration: model={model!r}, "
            f"role={checker_role!r}. Supported models: {supported}."
        ) from exc
    return dict(settings)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def case_key(paper_id, question_id):
    return str(paper_id), str(question_id)


def result_key(record):
    return (
        str(record.get("paper_id")),
        str(record.get("question_id")),
        str(record.get("model_id")),
    )


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def checker_context_digest(relevance_context, groundedness_context):
    payload = json.dumps(
        {
            "relevance": relevance_context,
            "groundedness": groundedness_context,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(payload)


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def read_jsonl(path, label):
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed {label} JSONL at line {line_number}."
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"{label} line {line_number} must be an object."
                )
            records.append(record)
    return records


def model_order(models, case_index):
    return list(models) if case_index % 2 == 0 else list(reversed(models))


def _is_absolute_path_text(value):
    return (
        isinstance(value, str)
        and (
            PureWindowsPath(value).is_absolute()
            or PurePosixPath(value).is_absolute()
        )
    )


def portable_document_metadata(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("Document metadata must be an object.")
    portable = {}
    for field in PORTABLE_METADATA_FIELDS:
        if field not in metadata:
            continue
        value = metadata[field]
        if field == "filename" and isinstance(value, str):
            value = value.replace("\\", "/").rsplit("/", 1)[-1]
        if _is_absolute_path_text(value):
            raise ValueError(
                f"Document metadata field {field} contains an absolute path."
            )
        if value is None or isinstance(value, (str, int, float, bool)):
            portable[field] = value
    return portable


def serialize_reranked_documents(documents):
    serialized = []
    for document in documents:
        portable_metadata = portable_document_metadata(document.metadata)
        restored = SimpleNamespace(
            page_content=document.page_content,
            metadata=portable_metadata,
        )
        structured_text = build_structured_chunk_text(document)
        if build_structured_chunk_text(restored) != structured_text:
            raise ValueError(
                "Portable document metadata cannot reconstruct the exact "
                "checker context."
            )
        serialized.append(
            {
                "page_content": document.page_content,
                "metadata": portable_metadata,
                "structured_text": structured_text,
            }
        )
    return serialized


def restore_reranked_documents(input_record):
    documents = []
    for item in input_record.get("reranked_documents", []):
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("page_content"), str)
            or not isinstance(item.get("metadata"), dict)
        ):
            raise ValueError("Stored checker input has invalid documents.")
        portable_metadata = portable_document_metadata(item["metadata"])
        if portable_metadata != item["metadata"]:
            raise ValueError("Stored checker input has non-portable metadata.")
        document = SimpleNamespace(
            page_content=item["page_content"],
            metadata=portable_metadata,
        )
        if build_structured_chunk_text(document) != item.get("structured_text"):
            raise ValueError("Stored checker context cannot be reconstructed exactly.")
        documents.append(document)
    return documents


def relevance_context_for_documents(documents):
    return "\n\n".join(
        f"[Context {index}]\n{build_structured_chunk_text(document)}"
        for index, document in enumerate(documents, start=1)
    )


def validate_input_record(record):
    for field in ("paper_id", "question_id", "question"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"Checker input field {field} is invalid.")
    if not isinstance(record.get("gold_answers"), dict):
        raise ValueError("Checker input gold_answers must be an object.")
    required_strings = (
        "paper_id",
        "question_id",
        "question",
        "rewritten_query",
        "checker_context",
        "relevance_checker_context",
        "generated_answer",
        "context_digest",
        "answer_digest",
    )
    if record.get("status") == "success":
        for field in required_strings:
            if not isinstance(record.get(field), str):
                raise ValueError(f"Checker input field {field} must be a string.")
        documents = restore_reranked_documents(record)
        relevance_context = relevance_context_for_documents(documents)
        groundedness_context = record["checker_context"]
        if relevance_context != record["relevance_checker_context"]:
            raise ValueError("Stored relevance checker context has changed.")
        if checker_context_digest(
            relevance_context,
            groundedness_context,
        ) != record["context_digest"]:
            raise ValueError("Stored checker context digest does not match.")
        if sha256_text(record["generated_answer"]) != record["answer_digest"]:
            raise ValueError("Stored generated-answer digest does not match.")
    elif record.get("status") != "input_error":
        raise ValueError("Checker input has an invalid status.")
    return record


def validate_result_record(record, input_by_key, selected_keys, models):
    key = result_key(record)
    if key[:2] not in selected_keys or key[2] not in models:
        raise ValueError(f"Checker result does not belong to this run: {key}.")
    input_record = input_by_key.get(key[:2])
    if input_record is None:
        raise ValueError(f"Checker result has no corresponding input: {key}.")
    if record.get("status") not in {"success", "error", "input_error"}:
        raise ValueError(f"Checker result has an invalid status: {key}.")
    if record.get("context_digest") != input_record.get("context_digest"):
        raise ValueError(f"Checker result context digest mismatch: {key}.")
    if record.get("answer_digest") != input_record.get("answer_digest"):
        raise ValueError(f"Checker result answer digest mismatch: {key}.")
    expected = {
        "relevance": comparison_checker_request_settings(
            key[2], "relevance"
        ),
        "groundedness": comparison_checker_request_settings(
            key[2], "groundedness"
        ),
    }
    if record.get("checker_requests") != expected:
        raise ValueError(f"Checker result request provenance mismatch: {key}.")
    return record


def prepare_checker_input(question, kb_state, profile):
    from generation.answer_generator import generate_answer
    from generation.query_rewriter import rewrite_query
    from ingestion.indexing import get_active_doc_ids
    from retrieval.bm25 import get_bm25_scores_filtered
    from retrieval.dense import get_dense_results_filtered
    from retrieval.fusion import hybrid_fusion
    from retrieval.reranking import rerank_results
    from retrieval.routing import route_retrieval_query

    rewritten_query = rewrite_query(question, [])
    active_doc_ids = get_active_doc_ids(kb_state, "all", [])
    retrieval_route, fusion_alpha = route_retrieval_query(
        rewritten_query,
        profile.retrieval_mode,
    )
    dense_results = get_dense_results_filtered(
        vectordb=kb_state["vectordb"],
        query=rewritten_query,
        allowed_doc_ids=active_doc_ids,
        top_n=profile.retrieval_top_n,
    )
    bm25_scores = get_bm25_scores_filtered(
        kb_state=kb_state,
        query=rewritten_query,
        allowed_doc_ids=active_doc_ids,
        top_n=profile.retrieval_top_n,
    )
    chunk_lookup = {
        document.metadata.get("chunk_id"): document
        for document in kb_state.get("bm25_chunks", [])
        if document.metadata.get("doc_id") in active_doc_ids
    }
    filtered_docs = hybrid_fusion(
        dense_results=dense_results,
        bm25_scores=bm25_scores,
        chunk_lookup=chunk_lookup,
        fusion_alpha=fusion_alpha,
        score_threshold=profile.fusion_threshold,
    )
    reranked_docs, _, _ = rerank_results(
        filtered_docs=filtered_docs[: profile.rerank_candidate_count],
        rerank_model=kb_state.get("rerank_model"),
        rewritten_query=rewritten_query,
        top_k=profile.rerank_top_k,
    )
    answer, context, _, _, _ = generate_answer(
        question=question,
        chat_history=[],
        reranked_docs=reranked_docs,
    )
    serialized_documents = serialize_reranked_documents(reranked_docs)
    reconstructed = [item["structured_text"] for item in serialized_documents]
    if context != "\n\n".join(reconstructed):
        raise RuntimeError("Generated and stored checker contexts differ.")
    return {
        "rewritten_query": rewritten_query,
        "retrieval_route": retrieval_route,
        "fusion_alpha": fusion_alpha,
        "reranked_documents": serialized_documents,
        "relevance_checker_context": relevance_context_for_documents(
            reranked_docs
        ),
        "checker_context": context,
        "generated_answer": answer,
    }


def _safe_relevance(input_record, documents, settings):
    try:
        result = compute_retrieval_relevance(
            input_record["rewritten_query"],
            documents,
            settings_override=settings,
        )
        if not isinstance(result, (tuple, list)) or len(result) != 4:
            raise ValueError("Malformed relevance checker result shape.")
        label, score, explanation, latency = result
        valid = (
            label in RELEVANCE_LABELS
            and isinstance(score, (int, float))
            and not isinstance(score, bool)
            and math.isfinite(score)
            and 0.0 <= score <= 1.0
            and isinstance(explanation, str)
            and bool(explanation.strip())
            and (
                latency is None
                or (
                    isinstance(latency, (int, float))
                    and not isinstance(latency, bool)
                    and math.isfinite(latency)
                    and latency >= 0
                )
            )
        )
        if not valid:
            return label if isinstance(label, str) else "unknown", None, (
                explanation if isinstance(explanation, str) else ""
            ), latency if isinstance(latency, (int, float)) else None, (
                sanitize_error(explanation)
                if isinstance(explanation, str) and explanation.strip()
                else "Relevance checker returned an unknown or malformed result."
            )
        return label, float(score), explanation.strip(), latency, None
    except Exception as exc:
        return "unknown", None, "", None, sanitize_error(exc)


def _safe_groundedness(input_record, settings):
    try:
        result = check_groundedness(
            input_record["question"],
            input_record["generated_answer"],
            input_record["checker_context"],
            settings_override=settings,
        )
        if not isinstance(result, (tuple, list)) or len(result) != 3:
            raise ValueError("Malformed groundedness checker result shape.")
        label, explanation, latency = result
        valid = (
            label in GROUNDEDNESS_LABELS
            and isinstance(explanation, str)
            and bool(explanation.strip())
            and (
                latency is None
                or (
                    isinstance(latency, (int, float))
                    and not isinstance(latency, bool)
                    and math.isfinite(latency)
                    and latency >= 0
                )
            )
        )
        if not valid:
            return label if isinstance(label, str) else "unknown", (
                explanation if isinstance(explanation, str) else ""
            ), latency if isinstance(latency, (int, float)) else None, (
                sanitize_error(explanation)
                if isinstance(explanation, str) and explanation.strip()
                else "Groundedness checker returned an unknown or malformed result."
            )
        return label, explanation.strip(), latency, None
    except Exception as exc:
        return "unknown", "", None, sanitize_error(exc)


def make_model_result(input_record, model, order_position):
    relevance_settings = comparison_checker_request_settings(
        model, "relevance"
    )
    groundedness_settings = comparison_checker_request_settings(
        model, "groundedness"
    )
    base = {
        "paper_id": input_record["paper_id"],
        "question_id": input_record["question_id"],
        "model_id": model,
        "evaluation_order_position": order_position,
        "context_digest": input_record.get("context_digest"),
        "answer_digest": input_record.get("answer_digest"),
        "checker_requests": {
            "relevance": relevance_settings,
            "groundedness": groundedness_settings,
        },
    }
    if input_record["status"] != "success":
        return {
            **base,
            "status": "input_error",
            "error": input_record.get("error") or "Checker input unavailable.",
            "relevance_label": "unknown",
            "relevance_rank_weighted_score": None,
            "relevance_explanation": "",
            "relevance_latency_seconds": None,
            "groundedness_label": "unknown",
            "groundedness_explanation": "",
            "groundedness_latency_seconds": None,
        }

    documents = restore_reranked_documents(input_record)
    relevance = _safe_relevance(
        input_record, documents, relevance_settings
    )
    groundedness = _safe_groundedness(
        input_record, groundedness_settings
    )
    errors = [error for error in (relevance[4], groundedness[3]) if error]
    return {
        **base,
        "status": "success" if not errors else "error",
        "error": "; ".join(errors) if errors else None,
        "relevance_label": relevance[0],
        "relevance_rank_weighted_score": relevance[1],
        "relevance_explanation": relevance[2],
        "relevance_latency_seconds": relevance[3],
        "groundedness_label": groundedness[0],
        "groundedness_explanation": groundedness[1],
        "groundedness_latency_seconds": groundedness[2],
    }


def latency_summary(values):
    usable = [
        float(value)
        for value in values
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    ]
    if not usable:
        return {"count": 0, "median": None, "mean": None, "maximum": None}
    return {
        "count": len(usable),
        "median": statistics.median(usable),
        "mean": statistics.fmean(usable),
        "maximum": max(usable),
    }


def _label_distribution(records, field, labels, selected_count):
    counts = Counter(record.get(field, "unknown") for record in records)
    distribution = {label: counts.get(label, 0) for label in labels}
    distribution["unknown"] = sum(
        count for label, count in counts.items() if label not in labels
    )
    distribution["missing"] = max(0, selected_count - len(records))
    return distribution


def _pairwise_metrics(selected_keys, results_by_key, models, field, labels):
    agreements = 0
    eligible = 0
    disagreements = []
    for paper_id, question_id in selected_keys:
        left = results_by_key.get((paper_id, question_id, models[0]))
        right = results_by_key.get((paper_id, question_id, models[1]))
        if not left or not right:
            continue
        left_label = left.get(field)
        right_label = right.get(field)
        if left_label not in labels or right_label not in labels:
            continue
        eligible += 1
        if left_label == right_label:
            agreements += 1
        else:
            disagreements.append(
                {
                    "paper_id": paper_id,
                    "question_id": question_id,
                    models[0]: left_label,
                    models[1]: right_label,
                }
            )
    return {
        "eligible_case_count": eligible,
        "agreement_count": agreements,
        "agreement_rate": agreements / eligible if eligible else None,
        "disagreement_case_ids": disagreements,
    }


def build_metrics(run_id, models, selected_keys, results, results_path):
    selected_count = len(selected_keys)
    results_by_key = {result_key(record): record for record in results}
    per_model = {}
    for model in models:
        model_records = [record for record in results if record.get("model_id") == model]
        relevance_valid = sum(
            record.get("relevance_label") in RELEVANCE_LABELS
            and isinstance(record.get("relevance_rank_weighted_score"), (int, float))
            and not isinstance(record.get("relevance_rank_weighted_score"), bool)
            for record in model_records
        )
        groundedness_valid = sum(
            record.get("groundedness_label") in GROUNDEDNESS_LABELS
            for record in model_records
        )
        combined_latencies = [
            sum(
                value
                for value in (
                    record.get("relevance_latency_seconds"),
                    record.get("groundedness_latency_seconds"),
                )
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            )
            for record in model_records
            if any(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in (
                    record.get("relevance_latency_seconds"),
                    record.get("groundedness_latency_seconds"),
                )
            )
        ]
        per_model[model] = {
            "expected_case_count": selected_count,
            "result_count": len(model_records),
            "result_coverage": len(model_records) / selected_count if selected_count else 0.0,
            "successful_result_count": sum(
                record.get("status") == "success" for record in model_records
            ),
            "error_result_count": sum(
                record.get("status") != "success" for record in model_records
            ),
            "relevance_valid_result_count": relevance_valid,
            "relevance_valid_result_rate": relevance_valid / selected_count if selected_count else 0.0,
            "groundedness_valid_result_count": groundedness_valid,
            "groundedness_valid_result_rate": groundedness_valid / selected_count if selected_count else 0.0,
            "relevance_label_distribution": _label_distribution(
                model_records, "relevance_label", RELEVANCE_LABELS, selected_count
            ),
            "groundedness_label_distribution": _label_distribution(
                model_records, "groundedness_label", GROUNDEDNESS_LABELS, selected_count
            ),
            "latency_seconds": {
                "relevance": latency_summary(
                    record.get("relevance_latency_seconds") for record in model_records
                ),
                "groundedness": latency_summary(
                    record.get("groundedness_latency_seconds") for record in model_records
                ),
                "combined": latency_summary(combined_latencies),
            },
        }
    relevance_pairwise = _pairwise_metrics(
        selected_keys, results_by_key, models, "relevance_label", RELEVANCE_LABELS
    )
    groundedness_pairwise = _pairwise_metrics(
        selected_keys,
        results_by_key,
        models,
        "groundedness_label",
        GROUNDEDNESS_LABELS,
    )
    disagreement_ids = sorted(
        {
            (item["paper_id"], item["question_id"])
            for item in (
                relevance_pairwise["disagreement_case_ids"]
                + groundedness_pairwise["disagreement_case_ids"]
            )
        }
    )
    return {
        "run_id": run_id,
        "comparison_version": COMPARISON_VERSION,
        "metric_name": "Controlled production-checker comparison",
        "path_base": PATH_BASE,
        "models": list(models),
        "selected_case_count": selected_count,
        "expected_model_case_result_count": selected_count * len(models),
        "recorded_model_case_result_count": len(results),
        "per_model": per_model,
        "pairwise_relevance_agreement": relevance_pairwise,
        "pairwise_groundedness_agreement": groundedness_pairwise,
        "disagreement_case_ids": [
            {"paper_id": paper_id, "question_id": question_id}
            for paper_id, question_id in disagreement_ids
        ],
        "manual_labels": {
            "available": False,
            "labels_path": None,
            "accuracy": None,
        },
        "accuracy": None,
        "interpretation_note": (
            "Model agreement measures consistency, not checker accuracy. "
            "Accuracy remains null until optional manual labels exist."
        ),
        "source_results_path": serialize_artifact_path(results_path),
        "evaluation_timestamp": utc_now(),
    }


def production_model_request_provenance():
    return {
        "generator": generation_request_settings(),
        "rewriter": rewriter_request_settings(),
        "relevance_checker": relevance_checker_request_settings(),
        "groundedness_checker": groundedness_checker_request_settings(),
    }


def make_manifest(args, run_id, comparison_dir, profile, models, selected_count):
    return {
        "run_id": run_id,
        "comparison_version": COMPARISON_VERSION,
        "stage": "controlled_checker_comparison",
        "status": "running",
        "started_at": utc_now(),
        "finished_at": None,
        "path_base": PATH_BASE,
        "evaluation_profile": profile_provenance(profile),
        "models": {
            "embedding": EMBEDDING_MODEL_NAME,
            "reranker": RERANK_MODEL_NAME,
            "rewriter": REWRITER_MODEL,
            "generator": GENERATOR_MODEL,
            "relevance_checker": RELEVANCE_CHECKER,
            "groundedness_checker": GROUNDEDNESS_CHECKER,
            "compared_checkers": list(models),
        },
        "production_model_requests": production_model_request_provenance(),
        "checker_requests": {
            model: {
                "relevance": comparison_checker_request_settings(
                    model, "relevance"
                ),
                "groundedness": comparison_checker_request_settings(
                    model, "groundedness"
                ),
            }
            for model in models
        },
        "operational": {
            "max_papers": args.max_papers,
            "max_questions": args.max_questions,
            "alternating_model_order": True,
            "parallel_checker_models": False,
            "independent_question_history": True,
        },
        "paths": {
            "inputs": serialize_artifact_path(comparison_dir / "inputs.jsonl"),
            "results": serialize_artifact_path(comparison_dir / "results.jsonl"),
            "metrics": serialize_artifact_path(comparison_dir / "metrics.json"),
            "manifest": serialize_artifact_path(
                comparison_dir / "comparison_manifest.json"
            ),
        },
        "counts": {
            "selected_cases": selected_count,
            "input_records": 0,
            "successful_inputs": 0,
            "input_errors": 0,
            "result_records": 0,
            "successful_results": 0,
            "result_errors": 0,
        },
    }


def validate_resume_manifest(manifest, args, run_id, profile, models, inputs_path):
    if manifest.get("run_id") != run_id:
        raise ValueError("Comparison resume run ID does not match.")
    if manifest.get("comparison_version") != COMPARISON_VERSION:
        raise ValueError("Comparison runner version does not match.")
    expected_operational = {
        "max_papers": args.max_papers,
        "max_questions": args.max_questions,
    }
    recorded = manifest.get("operational", {})
    if {key: recorded.get(key) for key in expected_operational} != expected_operational:
        raise ValueError("Comparison resume limits do not match the original run.")
    compared = manifest.get("models", {}).get("compared_checkers")
    if compared != list(models):
        raise ValueError("Comparison resume models do not match the original run.")
    expected_requests = {
        model: {
            "relevance": comparison_checker_request_settings(
                model, "relevance"
            ),
            "groundedness": comparison_checker_request_settings(
                model, "groundedness"
            ),
        }
        for model in models
    }
    if manifest.get("checker_requests") != expected_requests:
        raise ValueError("Comparison checker request settings have changed.")
    recorded_production_requests = manifest.get("production_model_requests")
    if (
        recorded_production_requests is not None
        and recorded_production_requests
        != production_model_request_provenance()
    ):
        raise ValueError("Production model request settings have changed.")
    validate_profile_provenance(
        manifest,
        profile,
        records_path=inputs_path,
        allow_legacy_read=False,
    )


def _refresh_outputs(
    manifest,
    manifest_path,
    metrics_path,
    models,
    selected_keys,
    inputs,
    results,
    results_path,
):
    manifest["counts"] = {
        "selected_cases": len(selected_keys),
        "input_records": len(inputs),
        "successful_inputs": sum(record.get("status") == "success" for record in inputs),
        "input_errors": sum(record.get("status") != "success" for record in inputs),
        "result_records": len(results),
        "successful_results": sum(record.get("status") == "success" for record in results),
        "result_errors": sum(record.get("status") != "success" for record in results),
    }
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(
        metrics_path,
        build_metrics(
            manifest["run_id"], models, selected_keys, results, results_path
        ),
    )


def run_comparison(args):
    run_id = args.run_id
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError("Invalid checker-comparison run ID.")
    models = tuple(args.models)
    if len(models) != 2 or len(set(models)) != 2:
        raise ValueError("--models requires exactly two distinct checker models.")
    for model in models:
        comparison_checker_request_settings(model, "relevance")
        comparison_checker_request_settings(model, "groundedness")

    run_dir = RUNS_DIR / run_id
    comparison_dir = run_dir / "checker_comparison"
    inputs_path = comparison_dir / "inputs.jsonl"
    results_path = comparison_dir / "results.jsonl"
    metrics_path = comparison_dir / "metrics.json"
    manifest_path = comparison_dir / "comparison_manifest.json"

    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError("Cannot resume; comparison manifest not found.")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        profile = load_evaluation_config()
        validate_resume_manifest(
            manifest, args, run_id, profile, models, inputs_path
        )
    else:
        if comparison_dir.exists() or run_dir.exists():
            raise FileExistsError(
                "Run directory already exists; choose a new --run-id or use "
                "--resume for an existing checker comparison."
            )
        profile = load_evaluation_config()
        comparison_dir.mkdir(parents=True)
        manifest = None

    records = load_qasper_records(profile.qa_path)
    grouped_cases = select_cases(records, args.max_papers, args.max_questions)
    selected = []
    for paper, indices in grouped_cases:
        for question_index in indices:
            selected.append((len(selected), paper, question_index))
    selected_keys = [
        case_key(paper["id"], paper["qas"]["question_id"][question_index])
        for _, paper, question_index in selected
    ]
    if not selected_keys:
        raise ValueError("The selected comparison subset contains no questions.")

    inputs = read_jsonl(inputs_path, "checker inputs")
    input_by_key = {}
    selected_key_set = set(selected_keys)
    for record in inputs:
        validate_input_record(record)
        key = case_key(record.get("paper_id"), record.get("question_id"))
        if key not in selected_key_set:
            raise ValueError(f"Checker input does not belong to this run: {key}.")
        if key in input_by_key:
            raise ValueError(f"Duplicate checker input record: {key}.")
        input_by_key[key] = record
    results = read_jsonl(results_path, "checker results")
    result_keys = set()
    for record in results:
        validate_result_record(
            record,
            input_by_key,
            selected_key_set,
            set(models),
        )
        key = result_key(record)
        if key in result_keys:
            raise ValueError(f"Duplicate checker result record: {key}.")
        result_keys.add(key)

    if manifest is None:
        manifest = make_manifest(
            args, run_id, comparison_dir, profile, models, len(selected_keys)
        )
    elif manifest.get("counts", {}).get("selected_cases") != len(selected_keys):
        raise ValueError("Resume selected-case count has changed.")
    manifest["status"] = "running"
    manifest["finished_at"] = None
    _refresh_outputs(
        manifest,
        manifest_path,
        metrics_path,
        models,
        selected_keys,
        inputs,
        results,
        results_path,
    )

    from core.models import initialize_models
    from core.state import delete_vectorstore, new_kb_state
    from ingestion.indexing import index_files

    reranker = None
    models_initialized = False
    kb_state = new_kb_state(mode="single")
    current_paper_id = None
    indexing_error = None
    try:
        for case_index, paper, question_index in selected:
            paper_id = paper["id"]
            qas = paper["qas"]
            question_id = qas["question_id"][question_index]
            key = case_key(paper_id, question_id)
            input_record = input_by_key.get(key)

            missing_models = [
                model for model in model_order(models, case_index)
                if (paper_id, question_id, model) not in result_keys
            ]
            if not missing_models:
                continue

            if input_record is None:
                if current_paper_id != paper_id:
                    current_paper_id = paper_id
                    indexing_error = None
                    try:
                        if not models_initialized:
                            _, reranker = initialize_models()
                            models_initialized = True
                        pdf_id = paper_id.removeprefix("arXiv:")
                        pdf_path = profile.pdf_path / f"{pdf_id}.pdf"
                        if not pdf_path.is_file():
                            raise FileNotFoundError("QASPER PDF is unavailable.")
                        kb_state, _, _, _ = index_files(
                            [str(pdf_path)],
                            kb_state,
                            "single",
                            profile.chunk_size,
                            profile.chunk_overlap,
                        )
                        kb_state["rerank_model"] = reranker
                        validate_kb_state(kb_state, paper_id)
                    except Exception as exc:
                        indexing_error = sanitize_error(exc)
                try:
                    if indexing_error:
                        raise RuntimeError(indexing_error)
                    prepared = prepare_checker_input(
                        qas["question"][question_index], kb_state, profile
                    )
                    context_digest = checker_context_digest(
                        prepared["relevance_checker_context"],
                        prepared["checker_context"],
                    )
                    input_record = {
                        "paper_id": paper_id,
                        "question_id": question_id,
                        "question": qas["question"][question_index],
                        "rewritten_query": prepared["rewritten_query"],
                        "retrieval_route": prepared["retrieval_route"],
                        "fusion_alpha": prepared["fusion_alpha"],
                        "reranked_documents": prepared["reranked_documents"],
                        "final_contexts": [
                            item["structured_text"]
                            for item in prepared["reranked_documents"]
                        ],
                        "relevance_checker_context": prepared[
                            "relevance_checker_context"
                        ],
                        "checker_context": prepared["checker_context"],
                        "generated_answer": prepared["generated_answer"],
                        "gold_answers": qas["answers"][question_index],
                        "context_digest": context_digest,
                        "answer_digest": sha256_text(
                            prepared["generated_answer"]
                        ),
                        "status": "success",
                        "error": None,
                    }
                    validate_input_record(input_record)
                except Exception as exc:
                    input_record = {
                        "paper_id": paper_id,
                        "question_id": question_id,
                        "question": qas["question"][question_index],
                        "rewritten_query": None,
                        "reranked_documents": [],
                        "final_contexts": [],
                        "relevance_checker_context": None,
                        "checker_context": None,
                        "generated_answer": None,
                        "gold_answers": qas["answers"][question_index],
                        "context_digest": None,
                        "answer_digest": None,
                        "status": "input_error",
                        "error": sanitize_error(exc),
                    }
                append_jsonl(inputs_path, input_record)
                inputs.append(input_record)
                input_by_key[key] = input_record
                _refresh_outputs(
                    manifest,
                    manifest_path,
                    metrics_path,
                    models,
                    selected_keys,
                    inputs,
                    results,
                    results_path,
                )

            ordered_models = model_order(models, case_index)
            for order_position, model in enumerate(ordered_models, start=1):
                key_with_model = (paper_id, question_id, model)
                if key_with_model in result_keys:
                    continue
                result = make_model_result(input_record, model, order_position)
                append_jsonl(results_path, result)
                results.append(result)
                result_keys.add(key_with_model)
                _refresh_outputs(
                    manifest,
                    manifest_path,
                    metrics_path,
                    models,
                    selected_keys,
                    inputs,
                    results,
                    results_path,
                )

        expected_results = len(selected_keys) * len(models)
        has_errors = any(record.get("status") != "success" for record in results)
        manifest["status"] = (
            "complete_with_errors" if has_errors else "complete"
        )
        if len(result_keys) != expected_results:
            manifest["status"] = "incomplete"
        manifest["finished_at"] = utc_now()
        _refresh_outputs(
            manifest,
            manifest_path,
            metrics_path,
            models,
            selected_keys,
            inputs,
            results,
            results_path,
        )
    except BaseException:
        manifest["status"] = "interrupted"
        manifest["finished_at"] = utc_now()
        _refresh_outputs(
            manifest,
            manifest_path,
            metrics_path,
            models,
            selected_keys,
            inputs,
            results,
            results_path,
        )
        raise
    finally:
        delete_vectorstore(kb_state.get("vectordb") if kb_state else None)

    print(
        f"Checker comparison {run_id} finished with status "
        f"{manifest['status']}. Results: {results_path}"
    )
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compare production checker models on identical RAG inputs."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-papers", type=positive_int)
    parser.add_argument("--max-questions", type=positive_int)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_comparison(args)


if __name__ == "__main__":
    main()
