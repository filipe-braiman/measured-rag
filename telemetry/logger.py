import json
import math
import sys
from datetime import datetime
from pathlib import Path

from config import DEPLOYMENT_MODE
from utils.helpers import distance_to_similarity, truncate_text

if DEPLOYMENT_MODE not in ("local", "cloud"):
    raise ValueError('DEPLOYMENT_MODE must be exactly "local" or "cloud".')

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_ROOT / "answer_logs"
if DEPLOYMENT_MODE == "local":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
EVAL_LOG_PATH = LOG_DIR / "rag_eval_log.jsonl"


def _cloud_number(value, integer=False):
    """Accept only operational numbers, never arbitrary strings or objects."""
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Invalid operational metric.")
    return int(value) if integer else round(float(value), 4)


def _cloud_label(value, allowed):
    return value if type(value) is str and value in allowed else "unknown"


def _emit_cloud_event(event):
    sys.stdout.write(json.dumps(event, allow_nan=False) + "\n")
    sys.stdout.flush()


def serialize_sources(docs, dists, rerank_scores):
    serialized = []
    for rank, (doc, dist, rerank_score) in enumerate(zip(docs, dists, rerank_scores), start=1):
        serialized.append({
            "rank": rank,
            "doc_id": doc.metadata.get("doc_id"),
            "file_name": doc.metadata.get("source_name"),
            "chunk_id": doc.metadata.get("chunk_id"),
            "page": doc.metadata.get("page"),
            "file_type": doc.metadata.get("file_type"),
            "doc_type": doc.metadata.get("doc_type"),
            "section_header": doc.metadata.get("section_header"),
            "distance": round(float(dist), 6),
            "similarity": round(float(distance_to_similarity(dist)), 6),
            "fusion_score": round(float(doc.metadata.get("fusion_score", 0.0)), 6),
            "rerank_score": round(float(rerank_score), 6),
            "content_preview": truncate_text(doc.page_content.strip(), 500),
        })
    return serialized

def log_evaluation_entry(
    kb_mode,
    active_doc_names,
    conversation_id,
    query_id,
    original_query,
    rewritten_query,
    retrieval_route,
    fusion_alpha,
    retrieval_top_n,
    rerank_candidate_n,
    rerank_top_k,
    fusion_threshold,
    retrieval_relevance,
    retrieval_relevance_score,
    retrieval_explanation,
    retrieval_eval_latency,
    groundedness_label,
    groundedness_explanation,
    groundedness_eval_latency,
    answer,
    raw_docs,
    reranked_docs,
    reranked_dists,
    rerank_scores,
    retrieval_latency,
    dense_latency,
    bm25_latency,
    fusion_latency,
    rerank_latency,
    tokens,
    llm_latency
):
    if DEPLOYMENT_MODE == "cloud":
        try:
            # Construct independently of the detailed local record. Only list
            # counts are used; document objects and sensitive arguments are
            # neither inspected, serialized, nor hashed in this branch.
            event = {
                "event": "rag_request_completed",
                "timestamp": datetime.now().isoformat(),
                "kb_mode": _cloud_label(kb_mode, {"single", "multi"}),
                "retrieval_route": _cloud_label(retrieval_route, {"keyword", "semantic", "balanced"}),
                "fusion_alpha": _cloud_number(fusion_alpha),
                "retrieval_top_n": _cloud_number(retrieval_top_n, integer=True),
                "rerank_candidate_n": _cloud_number(rerank_candidate_n, integer=True),
                "rerank_top_k": _cloud_number(rerank_top_k, integer=True),
                "fusion_threshold": _cloud_number(fusion_threshold),
                "num_fusion_survivors": len(raw_docs),
                "num_reranker_candidates": min(len(raw_docs), int(rerank_candidate_n)),
                "num_final_sources": len(reranked_docs),
                "retrieval_relevance": _cloud_label(retrieval_relevance, {"high", "medium", "low", "unknown"}),
                "relevance_score": _cloud_number(retrieval_relevance_score),
                "groundedness_label": _cloud_label(groundedness_label, {"grounded", "partially_grounded", "not_grounded", "unknown"}),
                "retrieval_latency": _cloud_number(retrieval_latency),
                "dense_latency": _cloud_number(dense_latency),
                "bm25_latency": _cloud_number(bm25_latency),
                "fusion_latency": _cloud_number(fusion_latency),
                "rerank_latency": _cloud_number(rerank_latency),
                "retrieval_eval_latency": _cloud_number(retrieval_eval_latency),
                "groundedness_eval_latency": _cloud_number(groundedness_eval_latency),
                "llm_latency": _cloud_number(llm_latency),
                "total_tokens": _cloud_number(tokens, integer=True),
            }
            _emit_cloud_event(event)
        except Exception:
            try:
                _emit_cloud_event({"event": "rag_telemetry_error"})
            except Exception:
                # A broken stdout must not fail the request or trigger a
                # fallback to content-bearing local logging.
                pass
        return

    try:
        raw_retrieval = []
        for rank, (doc, dist, fusion) in enumerate(raw_docs, start=1):
            raw_retrieval.append({
                "rank": rank,
                "doc_id": doc.metadata.get("doc_id"),
                "file_name": doc.metadata.get("source_name"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "page": doc.metadata.get("page"),
                "distance": round(float(dist), 6),
                "similarity": round(float(distance_to_similarity(dist)), 6),
                "fusion_score": round(float(fusion), 6),
                "content_preview": truncate_text(doc.page_content.strip(), 300),
                "file_type": doc.metadata.get("file_type"),
                "doc_type": doc.metadata.get("doc_type"),
                "section_header": doc.metadata.get("section_header"),
            })

        entry = {
            "timestamp": datetime.now().isoformat(),
            "conversation_id": conversation_id,
            "query_id": query_id,
            "kb_mode": kb_mode,
            "active_doc_names": active_doc_names,
            "query": {
                "original": original_query,
                "rewritten": rewritten_query,
            },
            "retrieval": {
                "route": retrieval_route,
                "fusion_alpha": round(float(fusion_alpha), 4),
                "retrieval_top_n": int(retrieval_top_n),
                "rerank_candidate_n": int(rerank_candidate_n),
                "rerank_top_k": int(rerank_top_k),
                "fusion_threshold": round(
                    float(fusion_threshold),
                    4,
                ),
                "latency": {
                    "total_seconds": (
                        round(float(retrieval_latency), 4)
                        if retrieval_latency is not None
                        else None
                    ),
                    "dense_seconds": (
                        round(float(dense_latency), 4)
                        if dense_latency is not None
                        else None
                    ),
                    "bm25_seconds": (
                        round(float(bm25_latency), 4)
                        if bm25_latency is not None
                        else None
                    ),
                    "fusion_seconds": (
                        round(float(fusion_latency), 4)
                        if fusion_latency is not None
                        else None
                    ),
                    "rerank_seconds": (
                        round(float(rerank_latency), 4)
                        if rerank_latency is not None
                        else None
                    ),
                },
                "num_fusion_survivors": len(raw_docs),
                "num_reranker_candidates": min(
                    len(raw_docs),
                    int(rerank_candidate_n),
                ),
                "raw_retrieval": raw_retrieval,
                "final_sources": serialize_sources(
                    reranked_docs,
                    reranked_dists,
                    rerank_scores,
                ),
                "rerank_margin": (
                    round(
                        float(rerank_scores[0]) -
                        float(rerank_scores[1]),
                        6,
                    )
                    if len(rerank_scores) >= 2
                    else None
                ),
                "retrieval_relevance": retrieval_relevance,
                "relevance_score": (
                    round(float(retrieval_relevance_score), 4)
                    if retrieval_relevance_score is not None
                    else None
                ),
                "relevance_explanation": (
                    retrieval_explanation
                ),
                "retrieval_eval_latency": (
                    round(float(retrieval_eval_latency), 4)
                    if retrieval_eval_latency is not None
                    else None
                ),
                "num_final_sources": len(reranked_docs),
            },
            "generation": {
                "answer": answer,
                "groundedness_label": groundedness_label,
                "groundedness_explanation": groundedness_explanation,
                "groundedness_eval_latency": (
                    round(float(groundedness_eval_latency), 4)
                    if groundedness_eval_latency is not None
                    else None
                ),
            },
            "llm_metrics": {
                "total_tokens": int(tokens) if tokens is not None else None,
                "latency_seconds": round(float(llm_latency), 4) if llm_latency is not None else None,
            },
        }

        with open(EVAL_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    except Exception as e:
        print(f"[Evaluation Logging Error] {e}")
