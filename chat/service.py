import gradio as gr
import time
from concurrent.futures import ThreadPoolExecutor
from retrieval.fusion import hybrid_fusion
from retrieval.reranking import rerank_results
from generation.answer_generator import generate_answer
from generation.query_rewriter import rewrite_query
from retrieval.routing import route_retrieval_query
from retrieval.dense import get_dense_results_filtered
from retrieval.bm25 import get_bm25_scores_filtered
from retrieval.relevance import compute_retrieval_relevance
from generation.groundedness import check_groundedness
from debug.builder import build_debug_info
from telemetry.logger import log_evaluation_entry
from utils.ids import get_query_id
from ingestion.indexing import get_active_doc_ids
from utils.source_formatter import format_sources


def _is_optional_number(value):
    return value is None or (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
    )


def _resolve_retrieval_checker(future):
    try:
        result = future.result()
    except Exception as exc:
        return (
            "unknown",
            None,
            f"Retrieval relevance check failed: {exc}",
            None,
        )

    if (
        not isinstance(result, (tuple, list))
        or len(result) != 4
    ):
        return (
            "unknown",
            None,
            "Retrieval relevance checker returned a malformed result.",
            None,
        )

    label, score, explanation, latency = result
    if (
        label not in {"high", "medium", "low", "unknown"}
        or not _is_optional_number(score)
        or not isinstance(explanation, str)
        or not _is_optional_number(latency)
    ):
        return (
            "unknown",
            None,
            "Retrieval relevance checker returned a malformed result.",
            None,
        )

    return label, score, explanation, latency


def _resolve_groundedness_checker(future):
    try:
        result = future.result()
    except Exception as exc:
        return (
            "unknown",
            f"Groundedness check failed: {exc}",
            None,
        )

    if (
        not isinstance(result, (tuple, list))
        or len(result) != 3
    ):
        return (
            "unknown",
            "Groundedness checker returned a malformed result.",
            None,
        )

    label, explanation, latency = result
    if (
        label not in {
            "grounded",
            "partially_grounded",
            "not_grounded",
            "unknown",
        }
        or not isinstance(explanation, str)
        or not _is_optional_number(latency)
    ):
        return (
            "unknown",
            "Groundedness checker returned a malformed result.",
            None,
        )

    return label, explanation, latency


def chat(
    user_query,
    chat_history,
    kb_state,
    retrieval_mode,
    retrieval_top_n,
    rerank_candidate_n,
    rerank_top_k,
    fusion_threshold,
    debug_enabled,
    query_scope_mode,
    selected_doc_names,
    conversation_id,
):
    if not user_query or not user_query.strip():
        return (
            chat_history,
            kb_state,
            gr.skip(),
            gr.skip(),
            conversation_id,
        )
    if (
        kb_state is None
        or kb_state.get("vectordb") is None
        or not kb_state.get("documents")
    ):
        chat_history.append(
            {
                "role": "assistant",
                "content": (
                    "**Document required.** Upload and index a PDF, DOC, or DOCX "
                    "document before asking a question."
                ),
            }
        )
        return chat_history, kb_state, "", "", conversation_id

    retrieval_top_n = max(1, int(retrieval_top_n))
    rerank_top_k = max(1, int(rerank_top_k))

    rerank_candidate_n = max(
        rerank_top_k,
        int(rerank_candidate_n),
    )

    fusion_threshold = max(
        0.0,
        min(1.0, fusion_threshold),
    )

    query_id = get_query_id(chat_history)

    active_doc_ids = get_active_doc_ids(
        kb_state,
        query_scope_mode,
        selected_doc_names,
    )

    if not active_doc_ids:
        chat_history.append(
            {
                "role": "assistant",
                "content": (
                    "**No documents selected.** Choose at least one indexed "
                    "document or switch the scope to all indexed documents."
                ),
            }
        )
        return chat_history, kb_state, "", "", conversation_id

    active_doc_names = [
        meta["file_name"]
        for doc_id, meta in kb_state["documents"].items()
        if doc_id in active_doc_ids
    ]

    rewritten_query = rewrite_query(
        user_query,
        chat_history,
    )

    retrieval_route, fusion_alpha = route_retrieval_query(
        rewritten_query,
        retrieval_mode,
    )

    retrieval_start = time.perf_counter()

    t0 = time.perf_counter()
    dense_results = get_dense_results_filtered(
        vectordb=kb_state["vectordb"],
        query=rewritten_query,
        allowed_doc_ids=active_doc_ids,
        top_n=retrieval_top_n,
    )
    dense_latency = time.perf_counter() - t0

    t0 = time.perf_counter()
    bm25_scores = get_bm25_scores_filtered(
        kb_state=kb_state,
        query=rewritten_query,
        allowed_doc_ids=active_doc_ids,
        top_n=retrieval_top_n,
    )
    bm25_latency = time.perf_counter() - t0

    chunk_lookup = {
        doc.metadata.get("chunk_id"): doc
        for doc in kb_state.get("bm25_chunks", [])
        if doc.metadata.get("doc_id") in active_doc_ids
    }

    t0 = time.perf_counter()
    filtered_docs = hybrid_fusion(
        dense_results=dense_results,
        bm25_scores=bm25_scores,
        chunk_lookup=chunk_lookup,
        fusion_alpha=fusion_alpha,
        score_threshold=fusion_threshold,
    )
    fusion_latency = time.perf_counter() - t0

    raw_docs = filtered_docs.copy()

    reranker_candidates = filtered_docs[:rerank_candidate_n]

    t0 = time.perf_counter()
    reranked_docs, reranked_dists, rerank_scores = rerank_results(
        filtered_docs=reranker_candidates,
        rerank_model=kb_state.get("rerank_model"),
        rewritten_query=rewritten_query,
        top_k=rerank_top_k,
    )
    rerank_latency = time.perf_counter() - t0

    retrieval_latency = time.perf_counter() - retrieval_start

    (
        answer,
        context,
        prompt,
        llm_latency,
        tokens,
    ) = generate_answer(
        question=user_query,
        chat_history=chat_history,
        reranked_docs=reranked_docs,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:

        retrieval_future = executor.submit(
            compute_retrieval_relevance,
            rewritten_query,
            reranked_docs,
        )

        groundedness_future = executor.submit(
            check_groundedness,
            question=user_query,
            answer=answer,
            context=context,
        )

        (
            retrieval_label,
            retrieval_relevance_score,
            retrieval_explanation,
            retrieval_eval_latency,
        ) = _resolve_retrieval_checker(retrieval_future)

        (
            groundedness_label,
            groundedness_explanation,
            groundedness_eval_latency,
        ) = _resolve_groundedness_checker(groundedness_future)

    retrieval_map = {
        "high": "High",
        "medium": "Medium",
        "low": "Low",
        "unknown": "Unknown",
    }

    retrieval_display = retrieval_map.get(
        retrieval_label,
        "Low",
    )

    groundedness_map = {
        "grounded": "Grounded",
        "partially_grounded": "Partially grounded",
        "not_grounded": "Not grounded",
        "unknown": "Unknown",
    }

    groundedness_display = groundedness_map.get(
        groundedness_label,
        "Partially grounded",
    )

    diagnostic_items = [
        f"- **Retrieval relevance:** {retrieval_display}",
        f"- **Groundedness:** {groundedness_display}",
    ]

    if kb_state["mode"] == "multi":
        if query_scope_mode == "all":
            diagnostic_items.append(
                f"- **Scope:** All indexed documents ({len(active_doc_names)})"
            )
        else:
            diagnostic_items.append(
                f"- **Scope:** Selected documents ({', '.join(active_doc_names)})"
            )

    # Retain the blank line after the rule for the existing suffix cleaner.
    answer += (
        "\n\n---\n\n"
        "**Answer diagnostics**\n\n"
        + "\n".join(diagnostic_items)
    )

    log_evaluation_entry(
    kb_mode=kb_state["mode"],
    active_doc_names=active_doc_names,
    conversation_id=conversation_id,
    query_id=query_id,
    original_query=user_query,
    rewritten_query=rewritten_query,
    retrieval_route=retrieval_route,
    fusion_alpha=fusion_alpha,
    retrieval_top_n=retrieval_top_n,
    rerank_top_k=rerank_top_k,
    fusion_threshold=fusion_threshold,
    rerank_candidate_n=rerank_candidate_n,
    retrieval_relevance=retrieval_label,
    retrieval_relevance_score=retrieval_relevance_score,
    retrieval_explanation=retrieval_explanation,
    retrieval_eval_latency=retrieval_eval_latency,
    groundedness_label=groundedness_label,
    groundedness_explanation=groundedness_explanation,
    groundedness_eval_latency=groundedness_eval_latency,
    answer=answer,
    raw_docs=raw_docs,
    reranked_docs=reranked_docs,
    reranked_dists=reranked_dists,
    rerank_scores=rerank_scores,
    retrieval_latency=retrieval_latency,
    dense_latency=dense_latency,
    bm25_latency=bm25_latency,
    fusion_latency=fusion_latency,
    rerank_latency=rerank_latency,
    tokens=tokens,
    llm_latency=llm_latency,
)

    chat_history.extend(
        [
            {
                "role": "user",
                "content": user_query,
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ]
    )

    sources = format_sources(
        reranked_docs,
        reranked_dists,
        rerank_scores,
    )

    debug_text = build_debug_info(
        original_query=user_query,
        rewritten_query=rewritten_query,
        retrieval_route=retrieval_route,
        fusion_alpha=fusion_alpha,
        raw_docs=raw_docs,
        reranked_docs=reranked_docs,
        reranked_dists=reranked_dists,
        rerank_scores=rerank_scores,
        rerank_candidate_n=rerank_candidate_n,
        num_reranker_candidates=len(reranker_candidates),
        prompt=prompt,
        retrieval_latency=retrieval_latency,
        dense_latency=dense_latency,
        bm25_latency=bm25_latency,
        fusion_latency=fusion_latency,
        rerank_latency=rerank_latency,
        retrieval_relevance=retrieval_label,
        retrieval_relevance_score=retrieval_relevance_score,
        retrieval_explanation=retrieval_explanation,
        retrieval_eval_latency=retrieval_eval_latency,
        groundedness_label=groundedness_label,
        groundedness_explanation=groundedness_explanation,
        groundedness_eval_latency=groundedness_eval_latency,
        tokens=tokens,
        llm_latency=llm_latency,
    )

    return (
        chat_history,
        kb_state,
        sources,
        debug_text,
        conversation_id,
    )
